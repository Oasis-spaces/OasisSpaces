#!/usr/bin/env python3
"""A local HTTPS proxy that sends uploads slowly enough to survive this network.

On this Mac's network, an upload sent at full speed dies with "bad record
MAC" once it passes about 30 KB (git pushes, storage uploads), while
downloads of any size are fine. The data is corrupted on its way out when it
leaves fast; handed over at about 170 KB/s it arrives intact (1 MB uploads
and a 46-commit push went through first time, where 80 tries of an 85 KB pack
at full speed all failed). Smaller packets alone did not help; the rate did.

This proxy does only that: it accepts CONNECT (what git, curl and Python use
for an https_proxy), relays bytes both ways untouched (TLS stays end to end),
and writes the outgoing direction in --chunk byte pieces, --gap seconds apart.

    python3 tools/paced_proxy.py &
    https_proxy=http://127.0.0.1:8899 git push origin main
"""

import argparse
import asyncio
import socket


async def relay_paced(reader: asyncio.StreamReader, writer: asyncio.StreamWriter, chunk: int, gap: float) -> None:
    try:
        while data := await reader.read(65536):
            for start in range(0, len(data), chunk):
                writer.write(data[start:start + chunk])
                await writer.drain()
                if gap:
                    await asyncio.sleep(gap)
    except (ConnectionError, asyncio.CancelledError):
        pass
    finally:
        writer.close()


async def relay(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        while data := await reader.read(65536):
            writer.write(data)
            await writer.drain()
    except (ConnectionError, asyncio.CancelledError):
        pass
    finally:
        writer.close()


def make_handler(chunk: int, gap: float, mss: int):
    async def handle(client_reader: asyncio.StreamReader, client_writer: asyncio.StreamWriter) -> None:
        try:
            head = await client_reader.readuntil(b"\r\n\r\n")
            method, target, _ = head.split(b"\r\n", 1)[0].decode().split(" ", 2)
            if method != "CONNECT":
                client_writer.write(b"HTTP/1.1 405 Method Not Allowed\r\n\r\n")
                await client_writer.drain()
                client_writer.close()
                return
            host, _, port = target.rpartition(":")
            upstream_reader, upstream_writer = await asyncio.open_connection(host, int(port), family=socket.AF_INET)
            # A small maximum segment size, so the kernel never sends a large packet on this
            # connection whatever is queued. (macOS only takes it once the connection is open:
            # before that the limit is the 512-byte default.)
            sock = upstream_writer.get_extra_info("socket")
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_MAXSEG, mss)
            client_writer.write(b"HTTP/1.1 200 Connection established\r\n\r\n")
            await client_writer.drain()
            await asyncio.gather(relay_paced(client_reader, upstream_writer, chunk, gap),
                                 relay(upstream_reader, client_writer))
        except Exception as error:
            print(f"connection failed: {type(error).__name__}: {error}", flush=True)
            client_writer.close()
    return handle


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8899)
    parser.add_argument("--chunk", type=int, default=1000, help="bytes per outgoing piece")
    parser.add_argument("--gap", type=float, default=0.005, help="seconds between pieces")
    parser.add_argument("--mss", type=int, default=1000, help="largest TCP segment on the outgoing connection")
    args = parser.parse_args()
    server = await asyncio.start_server(make_handler(args.chunk, args.gap, args.mss), "127.0.0.1", args.port)
    print(f"paced proxy on 127.0.0.1:{args.port} ({args.chunk} bytes every {args.gap * 1000:.0f} ms, "
          f"segments of at most {args.mss} bytes)", flush=True)
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())
