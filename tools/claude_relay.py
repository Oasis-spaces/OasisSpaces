#!/usr/bin/env python3
"""Answer Claude's checks for a pipeline running on a Colab session.

pipeline/agent.py on a Colab GPU asks Claude to look at frames and renders.
Colab has no `claude` login, so with OASIS_CLAUDE_RELAY set there the advisor
writes each question to <remote>/requests/<id>.json, with the paths of its
images on the VM, and waits (pipeline/advisor.py). This script runs on a
machine that does have a `claude` login. Through the Colab CLI it watches that
folder, downloads each new question and its images, asks Claude here, and
uploads the answer to <remote>/responses/<id>.json. Only the question and a few
images per check cross; point clouds and models stay on the VM.

Needs the Colab CLI (`uv tool install google-colab-cli`) signed in, a running
session, and `claude` on PATH. Stop it with Ctrl+C.

Usage:
    python3 tools/claude_relay.py --session oasis [--remote /content/claude-relay]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
os.environ.pop("OASIS_CLAUDE_RELAY", None)  # answer here, never relay onward
from advisor import Advisor  # noqa: E402


def colab(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["colab", *args], capture_output=True, text=True, timeout=600)


def remote_names(session: str, path: str) -> list[str]:
    """Entries of a folder on the VM; empty when it does not exist yet."""
    result = colab("ls", "-s", session, path)
    if result.returncode != 0:
        return []
    return [line.strip() for line in result.stdout.splitlines()
            if line.strip() and not line.startswith("[colab]")]


def answer(session: str, remote: str, name: str, work: Path) -> str:
    """Fetch one question, ask Claude, upload the reply. Returns a log line."""
    started = time.time()
    local = work / name
    fetched = colab("download", "-s", session, f"{remote}/requests/{name}", str(local))
    if fetched.returncode != 0 or not local.exists():
        return f"{name}: could not download the question, will retry"
    request = json.loads(local.read_text())
    images = []
    if request.get("bundle"):
        # All of the question's images in one tar (pipeline/advisor.py _bundle).
        bundle = work / f"{local.stem}.tar"
        if colab("download", "-s", session, request["bundle"], str(bundle)).returncode != 0:
            return f"{name}: could not download its images, will retry"
        folder = work / local.stem
        with tarfile.open(bundle) as tar:
            tar.extractall(folder, filter="data")
        images = sorted(folder.iterdir(), key=lambda p: int(p.name.split("-", 1)[0]))
    else:
        for i, path in enumerate(request.get("images", [])):
            target = work / f"{local.stem}-{i}-{Path(path).name}"
            if colab("download", "-s", session, path, str(target)).returncode == 0:
                images.append(target)
    text, advisor = None, None
    for _ in range(2):  # one retry: a rate limit or empty reply would switch Claude off on the VM
        advisor = Advisor(model=request.get("model", "claude-opus-5"))
        text = advisor.ask(request["prompt"], images) if advisor.available else None
        if text:
            break
        time.sleep(20)
    reply = {"result": text} if text else {"error": advisor.reason or "Claude gave no answer"}
    (work / f"reply-{name}").write_text(json.dumps(reply))
    sent = colab("upload", "-s", session, str(work / f"reply-{name}"),
                 f"{remote}/responses/{name}")
    if sent.returncode != 0:
        return f"{name}: answered but the upload failed ({sent.stderr.strip()[-120:]}), will retry"
    summary = (text or reply["error"]).replace("\n", " ")[:90]
    return (f"{name}: {len(images)} image(s), {time.time() - started:.0f}s, "
            f"{'answered' if text else 'error'}: {summary}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--session", "-s", required=True, help="Colab CLI session name")
    parser.add_argument("--remote", default="/content/claude-relay",
                        help="relay folder on the VM (OASIS_CLAUDE_RELAY there)")
    parser.add_argument("--poll", type=float, default=5.0, help="seconds between checks")
    args = parser.parse_args()

    backend = Advisor().backend
    if backend != "cli":
        sys.exit(f"No `claude` CLI here to answer with (backend: {backend}).")
    print(f"Relaying Claude checks for session {args.session} ({args.remote}); Ctrl+C to stop",
          flush=True)
    done: set[str] = set()
    with tempfile.TemporaryDirectory(prefix="claude-relay-") as tmp:
        work = Path(tmp)
        while True:
            answered = set(remote_names(args.session, f"{args.remote}/responses"))
            for name in sorted(remote_names(args.session, f"{args.remote}/requests")):
                if name.endswith(".json") and name not in answered and name not in done:
                    line = answer(args.session, args.remote, name, work)
                    print(time.strftime("%H:%M:%S"), line, flush=True)
                    if "will retry" not in line:
                        done.add(name)
            time.sleep(args.poll)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
