#!/usr/bin/env python3
"""Ask Claude the judgement calls that numbers cannot make.

The pipeline measures plenty: frames registered, how well keyframes agree on
scale, how flat the walls came out. Some questions are visual instead. Is this
a walkthrough or someone panning in place? Are the walls bare? Did the
detector miss the bed? Does this room render look like a room at all?
agent.py asks those here, and keeps its numeric guards as the safety net, so
an opinion never overrides a measurement.

Backends, tried in order:
  1. a relay, when OASIS_CLAUDE_RELAY names a folder: the question and the
     paths of its images are written there, and tools/claude_relay.py on
     another machine with a `claude` login answers it. This is how a pipeline
     on a Colab GPU gets Claude's checks without an API key;
  2. the `claude` CLI in headless mode, which uses your existing Claude login
     and needs no API key (run `claude login` once if the session expired);
  3. the Anthropic API, when the `anthropic` SDK is installed and a key is in
     the environment;
  4. offline: no model, so callers fall back to their own rules.

Standalone check:
    python3 pipeline/advisor.py [image ...]
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

# The model every question goes to. OASIS_CLAUDE_MODEL picks another (a plan that
# cannot use this one leaves the CLI waiting instead of failing); "default" lets the
# claude CLI use the account's own default model.
MODEL = os.environ.get("OASIS_CLAUDE_MODEL") or "claude-opus-5"
API_MODEL = "claude-opus-5"   # the API has no account default
RELAY_ENV = "OASIS_CLAUDE_RELAY"
# A relayed question waits for the other machine to fetch it (each Colab CLI
# call takes 20 s to 2 min) plus Claude's own answer.
RELAY_WAIT_SECONDS = 1800
# Relayed images travel as one bundle per question, each image no larger than
# this on its long side: the most Claude reads of an image (the high-resolution
# tier of Claude 4.7 and later; a 36-frame contact sheet needs every pixel).
RELAY_IMAGE_LONG_SIDE = 2576
# A photo (a video frame, 3840 px on its long side) is sent at this size: enough to
# tell what an object is, at about a third of the tokens of the full tier. Sheets
# the pipeline draws (plans, crops) are below RELAY_IMAGE_LONG_SIDE and go as drawn.
PHOTO_LONG_SIDE = 1568
PHOTO_ABOVE = 2600
# The brief every question is asked under: what Oasis does and what the model's part is.
BRIEF_PATH = Path(__file__).with_name("oasis_brief.md")


def image_block(path: Path) -> dict:
    """An image as a base64 content block, a photo shrunk to PHOTO_LONG_SIDE and
    anything else to RELAY_IMAGE_LONG_SIDE."""
    import base64
    import io
    import mimetypes

    from PIL import Image

    data, media = path.read_bytes(), mimetypes.guess_type(path.name)[0] or "image/png"
    with Image.open(path) as img:
        side = max(img.size)
        target = PHOTO_LONG_SIDE if side > PHOTO_ABOVE else RELAY_IMAGE_LONG_SIDE
        if side > target:
            img.thumbnail((target, target))
            buffer = io.BytesIO()
            img.convert("RGB").save(buffer, "JPEG", quality=88)
            data, media = buffer.getvalue(), "image/jpeg"
    return {"type": "image", "source": {"type": "base64", "media_type": media,
                                        "data": base64.standard_b64encode(data).decode()}}


def first_json(text: str) -> dict:
    """The claude CLI's answer from its --output-format json output. At a
    session limit the CLI prints its notice as a second JSON document after
    the first (seen 2026-10-10: "Extra data: line 2 column 1"), so the whole
    output is not one document: the first is taken, and when it carries no
    result and a later document is an error, that error is the answer."""
    text = (text or "").strip()
    if not text:
        return {}
    decoder = json.JSONDecoder()
    payload, end = decoder.raw_decode(text)
    payload = payload if isinstance(payload, dict) else {}
    rest = text[end:].strip()
    while rest and not payload.get("result") and not payload.get("is_error"):
        try:
            extra, end = decoder.raw_decode(rest)
        except ValueError:
            break
        if isinstance(extra, dict) and extra.get("is_error"):
            return extra
        rest = rest[end:].strip()
    return payload


class Advisor:
    """One place to ask Claude, whichever way this machine can reach it."""

    def __init__(self, model: str = MODEL, timeout: int = 600, enabled: bool = True,
                 system: str | None = None):
        self.model = model
        self.timeout = timeout
        self.system = system          # the system prompt every question is asked under (the brief)
        self.calls = 0
        self.usage: list[dict] = []   # tokens and cost of each answered call
        self.reason: str | None = None
        self.backend = self._pick_backend() if enabled else "offline"
        if not enabled:
            self.reason = "disabled by the caller"

    @property
    def available(self) -> bool:
        return self.backend != "offline"

    def _pick_backend(self) -> str:
        if os.getenv(RELAY_ENV):
            return "relay"
        if shutil.which("claude"):
            return "cli"
        if importlib.util.find_spec("anthropic") and (
                os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN")):
            return "api"
        self.reason = ("no `claude` CLI on PATH and no anthropic SDK with a key "
                       "in the environment")
        return "offline"

    def _disable(self, reason: str) -> None:
        self.backend = "offline"
        self.reason = reason.strip().splitlines()[0][:200] if reason else "unknown error"

    # ------------------------------------------------------------- backends
    def _ask_cli(self, prompt: str, images: list[Path]) -> str | None:
        """One turn of `claude -p`: the images go inline in the message (stream-json
        input), so the CLI needs no tool to open them; no tools, no MCP servers and
        the brief instead of Claude Code's own system prompt. About 3k input tokens
        for a question with a photo, against ~98k when the CLI read each image with
        its Read tool over three turns carrying its whole default context."""
        import tempfile

        message = {"type": "user", "message": {"role": "user", "content":
                   [image_block(p) for p in images] + [{"type": "text", "text": prompt}]}}
        command = ["claude", "-p", "--input-format", "stream-json", "--output-format", "stream-json",
                   "--verbose", *(["--model", self.model] if self.model != "default" else []),
                   "--tools", "", "--strict-mcp-config",
                   "--no-session-persistence"]
        system_file = None
        if self.system:
            system_file = tempfile.NamedTemporaryFile("w", suffix=".md", delete=False)
            system_file.write(self.system)
            system_file.close()
            command += ["--system-prompt-file", system_file.name]
        try:
            result = subprocess.run(command, input=json.dumps(message) + "\n", capture_output=True,
                                    text=True, timeout=self.timeout)
        except subprocess.TimeoutExpired:
            self._disable(f"the claude CLI did not answer within {self.timeout}s")
            return None
        except OSError as exc:
            self._disable(f"could not run the claude CLI: {exc}")
            return None
        finally:
            if system_file:
                Path(system_file.name).unlink(missing_ok=True)
        payload = {}
        for line in result.stdout.splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict) and event.get("type") == "result":
                payload = event
        if not payload:
            self._disable((result.stderr or "the claude CLI gave no result").strip())
            return None
        if payload.get("is_error"):
            self._disable(str(payload.get("result", "claude CLI reported an error")))
            return None
        self._record_usage(payload.get("usage") or {}, payload.get("total_cost_usd"))
        return payload.get("result")

    def _record_usage(self, usage: dict, cost=None) -> None:
        self.usage.append({"input": int(usage.get("input_tokens") or 0),
                           "cache_write": int(usage.get("cache_creation_input_tokens") or 0),
                           "cache_read": int(usage.get("cache_read_input_tokens") or 0),
                           "output": int(usage.get("output_tokens") or 0),
                           **({"cost_usd": round(float(cost), 4)} if cost is not None else {})})

    def usage_total(self) -> dict:
        """Tokens and cost over every call so far."""
        total = {"calls": len(self.usage)}
        for entry in self.usage:
            for k, v in entry.items():
                total[k] = round(total.get(k, 0) + v, 4)
        return total

    def _ask_relay(self, prompt: str, images: list[Path]) -> str | None:
        """Leave the question in the relay folder and wait for the answer that
        tools/claude_relay.py uploads: requests/<id>.json in, responses/<id>.json out."""
        root = Path(os.environ[RELAY_ENV])
        (root / "requests").mkdir(parents=True, exist_ok=True)
        (root / "responses").mkdir(parents=True, exist_ok=True)
        name = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}.json"
        request = {"prompt": prompt, "model": self.model, "system": self.system,
                   "images": [str(p.resolve()) for p in images]}
        if images:
            # One download for the whole question instead of one per image.
            request["bundle"] = str(self._bundle(root / "bundles" / f"{Path(name).stem}.tar", images))
        pending = root / "requests" / f".{name}.tmp"
        pending.write_text(json.dumps(request))
        pending.rename(root / "requests" / name)  # appears whole, never half-written
        answer = root / "responses" / name
        deadline = time.time() + RELAY_WAIT_SECONDS
        while time.time() < deadline:
            if answer.exists():
                try:
                    reply = json.loads(answer.read_text())
                except json.JSONDecodeError:
                    time.sleep(1)  # still arriving
                    continue
                if reply.get("error"):
                    self._disable(f"relay: {reply['error']}")
                    return None
                return reply.get("result")
            time.sleep(2)
        self._disable(f"no answer from the Claude relay within {RELAY_WAIT_SECONDS}s "
                      "(is tools/claude_relay.py running?)")
        return None

    @staticmethod
    def _bundle(path: Path, images: list[Path]) -> Path:
        """A tar of the question's images, in order (0-<name>, 1-<name>, ...),
        shrunk to RELAY_IMAGE_LONG_SIDE."""
        import io
        import tarfile

        from PIL import Image

        path.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(path, "w") as tar:
            for i, image in enumerate(images):
                img = Image.open(image)
                data = image.read_bytes()
                suffix = image.suffix.lower()
                if max(img.size) > RELAY_IMAGE_LONG_SIDE:
                    img.thumbnail((RELAY_IMAGE_LONG_SIDE, RELAY_IMAGE_LONG_SIDE))
                    buffer = io.BytesIO()
                    img.convert("RGB").save(buffer, "JPEG", quality=90)
                    data, suffix = buffer.getvalue(), ".jpg"
                info = tarfile.TarInfo(f"{i}-{image.stem}{suffix}")
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
        return path

    def _ask_api(self, prompt: str, images: list[Path], max_tokens: int) -> str | None:
        import anthropic

        content = [image_block(p) for p in images] + [{"type": "text", "text": prompt}]
        # The brief is the same on every call: cached, it costs a tenth after the first.
        system = ([{"type": "text", "text": self.system, "cache_control": {"type": "ephemeral"}}]
                  if self.system else anthropic.NOT_GIVEN)
        try:
            message = anthropic.Anthropic().messages.create(
                model=self.model if self.model != "default" else API_MODEL,
                max_tokens=max_tokens,
                system=system,
                output_config={"effort": "low"},  # short judgement calls
                messages=[{"role": "user", "content": content}],
            )
        except Exception as exc:  # auth, network, rate limit
            self._disable(f"{type(exc).__name__}: {exc}")
            return None
        u = message.usage
        self._record_usage({"input_tokens": u.input_tokens, "output_tokens": u.output_tokens,
                            "cache_creation_input_tokens": getattr(u, "cache_creation_input_tokens", 0),
                            "cache_read_input_tokens": getattr(u, "cache_read_input_tokens", 0)})
        return "".join(b.text for b in message.content if b.type == "text")

    # ----------------------------------------------------------------- asks
    def ask(self, prompt: str, images=(), max_tokens: int = 1024) -> str | None:
        """Claude's answer as text, or None when no backend is usable."""
        if not self.available:
            return None
        paths = [Path(p) for p in images if Path(p).exists()]
        if self.backend == "relay":
            answer = self._ask_relay(prompt, paths)
        elif self.backend == "cli":
            answer = self._ask_cli(prompt, paths)
        else:
            answer = self._ask_api(prompt, paths, max_tokens)
        if answer:
            self.calls += 1
        return answer

    def ask_json(self, prompt: str, images=(), max_tokens: int = 1024) -> dict | None:
        """Same, parsed as the JSON object in the reply. An empty or unreadable
        reply is asked once more before giving up."""
        for _ in range(2):
            answer = self.ask(prompt + "\n\nReply with a single JSON object and nothing else.",
                              images, max_tokens)
            start, end = (answer.find("{"), answer.rfind("}")) if answer else (-1, -1)
            if start >= 0 and end > start:
                try:
                    return json.loads(answer[start:end + 1])
                except json.JSONDecodeError:
                    pass
            if not self.available:
                break
        return None


def main() -> None:
    advisor = Advisor()
    print(f"backend: {advisor.backend}" + (f" ({advisor.reason})" if advisor.reason else ""))
    if not advisor.available:
        print("Run `claude login` to let the agent use your existing Claude access.")
        return
    images = sys.argv[1:]
    prompt = ("You are helping a 3D room-capture pipeline. Looking at the attached "
              "frame(s), describe the room, list the furniture you can see, and say "
              "whether the walls are bare (no texture to match on), whether the shot "
              "looks like walking through the room or turning on the spot, and "
              "whether any mirrors, windows or screens are visible.")
    answer = advisor.ask(prompt, images) if images else advisor.ask(
        "Reply with exactly: ready")
    print(answer or f"no answer ({advisor.reason})")


if __name__ == "__main__":
    main()
