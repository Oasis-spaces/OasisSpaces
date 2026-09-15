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

MODEL = "claude-opus-5"
RELAY_ENV = "OASIS_CLAUDE_RELAY"
# A relayed question waits for the other machine to fetch it (each Colab CLI
# call takes 20 s to 2 min) plus Claude's own answer.
RELAY_WAIT_SECONDS = 1800
# Relayed images travel as one bundle per question, each image no larger than
# this on its long side (about what Claude reads an image at anyway).
RELAY_IMAGE_LONG_SIDE = 1600


class Advisor:
    """One place to ask Claude, whichever way this machine can reach it."""

    def __init__(self, model: str = MODEL, timeout: int = 600, enabled: bool = True):
        self.model = model
        self.timeout = timeout
        self.calls = 0
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
        # The CLI reads files referenced with @path, so images go in as paths.
        refs = " ".join(f"@{p.resolve()}" for p in images)
        command = ["claude", "-p", f"{refs}\n\n{prompt}" if refs else prompt,
                   "--output-format", "json", "--model", self.model]
        if images:
            command += ["--allowedTools", "Read"]
        try:
            result = subprocess.run(command, capture_output=True, text=True,
                                    timeout=self.timeout)
            payload = json.loads(result.stdout or "{}")
        except subprocess.TimeoutExpired:
            self._disable(f"the claude CLI did not answer within {self.timeout}s")
            return None
        except (json.JSONDecodeError, OSError) as exc:
            self._disable(f"could not run the claude CLI: {exc}")
            return None
        if payload.get("is_error"):
            self._disable(str(payload.get("result", "claude CLI reported an error")))
            return None
        return payload.get("result")

    def _ask_relay(self, prompt: str, images: list[Path]) -> str | None:
        """Leave the question in the relay folder and wait for the answer that
        tools/claude_relay.py uploads: requests/<id>.json in, responses/<id>.json out."""
        root = Path(os.environ[RELAY_ENV])
        (root / "requests").mkdir(parents=True, exist_ok=True)
        (root / "responses").mkdir(parents=True, exist_ok=True)
        name = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}.json"
        request = {"prompt": prompt, "model": self.model,
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
        import base64
        import mimetypes

        import anthropic

        content: list[dict] = []
        for path in images:
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": mimetypes.guess_type(path.name)[0] or "image/png",
                    "data": base64.standard_b64encode(path.read_bytes()).decode(),
                },
            })
        content.append({"type": "text", "text": prompt})
        try:
            message = anthropic.Anthropic().messages.create(
                model=self.model,
                max_tokens=max_tokens,
                output_config={"effort": "low"},  # short judgement calls
                messages=[{"role": "user", "content": content}],
            )
        except Exception as exc:  # auth, network, rate limit
            self._disable(f"{type(exc).__name__}: {exc}")
            return None
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
