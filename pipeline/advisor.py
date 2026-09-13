#!/usr/bin/env python3
"""Ask Claude the judgement calls that numbers cannot make.

The pipeline measures plenty: frames registered, how well keyframes agree on
scale, how flat the walls came out. Some questions are visual instead. Is this
a walkthrough or someone panning in place? Are the walls bare? Did the
detector miss the bed? Does this room render look like a room at all?
agent.py asks those here, and keeps its numeric guards as the safety net, so
an opinion never overrides a measurement.

Backends, tried in order:
  1. the `claude` CLI in headless mode, which uses your existing Claude login
     and needs no API key (run `claude login` once if the session expired);
  2. the Anthropic API, when the `anthropic` SDK is installed and a key is in
     the environment;
  3. offline: no model, so callers fall back to their own rules.

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
from pathlib import Path

MODEL = "claude-opus-5"


class Advisor:
    """One place to ask Claude, whichever way this machine can reach it."""

    def __init__(self, model: str = MODEL, timeout: int = 300, enabled: bool = True):
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
        answer = (self._ask_cli(prompt, paths) if self.backend == "cli"
                  else self._ask_api(prompt, paths, max_tokens))
        if answer:
            self.calls += 1
        return answer

    def ask_json(self, prompt: str, images=(), max_tokens: int = 1024) -> dict | None:
        """Same, parsed as the JSON object in the reply."""
        answer = self.ask(prompt + "\n\nReply with a single JSON object and nothing else.",
                          images, max_tokens)
        if not answer:
            return None
        start, end = answer.find("{"), answer.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            return json.loads(answer[start:end + 1])
        except json.JSONDecodeError:
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
