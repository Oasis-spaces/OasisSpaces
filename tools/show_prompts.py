#!/usr/bin/env python3
"""Print every question the pipeline asks Claude, in the order a run asks them.

The prompts live next to the code that uses them, so this reads them out of the
source rather than keeping a copy: what it prints is what a run sends. Module
constants are printed as their text; a prompt built inside a method is printed
as the expression that builds it, with its {...} and f-string parts left in
place, so it is clear what is filled in per room.

Usage:
    python3 tools/show_prompts.py                 # every stage
    python3 tools/show_prompts.py --stage 4       # one stage
    python3 tools/show_prompts.py --markdown      # as a document
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Prompts the code fills in with .format(), where {{ }} stand for the braces Claude sees.
FORMATTED = {"FRAME_CHOICE_PROMPT", "OBJECT_NAMING_PROMPT", "LABEL_REVIEW_PROMPT", "RECHECK_NOTE", "START_VIEW_PROMPT",
             "FILL_REVIEW_PROMPT", "REVIEW_PROMPT", "PROMPT"}

# (stage, what it decides, file, what holds the prompt, the images that go with it)
ASKS = [
    (1, "Was the capture solved well enough, or should the solve be retried another way?",
     "pipeline/agent.py", ("method", "capture_verdict"),
     "3 frames from the start, middle and end"),
    (2, "Which frames of the video should every later question be asked with?",
     "pipeline/agent.py", ("constant", "FRAME_CHOICE_PROMPT"),
     "one contact sheet of 36 numbered frames from across the video"),
    (2, "What objects should the detector look for in this room, and what is each one for?",
     "pipeline/agent.py", ("constant", "OBJECT_NAMING_PROMPT"),
     "6 frames spread across the video"),
    (2, "Did the detector find them? Anything to add, rename or drop before we detect again?",
     "pipeline/agent.py", ("constant", "LABEL_REVIEW_PROMPT"),
     "the detections drawn on 6 frames"),
    (3, "Which measured walls and furniture boxes are real, and what is each one?",
     "pipeline/agent.py", ("method", "structure_review"),
     "the floor plan, each box in the 2 frames that saw it best, 3 frames of the room"),
    (3, "Does the built room look plausible?",
     "pipeline/agent.py", ("method", "render_verdict"),
     "a render of the room, the same from above, 1 frame"),
    (3, "Given those render problems, what should change?  (added to the structure question)",
     "pipeline/agent.py", ("constant", "RECHECK_NOTE"),
     "the renders, on top of the structure images"),
    (4, "Which trained splat is the best one, at trained and unseen views?",
     "tools/splat_choose.py", ("constant", "PROMPT"),
     "each splat rendered at the same views, beside the video's own frames"),
    (4, "Did filling the floor and walls make the splat better or worse?",
     "pipeline/agent.py", ("constant", "FILL_REVIEW_PROMPT"),
     "before and after, at the views the fill touched"),
    (4, "Which view should the splat open at?",
     "pipeline/agent.py", ("constant", "START_VIEW_PROMPT"),
     "6 candidate views rendered from the splat"),
    (4, "In the mixed scene: which textures, scans and photographed sides to keep, and what rests on what?",
     "tools/mixed_scene.py", ("constant", "REVIEW_PROMPT"),
     "the surfaces sheet, the pieces sheet (each scan as filmed and from an unfilmed angle, "
     "beside the real frame), the room with every piece named, the model sides sheet, 2 frames"),
    (4, "What should the person change when filming the next room?",
     "pipeline/agent.py", ("method", "finish"),
     "no images: the run's own numbers and judgements"),
]


def readable(node: ast.AST) -> str:
    """The text a prompt expression builds, with the parts filled in per room
    left as {their expression}."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        return "".join(readable(part) if isinstance(part, (ast.Constant, ast.JoinedStr))
                       else "{" + ast.unparse(part.value) + "}" for part in node.values)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return readable(node.left) + readable(node.right)
    if isinstance(node, ast.IfExp):          # a part added only in some runs
        return ("[only when " + ast.unparse(node.test) + ": " + readable(node.body).strip() + "]"
                + readable(node.orelse))
    if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "format":
        return readable(node.func.value)
    return "{" + ast.unparse(node) + "}"


def unescape(text: str, formatted: bool) -> str:
    """A .format() prompt writes {{ and }} for the braces Claude sees."""
    return text.replace("{{", "{").replace("}}", "}") if formatted else text
    return "{" + ast.unparse(node) + "}"


def source_of(path: Path, kind: str, name: str) -> str:
    """A prompt's text, whether it is a constant or built inside a method."""
    tree = ast.parse(path.read_text())
    if kind == "constant":
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets):
                return ast.literal_eval(node.value)
        raise KeyError(f"{name} not found in {path}")
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            for inner in ast.walk(node):
                if isinstance(inner, ast.Assign) and any(getattr(t, "id", None) == "prompt"
                                                         for t in inner.targets):
                    return readable(inner.value)
            for inner in ast.walk(node):          # asked without naming the prompt first
                if isinstance(inner, ast.Call) and ".ask" in ast.unparse(inner.func) + ast.unparse(inner):
                    texts = [a for a in inner.args if isinstance(a, (ast.Constant, ast.JoinedStr, ast.BinOp))
                             and isinstance(readable(a), str) and len(readable(a)) > 40]
                    if texts:
                        return readable(texts[0])
    raise KeyError(f"{name} not found in {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", type=int, choices=[1, 2, 3, 4], help="only this stage")
    parser.add_argument("--markdown", action="store_true", help="as a document")
    args = parser.parse_args()

    asked = [a for a in ASKS if args.stage is None or a[0] == args.stage]
    if args.markdown:
        print("# What the pipeline asks Claude\n")
        print("Read out of the source by `tools/show_prompts.py`, in the order a run asks them.\n")
    stage_now = None
    for stage, decides, where, (kind, name), images in asked:
        text = unescape(source_of(ROOT / where, kind, name), name in FORMATTED)
        if args.markdown:
            if stage != stage_now:
                print(f"\n## Stage {stage}\n")
                stage_now = stage
            print(f"### {decides}\n")
            print(f"`{where}` &middot; `{name}` &middot; with {images}\n")
            print("```text")
            print(text.strip())
            print("```\n")
        else:
            if stage != stage_now:
                print(f"\n{'=' * 78}\nSTAGE {stage}\n{'=' * 78}")
                stage_now = stage
            print(f"\n--- {decides}")
            print(f"    {where}  {name}  |  with {images}\n")
            print(text.strip())
            print()


if __name__ == "__main__":
    main()
