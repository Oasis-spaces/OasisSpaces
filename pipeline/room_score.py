#!/usr/bin/env python3
"""A measured score for a built room, so a review that makes it worse is undone.

Claude's render check names what looks wrong, but it cannot measure: asked
which of two reconstructions is the better one, frontier models agree with
the true geometry about as often as a coin (docs/research-weaknesses-2026-10.md).
So each pass of stage 3's structure review is scored by what can be measured,
and a pass that lowers the score is rolled back, whatever the judge said.

  agreement  For every furniture or storage name the frames show (its outlines
             in at least placement.MIN_FRAMES frames), how well the best built
             box of that kind covers those outlines when projected through the
             frames' own cameras (placement.score: trimmed-mean IoU, less the
             miss penalty for frames that show no such piece); 0 when no box of
             that kind is built. The mean over the names.
  collision  The largest share of one built piece's volume that lies inside
             another's (pillows and the like on furniture are not pieces).
  walk       The share of the camera positions standing inside a piece tall
             enough that nobody could have filmed from there.

  score = agreement - COLLISION_WEIGHT * collision - WALK_WEIGHT * walk

Usage:
    python3 pipeline/room_score.py spaces/<name>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import placement  # noqa: E402
from semantics import room_vocabulary  # noqa: E402

PIECES = ("bed", "seat", "table", "wardrobe")   # the furniture types scored as pieces
COLLISION_WEIGHT = 0.5
WALK_WEIGHT = 1.0
WALK_MIN_SHARE = 0.4      # a piece reaching this share of the room's height cannot be filmed from inside
EPS = 0.02                # scores closer than this are the same room


class Evidence:
    """placement.mask_evidence per name, loaded once per run."""

    def __init__(self, space: Path, log=None):
        self.space = Path(space)
        self.log = log or (lambda text: None)
        self._cache: dict[str, dict] = {}

    def __call__(self, label: str) -> dict:
        if label not in self._cache:
            self._cache[label] = placement.mask_evidence(self.space, label, log=self.log)
        return self._cache[label]


def built_pieces(shapes: dict) -> list[tuple[int, dict]]:
    return [(i, b) for i, b in enumerate(shapes.get("boxes") or [])
            if b.get("build", True) and b.get("label") in PIECES]


def boxes_of(shapes: dict, label: str, vocabulary) -> list[tuple[int, dict]]:
    """The built boxes that stand for `label`: detected as it (or as the same
    kind of storage), or built as its furniture type without a detection."""
    found = []
    for i, b in built_pieces(shapes):
        detected = b.get("detected")
        if detected:
            if vocabulary.same_kind(detected, label):
                found.append((i, b))
        elif b.get("label") == vocabulary.build_as(label):
            found.append((i, b))
    return found


def overlap_share(a: dict, b: dict) -> float:
    """The share of the smaller box's volume inside the other."""
    lo = np.maximum(np.array(a["min"], float), np.array(b["min"], float))
    hi = np.minimum(np.array(a["max"], float), np.array(b["max"], float))
    inter = float(np.prod(np.maximum(hi - lo, 0.0)))
    smallest = min(float(np.prod(np.array(a["max"], float) - np.array(a["min"], float))),
                   float(np.prod(np.array(b["max"], float) - np.array(b["min"], float))))
    return inter / smallest if smallest > 0 else 0.0


def room_score(space: Path, shapes: dict, evidence_for, vocabulary=None) -> dict:
    """The score of `shapes` (a shapes.json in memory), with its parts."""
    space = Path(space)
    vocabulary = vocabulary or room_vocabulary(space)
    pieces = {}
    for label in vocabulary.with_role("furniture", "storage"):
        evidence = evidence_for(label)
        if len(evidence["frames"]) < placement.MIN_FRAMES:
            continue
        candidates = boxes_of(shapes, label, vocabulary)
        best, best_id = 0.0, None
        for i, b in candidates:
            total, _ = placement.score(np.array(b["min"], float), np.array(b["max"], float), evidence)
            if best_id is None or total > best:
                best, best_id = total, f"B{i}"
        pieces[label] = {"agreement": round(best, 3), "frames": len(evidence["frames"]),
                         "box": best_id, "boxes": [f"B{i}" for i, _ in candidates]}
    agreement = float(np.mean([p["agreement"] for p in pieces.values()])) if pieces else None

    built = built_pieces(shapes)
    collision = 0.0
    for n, (_, a) in enumerate(built):
        for _, b in built[n + 1:]:
            collision = max(collision, overlap_share(a, b))

    level = shapes.get("room_level") or {}
    height = level.get("height")
    cameras = shapes.get("cameras") or []
    tall = [b for _, b in built
            if height and (b["max"][2] - level.get("floor_z", b["min"][2])) >= WALK_MIN_SHARE * height]
    inside = 0
    for x, y in cameras:
        if any(b["min"][0] <= x <= b["max"][0] and b["min"][1] <= y <= b["max"][1] for b in tall):
            inside += 1
    walk = inside / len(cameras) if cameras else 0.0

    score = (agreement or 0.0) - COLLISION_WEIGHT * collision - WALK_WEIGHT * walk
    return {"score": round(score, 3), "agreement": None if agreement is None else round(agreement, 3),
            "collision": round(collision, 3), "walk": round(walk, 3), "pieces": pieces}


def compare(before: dict, after: dict) -> str:
    """'better', 'worse' or 'same' (within EPS)."""
    delta = after["score"] - before["score"]
    return "better" if delta > EPS else "worse" if delta < -EPS else "same"


def describe(result: dict) -> str:
    parts = [f"{label} {p['agreement']:.2f}" + (f" ({p['box']})" if p["box"] else " (no box)")
             for label, p in sorted(result["pieces"].items())]
    text = f"room score {result['score']:.2f}"
    if parts:
        text += ": " + ", ".join(parts)
    else:
        text += ": no piece with outlines to measure"
    if result["collision"]:
        text += f"; pieces overlap by {result['collision']:.0%}"
    if result["walk"]:
        text += f"; {result['walk']:.0%} of the walk stands inside a piece"
    return text


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    space = Path(sys.argv[1])
    shapes = json.loads((space / "shapes.json").read_text())
    result = room_score(space, shapes, Evidence(space, log=print))
    print(describe(result))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
