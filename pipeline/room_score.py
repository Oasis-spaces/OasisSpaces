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
  doorway    The largest share of a built piece's silhouette, through the
             frames' cameras, that lies inside a fixture's outline (a door, a
             window): a "wardrobe" standing where the frames show the door
             (the phone's detector once named the pan's door a wardrobe).

  score = agreement - COLLISION_WEIGHT * collision - WALK_WEIGHT * walk
          - DOORWAY_WEIGHT * doorway

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
from tracking import FIXTURES, FIXTURE_VOTE_SHARE  # noqa: E402

PIECES = ("bed", "seat", "table", "wardrobe")   # the furniture types scored as pieces
COLLISION_WEIGHT = 0.5
WALK_WEIGHT = 1.0
DOORWAY_WEIGHT = 0.5
WALK_MIN_SHARE = 0.4      # a piece reaching this share of the room's height cannot be filmed from inside
MIN_SILHOUETTE = 0.005    # a piece filling less of a frame than this is not judged in that frame
DOORWAY_SHARE = 0.5       # a piece counts as standing in a fixture when this much of it lies in the outline
                          # (a piece beside a door overlaps its outline a little from some angles)
DOORWAY_SHARE_ONE = 0.6   # ... or this much in a single frame (a door is often seen in one keyframe only)
THIN_M = 0.2              # a piece thinner than this in such an outline is the door itself, not a wardrobe
                          # the detector sometimes calls a door: it counts in full, not by the vote share
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


def in_doorways(shapes: dict, evidence_for, vocabulary) -> dict[str, float]:
    """For each built piece, the mean share of its silhouette lying inside a
    fixture's outline over the frames that show the piece: DOORWAY_SHARE over
    at least MIN_FRAMES frames, or DOORWAY_SHARE_ONE in fewer (the pan's door
    was detected in two keyframes and the box standing in it shows in one).
    Keyed by box id; only pieces with a share are listed."""
    names = set(FIXTURES) | set(vocabulary.with_role("fixture"))
    found = {}
    for label in sorted(names):
        evidence = evidence_for(label)
        if not evidence["frames"]:
            continue
        for i, b in built_pieces(shapes):
            lo, hi = np.array(b["min"], float), np.array(b["max"], float)
            shares = []
            for view in evidence["frames"].values():
                mask = view["mask"]
                sil = placement.silhouette(lo, hi, view, mask.shape)
                if sil.sum() < MIN_SILHOUETTE * sil.size:
                    continue
                shares.append(float((sil & mask).sum() / sil.sum()))
            if not shares:
                continue
            share = float(np.mean(shares))
            if (len(shares) >= placement.MIN_FRAMES and share >= DOORWAY_SHARE) or share >= DOORWAY_SHARE_ONE:
                found[f"B{i}"] = max(found.get(f"B{i}", 0.0), share)
    return found


def doorways_by_votes(space: Path, shapes: dict, vocabulary) -> dict[str, float]:
    """With tracks (tracking.py): a built piece filling the outline of a
    tracked object that the detector named a door or window at least
    FIXTURE_VOTE_SHARE of the time stands in a doorway by that share, whatever
    name won the vote. The pan's door was "wardrobe 3.7, door 1.9": every
    outline of it counted as wardrobe evidence, and no door outline was left
    to catch the box built in it."""
    from tracking import Tracks

    tracks = Tracks.load(Path(space) / "workspace" / "tracks")
    if tracks is None:
        return {}
    fixtures = set(FIXTURES) | set(vocabulary.with_role("fixture"))
    doorish = tracks.doorish(fixtures)
    suspects = [(t, tracks.fixture_share(t, fixtures)) for t in tracks.tracks if t["id"] in doorish]
    if not suspects:
        return {}
    frames = sorted({f for t, _ in suspects for f in t["frames"]})
    views = placement.views_of(space, placement.spaced(frames))
    units = None
    try:
        units = json.loads((Path(space) / "densify.json").read_text()).get("colmap_units_per_metre")
    except (OSError, ValueError):
        pass
    found = {}
    for t, vote_share in suspects:
        for i, b in built_pieces(shapes):
            lo, hi = np.array(b["min"], float), np.array(b["max"], float)
            thin = units and float(min(hi[0] - lo[0], hi[1] - lo[1])) / units <= THIN_M
            weight = 1.0 if thin else vote_share
            shares = []
            for name, view in views.items():
                if name not in t["frames"]:
                    continue
                shape = (view["height"] // placement.GRID, view["width"] // placement.GRID)
                mask = tracks.mask(t["id"], name, shape)
                sil = placement.silhouette(lo, hi, view, shape)
                if mask is None or sil.sum() < MIN_SILHOUETTE * sil.size:
                    continue
                shares.append(float((sil & mask).sum() / sil.sum()))
            if shares and np.mean(shares) >= DOORWAY_SHARE:
                found[f"B{i}"] = max(found.get(f"B{i}", 0.0), float(np.mean(shares)) * weight)
    return found


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

    doorways = in_doorways(shapes, evidence_for, vocabulary)
    for box_id, share in doorways_by_votes(space, shapes, vocabulary).items():
        doorways[box_id] = max(doorways.get(box_id, 0.0), share)
    doorway = max(doorways.values(), default=0.0)

    score = (agreement or 0.0) - COLLISION_WEIGHT * collision - WALK_WEIGHT * walk - DOORWAY_WEIGHT * doorway
    return {"score": round(score, 3), "agreement": None if agreement is None else round(agreement, 3),
            "collision": round(collision, 3), "walk": round(walk, 3), "doorway": round(doorway, 3),
            "doorways": {k: round(v, 3) for k, v in doorways.items()}, "pieces": pieces}


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
    if result.get("doorway"):
        worst = max(result["doorways"].items(), key=lambda kv: kv[1])
        text += f"; {worst[0]} stands in a doorway or window ({worst[1]:.0%} of it)"
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
