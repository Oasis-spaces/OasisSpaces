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
  missing    The share of the measured wall support that the built room
             drops. A wall's support is read off the dense cloud in place:
             it meets the floor along its run (seam), little of the room
             lies beyond it (a bogus wall through the middle fails this),
             and no other measured wall stands between it and the cameras
             (a next room's wall glimpsed through a door fails this). So a
             review may drop an unsupported wall freely, but dropping a
             supported one is measured loss and is rolled back.
  cut        The largest share of the room's cloud lying beyond a *built*
             wall: a kept wall that slices the room. Dropping such a wall
             raises the score; nothing before measured walls at all.

  score = agreement - COLLISION_WEIGHT * collision - WALK_WEIGHT * walk
          - DOORWAY_WEIGHT * doorway - WALL_WEIGHT * missing
          - WALL_CUT_WEIGHT * cut

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

PIECES = ("bed", "seat", "table", "wardrobe")   # the box types always scored as pieces; a block
                                                # whose detected name is real furniture counts too
                                                # (built_pieces) — a wood stove is built as a block
COLLISION_WEIGHT = 0.5
WALK_WEIGHT = 1.0
DOORWAY_WEIGHT = 0.5
WALL_WEIGHT = 0.5
WALL_CUT_WEIGHT = 0.5
WALK_MIN_SHARE = 0.4      # a piece reaching this share of the room's height cannot be filmed from inside
MIN_SILHOUETTE = 0.005    # a piece filling less of a frame than this is not judged in that frame
DOORWAY_SHARE = 0.5       # a piece counts as standing in a fixture when this much of it lies in the outline
                          # (a piece beside a door overlaps its outline a little from some angles)
DOORWAY_SHARE_ONE = 0.6   # ... or this much in a single frame (a door is often seen in one keyframe only)
THIN_M = 0.2              # a piece thinner than this in such an outline is the door itself, not a wardrobe
                          # the detector sometimes calls a door: it counts in full, not by the vote share
EPS = 0.02                # scores closer than this are the same room

# Wall support, measured on the dense cloud in the shapes frame. Metres where
# densify recorded the scale; shares of the cloud's extent where it did not
# (the same fallback shapes.py uses).
SEAM_BINS = 12            # the wall's run, split into bins: how much of it meets the floor
SEAM_BIN_POINTS = 3       # a bin is met with at least this many points at the wall's base
SEAM_BAND_M = 0.30        # the floor band reaches this high up the wall
PLANE_BAND_M = 0.10       # a point this close to the plane lies on the wall
CLEAR_M = 0.30            # a point this far beyond the wall is outside the room it bounds
OCCLUDE_M = 0.40          # a parallel measured wall this much nearer the cameras hides this one
OCCLUDE_COS = 0.94        # ... parallel within about 20 degrees
OCCLUDE_OVERLAP = 0.5     # ... overlapping at least this share of the shorter wall's run
OCCLUDER_MAX_INTERIOR = 0.25  # ... and itself credible: a plane the room lies beyond is no
                              # wall, and must not hide the real one standing behind it
MAX_CLOUD_POINTS = 200_000


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


def built_pieces(shapes: dict, vocabulary=None) -> list[tuple[int, dict]]:
    """The built boxes the score treats as pieces: the furniture types, and —
    given the vocabulary — a block whose detected name is real furniture or
    storage. The review builds a wood stove as a block, and a box the room
    has must not read "no box"; a pillow's block stays out (on_furniture),
    and so does a block nothing detected."""
    return [(i, b) for i, b in enumerate(shapes.get("boxes") or [])
            if b.get("build", True)
            and (b.get("label") in PIECES
                 or (vocabulary is not None and b.get("label") == "block"
                     and vocabulary.role(b.get("detected")) in ("furniture", "storage")))]


def boxes_of(shapes: dict, label: str, vocabulary) -> list[tuple[int, dict]]:
    """The built boxes that stand for `label`: detected as it (or as the same
    kind of storage), or built as its furniture type without a detection."""
    found = []
    for i, b in built_pieces(shapes, vocabulary):
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
        for i, b in built_pieces(shapes, vocabulary):
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
        for i, b in built_pieces(shapes, vocabulary):
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


def wall_support(P: np.ndarray, plane: dict, floor_z: float, cameras: np.ndarray,
                 unit: float, others: list[dict] = ()) -> dict:
    """One wall plane measured in place against the cloud `P` (shapes frame).

    seam      the share of the wall's run with points at its base, in the
              floor band: a real wall meets the floor along its length.
    interior  the share of the points in the wall's own corridor lying
              beyond it (outward of the cameras): a bogus wall through the
              room has the room itself beyond it. Measured within the run,
              so a narrow plane cannot hide its slice in a big cloud.
    occluded  another measured wall — parallel, nearer the cameras, itself
              credible, with its own points covering this wall's run —
              stands in front: the next room's wall seen through a doorway.
              Coverage is by the occluder's measured points, not its fitted
              rectangle, so a short stub cannot hide a long wall, and a
              wall fitted across an alcove's opening (where it has no
              points) does not hide the alcove's real back wall.
    support   seam * (1 - interior), 0 when occluded.
    """
    def frame(p):
        n = np.asarray(p["normal"], float)
        return (n / np.linalg.norm(n), np.asarray(p["axis_a"], float),
                np.asarray(p["center"], float), float(p["half_a"]))

    def body(n, a, c, half):
        """The plane's own points above the floor band, within its run."""
        t = (P - c) @ a
        d = (P - c) @ n
        return (np.abs(t) <= half) & (np.abs(d) < PLANE_BAND_M * unit) \
            & (P[:, 2] > floor_z + SEAM_BAND_M * unit)

    def seam_interior(n, a, c, half):
        inner = float(np.sign(np.median(
            (np.column_stack([cameras, np.full(len(cameras), c[2])]) - c) @ n)))
        inner = inner or 1.0
        t = (P - c) @ a
        d = (P - c) @ n
        in_run = np.abs(t) <= half
        on_wall = in_run & (np.abs(d) < PLANE_BAND_M * unit)
        at_base = on_wall & (P[:, 2] > floor_z - PLANE_BAND_M * unit) \
            & (P[:, 2] < floor_z + SEAM_BAND_M * unit)
        bins = np.floor((t[at_base] + half) / (2 * half) * SEAM_BINS).astype(int) \
            .clip(0, SEAM_BINS - 1)
        seam = float((np.bincount(bins, minlength=SEAM_BINS) >= SEAM_BIN_POINTS).sum() / SEAM_BINS)
        beyond = in_run & (-inner * d > CLEAR_M * unit)
        return seam, float(beyond.sum() / max(int(in_run.sum()), 1)), inner

    n, a, c, half = frame(plane)
    seam, interior, inner = seam_interior(n, a, c, half)

    occluded = False
    for other in others:
        n2, a2, c2, half2 = frame(other)
        if abs(float(n @ n2)) < OCCLUDE_COS:
            continue
        if float((c2 - c) @ (inner * n)) <= OCCLUDE_M * unit:
            continue                       # not clearly on the cameras' side of this wall
        if seam_interior(n2, a2, c2, half2)[1] > OCCLUDER_MAX_INTERIOR:
            continue                       # the room lies beyond it: no wall, no occluder
        t_mine = (P[body(n2, a2, c2, half2)] - c) @ a
        t_mine = t_mine[np.abs(t_mine) <= half]
        bins = np.floor((t_mine + half) / (2 * half) * SEAM_BINS).astype(int) \
            .clip(0, SEAM_BINS - 1)
        covered = float((np.bincount(bins, minlength=SEAM_BINS) >= SEAM_BIN_POINTS).sum()
                        / SEAM_BINS)
        if covered >= OCCLUDE_OVERLAP:
            occluded = True
            break
    support = 0.0 if occluded else seam * (1.0 - interior)
    return {"seam": round(seam, 3), "interior": round(interior, 3),
            "occluded": occluded, "support": round(support, 3)}


class WallEvidence:
    """The space's dense cloud, loaded once; each wall's measured support
    cached by its geometry, so the review's edits re-measure only the walls
    they moved."""

    def __init__(self, space: Path, log=None):
        self.space = Path(space)
        self.log = log or (lambda text: None)
        self._positions = None   # raw solve-frame positions, subsampled
        self._loaded = False
        self._frames: dict[bytes, np.ndarray] = {}
        self._cache: dict[tuple, dict] = {}
        try:
            self.unit = json.loads((self.space / "densify.json").read_text()) \
                .get("colmap_units_per_metre")
        except (OSError, ValueError):
            self.unit = None

    def _points(self, world) -> np.ndarray | None:
        if not self._loaded:
            self._loaded = True
            from pointcloud import load_ply
            path = next((p for p in (self.space / "cloud-dense.ply", self.space / "cloud.ply")
                         if p.exists()), None)
            if path is not None:
                points = load_ply(path).points.astype(np.float64)
                points = points[np.isfinite(points).all(axis=1)]
                if len(points) > MAX_CLOUD_POINTS:
                    keep = np.random.default_rng(7).choice(len(points), MAX_CLOUD_POINTS,
                                                           replace=False)
                    points = points[keep]
                self._positions = points
                self.log(f"wall evidence: {len(points):,} points from {path.name}")
        if self._positions is None:
            return None
        key = np.asarray(world, float).tobytes()
        if key not in self._frames:
            self._frames[key] = self._positions @ np.asarray(world, float).T
        return self._frames[key]

    def measured(self, shapes: dict) -> dict[str, dict]:
        """{W<i>: wall_support(...)} for every RANSAC wall, built or not."""
        level = shapes.get("room_level") or {}
        cameras = np.asarray(shapes.get("cameras") or [], float)
        # A plane shapes.py suspects is a furniture front ("behind") is not
        # measured: it holds no support worth protecting, and it must not
        # occlude the real wall standing behind it.
        walls = [(i, p) for i, p in enumerate(shapes.get("planes") or [])
                 if p.get("kind") == "wall" and p.get("source") != "inferred"
                 and p.get("points") and not p.get("behind")]
        if "floor_z" not in level or not len(cameras) or not walls or shapes.get("world") is None:
            return {}
        P = self._points(shapes["world"])
        if P is None or not len(P):
            return {}
        extent = float(np.linalg.norm(np.percentile(P, 98, 0) - np.percentile(P, 2, 0)))
        unit = self.unit or extent * 0.12      # without the scale: bands as shares of the room
        found = {}

        def geometry(p):
            return (tuple(np.round(np.asarray(p["normal"], float), 4)),
                    tuple(np.round(np.asarray(p["center"], float), 3)),
                    round(float(p["half_a"]), 3))

        for i, p in walls:
            others = [q for j, q in walls if j != i]
            # Occlusion depends on the other walls too, so a moved neighbour
            # (the review's wall-move edit) re-measures this one as well.
            key = (round(float(level["floor_z"]), 4), geometry(p),
                   tuple(sorted(geometry(q) for q in others)))
            if key not in self._cache:
                self._cache[key] = wall_support(P, p, float(level["floor_z"]), cameras, unit, others)
            found[f"W{i}"] = {**self._cache[key], "points": int(p["points"]),
                              "built": bool(p.get("build", True))}
        return found


def wall_terms(shapes: dict, walls: "WallEvidence | None") -> tuple[float, float, dict]:
    """(missing, cut, per-wall detail): the dropped share of the measured
    wall support, and the worst built wall's slice through the room."""
    if walls is None:
        return 0.0, 0.0, {}
    measured = walls.measured(shapes)
    mass = {k: w["support"] * w["points"] for k, w in measured.items()}
    total = sum(mass.values())
    missing = sum(m for k, m in mass.items() if not measured[k]["built"]) / total if total else 0.0
    cut = max((w["interior"] for w in measured.values() if w["built"]), default=0.0)
    return missing, cut, measured


def room_score(space: Path, shapes: dict, evidence_for, vocabulary=None, walls=None) -> dict:
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

    built = built_pieces(shapes, vocabulary)
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

    missing, cut, measured_walls = wall_terms(shapes, walls)

    score = ((agreement or 0.0) - COLLISION_WEIGHT * collision - WALK_WEIGHT * walk
             - DOORWAY_WEIGHT * doorway - WALL_WEIGHT * missing - WALL_CUT_WEIGHT * cut)
    return {"score": round(score, 3), "agreement": None if agreement is None else round(agreement, 3),
            "collision": round(collision, 3), "walk": round(walk, 3), "doorway": round(doorway, 3),
            "doorways": {k: round(v, 3) for k, v in doorways.items()},
            "wall_missing": round(missing, 3), "wall_cut": round(cut, 3),
            "walls": measured_walls, "pieces": pieces}


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
    if result.get("wall_missing"):
        dropped = sorted(k for k, w in result["walls"].items() if not w["built"] and w["support"])
        text += (f"; dropped walls held {result['wall_missing']:.0%} of the measured wall "
                 f"support ({', '.join(dropped)})")
    if result.get("wall_cut"):
        worst = max((k for k, w in result["walls"].items() if w["built"]),
                    key=lambda k: result["walls"][k]["interior"])
        text += f"; {worst} cuts the room off ({result['walls'][worst]['interior']:.0%} of it beyond)"
    return text


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    space = Path(sys.argv[1])
    shapes = json.loads((space / "shapes.json").read_text())
    result = room_score(space, shapes, Evidence(space, log=print),
                        walls=WallEvidence(space, log=print))
    print(describe(result))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
