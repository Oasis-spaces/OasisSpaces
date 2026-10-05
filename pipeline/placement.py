"""Place a piece of furniture where the video's own masks say it stands.

A box for a piece the points never captured (a mirrored cupboard, a bed under
clutter) can be fitted from 2D evidence instead: the detector found the piece
in some keyframes, SAM outlined it, and every frame's camera is known. A
candidate box projects into each of those frames as a silhouette; the box
whose silhouettes overlap the masks best, across the frames, is where the
piece stands. The search runs over the room's walls (furniture stands against
one), the position along the wall and the size, and never puts a box where
the phone was.

    evidence = mask_evidence(space, "wardrobe")      # masks, poses, lenses
    best = search(space, shapes, "wardrobe", evidence)   # box record + score

The score is the intersection-over-union of silhouette and mask, averaged
over the frames that detected the piece with the worst quarter left out (a
few false detections should not sink a real piece; a piece that fits only a
few frames should not pass), less half the share of the picture a candidate
covers in keyframes that did not detect it (a piece standing where the
detector saw bare wall). Below ACCEPT_SCORE nothing is placed.

The search is two-stage: a coarse pass over every wall, position and size
with the trimmed score finds the piece; a fine pass then sizes it on every
frame the found piece explains, averaged without trimming and over the whole
size range, because the one frame that shows a bed's near edge is exactly the
frame trimming would drop for a box that stops short.

Two things make the masks fairer evidence. Loose things lying on a piece (the
clothes and bags on a bed, from the object list's roles) are outlined too and
joined to its mask, since SAM outlines the visible bedding and would leave the
cluttered half out. And where the detector's rectangle touches the picture's
edge, the piece runs out of frame, so that frame is scored inside the
rectangle (widened a little) rather than over the whole picture: a sliver
view then counts for what it shows.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

GRID = 8                    # the scoring raster is the frame at 1/GRID
NEAR = 0.05                 # solve units: surface points nearer than this are behind the lens
MIN_DETECTION = 0.3         # detector confidence for a keyframe to count as evidence
ACCEPT_SCORE = 0.35         # mean IoU a placement must reach
MIN_FRAMES = 2              # evidence frames a placement must agree with
MISS_WEIGHT = 0.5           # penalty weight for covering keyframes that saw no such piece
TRIM = 0.25                 # share of the worst evidence frames left out of the average
SIZES_M = {                 # (min, max, step) of width along the wall, depth into the room, height
    "wardrobe": ((0.6, 1.8, 0.2), (0.4, 0.7, 0.1), (1.8, 2.1, 0.3)),
    "bed": ((0.9, 2.1, 0.3), (0.9, 2.1, 0.3), (0.4, 0.7, 0.15)),
    "table": ((0.8, 1.6, 0.2), (0.5, 0.8, 0.1), (0.7, 0.8, 0.1)),
    "seat": ((0.4, 0.9, 0.1), (0.4, 0.9, 0.1), (0.4, 0.5, 0.1)),
}
OFFSET_STEP_M = 0.1
TYPICAL_M = {"wardrobe": (1.0, 0.55, 2.0), "bed": (1.5, 1.9, 0.5), "table": (1.2, 0.6, 0.75), "seat": (0.5, 0.5, 0.45)}
PRIOR_WEIGHT = 0.03         # breaks ties between sizes the masks cannot tell apart, toward the typical size
MAX_INSTANCES = 2           # a room can hold two beds or two cupboards under one label
LOOSE_ROLES = {"loose"}                                  # object-list roles that lie on furniture
BEDDING = {"pillow", "blanket", "cushion", "duvet", "quilt", "bedsheet", "bed sheet"}
ON_TOP_SHARE = 0.5          # a loose detection joins a piece when this much of it lies inside the piece's box
EDGE = 0.02                 # a detection within this share of the frame's edge is cut off by it
WINDOW = 0.25               # how much a cut-off detection's rectangle is widened for scoring
FAMILY_WORDS = {"table": ("table", "desk"), "seat": ("chair", "stool", "sofa", "seat", "bench"),
                "bed": ("bed", "mattress"), "wardrobe": ("wardrobe", "cupboard", "almirah", "cabinet", "drawers", "shelf")}


def furniture_family(space: Path, label: str) -> str | None:
    """The furniture family a label belongs to, or None when it names no
    furniture (a fridge, a TV, a basket): the object list's build_as, else
    the name's words. Unlike size_family, nothing is assumed."""
    try:
        for o in json.loads((Path(space) / "objects.json").read_text()).get("objects") or []:
            if o.get("name") == label and o.get("build_as") in SIZES_M:
                return o["build_as"]
    except (OSError, ValueError):
        pass
    for family, words in FAMILY_WORDS.items():
        if any(w in label for w in words):
            return family
    return None


def size_family(space: Path, label: str) -> str:
    """Which SIZES_M entry fits `label`: the object list's build_as (objects.json,
    the pipeline's own mapping) when it has one, else by the name's words."""
    try:
        objects = json.loads((Path(space) / "objects.json").read_text()).get("objects") or []
        for o in objects:
            if o.get("name") == label and o.get("build_as") in SIZES_M:
                return o["build_as"]
    except (OSError, ValueError):
        pass
    for family, words in FAMILY_WORDS.items():
        if any(w in label for w in words):
            return family
    return "wardrobe"


# ----------------------------------------------------------------- geometry
def convex_hull(points: np.ndarray) -> np.ndarray:
    """Andrew's monotone chain; points (n, 2) -> hull vertices in order."""
    pts = np.unique(points, axis=0)
    if len(pts) < 3:
        return pts
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def turn(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    def half(seq):
        out = []
        for p in seq:
            while len(out) >= 2 and turn(out[-2], out[-1], p) <= 0:
                out.pop()
            out.append(p)
        return out

    lower, upper = half(pts), half(pts[::-1])
    return np.array(lower[:-1] + upper[:-1])


def box_surface_points(lo, hi, per_edge: int = 6) -> np.ndarray:
    """Corners and points along every edge of the box (lo, hi)."""
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    edges = [(0, 1), (0, 2), (1, 3), (2, 3), (4, 5), (4, 6), (5, 7), (6, 7), (0, 4), (1, 5), (2, 6), (3, 7)]
    along = [corners[a] + (corners[b] - corners[a]) * k / (per_edge + 1)
             for a, b in edges for k in range(1, per_edge + 1)]
    return np.vstack([corners, along])


def silhouette(lo, hi, view: dict, shape: tuple[int, int]) -> np.ndarray:
    """The box (room frame) seen by `view`, as a boolean raster of `shape`
    (rows, cols) at 1/GRID of the frame: the convex hull of its projected
    surface points, those in front of the lens."""
    from PIL import Image, ImageDraw

    X = box_surface_points(lo, hi) @ view["world"]            # room -> COLMAP frame (world is a rotation)
    Xc = (view["R"] @ X.T).T + view["t"]
    ahead = Xc[:, 2] > NEAR
    if ahead.sum() < 3:
        return np.zeros(shape, bool)
    Xc = Xc[ahead]
    uv = np.stack([view["fx"] * Xc[:, 0] / Xc[:, 2] + view["cx"],
                   view["fy"] * Xc[:, 1] / Xc[:, 2] + view["cy"]], 1) / GRID
    hull = convex_hull(uv)
    if len(hull) < 3:
        return np.zeros(shape, bool)
    canvas = Image.new("L", (shape[1], shape[0]), 0)
    ImageDraw.Draw(canvas).polygon([tuple(p) for p in hull], fill=1)
    return np.asarray(canvas, bool)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


# ----------------------------------------------------------------- evidence
def views_of(space: Path, names: list[str]) -> dict[str, dict]:
    """Pose and lens of these frames: {name: {R, t, world, fx, fy, cx, cy, width, height}}."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from densify import read_cameras_bin, read_images_bin
    from pointcloud import space_model_dir

    space = Path(space)
    model = space_model_dir(space)
    world = np.array(json.loads((space / "shapes.json").read_text())["world"], dtype=float)
    cameras = read_cameras_bin(model / "cameras.bin")
    images = {v["name"]: v for v in read_images_bin(model / "images.bin").values()}
    views = {}
    for name in names:
        info = images.get(name)
        if info is None:
            continue
        cam = cameras[info["camera_id"]]
        fx, fy, cx, cy = cam["params"][:4]
        views[name] = {"R": np.array(info["R"], float), "t": np.array(info["t"], float).reshape(3),
                       "world": world, "fx": fx, "fy": fy, "cx": cx, "cy": cy,
                       "width": cam["width"], "height": cam["height"]}
    return views


def loose_roles(space: Path) -> set[str]:
    """Names in the object list whose role says they lie on furniture, plus bedding."""
    names = set(BEDDING)
    try:
        for o in json.loads((Path(space) / "objects.json").read_text()).get("objects") or []:
            if o.get("role") in LOOSE_ROLES:
                names.add(o["name"])
    except (OSError, ValueError):
        pass
    return names


def on_top(piece: list, loose: list) -> bool:
    """Does at least ON_TOP_SHARE of the loose detection's rectangle lie inside the piece's?"""
    ix = max(0, min(piece[2], loose[2]) - max(piece[0], loose[0]))
    iy = max(0, min(piece[3], loose[3]) - max(piece[1], loose[1]))
    area = max(1, (loose[2] - loose[0]) * (loose[3] - loose[1]))
    return ix * iy / area >= ON_TOP_SHARE


def with_what_lies_on_it(piece_dets: list[dict], frame_dets: list[dict], loose: set[str]) -> list[dict]:
    """The piece's detections plus the loose things lying on it in this frame."""
    def touches(a, b):
        return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])

    extra = [d for d in frame_dets
             if d["label"] in loose and d.get("score", 1) >= MIN_DETECTION
             and any(on_top(p["box"], d["box"]) or (d["label"] in BEDDING and touches(p["box"], d["box"]))
                     for p in piece_dets)]
    return [dict(d) for d in piece_dets] + [dict(d) for d in extra]


def scoring_window(dets: list[dict], width: int, height: int) -> tuple[int, int, int, int] | None:
    """The raster rectangle to score within when a detection is cut off by
    the frame's edge (None otherwise): the detections' union, widened by WINDOW."""
    boxes = [d["box"] for d in dets]
    x0, y0 = min(b[0] for b in boxes), min(b[1] for b in boxes)
    x1, y1 = max(b[2] for b in boxes), max(b[3] for b in boxes)
    cut = x0 <= EDGE * width or y0 <= EDGE * height or x1 >= (1 - EDGE) * width or y1 >= (1 - EDGE) * height
    if not cut:
        return None
    w, h = x1 - x0, y1 - y0
    x0, x1 = max(0, x0 - WINDOW * w), min(width, x1 + WINDOW * w)
    y0, y1 = max(0, y0 - WINDOW * h), min(height, y1 + WINDOW * h)
    return int(x0 / GRID), int(y0 / GRID), int(np.ceil(x1 / GRID)), int(np.ceil(y1 / GRID))


def mask_evidence(space: Path, label: str, segmenter=None, log=print) -> dict:
    """The keyframes that detected `label` (densify.json), each with its SAM
    mask at the scoring raster (joined with what lies on it), its scoring
    window when the frame cuts it off, its pose and lens; and the keyframes
    that did not. Masks are cached in workspace/masks/<label>.v3.npz."""
    from PIL import Image

    space = Path(space)
    meta = json.loads((space / "densify.json").read_text())
    detections = meta.get("detections") or {}
    loose = loose_roles(space)
    hits = {name: [d for d in dets if d["label"] == label and d.get("score", 1) >= MIN_DETECTION]
            for name, dets in detections.items()}
    seen = sorted(n for n, d in hits.items() if d)
    unseen = sorted(n for n, d in hits.items() if not d)
    views = views_of(space, seen + unseen)
    cache = space / "workspace" / "masks" / f"{label.replace(' ', '-')}.v3.npz"
    masks: dict[str, np.ndarray] = {}
    if cache.exists():
        with np.load(cache) as stored:
            masks = {k: stored[k] for k in stored.files}
    missing = [n for n in seen if n not in masks and n in views]
    if missing:
        if segmenter is None:
            import sys

            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from semantics import Segmenter

            segmenter = Segmenter()
        for name in missing:
            view = views[name]
            shape = (view["height"] // GRID, view["width"] // GRID)
            dets = with_what_lies_on_it(hits[name], detections[name], loose)
            img = Image.open(space / "workspace" / "images" / name)
            segmenter.outline(img, dets, clip=False)        # the outline may run past the rectangle
            union = np.zeros(shape, bool)
            for d in dets:
                if "mask" in d:
                    small = Image.fromarray(d["mask"].astype(np.uint8) * 255).resize((shape[1], shape[0]), Image.NEAREST)
                    union |= np.asarray(small) > 0
                else:                                         # no outline: the detector's rectangle
                    x0, y0, x1, y1 = (int(v / GRID) for v in d["box"])
                    union[max(y0, 0):y1 + 1, max(x0, 0):x1 + 1] = True
            masks[name] = union
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, **masks)
        log(f"    outlined '{label}' in {len(missing)} keyframe(s), with what lies on it")
    frames = {}
    for n in seen:
        if n in masks and n in views:
            frames[n] = {"mask": masks[n], **views[n],
                         "window": scoring_window(hits[n], views[n]["width"], views[n]["height"])}
    return {"label": label, "frames": frames, "unseen": {n: views[n] for n in unseen if n in views}}


# ----------------------------------------------------------------- the search
def trimmed_mean(values: list[float]) -> float:
    """The mean with the lowest TRIM share dropped (never below MIN_FRAMES kept)."""
    kept = sorted(values, reverse=True)
    keep = max(MIN_FRAMES, int(round(len(kept) * (1 - TRIM))))
    kept = kept[:keep]
    return float(np.mean(kept)) if kept else 0.0


def score(lo, hi, evidence: dict, trim: bool = True) -> tuple[float, dict[str, float]]:
    """Trimmed-mean IoU over the evidence frames (plain mean with trim=False),
    less MISS_WEIGHT times the mean share of the picture covered in keyframes
    that saw no such piece."""
    per_frame = {}
    for name, view in evidence["frames"].items():
        mask = view["mask"]
        sil = silhouette(lo, hi, view, mask.shape)
        window = view.get("window")
        if window:                                        # cut off by the frame: judge what it shows
            x0, y0, x1, y1 = window
            per_frame[name] = iou(mask[y0:y1, x0:x1], sil[y0:y1, x0:x1])
        else:
            per_frame[name] = iou(mask, sil)
    if not per_frame:
        return 0.0, {}
    miss = 0.0
    for name, view in evidence["unseen"].items():
        shape = (view["height"] // GRID, view["width"] // GRID)
        miss += silhouette(lo, hi, view, shape).mean()
    miss = miss / len(evidence["unseen"]) if evidence["unseen"] else 0.0
    agreement = trimmed_mean(list(per_frame.values())) if trim else float(np.mean(list(per_frame.values())))
    return agreement - MISS_WEIGHT * miss, per_frame


def candidates(shapes: dict, label: str, units: float, fine: tuple | None = None, family: str | None = None):
    """Boxes against every built wall: (wall index, offset from the wall's
    start, width, depth, height), sizes from SIZES_M[family] in solve units;
    `fine` keeps the wall and nearly the offset of a (wall, offset, width,
    depth, height) result and tries every size for it."""
    sizes = SIZES_M[family or SIZES_M.get(label) and label or "wardrobe"]
    level = shapes.get("room_level") or {}
    room_centre = np.array((shapes.get("room") or {}).get("center", [0, 0])[:2], float)
    cameras = np.array(shapes.get("cameras") or np.zeros((0, 2)), float)
    for i, p in enumerate(shapes["planes"]):
        if p["kind"] != "wall" or not p.get("build", True):
            continue
        if fine and i != fine[0]:
            continue
        c = np.array(p["center"][:2], float)
        a = np.array(p["axis_a"][:2], float)
        a /= np.linalg.norm(a)
        n = np.array(p["normal"][:2], float)
        n /= np.linalg.norm(n)
        if np.dot(room_centre - c, n) < 0:
            n = -n
        length = 2 * p["half_a"]
        floor_z = level.get("floor_z", p["center"][2] - p["half_b"])
        ranges = []
        for k, (lo_m, hi_m, step_m) in enumerate(sizes):
            if fine:                                      # the whole range, at half the step
                ranges.append(np.arange(lo_m, hi_m + 1e-9, step_m / 2) * units)
            else:
                ranges.append(np.arange(lo_m, hi_m + 1e-9, step_m) * units)
        for width in ranges[0]:
            if width > length:
                continue
            if fine:
                offsets = fine[1] + np.array([-1.0, -0.5, 0.0, 0.5, 1.0]) * OFFSET_STEP_M * units * 0.5
                offsets = offsets[(offsets >= 0) & (offsets <= length - width)]
            else:
                offsets = np.arange(0.0, length - width + 1e-9, OFFSET_STEP_M * units)
            for depth in ranges[1]:
                for height in ranges[2]:
                    for offset in offsets:
                        t0 = -p["half_a"] + offset
                        corners = [c + t * a + k * depth * n for t in (t0, t0 + width) for k in (0.0, 1.0)]
                        xs, ys = [q[0] for q in corners], [q[1] for q in corners]
                        lo = np.array([min(xs), min(ys), floor_z])
                        hi = np.array([max(xs), max(ys), floor_z + height])
                        if len(cameras) and np.any((cameras[:, 0] >= lo[0]) & (cameras[:, 0] <= hi[0])
                                                   & (cameras[:, 1] >= lo[1]) & (cameras[:, 1] <= hi[1])):
                            continue                          # the phone was there
                        yield (i, offset, width, depth, height), lo, hi


def size_prior(key: tuple, family: str, units: float) -> float:
    """How far a candidate's size is from the family's typical one, 0 (typical) to 1."""
    typical = TYPICAL_M[family]
    width, depth, height = key[2] / units, key[3] / units, key[4] / units
    return float(np.mean([min(1.0, abs(v - t) / t) for v, t in zip((width, depth, height), typical)]))


def search(space: Path, shapes: dict, label: str, evidence: dict, units: float, log=print,
           min_frames: int = MIN_FRAMES) -> dict | None:
    """The best-scoring box for `label`, or None when nothing reaches
    ACCEPT_SCORE in at least `min_frames` frames. Coarse over every wall, then
    fine around the best. Sizes the masks cannot tell apart (a bed seen along
    its length, cut off by the frame) go to the typical size for the family."""
    if not evidence["frames"]:
        log(f"    no keyframe detected a {label}: nothing to place it by")
        return None
    best = None
    family = size_family(space, label)
    for key_, lo, hi in candidates(shapes, label, units, family=family):
        total, per_frame = score(lo, hi, evidence)
        total -= PRIOR_WEIGHT * size_prior(key_, family, units)
        if best is None or total > best["score"]:
            best = {"key": key_, "min": lo.tolist(), "max": hi.tolist(), "score": total, "frames": per_frame}
    if best is not None:
        # Size it on the frames it explains, every one counting.
        explained = {n: f for n, f in evidence["frames"].items() if best["frames"].get(n, 0) >= 0.2}
        if len(explained) >= MIN_FRAMES:
            sizing = {**evidence, "frames": explained}
            sized = None
            for key_, lo, hi in candidates(shapes, label, units, fine=best["key"], family=family):
                total, per_frame = score(lo, hi, sizing, trim=False)
                total -= PRIOR_WEIGHT * size_prior(key_, family, units)
                if sized is None or total > sized["score"]:
                    sized = {"key": key_, "min": lo.tolist(), "max": hi.tolist(), "score": total, "frames": per_frame}
            if sized is not None:
                # Keep the found piece's score and frames as its record; take the fine size and place.
                full, per_frame = score(np.array(sized["min"]), np.array(sized["max"]), evidence)
                best = {"key": sized["key"], "min": sized["min"], "max": sized["max"],
                        "score": full - PRIOR_WEIGHT * size_prior(sized["key"], family, units), "frames": per_frame}
    if best is None:
        log(f"    no room for a {label} against any wall off the walk")
        return None
    agreeing = sum(1 for v in best["frames"].values() if v >= 0.2)
    wall, offset, width, depth, height = best["key"]
    log(f"    {label} (sized as a {family}): best fit against W{wall}, {offset / units:.2f} m along it, "
        f"{width / units:.2f} x {depth / units:.2f} x {height / units:.2f} m, score {best['score']:.2f} "
        f"({agreeing} of {len(best['frames'])} frames agree: "
        + ", ".join(f"{n[6:11]} {v:.2f}" for n, v in sorted(best["frames"].items())) + ")")
    if best["score"] < ACCEPT_SCORE or agreeing < min_frames:
        return None
    return {"min": best["min"], "max": best["max"], "points": 0, "source": "masks", "detected": label,
            "label": label, "build": True, "color": [190, 185, 175],
            "reason": f"placed by its masks in {agreeing} keyframe(s), score {best['score']:.2f}, against W{wall}",
            "placement": {"wall": wall, "offset_m": round(offset / units, 2), "score": round(best["score"], 3),
                          "frames": {n: round(v, 3) for n, v in best["frames"].items()}}}


def instances(space: Path, shapes: dict, label: str, evidence: dict, units: float, log=print) -> list[dict]:
    """Up to MAX_INSTANCES boxes for `label`: the best fit, then the best fit
    to the frames it does not explain (a second cupboard under the same
    label), each needing one more agreeing frame than the last."""
    found = []
    remaining = dict(evidence["frames"])
    for n in range(MAX_INSTANCES):
        if len(remaining) < MIN_FRAMES + n:
            break
        box = search(space, shapes, label, {**evidence, "frames": remaining}, units, log=log,
                     min_frames=MIN_FRAMES + n)
        if box is None:
            break
        found.append(box)
        explained = {name for name, v in box["placement"]["frames"].items() if v >= 0.2}
        remaining = {name: f for name, f in remaining.items() if name not in explained}
        # the next instance must stand clear of this one
        shapes = {**shapes, "cameras": list(shapes.get("cameras") or []) + [box_footprint_centre(box)]}
    return found


def box_footprint_centre(box: dict) -> list[float]:
    return [(box["min"][0] + box["max"][0]) / 2, (box["min"][1] + box["max"][1]) / 2]


def evidence_sheet(evidence: dict, box: dict | None, out: Path, space: Path) -> Path | None:
    """Each evidence frame with its mask (blue) and the box's silhouette
    (red) over it, for the record and for Claude."""
    from PIL import Image

    if not evidence["frames"]:
        return None
    tiles = []
    for name, view in sorted(evidence["frames"].items()):
        frame = Image.open(Path(space) / "workspace" / "images" / name).convert("RGB")
        mask = view["mask"]
        small = frame.resize((mask.shape[1], mask.shape[0]))
        arr = np.asarray(small).astype(float)
        arr[mask] = arr[mask] * 0.5 + np.array([40, 90, 220]) * 0.5
        if box:
            sil = silhouette(np.array(box["min"]), np.array(box["max"]), view, mask.shape)
            arr[sil] = arr[sil] * 0.5 + np.array([230, 50, 50]) * 0.5
        tiles.append((name, Image.fromarray(arr.astype(np.uint8))))
    w, h = tiles[0][1].size
    sheet = Image.new("RGB", (len(tiles) * (w + 6) + 6, h + 24), "white")
    from PIL import ImageDraw

    d = ImageDraw.Draw(sheet)
    for n, (name, tile) in enumerate(tiles):
        d.text((6 + n * (w + 6), 4), name, fill="black")
        sheet.paste(tile, (6 + n * (w + 6), 20))
    out = Path(out)
    sheet.save(out)
    return out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Place a piece of furniture by its masks in the keyframes.")
    parser.add_argument("space", type=Path)
    parser.add_argument("label")
    parser.add_argument("--sheet", type=Path, help="write the evidence picture here")
    args = parser.parse_args()
    shapes = json.loads((args.space / "shapes.json").read_text())
    units = json.loads((args.space / "densify.json").read_text())["colmap_units_per_metre"]
    evidence = mask_evidence(args.space, args.label)
    found = instances(args.space, shapes, args.label, evidence, units)
    if args.sheet:
        print("wrote", evidence_sheet(evidence, found[0] if found else None, args.sheet, args.space))
    print(json.dumps([{k: box[k] for k in ("min", "max", "reason", "placement")} for box in found], indent=1))
