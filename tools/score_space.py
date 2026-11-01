#!/usr/bin/env python3
"""Score a processed space against measured ground truth.

Every number the pipeline reports about a room is relative until it is set
against a room measured another way. This does that, for any room that has a
truth folder:

  truth.json      {"cameras": [{"t": seconds from the video's start,
                                "centre": [x, y, z]}, ...],      metres, z up
                   "boxes": [{"label", "centre", "size", "axes"}, ...],
                   "source": "...", "room_type": "..."}
                  (a box's axes are three unit rows; size[i] runs along axes[i])
  surface.npy     points on the room's real surfaces in the same frame
                  (a LiDAR mesh's vertices, a laser scan), float32 N x 3

tools/benchmark_arkitscenes.py writes these for ARKitScenes scenes (real rooms
filmed handheld, with a LiDAR mesh and hand-labelled furniture boxes); a
tape-measured room of our own can be written the same way.

How the two are put in one frame: the space's camera centres, each at its
time in the video (workspace/video-grid/frames.json), are matched to the truth's
camera path, searching the clock offset between the two, and a similarity
(scale, rotation, shift) is fitted by least squares (Umeyama). The fitted
scale says how wrong the space's metric scale is; everything else is then
measured in the truth's metres:

  scale       the space's metre against the true metre (+10% = room 10% too big)
  path        camera centres after the fit (how well the two agree at all)
  floor, ceiling, height   the room's levels against the truth surface
  walls       for each built wall, the median distance from points on it to
              the nearest true surface point
  coverage    the share of true surface points at wall height (1.0-1.8 m above
              the floor) that lie within 15 cm of the room model (walls, boxes)
  footprint   the room's floor rectangle against the true floor (IoU, areas)
  furniture   each true box matched to the space's boxes of a compatible name
              (3D IoU, centre error, size error); recall and precision

Usage:
    python3 tools/score_space.py spaces/<name> <truth folder> [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(ROOT / "tools"))

OFFSET_SEARCH_S = 3.0      # clock offset between the space's video and the truth's path, searched both ways
OFFSET_STEP_S = 0.05
WALL_SAMPLES = (24, 12)    # points along x up each built wall
COVERAGE_BAND_M = (1.0, 1.8)
COVERAGE_REACH_M = 0.15
FLOOR_BAND_M = 0.04        # true floor points lie this close to the floor level
GRID_M = 0.05              # raster cell for footprints
MATCH_MIN_IOU = 0.05       # a true box and a space box overlapping less than this are not the same object
MAX_SURFACE_POINTS = 400_000

# The space's names (Claude's room vocabulary, GroundingDINO's phrases) folded onto
# ARKitScenes' 17 classes. A name not listed matches only by geometry.
CLASS_OF = {
    "bed": "bed", "sofa": "sofa", "couch": "sofa", "armchair": "chair", "chair": "chair",
    "office chair": "chair", "dining chair": "chair", "stool": "stool", "bench": "stool",
    "table": "table", "desk": "table", "coffee table": "table", "dining table": "table",
    "side table": "table", "bedside table": "table", "nightstand": "table",
    "wardrobe": "cabinet", "cabinet": "cabinet", "cupboard": "cabinet", "almirah": "cabinet",
    "chest of drawers": "cabinet", "dresser": "cabinet", "tv unit": "cabinet", "sideboard": "cabinet",
    "shelf": "shelf", "bookcase": "shelf", "bookshelf": "shelf",
    "television": "tv_monitor", "tv": "tv_monitor", "computer monitor": "tv_monitor", "monitor": "tv_monitor",
    "refrigerator": "refrigerator", "fridge": "refrigerator", "oven": "oven", "stove": "stove",
    "sink": "sink", "toilet": "toilet", "bathtub": "bathtub", "washing machine": "washer",
    "washer": "washer", "dishwasher": "dishwasher", "fireplace": "fireplace",
}
# Classes that stand for each other when matching (a dresser is filed as a cabinet in one
# dataset and a shelf unit in another).
SAME_KIND = [{"cabinet", "shelf"}, {"chair", "stool", "sofa"}, {"stove", "oven"}]


def compatible(a: str | None, b: str | None) -> bool:
    if not a or not b:
        return False
    return a == b or any(a in k and b in k for k in SAME_KIND)


def room_type(labels) -> str:
    """What kind of room the true furniture says it is."""
    labels = set(labels)
    if labels & {"toilet", "bathtub"}:
        return "bathroom"
    if labels & {"stove", "oven", "refrigerator", "dishwasher"}:
        return "kitchen"
    if "bed" in labels:
        return "bedroom"
    if labels & {"sofa", "tv_monitor", "fireplace"}:
        return "living room"
    if "table" in labels and "chair" in labels:
        return "dining or office"
    return "other"


# ------------------------------------------------------------------ alignment
def umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """s, R, t minimising |dst - (s R src + t)|²."""
    mu_s, mu_d = src.mean(axis=0), dst.mean(axis=0)
    a, b = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(b.T @ a / len(src))
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    s = float(np.trace(np.diag(S) @ D) / a.var(axis=0).sum())
    return s, R, mu_d - s * R @ mu_s


def fit_to_truth(times: np.ndarray, centres: np.ndarray, truth_t: np.ndarray, truth_c: np.ndarray):
    """The similarity from the space's solve frame to the truth's, over the clock
    offset that fits best. Returns (s, R, t, offset, rmse, used)."""
    best = None
    for offset in np.arange(-OFFSET_SEARCH_S, OFFSET_SEARCH_S + 1e-9, OFFSET_STEP_S):
        tt = times + offset
        inside = (tt >= truth_t[0]) & (tt <= truth_t[-1])
        if inside.sum() < 4:
            continue
        target = np.column_stack([np.interp(tt[inside], truth_t, truth_c[:, k]) for k in range(3)])
        s, R, t = umeyama(centres[inside], target)
        rmse = float(np.sqrt(np.mean(np.sum((s * centres[inside] @ R.T + t - target) ** 2, axis=1))))
        if best is None or rmse < best[4]:
            best = (s, R, t, float(offset), rmse, int(inside.sum()))
    if best is None:
        raise SystemExit("the space's frames and the truth's camera path never overlap in time")
    return best


# ------------------------------------------------------------------ the space
def space_cameras(space: Path) -> tuple[np.ndarray, np.ndarray]:
    """(time in the video, camera centre in the solve frame) for every placed frame."""
    from densify import read_images_bin
    from pointcloud import space_model_dir

    grid = json.loads((space / "workspace" / "video-grid" / "frames.json").read_text())
    images = read_images_bin(space_model_dir(space) / "images.bin")
    times, centres = [], []
    for info in images.values():
        step = grid["frames"].get(info["name"])
        if step is None:
            continue
        times.append(step / grid["fps"])
        centres.append(-info["R"].T @ info["t"])
    order = np.argsort(times)
    return np.array(times)[order], np.array(centres)[order]


def rect_points(center, axis_a, axis_b, half_a, half_b, n=WALL_SAMPLES) -> np.ndarray:
    a = np.linspace(-half_a, half_a, n[0])
    b = np.linspace(-half_b, half_b, n[1])
    A, B = np.meshgrid(a, b)
    return (np.asarray(center) + A.reshape(-1, 1) * np.asarray(axis_a) + B.reshape(-1, 1) * np.asarray(axis_b))


def box_corners(lo, hi) -> np.ndarray:
    return np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])


def box_surface(centre, size, axes, per_face=200) -> np.ndarray:
    """Points over an oriented box's six faces."""
    centre, size, axes = np.asarray(centre, float), np.asarray(size, float), np.asarray(axes, float)
    rng = np.random.default_rng(0)
    out = []
    for k in range(3):
        for sign in (-1, 1):
            uv = rng.uniform(-0.5, 0.5, (per_face, 3)) * size
            uv[:, k] = sign * size[k] / 2
            out.append(centre + uv @ axes)
    return np.vstack(out)


# ------------------------------------------------------------------ geometry on a raster
def footprint_mask(polygons, lo, shape) -> np.ndarray:
    from PIL import Image, ImageDraw

    img = Image.new("L", (shape[1], shape[0]), 0)
    draw = ImageDraw.Draw(img)
    for poly in polygons:
        draw.polygon([((x - lo[0]) / GRID_M, (y - lo[1]) / GRID_M) for x, y in poly], fill=1)
    return np.asarray(img, bool)


def oriented_iou(a: dict, b: dict) -> float:
    """3D IoU of two boxes standing upright (an oriented footprint times a height)."""
    pa, pb = footprint(a), footprint(b)
    allp = np.vstack([pa, pb])
    lo = allp.min(axis=0) - GRID_M
    shape = (int(np.ceil((allp[:, 1].max() - lo[1]) / GRID_M)) + 2, int(np.ceil((allp[:, 0].max() - lo[0]) / GRID_M)) + 2)
    ma, mb = footprint_mask([pa], lo, shape), footprint_mask([pb], lo, shape)
    inter_area = (ma & mb).sum() * GRID_M ** 2
    za, zb = z_range(a), z_range(b)
    inter_h = max(0.0, min(za[1], zb[1]) - max(za[0], zb[0]))
    inter = inter_area * inter_h
    union = ma.sum() * GRID_M ** 2 * (za[1] - za[0]) + mb.sum() * GRID_M ** 2 * (zb[1] - zb[0]) - inter
    return float(inter / union) if union > 0 else 0.0


def vertical_axis(box: dict) -> int:
    return int(np.argmax(np.abs(np.asarray(box["axes"])[:, 2])))


def footprint(box: dict) -> np.ndarray:
    axes, size, c = np.asarray(box["axes"], float), np.asarray(box["size"], float), np.asarray(box["centre"], float)
    up = vertical_axis(box)
    i, j = [k for k in range(3) if k != up]
    u, v = axes[i][:2] * size[i] / 2, axes[j][:2] * size[j] / 2
    return np.array([c[:2] + u + v, c[:2] + u - v, c[:2] - u - v, c[:2] - u + v])


def z_range(box: dict) -> tuple[float, float]:
    up = vertical_axis(box)
    h = box["size"][up] / 2
    return box["centre"][2] - h, box["centre"][2] + h


def plan_dims(box: dict) -> tuple[float, float, float]:
    """(longer side, shorter side, height) in metres."""
    up = vertical_axis(box)
    sides = sorted((box["size"][k] for k in range(3) if k != up), reverse=True)
    return sides[0], sides[1], box["size"][up]


def level(z: np.ndarray, near: str) -> float | None:
    """The floor (lowest dense level) or the ceiling (highest) of a surface cloud."""
    lo, hi = np.percentile(z, 0.5), np.percentile(z, 99.5)
    bins = np.arange(lo - 0.05, hi + 0.07, 0.02)
    hist, edges = np.histogram(z, bins)
    span = hi - lo
    window = (edges[:-1] < lo + 0.3 * span) if near == "floor" else (edges[:-1] > hi - 0.3 * span)
    if not window.any():
        return None
    k = np.flatnonzero(window)[np.argmax(hist[window])]
    if hist[k] < 0.01 * len(z):     # no real slab there (a scan with no ceiling)
        return None
    return float((edges[k] + edges[k + 1]) / 2)


# ------------------------------------------------------------------ the score
def score(space: Path, truth_dir: Path) -> dict:
    from scipy.spatial import cKDTree

    truth = json.loads((truth_dir / "truth.json").read_text())
    surface = np.load(truth_dir / "surface.npy").astype(np.float64)
    if len(surface) > MAX_SURFACE_POINTS:
        surface = surface[np.random.default_rng(0).choice(len(surface), MAX_SURFACE_POINTS, replace=False)]
    tree = cKDTree(surface)
    shapes = json.loads((space / "shapes.json").read_text())
    metre = json.loads((space / "densify.json").read_text())["colmap_units_per_metre"]
    world = np.array(shapes["world"])
    to_solve = lambda p: np.asarray(p, float) @ world          # scene -> camera solve (scene = world @ solve)

    times, centres = space_cameras(space)
    cams = truth["cameras"]
    truth_t = np.array([c["t"] for c in cams]); truth_c = np.array([c["centre"] for c in cams])
    s, R, t, offset, rmse, used = fit_to_truth(times, centres, truth_t, truth_c)
    to_truth = lambda p: s * to_solve(p) @ R.T + t             # scene -> truth metres
    out = {"truth": truth.get("source", str(truth_dir)), "room_type": truth.get("room_type") or room_type(b["label"] for b in truth["boxes"]),
           "frames_used": used, "clock_offset_s": round(offset, 2), "path_rmse_m": round(rmse, 3),
           # s turns solve units into true metres; the space believed `metre` units make a metre.
           "scale_error": round(s * metre - 1, 3)}

    # Floor, ceiling, height.
    lv = shapes.get("room_level") or {}
    room = shapes.get("room") or {}
    if lv and room:
        cx, cy = room["center"]
        floor_z = float(to_truth([[cx, cy, lv["floor_z"]]])[0, 2])
        ceil_z = float(to_truth([[cx, cy, lv["floor_z"] + lv["height"]]])[0, 2])
        true_floor, true_ceil = level(surface[:, 2], "floor"), level(surface[:, 2], "ceiling")
        up = to_truth([[cx, cy, 1.0]])[0] - to_truth([[cx, cy, 0.0]])[0]
        out.update(floor_error_m=round(floor_z - true_floor, 3) if true_floor is not None else None,
                   ceiling_error_m=round(ceil_z - true_ceil, 3) if true_ceil is not None else None,
                   height_m=round(ceil_z - floor_z, 3),
                   true_height_m=round(true_ceil - true_floor, 3) if None not in (true_floor, true_ceil) else None,
                   tilt_deg=round(float(np.degrees(np.arccos(abs(up[2]) / np.linalg.norm(up)))), 2))
    else:
        true_floor = level(surface[:, 2], "floor")

    # Walls: how far each built wall stands from the real surface.
    walls, model_points = [], []
    for p in shapes.get("planes", []):
        if p.get("label") != "wall" or p.get("build") is False:
            continue
        pts = to_truth(rect_points(p["center"], p["axis_a"], p["axis_b"], p["half_a"], p["half_b"]))
        d, _ = tree.query(pts)
        walls.append({"median_m": round(float(np.median(d)), 3), "within_10cm": round(float((d < 0.10).mean()), 3),
                      "length_m": round(2 * p["half_a"] * s, 2), "inferred": p.get("source") == "inferred"})
        model_points.append(pts)
    out["walls"] = walls
    out["wall_error_m"] = round(float(np.median([w["median_m"] for w in walls])), 3) if walls else None

    # Furniture: the space's built boxes in the truth's frame.
    ours = []
    for b in shapes.get("boxes", []):
        if b.get("build") is False:
            continue
        corners = to_truth(box_corners(b["min"], b["max"]))
        axes = (R @ world.T).T                                  # the scene's x, y, z in the truth frame (rows)
        size = (np.asarray(b["max"]) - np.asarray(b["min"])) * s
        name = (b.get("detected") or b.get("label") or "").lower()
        ours.append({"label": name, "class": CLASS_OF.get(name), "centre": corners.mean(axis=0).tolist(),
                     "size": size.tolist(), "axes": axes.tolist()})
        model_points.append(box_surface(corners.mean(axis=0), size, axes))
    truths = [{**b, "class": b["label"]} for b in truth["boxes"]]
    pairs = [(oriented_iou(a, b), i, j) for i, a in enumerate(truths) for j, b in enumerate(ours)
             if compatible(a["class"], b["class"])]
    taken_t, taken_o, matches = set(), set(), []
    for iou, i, j in sorted(pairs, reverse=True):
        if iou < MATCH_MIN_IOU or i in taken_t or j in taken_o:
            continue
        taken_t.add(i); taken_o.add(j)
        a, b = truths[i], ours[j]
        da, db = plan_dims(a), plan_dims(b)
        matches.append({"truth": a["label"], "space": b["label"], "iou": round(iou, 3),
                        "centre_error_m": round(float(np.linalg.norm(np.subtract(a["centre"], b["centre"]))), 3),
                        "size_error_m": [round(db[k] - da[k], 3) for k in range(3)]})
    missed = [truths[i]["label"] for i in range(len(truths)) if i not in taken_t]
    out["furniture"] = {"truth_objects": len(truths), "space_boxes": len(ours), "matched": matches, "missed": missed,
                        "unmatched_space_boxes": [ours[j]["label"] or "unlabelled" for j in range(len(ours)) if j not in taken_o],
                        "recall": round(len(matches) / len(truths), 3) if truths else None,
                        "precision": round(len(matches) / len(ours), 3) if ours else None,
                        "median_iou": round(float(np.median([m["iou"] for m in matches])), 3) if matches else None,
                        "median_centre_error_m": round(float(np.median([m["centre_error_m"] for m in matches])), 3) if matches else None}

    # Coverage: true surface at wall height that the room model accounts for.
    if true_floor is not None and model_points:
        band = surface[(surface[:, 2] > true_floor + COVERAGE_BAND_M[0]) & (surface[:, 2] < true_floor + COVERAGE_BAND_M[1])]
        if len(band):
            d, _ = cKDTree(np.vstack(model_points)).query(band, distance_upper_bound=COVERAGE_REACH_M)
            out["coverage"] = round(float(np.isfinite(d).mean()), 3)

    # Footprint: the room's floor rectangle against the true floor.
    if room and true_floor is not None:
        u, v = np.array(room["axis_u"]), np.array(room["axis_v"])
        c = np.array(room["center"])
        corners = [c + su * room["half_u"] * u + sv * room["half_v"] * v for su, sv in ((1, 1), (1, -1), (-1, -1), (-1, 1))]
        poly = to_truth([[x, y, lv.get("floor_z", 0.0)] for x, y in corners])[:, :2]
        floor_pts = surface[np.abs(surface[:, 2] - true_floor) < FLOOR_BAND_M][:, :2]
        if len(floor_pts):
            from scipy.ndimage import binary_closing, binary_fill_holes

            allp = np.vstack([poly, floor_pts])
            lo = allp.min(axis=0) - 2 * GRID_M
            shape = (int(np.ceil((allp[:, 1].max() - lo[1]) / GRID_M)) + 4, int(np.ceil((allp[:, 0].max() - lo[0]) / GRID_M)) + 4)
            true_mask = np.zeros(shape, bool)
            ij = ((floor_pts - lo) / GRID_M).astype(int)
            true_mask[ij[:, 1], ij[:, 0]] = True
            true_mask = binary_fill_holes(binary_closing(true_mask, iterations=3))
            ours_mask = footprint_mask([poly], lo, shape)
            inter, union = (true_mask & ours_mask).sum(), (true_mask | ours_mask).sum()
            out["footprint"] = {"iou": round(float(inter / union), 3) if union else None,
                                "area_m2": round(float(ours_mask.sum() * GRID_M ** 2), 2),
                                "true_area_m2": round(float(true_mask.sum() * GRID_M ** 2), 2)}
    return out


def summary(r: dict) -> str:
    f = r.get("furniture", {})
    fp = r.get("footprint", {})
    bits = [f"{r['room_type']}", f"scale {r['scale_error']:+.0%}", f"path {r['path_rmse_m']:.2f} m"]
    if r.get("floor_error_m") is not None:
        bits.append(f"floor {r['floor_error_m']:+.2f} m")
    if r.get("true_height_m") is not None:
        bits.append(f"height {r['height_m']:.2f} vs {r['true_height_m']:.2f} m")
    if r.get("wall_error_m") is not None:
        bits.append(f"walls {r['wall_error_m']:.2f} m off ({len(r['walls'])})")
    if fp.get("iou") is not None:
        bits.append(f"footprint IoU {fp['iou']:.2f} ({fp['area_m2']:.1f} vs {fp['true_area_m2']:.1f} m²)")
    if r.get("coverage") is not None:
        bits.append(f"coverage {r['coverage']:.0%}")
    if f.get("truth_objects"):
        bits.append(f"furniture {len(f['matched'])}/{f['truth_objects']} found"
                    + (f", IoU {f['median_iou']:.2f}, centres {f['median_centre_error_m']:.2f} m" if f.get("matched") else ""))
    return " · ".join(bits)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("space", type=Path)
    parser.add_argument("truth", type=Path)
    parser.add_argument("--json", type=Path, help="write the full score here (default <space>/score.json)")
    args = parser.parse_args()
    result = score(args.space.resolve(), args.truth.resolve())
    path = args.json or args.space / "score.json"
    path.write_text(json.dumps(result, indent=1) + "\n")
    print(summary(result))
    print(f"full score: {path}")


if __name__ == "__main__":
    main()
