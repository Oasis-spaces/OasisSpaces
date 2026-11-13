#!/usr/bin/env python3
"""Find the gaps in a room's surfaces, and fill them only with what the video saw.

After stage 3 the room has its walls, floor and ceiling as planes, but the cloud
on them has holes. Some are holes only in the cloud: the dense cloud is built
from a few dozen keyframes, and other frames of the walk did see that floor or
wall. Others nobody filmed: the floor under the bed, the wall behind the
wardrobe, the corner the phone never turned to.

Nothing here is invented. This step:

  1. maps each surface (the floor and ceiling inside the room's outline, every
     built wall from floor to ceiling) on a 2 cm grid from the dense cloud, and
     marks what the cloud does not cover;
  2. sorts each gap by what explains it: hidden behind or under a piece, an
     opening (a door in that wall), or simply never filmed;
  3. recovers what it can from the frames themselves: each gap cell is
     projected into every solved frame; a frame counts only where nothing
     nearer stands in the way (the dense cloud's own depth, and the tracked
     masks of furniture and belongings), it faces the surface, and at least
     MIN_VIEWS frames from different places agree on the colour. Agreement
     from different viewpoints is what tells a surface apart from something
     standing in front of it;
  4. closes the cloud's own sampling gaps (a few centimetres between measured
     points) by interpolating the measured colours around them;
  5. leaves everything else unseen. It is written as unseen, not painted;
  6. shows Claude each surface: as measured, and with the recovered cells,
     beside frames that face it. Claude checks per gap that what was
     recovered is this surface (not clothes, a bed edge or a reflection), and
     which gaps are openings that must stay open.

cloud-gaps.ply holds the recovered and interpolated points (real pixels, but
not measured by the dense cloud, so the viewer can mark them); cloud-unseen.ply
holds the cells nothing saw, for the viewer to show as unknown. gaps.json
records each surface's coverage and each gap with Claude's decision.

    python3 pipeline/gaps.py spaces/<name> [--no-claude] [--cell 0.02]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pointcloud import PointCloud, load_ply, save_ply  # noqa: E402

CELL_M = 0.02                 # grid cell on each surface, metres
NEAR_SURFACE_M = 0.05         # points this close to a surface belong to it
CLOSE_CELLS = 3               # seen cells closed over this many cells: sparse sampling is not a gap
MIN_GAP_M2 = 0.02             # smaller holes are not shown to Claude
IN_FRONT_M = 0.9              # a piece this close in front of a wall hides it
MAX_SHOWN = 8                 # gaps per surface Claude decides on, largest first
SHOW_FRAMES = 3               # frames shown to Claude per surface
GAP_KINDS = {"hidden": "behind or under a piece", "opening": "an opening (a door in this wall)",
             "unfilmed": "never filmed by the dense cloud's frames"}
# Recovering from frames
FRAME_REDUCE = 4              # frames read at 1/4 size for colour
DEPTH_REDUCE = 8              # the cloud's depth drawn at 1/8 size, to test what is in the way
DEPTH_VOXEL_M = 0.01          # the cloud thinned to this before drawing its depth
DEPTH_WIDEN = 3               # each depth pixel takes the nearest of this many around it: closes the
                              # spaces between the cloud's points so no view slips through a piece
IN_THE_WAY_M = 0.04           # something this much nearer than the cell hides it (plus 3% of the distance)
MIN_FACING = 0.3              # cosine between the view ray and the surface normal
MIN_VIEWS = 2                 # frames that must see a cell
GAIN_MIN_CELLS = 150          # measured cells a frame needs to have its exposure corrected
REFERENCE_CELLS = 4000        # measured cells per surface that give each frame's exposure
PERIOD_RANGE_M = (0.15, 1.3)  # tile and panel sizes looked for
PERIOD_MIN_CORR = 0.3         # autocorrelation peak that counts as a repeating pattern
PERIOD_MIN_CONTRAST = 20.0    # measured luminance spread (0..255) below which a surface is plain
PERIOD_REPEAT_SHARE = 0.5     # the second repeat must keep this share of the first's correlation
MIN_PATCH_M2 = 0.01           # recovered cells must form patches this large: lone cells are chance agreement
MIN_BASELINE_M = 0.15         # and stand at least this far apart
MAX_SPREAD = 18.0             # their brightness may differ this much (median absolute, 0..255)
UNSEEN_COLOUR = (128, 128, 128)


# ---------------------------------------------------------------- surfaces
def surfaces(shapes: dict, metre: float) -> list[dict]:
    """Each surface as a 2D frame in scene units: origin, two in-plane axes (u
    along, v up or across), normal, its size, and a mask of the cells that lie
    inside the room (the outline for floor and ceiling)."""
    level = shapes["room_level"]
    floor_z, height = level["floor_z"], level["height"]
    out = []
    polygons = (shapes.get("outline") or {}).get("polygons")
    if not polygons:
        room = shapes["room"]
        u, v, c = np.array(room["axis_u"]), np.array(room["axis_v"]), np.array(room["center"])
        polygons = [[(c + su * room["half_u"] * u + sv * room["half_v"] * v).tolist()
                     for su, sv in ((1, 1), (1, -1), (-1, -1), (-1, 1))]]
    pts = np.array([p for poly in polygons for p in poly], float)
    lo, hi = pts.min(0), pts.max(0)
    for name, z, up in (("floor", floor_z, 1.0), ("ceiling", floor_z + height, -1.0)):
        out.append({"name": name, "kind": name, "origin": np.array([lo[0], lo[1], z]),
                    "u": np.array([1.0, 0, 0]), "v": np.array([0, 1.0, 0]), "normal": np.array([0, 0, up]),
                    "size": (hi[0] - lo[0], hi[1] - lo[1]), "polygons": polygons})
    for i, p in enumerate(shapes.get("planes", [])):
        if p.get("label") != "wall" or p.get("build") is False:
            continue
        n = np.array(p["normal"], float)
        if abs(n[2]) > 0.2:
            continue
        a = np.array(p["axis_a"], float)
        a[2] = 0
        a /= np.linalg.norm(a)
        c = np.array(p["center"], float)
        origin = np.array([*(c[:2] - p["half_a"] * a[:2]), floor_z])
        # Faces into the room: towards the outline's middle.
        middle = np.array([*(pts.mean(0)), floor_z])
        n = np.array([n[0], n[1], 0.0]) / np.linalg.norm(n[:2])
        if np.dot(middle - origin, n) < 0:
            n = -n
        out.append({"name": f"wall W{i}", "kind": "wall", "index": i, "origin": origin, "u": a,
                    "v": np.array([0, 0, 1.0]), "normal": n, "size": (2 * p["half_a"], height),
                    "inferred": p.get("source") == "inferred"})
    return out


def inside_polygons(polygons, x, y) -> np.ndarray:
    """Even-odd rule over several polygons, for arrays of points."""
    inside = np.zeros(len(x), bool)
    for poly in polygons:
        poly = np.asarray(poly, float)
        j = len(poly) - 1
        for i in range(len(poly)):
            xi, yi = poly[i]
            xj, yj = poly[j]
            crosses = ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / np.where(yj - yi == 0, 1e-12, yj - yi) + xi)
            inside ^= crosses
            j = i
    return inside


def grid(surface: dict, scene: np.ndarray, colours: np.ndarray, cell: float, near: float, labels=None,
         in_front=(), weights=None):
    """The surface's colour image (rows: v, cols: u), the cells seen, the cells
    that are part of the surface, and the in-plane coordinates of each cell.
    With `weights` (each point's opacity, for a cloud made from a splat), a cell
    is seen when the opacity stacked on it reaches SEEN_ALPHA, as a camera
    looking at the surface would see it, and colours are opacity-weighted."""
    w, h = surface["size"]
    cols, rows = max(2, int(np.ceil(w / cell))), max(2, int(np.ceil(h / cell)))
    rel = scene - surface["origin"]
    d = rel @ surface["normal"]
    su, sv = rel @ surface["u"], rel @ surface["v"]
    on = (np.abs(d) < near) & (su >= 0) & (su < w) & (sv >= 0) & (sv < h)
    c, r = (su[on] / cell).astype(int), (sv[on] / cell).astype(int)
    flat = r * cols + c
    w_on = np.ones(int(on.sum())) if weights is None else np.asarray(weights, float)[on]
    count = np.bincount(flat, weights=w_on, minlength=rows * cols).reshape(rows, cols)
    image = np.zeros((rows, cols, 3))
    for k in range(3):
        image[..., k] = np.bincount(flat, weights=colours[on, k] * w_on, minlength=rows * cols).reshape(rows, cols)
    if weights is None:
        seen = count > 0
    else:
        clear = np.bincount(flat, weights=np.log1p(-np.clip(w_on, 0, 0.999)), minlength=rows * cols)
        seen = (1 - np.exp(clear.reshape(rows, cols))) >= SEEN_ALPHA
    image[count > 0] /= count[count > 0][:, None]
    named = np.zeros((rows, cols))
    front = np.zeros((rows, cols))
    if labels is not None:
        named = np.bincount(flat, weights=(labels[on] > 0) * w_on, minlength=rows * cols).reshape(rows, cols)
        if len(in_front):
            front = np.bincount(flat, weights=np.isin(labels[on], in_front) * w_on,
                                minlength=rows * cols).reshape(rows, cols)
    surface["named_share"] = np.where(seen, named / np.maximum(count, 1e-9), 0.0)
    # A cell whose points are mostly a piece standing in front (the headboard, a
    # wardrobe's doors, clothes) shows that piece, not this surface: unseen.
    seen = seen & ~(front > 0.5 * np.maximum(count, 1e-9))
    uu, vv = np.meshgrid((np.arange(cols) + 0.5) * cell, (np.arange(rows) + 0.5) * cell)
    if "polygons" in surface:
        world = surface["origin"][:2] + uu[..., None] * surface["u"][:2] + vv[..., None] * surface["v"][:2]
        part = inside_polygons(surface["polygons"], world[..., 0].ravel(), world[..., 1].ravel()).reshape(rows, cols)
    else:
        part = np.ones((rows, cols), bool)
    return image.astype(np.float32), seen, part, (uu, vv)


# ---------------------------------------------------------------- gaps
def find_gaps(seen: np.ndarray, part: np.ndarray, cell: float):
    """Connected regions of the surface that nothing saw, after closing over
    the spacing of the cloud's own samples."""
    import cv2

    kernel = np.ones((2 * CLOSE_CELLS + 1, 2 * CLOSE_CELLS + 1), np.uint8)
    closed = cv2.morphologyEx(seen.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
    holes = part & ~closed
    count, labels = cv2.connectedComponents(holes.astype(np.uint8), connectivity=8)
    regions = []
    for k in range(1, count):
        mask = labels == k
        area = mask.sum() * cell * cell
        regions.append({"mask": mask, "area": area})
    return holes, closed, sorted(regions, key=lambda r: -r["area"])


def cover(surface: dict, boxes: list[dict], uv, metre: float, cell: float, shape) -> dict[str, np.ndarray]:
    """For each piece, the surface cells it stands on or in front of."""
    uu, vv = uv
    covered = {}
    for b in boxes:
        lo, hi = np.array(b["min"], float), np.array(b["max"], float)
        corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        rel = corners - surface["origin"]
        d = rel @ surface["normal"]
        if surface["kind"] == "wall" and (d.min() > IN_FRONT_M * metre or d.max() < -0.3 * metre):
            continue
        if surface["kind"] == "ceiling" and d.min() > 0.4 * metre:
            continue
        if surface["kind"] == "floor" and d.min() > 0.4 * metre:
            continue                       # a piece up on the wall does not hide the floor
        pu, pv = rel @ surface["u"], rel @ surface["v"]
        mask = (uu >= pu.min()) & (uu <= pu.max()) & (vv >= pv.min()) & (vv <= pv.max())
        if mask.any():
            covered[b["name"]] = mask
    return covered


# ---------------------------------------------------------------- frames
def project(points: np.ndarray, view: dict) -> tuple[np.ndarray, np.ndarray]:
    """Pixels (full-size, distorted as the frame was taken) and camera depth of
    solve-frame points."""
    cam = points @ view["R"].T + view["t"]
    z = cam[:, 2]
    zs = np.where(z > 1e-6, z, 1e-6)
    x, y = cam[:, 0] / zs, cam[:, 1] / zs
    k1, k2, p1, p2 = view["dist"]
    r2 = x * x + y * y
    radial = 1 + k1 * r2 + k2 * r2 * r2
    xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    K = view["K"]
    return np.column_stack([K[0, 0] * xd + K[0, 2], K[1, 1] * yd + K[1, 2]]), z


def load_views(space: Path, metre: float, mask_names: list[str], log=print) -> list[dict]:
    """Every solved frame: pose, lens, the frame at 1/FRAME_REDUCE, and the
    tracked masks of the pieces and belongings that stand in front of
    surfaces."""
    import cv2
    from densify import read_cameras_bin, read_images_bin
    from splat_gsplat import lens

    model = space / "workspace" / "sparse" / "0"
    cameras = read_cameras_bin(model / "cameras.bin")
    images = read_images_bin(model / "images.bin")
    masks = {}
    for name in mask_names:
        path = space / "workspace" / "masks" / f"{name.replace(' ', '-')}.v3.npz"
        if path.exists():
            with np.load(path) as m:
                for frame in m.files:
                    masks[frame] = masks[frame] | m[frame] if frame in masks else m[frame].copy()
    views = []
    for info in sorted(images.values(), key=lambda i: i["name"]):
        cam = cameras[info["camera_id"]]
        K, dist = lens(cam)
        img = cv2.imread(str(space / "workspace" / "images" / info["name"]),
                         {4: cv2.IMREAD_REDUCED_COLOR_4, 2: cv2.IMREAD_REDUCED_COLOR_2}.get(FRAME_REDUCE,
                                                                                         cv2.IMREAD_COLOR))
        if img is None:
            continue
        views.append({"name": info["name"], "R": info["R"], "t": info["t"], "K": K, "dist": dist,
                      "size": (cam["width"], cam["height"]), "image": img[:, :, ::-1],
                      "centre": -info["R"].T @ info["t"], "mask": masks.get(info["name"])})
    log(f"  {len(views)} frames read to recover gaps from")
    return views


def occluders(cloud: PointCloud, metre: float, in_front_labels: set[int]) -> tuple[np.ndarray, np.ndarray]:
    """The dense cloud thinned to DEPTH_VOXEL_M, and which of its points are
    pieces or belongings (things that stand in front of surfaces)."""
    pts = cloud.points.astype(np.float64)
    keys = np.floor(pts / (DEPTH_VOXEL_M * metre)).astype(np.int64)
    _, keep = np.unique(keys, axis=0, return_index=True)
    front = (np.isin(cloud.labels[keep], list(in_front_labels)) if cloud.labels is not None and in_front_labels
             else np.zeros(len(keep), bool))
    return pts[keep], front


def depth_buffer(px: np.ndarray, z: np.ndarray, front: np.ndarray, size) -> tuple[np.ndarray, np.ndarray]:
    """The nearest depth per 1/DEPTH_REDUCE pixel of the points given, widened
    by DEPTH_WIDEN so the spaces between points do not let a view through, and
    whether that nearest thing is a piece or belonging."""
    import cv2

    w, h = size[0] // DEPTH_REDUCE, size[1] // DEPTH_REDUCE
    ok = (z > 0) & np.isfinite(px).all(1)
    q = np.where(ok[:, None], px / DEPTH_REDUCE, -1).astype(int)
    ok &= (q[:, 0] >= 0) & (q[:, 0] < w) & (q[:, 1] >= 0) & (q[:, 1] < h)
    flat, zz, ff = q[ok, 1] * w + q[ok, 0], z[ok], front[ok]
    order = np.lexsort((zz, flat))
    first = np.unique(flat[order], return_index=True)[1]
    depth = np.full(h * w, np.inf, np.float32)
    is_front = np.zeros(h * w, np.uint8)
    depth[flat[order][first]] = zz[order][first]
    is_front[flat[order][first]] = ff[order][first]
    kernel = np.ones((DEPTH_WIDEN, DEPTH_WIDEN), np.uint8)
    return cv2.erode(depth.reshape(h, w), kernel), cv2.dilate(is_front.reshape(h, w), kernel).astype(bool)


def through_box(start: np.ndarray, ends: np.ndarray, lo: np.ndarray, hi: np.ndarray, stop: float) -> np.ndarray:
    """Whether each segment from start to an end passes through the box,
    leaving out the last `stop` of its length (the cell may sit on the box)."""
    d = ends - start
    length = np.linalg.norm(d, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = (lo - start) / d
        t2 = (hi - start) / d
    t_near = np.nanmax(np.minimum(t1, t2), axis=1)
    t_far = np.nanmin(np.maximum(t1, t2), axis=1)
    t_end = 1 - stop / np.maximum(length, 1e-9)
    return (t_near <= t_far) & (t_far > 0) & (t_near < t_end)


def recover(targets: list[dict], views: list[dict], pts: np.ndarray, front: np.ndarray, metre: float,
            boxes=(), world=None) -> None:
    """Colour for each target's cells (solve frame) that frames saw with
    nothing in the way, agreeing across viewpoints. A target is {cells,
    normal, own}: `own` marks the occluder points on that surface itself,
    which never count as being in its way. Nothing is seen through the
    furniture's boxes either (`boxes`: (min, max) in the scene frame, `world`
    the solve-to-scene rotation): the cloud of a bed has holes, the bed does
    not. Adds `colours`, `ok` and `views` (how many frames saw each cell
    clearly) to each target."""
    found = [([], [], [], []) for _ in targets]
    for vi, view in enumerate(views):
        P, Z = project(pts, view)
        W, H = view["size"]
        centre_scene = view["centre"] @ world.T if world is not None else None
        for t, (idx_all, rgb_all, centre_all, view_all) in zip(targets, found):
            cells = t["cells"]
            px, z = project(cells, view)
            inside = (z > 0) & np.isfinite(px).all(1)
            inside &= (px[:, 0] >= 0) & (px[:, 0] < W - 1) & (px[:, 1] >= 0) & (px[:, 1] < H - 1)
            if not inside.any():
                continue
            ray = cells[inside] - view["centre"]
            facing = -(ray @ t["normal"]) / np.maximum(np.linalg.norm(ray, axis=1), 1e-9)
            idx = np.nonzero(inside)[0][facing >= MIN_FACING]
            if not len(idx):
                continue
            others = ~t["own"]
            depth, is_front = depth_buffer(P[others], Z[others], front[others], (W, H))
            p, z_in = px[idx], z[idx]
            q = np.minimum((p / DEPTH_REDUCE).astype(int), [depth.shape[1] - 1, depth.shape[0] - 1])
            nearest = depth[q[:, 1], q[:, 0]]
            margin = IN_THE_WAY_M * metre + 0.03 * z_in
            clear = z_in <= nearest + margin                        # nothing nearer in the way
            clear &= ~(is_front[q[:, 1], q[:, 0]] & (nearest < z_in + margin))   # nor a piece at the cell
            m = view["mask"]
            if m is not None:
                mq = np.minimum((p * m.shape[0] / H).astype(int), [m.shape[1] - 1, m.shape[0] - 1])
                clear &= ~m[mq[:, 1], mq[:, 0]]
            idx, p = idx[clear], p[clear]
            if len(idx) and len(boxes) and centre_scene is not None:
                ends = cells[idx] @ world.T
                blocked = np.zeros(len(idx), bool)
                for lo, hi in boxes:
                    blocked |= through_box(centre_scene, ends, lo, hi, NEAR_SURFACE_M * metre)
                idx, p = idx[~blocked], p[~blocked]
            if not len(idx):
                continue
            img = view["image"]
            c = np.minimum((p / FRAME_REDUCE).astype(int), [img.shape[1] - 1, img.shape[0] - 1])
            idx_all.append(idx)
            rgb_all.append(img[c[:, 1], c[:, 0]].astype(np.float32))
            centre_all.append(np.repeat(view["centre"][None], len(idx), 0))
            view_all.append(np.full(len(idx), vi))
    # Auto exposure and white balance change from frame to frame, so two frames that
    # saw the same spot disagree on its colour. Each frame's gain per channel is
    # measured against cells the dense cloud measured (targets with `truth`), then
    # divided out of every sample before frames are compared.
    gain = np.ones((len(views), 3), np.float32)
    ratios = [[] for _ in views]
    for t, (idx_all, rgb_all, _c, view_all) in zip(targets, found):
        if "truth" in t and idx_all:
            idx, rgb, vw = np.concatenate(idx_all), np.concatenate(rgb_all), np.concatenate(view_all)
            ratio = t["truth"][idx] / np.maximum(rgb, 8.0)
            for v in np.unique(vw):
                ratios[v].append(ratio[vw == v])
    for v, parts in enumerate(ratios):
        if parts and sum(len(r) for r in parts) >= GAIN_MIN_CELLS:
            gain[v] = np.clip(np.median(np.concatenate(parts), axis=0), 0.5, 2.0)
    for t, (idx_all, rgb_all, centre_all, view_all) in zip(targets, found):
        if "truth" in t:
            continue
        corrected = [r * gain[vw] for r, vw in zip(rgb_all, view_all)]
        t["colours"], t["ok"], t["views"] = agree(len(t["cells"]), idx_all, corrected, centre_all, metre)
    return gain


def agree(n: int, idx_all, rgb_all, centre_all, metre: float):
    """Per cell: the median colour of the frames that saw it, kept when
    MIN_VIEWS frames at least MIN_BASELINE_M apart agree on its brightness."""
    colours = np.zeros((n, 3), np.float32)
    views_seen = np.zeros(n, int)
    ok = np.zeros(n, bool)
    if not idx_all:
        return colours, ok, views_seen
    idx = np.concatenate(idx_all)
    rgb = np.concatenate(rgb_all)
    centres = np.concatenate(centre_all)
    order = np.argsort(idx, kind="stable")
    idx, rgb, centres = idx[order], rgb[order], centres[order]
    starts = np.r_[0, np.nonzero(np.diff(idx))[0] + 1]
    ends = np.r_[starts[1:], len(idx)]
    lum = rgb @ np.array([0.299, 0.587, 0.114], np.float32)
    for a, b in zip(starts, ends):
        cell = idx[a]
        views_seen[cell] = b - a
        if b - a < MIN_VIEWS:
            continue
        if np.linalg.norm(centres[a:b] - centres[a], axis=1).max() < MIN_BASELINE_M * metre:
            continue
        lv = lum[a:b]
        if np.median(np.abs(lv - np.median(lv))) > MAX_SPREAD:
            continue
        colours[cell] = np.median(rgb[a:b], axis=0)
        ok[cell] = True
    return colours, ok, views_seen


# ---------------------------------------------------------------- pattern
def period(lum: np.ndarray, seen: np.ndarray, cell: float) -> tuple[int | None, int | None, float]:
    """The repeat along columns and along rows (in cells) of a pattern, from the
    masked autocorrelation of what was seen; None along an axis without one."""
    m = seen.astype(np.float64)
    if m.sum() < 400 or lum[seen].std() < PERIOD_MIN_CONTRAST:
        return None, None, 0.0          # too little measured, or too plain to have a pattern
    x = np.where(seen, lum - lum[seen].mean(), 0.0)
    shape = [2 * s for s in x.shape]
    fx, fm = np.fft.rfft2(x, shape), np.fft.rfft2(m, shape)
    num = np.fft.irfft2(fx * np.conj(fx), shape)
    den = np.fft.irfft2(fm * np.conj(fm), shape)
    var = num[0, 0] / max(den[0, 0], 1)
    corr = num / np.maximum(den, 1) / max(var, 1e-9)
    corr[den < 0.2 * den[0, 0]] = 0
    lo, hi = int(PERIOD_RANGE_M[0] / cell), int(PERIOD_RANGE_M[1] / cell)
    found, strength = [], []
    for axis in (1, 0):                     # columns (u), then rows (v)
        line = corr[0, lo:hi] if axis == 1 else corr[lo:hi, 0]
        if not len(line):
            found.append(None)
            continue
        k = int(np.argmax(line))
        # A real pattern repeats: the second repeat must also correlate (checked when it lies in range).
        twice = corr[0, 2 * (lo + k)] if axis == 1 and 2 * (lo + k) < corr.shape[1] else (
            corr[2 * (lo + k), 0] if axis == 0 and 2 * (lo + k) < corr.shape[0] else None)
        repeats = twice is None or twice >= PERIOD_REPEAT_SHARE * line[k]
        if (line[k] >= PERIOD_MIN_CORR and repeats and 0 < k < len(line) - 1
                and line[k] >= line[k - 1] and line[k] >= line[k + 1]):
            found.append(lo + k)
            strength.append(float(line[k]))
        else:
            found.append(None)
    return found[0], found[1], (min(strength) if strength else 0.0)


def pattern_text(pu, pv, cell_m: float) -> str | None:
    found = [f"{p * cell_m * 100:.0f} cm" for p in (pu, pv) if p]
    return " by ".join(found) if found else None


def patches(mask: np.ndarray, min_cells: float) -> np.ndarray:
    """The mask without its lone cells and specks: opened by a 3x3 square,
    then only the connected patches of at least min_cells kept."""
    import cv2

    opened = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
    keep = np.zeros(count, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_cells
    return keep[labels] & mask


def interpolate(image: np.ndarray, seen: np.ndarray, todo: np.ndarray) -> np.ndarray:
    """The measured colours carried a few cells into the cloud's own sampling
    gaps (todo), nothing further."""
    import cv2

    src = np.clip(image, 0, 255).astype(np.uint8)
    mask = (todo & ~seen).astype(np.uint8)
    if not mask.any():
        return image
    out = cv2.inpaint(src, mask, CLOSE_CELLS, cv2.INPAINT_TELEA).astype(np.float32)
    return np.where(seen[..., None], image, out)


def picture(surface: dict, image, seen, part, gaps, filled, known, path: Path, cell: float):
    """Two panels: the surface as the cloud measured it, each gap outlined and
    numbered (red never filmed, yellow behind a piece, blue an opening); and
    with what was recovered from frames, unseen cells in a grey hatch."""
    import cv2

    def panel(img, have):
        p = np.clip(img, 0, 255).astype(np.uint8).copy()
        rows, cols = np.indices(have.shape)
        hatch = ((rows + cols) // 3) % 2 == 0
        p[~have] = np.where(hatch[~have, None], 90, 60)
        p[~part] = (25, 25, 25)
        return cv2.cvtColor(p[::-1].copy(), cv2.COLOR_RGB2BGR)   # rows run up the surface

    left = panel(image, seen)
    right = panel(filled, known)
    colour = {"unfilmed": (60, 60, 255), "hidden": (0, 220, 255), "opening": (255, 140, 40)}
    for g in gaps:
        m = g["mask"][::-1].astype(np.uint8)
        for side in (left, right):
            contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(side, contours, -1, colour[g["kind"]], 2)
            ys, xs = np.nonzero(m)
            cv2.putText(side, str(g["id"]), (int(xs.mean()), int(ys.mean())), 0, 0.8, (255, 255, 255), 3)
            cv2.putText(side, str(g["id"]), (int(xs.mean()), int(ys.mean())), 0, 0.8, colour[g["kind"]], 2)
    gap = np.full((left.shape[0], 12, 3), 255, np.uint8)
    both = np.hstack([left, gap, right])
    scale = min(1400 / both.shape[1], 900 / both.shape[0])      # large enough for Claude to read
    both = cv2.resize(both, None, fx=scale, fy=scale,
                      interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_NEAREST)
    cv2.imwrite(str(path), both)
    return path


def facing_frames(space: Path, surface: dict, world: np.ndarray, metre: float, count: int) -> list[Path]:
    """The chosen frames that look most squarely at the surface's middle."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from densify import read_images_bin

    model = space / "workspace" / "sparse" / "0"
    chosen_path = space / "frames-chosen.json"
    chosen = set()
    if chosen_path.exists():
        data = json.loads(chosen_path.read_text())
        chosen = set(data.get("room", []) + data.get("objects", []))
    images = read_images_bin(model / "images.bin")
    w, h = surface["size"]
    middle = surface["origin"] + surface["u"] * w / 2 + surface["v"] * h / 2
    scored = []
    for info in images.values():
        if chosen and info["name"] not in chosen:
            continue
        centre = (-info["R"].T @ info["t"]) @ world.T
        look = (info["R"].T @ np.array([0, 0, 1.0])) @ world.T
        to = middle - centre
        dist = np.linalg.norm(to)
        if dist < 1e-6:
            continue
        facing = np.dot(look, to / dist) * max(0.0, -np.dot(surface["normal"], to / dist))
        scored.append((facing, info["name"]))
    return [space / "workspace" / "images" / n for _, n in sorted(scored, reverse=True)[:count]]


PROMPT = """You are checking the room model of a real room filmed with a phone. Nothing in it may be invented: every colour must come from the video. This is the {name} ({kind}, {w:.2f} m x {h:.2f} m).

The first image has two panels, looking straight at the {kind} (for the floor and ceiling, from above), dark outside the room. Left: what the dense cloud measured; grey hatching where it has nothing. Each gap is outlined and numbered: red not in the cloud's frames, yellow behind or under a piece of furniture, blue an opening (a door detected in this wall). Right: the same, with cells recovered from other frames of the video that saw that spot with nothing in the way and agreed on its colour; grey hatching is what no frame saw, which stays unknown. The other images are frames of the video that face this {kind}.

The gaps:
{gaps}

For each gap decide:
- "keep": what was recovered there is this {kind} as the frames show it (or nothing was recovered and the hatching is right).
- "drop": what was recovered there is something else painted onto the {kind}: clothes, the side of the bed, a bag, a reflection, a shadow of a piece. It will be left unknown instead.
- "open": it is not this surface (a doorway, window, mirror or niche, a room beyond), so it stays empty.
Also say in a few words what the {kind} looks like in the frames ("beige 60 cm tiles", "plain white paint").

Reply as JSON: {{"looks": "...", "gaps": [{{"id": 1, "decision": "keep" or "drop" or "open", "why": "..."}}], "record": {{"doubts": [...]}}}}"""


def ask_claude(space: Path, surface: dict, gaps: list[dict], pic: Path, world, metre, advisor,
               pattern: str | None = None, pattern_after: str | None = None) -> dict | None:
    frames = facing_frames(space, surface, world, metre, SHOW_FRAMES)
    listing = "\n".join(f"{g['id']}: {g['area'] / metre ** 2:.2f} m², {GAP_KINDS[g['kind']]}"
                        + (f" ({', '.join(g['behind'])})" if g.get("behind") else "")
                        + f"; {g['recovered_share']:.0%} recovered from frames" for g in gaps)
    w, h = (s / metre for s in surface["size"])
    note = ("Pattern recognition on the measured part: " + (f"it repeats every {pattern}" if pattern else "no repeating pattern")
            + "; with the recovered cells added: " + (f"every {pattern_after}" if pattern_after else "none")
            + ". A recovered patch that breaks a pattern the measured part clearly has is suspect.")
    return advisor.ask_json(PROMPT.format(name=surface["name"], kind=surface["kind"], w=w, h=h, gaps=listing + "\n" + note),
                            [pic, *frames], max_tokens=1500)


# ---------------------------------------------------------------- main
SPLAT_MAX_THICK_M = 0.02      # splats thicker than this are haze filling the air, not a surface
SEEN_ALPHA = 0.5              # opacity stacked on a cell (from a splat) that counts as seeing it
SPLAT_SAMPLES = 32            # most samples laid across one splat's footprint
SPLAT_MAX_POINTS = 8_000_000  # the whole cloud made from the splat
SPLAT_LABEL_REACH_M = 0.05    # a sample takes the label of the dense-cloud point this close, if any


def quaternion_matrices(q: np.ndarray) -> np.ndarray:
    """(N, 3, 3) rotations of (N, 4) quaternions w, x, y, z."""
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    return np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], 1),
                     np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], 1),
                     np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], 1)], 1)


def splat_cloud(space: Path, splat: Path, metre: float, cell_m: float = CELL_M, log=print):
    """The trained splat as a cloud. A splat trains on every frame, so it covers
    what the dense cloud (a few dozen keyframes) misses. Walls and floors are
    mostly large, flat, thin splats, each covering an area: every opaque, thin
    splat is sampled across its footprint (within one standard deviation, about
    one sample per grid cell, at most SPLAT_SAMPLES) in its base colour, each
    sample carrying the splat's opacity: a surface is many faint splats stacked,
    so grid() adds their opacity up per cell. Thick soft splats are haze in the
    air and are left out. Labels come from the nearest dense-cloud point, so
    pieces and belongings are still known. Returns the cloud and the opacities."""
    from scipy.spatial import cKDTree
    from splat_gsplat import SH_C0, read_ply

    raw = read_ply(splat)
    opacity = 1 / (1 + np.exp(-raw["opacities"]))
    scales = np.exp(raw["scales"])
    keep = (opacity >= 0.01) & (scales.min(1) <= SPLAT_MAX_THICK_M * metre)
    means, scales, quats = raw["means"][keep], scales[keep], raw["quats"][keep]
    colours = np.clip((0.5 + SH_C0 * raw["sh0"][keep, 0]) * 255, 0, 255)
    order = np.sort(scales, axis=1)
    cell = cell_m * metre
    count = np.clip(np.ceil(np.pi * order[:, 2] * order[:, 1] / cell ** 2), 1, SPLAT_SAMPLES).astype(int)
    if count.sum() > SPLAT_MAX_POINTS:
        count = np.maximum(1, (count * SPLAT_MAX_POINTS / count.sum()).astype(int))
    owner = np.repeat(np.arange(len(means)), count)
    rng = np.random.default_rng(0)
    local = rng.standard_normal((len(owner), 3))
    local /= np.maximum(1.0, np.linalg.norm(local, axis=1, keepdims=True))   # within one standard deviation
    R = quaternion_matrices(quats.astype(np.float64))
    points = (means[owner] + np.einsum("nij,nj->ni", R[owner], local * scales[owner])).astype(np.float32)
    dense = load_ply(space / "cloud-dense.ply")
    labels = None
    if dense.labels is not None:
        dist, idx = cKDTree(dense.points).query(points, workers=-1)
        labels = np.where(dist < SPLAT_LABEL_REACH_M * metre, dense.labels[idx], 0).astype(dense.labels.dtype)
    log(f"  {splat.name}: {int(keep.sum()):,} of {len(keep):,} splats are thin enough to be surfaces; "
        f"{len(points):,} points laid across them")
    return PointCloud(points, colours[owner].astype(np.uint8), None, labels, dense.label_names), opacity[keep][owner]


def run(space: Path, use_claude: bool = True, cell_m: float = CELL_M, log=print, splat: Path | None = None) -> dict:
    import cv2

    shapes = json.loads((space / "shapes.json").read_text())
    metre = json.loads((space / "densify.json").read_text())["colmap_units_per_metre"]
    world = np.array(shapes["world"], float)
    dense = load_ply(space / "cloud-dense.ply")
    cloud, weights = splat_cloud(space, splat, metre, cell_m, log) if splat else (dense, None)
    scene = cloud.points.astype(np.float64) @ world.T
    colours = cloud.colors.astype(np.float64)
    cell, near = cell_m * metre, NEAR_SURFACE_M * metre
    pieces = [dict(b, name=b.get("identity") or b.get("detected") or b.get("label") or f"B{i}")
              for i, b in enumerate(shapes.get("boxes", [])) if b.get("build", True)]
    advisor = None
    if use_claude:
        from advisor import BRIEF_PATH, Advisor
        advisor = Advisor(system=BRIEF_PATH.read_text() if BRIEF_PATH.exists() else None)
        if not advisor.available:
            log(f"Claude is not available ({advisor.reason}); recovered cells are kept without its check")
            advisor = None
    # Labels of things that stand in front of a surface rather than being part
    # of it (furniture, storage, what sits on them, belongings).
    from semantics import planned_vocabulary
    vocabulary = planned_vocabulary(space)
    names = cloud.label_names or []
    front_roles = ("furniture", "storage", "on_furniture", "loose")
    in_front = np.array([i for i, n in enumerate(names) if vocabulary.role(n) in front_roles], int)
    masks_dir = space / "workspace" / "masks"
    mask_names = sorted({p.name.split(".")[0].replace("-", " ") for p in masks_dir.glob("*.v3.npz")}
                        if masks_dir.exists() else set())
    front_masks = [n for n in mask_names if vocabulary.role(n) in front_roles]
    views = load_views(space, metre, front_masks, log)
    pts, front = occluders(dense, metre, set(in_front.tolist()))
    pts_scene = pts @ world.T
    record = {"cell_m": cell_m, "rule": "only colours the video saw; the rest is unseen",
              "measured_from": splat.name if splat else "cloud-dense.ply", "surfaces": []}
    out = {"real": ([], [], []), "unseen": ([], [], [])}
    work = space / "gaps"
    work.mkdir(exist_ok=True)
    states, references = [], []
    for surface in surfaces(shapes, metre):
        image, seen, part, uv = grid(surface, scene, colours, cell, near, cloud.labels, in_front, weights)
        holes, closed, _ = find_gaps(seen, part, cell)
        covered = cover(surface, pieces, uv, metre, cell, seen.shape)
        # Each unseen cell takes its cause: an opening in the wall, the piece in
        # front of it (or standing on it), or none (never filmed). Gaps are the
        # connected regions of one cause.
        cause = np.full(seen.shape, "", object)
        for name, m in covered.items():
            cause[holes & m & (cause == "")] = name
        cause[holes & (cause == "")] = "-"
        gaps = []
        for name in sorted({c for c in cause.ravel() if c}):
            count, labels = cv2.connectedComponents((cause == name).astype(np.uint8), connectivity=8)
            for k in range(1, count):
                mask = labels == k
                g = {"mask": mask, "area": mask.sum() * cell * cell}
                if name == "-":
                    g["kind"] = "unfilmed"
                elif name in ("door", "window"):
                    g["kind"], g["behind"] = "opening", [name]
                else:
                    g["kind"], g["behind"] = "hidden", [name]
                g["small"] = g["area"] < MIN_GAP_M2 * metre ** 2
                gaps.append(g)
        gaps.sort(key=lambda g: -g["area"])
        # Pattern recognition: the repeat (tiles, panels) of what was measured. It is
        # an identifier, never a paint: it tells Claude what to expect and is
        # checked against what the frames gave back.
        lum = image @ np.array([0.299, 0.587, 0.114], np.float32)
        pu, pv, strength = period(lum, seen & part, cell_m)
        pattern = pattern_text(pu, pv, cell_m)
        # The cloud's sampling gaps: measured colours carried a few cells.
        between = closed & ~seen & part
        filled = interpolate(image, seen, between)
        # Real gaps: what other frames saw. Openings are not looked into.
        look = holes.copy()
        for g in gaps:
            if g["kind"] == "opening":
                look &= ~g["mask"]
        rr, cc = np.nonzero(look)
        uu, vv = uv
        scene_cells = surface["origin"] + uu[rr, cc, None] * surface["u"] + vv[rr, cc, None] * surface["v"]
        rel = pts_scene - surface["origin"]
        own = np.abs(rel @ surface["normal"]) < near
        ref_r, ref_c = np.nonzero(seen & part)
        if len(ref_r) > REFERENCE_CELLS:
            pick = np.random.default_rng(0).choice(len(ref_r), REFERENCE_CELLS, replace=False)
            ref_r, ref_c = ref_r[pick], ref_c[pick]
        ref_cells = surface["origin"] + uu[ref_r, ref_c, None] * surface["u"] + vv[ref_r, ref_c, None] * surface["v"]
        references.append({"cells": ref_cells @ world, "normal": surface["normal"] @ world,
                           "own": np.abs((pts_scene - surface["origin"]) @ surface["normal"]) < near,
                           "truth": image[ref_r, ref_c].astype(np.float32)})
        states.append({"surface": surface, "image": image, "seen": seen, "part": part, "uv": uv,
                       "gaps": gaps, "between": between, "filled": filled, "rr": rr, "cc": cc,
                       "target": {"cells": scene_cells @ world, "normal": surface["normal"] @ world, "own": own}})
    # Every built piece is solid except openings (a door or window is looked into, not through).
    boxes = [(np.array(b["min"], float), np.array(b["max"], float)) for b in pieces
             if b["name"] not in ("door", "window")]
    gain = recover([st["target"] for st in states] + references, views, pts, front, metre, boxes, world)
    log(f"  exposure corrected per frame: gain {np.percentile(gain, 5):.2f} to {np.percentile(gain, 95):.2f}")
    for st in states:
        surface, image, seen, part, uv = st["surface"], st["image"], st["seen"], st["part"], st["uv"]
        gaps, between, filled, rr, cc = st["gaps"], st["between"], st["filled"], st["rr"], st["cc"]
        uu, vv = uv
        got, ok = st["target"]["colours"], st["target"]["ok"]
        recovered = np.zeros(seen.shape, bool)
        recovered[rr[ok], cc[ok]] = True
        recovered = patches(recovered, MIN_PATCH_M2 / cell_m ** 2)
        filled[rr[ok], cc[ok]] = got[ok]
        for g in gaps:
            g["recovered_share"] = float((recovered & g["mask"]).sum() / max(g["mask"].sum(), 1))
        shown = [g for g in gaps if not g.get("small")][:MAX_SHOWN]
        for k, g in enumerate(shown, 1):
            g["id"] = k
        known = seen | between | recovered
        pu2, pv2, strength2 = period(image @ np.array([0.299, 0.587, 0.114], np.float32) * 0 + np.where(
            recovered | seen, filled @ np.array([0.299, 0.587, 0.114], np.float32), 0), recovered | seen, cell_m)
        pattern_after = pattern_text(pu2, pv2, cell_m)
        pic = picture(surface, image, seen, part, shown, filled, known,
                      work / f"{surface['name'].replace(' ', '-')}.jpg", cell)
        verdict = None
        if advisor is not None and shown:
            verdict = ask_claude(space, surface, shown, pic, world, metre, advisor, pattern, pattern_after)
        decisions = {}
        for d in (verdict or {}).get("gaps") or []:
            try:
                decisions[int(d.get("id"))] = d
            except (TypeError, ValueError):
                continue
        opened = np.zeros(seen.shape, bool)
        for g in gaps:
            d = decisions.get(g.get("id"), {}) if "id" in g else {}
            decision = d.get("decision") or ("open" if g["kind"] == "opening" else "keep")
            g["decision"], g["why"] = decision, d.get("why")
            if decision in ("drop", "open"):
                recovered &= ~g["mask"]
            if decision == "open":
                opened |= g["mask"]
        real = (between | recovered) & part
        unseen = part & ~seen & ~real & ~opened
        for key, mask, colour in (("real", real, None), ("unseen", unseen, UNSEEN_COLOUR)):
            r, c = np.nonzero(mask)
            pos = surface["origin"] + uu[r, c, None] * surface["u"] + vv[r, c, None] * surface["v"]
            out[key][0].append(pos)
            out[key][1].append(np.clip(filled[r, c], 0, 255) if colour is None
                               else np.tile(np.array(colour, float), (len(r), 1)))
            out[key][2].append(np.repeat(surface["normal"][None], len(pos), 0))
        area = part.sum() * cell_m ** 2
        m2 = lambda m: round(float((m & part).sum()) * cell_m ** 2, 2)   # noqa: E731
        entry = {"surface": surface["name"], "kind": surface["kind"], "area_m2": round(area, 2),
                 "seen_share": round(float((seen & part).sum() / max(part.sum(), 1)), 3),
                 "measured_m2": m2(seen), "interpolated_m2": m2(between), "recovered_m2": m2(recovered),
                 "unseen_m2": m2(unseen), "open_m2": m2(opened),
                 "known_share": round(float(((seen | real) & part).sum() / max(part.sum(), 1)), 3),
                 "looks": (verdict or {}).get("looks"), "pattern_measured": pattern,
                 "pattern_with_recovered": pattern_after,
                 "gaps": [{"id": g["id"], "area_m2": round(g["area"] / metre ** 2, 2), "kind": g["kind"],
                           "behind": g.get("behind"), "recovered_share": round(g["recovered_share"], 3),
                           "decision": g.get("decision"), "why": g.get("why")} for g in shown]}
        if verdict and (verdict.get("record") or {}).get("doubts"):
            entry["doubts"] = verdict["record"]["doubts"]
        record["surfaces"].append(entry)
        log(f"  {surface['name']}: {entry['seen_share']:.0%} measured, {entry['recovered_m2']} m² recovered "
            f"from frames, {entry['unseen_m2']} m² unseen; gaps: "
            + (", ".join(f"{g['id']} {g['kind']} {g['area_m2']} m² ({g['recovered_share']:.0%} recovered)"
                         f" -> {g['decision']}" for g in entry["gaps"]) or "none"))
    for key, path in (("real", "cloud-gaps.ply"), ("unseen", "cloud-unseen.ply")):
        pts, cols, nrm = out[key]
        points = np.vstack(pts) if pts else np.zeros((0, 3))
        save_ply(PointCloud((points @ world).astype(np.float32),        # back to the solve's frame
                            (np.vstack(cols) if cols else np.zeros((0, 3))).astype(np.uint8),
                            ((np.vstack(nrm) if nrm else np.zeros((0, 3))) @ world).astype(np.float32)),
                 space / path)
        record[f"{key}_points"] = int(len(points))
    record["claude"] = advisor is not None
    (space / "gaps.json").write_text(json.dumps(record, indent=1) + "\n")
    log(f"{record['real_points']:,} points from the video's own pixels -> cloud-gaps.ply; "
        f"{record['unseen_points']:,} unseen cells -> cloud-unseen.ply")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("space", type=Path)
    parser.add_argument("--no-claude", action="store_true")
    parser.add_argument("--cell", type=float, default=CELL_M, help="grid cell in metres")
    parser.add_argument("--splat", type=Path, help="measure the surfaces from this trained splat (e.g. "
                        "splat-gsplat.ply) instead of the keyframes' dense cloud")
    args = parser.parse_args()
    splat = args.splat if args.splat is None or args.splat.is_absolute() else args.space / args.splat
    run(args.space, use_claude=not args.no_claude, cell_m=args.cell, splat=splat)


if __name__ == "__main__":
    main()
