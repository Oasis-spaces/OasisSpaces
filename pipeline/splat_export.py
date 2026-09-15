#!/usr/bin/env python3
"""Write a trained splat in the viewer's compact .splat format.

OpenSplat saves 62 float32 properties per Gaussian (248 bytes, spherical
harmonics included). splat-viewer/ also reads .splat: position, scale, colour
with opacity and rotation in 32 bytes, sorted largest and most opaque first so
the scene appears as it streams in. The walkthrough's 46 MB splat.ply becomes
6 MB, and loads where the .ply broke off: Python's http.server on macOS failed
mid-transfer with "No buffer space available". The view-dependent colour
(higher spherical harmonics) is dropped, as the viewer ignores it anyway.

It also writes <name>.view.json: a starting camera and field of view chosen
from the capture positions (see start_view). splat-viewer opens there instead
of at its demo camera, which sits somewhere no frame was taken and shows a smear.

Usage:
    python3 pipeline/splat_export.py spaces/<name>/splat.ply [out.splat] [--view-only]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

SH_C0 = 0.28209479177387814
NEEDED = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
          "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]


def read_splat_ply(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    end = raw.index(b"end_header\n") + len(b"end_header\n")
    header = raw[:end].decode("ascii").splitlines()
    if "format binary_little_endian 1.0" not in header:
        sys.exit(f"{path}: expected a binary little-endian PLY")
    props = []
    for line in header:
        tokens = line.split()
        if tokens[:1] == ["property"]:
            if tokens[1] != "float":
                sys.exit(f"{path}: property {tokens[-1]} is {tokens[1]}, expected float")
            props.append(tokens[-1])
    missing = [name for name in NEEDED if name not in props]
    if missing:
        sys.exit(f"{path}: not a Gaussian splat (missing {', '.join(missing)})")
    return np.frombuffer(raw[end:], dtype=np.dtype([(p, "<f4") for p in props]))


def ply_to_splat(src: Path, dst: Path) -> int:
    v = read_splat_ply(src)
    log_scales = np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=1)
    order = np.argsort(-np.exp(log_scales.sum(axis=1)) / (1 + np.exp(-v["opacity"])))
    v, log_scales = v[order], log_scales[order]
    out = np.zeros(len(v), dtype=[("position", "<f4", 3), ("scale", "<f4", 3),
                                  ("rgba", "u1", 4), ("rotation", "u1", 4)])
    out["position"] = np.stack([v["x"], v["y"], v["z"]], axis=1)
    out["scale"] = np.exp(log_scales)
    rgb = 0.5 + SH_C0 * np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1)
    alpha = 1 / (1 + np.exp(-v["opacity"]))
    out["rgba"] = np.clip(np.column_stack([rgb, alpha]) * 255, 0, 255).astype(np.uint8)
    quat = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=1)
    quat /= np.maximum(np.linalg.norm(quat, axis=1, keepdims=True), 1e-12)
    out["rotation"] = np.clip(quat * 128 + 128, 0, 255).astype(np.uint8)
    dst.write_bytes(out.tobytes())
    return len(out)


# The viewer shows this vertical field of view whatever the window's size (see
# splat-viewer/main.js), close to a phone's portrait frame (about 62 degrees).
VIEW_FOV_Y = 55.0
VIEW_ASPECT = 16 / 10
PITCH_DOWN = 10.0        # degrees: a level view, tipped a little towards the floor
NEAR_METRES = 1.0        # a surface closer than this fills the view out of focus
CLEAR_METRES = 0.35      # the camera must stand at least this far from any surface
BACK_OFF_METRES = (-0.5, 0.0, 0.5, 1.0)   # negative: a step forward
TURNS = (-25.0, 0.0, 25.0)
MAX_CAMERAS = 60
GRID = (32, 20)          # screen cells for scoring


def model_dir(space: Path) -> Path | None:
    """The camera model the splat was trained with."""
    candidates = []
    meta = space / "densify.json"
    if meta.exists():
        named = Path(json.loads(meta.read_text())["model_dir"])
        candidates += [named, space / "workspace" / "sparse" / named.name]
    candidates += sorted((p.parent for p in (space / "workspace" / "sparse").glob("*/images.bin")),
                         key=lambda d: -(d / "images.bin").stat().st_size)
    candidates.append(space / "splat-project")
    return next((d for d in candidates
                 if (d / "images.bin").exists() and (d / "points3D.bin").exists()), None)


def look_matrix(position: np.ndarray, forward: np.ndarray, up: np.ndarray) -> np.ndarray:
    """World-to-camera rotation rows (right, down, forward) for a level camera
    at `position` looking along `forward`; splat-viewer's camera looks along +z
    with image-y pointing down, like COLMAP's."""
    f = forward / np.linalg.norm(forward)
    down = -(up - f * (up @ f))
    down /= np.linalg.norm(down)
    right = np.cross(down, f)
    return np.stack([right, down, f])


def view_json(R: np.ndarray, position: np.ndarray, **extra) -> dict:
    t = -R @ position
    # Column-major world-to-camera matrix, the layout splat-viewer's view uses.
    view = [R[0][0], R[1][0], R[2][0], 0, R[0][1], R[1][1], R[2][1], 0,
            R[0][2], R[1][2], R[2][2], 0, t[0], t[1], t[2], 1]
    return {"viewMatrix": [round(float(v), 5) for v in view], "fovY": VIEW_FOV_Y, **extra}


def splat_blobs(path: Path, count: int = 120_000):
    """Positions, largest radius and opacity of the splat's Gaussians."""
    v = read_splat_ply(path)
    alpha = 1 / (1 + np.exp(-v["opacity"].astype(np.float64)))
    keep = np.flatnonzero(alpha > 0.1)
    if len(keep) > count:
        keep = np.random.default_rng(0).choice(keep, count, replace=False)
    v = v[keep]
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)
    radius = np.exp(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=1)
                    .astype(np.float64)).max(axis=1)
    return xyz, radius, alpha[keep]


# A patch of screen is blurry when most of the Gaussians making its surface
# are drawn wider than this share of the screen's height: a surface the camera
# only saw edge-on, or one right in front of the lens.
BLUR_SCREEN_SHARE = 0.09
SURFACE_BAND_METRES = 0.25
SMEAR_METRES = 1.5


def score_view(blobs, R: np.ndarray, position: np.ndarray, metre: float) -> tuple:
    """(score, coverage, blurry share, median depth in metres) of the viewer's
    picture from here: how much of the screen shows the splat, and how little
    of it is blur."""
    xyz, radius, alpha = blobs
    local = (xyz - position) @ R.T
    z = local[:, 2]
    tan_y = np.tan(np.radians(VIEW_FOV_Y / 2))
    tan_x = tan_y * VIEW_ASPECT
    ok = z > 0.05 * metre
    sx = local[ok, 0] / z[ok] / tan_x
    sy = local[ok, 1] / z[ok] / tan_y
    zs, rs, al = z[ok], radius[ok], alpha[ok]
    inside = (np.abs(sx) < 1) & (np.abs(sy) < 1)
    sx, sy, zs, rs, al = sx[inside], sy[inside], zs[inside], rs[inside], al[inside]
    cols, rows = GRID
    if len(zs) < 200:
        return (0.0, 0.0, 1.0, 0.0)
    cell = (((sy + 1) / 2 * rows).astype(int).clip(0, rows - 1) * cols
            + ((sx + 1) / 2 * cols).astype(int).clip(0, cols - 1))
    front = np.full(rows * cols, np.inf)
    solid = al > 0.3
    np.minimum.at(front, cell[solid], zs[solid])
    surface = zs <= front[cell] + SURFACE_BAND_METRES * metre
    counts = np.bincount(cell[surface], minlength=rows * cols)
    wide = (2 * 3 * rs / zs / tan_y / 2) > BLUR_SCREEN_SHARE   # drawn width / screen height
    wide_counts = np.bincount(cell[surface], weights=wide[surface].astype(float),
                              minlength=rows * cols)
    covered = counts >= 4
    blurry = covered & (wide_counts > 0.5 * np.maximum(counts, 1))
    near = covered & (front < NEAR_METRES * metre)
    # Big blobs close to the lens smear across the screen from wherever their
    # centre is, even just outside it: mark every cell they reach.
    close = (z > 0.05 * metre) & (z < SMEAR_METRES * metre) & (alpha > 0.15)
    reach = 3 * radius[close] / z[close] / tan_y          # drawn radius, in half-screen-heights
    big = reach > BLUR_SCREEN_SHARE * 2
    if big.any():
        cx = local[close, 0][big] / z[close][big] / tan_y   # centre, half-screen-heights
        cy = local[close, 1][big] / z[close][big] / tan_y
        gx = ((np.arange(cols) + 0.5) / cols * 2 - 1) * VIEW_ASPECT
        gy = (np.arange(rows) + 0.5) / rows * 2 - 1
        GX, GY = np.meshgrid(gx, gy)
        hits = np.zeros(rows * cols)
        for x, y, r in zip(cx, cy, reach[big]):
            hits += (((GX - x) ** 2 + (GY - y) ** 2) < (0.6 * r) ** 2).ravel()
        blurry |= hits >= 2
    coverage = float(covered.mean())
    bad = float((blurry | near).mean())
    depth = float(np.median(zs[surface])) / metre
    score = coverage * (1 - bad) ** 2 * min(1.0, depth / 1.5)
    return (score, coverage, bad, depth)


def start_view(space: Path, splat: Path | None = None) -> dict | None:
    """A viewer camera where the splat looks its best: at or a step behind a
    capture position, level, turned a little either way, and scored by
    score_view on the splat itself. The exact capture pose often sits pressed
    against a wall, or beside a cupboard filmed edge-on, which then fills part
    of the screen as a blur."""
    views = start_views(space, splat, 1)
    return views[0] if views else None


def start_views(space: Path, splat: Path | None = None, count: int = 6) -> list[dict]:
    """The `count` best-scoring start views (see start_view), best first, from
    capture positions spread through the video, so they show different parts
    of the room; the agent lets Claude choose among them."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from densify import read_images_bin

    model = model_dir(space)
    splat = splat or space / "splat.ply"
    if model is None or not splat.exists():
        return []
    images = list(read_images_bin(model / "images.bin").values())
    if not images:
        return []
    blobs = splat_blobs(splat)
    shapes_path = space / "shapes.json"
    if shapes_path.exists():
        up = np.array(json.loads(shapes_path.read_text())["world"])[2]
    else:
        from shapes import estimate_up

        up = estimate_up(model)
    meta = space / "densify.json"
    metre = (json.loads(meta.read_text()).get("colmap_units_per_metre") if meta.exists()
             else None) or float(np.linalg.norm(np.percentile(blobs[0], 95, 0)
                                                - np.percentile(blobs[0], 5, 0))) / 5
    from scipy.spatial import cKDTree

    tree = cKDTree(blobs[0][blobs[2] > 0.3])
    step = max(1, len(images) // MAX_CAMERAS)
    scored = []
    for order, info in enumerate(sorted(images, key=lambda im: im["name"])[::step]):
        centre = -info["R"].T @ info["t"]
        heading = info["R"][2] - up * (info["R"][2] @ up)
        if np.linalg.norm(heading) < 1e-6:
            continue
        heading /= np.linalg.norm(heading)
        side = np.cross(up, heading)
        for turn in TURNS:
            a = np.radians(turn)
            level = np.cos(a) * heading + np.sin(a) * side
            p = np.radians(PITCH_DOWN)
            forward = np.cos(p) * level - np.sin(p) * up
            R = look_matrix(centre, forward, up)
            for back in BACK_OFF_METRES:
                position = centre - level * back * metre
                if tree.query(position)[0] < CLEAR_METRES * metre:
                    continue  # standing inside furniture or a wall
                score, coverage, bad, depth = score_view(blobs, R, position, metre)
                scored.append((score, order, R, position, info["name"], turn, back,
                               coverage, bad, depth))
    # Best first, each from a capture position at least this far through the
    # video from the ones already taken.
    spacing = max(2, len(images) // step // (3 * count))
    views, taken = [], []
    for score, order, R, position, name, turn, back, coverage, bad, depth in sorted(
            scored, key=lambda v: -v[0]):
        if score <= 0 or any(abs(order - o) < spacing for o in taken):
            continue
        taken.append(order)
        views.append(view_json(R, position, frame=name, turn=turn, backMetres=back,
                               coverage=round(coverage, 2), blurShare=round(bad, 2),
                               medianDepthMetres=round(depth, 2), score=round(float(score), 3),
                               # The room's up direction and scale, so a viewer can walk
                               # level at walking speed (apps/SplatViewer).
                               up=[round(float(v), 5) for v in up], metre=round(float(metre), 5)))
        if len(views) == count:
            break
    return views


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    args = [a for a in sys.argv[1:] if a != "--view-only"]
    src = Path(args[0])
    dst = Path(args[1]) if len(args) > 1 else src.with_suffix(".splat")
    if "--view-only" not in sys.argv:
        count = ply_to_splat(src, dst)
        print(f"{count:,} gaussians -> {dst} ({dst.stat().st_size / 1e6:.1f} MB, "
              f"from {src.stat().st_size / 1e6:.1f} MB)")
    view = start_view(src.parent, src)
    if view:
        view_path = dst.with_name(dst.stem + ".view.json")
        view_path.write_text(json.dumps(view) + "\n")
        print(f"starting camera: near {view['frame']} (turned {view['turn']:+.0f} deg, "
              f"{view['backMetres']:.1f} m back; {view['coverage']:.0%} of the screen "
              f"covered, {view['blurShare']:.0%} blurred or too close) -> {view_path}")


if __name__ == "__main__":
    main()
