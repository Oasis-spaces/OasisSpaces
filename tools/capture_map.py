#!/usr/bin/env python3
"""A map of what a capture covered: the room from above, where the camera
went, which parts of the floor it saw and how often, which way it looked.

This is the picture a person needs right after filming, and the one the
pipeline needs before it spends an hour on a room: the room's footprint from
the measured points, the walked path from start to end, the floor shaded by
how many frames saw each part of it, the gaps no frame saw, and a compass of
the directions the camera faced (a wall nobody turned to has no texture and
no geometry, whatever the trainer does).

It works from any camera solve:
  - MapAnything's (workspace/mapanything/views.npz + points.ply), metric;
  - else the space's COLMAP model with cloud-dense.ply or the sparse points,
    in metres when densify.json has recorded the scale.

Writes <space>/capture-map.png and <space>/capture-map.json.

Usage:
    python3 tools/capture_map.py spaces/<name> [--source mapanything|colmap]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))

CELL_M = 0.05              # the plan's resolution
SEEN_RANGE_M = 5.0         # a floor cell counts as seen by a camera within this distance
WELL_SEEN = 3              # frames that make a part of the floor well covered
SECTORS = 12               # the compass of directions faced
PX_PER_M = 120             # plan scale
HANDHELD_HEIGHT_M = 1.4    # a phone filming a room is held about this high: the scale before densify measures it


def read_ply_points(path: Path):
    """xyz (and rgb if present) of a binary little-endian PLY of floats and uchars."""
    blob = path.read_bytes()
    end = blob.index(b"end_header\n") + len(b"end_header\n")
    header = blob[:end].decode().splitlines()
    count = next(int(l.split()[-1]) for l in header if l.startswith("element vertex"))
    props = [l.split() for l in header if l.startswith("property")]
    kinds = {"float": "<f4", "double": "<f8", "uchar": "u1", "uint8": "u1", "int": "<i4"}
    dtype = np.dtype([(p[2], kinds[p[1]]) for p in props])
    rows = np.frombuffer(blob, dtype=dtype, count=count, offset=end)
    xyz = np.stack([rows["x"], rows["y"], rows["z"]], axis=1).astype(np.float64)
    names = rows.dtype.names
    rgb = (np.stack([rows[c] for c in ("red", "green", "blue")], axis=1).astype(np.float64)
           if "red" in names else np.full((count, 3), 170.0))
    return xyz, rgb


def load_mapanything(space: Path):
    folder = space / "workspace" / "mapanything"
    views = np.load(folder / "views.npz")
    names = [str(n) for n in views["names"]]
    cam2world = views["cam2world"]
    K = views["intrinsics"]
    width, height = (int(v) for v in views["frame_size"])
    points, colours = read_ply_points(folder / "points.ply")
    return names, cam2world, K, (width, height), points, colours, 1.0, "MapAnything"


def load_colmap(space: Path, dense: bool = True, units_override: float | None = None):
    from densify import read_cameras_bin, read_images_bin, read_points3d_bin
    from pointcloud import space_model_dir

    model = space_model_dir(space)
    if model is None:
        raise SystemExit(f"no camera solve in {space / 'workspace'}")
    cameras = read_cameras_bin(model / "cameras.bin")
    infos = sorted(read_images_bin(model / "images.bin").values(), key=lambda v: v["name"])
    cam2world, K = [], []
    for info in infos:
        pose = np.eye(4)
        pose[:3, :3] = info["R"].T
        pose[:3, 3] = -info["R"].T @ info["t"]
        cam2world.append(pose)
        fx, fy, cx, cy = cameras[info["camera_id"]]["params"][:4]
        K.append(np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]))
    camera = cameras[infos[0]["camera_id"]]
    cloud = space / "cloud-dense.ply"
    if dense and cloud.exists():
        points, colours = read_ply_points(cloud)
        pick = np.random.default_rng(0).choice(len(points), min(len(points), 800_000), replace=False)
        points, colours = points[pick], colours[pick]
    else:
        pts = read_points3d_bin(model / "points3D.bin")
        points = np.array([v[:3] for v in pts.values()], float)
        colours = np.full((len(points), 3), 170.0)
    meta = space / "densify.json"
    units = json.loads(meta.read_text()).get("colmap_units_per_metre") if meta.exists() else None
    units = units_override or units              # None: not measured yet (build estimates it)
    return ([i["name"] for i in infos], np.stack(cam2world), np.stack(K),
            (camera["width"], camera["height"]), points, colours, units, "COLMAP")


def units_from_height(cam2world: np.ndarray, points: np.ndarray, up: np.ndarray) -> float:
    """Solve units per metre, guessed from how high the phone was held: the
    median camera above the floor (the low end of the measured points) is
    taken as HANDHELD_HEIGHT_M. Good to a few tens of percent, which is enough
    to read a coverage map before densify has measured the scale."""
    floor = np.percentile(points @ up, 2)
    above = np.median(cam2world[:, :3, 3] @ up) - floor
    return float(above / HANDHELD_HEIGHT_M) if above > 0 else 1.0


def room_frame(cam2world: np.ndarray, points: np.ndarray):
    """(up, right, forward). A handheld phone pitches and turns but does not
    roll, so its image's sideways axis stays level: up is the direction most
    at right angles to every frame's sideways axis, whatever the pitch
    (averaging the image's up axis instead tilts it back whenever the camera
    mostly looks down one way). The plan's axes follow the spread of the
    points across the floor."""
    sideways = cam2world[:, :3, 0]
    values, vectors = np.linalg.eigh(sideways.T @ sideways)
    up = vectors[:, 0]
    if up @ -cam2world[:, :3, 1].mean(axis=0) < 0:        # the image's up points up, not down
        up = -up
    flat = points - np.outer(points @ up, up)
    flat -= flat.mean(axis=0)
    _, _, vt = np.linalg.svd(flat[np.random.default_rng(1).choice(len(flat), min(len(flat), 50_000), replace=False)],
                             full_matrices=False)
    right = vt[0] - (vt[0] @ up) * up
    right /= np.linalg.norm(right)
    forward = np.cross(up, right)
    return up, right, forward


def hull_mask(cells: np.ndarray) -> np.ndarray:
    """The convex hull of the True cells, filled, as a mask of the same shape."""
    from scipy.spatial import ConvexHull

    ys, xs = np.nonzero(cells)
    if len(xs) < 3:
        return cells.copy()
    corners = np.stack([xs, ys], axis=1).astype(float)
    corners = np.vstack([corners, corners + [1, 0], corners + [0, 1], corners + [1, 1]])   # cell corners
    hull = corners[ConvexHull(corners).vertices]
    picture = Image.new("L", (cells.shape[1], cells.shape[0]), 0)
    ImageDraw.Draw(picture).polygon([tuple(v) for v in hull], fill=1)
    return np.asarray(picture).astype(bool)


OCCLUSION_GRID = 16        # the depth buffer that decides what blocks a view: frame pixels per cell


def floor_coverage(cells_xyz: np.ndarray, cam2world: np.ndarray, K: np.ndarray, size, reach: float,
                   occluders: np.ndarray, margin: float) -> np.ndarray:
    """How many frames saw each floor cell: in front of the camera, inside its
    picture, within `reach`, and not behind anything measured. Each frame's
    view is blocked where the measured points stand nearer than the floor: a
    coarse depth buffer of them, spread one cell so gaps between points do not
    let the floor under a bed show through."""
    from scipy.ndimage import minimum_filter

    width, height = size
    gw, gh = -(-width // OCCLUSION_GRID), -(-height // OCCLUSION_GRID)
    seen = np.zeros(len(cells_xyz), np.int32)
    for pose, k in zip(cam2world, K):
        R, centre = pose[:3, :3], pose[:3, 3]
        local = (occluders - centre) @ R                  # world -> camera: R^T (x - c), as rows
        z = local[:, 2]
        front = z > 0.1
        u = (k[0, 0] * local[front, 0] / z[front] + k[0, 2]) // OCCLUSION_GRID
        v = (k[1, 1] * local[front, 1] / z[front] + k[1, 2]) // OCCLUSION_GRID
        on = (u >= 0) & (u < gw) & (v >= 0) & (v < gh)
        depth = np.full((gh, gw), np.inf)
        np.minimum.at(depth, (v[on].astype(int), u[on].astype(int)), z[front][on])
        depth = minimum_filter(depth, size=3)
        local = (cells_xyz - centre) @ R
        z = local[:, 2]
        ahead = (z > 0.1) & (np.linalg.norm(cells_xyz - centre, axis=1) < reach)
        u = k[0, 0] * local[:, 0] / np.where(ahead, z, 1) + k[0, 2]
        v = k[1, 1] * local[:, 1] / np.where(ahead, z, 1) + k[1, 2]
        inside = ahead & (u >= 0) & (u < width) & (v >= 0) & (v < height)
        index = np.flatnonzero(inside)
        blocker = depth[(v[index] // OCCLUSION_GRID).astype(int), (u[index] // OCCLUSION_GRID).astype(int)]
        visible = np.zeros(len(cells_xyz), bool)
        visible[index] = z[index] <= blocker + margin
        seen += visible
    return seen


def build(space: Path, source: str = "auto", log=print, points_from: str = "dense", out: Path | None = None,
          units: float | None = None) -> dict:
    use_ma = source == "mapanything" or (source == "auto" and (space / "workspace" / "mapanything" / "views.npz").exists())
    names, cam2world, K, size, points, colours, units, where = (
        load_mapanything(space) if use_ma else load_colmap(space, dense=points_from == "dense", units_override=units))
    up, right, forward = room_frame(cam2world, points)
    if units:
        scale_from = "MapAnything's metric depth" if use_ma else "densify's measurement"
    else:
        units, scale_from = units_from_height(cam2world, points, up), f"the phone held {HANDHELD_HEIGHT_M} m up (guessed)"
    m = units                                            # solve units per metre
    heights = points @ up
    floor, ceiling = np.percentile(heights, 2), np.percentile(heights, 98)
    plan = np.stack([points @ right, points @ forward], axis=1)
    lo, hi = np.percentile(plan, 0.5, axis=0) - 0.3 * m, np.percentile(plan, 99.5, axis=0) + 0.3 * m
    path = np.stack([cam2world[:, :3, 3] @ right, cam2world[:, :3, 3] @ forward], axis=1)
    lo, hi = np.minimum(lo, path.min(axis=0) - 0.3 * m), np.maximum(hi, path.max(axis=0) + 0.3 * m)
    cell = CELL_M * m
    cols, rows = (np.ceil((hi - lo) / cell).astype(int) + 1)

    # The room's footprint. Walls and furniture (points between knee and head
    # height) outline it; the floor alone would leak out through a doorway.
    # What was seen through a door or a window lies apart from the room, so
    # only the largest connected body of them counts, and the floor is its
    # convex hull: a video filmed from inside rarely closes the ring of walls
    # all round, and the floor inside is the room whether it was seen or not.
    from scipy.ndimage import binary_closing, label

    ij = np.floor((plan - lo) / cell).astype(int)
    within = np.all((ij >= 0) & (ij < [cols, rows]), axis=1)
    body = (heights > floor + 0.15 * m) & (heights < ceiling - 0.15 * m) & within
    occupied = np.zeros((rows, cols), np.int32)
    np.add.at(occupied, (ij[body, 1], ij[body, 0]), 1)
    tint = np.zeros((rows, cols, 3))
    np.add.at(tint, (ij[body, 1], ij[body, 0]), colours[body])
    joined, found = label(binary_closing(occupied > 0, iterations=max(2, round(0.3 / CELL_M))))
    if found > 1:
        weight = np.bincount(joined.ravel(), weights=occupied.ravel())[1:]
        room_cells = joined == 1 + int(np.argmax(weight))
    else:
        room_cells = joined > 0
    has = (occupied > 0) & room_cells
    tint[has] /= occupied[has][:, None]
    footprint = hull_mask(room_cells)

    # How often each floor cell was seen.
    gy, gx = np.mgrid[0:rows, 0:cols]
    centres2 = lo + (np.stack([gx.ravel(), gy.ravel()], axis=1) + 0.5) * cell
    cells_xyz = centres2[:, :1] * right + centres2[:, 1:] * forward + floor * up
    # What blocks a view: everything measured above the floor (the floor itself cannot hide the floor).
    raised = points[(heights > floor + 0.05 * m)]
    raised = raised[np.random.default_rng(2).choice(len(raised), min(len(raised), 200_000), replace=False)]
    seen = floor_coverage(cells_xyz, cam2world, K, size, SEEN_RANGE_M * m, raised, 0.08 * m).reshape(rows, cols)
    inside = footprint
    well = inside & (seen >= WELL_SEEN)
    unseen = inside & (seen == 0)

    # Which way the camera looked, on the plan.
    look = cam2world[:, :3, 2]
    angles = np.arctan2(look @ forward, look @ right)
    sectors = np.bincount(((angles + np.pi) / (2 * np.pi) * SECTORS).astype(int) % SECTORS, minlength=SECTORS)

    # ---- the picture
    scale = PX_PER_M / m * cell
    W, H = int(cols * scale) + 40, int(rows * scale) + 40
    board = Image.new("RGB", (W + 300, max(H, 420)), (248, 247, 244))
    draw = ImageDraw.Draw(board)
    font = ImageFont.load_default(size=15)
    small = ImageFont.load_default(size=12)
    to_px = lambda xy: (20 + (xy[0] - lo[0]) / cell * scale, 20 + (hi[1] - xy[1]) / cell * scale)
    shade = np.full((rows, cols, 3), 248.0)
    shade[inside & (seen == 0)] = (236, 120, 110)                          # never seen: red
    shade[inside & (seen > 0) & (seen < WELL_SEEN)] = (244, 205, 120)     # seen by one or two frames: amber
    shade[well] = (214, 232, 214)                                          # seen by several: green
    shade[has] = 0.5 * shade[has] + 0.5 * tint[has]                        # what stands there, over it
    picture = Image.fromarray(np.clip(shade[::-1], 0, 255).astype(np.uint8)).resize(
        (int(cols * scale), int(rows * scale)), Image.NEAREST)
    board.paste(picture, (20, 20))
    track = [to_px(p) for p in path]
    for n in range(len(track) - 1):
        t = n / max(1, len(track) - 2)
        draw.line([track[n], track[n + 1]], fill=(int(40 + 180 * t), int(90 + 40 * (1 - t)), int(200 - 150 * t)), width=3)
    for n in range(0, len(track), max(1, len(track) // 24)):
        x, y = track[n]
        d = np.array([look[n] @ right, look[n] @ forward])
        d = d / (np.linalg.norm(d) + 1e-9) * 16
        draw.line([(x, y), (x + d[0], y - d[1])], fill=(60, 60, 60), width=1)
    for label, xy, colour in (("start", track[0], (40, 140, 60)), ("end", track[-1], (190, 60, 50))):
        draw.ellipse([xy[0] - 6, xy[1] - 6, xy[0] + 6, xy[1] + 6], fill=colour)
        draw.text((xy[0] + 8, xy[1] - 8), label, fill=colour, font=small)
    # scale bar
    bar = PX_PER_M
    draw.line([(24, H - 12), (24 + bar, H - 12)], fill=(40, 40, 40), width=3)
    draw.text((28 + bar, H - 20), "1 m", fill=(40, 40, 40), font=small)

    # compass of directions faced, and the legend
    cx0, cy0, r = W + 150, 130, 90
    draw.text((W + 20, 12), "Directions the camera faced", fill=(30, 30, 30), font=font)
    top = sectors.max() or 1
    for k in range(SECTORS):
        a0 = -np.pi + 2 * np.pi * k / SECTORS
        a1 = a0 + 2 * np.pi / SECTORS
        length = 18 + (r - 18) * sectors[k] / top
        poly = [(cx0, cy0)] + [(cx0 + length * np.cos(a), cy0 - length * np.sin(a)) for a in np.linspace(a0, a1, 6)]
        draw.polygon(poly, fill=(120, 150, 200) if sectors[k] else (236, 120, 110))
    draw.ellipse([cx0 - r, cy0 - r, cx0 + r, cy0 + r], outline=(150, 150, 150))
    legend = [((214, 232, 214), f"floor seen by {WELL_SEEN}+ frames"), ((244, 205, 120), "seen by 1-2 frames"),
              ((236, 120, 110), "never seen"), ((90, 110, 200), "the walk, start to end")]
    for n, (colour, text) in enumerate(legend):
        y = 250 + n * 24
        draw.rectangle([W + 20, y, W + 36, y + 16], fill=colour)
        draw.text((W + 44, y), text, fill=(30, 30, 30), font=small)

    area = float(inside.sum()) * CELL_M ** 2
    record = {
        "source": where, "frames": len(names),
        "path_m": round(float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()) / m, 2),
        "footprint_m2": round(area, 2),
        "size_m": [round(float((hi - lo)[0]) / m, 2), round(float((hi - lo)[1]) / m, 2)],
        "height_m": round(float(ceiling - floor) / m, 2),
        "floor_well_seen": round(float(well.sum()) / max(1, inside.sum()), 3),
        "floor_never_seen": round(float(unseen.sum()) / max(1, inside.sum()), 3),
        "directions_faced": f"{int((sectors > 0).sum())} of {SECTORS}",
        "directions_never_faced_deg": [int(-180 + 360 * k / SECTORS) for k in range(SECTORS) if sectors[k] == 0],
        "scale_from": scale_from,
    }
    draw.text((W + 20, 342), f"scale from {scale_from}", fill=(90, 90, 90), font=small)
    draw.text((W + 20, 360), f"{record['footprint_m2']} sq m room, {record['height_m']} m high", fill=(30, 30, 30), font=small)
    draw.text((W + 20, 378), f"{record['frames']} frames over a {record['path_m']} m walk ({where})", fill=(30, 30, 30), font=small)
    draw.text((W + 20, 396), f"floor well seen {record['floor_well_seen']:.0%}, never {record['floor_never_seen']:.0%}", fill=(30, 30, 30), font=small)
    out = out or space / "capture-map"
    board.save(out.with_suffix(".png"))
    out.with_suffix(".json").write_text(json.dumps(record, indent=1) + "\n")
    log(f"capture map: {record['footprint_m2']} m², floor well seen {record['floor_well_seen']:.0%}, "
        f"never {record['floor_never_seen']:.0%}, {record['directions_faced']} directions faced -> {out.with_suffix('.png')}")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("space", type=Path)
    parser.add_argument("--source", choices=["auto", "mapanything", "colmap"], default="auto")
    parser.add_argument("--points", choices=["dense", "sparse"], default="dense",
                        help="a COLMAP solve's dense cloud (when densify has run) or its own sparse points")
    parser.add_argument("--out", type=Path, help="where to write (default <space>/capture-map.png)")
    parser.add_argument("--units-per-metre", type=float,
                        help="the solve's units per metre, when densify.json has not recorded it")
    args = parser.parse_args()
    build(args.space.resolve(), args.source, points_from=args.points,
          out=args.out.resolve().with_suffix("") if args.out else None, units=args.units_per_metre)


if __name__ == "__main__":
    main()
