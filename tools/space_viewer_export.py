#!/usr/bin/env python3
"""A space for the browser, straight after stage 3 (no GPU stage needed).

space-viewer/ shows a space as its dense cloud with the room model over it
(walls, the floor's outline, one box per piece of furniture), lets a person
take pieces out and put furniture in. This writes what it reads:

  spaces/<name>/viewer/points.bin   the dense cloud, subsampled: float32 x, y, z
                                    (metres; y up; the room's centre on the floor
                                    at the origin), then uint8 r, g, b, then a
                                    uint8 piece index per point (0: none)
  spaces/<name>/viewer/room.json    the room model in the same frame: walls, floor
                                    outline, pieces (id, name, box), the camera
                                    path, the floor's colour, the point count

A point belongs to a piece when it lies inside the piece's box (a few cm
around it) and above the floor, so removing the piece takes its points away.

    python3 tools/space_viewer_export.py spaces/<name> [--points 450000]
    python3 -m http.server 8734        # then open
    http://localhost:8734/space-viewer/?space=<name>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))

TO_VIEWER = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], float)   # scene z-up -> viewer y-up
PIECE_MARGIN_M = 0.03
FLOOR_BAND_M = 0.03
KEEP_BEYOND_M = 1.0          # points this far past the room's rectangle are another space: left out
MAX_PIECES = 250


def name_of(box: dict) -> str:
    return box.get("identity") or box.get("detected") or box.get("label") or "piece"


def export(space: Path, max_points: int = 450_000) -> Path:
    from pointcloud import load_ply

    space = Path(space)
    shapes = json.loads((space / "shapes.json").read_text())
    meta = json.loads((space / "densify.json").read_text())
    metre = meta["colmap_units_per_metre"]
    world = np.array(shapes["world"], float)
    level, room = shapes["room_level"], shapes["room"]
    origin = np.array([room["center"][0], room["center"][1], level["floor_z"]], float)

    def to_viewer(scene_points: np.ndarray) -> np.ndarray:
        return ((np.asarray(scene_points, float) - origin) / metre) @ TO_VIEWER.T

    cloud = load_ply(space / "cloud-dense.ply")
    scene = cloud.points.astype(np.float64) @ world.T
    colours = cloud.colors
    # Keep the room and what is right around it.
    u, v = np.array(room["axis_u"]), np.array(room["axis_v"])
    rel = scene[:, :2] - origin[:2]
    du, dv = rel @ u, rel @ v
    reach = KEEP_BEYOND_M * metre
    keep = ((np.abs(du) <= room["half_u"] + reach) & (np.abs(dv) <= room["half_v"] + reach)
            & (scene[:, 2] > level["floor_z"] - 0.1 * metre)
            & (scene[:, 2] < level["floor_z"] + level["height"] + 0.1 * metre))
    idx = np.flatnonzero(keep)
    if len(idx) > max_points:
        idx = np.sort(np.random.default_rng(0).choice(idx, max_points, replace=False))
    scene, colours = scene[idx], colours[idx]

    # Pieces: the built furniture boxes; a point inside one (smallest box wins) is that piece's.
    pieces, piece_of = [], np.zeros(len(scene), np.uint8)
    built = [(i, b) for i, b in enumerate(shapes.get("boxes", []))
             if b.get("build", True) and b.get("label") not in (None, "clutter")][:MAX_PIECES]
    margin = PIECE_MARGIN_M * metre
    order = sorted(built, key=lambda ib: -np.prod(np.subtract(ib[1]["max"], ib[1]["min"])))
    for i, b in order:                       # biggest first, so a smaller box inside it wins
        lo, hi = np.array(b["min"], float) - margin, np.array(b["max"], float) + margin
        lo[2] = max(lo[2], level["floor_z"] + FLOOR_BAND_M * metre)
        inside = np.all((scene >= lo) & (scene <= hi), axis=1)
        piece_of[inside] = len(pieces) + 1
        corners = to_viewer(np.array([b["min"], b["max"]], float))
        pieces.append({"index": len(pieces) + 1, "id": f"B{i}", "name": name_of(b), "type": b.get("label"),
                       "min": corners.min(axis=0).round(4).tolist(), "max": corners.max(axis=0).round(4).tolist(),
                       "mounted": bool(b.get("mounted")), "points": int(inside.sum())})

    floor = (np.abs(scene[:, 2] - level["floor_z"]) < FLOOR_BAND_M * metre) & (piece_of == 0)
    floor_colour = (np.median(colours[floor], axis=0) if floor.any() else np.array([180, 170, 160])).astype(int)

    walls = []
    for i, p in enumerate(shapes.get("planes", [])):
        if p.get("label") != "wall" or p.get("build") is False:
            continue
        c, a = np.array(p["center"], float), np.array(p["axis_a"], float)
        ends = to_viewer(np.array([c - p["half_a"] * a, c + p["half_a"] * a]))
        walls.append({"id": f"W{i}", "from": ends[0][[0, 2]].round(4).tolist(), "to": ends[1][[0, 2]].round(4).tolist(),
                      "inferred": p.get("source") == "inferred"})
    outline = (shapes.get("outline") or {}).get("polygons")
    if not outline:
        outline = [[room["center"] + (su * room["half_u"] * u + sv * room["half_v"] * v)
                    for su, sv in ((1, 1), (1, -1), (-1, -1), (-1, 1))]]
    floor_polys = [to_viewer(np.column_stack([np.asarray(poly, float), np.full(len(poly), level["floor_z"])]))[:, [0, 2]]
                   .round(4).tolist() for poly in outline]
    cams = np.asarray(shapes.get("cameras") or [], float).reshape(-1, 2)
    path = (to_viewer(np.column_stack([cams, np.full(len(cams), level["floor_z"])]))[:, [0, 2]].round(3).tolist()
            if len(cams) else [])

    out = space / "viewer"
    out.mkdir(exist_ok=True)
    pts = to_viewer(scene).astype(np.float32)
    with open(out / "points.bin", "wb") as f:
        f.write(pts.tobytes())
        f.write(np.ascontiguousarray(colours, np.uint8).tobytes())
        f.write(piece_of.tobytes())
    (out / "room.json").write_text(json.dumps({
        "space": space.name, "count": int(len(pts)),
        # "video": metres from monocular depth; "measured": set from a real length (set_real_scale.py)
        "scaleSource": "measured" if meta.get("scale_measured") else "video",
        "scaleMeasured": meta.get("scale_measured"), "height": round(level["height"] / metre, 3),
        "floorColor": floor_colour.tolist(), "floor": floor_polys, "walls": walls, "pieces": pieces,
        "path": path}, indent=1) + "\n")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("space", type=Path)
    parser.add_argument("--points", type=int, default=450_000)
    args = parser.parse_args()
    out = export(args.space, args.points)
    room = json.loads((out / "room.json").read_text())
    print(f"{out}: {room['count']:,} points, {len(room['pieces'])} pieces "
          f"({', '.join(p['name'] for p in room['pieces'])}), {len(room['walls'])} walls")


if __name__ == "__main__":
    main()
