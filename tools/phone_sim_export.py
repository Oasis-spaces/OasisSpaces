#!/usr/bin/env python3
"""Export what the phone would have known while filming a processed space, so
the phone's perception can be run on the video's frames on the Mac
(apps/OasisCapture/Packages/CaptureRules, the `phonesim` tool) and judged
against stage 3's room.

Writes spaces/<name>/phone-sim/frames.json: for every registered frame, its
camera (focal length and centre in pixels, size, camera-to-world transform in
metres, y up, the room's centre on the floor as origin: ARKit's conventions)
and the sparse points it saw (world, metres), standing in for ARKit's
tracking points; and the room's measured objects as boxes in the same frame.

    python3 tools/phone_sim_export.py spaces/<name> [--every 1]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(ROOT / "tools"))

# scene (x, y along the walls, z up) -> ARKit-like (x, y up, z towards the viewer)
TO_VIEWER = np.array([[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]])


def main() -> None:
    from densify import read_cameras_bin, read_images_bin, read_points3d_bin
    from splat_edit import Room
    from splat_export import model_dir

    parser = argparse.ArgumentParser()
    parser.add_argument("space", type=Path)
    parser.add_argument("--every", type=int, default=1, help="use every n-th registered frame")
    args = parser.parse_args()
    space = args.space.resolve()
    room = Room(space)
    m = room.metre
    model = model_dir(space)
    cams = read_cameras_bin(model / "cameras.bin")
    images = sorted(read_images_bin(model / "images.bin").values(), key=lambda v: v["name"])
    pts3d = read_points3d_bin(model / "points3D.bin")
    origin_scene = np.array([room.centre[0], room.centre[1], room.floor_z])
    # splat/colmap -> scene -> viewer frame, metres
    rot = TO_VIEWER @ room.world

    def to_world(p_colmap):
        return (rot @ (np.asarray(p_colmap, float)) - TO_VIEWER @ origin_scene) / m

    frames = []
    for info in images[:: max(1, args.every)]:
        cam = cams[info["camera_id"]]
        params = cam["params"]
        fx = params[0]
        fy = params[1] if len(params) >= 4 else params[0]
        cx, cy = (params[2], params[3]) if len(params) >= 4 else (params[1], params[2])
        R, t = np.asarray(info["R"], float), np.asarray(info["t"], float)
        # COLMAP: camera looks along +z, y down. ARKit: looks along -z, y up. Camera-to-world:
        C_colmap = -R.T @ t
        R_c2w = R.T
        flip = np.diag([1.0, -1.0, -1.0])                 # colmap camera axes -> arkit camera axes
        R_world = rot @ R_c2w @ flip
        position = to_world(C_colmap)
        T = np.eye(4)
        T[:3, :3] = R_world
        T[:3, 3] = position
        sparse = []
        for pid in info.get("point3D_ids", []):
            if pid < 0 or pid not in pts3d:
                continue
            entry = pts3d[pid]
            xyz = entry["xyz"] if isinstance(entry, dict) else entry
            sparse.append([round(float(v), 4) for v in to_world(xyz)])
        frames.append({"name": info["name"], "width": cam["width"], "height": cam["height"],
                       "fx": fx, "fy": fy, "cx": cx, "cy": cy,
                       "transform": [round(float(v), 6) for v in T.T.flatten()],   # column-major, like simd
                       "points": sparse})
    objects = []
    for i, box in enumerate(room.shapes["boxes"]):
        if not box.get("build", True):
            continue
        corners = np.array([[x, y, z] for x in (box["min"][0], box["max"][0]) for y in (box["min"][1], box["max"][1])
                            for z in (box["min"][2], box["max"][2])])
        world = np.array([(TO_VIEWER @ (c - origin_scene)) / m for c in corners])
        objects.append({"id": f"B{i}", "label": box.get("detected") or box.get("label"),
                        "min": world.min(axis=0).round(3).tolist(), "max": world.max(axis=0).round(3).tolist()})
    # Only walls the room keeps: a plane the review dropped (a diagonal through the room, a
    # curtain's plane) is no wall, and the phone holds back what it sees on a bare wall.
    planes = [p for p in room.shapes["planes"] if p.get("label") == "wall" and p.get("build", True)]
    walls = []
    for p in planes:
        c = (TO_VIEWER @ (np.array(p["center"]) - origin_scene)) / m
        a = TO_VIEWER @ np.array(p["axis_a"]); n = TO_VIEWER @ np.array(p["normal"])
        walls.append({"center": c.round(3).tolist(), "along": a.round(4).tolist(), "normal": n.round(4).tolist(),
                      "half": round(p["half_a"] / m, 3), "height": round(room.shapes["room_level"]["height"] / m, 3)})
    out = space / "phone-sim"
    out.mkdir(exist_ok=True)
    (out / "frames.json").write_text(json.dumps({
        "space": space.name, "images": "workspace/images", "metresPerUnit": 1.0,
        "room": {"width": round(2 * room.half[0] / m, 3), "depth": round(2 * room.half[1] / m, 3),
                 "height": round(room.shapes["room_level"]["height"] / m, 3)},
        "walls": walls, "objects": objects, "frames": frames}))
    print(f"wrote {out / 'frames.json'}: {len(frames)} frames, {len(objects)} objects "
          f"({', '.join(o['label'] for o in objects)}), {len(walls)} walls")


if __name__ == "__main__":
    main()
