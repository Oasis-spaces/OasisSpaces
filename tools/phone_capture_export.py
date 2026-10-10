#!/usr/bin/env python3
"""A recording made by Oasis Capture, laid out for the phone simulator.

`phonesim` (apps/OasisCapture/Packages/CaptureRules) runs the phone's own
models and tracker over a space's frames. tools/phone_sim_export.py gives it
a processed space, with camera poses from the capture solve and a measured
room. This gives it a recording as the phone made it: the video, with the
camera pose ARKit gave every frame (frames.jsonl) and the planes ARKit found
(capture.json). Nothing is solved or measured, so any recording can be played
back through the current code minutes after it was made, floor and walls as
wrong as the phone had them.

    ~/.venvs/oasis-coreml/bin/python tools/phone_capture_export.py \\
        <folder with frames.jsonl and capture.json> --video <video.mov> --space spaces/<name>
    cd apps/OasisCapture/Packages/CaptureRules
    swift run -c release phonesim ../../../../spaces/<name> --dump <folder>

Writes spaces/<name>/phone-sim/frames.json and phone-sim/images/. Frames are
taken a few a second, as the phone analyses them, upright, 1080 wide.

ARKit's tracking points were not recorded before October 2026 (frames.jsonl
has them from then on, as "points"). Where they are missing they are made
here the way ARKit makes them: corners followed over a few frames and
triangulated with the recorded poses. Needs OpenCV (the Core ML environment
has it).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

WIDTH = 1080            # of the frames written, upright
TRACK = 540             # of the frames corners are followed on
WINDOW = 12             # sampled frames a corner is followed over (four seconds: a turn on the spot moves little)
FLIP = np.diag([1.0, -1.0, -1.0])   # our camera (x right, y up, looking along -z) to OpenCV's


def upright_camera(record: dict, scale: float) -> dict:
    """The camera of the upright picture, from ARKit's (which is of the landscape sensor image).

    The upright picture is the sensor's turned a quarter clockwise: sensor (u, v) lands at
    (H - v, u). So the upright camera's x is the sensor camera's y, its y the sensor's -x.
    """
    fx, fy, cx, cy = record["intrinsics"]
    w, h = record["size"]
    t = np.array(record["transform"], dtype=float).reshape(4, 4).T    # column-major, camera to world
    turned = t.copy()
    turned[:3, 0] = t[:3, 1]
    turned[:3, 1] = -t[:3, 0]
    return {"fx": fy * scale, "fy": fx * scale, "cx": (h - cy) * scale, "cy": cx * scale,
            "width": int(round(h * scale)), "height": int(round(w * scale)), "matrix": turned}


def projection(camera: dict, scale: float) -> np.ndarray:
    """3x4 OpenCV projection of world points into the camera's picture, shrunk by `scale`."""
    k = np.array([[camera["fx"] * scale, 0, camera["cx"] * scale], [0, camera["fy"] * scale, camera["cy"] * scale], [0, 0, 1]])
    r, c = camera["matrix"][:3, :3], camera["matrix"][:3, 3]
    return k @ FLIP @ np.hstack([r.T, (-r.T @ c)[:, None]])


def triangulate(projections: list[np.ndarray], pixels: list[np.ndarray]) -> np.ndarray:
    rows = []
    for p, (u, v) in zip(projections, pixels):
        rows.append(u * p[2] - p[0])
        rows.append(v * p[2] - p[1])
    _, _, vt = np.linalg.svd(np.array(rows))
    x = vt[-1]
    return x[:3] / x[3]


def tracked_points(grays: list[np.ndarray], cameras: list[dict]) -> list[list[list[float]]]:
    """World points for each frame: corners followed over WINDOW frames and triangulated."""
    scale = TRACK / WIDTH
    projections = [projection(c, scale) for c in cameras]
    centres = [c["matrix"][:3, 3] for c in cameras]
    points: list[list[list[float]]] = [[] for _ in cameras]
    flow = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    for start in range(len(grays) - 1):
        corners = cv2.goodFeaturesToTrack(grays[start], maxCorners=400, qualityLevel=0.01, minDistance=12)
        if corners is None:
            continue
        tracks = [[(start, c.ravel())] for c in corners]
        alive = list(range(len(tracks)))
        previous = corners.reshape(-1, 1, 2).astype(np.float32)
        for k in range(start + 1, min(start + WINDOW, len(grays))):
            if not alive:
                break
            forward, ok, _ = cv2.calcOpticalFlowPyrLK(grays[k - 1], grays[k], previous, None, **flow)
            back, ok2, _ = cv2.calcOpticalFlowPyrLK(grays[k], grays[k - 1], forward, None, **flow)
            drift = np.linalg.norm((back - previous).reshape(-1, 2), axis=1)
            keep = (ok.ravel() == 1) & (ok2.ravel() == 1) & (drift < 1.0)
            h, w = grays[k].shape
            inside = (forward[:, 0, 0] > 2) & (forward[:, 0, 0] < w - 2) & (forward[:, 0, 1] > 2) & (forward[:, 0, 1] < h - 2)
            keep &= inside
            for i, good in zip(list(alive), keep):
                if good:
                    tracks[i].append((k, forward[alive.index(i)].ravel()))
            alive = [i for i, good in zip(alive, keep) if good]
            previous = forward[keep].reshape(-1, 1, 2)
        for track in tracks:
            if len(track) < 3:
                continue
            frames = [f for f, _ in track]
            world = triangulate([projections[f] for f in frames], [p for _, p in track])
            # Seen from far enough apart to have a depth, in front of every camera, and landing where it was seen.
            rays = [world - centres[f] for f in frames]
            depths = [np.linalg.norm(r) for r in rays]
            if min(depths) < 0.2 or max(depths) > 8:
                continue
            a, b = rays[0] / depths[0], rays[-1] / depths[-1]
            # (Corners are followed to a third of a pixel, a fortieth of a degree: at 0.7 degrees
            # between the two sight lines the depth is good to a few per cent.)
            if np.degrees(np.arccos(np.clip(a @ b, -1, 1))) < 0.7:
                continue
            good = True
            for f, pixel in track:
                x = projections[f] @ np.append(world, 1)
                if x[2] <= 0 or np.linalg.norm(x[:2] / x[2] - pixel) > 1.2:
                    good = False
                    break
            if good:
                for f in frames:
                    points[f].append([round(float(v), 4) for v in world])
    return points


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("capture", type=Path, help="folder with frames.jsonl and capture.json")
    parser.add_argument("--video", type=Path, help="the recording's video (default: video.mov in the folder)")
    parser.add_argument("--space", type=Path, required=True, help="where to write phone-sim/")
    parser.add_argument("--fps", type=float, default=3, help="frames a second to take, as the phone analyses them")
    args = parser.parse_args()

    records = [json.loads(line) for line in (args.capture / "frames.jsonl").read_text().splitlines() if line.strip()]
    capture = json.loads((args.capture / "capture.json").read_text())
    video = args.video or args.capture / "video.mov"
    out = args.space / "phone-sim"
    images = out / "images"
    images.mkdir(parents=True, exist_ok=True)

    reader = cv2.VideoCapture(str(video))
    rate = reader.get(cv2.CAP_PROP_FPS) or 30
    step = max(1, int(round(rate / args.fps)))
    scale = WIDTH / records[0]["size"][1]
    frames, grays, cameras = [], [], []
    index = 0
    while True:
        ok, image = reader.read()
        if not ok or index >= len(records):
            break
        if index % step == 0:
            if image.shape[1] > image.shape[0]:                    # not turned upright by the reader
                image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            height = int(round(image.shape[0] * WIDTH / image.shape[1]))
            image = cv2.resize(image, (WIDTH, height), interpolation=cv2.INTER_AREA)
            name = f"frame_{index + 1:05d}.jpg"
            cv2.imwrite(str(images / name), image, [cv2.IMWRITE_JPEG_QUALITY, 90])
            camera = upright_camera(records[index], scale)
            cameras.append(camera)
            grays.append(cv2.cvtColor(cv2.resize(image, (TRACK, int(round(height * TRACK / WIDTH)))), cv2.COLOR_BGR2GRAY))
            frames.append({"name": name, "record": records[index]})
        index += 1
    reader.release()

    # The phone records its tracking points every tenth frame: a sampled frame takes the nearest set.
    with_points = [i for i, r in enumerate(records) if "points" in r]
    recorded = len(with_points) >= len(records) // 20
    if recorded:
        taken = np.array(with_points)
        points = [records[int(taken[np.abs(taken - f["record"]["frame"]).argmin()])]["points"] for f in frames]
    else:
        points = tracked_points(grays, cameras)

    # The planes as the phone had them at the end of the recording, wrong ones included.
    planes = [{"kind": p["kind"], "vertical": p["vertical"], "center": p["center"], "xAxis": p["xAxis"], "zAxis": p["zAxis"],
               "extent": p["extent"]} for p in capture.get("roomMap", {}).get("planes", [])]
    walls = []
    for p in planes:
        if not p["vertical"]:
            continue
        x, z = np.array(p["xAxis"]), np.array(p["zAxis"])
        along, length, height = (x, p["extent"][0], p["extent"][1]) if abs(x[1]) <= abs(z[1]) else (z, p["extent"][1], p["extent"][0])
        normal = np.cross(along, [0, 1, 0])
        walls.append({"center": p["center"], "along": along.round(4).tolist(), "normal": normal.round(4).tolist(),
                      "half": round(length / 2, 3), "height": round(height, 3)})

    result = {
        "space": args.space.name, "images": "phone-sim/images", "metresPerUnit": 1,
        "room": {"width": 0, "depth": 0, "height": 0}, "walls": walls, "planes": planes, "objects": [],
        "frames": [{"name": f["name"], "width": c["width"], "height": c["height"],
                    "fx": round(c["fx"], 3), "fy": round(c["fy"], 3), "cx": round(c["cx"], 3), "cy": round(c["cy"], 3),
                    "transform": [round(float(v), 6) for v in c["matrix"].T.reshape(-1)], "points": p}
                   for f, c, p in zip(frames, cameras, points)],
    }
    (out / "frames.json").write_text(json.dumps(result))
    counts = sorted(len(p) for p in points)
    print(f"{out / 'frames.json'}: {len(frames)} frames at {rate / step:.1f} a second, {len(planes)} planes; "
          f"tracking points per frame: median {counts[len(counts) // 2]}, least {counts[0]}"
          + ("" if recorded else " (made here: the recording has none)"))


if __name__ == "__main__":
    main()
