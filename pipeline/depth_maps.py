#!/usr/bin/env python3
"""MoGe-2 depth for every registered frame, in the camera solve's units.

densify.py predicts depth for a dozen keyframes to build the dense cloud.
Depth-supervised splat training (pipeline/splat_train.py) wants a depth map
for every training frame instead: each rendered frame is then held to the
depth the model measured, so surfaces stay where they are and the training
cannot explain a frame with blobs in mid-air.

Each map is scaled to COLMAP units the way densify.py does it: a robust
single scale fitted on the sparse points the frame sees, or the median scale
over all frames when a frame has too few points or fits more than 25% off it.
Maps are written at MoGe's working resolution as float16, 0 where invalid,
to <space>/workspace/depth/<frame>.npy, with depth.json beside them saying
how each frame was scaled.

Usage:
    python3 pipeline/depth_maps.py spaces/<name> [--every 1] [--work-size 1280]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from densify import (MIN_ANCHORS, MoGeDepth, fit_scale, frame_anchors,  # noqa: E402
                     read_cameras_bin, read_images_bin, read_points3d_bin,
                     release_model_memory)
from pointcloud import space_model_dir  # noqa: E402
from semantics import default_device  # noqa: E402

MAX_SCALE_OFF = 1.25    # a frame's own scale further off the median than this is not trusted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("space")
    parser.add_argument("--every", type=int, default=1, help="every Nth registered frame")
    parser.add_argument("--work-size", type=int, default=1280, help="MoGe's working long side")
    args = parser.parse_args()

    space = Path(args.space).resolve()
    model = space_model_dir(space)
    if model is None:
        sys.exit(f"no camera model in {space / 'workspace' / 'sparse'}")
    cameras = read_cameras_bin(model / "cameras.bin")
    images = read_images_bin(model / "images.bin")
    points3d = read_points3d_bin(model / "points3D.bin")
    frames = sorted(images.values(), key=lambda v: v["name"])[::args.every]
    out_dir = space / "workspace" / "depth"
    out_dir.mkdir(parents=True, exist_ok=True)

    import torch
    from PIL import Image

    device = default_device()
    predictor = MoGeDepth(device, args.work_size)
    print(f"MoGe-2 on {device}: depth for {len(frames)} frames -> {out_dir}")
    started = time.time()
    records = {}
    for i, info in enumerate(frames):
        cam = cameras[info["camera_id"]]
        img = Image.open(space / "workspace" / "images" / info["name"])
        W, H = img.size
        px, py, depth_true = frame_anchors(info, points3d, W, H)
        depth, k, _ = predictor(img, cam["params"][0])
        s = None
        if len(px) >= MIN_ANCHORS:
            sy = np.minimum((py * k).astype(int), depth.shape[0] - 1)
            sx = np.minimum((px * k).astype(int), depth.shape[1] - 1)
            s = fit_scale(depth[sy, sx], depth_true)
        np.save(out_dir / (Path(info["name"]).stem + ".npy"), depth.astype(np.float16))
        records[info["name"]] = {"scale": s, "anchors": int(len(px)), "work_scale": k,
                                 "shape": list(depth.shape)}
        if (i + 1) % 20 == 0 or i + 1 == len(frames):
            print(f"  {i + 1}/{len(frames)} frames ({time.time() - started:.0f}s)")
    del predictor
    release_model_memory(torch)

    fitted = [r["scale"] for r in records.values() if r["scale"]]
    if not fitted:
        sys.exit("no frame had enough sparse points to fit a depth scale")
    median = float(np.median(fitted))
    on_median = 0
    for r in records.values():
        own = r["scale"]
        if own is None or abs(np.log(own / median)) > np.log(MAX_SCALE_OFF):
            r["scale"], r["own_scale"], on_median = median, own, on_median + 1
    spread = float(np.percentile(fitted, 90) / np.percentile(fitted, 10) - 1)
    (out_dir / "depth.json").write_text(json.dumps({
        "work_size": args.work_size, "median_scale": median, "scale_spread": round(spread, 4),
        "frames_on_median_scale": on_median, "frames": records}, indent=1) + "\n")
    print(f"{len(records)} depth maps; scale to COLMAP units: median {median:.4f} "
          f"(frames spread {spread:.0%}), {on_median} frame(s) on the median scale")


if __name__ == "__main__":
    main()
