#!/usr/bin/env python3
"""Remove a splat's floaters: Gaussians hanging in the room's free air.

Training fits the video frames, and it happily explains a frame with a faint
blob in mid-air instead of a surface. From the frames those blobs look right;
from anywhere else they are the haze, grey veils and floating specks a
viewer walks into. Claude's reviews name exactly these ("milky haze", "grey
veil over the wall", "floating specks").

A floater is decided from the surfaces the dense cloud measured. Each
Gaussian is projected into every keyframe, and compared with the nearest
measured surface along that pixel's ray (surface_fill.depth_buffer). A
Gaussian well in front of that surface is in free air in that frame; one in
free air in at least MIN_FRAMES frames, and in at least half of the frames
that have a surface behind it, is pruned. A surface Gaussian only ever sits
at or behind the nearest measured surface, so real detail survives; a thin
thing the dense cloud missed is protected by the half-of-frames rule.
Gaussians farther than FAR_M from every dense point (junk outside the room)
go too. The trained splat is not touched: the result is a new file.

What it found on this project's captures (Sep 2026), measured with
tools/splat_choose.compare at the trained views and at views between them:
pruning never helped. Pan room: 5.2% of the Gaussians float by this test, and
removing them cost 0.04 SSIM and tripled the see-through gaps; even the 853
farthest from anything cost 0.003. Walkthrough: 0.2% float, no change. The
floaters are not dead weight: training put the image on them (a wall's paint
sits 20-40 cm in front of the measured wall, and the pan's keyframes disagree
on scale by 13%), so deleting them uncovers nothing behind. The fix is to make
training put the content on the measured surfaces (depth-supervised training),
not to delete it afterwards. This stays a diagnostic: it reports how much of a
splat floats and where, and is not a stage of the pipeline.

Usage:
    python3 tools/splat_prune.py spaces/<name> [--in splat.ply] [--out splat-pruned.ply]
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
from densify import read_cameras_bin, read_images_bin  # noqa: E402
from pointcloud import load_ply, space_model_dir  # noqa: E402
from splat_tools import read_splat, write_splat  # noqa: E402
from surface_fill import DEPTH_SHRINK, depth_buffer, project  # noqa: E402

FRONT_M = 0.15        # in free air: this far in front of the measured surface...
FRONT_SHARE = 0.03    # ...plus this share of the distance (depth noise grows with it)
MIN_FRAMES = 2        # seen in free air in at least this many frames...
MIN_SHARE = 0.5       # ...and in at least this share of the frames with a surface behind
FAR_M = 0.5           # or farther than this from every measured point
DENSE_SAMPLE = 2_500_000


def prune(space: Path, src: Path, dst: Path, log=print, front_m: float = FRONT_M,
          alpha_max: float = 1.0, far_m: float = FAR_M) -> dict:
    """Write `dst`: `src` without its floaters. Returns what was removed.
    `alpha_max` limits pruning to Gaussians fainter than that; `front_m` is
    how far in front of the measured surface counts as free air."""
    space = Path(space)
    meta = json.loads((space / "densify.json").read_text())
    metre = meta.get("colmap_units_per_metre") or 1.0
    arr, trailing = read_splat(src)
    xyz = np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype(np.float64)
    alpha = 1 / (1 + np.exp(-arr["opacity"].astype(np.float64)))
    size = np.exp(np.stack([arr["scale_0"], arr["scale_1"], arr["scale_2"]], axis=1)
                  .astype(np.float64)).max(axis=1) / metre

    dense = load_ply(space / "cloud-dense.ply").points
    if len(dense) > DENSE_SAMPLE:
        dense = dense[np.random.default_rng(0).choice(len(dense), DENSE_SAMPLE, replace=False)]
    dense = dense.astype(np.float64)
    model = space_model_dir(space)
    if model is None:
        raise SystemExit(f"no camera model in {space}")
    cameras = read_cameras_bin(model / "cameras.bin")
    images = sorted(read_images_bin(model / "images.bin").values(), key=lambda v: v["name"])

    in_front = np.zeros(len(arr), int)   # frames in which the Gaussian floats in free air
    behind = np.zeros(len(arr), int)     # frames that measured a surface behind it
    for info in images:
        cam = cameras[info["camera_id"]]
        buf = depth_buffer(dense, info, cam)
        u, v, z = project(xyz, info, cam)
        ok = (z > 0.1 * metre) & (u >= 0) & (v >= 0) & (u < cam["width"] - 1) & (v < cam["height"] - 1)
        idx = np.flatnonzero(ok)
        nearest = buf[(v[idx] / DEPTH_SHRINK).astype(int), (u[idx] / DEPTH_SHRINK).astype(int)]
        measured = np.isfinite(nearest)
        idx, nearest, zz = idx[measured], nearest[measured], z[idx][measured]
        behind[idx] += 1
        in_front[idx] += (zz < nearest - front_m * metre - FRONT_SHARE * zz)
    floating = ((in_front >= MIN_FRAMES) & (in_front >= MIN_SHARE * np.maximum(behind, 1))
                & (alpha <= alpha_max))

    from scipy.spatial import cKDTree

    far = cKDTree(dense).query(xyz, workers=-1)[0] > far_m * metre
    drop = floating | far
    keep = ~drop
    write_splat(dst, arr[keep], trailing)
    stats = {
        "source": str(src), "result": str(dst), "frames": len(images),
        "gaussians": int(len(arr)), "kept": int(keep.sum()), "pruned": int(drop.sum()),
        "pruned_floating": int(floating.sum()), "pruned_far": int((far & ~floating).sum()),
        "pruned_share": round(float(drop.mean()), 4),
        "pruned_median_alpha": round(float(np.median(alpha[drop])), 3) if drop.any() else None,
        "pruned_median_size_m": round(float(np.median(size[drop])), 3) if drop.any() else None,
        "pruned_big_share": round(float((size[drop] > 0.10).mean()), 3) if drop.any() else None,
    }
    log(f"  pruned {stats['pruned']:,} of {stats['gaussians']:,} Gaussians ({stats['pruned_share']:.1%}): "
        f"{stats['pruned_floating']:,} floating in free air, {stats['pruned_far']:,} far from everything"
        + (f"; median opacity {stats['pruned_median_alpha']:.2f}, median size "
           f"{stats['pruned_median_size_m']:.3f} m" if drop.any() else ""))
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("space")
    parser.add_argument("--in", dest="src", default="splat.ply", help="splat in the space (default splat.ply)")
    parser.add_argument("--out", dest="dst", default="splat-pruned.ply")
    args = parser.parse_args()
    space = Path(args.space)
    stats = prune(space, space / args.src, space / args.dst)
    (space / (Path(args.dst).stem + ".json")).write_text(json.dumps(stats, indent=1) + "\n")
    print(f"wrote {space / args.dst}")


if __name__ == "__main__":
    main()
