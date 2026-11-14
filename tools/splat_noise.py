#!/usr/bin/env python3
"""A noise score for every Gaussian of a splat, and a cleaned copy at any amount.

The floater eraser: instead of one fixed rule (tools/splat_prune.py), every
Gaussian gets a score from 0 (on a measured surface) to 1 (hanging in free air),
and a person chooses how much to remove, looking at the result.

The score comes from the same test as splat_prune: each Gaussian is projected
into every solved frame and compared with the nearest surface the dense cloud
measured along that pixel's ray. A Gaussian well in front of that surface is in
free air in that frame. Its score is the share of the frames with a surface
behind it in which it floats (needing at least MIN_FRAMES of them), raised for
Gaussians far from every measured point. Ties are broken by distance, so the
order is stable from the first percent removed to the last.

    score   writes splat-noise.npz (score per Gaussian, in the PLY's order) and
            prints how much of the splat floats
    apply   writes a cleaned copy without the given share of the noisiest
            Gaussians (the full splat, view-dependent colour and all)

Removing floaters can cost as much as it gains: training can paint the room's
colour onto them (see splat_prune.py's findings), so check the cleaned copy from
the filmed views as well as from beside them.

Usage:
    python3 tools/splat_noise.py score spaces/<name> splat-gsplat-30k.ply
    python3 tools/splat_noise.py apply spaces/<name> splat-gsplat-30k.ply --remove 12
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

FRONT_M = 0.15        # in free air: this far in front of the measured surface...
FRONT_SHARE = 0.03    # ...plus this share of the distance (depth noise grows with it)
MIN_FRAMES = 2        # frames with a surface behind it needed before a share means anything
FAR_M = 0.5           # this far from every measured point counts as fully in free air
DENSE_SAMPLE = 2_500_000


def scores(space: Path, ply: Path, log=print) -> dict:
    """Noise score per Gaussian (0..1) and the facts behind it."""
    from densify import read_cameras_bin, read_images_bin
    from pointcloud import load_ply, space_model_dir
    from scipy.spatial import cKDTree
    from splat_gsplat import read_ply
    from surface_fill import DEPTH_SHRINK, depth_buffer, project

    metre = json.loads((space / "densify.json").read_text()).get("colmap_units_per_metre") or 1.0
    raw = read_ply(ply)
    xyz = raw["means"].astype(np.float64)
    dense = load_ply(space / "cloud-dense.ply").points
    if len(dense) > DENSE_SAMPLE:
        dense = dense[np.random.default_rng(0).choice(len(dense), DENSE_SAMPLE, replace=False)]
    dense = dense.astype(np.float64)
    model = space_model_dir(space)
    cameras = read_cameras_bin(model / "cameras.bin")
    images = sorted(read_images_bin(model / "images.bin").values(), key=lambda v: v["name"])
    in_front = np.zeros(len(xyz), np.int32)
    behind = np.zeros(len(xyz), np.int32)
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
        in_front[idx] += zz < nearest - FRONT_M * metre - FRONT_SHARE * zz
    distance = cKDTree(dense).query(xyz, workers=-1)[0] / metre
    share = np.where(behind >= MIN_FRAMES, in_front / np.maximum(behind, 1), 0.0)
    far = np.clip(distance / FAR_M, 0, 1)
    score = np.maximum(share, far)
    # Ties (most Gaussians sit at 0) are ordered by distance, a little at a time.
    score = np.clip(score * 0.98 + 0.02 * far, 0, 1)
    opacity = 1 / (1 + np.exp(-raw["opacities"]))
    floating = score >= 0.5
    log(f"  {len(xyz):,} Gaussians, {len(images)} frames: {int(floating.sum()):,} ({floating.mean():.1%}) float in "
        f"free air in at least half the frames that see a surface behind them; they carry "
        f"{opacity[floating].sum() / opacity.sum():.1%} of the splat's opacity")
    return {"score": score.astype(np.float32), "distance_m": distance.astype(np.float32),
            "free_share": share.astype(np.float32), "frames_behind": behind,
            "summary": {"gaussians": int(len(xyz)), "frames": len(images),
                        "floating_share": round(float(floating.mean()), 4),
                        "floating_opacity_share": round(float(opacity[floating].sum() / opacity.sum()), 4),
                        "score_percentiles": {str(p): round(float(np.percentile(score, p)), 3)
                                              for p in (50, 75, 90, 95, 99)}}}


def threshold_for(score: np.ndarray, remove_percent: float) -> float:
    """The score above which the noisiest `remove_percent` of Gaussians lie."""
    if remove_percent <= 0:
        return np.inf
    return float(np.quantile(score, 1 - min(remove_percent, 100) / 100))


def apply(space: Path, ply: Path, remove_percent: float, out: Path | None = None, log=print) -> Path:
    from splat_gsplat import read_ply, write_ply

    noise = np.load(space / "splat-noise.npz")
    score = noise["score"]
    raw = read_ply(ply)
    if len(score) != len(raw["means"]):
        raise SystemExit(f"splat-noise.npz scores {len(score):,} Gaussians, {ply.name} has {len(raw['means']):,}: "
                         "score this splat first")
    keep = score < threshold_for(score, remove_percent)
    out = out or space / f"{ply.stem}-clean{remove_percent:g}.ply"
    write_ply(out, *(raw[k][keep] for k in ("means", "scales", "quats", "opacities", "sh0", "shN")),
              comment=f"{ply.name} without its noisiest {remove_percent:g}% (tools/splat_noise.py)")
    log(f"  wrote {out.name}: removed {int((~keep).sum()):,} of {len(keep):,} Gaussians")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("space", type=Path)
    s.add_argument("ply")
    a = sub.add_parser("apply")
    a.add_argument("space", type=Path)
    a.add_argument("ply")
    a.add_argument("--remove", type=float, required=True, help="percent of the noisiest Gaussians to remove")
    args = ap.parse_args()
    ply = args.space / args.ply
    if args.cmd == "score":
        result = scores(args.space, ply)
        np.savez_compressed(args.space / "splat-noise.npz", **{k: v for k, v in result.items() if k != "summary"})
        (args.space / "splat-noise.json").write_text(json.dumps({"splat": ply.name, **result["summary"]}, indent=1))
    else:
        apply(args.space, ply, args.remove)


if __name__ == "__main__":
    main()
