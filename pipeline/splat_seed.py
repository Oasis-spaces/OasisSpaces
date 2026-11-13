#!/usr/bin/env python3
"""Build a space's OpenSplat project, seeded from the dense cloud.

Splat training grows Gaussians from a starting point cloud. COLMAP's sparse
points leave white walls and other low-texture surfaces almost empty, so
Gaussians there start late and often end up see-through. This writes
spaces/<name>/splat-project/ with the solved cameras, the frames, and a
points3D.bin made from cloud-dense.ply (voxel-downsampled to --max-points).
Without a dense cloud, or with --sparse, it uses COLMAP's sparse points.

Usage:
    python3 pipeline/splat_seed.py spaces/<name> [--max-points 250000]
    tools/opensplat spaces/<name>/splat-project -n 10000 -d 4 -o spaces/<name>/splat.ply
"""

from __future__ import annotations

import argparse
import json
import shutil
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from pointcloud import PointCloud, load_ply, space_model_dir, voxel_downsample


def without_images(src: Path, dst: Path, drop: set[str]) -> int:
    """images.bin without the named frames (their records copied byte for byte):
    a splat trained on the rest has never seen them. Returns how many were dropped."""
    data = src.read_bytes()
    count = struct.unpack_from("<Q", data, 0)[0]
    at, kept, dropped = 8, [], 0
    for _ in range(count):
        start = at
        at += 4 + 32 + 24 + 4                       # id, quaternion, translation, camera id
        end = data.index(b"\x00", at)
        name = data[at:end].decode()
        at = end + 1
        points = struct.unpack_from("<Q", data, at)[0]
        at += 8 + 24 * points
        if name in drop:
            dropped += 1
        else:
            kept.append(data[start:at])
    dst.write_bytes(struct.pack("<Q", len(kept)) + b"".join(kept))
    return dropped


def write_points3d_bin(cloud: PointCloud, path: Path) -> None:
    """COLMAP points3D.bin with empty tracks; OpenSplat reads xyz and rgb."""
    record = np.dtype([("id", "<u8"), ("xyz", "<f8", 3), ("rgb", "u1", 3),
                       ("error", "<f8"), ("track_length", "<u8")])
    data = np.zeros(len(cloud), dtype=record)
    data["id"] = np.arange(1, len(cloud) + 1)
    data["xyz"] = cloud.points
    data["rgb"] = cloud.colors
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(cloud)))
        f.write(data.tobytes())


def downsample_to(cloud: PointCloud, max_points: int) -> PointCloud:
    """Coarsen the voxel grid until the cloud fits the point budget."""
    if len(cloud) <= max_points:
        return cloud
    extent = float(np.linalg.norm(cloud.points.max(0) - cloud.points.min(0)))
    voxel = extent / 1000
    while True:
        reduced = voxel_downsample(cloud, voxel)
        if len(reduced) <= max_points:
            return reduced
        voxel *= 1.3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("space", help="space folder (e.g. spaces/walkthrough)")
    parser.add_argument("--max-points", type=int, default=250_000,
                        help="seed point budget (default 250000; more uses more memory)")
    parser.add_argument("--hold-out", type=Path, metavar="REPORT.json",
                        help="splat-gsplat.json: leave its held-out frames out of the project (written to "
                             "splat-project-heldout), for a splat that has never seen them")
    parser.add_argument("--sparse", action="store_true",
                        help="seed from COLMAP's sparse points instead of the dense cloud")
    args = parser.parse_args()

    space = Path(args.space).resolve()
    # The dense cloud lives in the frame of the model densify used.
    model_dir = space_model_dir(space)
    if model_dir is None:
        sys.exit(f"no camera model in {space / 'workspace' / 'sparse'}")
    print(f"Cameras from {model_dir}")

    drop = set(json.loads(args.hold_out.read_text())["held_out"]) if args.hold_out else set()
    project = space / ("splat-project-heldout" if drop else "splat-project")
    if project.exists():
        shutil.rmtree(project)
    project.mkdir()
    (project / "cameras.bin").symlink_to((model_dir / "cameras.bin").resolve())
    if drop:
        dropped = without_images(model_dir / "images.bin", project / "images.bin", drop)
        print(f"Left out {dropped} held-out frames")
    else:
        (project / "images.bin").symlink_to((model_dir / "images.bin").resolve())
    (project / "images").symlink_to((space / "workspace" / "images").resolve())

    dense = space / "cloud-dense.ply"
    if dense.exists() and not args.sparse:
        cloud = downsample_to(load_ply(dense), args.max_points)
        write_points3d_bin(cloud, project / "points3D.bin")
        print(f"Seeded from the dense cloud: {len(cloud):,} points")
    else:
        (project / "points3D.bin").symlink_to((model_dir / "points3D.bin").resolve())
        print("Seeded from COLMAP's sparse points")


if __name__ == "__main__":
    main()
