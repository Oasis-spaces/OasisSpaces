#!/usr/bin/env python3
"""Render a Gaussian splat to an image on the CPU, without a browser.

The agent needs pictures of a splat to show Claude (before and after a fill),
and a headless run has no WebGL. This follows the standard splatting maths the
viewer uses: each Gaussian's 3D covariance (scale and rotation) projected to a
2D ellipse through the camera's Jacobian, colour from its base (DC) term, and
front-to-back alpha compositing. It is slow compared with a GPU (seconds per
small image) and meant for a few review images, not for viewing.

Cameras use splat-viewer's view matrix (column-major, world to camera, +z
forward, image y down) and a vertical field of view, as in <name>.view.json.

Usage:
    python3 tools/splat_render.py spaces/<name>/splat.ply out.png [--view view.json] [--size 480 360]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from splat_tools import SH_C0, read_splat  # noqa: E402

NEAR = 0.2      # scene units, as the viewer's projection


def quaternion_matrices(q: np.ndarray) -> np.ndarray:
    """(n, 4) w-x-y-z quaternions -> (n, 3, 3) rotation matrices."""
    q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], axis=1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], axis=1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], axis=1),
    ], axis=1)


def render(arr: np.ndarray, view_matrix, width: int = 480, height: int = 360,
           fov_y: float = 55.0, with_coverage: bool = False):
    """HxWx3 uint8 image of the splat `arr` (splat_tools layout); with
    `with_coverage`, also the HxW share of each pixel the splat covers (1 is
    solid, 0 is a see-through gap)."""
    V = np.array(view_matrix, float).reshape(4, 4).T          # column-major -> rows
    W, t = V[:3, :3], V[:3, 3]
    xyz = np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype(np.float64)
    cam = xyz @ W.T + t
    z = cam[:, 2]
    focal = height / 2 / np.tan(np.radians(fov_y) / 2)
    keep = z > NEAR
    u = focal * cam[:, 0] / np.where(keep, z, 1) + width / 2
    v = focal * cam[:, 1] / np.where(keep, z, 1) + height / 2
    keep &= (u > -0.5 * width) & (u < 1.5 * width) & (v > -0.5 * height) & (v < 1.5 * height)
    alpha = 1 / (1 + np.exp(-arr["opacity"].astype(np.float64)))
    keep &= alpha > 1 / 255
    idx = np.flatnonzero(keep)
    if len(idx) == 0:
        empty = np.zeros((height, width, 3), np.uint8)
        return (empty, np.zeros((height, width))) if with_coverage else empty

    scales = np.exp(np.stack([arr["scale_0"], arr["scale_1"], arr["scale_2"]], axis=1)[idx]
                    .astype(np.float64))
    rot = quaternion_matrices(np.stack([arr["rot_0"], arr["rot_1"], arr["rot_2"], arr["rot_3"]],
                                       axis=1)[idx].astype(np.float64))
    M = rot * scales[:, None, :]                               # R @ diag(s)
    sigma = M @ M.transpose(0, 2, 1)                           # world covariance
    x, y, zz = cam[idx, 0], cam[idx, 1], z[idx]
    J = np.zeros((len(idx), 2, 3))
    J[:, 0, 0] = focal / zz
    J[:, 0, 2] = -focal * x / zz ** 2
    J[:, 1, 1] = focal / zz
    J[:, 1, 2] = -focal * y / zz ** 2
    T = J @ W
    cov = T @ sigma @ T.transpose(0, 2, 1)
    a = cov[:, 0, 0] + 0.3
    b = cov[:, 0, 1]
    c = cov[:, 1, 1] + 0.3
    det = a * c - b * b
    ok = det > 1e-12
    mid = (a + c) / 2
    radius = np.ceil(3 * np.sqrt(np.maximum(mid + np.sqrt(np.maximum(mid * mid - det, 0)), 0)))
    rgb = np.clip(0.5 + SH_C0 * np.stack([arr["f_dc_0"], arr["f_dc_1"], arr["f_dc_2"]], axis=1)[idx]
                  .astype(np.float64), 0, 1)
    order = np.argsort(zz)

    image = np.zeros((height, width, 3))
    trans = np.ones((height, width))
    uu, vv, al = u[idx], v[idx], alpha[idx]
    for k in order:
        if not ok[k]:
            continue
        r = radius[k]
        x0, x1 = int(max(uu[k] - r, 0)), int(min(uu[k] + r + 1, width))
        y0, y1 = int(max(vv[k] - r, 0)), int(min(vv[k] + r + 1, height))
        if x0 >= x1 or y0 >= y1:
            continue
        t_view = trans[y0:y1, x0:x1]
        if t_view.max() < 1e-3:
            continue
        ys, xs = np.mgrid[y0:y1, x0:x1]
        dx, dy = xs + 0.5 - uu[k], ys + 0.5 - vv[k]
        inv = 1 / det[k]
        power = -0.5 * (c[k] * dx * dx - 2 * b[k] * dx * dy + a[k] * dy * dy) * inv
        w = np.minimum(0.99, al[k] * np.exp(np.minimum(power, 0)))
        w[w < 1 / 255] = 0
        image[y0:y1, x0:x1] += (t_view * w)[..., None] * rgb[k]
        trans[y0:y1, x0:x1] = t_view * (1 - w)
    picture = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    return (picture, 1 - trans) if with_coverage else picture


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("splat")
    parser.add_argument("out")
    parser.add_argument("--view", help="a .view.json (default: the splat's own)")
    parser.add_argument("--size", nargs=2, type=int, default=(480, 360))
    args = parser.parse_args()
    splat = Path(args.splat)
    view_path = Path(args.view) if args.view else splat.with_suffix(".view.json")
    view = json.loads(view_path.read_text())
    arr, _ = read_splat(splat)
    image = render(arr, view["viewMatrix"], *args.size, fov_y=view.get("fovY", 55.0))
    Image.fromarray(image).save(args.out)
    print(args.out)


if __name__ == "__main__":
    main()
