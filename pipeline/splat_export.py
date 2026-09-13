#!/usr/bin/env python3
"""Write a trained splat in the viewer's compact .splat format.

OpenSplat saves 62 float32 properties per Gaussian (248 bytes, spherical
harmonics included). splat-viewer/ also reads .splat: position, scale, colour
with opacity and rotation in 32 bytes, sorted largest and most opaque first so
the scene appears as it streams in. The walkthrough's 46 MB splat.ply becomes
6 MB, and loads where the .ply broke off: Python's http.server on macOS failed
mid-transfer with "No buffer space available". The view-dependent colour
(higher spherical harmonics) is dropped, as the viewer ignores it anyway.

Usage:
    python3 pipeline/splat_export.py spaces/<name>/splat.ply [out.splat]
"""

from __future__ import annotations

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


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else src.with_suffix(".splat")
    count = ply_to_splat(src, dst)
    print(f"{count:,} gaussians -> {dst} ({dst.stat().st_size / 1e6:.1f} MB, "
          f"from {src.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
