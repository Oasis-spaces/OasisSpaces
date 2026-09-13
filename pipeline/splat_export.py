#!/usr/bin/env python3
"""Write a trained splat in the viewer's compact .splat format.

OpenSplat saves 62 float32 properties per Gaussian (248 bytes, spherical
harmonics included). splat-viewer/ also reads .splat: position, scale, colour
with opacity and rotation in 32 bytes, sorted largest and most opaque first so
the scene appears as it streams in. The walkthrough's 46 MB splat.ply becomes
6 MB, and loads where the .ply broke off: Python's http.server on macOS failed
mid-transfer with "No buffer space available". The view-dependent colour
(higher spherical harmonics) is dropped, as the viewer ignores it anyway.

It also writes <name>.view.json: the pose of the capture frame that shows the
most of the room (see start_view). splat-viewer opens there instead of
at its demo camera, which sits somewhere no frame was taken and shows a smear.

Usage:
    python3 pipeline/splat_export.py spaces/<name>/splat.ply [out.splat]
"""

from __future__ import annotations

import json
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


START_STEP_IN = 0.0  # stepping in made the viewer draw nothing for the walkthrough; revisit


def start_view(space: Path) -> dict | None:
    """A viewer camera at the capture frame that shows the most of the room:
    it faces the middle of the solved points and sees many of them from far
    away. (Most matched points alone picks close-ups of textured surfaces.)"""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from densify import read_images_bin, read_points3d_bin

    candidates = []
    meta = space / "densify.json"
    if meta.exists():
        candidates.append(Path(json.loads(meta.read_text())["model_dir"]))
    candidates += sorted((p.parent for p in (space / "workspace" / "sparse").glob("*/images.bin")),
                         key=lambda d: -(d / "images.bin").stat().st_size)
    candidates.append(space / "splat-project")
    model = next((d for d in candidates
                  if (d / "images.bin").exists() and (d / "points3D.bin").exists()), None)
    if model is None:
        return None
    images = read_images_bin(model / "images.bin")
    points = read_points3d_bin(model / "points3D.bin")
    if not points:
        return None
    xyz = np.array(list(points.values()))
    centre = np.median(xyz, axis=0)
    best, best_score = None, -1.0
    for im in images.values():
        R, t = im["R"], im["t"]
        ahead = R @ centre + t
        if ahead[2] <= 0 or ahead[2] / np.linalg.norm(ahead) < np.cos(np.radians(45)):
            continue  # the middle of the room is not in front of this camera
        seen = [points[i] for i in im["point3D_ids"] if i >= 0 and i in points]
        if len(seen) < 50:
            continue
        depth = float(np.median((np.array(seen) @ R.T + t)[:, 2]))
        score = np.sqrt(len(seen)) * depth
        if score > best_score:
            best, best_score = im, score
    if best is None:
        best = max(images.values(), key=lambda im: int((im["point3D_ids"] >= 0).sum()))
    R, t = best["R"], best["t"].copy()
    # Splats often keep a haze of stray Gaussians right around the recorded
    # camera positions, so step forward a third of the way to what the frame
    # looks at (moving the camera forward lowers every point's depth).
    seen = [points[i] for i in best["point3D_ids"] if i >= 0 and i in points]
    if seen:
        t[2] -= START_STEP_IN * float(np.median((np.array(seen) @ R.T + t)[:, 2]))
    # Column-major world-to-camera matrix, the layout splat-viewer's view uses.
    view = [R[0][0], R[1][0], R[2][0], 0, R[0][1], R[1][1], R[2][1], 0,
            R[0][2], R[1][2], R[2][2], 0, t[0], t[1], t[2], 1]
    return {"frame": best["name"], "viewMatrix": [round(float(v), 5) for v in view]}


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else src.with_suffix(".splat")
    count = ply_to_splat(src, dst)
    print(f"{count:,} gaussians -> {dst} ({dst.stat().st_size / 1e6:.1f} MB, "
          f"from {src.stat().st_size / 1e6:.1f} MB)")
    view = start_view(src.parent)
    if view:
        view_path = dst.with_name(dst.stem + ".view.json")
        view_path.write_text(json.dumps(view) + "\n")
        print(f"starting camera: {view['frame']} -> {view_path}")


if __name__ == "__main__":
    main()
