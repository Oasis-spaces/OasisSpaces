#!/usr/bin/env python3
"""Build RoomMetricDepth.mlpackage: MoGe-2 small, the phone's depth in metres.

The depth model used to be Depth Anything V2 Small: a relative depth that the
phone could only turn into metres by fitting a line through ARKit's tracking
points, so on a plain wall or a fast move, with too few points, nothing could
be placed. MoGe-2 (Microsoft, MIT, https://github.com/microsoft/MoGe) predicts
an affine-invariant point map plus its own metric scale; with the camera's
focal length known (ARKit gives it), the only unknown is a depth offset, which
the phone solves in closed form. Measured against the pipeline's own
reconstruction of two rooms: MoGe-2 small alone is 8.5% off per pixel (scale
0.99), 5.9% with the tracking points as a correction; the old model had no
answer at all without points.

Input: the sensor's landscape image squeezed to 518x392 (RGB). Outputs:
  points       (1, 392, 518, 3)  camera-space x, y, z, z up to a shift
  mask         (1, 392, 518)     1 where the point is valid
  metric_scale (1,)              multiply the shifted points by this for metres
Conventions (MoGe): the optical centre is the image centre; u, v span
+-aspect/sqrt(1+aspect^2) and +-1/sqrt(1+aspect^2); focal is relative to
half the diagonal; focal * x / (z + shift) = u.

Needs the oasis-coreml venv with `moge` (pip install --no-deps from the pinned
commit in the README) and `utils3d`:
    ~/.venvs/oasis-coreml/bin/python apps/OasisCapture/scripts/convert_depth_moge.py [--check image.jpg]
"""

import argparse
import sys
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "Resources" / "RoomMetricDepth.mlpackage"
CHECKPOINT = "Ruicheng/moge-2-vits-normal"
WIDTH, HEIGHT = 518, 392
TOKENS = 1200


def load_patched():
    """MoGe with two things Core ML cannot trace replaced: its bicubic resize
    (bilinear instead; the input is already the token grid's size) and the
    view-plane grids made from meshgrid (constants, for a fixed input)."""
    import moge.model.v2 as v2
    from moge.model.v2 import MoGeModel

    original = F.interpolate

    def interpolate(x, *args, **kwargs):
        if kwargs.get("mode") == "bicubic":
            kwargs["mode"] = "bilinear"
            kwargs.pop("antialias", None)
        return original(x, *args, **kwargs)

    v2.F.interpolate = interpolate
    grid = v2.normalized_view_plane_uv
    cache = {}

    def constant_uv(width, height, aspect_ratio=None, dtype=None, device=None):
        key = (int(width), int(height))
        if key not in cache:
            with torch.no_grad():
                cache[key] = grid(width=key[0], height=key[1], aspect_ratio=aspect_ratio, dtype=dtype, device=device).clone()
        return cache[key]

    v2.normalized_view_plane_uv = constant_uv
    return MoGeModel.from_pretrained(CHECKPOINT).eval()


class Wrapped(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image):   # RGB 0...1
        out = self.model.forward(image, num_tokens=TOKENS)
        return out["points"], out["mask"], out["metric_scale"]


def convert() -> None:
    wrapped = Wrapped(load_patched()).eval()
    example = torch.rand(1, 3, HEIGHT, WIDTH)
    with torch.no_grad():
        wrapped(example)                       # fills the grid cache before tracing
    traced = torch.jit.trace(wrapped, example, check_trace=False)
    model = ct.convert(
        traced,
        inputs=[ct.ImageType(name="image", shape=(1, 3, HEIGHT, WIDTH), scale=1 / 255.0, color_layout=ct.colorlayout.RGB)],
        outputs=[ct.TensorType(name="points"), ct.TensorType(name="mask"), ct.TensorType(name="metric_scale")],
        minimum_deployment_target=ct.target.iOS17, compute_precision=ct.precision.FLOAT16, convert_to="mlprogram")
    model.short_description = f"MoGe-2 small ({CHECKPOINT}): {WIDTH}x{HEIGHT} image -> point map (z up to a shift), mask, metric scale"
    model.user_defined_metadata["source"] = CHECKPOINT
    model.save(str(OUT))
    size = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file())
    print(f"wrote {OUT} ({size / 1e6:.0f} MB)")


def view_plane(width: int, height: int):
    """(u, v) grids as MoGe defines them."""
    aspect = width / height
    span_x, span_y = aspect / (1 + aspect ** 2) ** 0.5, 1 / (1 + aspect ** 2) ** 0.5
    u = np.linspace(-span_x * (width - 1) / width, span_x * (width - 1) / width, width)
    v = np.linspace(-span_y * (height - 1) / height, span_y * (height - 1) / height, height)
    return np.meshgrid(u, v)


def recover_shift(points: np.ndarray, mask: np.ndarray, focal_rel: float) -> float:
    """The depth shift, in closed form: focal * x / (z + s) = u gives s per pixel;
    the median over confident pixels away from the centre (the same maths the
    phone runs, see CaptureRules/MetricDepth.swift)."""
    height, width = mask.shape
    u, v = view_plane(width, height)
    x, y, z = points[..., 0], points[..., 1], points[..., 2]
    ok = mask > 0.5
    estimates = []
    for coord, axis in ((u, x), (v, y)):
        use = ok & (np.abs(coord) > 0.15)
        estimates.append((focal_rel * axis[use] / coord[use] - z[use]))
    estimates = np.concatenate(estimates)
    return float(np.median(estimates)) if len(estimates) else 0.0


def metres(result: dict, focal_px: float, width_px: float, height_px: float) -> np.ndarray:
    """Depth in metres from the model's outputs, given the camera's focal length
    in pixels of a width_px x height_px image (the same aspect as the input)."""
    points = np.asarray(result["points"])[0].astype(np.float32)
    mask = np.asarray(result["mask"])[0].astype(np.float32)
    scale = float(np.asarray(result["metric_scale"]).ravel()[0])
    focal_rel = focal_px / ((width_px ** 2 + height_px ** 2) ** 0.5 / 2)
    shift = recover_shift(points, mask, focal_rel)
    depth = (points[..., 2] + shift) * scale
    return np.where(mask > 0.5, depth, np.nan)


def check(image_path: str) -> None:
    from PIL import Image

    model = ct.models.MLModel(str(OUT))
    source = Image.open(image_path).convert("RGB")
    if source.height > source.width:
        source = source.rotate(90, expand=True)           # the sensor is landscape
    result = model.predict({"image": source.resize((WIDTH, HEIGHT))})
    # Without the real focal: assume a phone's wide camera (about 70 degrees across the long side).
    focal = source.width / (2 * np.tan(np.radians(70) / 2))
    depth = metres(result, focal, source.width, source.height)
    print(f"metric scale {float(np.asarray(result['metric_scale']).ravel()[0]):.3f}; depth metres: "
          f"min {np.nanmin(depth):.2f} median {np.nanmedian(depth):.2f} max {np.nanmax(depth):.2f}")
    picture = np.clip((1 - (depth - np.nanmin(depth)) / (np.nanmax(depth) - np.nanmin(depth) + 1e-6)) * 255, 0, 255)
    target = Path(image_path).with_name(Path(image_path).stem + "-metric-depth.png")
    Image.fromarray(np.nan_to_num(picture).astype(np.uint8)).save(target)
    print("drew", target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", metavar="IMAGE")
    args = parser.parse_args()
    check(args.check) if args.check else convert()
