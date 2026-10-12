#!/usr/bin/env python3
"""Build the surface fill for the phone from Big-LaMa: SurfaceFill.mlpackage.

Big-LaMa (Samsung AI, Apache-2.0 code and weights) is what tools/surface_fill.py
runs today to continue a floor or wall into the parts no frame saw. This is the
same model and the same weights (IOPaint's TorchScript big-lama.pt from GitHub),
so the phone fills exactly as the Mac does:
  SurfaceFill: image (1, 3, 512, 512), RGB in 0-1, and mask (1, 1, 512, 512), 1 where
               unknown -> filled (1, 3, 512, 512), RGB in 0-1, the known pixels kept.
The caller shrinks a surface photo so its longer side is 512 (box filter; a pixel
is unknown when any of it was), pads it to the square, and puts the fill back only
where the photo was unknown, as surface_fill.inpaint does.

Why these choices, all measured on frames from our clips (docs/on-device-plan-2026-10.md):
  * 512 and fixed. LaMa's Fourier units become fixed-size DFT matrix products in
    Core ML, so the shape cannot vary (enumerated shapes fail to convert). 512 is also
    better than the photo's own size: at 1024 a footprint-sized hole came back as a
    grey waffle, at 512 it carried on the planks (+3.6 dB in the hole).
  * Big-LaMa, not MI-GAN-512 (the mobile GAN): LaMa scored higher in the hole in 23
    of 24 cases (+4.5 dB on footprints) and MI-GAN streaked wood and invented objects.
  * fp32 compute on the GPU (CPU_AND_GPU). The Neural Engine runs fp16 only, and LaMa
    breaks there: mlx-lama-swift found fp16 collapses the Fourier convolutions, and
    CoreMLaMa's author has had no working fp16 Neural Engine reports.
  * int8 weights for the plain convolutions but the first and last: 58 MB instead
    of 196 MB. The package's own weights, read back and run in PyTorch, moved the fill
    by -0.05 dB on average (worst -0.61) over the 24 fills. Quantizing the first and
    last as well cost 11 dB; quantizing the three transposed convolutions doubled the
    loss (probably because their batch norm folds in along the other axis).
    --fp32-weights keeps every weight in fp32.
Its time and memory on an iPhone 13 are not measured yet: --check on a Mac compares
the package with PyTorch, and the first device run times it.

Not built by build.sh and not in project.yml yet: the app has no fill stage until
splats are trained on the phone (third_party/arkit-3dgs-scanner).

    ~/.venvs/oasis-coreml/bin/python apps/OasisCapture/scripts/convert_fill_lama.py [--fp32-weights]
    ~/.venvs/oasis-coreml/bin/python apps/OasisCapture/scripts/convert_fill_lama.py --check frame.jpg ...   (macOS)
The weights are fetched from GitHub into ~/.cache/oasisspaces/big-lama.pt, where
surface_fill.py looks for them.
"""

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = Path.home() / ".cache" / "oasisspaces" / "big-lama.pt"
WEIGHTS_URL = "https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt"
WEIGHTS_MD5 = "e3aa4aaa15225a33ec84f9f4bc47e500"
OUT = ROOT / "Resources" / "SurfaceFill.mlpackage"
SIZE = 512


def load() -> torch.jit.ScriptModule:
    if not WEIGHTS.exists():
        WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
        print(f"fetching {WEIGHTS_URL}")
        urllib.request.urlretrieve(WEIGHTS_URL, WEIGHTS)
    digest = hashlib.md5(WEIGHTS.read_bytes()).hexdigest()
    if digest != WEIGHTS_MD5:
        sys.exit(f"{WEIGHTS} has md5 {digest}, expected {WEIGHTS_MD5}")
    return torch.jit.load(str(WEIGHTS), map_location="cpu").eval()


class Fill(torch.nn.Module):
    """As surface_fill.inpaint calls LaMa: the unknown is blanked before it goes in."""

    def __init__(self, lama):
        super().__init__()
        self.lama = lama

    def forward(self, image, mask):
        return self.lama(image * (1 - mask), mask)


def convert(fp32_weights: bool) -> None:
    import coremltools as ct

    fill = Fill(load()).eval()
    example = (torch.rand(1, 3, SIZE, SIZE), (torch.rand(1, 1, SIZE, SIZE) > 0.7).float())
    with torch.inference_mode():
        traced = torch.jit.trace(fill, example)
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="image", shape=(1, 3, SIZE, SIZE)),
                ct.TensorType(name="mask", shape=(1, 1, SIZE, SIZE))],
        outputs=[ct.TensorType(name="filled")],
        minimum_deployment_target=ct.target.iOS18, compute_precision=ct.precision.FLOAT32,
        compute_units=ct.ComputeUnit.CPU_AND_GPU, convert_to="mlprogram")
    if not fp32_weights:
        import coremltools.optimize.coreml as cto

        # Plain convolutions only: the Fourier units' DFT matrices stay fp32 (their sine
        # matrix has an all-zero row, which int8 per-channel scaling turns into NaN), and
        # so do the transposed convolutions and the first and last convolution.
        convs = [op.name for f in mlmodel._mil_program.functions.values() for op in f.operations
                 if op.op_type == "conv"]
        int8 = cto.OpLinearQuantizerConfig(mode="linear_symmetric", granularity="per_channel")
        config = cto.OptimizationConfig(op_type_configs={"conv": int8},
                                        op_name_configs={convs[0]: None, convs[-1]: None})
        mlmodel = cto.linear_quantize_weights(mlmodel, config)
    mlmodel.short_description = ("Big-LaMa (Apache-2.0) surface fill: 512x512 RGB 0-1 and mask (1 = unknown) "
                                 "-> filled RGB 0-1; run on CPU_AND_GPU in fp32")
    mlmodel.save(str(OUT))
    size = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file())
    print(f"wrote {OUT} ({size / 1e6:.1f} MB)")


def masks() -> dict:
    """Holes like surface_fill's: a furniture footprint, a region no frame saw, thin occluders."""
    from PIL import Image, ImageDraw

    out = {}
    m = Image.new("L", (SIZE, SIZE), 0)
    ImageDraw.Draw(m).rectangle([100, 250, 412, 511], fill=255)
    out["footprint"] = np.array(m) > 0
    m = Image.new("L", (SIZE, SIZE), 0)
    d = ImageDraw.Draw(m)
    for (x, y, r) in [(200, 200, 90), (290, 260, 100), (240, 330, 80), (330, 180, 60)]:
        d.ellipse([x - r, y - r, x + r, y + r], fill=255)
    out["unseen blob"] = np.array(m) > 0
    m = Image.new("L", (SIZE, SIZE), 0)
    d = ImageDraw.Draw(m)
    for x in (90, 230, 380):
        d.rectangle([x, 0, x + 14, SIZE - 1], fill=255)
    d.rectangle([0, 440, SIZE - 1, 458], fill=255)
    out["strips"] = np.array(m) > 0
    return out


def check(images: list) -> None:
    """The package against PyTorch on each image's centre square, three holes each."""
    import coremltools as ct
    from PIL import Image

    if sys.platform != "darwin":
        sys.exit("--check runs the Core ML package, which needs macOS")
    lama = load()
    mlmodel = ct.models.MLModel(str(OUT), compute_units=ct.ComputeUnit.CPU_AND_GPU)
    for path in images:
        photo = Image.open(path).convert("RGB")
        side = min(photo.size)
        left, top = (photo.width - side) // 2, (photo.height - side) // 2
        img = np.asarray(photo.crop((left, top, left + side, top + side)).resize((SIZE, SIZE), Image.BOX),
                         np.float32) / 255.0
        line = [Path(path).name]
        for name, hole in masks().items():
            x = torch.from_numpy(img.transpose(2, 0, 1)[None].copy())
            m = torch.from_numpy(hole.astype(np.float32))[None, None]
            with torch.inference_mode():
                ref = lama(x * (1 - m), m)[0].numpy()
            ours = mlmodel.predict({"image": x.numpy(), "mask": m.numpy()})["filled"][0]
            truth = img.transpose(2, 0, 1)
            psnr = lambda a: 10 * np.log10(1 / max(1e-12, float(((a - truth)[:, hole] ** 2).mean())))
            agree = 10 * np.log10(1 / max(1e-12, float(((ours - ref)[:, hole] ** 2).mean())))
            line.append(f"{name}: Core ML {psnr(ours):.2f} dB, PyTorch {psnr(ref):.2f} dB, "
                        f"the two agree to {agree:.1f} dB")
        print("  " + "; ".join(line))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fp32-weights", action="store_true", help="keep every weight in fp32 (196 MB)")
    parser.add_argument("--check", nargs="+", metavar="IMAGE", help="compare the built package with PyTorch (macOS)")
    args = parser.parse_args()
    if args.check:
        check(args.check)
    else:
        convert(args.fp32_weights)
