#!/usr/bin/env python3
"""Build the mask refiner: MaskEncoder.mlpackage and MaskDecoder.mlpackage.

The object detector finds each thing and a rough mask of it at a quarter of its
input (160x120): outlines come out blocky, and a cluttered bed or desk
fragments into pieces. MobileSAM (Segment Anything with a 5 M parameter
Tiny-ViT encoder, Apache-2.0, https://github.com/ChaoningZhang/MobileSAM)
turns the detector's box into one clean whole-object mask:

  MaskEncoder: a 1024x1024 RGB image -> image embedding (1, 256, 64, 64), once a frame.
  MaskDecoder: embedding + a box (x0, y0, x1, y1 in the 1024 square's pixels) ->
               a 256x256 mask (logits; > 0 is inside) and a score, once per thing.

Needs the oasis-coreml venv with mobile_sam installed
(pip install git+https://github.com/ChaoningZhang/MobileSAM.git) and the
checkpoint in ~/.cache/oasis-models/mobile_sam.pt:
    ~/.venvs/oasis-coreml/bin/python apps/OasisCapture/scripts/convert_refiner.py [--check image.jpg]
"""

import argparse
import json
import sys
from pathlib import Path

import coremltools as ct
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
CACHE = Path.home() / ".cache" / "oasis-models"
CHECKPOINT = CACHE / "mobile_sam.pt"
ENCODER_OUT = ROOT / "Resources" / "MaskEncoder.mlpackage"
DECODER_OUT = ROOT / "Resources" / "MaskDecoder.mlpackage"
SIZE = 1024


class Encoder(torch.nn.Module):
    def __init__(self, sam):
        super().__init__()
        self.sam = sam
        self.register_buffer("mean", sam.pixel_mean.view(1, 3, 1, 1))
        self.register_buffer("std", sam.pixel_std.view(1, 3, 1, 1))

    def forward(self, image):   # RGB 0...255
        return self.sam.image_encoder((image - self.mean) / self.std)


class Decoder(torch.nn.Module):
    """Prompt encoder and mask decoder for one box, the single best mask."""

    def __init__(self, sam):
        super().__init__()
        self.sam = sam

    def forward(self, embedding, box):
        # box: (1, 4) as x0, y0, x1, y1 in 1024-space pixels. (Corner points with labels 2 and 3,
        # the ONNX export's trick, do not work here: this prompt encoder only embeds labels 0 and 1.)
        sparse, dense = self.sam.prompt_encoder(points=None, boxes=box, masks=None)
        masks, scores = self.sam.mask_decoder(
            image_embeddings=embedding, image_pe=self.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense, multimask_output=False)
        return masks[:, 0], scores[:, 0]


def load():
    from mobile_sam import sam_model_registry

    if not CHECKPOINT.exists():
        sys.exit(f"download MobileSAM's weights to {CHECKPOINT} first")
    return sam_model_registry["vit_t"](checkpoint=str(CHECKPOINT)).eval()


def convert(encoder_too: bool = True) -> None:
    sam = load()
    if not encoder_too:
        convert_decoder(sam)
        return
    encoder = Encoder(sam).eval()
    traced = torch.jit.trace(encoder, torch.rand(1, 3, SIZE, SIZE) * 255)
    model = ct.convert(
        traced,
        inputs=[ct.ImageType(name="image", shape=(1, 3, SIZE, SIZE), color_layout=ct.colorlayout.RGB)],
        outputs=[ct.TensorType(name="embedding")],
        minimum_deployment_target=ct.target.iOS17, compute_precision=ct.precision.FLOAT16,
        convert_to="mlprogram")
    model.short_description = "MobileSAM image encoder (Tiny-ViT): 1024x1024 image -> 256x64x64 embedding"
    model.save(str(ENCODER_OUT))
    print(f"wrote {ENCODER_OUT}")
    convert_decoder(sam)


def convert_decoder(sam) -> None:
    decoder = Decoder(sam).eval()
    example = (torch.rand(1, 256, 64, 64), torch.tensor([[200.0, 300.0, 700.0, 900.0]]))
    traced = torch.jit.trace(decoder, example)
    model = ct.convert(
        traced,
        inputs=[ct.TensorType(name="embedding", shape=(1, 256, 64, 64)),
                ct.TensorType(name="box", shape=(1, 4))],
        outputs=[ct.TensorType(name="mask"), ct.TensorType(name="score")],
        minimum_deployment_target=ct.target.iOS17, compute_precision=ct.precision.FLOAT16,
        convert_to="mlprogram")
    model.short_description = "MobileSAM prompt encoder + mask decoder: embedding + box -> 256x256 mask logits"
    model.save(str(DECODER_OUT))
    print(f"wrote {DECODER_OUT}")
    for out in (ENCODER_OUT, DECODER_OUT):
        size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
        print(f"  {out.name}: {size / 1e6:.1f} MB")


def check(image_path: str) -> None:
    """Runs the detector, then the refiner on each detected box, and draws both outlines."""
    from PIL import Image, ImageDraw

    spec = json.loads((ROOT / "Packages/CaptureRules/Sources/CaptureRules/Resources/object-classes.json").read_text())
    detector = ct.models.MLModel(str(ROOT / "Resources" / "RoomObjects.mlpackage"))
    encoder = ct.models.MLModel(str(ENCODER_OUT))
    decoder = ct.models.MLModel(str(DECODER_OUT))
    source = Image.open(image_path).convert("RGB")
    w, h = spec["inputSize"]
    small = source.resize((w, h))
    out = {k: np.asarray(v) for k, v in detector.predict({"image": small}).items()}
    protos = next(v for v in out.values() if v.ndim == 4)[0]
    preds = next(v for v in out.values() if v.ndim == 3)[0]
    nc = len(spec["classes"])
    boxes, scores, coeffs = preds[:4].T, preds[4:4 + nc].T, preds[4 + nc:].T
    best, conf = scores.argmax(1), scores.max(1)
    order = np.flatnonzero(conf >= spec["confidence"])
    order = order[np.argsort(-conf[order])]

    def iou(a, b):
        ax0, ay0, ax1, ay1 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
        bx0, by0, bx1, by1 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
        iw, ih = max(0, min(ax1, bx1) - max(ax0, bx0)), max(0, min(ay1, by1) - max(ay0, by0))
        return iw * ih / (a[2] * a[3] + b[2] * b[3] - iw * ih + 1e-6)

    kept = []
    for i in order:
        if all(iou(boxes[i], boxes[j]) < spec["iou"] for j in kept):
            kept.append(i)
    # The refiner sees the image letterboxed into 1024x1024 (long side 1024).
    scale = SIZE / max(source.size)
    padded = Image.new("RGB", (SIZE, SIZE))
    padded.paste(source.resize((round(source.width * scale), round(source.height * scale))), (0, 0))
    import time
    t = time.time()
    embedding = encoder.predict({"image": padded})["embedding"]
    print(f"encoder {time.time() - t:.2f} s")
    canvas = source.copy()
    draw = ImageDraw.Draw(canvas)
    for i in kept[:10]:
        x, y, bw, bh = boxes[i]
        x0, y0, x1, y1 = (x - bw / 2) / w, (y - bh / 2) / h, (x + bw / 2) / w, (y + bh / 2) / h
        box = np.array([[x0 * source.width * scale, y0 * source.height * scale,
                         x1 * source.width * scale, y1 * source.height * scale]], np.float32)
        t = time.time()
        result = decoder.predict({"embedding": embedding, "box": box})
        mask = np.asarray(result["mask"])[0] > 0           # 256x256 of the 1024 square
        took = time.time() - t
        # Crop the 256 mask to the image's part of the square, draw its edge.
        mw, mh = round(source.width * scale / 4), round(source.height * scale / 4)
        big = np.array(Image.fromarray((mask[:mh, :mw] * 255).astype(np.uint8)).resize(source.size, Image.BILINEAR)) > 127
        edge = (big ^ np.roll(big, 1, 0)) | (big ^ np.roll(big, 1, 1))
        overlay = Image.new("RGBA", source.size, (48, 209, 88, 0))
        overlay.putalpha(Image.fromarray((edge * 255).astype(np.uint8)))
        canvas.paste(overlay, (0, 0), overlay)
        label = spec["classes"][best[i]]["label"]
        draw.text((x0 * source.width + 3, y0 * source.height + 2), f"{label} {float(result['score'][0]):.2f}", fill=(255, 255, 255))
        print(f"  {label:14s} detector {conf[i]:.2f}  refiner score {float(result['score'][0]):.2f}  mask {int(mask.sum())} px  {took * 1000:.0f} ms")
    target = Path(image_path).with_name(Path(image_path).stem + "-refined.png")
    canvas.save(target)
    print("drew", target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", metavar="IMAGE")
    parser.add_argument("--decoder-only", action="store_true")
    args = parser.parse_args()
    check(args.check) if args.check else convert(encoder_too=not args.decoder_only)
