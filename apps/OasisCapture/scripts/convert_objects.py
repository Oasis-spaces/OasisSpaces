#!/usr/bin/env python3
"""Build RoomObjects.mlpackage, the on-device object detector: one mask per
piece of furniture, so each becomes one box in the room map.

YOLOE-11m (segmentation, https://docs.ultralytics.com/models/yoloe/) is an
open-vocabulary detector: it is prompted with the class names in
object-classes.json (bed, sofa, wardrobe, ... 71 room things, no clothes), the
text embeddings are baked into the head, and the exported model needs no text
encoder. Input: a 480x640 (w x h) upright RGB image. Outputs: boxes with a score
per class for every anchor, and 32 mask prototypes at a quarter of the input;
the phone decodes both (see InstanceDecoder in CaptureRules).

Needs the oasis-coreml venv (torch 2.5.1, coremltools 8.3, ultralytics 8.4) and
the weights in ~/.cache/oasis-models (yoloe-11m-seg.pt and the MobileCLIP text
encoder mobileclip_blt.ts, both from the ultralytics assets release):
    ~/.venvs/oasis-coreml/bin/python apps/OasisCapture/scripts/convert_objects.py [--check image.jpg]
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPEC = ROOT / "Packages" / "CaptureRules" / "Sources" / "CaptureRules" / "Resources" / "object-classes.json"
OUT = ROOT / "Resources" / "RoomObjects.mlpackage"
CACHE = Path.home() / ".cache" / "oasis-models"
WEIGHTS = "yoloe-11m-seg.pt"
ASSETS = "https://github.com/ultralytics/assets/releases/download/v8.3.0/"


def fetch(name: str) -> Path:
    path = CACHE / name
    if not path.exists():
        import urllib.request
        CACHE.mkdir(parents=True, exist_ok=True)
        print(f"downloading {name}...")
        urllib.request.urlretrieve(ASSETS + name, path)
    return path


def convert() -> None:
    spec = json.loads(SPEC.read_text())
    names = [c["prompt"] for c in spec["classes"]]
    width, height = spec["inputSize"]
    weights = fetch(WEIGHTS)
    fetch("mobileclip_blt.ts")
    os.chdir(CACHE)   # ultralytics looks for the text encoder in the working directory
    from ultralytics import YOLOE

    model = YOLOE(str(weights))
    model.set_classes(names, model.get_text_pe(names))
    exported = Path(model.export(format="coreml", imgsz=(height, width), half=True, nms=False, device="cpu"))
    if OUT.exists():
        shutil.rmtree(OUT)
    shutil.move(str(exported), str(OUT))

    import coremltools as ct
    mlmodel = ct.models.MLModel(str(OUT))
    desc = mlmodel.get_spec().description
    print("inputs:", [(i.name, i.type.WhichOneof("Type")) for i in desc.input])
    print("outputs:", [(o.name, list(o.type.multiArrayType.shape)) for o in desc.output])
    size = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file())
    print(f"wrote {OUT} ({size / 1e6:.0f} MB), {len(names)} classes")


def check(image: str) -> None:
    """Runs the Core ML model on the Mac and draws what it finds next to the image."""
    import coremltools as ct
    import numpy as np
    from PIL import Image, ImageDraw

    spec = json.loads(SPEC.read_text())
    width, height = spec["inputSize"]
    mlmodel = ct.models.MLModel(str(OUT))
    source = Image.open(image).convert("RGB")
    resized = source.resize((width, height))
    out = mlmodel.predict({"image": resized})
    arrays = {k: np.asarray(v) for k, v in out.items()}
    protos = next(v for v in arrays.values() if v.ndim == 4 and v.shape[1] == 32)[0]
    preds = next(v for v in arrays.values() if v.ndim == 3)[0]   # (4 + classes, anchors)
    classes = spec["classes"]
    nc = len(classes)
    boxes = preds[:4].T
    scores = preds[4:4 + nc].T
    coeffs = preds[4 + nc:].T
    best = scores.argmax(1)
    conf = scores.max(1)
    keep = conf >= spec["confidence"]
    order = np.argsort(-conf[keep])
    idx = np.flatnonzero(keep)[order]

    def iou(a, b):
        ax0, ay0, ax1, ay1 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
        bx0, by0, bx1, by1 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
        w = max(0, min(ax1, bx1) - max(ax0, bx0)); h = max(0, min(ay1, by1) - max(ay0, by0))
        inter = w * h
        return inter / (a[2] * a[3] + b[2] * b[3] - inter + 1e-6)

    kept = []
    for i in idx:
        if all(iou(boxes[i], boxes[j]) < spec["iou"] for j in kept):
            kept.append(i)
    draw = ImageDraw.Draw(resized, "RGBA")
    mh, mw = protos.shape[1:]
    for i in kept[:20]:
        c = classes[best[i]]
        x, y, w, h = boxes[i]
        mask = 1 / (1 + np.exp(-(coeffs[i] @ protos.reshape(32, -1)).reshape(mh, mw)))
        ys, xs = np.mgrid[0:mh, 0:mw]
        inside = (xs * width / mw >= x - w / 2) & (xs * width / mw <= x + w / 2) & (ys * height / mh >= y - h / 2) & (ys * height / mh <= y + h / 2)
        mask = (mask > spec["maskThreshold"]) & inside
        overlay = Image.fromarray((mask * 110).astype(np.uint8)).resize((width, height))
        colour = Image.new("RGBA", (width, height), (255, 159, 10, 0))
        colour.putalpha(overlay)
        resized.alpha_composite(colour) if resized.mode == "RGBA" else resized.paste(colour, (0, 0), colour)
        draw.rectangle([x - w / 2, y - h / 2, x + w / 2, y + h / 2], outline=(255, 159, 10, 255), width=2)
        draw.text((x - w / 2 + 3, y - h / 2 + 2), f"{c['label']} {conf[i]:.2f}", fill=(255, 255, 255, 255))
        print(f"{c['label']:18s} {conf[i]:.2f}  box {x:.0f},{y:.0f} {w:.0f}x{h:.0f}  mask px {int(mask.sum())}")
    target = Path(image).with_name(Path(image).stem + "-objects.png")
    resized.save(target)
    print("drew", target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", metavar="IMAGE", help="run the built model on an image and draw the result")
    args = parser.parse_args()
    if args.check:
        check(args.check)
    else:
        convert()
