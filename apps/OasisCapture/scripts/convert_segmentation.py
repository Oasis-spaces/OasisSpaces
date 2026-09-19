#!/usr/bin/env python3
"""Build RoomSegmentation.mlpackage, the on-device model for the room's surfaces:
walls, floor, ceiling, doors and windows (things come from the object
detector, see convert_objects.py).

SegFormer-B0 fine-tuned on ADE20K (150 classes), 3.8 M parameters, from
https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512. It compiles
for the Neural Engine and answers in tens of milliseconds. B2 (27 M) was tried
for better furniture labels: the Neural Engine's compiler crashes on it, so it
ran on the GPU at 0.3 to 0.9 s a frame, which starved ARKit's tracking; and
furniture is the detector's job now, which B0's big flat classes do not need.
The Core ML model takes a 512x512 RGB image and returns a 256x256 map of class
ids (ImageNet normalisation, upsampling and the per-pixel argmax are inside the
model), so the phone gets classes directly.

Needs the oasis-coreml venv (Python 3.12, torch 2.5, transformers 4.46, coremltools 8 or 9):
    ~/.venvs/oasis-coreml/bin/python apps/OasisCapture/scripts/convert_segmentation.py [--check image.jpg]
The model is not checked in; build.sh runs this when it is missing.
"""

import argparse
import json
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
from transformers import SegformerForSemanticSegmentation

MODEL_ID = "nvidia/segformer-b0-finetuned-ade-512-512"
OUT = Path(__file__).resolve().parent.parent / "Resources" / "RoomSegmentation.mlpackage"
SIZE = 512
# The decode head answers at 1/4 resolution; the logits are upsampled inside
# the model before the argmax, so the class map is as fine as the input and
# outlines are not jagged.
OUT_SIZE = 256


class Wrapped(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, image):  # image: RGB in 0...1
        logits = self.model(pixel_values=(image - self.mean) / self.std).logits
        logits = torch.nn.functional.interpolate(logits, size=(OUT_SIZE, OUT_SIZE), mode="bilinear", align_corners=False)
        return torch.argmax(logits, dim=1).to(torch.int32)


def convert() -> dict:
    model = SegformerForSemanticSegmentation.from_pretrained(MODEL_ID).eval()
    labels = {int(k): v for k, v in model.config.id2label.items()}
    wrapped = Wrapped(model).eval()
    example = torch.rand(1, 3, SIZE, SIZE)
    traced = torch.jit.trace(wrapped, example)
    mlmodel = ct.convert(
        traced,
        inputs=[ct.ImageType(name="image", shape=(1, 3, SIZE, SIZE), scale=1 / 255.0,
                             color_layout=ct.colorlayout.RGB)],
        outputs=[ct.TensorType(name="classes")],
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=ct.precision.FLOAT16,
    )
    mlmodel.short_description = f"Room segmentation ({MODEL_ID.split('/')[1]}, ADE20K): {OUT_SIZE}x{OUT_SIZE} class ids"
    mlmodel.user_defined_metadata["labels"] = json.dumps(labels)
    mlmodel.user_defined_metadata["source"] = MODEL_ID
    OUT.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(OUT))
    size = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file())
    print(f"wrote {OUT} ({size / 1e6:.1f} MB)")
    return labels


def check(image_path: Path, labels: dict) -> None:
    """Runs the saved model on a photo and lists what it found."""
    from PIL import Image

    mlmodel = ct.models.MLModel(str(OUT))
    img = Image.open(image_path).convert("RGB").resize((SIZE, SIZE))
    classes = np.asarray(mlmodel.predict({"image": img})["classes"])[0]
    ids, counts = np.unique(classes, return_counts=True)
    order = np.argsort(-counts)
    print(f"{image_path.name}:")
    for i in order[:14]:
        print(f"  {labels[int(ids[i])]:>20}  {counts[i] / classes.size:6.1%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", nargs="*", default=[])
    args = parser.parse_args()
    labels = convert()
    for path in args.check:
        check(Path(path), labels)
