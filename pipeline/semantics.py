#!/usr/bin/env python3
"""Open-vocabulary object detection on keyframes, locally on the Mac's GPU.

densify.py uses this to label the dense cloud. Every dense point knows the
frame and the pixel it came from, so a detection box on that frame labels
those points directly, with no reprojection needed. Points on surfaces whose
monocular depth cannot be trusted (mirrors, windows, screens: the model
either sees through them or into a reflection) are dropped instead of
labelled, which is what the glossy wardrobe and the blown-out window in the
test captures need.

shapes.py then builds furniture boxes from the labelled points, instead of
inferring furniture from whatever the plane fitter left over.

The detector is GroundingDINO-tiny through transformers. It answers in free
text ("a wardrobe cabinet"), so phrases are mapped back onto VOCABULARY.

Standalone check:
    python3 pipeline/semantics.py <image> [more images ...]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

MODEL_ID = "IDEA-Research/grounding-dino-tiny"

# Worth building as furniture in the Blender room.
FURNITURE = [
    "bed", "sofa", "armchair", "chair", "stool", "table", "desk", "wardrobe",
    "cabinet", "chest of drawers", "shelf", "bookcase", "lamp", "rug",
    "potted plant", "pillow",
]
# Depth here is meaningless: the model sees through the surface or into a
# reflection, so these points are dropped from the cloud.
UNRELIABLE = ["mirror", "window", "television", "computer monitor"]
# Named for context; not built as furniture.
STRUCTURE = ["door", "curtain"]
# Hangs in front of walls and windows: as a plane it becomes a duplicate wall,
# and as leftover points a junk box, so shapes.py leaves it out of both.
HANGING = ["curtain"]
VOCABULARY = FURNITURE + UNRELIABLE + STRUCTURE


def canonical(phrase: str) -> str | None:
    """Map the detector's free text back onto VOCABULARY."""
    text = phrase.lower()
    for term in sorted(VOCABULARY, key=len, reverse=True):
        if term in text:
            return term
    return None


class Detector:
    """GroundingDINO-tiny, loaded once and run per keyframe."""

    def __init__(self, device: str | None = None, work_size: int = 1024,
                 box_threshold: float = 0.3, text_threshold: float = 0.25):
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.torch = torch
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(MODEL_ID)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            MODEL_ID).to(self.device).eval()
        self.prompt = ". ".join(f"a {term}" for term in VOCABULARY) + "."
        self.work_size = work_size
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

    def detect(self, img) -> list[dict]:
        """Detections as {label, score, box}, boxes in the image's own pixels."""
        from PIL import Image

        full = img.convert("RGB")
        scale = min(1.0, self.work_size / max(full.size))
        small = full.resize((round(full.width * scale), round(full.height * scale)),
                            Image.BILINEAR) if scale < 1.0 else full
        inputs = self.processor(images=small, text=self.prompt,
                                return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            outputs = self.model(**inputs)
        result = self.processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[(small.height, small.width)],
        )[0]
        phrases = result.get("text_labels", result.get("labels"))
        found = []
        for phrase, score, box in zip(phrases, result["scores"], result["boxes"]):
            label = canonical(str(phrase))
            if label is None:
                continue
            x0, y0, x1, y1 = (float(v) / scale for v in box)
            found.append({"label": label, "score": round(float(score), 3),
                          "box": [x0, y0, x1, y1]})
        return found


def pixel_labels(us: np.ndarray, vs: np.ndarray, detections: list[dict],
                 index: dict[str, int]) -> np.ndarray:
    """Label index per pixel: the highest-scoring detection covering it."""
    labels = np.zeros(len(us), dtype=np.uint8)
    for det in sorted(detections, key=lambda d: d["score"]):
        x0, y0, x1, y1 = det["box"]
        inside = (us >= x0) & (us <= x1) & (vs >= y0) & (vs <= y1)
        labels[inside] = index[det["label"]]
    return labels


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    detector = Detector()
    print(f"{MODEL_ID} on {detector.device}")
    from PIL import Image
    for path in sys.argv[1:]:
        detections = detector.detect(Image.open(path))
        summary = ", ".join(f"{d['label']} {d['score']:.2f}" for d in detections)
        print(f"{Path(path).name}: {summary or 'nothing detected'}")


if __name__ == "__main__":
    main()
