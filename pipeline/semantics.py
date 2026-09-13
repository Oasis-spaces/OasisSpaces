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
text ("a wardrobe cabinet"), so phrases are mapped back onto VOCABULARY. A
detection is a rectangle, and a rectangle around a bed also holds floor, wall
and curtain, so SAM 2.1 (hiera-tiny) then cuts each rectangle down to the
object's own outline, and only pixels inside the outline are labelled.

Standalone check (draws the outlines to <image>-outlines.png in the current
folder, so a space's frame folder is never touched):
    python3 pipeline/semantics.py <image> [more images ...]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

MODEL_ID = "IDEA-Research/grounding-dino-tiny"
SEGMENTER_ID = "facebook/sam2.1-hiera-tiny"
# An outline covering less than this share of its rectangle means SAM
# found nothing there, so the whole rectangle is used instead.
MIN_OUTLINE_SHARE = 0.02

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


class Segmenter:
    """SAM 2.1 hiera-tiny, prompted with the detector's rectangles.

    Preprocessing is done here (resize to 1024x1024, ImageNet normalisation,
    masks upsampled from 256x256) rather than by transformers' Sam2Processor,
    which needs torchvision; torch here is Homebrew's build, which OpenSplat is
    linked against, so it is not swapped for a pip one."""

    INPUT = 1024
    MEAN = np.array([0.485, 0.456, 0.406], np.float32)
    STD = np.array([0.229, 0.224, 0.225], np.float32)

    def __init__(self, device: str | None = None, work_size: int = 1024):
        import torch
        from transformers import Sam2Model

        self.torch = torch
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = Sam2Model.from_pretrained(SEGMENTER_ID).to(self.device).eval()
        self.work_size = work_size

    def outline(self, img, detections: list[dict]) -> int:
        """Give each detection a "mask" (booleans at the working size, clipped
        to its rectangle) and the "mask_scale" from image pixels to it.
        Returns how many detections got an outline."""
        from PIL import Image

        torch = self.torch
        if not detections:
            return 0
        full = img.convert("RGB")
        W, H = full.size
        x = np.asarray(full.resize((self.INPUT, self.INPUT), Image.BILINEAR), np.float32)
        x = (x / 255.0 - self.MEAN) / self.STD
        pixels = torch.from_numpy(x.transpose(2, 0, 1).copy())[None].to(self.device)
        sx, sy = self.INPUT / W, self.INPUT / H
        boxes = torch.tensor([[[d["box"][0] * sx, d["box"][1] * sy,
                                d["box"][2] * sx, d["box"][3] * sy] for d in detections]],
                             dtype=torch.float32, device=self.device)
        with torch.no_grad():
            out = self.model(pixel_values=pixels, input_boxes=boxes, multimask_output=False)
        scale = min(1.0, self.work_size / max(W, H))
        size = (round(H * scale), round(W * scale))
        low = out.pred_masks[0, :, :1].float().cpu()  # boxes x 1 x 256 x 256 logits
        masks = (torch.nn.functional.interpolate(low, size, mode="bilinear",
                                                 align_corners=False)[:, 0] > 0).numpy()
        outlined = 0
        for det, mask in zip(detections, masks):
            x0, y0, x1, y1 = (int(round(v * scale)) for v in det["box"])
            x0, y0 = max(x0, 0), max(y0, 0)
            clipped = np.zeros_like(mask)
            clipped[y0:y1 + 1, x0:x1 + 1] = mask[y0:y1 + 1, x0:x1 + 1]
            box_area = max((x1 - x0 + 1) * (y1 - y0 + 1), 1)
            if clipped.sum() >= MIN_OUTLINE_SHARE * box_area:
                det["mask"], det["mask_scale"] = clipped, scale
                outlined += 1
        return outlined


def pixel_labels(us: np.ndarray, vs: np.ndarray, detections: list[dict],
                 index: dict[str, int]) -> np.ndarray:
    """Label index per pixel: the highest-scoring detection covering it, by
    its outline when Segmenter gave it one, else by its rectangle."""
    labels = np.zeros(len(us), dtype=np.uint8)
    for det in sorted(detections, key=lambda d: d["score"]):
        x0, y0, x1, y1 = det["box"]
        inside = (us >= x0) & (us <= x1) & (vs >= y0) & (vs <= y1)
        mask = det.get("mask")
        if mask is not None:
            at = np.flatnonzero(inside)
            k = det["mask_scale"]
            rows = np.minimum((vs[at] * k).astype(int), mask.shape[0] - 1)
            cols = np.minimum((us[at] * k).astype(int), mask.shape[1] - 1)
            inside = at[mask[rows, cols]]
        labels[inside] = index[det["label"]]
    return labels


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    from PIL import Image, ImageDraw

    detector = Detector()
    print(f"{MODEL_ID} on {detector.device}")
    found = {path: detector.detect(Image.open(path)) for path in sys.argv[1:]}
    del detector
    segmenter = Segmenter()
    print(f"{SEGMENTER_ID} on {segmenter.device}")
    for path, detections in found.items():
        img = Image.open(path).convert("RGB")
        outlined = segmenter.outline(img, detections)
        summary = ", ".join(f"{d['label']} {d['score']:.2f}" for d in detections)
        print(f"{Path(path).name}: {summary or 'nothing detected'} "
              f"({outlined} outlined)")
        # Each outline tinted over the photo, with its rectangle drawn round it.
        overlay = np.asarray(img).astype(np.float32)
        palette = [(230, 60, 60), (60, 180, 75), (60, 110, 230), (240, 170, 40),
                   (170, 70, 220), (40, 190, 200)]
        draw_boxes = []
        for i, det in enumerate(sorted(detections, key=lambda d: d["score"])):
            colour = np.array(palette[i % len(palette)], np.float32)
            if "mask" in det:
                big = np.asarray(Image.fromarray(det["mask"]).resize(img.size))
                overlay[big] = overlay[big] * 0.45 + colour * 0.55
            draw_boxes.append((det, tuple(int(c) for c in colour)))
        out = Image.fromarray(overlay.astype(np.uint8))
        draw = ImageDraw.Draw(out)
        for det, colour in draw_boxes:
            draw.rectangle(det["box"], outline=colour, width=3)
            draw.text((det["box"][0] + 6, det["box"][1] + 4), det["label"], fill=colour)
        target = Path.cwd() / (Path(path).stem + "-outlines.png")
        out.save(target)
        print(f"  -> {target}")


if __name__ == "__main__":
    main()
