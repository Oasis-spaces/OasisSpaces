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

What it searches for is a per-room Vocabulary: before this stage the agent
has Claude look at frames of the video and name what is actually in the room
(objects.json), each name with a role that tells later stages what to do with
it. Without that, a general default list is used.

The detector is GroundingDINO-tiny through transformers. It answers in free
text ("a wardrobe cabinet"), so phrases are mapped back onto the names. A
detection is a rectangle, and a rectangle around a bed also holds floor, wall
and curtain, so SAM 2.1 (hiera-tiny) then cuts each rectangle down to the
object's own outline, and only pixels inside the outline are labelled.

Standalone check (draws the outlines to <image>-outlines.png in the current
folder, so a space's frame folder is never touched):
    python3 pipeline/semantics.py <image> [more images ...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

MODEL_ID = "IDEA-Research/grounding-dino-tiny"
SEGMENTER_ID = "facebook/sam2.1-hiera-tiny"
# An outline covering less than this share of its rectangle means SAM
# found nothing there, so the whole rectangle is used instead.
MIN_OUTLINE_SHARE = 0.02

# What a named object means to the rest of the pipeline.
ROLES = {
    "furniture": "stands on the floor; its points become a box built as build_as",
    "storage": "wardrobes, cupboards, almirahs, shelves, chests of drawers: built as a "
               "wardrobe. All storage names are clustered together, because a cupboard "
               "filmed side-on comes out as a strip under one name down its edge and "
               "another along its base",
    "on_furniture": "small things that may sit on other furniture (pillow, lamp, plant): "
                    "a box, stood on the floor only when it already starts near it",
    "floor_covering": "lies flat on the floor (rug, mat): one seen up on a bed is a "
                      "mislabel, a blanket seen as a rug",
    "loose": "belongings that are not part of a model of the room (clothes, bags, "
             "laptops, towels): neither a box nor part of a wall or floor",
    "unreliable": "depth is meaningless there (mirror, window, glass, screen): the model "
                  "sees through the surface or into a reflection, so the points are dropped",
    "hanging": "hangs in front of walls and windows (curtain): as a plane it would become "
               "a duplicate wall and as leftover points a junk box, so it is neither",
    "fixture": "part of the room itself (door, pinboard, switchboard, wall art): named "
               "for context only; its points stay with the wall",
}
# Builders in the Blender furniture library (tools/blender_room.py).
BUILD_TYPES = ("bed", "seat", "table", "wardrobe", "block")
BOXED_ROLES = ("furniture", "storage", "on_furniture", "floor_covering")
# GroundingDINO reads at most 256 text tokens, and a long list dilutes it.
MAX_OBJECTS = 30

# The list used when nobody has looked at the room first (Claude unavailable,
# or clouds made before per-room lists existed; those were labelled with
# exactly these names).
DEFAULT_OBJECTS = [
    {"name": "bed", "role": "furniture", "build_as": "bed"},
    {"name": "sofa", "role": "furniture", "build_as": "seat"},
    {"name": "armchair", "role": "furniture", "build_as": "seat"},
    {"name": "chair", "role": "furniture", "build_as": "seat"},
    {"name": "stool", "role": "furniture", "build_as": "seat"},
    {"name": "table", "role": "furniture", "build_as": "table"},
    {"name": "desk", "role": "furniture", "build_as": "table"},
    {"name": "wardrobe", "role": "storage", "build_as": "wardrobe"},
    {"name": "cabinet", "role": "storage", "build_as": "wardrobe"},
    {"name": "chest of drawers", "role": "storage", "build_as": "wardrobe"},
    {"name": "shelf", "role": "storage", "build_as": "wardrobe"},
    {"name": "bookcase", "role": "storage", "build_as": "wardrobe"},
    {"name": "lamp", "role": "on_furniture", "build_as": "block"},
    {"name": "rug", "role": "floor_covering", "build_as": "block"},
    {"name": "potted plant", "role": "on_furniture", "build_as": "block"},
    {"name": "pillow", "role": "on_furniture", "build_as": "block"},
    {"name": "mirror", "role": "unreliable", "build_as": None},
    {"name": "window", "role": "unreliable", "build_as": None},
    {"name": "television", "role": "unreliable", "build_as": None},
    {"name": "computer monitor", "role": "unreliable", "build_as": None},
    {"name": "door", "role": "fixture", "build_as": None},
    {"name": "curtain", "role": "hanging", "build_as": None},
]


def clean_name(text) -> str:
    """A detector phrase that is safe everywhere it goes: lower case, no
    periods (they separate the detector's phrases) and no commas (they
    separate label names in the cloud's PLY header)."""
    words = str(text).lower().replace(",", " ").replace(".", " ").replace(";", " ").split()
    while words and words[0] in ("a", "an", "the"):
        words = words[1:]
    return " ".join(words)[:40]


class Vocabulary:
    """The names the detector searches one room for, and what each one is."""

    def __init__(self, objects: list[dict], source: str = "default"):
        cleaned, seen = [], set()
        for o in objects:
            name, role = clean_name(o.get("name", "")), o.get("role")
            if not name or name in seen or role not in ROLES:
                continue
            build = o.get("build_as")
            if role == "storage" or (role == "furniture" and build == "wardrobe"):
                role, build = "storage", "wardrobe"
            elif role in BOXED_ROLES:
                build = build if build in BUILD_TYPES and build != "wardrobe" else "block"
                if role != "furniture":
                    build = "block"
            else:
                build = None
            seen.add(name)
            cleaned.append({"name": name, "role": role, "build_as": build})
        self.objects = cleaned[:MAX_OBJECTS]
        self.source = source
        self._by_name = {o["name"]: o for o in self.objects}

    @classmethod
    def default(cls) -> "Vocabulary":
        return cls(DEFAULT_OBJECTS, "default")

    @classmethod
    def from_json(cls, data) -> "Vocabulary":
        if isinstance(data, dict) and data.get("objects"):
            return cls(data["objects"], data.get("source", "file"))
        return cls.default()

    def to_json(self) -> dict:
        return {"source": self.source, "objects": self.objects}

    @property
    def names(self) -> list[str]:
        return [o["name"] for o in self.objects]

    def with_role(self, *roles: str) -> list[str]:
        return [o["name"] for o in self.objects if o["role"] in roles]

    @property
    def furniture(self) -> list[str]:
        """Names whose points become boxes."""
        return self.with_role(*BOXED_ROLES)

    @property
    def storage(self) -> list[str]:
        return self.with_role("storage")

    @property
    def unreliable(self) -> list[str]:
        return self.with_role("unreliable")

    @property
    def hanging(self) -> list[str]:
        return self.with_role("hanging")

    @property
    def left_out(self) -> list[str]:
        """Names whose points are neither fitted as planes nor boxed."""
        return self.with_role("hanging", "loose")

    def role(self, name: str | None) -> str | None:
        o = self._by_name.get(name)
        return o["role"] if o else None

    def build_as(self, name: str | None) -> str:
        """The library builder for a detected name; a name from an older list
        that is itself a builder (a "wardrobe" front) keeps it."""
        o = self._by_name.get(name)
        if o and o["build_as"]:
            return o["build_as"]
        return name if name in BUILD_TYPES else "block"

    def same_kind(self, a: str | None, b: str | None) -> bool:
        """Two names for the same object: equal, or both storage (a cupboard
        is seen as a wardrobe from one side and a shelf from another)."""
        return a == b or (self.build_as(a) == "wardrobe" == self.build_as(b))

    def canonical(self, phrase: str) -> str | None:
        """Map the detector's free text back onto a name. The detector answers
        with the prompt words it matched, sometimes only part of a phrase
        ("drawers" for "chest of drawers"), so when no whole name appears in the
        text, a name sharing a distinctive word with it is used, if only one does."""
        text = clean_name(phrase)
        for term in sorted(self.names, key=len, reverse=True):
            if term in text:
                return term
        words = set(text.split())
        sharing = [term for term in self.names
                   if any(len(w) >= 4 and w in words for w in term.split())]
        return sharing[0] if len(sharing) == 1 else None

    @property
    def prompt(self) -> str:
        return ". ".join(f"{'an' if term[0] in 'aeiou' else 'a'} {term}"
                         for term in self.names) + "."


def room_vocabulary(space: Path) -> Vocabulary:
    """The list a space's cloud was labelled with: recorded in densify.json by
    densify.py; the default list for clouds made before that."""
    try:
        meta = json.loads((Path(space) / "densify.json").read_text())
    except (OSError, ValueError):
        meta = {}
    return Vocabulary.from_json(meta.get("objects"))


def planned_vocabulary(space: Path) -> Vocabulary:
    """The list densify.py should search for: the room's own objects.json (the
    agent writes it from Claude's look at the video) plus the default
    unreliable surfaces, or the default list when there is no objects.json.

    Mirrors, windows and screens are always searched for: their depth is wrong
    whatever the room holds, and missing one leaves a wall of fake points."""
    path = Path(space) / "objects.json"
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return Vocabulary.default()
    vocabulary = Vocabulary.from_json(data)
    if vocabulary.source == "default":
        return vocabulary
    extra = [o for o in DEFAULT_OBJECTS if o["role"] == "unreliable"
             and not any(o["name"] in n or n in o["name"] for n in vocabulary.names)]
    return Vocabulary(vocabulary.objects[:MAX_OBJECTS - len(extra)] + extra,
                      vocabulary.source)


def default_device() -> str:
    """The GPU this machine has: CUDA (Colab), Apple's MPS (Mac), or CPU."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


class Detector:
    """GroundingDINO-tiny, loaded once and run per keyframe."""

    def __init__(self, device: str | None = None, work_size: int = 1024,
                 box_threshold: float = 0.3, text_threshold: float = 0.25,
                 vocabulary: Vocabulary | None = None):
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.torch = torch
        self.device = device or default_device()
        self.processor = AutoProcessor.from_pretrained(MODEL_ID)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            MODEL_ID).to(self.device).eval()
        self.vocabulary = vocabulary or Vocabulary.default()
        self.prompt = self.vocabulary.prompt
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
            label = self.vocabulary.canonical(str(phrase))
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
        self.device = device or default_device()
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
