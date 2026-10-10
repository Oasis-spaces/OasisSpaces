#!/usr/bin/env python3
"""A second opinion on each tracked object's name: CLIP reads its crops.

The detector's votes all come from one model, so its systematic confusions
survive the election — the pan's door was named "wardrobe" two frames in
three, and tracking faithfully consolidated the mistake. CLIP judges each
object's crops independently, against the room's own vocabulary, and its
vote joins the track's at a bounded weight: strong enough to flip a near-tie
like that door, never enough to overturn a detector that is sure. An unsure
CLIP (a thin margin between its top two names) abstains entirely — one weak
opinion is not a verdict.

    python3 tools/crop_vote.py spaces/<space>
    python3 tools/crop_vote.py spaces/<space> --boxes boxes.json

The first form votes on the space's tracks (workspace/tracks/, written by
densify with tracking on) and writes crop-votes.json in the space. The second
scores hand-marked boxes instead — {"frame_00001.jpg": [[x0, y0, x1, y1],
...], ...} in frame pixels — for trying the scorer without a tracked space.
Weights (openai/clip-vit-base-patch32) download from Hugging Face on first
use, like the detector's.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
from semantics import default_device, room_vocabulary  # noqa: E402

CLIP_CHECKPOINT = "openai/clip-vit-base-patch32"
PROMPT = "a photo of a {} in a room"
CLIP_WEIGHT = 0.5         # CLIP's whole say, as a share of the detector's total votes
MIN_MARGIN = 0.15         # closer than this between its top two names, CLIP abstains
CROPS_PER_TRACK = 5       # the track's clearest sightings
CROP_CONTEXT = 0.12       # each box widened by this share of its size: a little room around it


class Scorer:
    """CLIP over the room's names; weights load when the scorer is built."""

    def __init__(self, names: list[str]):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.names = list(names)
        self.device = default_device()
        self.processor = CLIPProcessor.from_pretrained(CLIP_CHECKPOINT)
        self.model = CLIPModel.from_pretrained(CLIP_CHECKPOINT).to(self.device).eval()
        self.torch = torch

    def scores(self, crops: list[Image.Image]) -> np.ndarray:
        """Each crop's probability over the names: rows sum to one."""
        texts = [PROMPT.format(name) for name in self.names]
        inputs = self.processor(text=texts, images=crops, return_tensors="pt", padding=True)
        with self.torch.no_grad():
            logits = self.model(**inputs.to(self.device)).logits_per_image
        return logits.softmax(dim=1).cpu().numpy()


def crop(image: Image.Image, box: list[float], context: float = CROP_CONTEXT) -> Image.Image:
    """The box widened a little, clamped to the frame, never thinner than a
    pixel — a hand-marked box may be inverted or sit outside the frame."""
    x0, x1 = sorted((box[0], box[2]))
    y0, y1 = sorted((box[1], box[3]))
    dx, dy = (x1 - x0) * context, (y1 - y0) * context
    left = min(max(0, int(x0 - dx)), image.width - 1)
    top = min(max(0, int(y0 - dy)), image.height - 1)
    return image.crop((left, top, max(min(image.width, int(x1 + dx)), left + 1),
                       max(min(image.height, int(y1 + dy)), top + 1)))


def clearest_sightings(track: dict, images_dir: Path,
                       count: int = CROPS_PER_TRACK) -> list[tuple[str, list[float]]]:
    """(frame, box) of the track's highest-scoring sightings still on disk."""
    frames = sorted(track["frames"].items(), key=lambda kv: -kv[1]["score"])
    kept = [(name, sighting["box"]) for name, sighting in frames
            if (Path(images_dir) / name).exists()]
    return kept[:count]


def clip_vote(scores: np.ndarray, names: list[str]) -> dict[str, float] | None:
    """The mean of the crops' probabilities, or None when CLIP is unsure."""
    if scores.size == 0:
        return None
    mean = scores.mean(axis=0)
    top = np.sort(mean)[::-1]
    if len(top) > 1 and float(top[0] - top[1]) < MIN_MARGIN:
        return None
    return {name: round(float(p), 4) for name, p in zip(names, mean)}


def merged_votes(votes: dict[str, float], opinion: dict[str, float] | None) -> dict[str, float]:
    """The detector's votes plus CLIP's, scaled so CLIP holds CLIP_WEIGHT of
    the detector's total mass — it can flip a near-tie, not a sure thing."""
    merged = dict(votes)
    if opinion and votes:
        say = CLIP_WEIGHT * sum(votes.values())
        for name, p in opinion.items():
            merged[name] = round(merged.get(name, 0.0) + say * p, 3)
    return merged


def winner(votes: dict[str, float]) -> str | None:
    return max(votes, key=votes.get) if votes else None


def second_opinion(tracks: list[dict], images_dir: Path, names: list[str], scorer) -> list[dict]:
    """Every track's crops judged; the merged election beside the detector's."""
    rows = []
    for track in tracks:
        crops = [crop(Image.open(Path(images_dir) / frame).convert("RGB"), box)
                 for frame, box in clearest_sightings(track, images_dir)]
        if not crops:
            continue
        opinion = clip_vote(scorer.scores(crops), names)
        merged = merged_votes(track["votes"], opinion)
        rows.append({"id": track["id"], "label": track["label"], "crops": len(crops),
                     "clip": ({winner(opinion): opinion[winner(opinion)]} if opinion else None),
                     "merged": merged, "elected": winner(merged),
                     "changed": winner(merged) != winner(track["votes"])})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("space", help="the space folder (spaces/<name>)")
    parser.add_argument("--boxes", help="JSON of {frame: [[x0,y0,x1,y1], ...]} to score instead of tracks")
    args = parser.parse_args()
    space = Path(args.space)
    images_dir = space / "workspace" / "images"
    names = room_vocabulary(space).names

    if args.boxes:
        marked = json.loads(Path(args.boxes).read_text())
        scorer = Scorer(names)
        for frame, boxes in marked.items():
            image = Image.open(images_dir / frame).convert("RGB")
            scores = scorer.scores([crop(image, box) for box in boxes])
            for box, row in zip(boxes, scores):
                best = np.argsort(row)[::-1][:3]
                print(f"{frame} {[round(v) for v in box]}: "
                      + ", ".join(f"{names[i]} {row[i]:.2f}" for i in best))
        return 0

    from tracking import Tracks
    tracks = Tracks.load(space / "workspace" / "tracks")
    if tracks is None:
        sys.exit(f"{space} has no workspace/tracks (densify with tracking writes it)")
    rows = second_opinion(tracks.tracks, images_dir, names, Scorer(names))
    for row in rows:
        note = f" -> {row['elected']}" if row["changed"] else ""
        print(f"  {row['id']:>3} {row['label']:<18} clip: "
              + (f"{row['clip']}" if row["clip"] else "abstained") + note)
    changed = sum(1 for r in rows if r["changed"])
    (space / "crop-votes.json").write_text(json.dumps(
        {"checkpoint": CLIP_CHECKPOINT, "weight": CLIP_WEIGHT, "votes": rows}, indent=1) + "\n")
    print(f"{len(rows)} object(s), {changed} election(s) changed -> {space / 'crop-votes.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
