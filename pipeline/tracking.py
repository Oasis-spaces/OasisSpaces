#!/usr/bin/env python3
"""Each detected object carried through every frame of the video.

GroundingDINO names what it sees on a few spaced frames, and names it a
little differently each time: a wardrobe is a "hanging cloth" in one frame,
a door is a "wardrobe" in another, and a thing seen in three keyframes of
twelve is unknown in the other nine. SAM 2.1's video model carries each
detected thing through all the frames in between, so one object keeps one
identity over the whole video, takes the name the detector gave it most
often, and has an outline in every frame it shows in, not only the keyframes.

How it runs:
  1. The detector runs on the prompt frames: the densify keyframes plus every
     Nth frame, about PROMPT_FRAMES in all.
  2. Forward pass, in time order. At each prompt frame a detection that lands
     on an object already tracked there (rectangle IoU >= MATCH_IOU with the
     tracked outline's rectangle) votes for that object's name and refreshes
     its outline; one that lands on nothing starts a new object. Between
     prompt frames every object is propagated.
  3. Backward pass: each object is seeded with its first outline and carried
     back to the start of the video, so it is known in the frames before the
     detector first named it too.
  4. Objects tracing the same thing (outline IoU >= DUPLICATE_IOU over the
     frames both show in) are merged, their votes added.

Output, in <space>/workspace/tracks/: tracks.json (each object's name, the
votes behind it, and its rectangle and score per frame) and masks.npz (its
outline per frame, 256x256 bits). densify.py labels its points with these
instead of per-keyframe detections, and placement.py scores boxes against
these outlines in every frame.

The model runs object by object, which is fast on CUDA and slow on this Mac's
GPU (about a second per object and frame), so densify runs it by default
only on CUDA (--track on forces it elsewhere).

SAM 3 (facebook/sam3, gated: an access request on Hugging Face and a login)
does all of this in one model: given the room's names as text, it finds every
object of each name in every frame and keeps one identity per object through
the video (Promptable Concept Segmentation), where GroundingDINO only names
~20 frames and is the source of the systematic mislabels (a door called a
wardrobe two times in three). Sam3Tracker gives the same Tracks: one object,
the name its frames voted for, an outline per frame. densify.py --tracker
auto uses it when its weights can be loaded, and SAM 2.1 otherwise.

Usage:
    python3 pipeline/tracking.py spaces/<name> [--sheet out.png] [--update-densify]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from semantics import SEGMENTER_ID, default_device  # noqa: E402

MASK_SIZE = 256           # the model's own mask raster, kept as the stored one
SAM3_ID = "facebook/sam3"   # gated on Hugging Face: request access, then log in
SAM3_WORK = 1008          # SAM 3's own input size: frames go to it at this longest side
# The names worth carrying through the video: the pieces, the surfaces and the
# openings a room model is made of. Clothes, bags and brushes are labelled per
# keyframe as before; tracking costs frames x objects, and a walkthrough with
# forty names did not finish inside a Colab session.
TRACKED_ROLES = ("furniture", "storage", "fixture", "unreliable", "hanging", "floor_covering")
HALF_ON_CUDA = True       # float16 on CUDA: half the memory traffic of the per-object loop
# A tracked object the detector named a door or window this share of the time
# is partly one, whatever name won the vote: the pan's door was "wardrobe 3.7,
# door 1.9". Its outlines are evidence for the fixture, not for a piece.
FIXTURES = ("door", "window")
FIXTURE_VOTE_SHARE = 0.25
MIN_FIXTURE_VOTES = 0.6   # ... with at least this much vote behind it: one weak "door" (0.3) on a
                          # cupboard seen eight times is not a verdict
MATCH_IOU = 0.5           # a detection lands on a tracked object
DUPLICATE_IOU = 0.7       # two objects trace the same thing
DUPLICATE_FRAMES = 3      # ... judged over at least this many shared frames
MIN_AREA_SHARE = 0.0005   # an outline smaller than this share of the frame is noise
PROMPT_FRAMES = 20        # about how many frames the detector runs on
MEMORY_FRAMES = 16        # SAM 2 looks back at most this far: older outputs are dropped
INPUT = 1024              # the model's input size
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


# ------------------------------------------------------------------ the result
def box_iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def mask_box(mask: np.ndarray, width: int, height: int) -> list[float] | None:
    """The outline's rectangle in frame pixels, or None for an empty outline."""
    rows, cols = np.flatnonzero(mask.any(axis=1)), np.flatnonzero(mask.any(axis=0))
    if not len(rows):
        return None
    sx, sy = width / mask.shape[1], height / mask.shape[0]
    return [float(cols[0] * sx), float(rows[0] * sy), float((cols[-1] + 1) * sx), float((rows[-1] + 1) * sy)]


def tracked_names(vocabulary) -> set[str]:
    """The vocabulary's names in TRACKED_ROLES."""
    return set(vocabulary.with_role(*TRACKED_ROLES))


def split_prompts(prompts: dict[str, list[dict]], tracked: set[str]) -> tuple[dict, dict]:
    """The detections to track and the rest, each as {frame: detections}."""
    to_track = {name: [d for d in dets if d["label"] in tracked] for name, dets in prompts.items()}
    rest = {name: [d for d in dets if d["label"] not in tracked] for name, dets in prompts.items()}
    return {n: d for n, d in to_track.items() if d}, {n: d for n, d in rest.items() if d}


def merge_detections(tracked: dict[str, list[dict]], rest: dict[str, list[dict]]) -> dict[str, list[dict]]:
    """densify.json's "detections": the tracked objects in every frame they
    show in, plus the untracked detections of the prompt frames."""
    merged = {name: list(dets) for name, dets in tracked.items()}
    for name, dets in rest.items():
        merged.setdefault(name, []).extend(
            {"label": d["label"], "score": d["score"], "box": [round(v) for v in d["box"]]} for d in dets)
    return merged


def prompt_frames(frames: list[str], keyframes: list[str], count: int = PROMPT_FRAMES) -> list[str]:
    """The keyframes plus every Nth frame, about `count` in all, in time order."""
    step = max(1, len(frames) // count)
    chosen = set(keyframes) & set(frames)
    chosen.update(frames[::step])
    return [f for f in frames if f in chosen]


class Tracks:
    """What the tracker found: objects with a name each and an outline per frame."""

    def __init__(self, frames: list[str], size: tuple[int, int], tracks: list[dict],
                 masks: dict[str, np.ndarray], model: str = SEGMENTER_ID):
        self.frames = frames                    # every frame name, in time order
        self.width, self.height = size
        self.tracks = tracks                    # [{id, label, votes, seed, frames: {name: {box, score}}}]
        self._masks = masks                     # "t<id>/<frame>" -> packed MASK_SIZE x MASK_SIZE bits
        self.model = model                      # the video model that made them

    @staticmethod
    def key(track_id: int, frame: str) -> str:
        return f"t{track_id}/{frame}"

    def mask(self, track_id: int, frame: str, shape: tuple[int, int] | None = None) -> np.ndarray | None:
        """The object's outline in that frame as booleans, at MASK_SIZE square
        or resized to `shape` (rows, cols); None when it is not in the frame."""
        packed = self._masks.get(self.key(track_id, frame))
        if packed is None:
            return None
        mask = np.unpackbits(packed)[:MASK_SIZE * MASK_SIZE].reshape(MASK_SIZE, MASK_SIZE).astype(bool)
        if shape is not None and tuple(shape) != mask.shape:
            from PIL import Image

            mask = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255)
                              .resize((shape[1], shape[0]), Image.NEAREST)) > 0
        return mask

    def detections(self, frame: str, work_size: int | None = None) -> list[dict]:
        """The objects in this frame as detections, {label, score, box, track},
        largest first so a small thing on a big one wins the pixel. With
        `work_size`, each also carries its "mask" at that longest side and the
        "mask_scale" from frame pixels to it, as Segmenter.outline gives."""
        found = []
        for track in self.tracks:
            seen = track["frames"].get(frame)
            if seen is None:
                continue
            det = {"label": track["label"], "score": seen["score"], "box": list(seen["box"]),
                   "track": track["id"]}
            if work_size is not None:
                scale = min(1.0, work_size / max(self.width, self.height))
                shape = (round(self.height * scale), round(self.width * scale))
                det["mask"], det["mask_scale"] = self.mask(track["id"], frame, shape), scale
            found.append(det)
        found.sort(key=lambda d: -(d["box"][2] - d["box"][0]) * (d["box"][3] - d["box"][1]))
        return found

    @staticmethod
    def fixture_share(track: dict, fixtures: set[str]) -> float:
        """The share of the detector's votes for this object that named a fixture."""
        total = sum(track["votes"].values())
        return sum(v for k, v in track["votes"].items() if k in fixtures) / total if total else 0.0

    def doorish(self, fixtures: set[str] | None = None) -> set[int]:
        """The ids of objects named a fixture at least FIXTURE_VOTE_SHARE of the
        time but not called one: a door that won as "wardrobe"."""
        fixtures = set(fixtures or ()) | set(FIXTURES)
        return {t["id"] for t in self.tracks
                if t["label"] not in fixtures and self.fixture_share(t, fixtures) >= FIXTURE_VOTE_SHARE
                and sum(v for k, v in t["votes"].items() if k in fixtures) >= MIN_FIXTURE_VOTES}

    def all_detections(self) -> dict[str, list[dict]]:
        """{frame: detections} for every frame something shows in (densify.json's "detections")."""
        out = {}
        for frame in self.frames:
            dets = self.detections(frame)
            if dets:
                out[frame] = [{**d, "box": [round(v) for v in d["box"]]} for d in dets]
        return out

    def summary(self) -> dict:
        labels: dict[str, int] = {}
        for t in self.tracks:
            labels[t["label"]] = labels.get(t["label"], 0) + 1
        return {"objects": len(self.tracks), "labels": labels, "frames": len(self.frames),
                "frames_with_objects": sum(1 for f in self.frames if any(f in t["frames"] for t in self.tracks)),
                "model": self.model + " (video)"}

    def save(self, folder: Path) -> None:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        record = {"frames": self.frames, "width": self.width, "height": self.height,
                  "mask_size": MASK_SIZE, "model": self.model,
                  "tracks": [{**t, "frames": {n: {"box": [round(v, 1) for v in s["box"]], "score": s["score"]}
                                              for n, s in t["frames"].items()}} for t in self.tracks]}
        (folder / "tracks.json").write_text(json.dumps(record, indent=1) + "\n")
        np.savez_compressed(folder / "masks.npz", **self._masks)

    @classmethod
    def load(cls, folder: Path) -> "Tracks | None":
        folder = Path(folder)
        if not (folder / "tracks.json").exists() or not (folder / "masks.npz").exists():
            return None
        record = json.loads((folder / "tracks.json").read_text())
        with np.load(folder / "masks.npz") as stored:
            masks = {k: stored[k] for k in stored.files}
        return cls(record["frames"], (record["width"], record["height"]), record["tracks"], masks,
                   record.get("model", SEGMENTER_ID))


# ------------------------------------------------------------------ the tracker
class Tracker:
    """SAM 2.1 hiera-tiny's video model, driven frame by frame without
    transformers' processor (which needs torchvision; see Segmenter)."""

    def __init__(self, device: str | None = None):
        import torch
        from transformers import Sam2VideoModel

        self.torch = torch
        self.device = device or default_device()
        self.dtype = torch.float16 if (self.device == "cuda" and HALF_ON_CUDA) else torch.float32
        self.model = Sam2VideoModel.from_pretrained(SEGMENTER_ID, dtype=self.dtype).to(self.device).eval()

    def pixels(self, path: Path):
        from PIL import Image

        img = Image.open(path).convert("RGB")
        x = np.asarray(img.resize((INPUT, INPUT), Image.BILINEAR), np.float32)
        return self.torch.from_numpy(((x / 255.0 - MEAN) / STD).transpose(2, 0, 1).copy())

    def session(self, width: int, height: int):
        from transformers.models.sam2_video.modeling_sam2_video import Sam2VideoInferenceSession

        return Sam2VideoInferenceSession(video=None, video_height=height, video_width=width,
                                         inference_device=self.device, inference_state_device=self.device,
                                         video_storage_device="cpu", dtype=self.dtype)

    def run(self, image_dir: Path, frames: list[str], prompts: dict[str, list[dict]], log=print) -> Tracks:
        """Track everything in `prompts` ({frame: detections}) through `frames`."""
        from PIL import Image

        torch = self.torch
        image_dir = Path(image_dir)
        index = {name: i for i, name in enumerate(frames)}
        prompt_at = {index[n]: d for n, d in prompts.items() if n in index and d}
        width, height = Image.open(image_dir / frames[0]).size
        objects: dict[int, dict] = {}      # id -> {votes, seed, frames: {idx: (box, score)}}
        masks: dict[str, np.ndarray] = {}
        if not prompt_at:
            return Tracks(frames, (width, height), [], masks)

        def step(session, px, f, reverse):
            """Run the frame; record every present object's outline. Returns {id: mask}."""
            out = self.model(session, frame=px, frame_idx=f, reverse=reverse)
            present = {}
            pred = (out.pred_masks[:, 0] > 0).cpu().numpy()
            logits = out.object_score_logits.flatten().float().cpu().numpy()
            for obj_id, mask, logit in zip(out.object_ids, pred, logits):
                box = mask_box(mask, width, height)
                if logit <= 0 or box is None or mask.sum() < MIN_AREA_SHARE * mask.size:
                    continue
                present[obj_id] = mask
                if reverse and f >= objects[obj_id]["seed"]:
                    continue                                  # the forward pass has this frame
                objects[obj_id]["frames"][f] = (box, round(float(1 / (1 + np.exp(-logit))), 3))
                masks[Tracks.key(obj_id, frames[f])] = np.packbits(mask)
            session.processed_frames.pop(f, None)
            # The model needs the last MEMORY_FRAMES outputs and the prompted
            # frames; everything else (1 MB an object and frame) goes.
            for store in session.output_dict_per_obj.values():
                old = [k for k in store["non_cond_frame_outputs"]
                       if (k > f + MEMORY_FRAMES if reverse else k < f - MEMORY_FRAMES)]
                for k in old:
                    del store["non_cond_frame_outputs"][k]
                for kind in ("cond_frame_outputs", "non_cond_frame_outputs"):
                    for o in store[kind].values():
                        o.pop("high_res_masks", None)
            return present

        def add_box(session, obj_id, f, box):
            x0, y0, x1, y1 = box
            coords = torch.tensor([[[[x0 * INPUT / width, y0 * INPUT / height],
                                     [x1 * INPUT / width, y1 * INPUT / height]]]], dtype=self.dtype)
            labels = torch.tensor([[[2, 3]]], dtype=torch.int32)        # SAM's box corners
            session.add_point_inputs(session.obj_id_to_idx(obj_id), f,
                                     {"point_coords": coords, "point_labels": labels})

        # Forward: from the first prompt frame to the end.
        first = min(prompt_at)
        session = self.session(width, height)
        started = time.time()
        next_id = 0
        with torch.no_grad():
            for f in range(first, len(frames)):
                px = self.pixels(image_dir / frames[f])
                present = step(session, px, f, False) if session.get_obj_num() else {}
                if f in prompt_at:
                    matched, started_here, prompted = 0, 0, []
                    for det in sorted(prompt_at[f], key=lambda d: -d["score"]):
                        best, best_iou = None, 0.0
                        for obj_id, mask in present.items():
                            v = box_iou(det["box"], mask_box(mask, width, height))
                            if v > best_iou:
                                best, best_iou = obj_id, v
                        if best is not None and best_iou >= MATCH_IOU:
                            votes = objects[best]["votes"]
                            votes[det["label"]] = round(votes.get(det["label"], 0.0) + det["score"], 3)
                            obj_id, matched = best, matched + 1
                        else:
                            obj_id, next_id = next_id, next_id + 1
                            objects[obj_id] = {"votes": {det["label"]: det["score"]}, "seed": f, "frames": {}}
                            started_here += 1
                        if obj_id not in prompted:              # the strongest detection prompts it
                            add_box(session, obj_id, f, det["box"])
                            prompted.append(obj_id)
                    if prompted:
                        session.obj_with_new_inputs = prompted
                        step(session, px, f, False)
                    log(f"  {frames[f]}: {len(prompt_at[f])} detections, {matched} on tracked objects, "
                        f"{started_here} new; {session.get_obj_num()} tracked")
                elif (f - first) % 25 == 0 and f > first:
                    rate = (time.time() - started) / (f - first)
                    log(f"  {frames[f]}: {session.get_obj_num()} objects, {rate:.1f} s a frame, "
                        f"about {rate * (len(frames) - f) / 60:.0f} min to go")
            # Backward: each object from its first outline back to the start.
            seeds = {o["seed"] for o in objects.values() if o["seed"] > 0}
            if seeds:
                session = self.session(width, height)
                for f in range(max(seeds), -1, -1):
                    px = self.pixels(image_dir / frames[f])
                    seeded = [i for i, o in objects.items() if o["seed"] == f and f in o["frames"]]
                    for obj_id in seeded:
                        mask = np.unpackbits(masks[Tracks.key(obj_id, frames[f])])[:MASK_SIZE * MASK_SIZE]
                        mask = torch.from_numpy(mask.reshape(1, 1, MASK_SIZE, MASK_SIZE).astype(np.float32))
                        mask = torch.nn.functional.interpolate(mask, size=(INPUT, INPUT), mode="bilinear",
                                                               align_corners=False) >= 0.5
                        session.add_mask_inputs(session.obj_id_to_idx(obj_id), f, mask.to(self.dtype))
                    if seeded:
                        session.obj_with_new_inputs = seeded
                    if session.get_obj_num():
                        step(session, px, f, True)
                    if f % 25 == 0:
                        log(f"  back at {frames[f]}: {session.get_obj_num()} objects")
        tracks = self.finish(objects, masks, frames, width, height, log)
        log(f"  tracked {len(tracks.tracks)} objects through {len(frames)} frames "
            f"in {(time.time() - started) / 60:.1f} min")
        return tracks

    @staticmethod
    def finish(objects: dict, masks: dict, frames: list[str], width: int, height: int, log=print,
               model: str = SEGMENTER_ID) -> Tracks:
        """Merge objects tracing one thing; name each by its votes."""
        def unpack(obj_id, f):
            return np.unpackbits(masks[Tracks.key(obj_id, frames[f])])[:MASK_SIZE * MASK_SIZE].astype(bool)

        ids = sorted(objects)
        merged_into: dict[int, int] = {}
        for i, a in enumerate(ids):
            if a in merged_into:
                continue
            for b in ids[i + 1:]:
                if b in merged_into:
                    continue
                shared = sorted(set(objects[a]["frames"]) & set(objects[b]["frames"]))
                if len(shared) < DUPLICATE_FRAMES:
                    continue
                ious = []
                for f in shared[:: max(1, len(shared) // 12)]:
                    ma, mb = unpack(a, f), unpack(b, f)
                    ious.append((ma & mb).sum() / max((ma | mb).sum(), 1))
                if np.mean(ious) >= DUPLICATE_IOU:
                    merged_into[b] = a
                    for label, v in objects[b]["votes"].items():
                        objects[a]["votes"][label] = round(objects[a]["votes"].get(label, 0.0) + v, 3)
                    for f, seen in objects[b]["frames"].items():
                        if f not in objects[a]["frames"]:
                            objects[a]["frames"][f] = seen
                            masks[Tracks.key(a, frames[f])] = masks[Tracks.key(b, frames[f])]
                    objects[a]["seed"] = min(objects[a]["seed"], objects[b]["seed"])
        for b in merged_into:
            for f in objects[b]["frames"]:
                masks.pop(Tracks.key(b, frames[f]), None)
            del objects[b]
        if merged_into:
            log(f"  merged {len(merged_into)} duplicate object(s)")
        tracks = []
        for obj_id in sorted(objects):
            o = objects[obj_id]
            if not o["frames"]:
                continue
            label = max(o["votes"].items(), key=lambda kv: kv[1])[0]
            tracks.append({"id": obj_id, "label": label, "votes": o["votes"], "seed": frames[o["seed"]],
                           "frames": {frames[f]: {"box": box, "score": score}
                                      for f, (box, score) in sorted(o["frames"].items())}})
        return Tracks(frames, (width, height), tracks, masks, model)


class Sam3Tracker:
    """SAM 3's video model: the room's names as text prompts; every object of
    each name found, outlined and carried through the video by the one model.

    Each frame's outputs (an object's prompt, score, box and mask) become the
    same records the SAM 2.1 tracker keeps, and Tracker.finish merges and names
    them the same way. An object's name is the prompt it was found under; one
    thing found under two names (a door as "door" and as "wardrobe") is two
    objects tracing one outline, merged with both names' votes, so
    Tracks.doorish still sees a door that lost the vote."""

    def __init__(self, device: str | None = None):
        import torch
        from transformers import Sam3VideoModel, Sam3VideoProcessor

        self.torch = torch
        self.device = device or default_device()
        # SAM 3 is trained in bfloat16; a T4 has none, and float16 overflows, so float32 there.
        self.dtype = (torch.bfloat16 if self.device == "cuda" and torch.cuda.is_bf16_supported()
                      else torch.float32)
        self.processor = Sam3VideoProcessor.from_pretrained(SAM3_ID)
        self.model = Sam3VideoModel.from_pretrained(SAM3_ID, dtype=self.dtype).to(self.device).eval()

    def run(self, image_dir: Path, frames: list[str], names: list[str], log=print) -> Tracks:
        """Find and track every object named in `names` through `frames`."""
        from PIL import Image

        image_dir = Path(image_dir)
        width, height = Image.open(image_dir / frames[0]).size
        scale = min(1.0, SAM3_WORK / max(width, height))
        size = (round(width * scale), round(height * scale))
        video = [Image.open(image_dir / f).convert("RGB").resize(size, Image.BILINEAR) for f in frames]
        session = self.processor.init_video_session(video=video, inference_device=self.device,
                                                    processing_device="cpu", video_storage_device="cpu",
                                                    dtype=self.dtype)
        self.processor.add_text_prompt(inference_session=session, text=list(names))
        started = time.time()

        def outputs():
            with self.torch.no_grad():
                for out in self.model.propagate_in_video_iterator(inference_session=session):
                    result = self.processor.postprocess_outputs(session, out)
                    if out.frame_idx and out.frame_idx % 25 == 0:
                        rate = (time.time() - started) / out.frame_idx
                        log(f"  {frames[out.frame_idx]}: {len(result['object_ids'])} objects, {rate:.1f} s a frame, "
                            f"about {rate * (len(frames) - out.frame_idx) / 60:.0f} min to go")
                    yield out.frame_idx, result

        tracks = sam3_tracks(outputs(), frames, (width, height), scale, log)
        log(f"  tracked {len(tracks.tracks)} objects through {len(frames)} frames "
            f"in {(time.time() - started) / 60:.1f} min")
        return tracks


def as_numpy(value) -> np.ndarray:
    """A tensor or a list as a numpy array (bfloat16 has no numpy type)."""
    if hasattr(value, "detach"):
        value = value.detach().float().cpu() if value.is_floating_point() else value.detach().cpu()
        return value.numpy()
    return np.asarray(value)


def sam3_tracks(outputs, frames: list[str], size: tuple[int, int], scale: float, log=print) -> Tracks:
    """SAM 3's processed outputs, (frame index, {object_ids, scores, boxes, masks,
    prompt_to_obj_ids}) per frame at `scale` of the frames' size, as Tracks.

    Votes: every frame an object shows in adds its score under its prompt,
    weighted so that a whole video adds up like the detector's PROMPT_FRAMES
    frames do, which keeps Tracks.doorish's vote thresholds meaning the same."""
    from PIL import Image

    width, height = size
    weight = min(1.0, PROMPT_FRAMES / max(len(frames), 1))
    objects: dict[int, dict] = {}
    masks: dict[str, np.ndarray] = {}
    for f, result in outputs:
        prompt_of = {int(i): name for name, ids in result["prompt_to_obj_ids"].items()
                     for i in as_numpy(ids).ravel()}
        ids, scores = as_numpy(result["object_ids"]).ravel(), as_numpy(result["scores"]).ravel()
        boxes, outlines = as_numpy(result["boxes"]).reshape(-1, 4), as_numpy(result["masks"])
        for obj_id, score, box, mask in zip(ids, scores, boxes, outlines):
            obj_id, mask = int(obj_id), np.asarray(mask) > 0
            label = prompt_of.get(obj_id)
            if label is None or mask.sum() < MIN_AREA_SHARE * mask.size:
                continue
            seen = objects.setdefault(obj_id, {"votes": {}, "seed": f, "frames": {}})
            seen["seed"] = min(seen["seed"], f)
            seen["votes"][label] = round(seen["votes"].get(label, 0.0) + float(score) * weight, 3)
            seen["frames"][f] = ([float(v) / scale for v in box], round(float(score), 3))
            square = Image.fromarray(mask.astype(np.uint8) * 255).resize((MASK_SIZE, MASK_SIZE), Image.NEAREST)
            masks[Tracks.key(obj_id, frames[f])] = np.packbits(np.asarray(square) > 0)
    return Tracker.finish(objects, masks, frames, width, height, log, model=SAM3_ID)


TRACKERS = ("auto", "sam3", "sam2")


def track(image_dir: Path, frames: list[str], prompts: dict[str, list[dict]], vocabulary,
          choice: str = "auto", device: str | None = None, log=print) -> tuple[Tracks, dict]:
    """The room's names (tracked_names) carried through `frames`; returns the
    tracks and the detections of the names that are not tracked ({frame: dets}),
    which keep their per-keyframe detections. `choice`: "sam3" finds and tracks
    by name in one model; "sam2" tracks the detector's `prompts`; "auto" is SAM 3
    when its weights load (access granted, logged in) and SAM 2.1 otherwise."""
    tracked = tracked_names(vocabulary)
    to_track, rest = split_prompts(prompts, tracked)
    if choice in ("auto", "sam3"):
        names = [n for n in vocabulary.names if n in tracked]
        try:
            tracker = Sam3Tracker(device)
        except Exception as exc:     # gated and not granted, not logged in, or an older transformers
            if choice == "sam3":
                raise
            log(f"SAM 3 not available ({str(exc).strip().splitlines()[0][:160]}); tracking with SAM 2.1")
        else:
            log(f"Object tracking: {SAM3_ID} (video) on {tracker.device} ({tracker.dtype}), "
                f"{len(frames)} frames; finding and tracking " + ", ".join(names))
            return tracker.run(image_dir, frames, names, log), rest
    tracker = Tracker(device=device)
    log(f"Object tracking: {SEGMENTER_ID} (video) on {tracker.device} ({tracker.dtype}), "
        f"{len(prompts)} prompt frames of {len(frames)}; tracking "
        + (", ".join(sorted({d["label"] for dets in to_track.values() for d in dets})) or "nothing"))
    return tracker.run(image_dir, frames, to_track, log), rest


# ------------------------------------------------------------------ for the eye
PALETTE = [(230, 60, 60), (60, 180, 75), (60, 110, 230), (240, 170, 40), (170, 70, 220),
           (40, 190, 200), (250, 120, 180), (140, 200, 60), (200, 120, 60), (90, 90, 230)]


def sheet(tracks: Tracks, image_dir: Path, out: Path, count: int = 8, tile: int = 360) -> Path:
    """Evenly spaced frames with every tracked object tinted (one colour per
    object, the same in every frame) and named."""
    from PIL import Image, ImageDraw

    shown = [f for f in tracks.frames if any(f in t["frames"] for t in tracks.tracks)] or tracks.frames
    picks = shown[:: max(1, len(shown) // count)][:count]
    tiles = []
    for name in picks:
        img = Image.open(Path(image_dir) / name).convert("RGB")
        img.thumbnail((tile, tile))
        over = np.asarray(img).astype(np.float32)
        draw_later = []
        for track in tracks.tracks:
            mask = tracks.mask(track["id"], name, over.shape[:2])
            if mask is None:
                continue
            colour = np.array(PALETTE[track["id"] % len(PALETTE)], np.float32)
            over[mask] = over[mask] * 0.5 + colour * 0.5
            rows, cols = np.flatnonzero(mask.any(1)), np.flatnonzero(mask.any(0))
            draw_later.append((f"{track['id']} {track['label']}", (int(cols[0]) + 2, int(rows[0]) + 1),
                               tuple(int(c) for c in colour)))
        img = Image.fromarray(over.astype(np.uint8))
        draw = ImageDraw.Draw(img)
        for text, at, colour in draw_later:
            draw.text(at, text, fill=colour)
        draw.text((4, img.height - 14), name, fill=(255, 255, 255))
        tiles.append(img)
    cols = min(4, len(tiles))
    rows = (len(tiles) + cols - 1) // cols
    w, h = max(t.width for t in tiles), max(t.height for t in tiles)
    page = Image.new("RGB", (cols * w, rows * h), (20, 20, 20))
    for i, t in enumerate(tiles):
        page.paste(t, ((i % cols) * w, (i // cols) * h))
    out = Path(out)
    page.save(out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("space")
    parser.add_argument("--sheet", default=None, help="write a contact sheet of the tracks here")
    parser.add_argument("--update-densify", action="store_true",
                        help="rewrite densify.json's detections from the tracks (the cloud's own "
                             "point labels stay as densified)")
    parser.add_argument("--prompt-frames", type=int, default=PROMPT_FRAMES)
    parser.add_argument("--tracker", choices=TRACKERS, default="auto",
                        help="sam3: find and track by name with SAM 3 (gated); sam2: track the "
                             "detector's finds with SAM 2.1; auto: SAM 3 when it loads")
    args = parser.parse_args()
    from PIL import Image

    from semantics import Detector, planned_vocabulary

    space = Path(args.space)
    image_dir = space / "workspace" / "images"
    frames = sorted(p.name for p in image_dir.glob("*.jpg"))
    meta_path = space / "densify.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    keyframes = sorted(meta.get("detections") or {})
    prompt_names = prompt_frames(frames, keyframes, args.prompt_frames)
    vocabulary = planned_vocabulary(space)
    detector = Detector(vocabulary=vocabulary)
    print(f"Detecting on {len(prompt_names)} of {len(frames)} frames ({detector.device}): "
          + ", ".join(vocabulary.names))
    prompts = {}
    for name in prompt_names:
        prompts[name] = detector.detect(Image.open(image_dir / name))
        print(f"  {name}: " + (", ".join(sorted({d['label'] for d in prompts[name]})) or "nothing"))
    del detector
    tracks, rest = track(image_dir, frames, prompts, vocabulary, args.tracker)
    tracks.save(space / "workspace" / "tracks")
    for t in tracks.tracks:
        votes = ", ".join(f"{k} {v:.1f}" for k, v in sorted(t["votes"].items(), key=lambda kv: -kv[1]))
        print(f"  object {t['id']}: {t['label']} in {len(t['frames'])} frames (votes: {votes})")
    if args.update_densify and meta:
        meta["detections"] = merge_detections(tracks.all_detections(), rest)
        meta["tracks"] = tracks.summary()
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")
        print(f"densify.json: detections now from the tracks, {len(meta['detections'])} frames")
    if args.sheet:
        print(f"sheet -> {sheet(tracks, image_dir, Path(args.sheet))}")


if __name__ == "__main__":
    main()
