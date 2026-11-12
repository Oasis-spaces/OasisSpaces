#!/usr/bin/env python3
"""SAM 3.1 on a space's frames: what it finds and tracks, how fast, in what form.

A first look before SAM 3.1 becomes a tracker in pipeline/tracking.py: the room's
object names are given as text prompts to Meta's multiplex video predictor (the
sam3 package; a CUDA GPU), and for each name this records how many objects it
tracks through the frames, their scores, the time taken, and the structure of the
predictor's outputs. Writes <space>/sam31-probe.json and overlays of a few frames
in <space>/sam31-probe/.

    python3 tools/sam31_probe.py spaces/<name> [--every 2]
"""
import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
from semantics import planned_vocabulary  # noqa: E402

SHOW = 4            # frames drawn with their outlines


def describe(value, depth=0):
    """The shape of a nested output, for the record."""
    if isinstance(value, dict):
        return {str(k): describe(v, depth + 1) for k, v in list(value.items())[:20]} if depth < 4 else "dict"
    if isinstance(value, (list, tuple)):
        return [describe(value[0], depth + 1), f"x{len(value)}"] if value else []
    if hasattr(value, "shape"):
        return f"{type(value).__name__}{tuple(value.shape)} {getattr(value, 'dtype', '')}"
    return type(value).__name__


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("space", type=Path)
    parser.add_argument("--every", type=int, default=2)
    args = parser.parse_args()
    import torch
    from sam3.model_builder import build_sam3_multiplex_video_predictor

    frames = sorted((args.space / "workspace" / "images").glob("*.jpg"))[::args.every]
    folder = Path(tempfile.mkdtemp())
    for i, f in enumerate(frames):            # the predictor wants frames named by their index
        img = Image.open(f).convert("RGB")
        img.thumbnail((1008, 1008))
        img.save(folder / f"{i:05d}.jpg", quality=92)
    vocabulary = planned_vocabulary(args.space)
    names = vocabulary.with_role("furniture", "storage", "fixture", "hanging", "unreliable")
    record = {"frames": len(frames), "every": args.every, "gpu": torch.cuda.get_device_name(0), "names": {}}
    started = time.time()
    predictor = build_sam3_multiplex_video_predictor()
    record["load_seconds"] = round(time.time() - started, 1)
    # Meta's start_session passes offload_state_to_cpu, which the multiplex model's
    # init_state does not take (sam3 main, Nov 2026): drop it there.
    init_state = predictor.model.init_state

    def init_without_offload(*a, **kw):
        kw.pop("offload_state_to_cpu", None)
        return init_state(*a, **kw)

    predictor.model.init_state = init_without_offload
    session = predictor.handle_request(request=dict(type="start_session", resource_path=str(folder)))["session_id"]
    shown = {}
    for name in names:
        t = time.time()
        predictor.handle_request(request=dict(type="reset_session", session_id=session))
        first = predictor.handle_request(request=dict(type="add_prompt", session_id=session,
                                                      frame_index=0, text=name))
        per_frame = {}
        for response in predictor.handle_stream_request(request=dict(type="propagate_in_video", session_id=session)):
            per_frame[response["frame_index"]] = response["outputs"]
        if "structure" not in record:
            record["structure"] = {"add_prompt": describe(first), "frame_output": describe(next(iter(per_frame.values())))}
        ids, scores = set(), []
        for out in per_frame.values():
            obj_ids = out.get("out_obj_ids") if isinstance(out, dict) else None
            if obj_ids is not None:
                ids |= {int(i) for i in np.asarray(obj_ids).ravel()}
            probs = out.get("out_probs") if isinstance(out, dict) else None
            if probs is not None:
                scores += [float(p) for p in np.asarray(probs).ravel()]
        record["names"][name] = {"objects": len(ids), "frames_with_any": sum(
            1 for out in per_frame.values() if isinstance(out, dict) and len(np.asarray(out.get("out_obj_ids", []))) > 0),
            "mean_score": round(float(np.mean(scores)), 3) if scores else None, "seconds": round(time.time() - t, 1)}
        print(name, record["names"][name], flush=True)
        for index, out in per_frame.items():
            masks = out.get("out_binary_masks") if isinstance(out, dict) else None
            if masks is not None and len(masks) and len(shown) < SHOW * 3:
                shown.setdefault(index, []).append((name, np.asarray(masks)))
    record["total_seconds"] = round(time.time() - started, 1)
    out_dir = args.space / "sam31-probe"
    out_dir.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    for index in sorted(shown)[:SHOW]:
        img = np.asarray(Image.open(folder / f"{index:05d}.jpg").convert("RGB")).astype(float)
        for name, masks in shown[index]:
            colour = rng.integers(60, 255, 3)
            for m in masks.reshape(-1, *masks.shape[-2:]):
                if m.shape == img.shape[:2]:
                    img[m > 0] = 0.55 * img[m > 0] + 0.45 * colour
        Image.fromarray(img.astype(np.uint8)).save(out_dir / f"frame-{index:03d}.jpg", quality=88)
    (args.space / "sam31-probe.json").write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps(record, indent=1))


if __name__ == "__main__":
    main()
