#!/usr/bin/env python3
"""SAM 3 tracking in chunks: objects carry one identity across chunk borders.

A stand-in for SAM 3's processor and model plays a walk of 100 frames with two
objects: a bed seen throughout and a door seen from frame 60. Tracked in chunks
of 48 frames overlapping by 6, the bed must come out as one track over every
frame and the door as one track from frame 60, with no duplicates."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "pipeline"))
import tracking  # noqa: E402

H = W = 64


def mask(x0, x1):
    m = np.zeros((H, W), bool)
    m[10:50, x0:x1] = True
    return m


class Processor:
    def init_video_session(self, video, **kw):
        return SimpleNamespace(n=len(video), start=None)

    def add_text_prompt(self, inference_session, text):
        pass

    def postprocess_outputs(self, session, out):
        f = out.global_frame
        ids, masks, names = [], [], {"bed": [], "door": []}
        # Each chunk numbers its objects afresh, as SAM 3 does per session.
        ids.append(7); masks.append(mask(5, 30)); names["bed"].append(7)
        if f >= 60:
            ids.append(3); masks.append(mask(40, 60)); names["door"].append(3)
        return {"object_ids": np.array(ids), "scores": np.full(len(ids), 0.9),
                "boxes": np.array([[0, 0, 10, 10]] * len(ids), float), "masks": np.array(masks),
                "prompt_to_obj_ids": {k: np.array(v) for k, v in names.items()}}


class Model:
    def __init__(self, starts):
        self.starts = starts

    def propagate_in_video_iterator(self, inference_session):
        start = self.starts.pop(0)
        for i in range(inference_session.n):
            yield SimpleNamespace(frame_idx=i, global_frame=start + i)


def main():
    import tempfile
    folder = Path(tempfile.mkdtemp())
    frames = [f"frame_{i:05d}.jpg" for i in range(100)]
    for f in frames:
        Image.new("RGB", (W, H)).save(folder / f)
    starts = list(range(0, 100 - tracking.SAM3_OVERLAP, tracking.SAM3_CHUNK - tracking.SAM3_OVERLAP))
    tracker = tracking.Sam3Tracker.__new__(tracking.Sam3Tracker)
    import torch
    tracker.torch, tracker.device, tracker.dtype = torch, "cpu", torch.float32
    tracker.processor, tracker.model = Processor(), Model(list(starts))
    tracks = tracker.run(folder, frames, ["bed", "door"], log=lambda *_: None)
    by_label = {}
    for t in tracks.tracks:
        by_label.setdefault(t["label"], []).append(t)
    ok = True

    def check(what, cond):
        nonlocal ok
        print(("ok   " if cond else "FAIL ") + what)
        ok &= bool(cond)

    check("one bed track", len(by_label.get("bed", [])) == 1)
    check("one door track", len(by_label.get("door", [])) == 1)
    check("the bed is seen in every frame", len(by_label["bed"][0]["frames"]) == 100)
    check("the door is seen from frame 60 on", sorted(by_label["door"][0]["frames"])[0] == "frame_00060.jpg"
          and len(by_label["door"][0]["frames"]) == 40)
    print("ok" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
