#!/usr/bin/env python3
"""SAM 3.1 tracking, in chunks and one pass per name, with a stand-in for Meta's
predictor: a bed seen throughout and a door from frame 60, over 100 frames. The
bed must come out as one track over every frame, the door as one from frame 60."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "pipeline"))
import tracking  # noqa: E402

H = W = 64


class Predictor:
    def __init__(self):
        self.sessions = {}

    def handle_request(self, request):
        kind = request["type"]
        if kind == "start_session":
            names = sorted(p.name for p in Path(request["resource_path"]).glob("*.jpg"))
            sid = f"s{len(self.sessions)}"
            start = int(Path(request["resource_path"]).name.split("-")[-1])
            self.sessions[sid] = {"n": len(names), "start": start, "text": None}
            return {"session_id": sid}
        if kind == "add_prompt":
            self.sessions[request["session_id"]]["text"] = request["text"]
        return {}

    def handle_stream_request(self, request):
        s = self.sessions[request["session_id"]]
        for i in range(s["n"]):
            f = s["start"] + i
            ids, masks = [], []
            if s["text"] == "bed":
                m = np.zeros((H, W), bool); m[10:50, 5:30] = True
                ids, masks = [11], [m]
            elif s["text"] == "door" and f >= 60:
                m = np.zeros((H, W), bool); m[10:50, 40:60] = True
                ids, masks = [4], [m]
            yield {"frame_index": i, "outputs": {"out_obj_ids": np.array(ids), "out_probs": np.full(len(ids), 0.9),
                                                  "out_boxes_xywh": np.full((len(ids), 4), 0.2),
                                                  "out_binary_masks": np.array(masks).reshape(len(ids), H, W)}}


def main():
    import tempfile
    folder = Path(tempfile.mkdtemp())
    frames = [f"frame_{i:05d}.jpg" for i in range(100)]
    for f in frames:
        Image.new("RGB", (W, H)).save(folder / f)
    starts = iter(range(0, 100, tracking.SAM3_CHUNK - tracking.SAM3_OVERLAP))
    real_mkdtemp = tempfile.mkdtemp
    def chunk_folder(prefix=""):              # the folder's name tells the stand-in which chunk it plays
        path = Path(real_mkdtemp(prefix=prefix)) / f"chunk-{next(starts)}"
        path.mkdir()
        return str(path)

    tempfile.mkdtemp = chunk_folder
    tracker = tracking.Sam31Tracker.__new__(tracking.Sam31Tracker)
    import torch
    tracker.torch, tracker.predictor = torch, Predictor()
    try:
        tracks = tracker.run(folder, frames, ["bed", "door"], log=lambda *_: None)
    finally:
        tempfile.mkdtemp = real_mkdtemp
    by = {}
    for t in tracks.tracks:
        by.setdefault(t["label"], []).append(t)
    checks = [("one bed track", len(by.get("bed", [])) == 1),
              ("one door track", len(by.get("door", [])) == 1),
              ("the bed is seen in every frame", by.get("bed") and len(by["bed"][0]["frames"]) == 100),
              ("the door from frame 60 on", by.get("door") and min(by["door"][0]["frames"]) == "frame_00060.jpg"
               and len(by["door"][0]["frames"]) == 40)]
    for what, ok in checks:
        print(("ok   " if ok else "FAIL ") + what)
    ok = all(c for _, c in checks)
    print("ok" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
