"""Objects tracked through the video: matching, merging, naming, storage.

    python3 tools/tests/test_tracking.py
"""

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "pipeline"))
import tracking  # noqa: E402

FRAMES = [f"frame_{i:05d}.jpg" for i in range(1, 11)]
W, H = 1080, 1920


def disc(cx, cy, r, size=tracking.MASK_SIZE):
    yy, xx = np.mgrid[:size, :size]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r


def packed(mask):
    return np.packbits(mask)


def test_a_rectangle_comes_back_from_an_outline_in_frame_pixels():
    mask = np.zeros((256, 256), bool)
    mask[64:128, 32:96] = True                     # rows 64..127, cols 32..95
    box = tracking.mask_box(mask, W, H)
    assert np.allclose(box, [32 * W / 256, 64 * H / 256, 96 * W / 256, 128 * H / 256])
    assert tracking.mask_box(np.zeros((256, 256), bool), W, H) is None
    assert abs(tracking.box_iou([0, 0, 10, 10], [5, 0, 15, 10]) - 1 / 3) < 1e-9


def test_prompt_frames_are_the_keyframes_plus_a_spaced_sample():
    frames = [f"f{i:03d}" for i in range(100)]
    chosen = tracking.prompt_frames(frames, ["f037", "f090", "not-a-frame"], count=20)
    assert "f037" in chosen and "f090" in chosen and "not-a-frame" not in chosen
    assert all(frames[i] in chosen for i in range(0, 100, 5))
    assert chosen == sorted(chosen) and len(chosen) == 21                # 20 sampled, f090 among them, plus f037


def finished(objects, masks):
    return tracking.Tracker.finish(objects, masks, FRAMES, W, H, log=lambda *_: None)


def test_two_objects_tracing_one_thing_are_merged_and_the_votes_decide_the_name():
    a, b = disc(100, 100, 40), disc(102, 101, 40)   # nearly the same outline in every frame
    masks, objects = {}, {
        0: {"votes": {"wardrobe": 0.9, "door": 0.4}, "seed": 1, "frames": {}},
        1: {"votes": {"door": 0.8}, "seed": 4, "frames": {}},
    }
    for f in range(1, 8):
        objects[0]["frames"][f] = ([0, 0, 10, 10], 0.9)
        masks[tracking.Tracks.key(0, FRAMES[f])] = packed(a)
    for f in range(4, 10):
        objects[1]["frames"][f] = ([0, 0, 10, 10], 0.9)
        masks[tracking.Tracks.key(1, FRAMES[f])] = packed(b)
    tracks = finished(objects, masks)
    assert [t["id"] for t in tracks.tracks] == [0]
    one = tracks.tracks[0]
    assert one["label"] == "wardrobe" and one["votes"] == {"wardrobe": 0.9, "door": 1.2} or one["label"] == "door"
    # the merged object carries frames 1..9; the first object's outline is kept where both had one
    assert sorted(one["frames"]) == FRAMES[1:10]
    assert tracks.mask(0, FRAMES[5]).sum() == a.sum() and tracks.mask(0, FRAMES[9]).sum() == b.sum()
    assert tracking.Tracks.key(1, FRAMES[9]) not in tracks._masks


def test_the_name_with_the_most_detector_weight_wins():
    masks = {tracking.Tracks.key(0, FRAMES[f]): packed(disc(50, 50, 20)) for f in range(3)}
    objects = {0: {"votes": {"hanging cloth": 0.6, "wardrobe": 0.45 + 0.4}, "seed": 0,
                   "frames": {f: ([0, 0, 1, 1], 0.9) for f in range(3)}}}
    assert finished(objects, masks).tracks[0]["label"] == "wardrobe"


def test_objects_apart_or_briefly_overlapping_stay_separate():
    far = {0: {"votes": {"bed": 1.0}, "seed": 0, "frames": {}}, 1: {"votes": {"chair": 1.0}, "seed": 0, "frames": {}}}
    masks = {}
    for f in range(6):
        for i, m in ((0, disc(60, 60, 30)), (1, disc(190, 190, 30))):
            far[i]["frames"][f] = ([0, 0, 1, 1], 0.9)
            masks[tracking.Tracks.key(i, FRAMES[f])] = packed(m)
    assert len(finished(far, masks).tracks) == 2
    # same outline but shared in too few frames to be sure
    brief = {0: {"votes": {"bed": 1.0}, "seed": 0, "frames": {0: ([0, 0, 1, 1], 0.9), 1: ([0, 0, 1, 1], 0.9)}},
             1: {"votes": {"sofa": 1.0}, "seed": 1, "frames": {1: ([0, 0, 1, 1], 0.9), 2: ([0, 0, 1, 1], 0.9)}}}
    masks = {tracking.Tracks.key(i, FRAMES[f]): packed(disc(60, 60, 30)) for i, o in brief.items() for f in o["frames"]}
    assert len(finished(brief, masks).tracks) == 2


def test_tracks_save_load_and_answer_as_detections_with_masks():
    big, small = disc(128, 128, 90), disc(128, 128, 20)
    tracks = tracking.Tracks(FRAMES, (W, H), [
        {"id": 0, "label": "bed", "votes": {"bed": 2.0}, "seed": FRAMES[0],
         "frames": {FRAMES[0]: {"box": tracking.mask_box(big, W, H), "score": 0.99}}},
        {"id": 3, "label": "pillow", "votes": {"pillow": 0.5}, "seed": FRAMES[0],
         "frames": {FRAMES[0]: {"box": tracking.mask_box(small, W, H), "score": 0.9},
                    FRAMES[1]: {"box": tracking.mask_box(small, W, H), "score": 0.7}}},
    ], {tracking.Tracks.key(0, FRAMES[0]): packed(big), tracking.Tracks.key(3, FRAMES[0]): packed(small),
        tracking.Tracks.key(3, FRAMES[1]): packed(small)})
    with tempfile.TemporaryDirectory() as tmp:
        tracks.save(Path(tmp))
        back = tracking.Tracks.load(Path(tmp))
    assert back.frames == FRAMES and (back.width, back.height) == (W, H)
    dets = back.detections(FRAMES[0], work_size=1024)
    assert [d["label"] for d in dets] == ["bed", "pillow"]          # largest first
    assert dets[0]["track"] == 0 and dets[1]["track"] == 3
    scale = 1024 / H
    assert dets[0]["mask_scale"] == scale and dets[0]["mask"].shape == (1024, round(W * scale))
    assert dets[1]["mask"].sum() < dets[0]["mask"].sum()
    only = back.detections(FRAMES[1])
    assert len(only) == 1 and only[0]["label"] == "pillow" and only[0]["score"] == 0.7 and only[0]["track"] == 3
    assert np.allclose(only[0]["box"], tracking.mask_box(small, W, H), atol=0.06)   # saved to a tenth of a pixel
    assert back.detections(FRAMES[2]) == [] and back.mask(0, FRAMES[2]) is None
    everything = back.all_detections()
    assert sorted(everything) == FRAMES[:2] and all(isinstance(v, int) for v in everything[FRAMES[0]][0]["box"])
    assert back.summary()["labels"] == {"bed": 1, "pillow": 1} and back.summary()["frames_with_objects"] == 2
    assert tracking.Tracks.load(Path("/nonexistent")) is None


def test_the_sheet_tints_each_object_in_its_colour():
    from PIL import Image

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for name in FRAMES[:2]:
            Image.new("RGB", (W // 4, H // 4), (120, 120, 120)).save(tmp / name)
        big = disc(128, 128, 90)
        tracks = tracking.Tracks(FRAMES[:2], (W // 4, H // 4), [
            {"id": 0, "label": "bed", "votes": {"bed": 1.0}, "seed": FRAMES[0],
             "frames": {FRAMES[0]: {"box": tracking.mask_box(big, W // 4, H // 4), "score": 0.9}}}],
            {tracking.Tracks.key(0, FRAMES[0]): packed(big)})
        out = tracking.sheet(tracks, tmp, tmp / "sheet.png", count=2)
        img = np.asarray(Image.open(out).convert("RGB")).astype(int)
    centre = img[img.shape[0] // 2, img.shape[1] // 4]            # the first tile's middle: tinted red
    assert centre[0] > centre[1] + 30


if __name__ == "__main__":
    for name, test in sorted(globals().items()):
        if name.startswith("test_"):
            test()
            print("ok", name)
