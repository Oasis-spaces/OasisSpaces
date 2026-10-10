"""The CLIP crop vote: cropping, abstaining, the bounded merge, the election.

    python3 tools/tests/test_crop_vote.py

A fake scorer stands in for CLIP, so the mechanics are tested without
weights or a network — the same way the tracking tests stand in for SAM.
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "pipeline"))
import crop_vote  # noqa: E402

NAMES = ["wardrobe", "door", "bed"]


class FakeScorer:
    """Returns the rows it was built with, however many crops come in."""

    def __init__(self, row):
        self.row = np.array(row, float)

    def scores(self, crops):
        return np.tile(self.row, (len(crops), 1))


def test_a_crop_is_widened_a_little_and_clamped_to_the_frame():
    image = Image.new("RGB", (100, 80))
    inside = crop_vote.crop(image, [20, 20, 60, 60], context=0.1)
    assert inside.size == (48, 48)                 # 40 px box widened by 4 px each side
    corner = crop_vote.crop(image, [0, 0, 50, 50], context=0.1)
    assert corner.size == (55, 55)                 # nothing to widen past the frame's edge


def test_clips_mean_is_taken_and_a_thin_margin_abstains():
    sure = crop_vote.clip_vote(np.array([[0.1, 0.8, 0.1], [0.1, 0.6, 0.3]]), NAMES)
    assert sure is not None and abs(sure["door"] - 0.7) < 1e-9
    unsure = crop_vote.clip_vote(np.array([[0.45, 0.40, 0.15]]), NAMES)
    assert unsure is None                          # 0.05 between its top two names
    assert crop_vote.clip_vote(np.zeros((0, 3)), NAMES) is None


def test_the_pans_door_flips_and_a_sure_detector_does_not():
    # The real mistake: the pan's door came out "wardrobe 3.7, door 1.9"
    # (tracking.py's own record). A confident CLIP flips it.
    votes = {"wardrobe": 3.7, "door": 1.9}
    merged = crop_vote.merged_votes(votes, {"wardrobe": 0.05, "door": 0.9, "bed": 0.05})
    assert crop_vote.winner(merged) == "door"
    assert abs(merged["door"] - (1.9 + 0.5 * 5.6 * 0.9)) < 1e-3
    # A detector that is sure keeps its name whatever CLIP says.
    sure = crop_vote.merged_votes({"wardrobe": 9.0, "door": 1.0},
                                  {"wardrobe": 0.0, "door": 1.0, "bed": 0.0})
    assert crop_vote.winner(sure) == "wardrobe"
    # An abstention changes nothing at all, and nothing is added to no votes.
    assert crop_vote.merged_votes(votes, None) == votes
    assert crop_vote.merged_votes({}, {"door": 1.0}) == {}


def test_a_degenerate_hand_box_still_crops_at_least_a_pixel():
    image = Image.new("RGB", (100, 80))
    assert crop_vote.crop(image, [60, 60, 20, 20], context=0.1).size == (48, 48)  # corners sorted
    assert crop_vote.crop(image, [10, 10, 10, 10]).size == (1, 1)     # zero-size box
    assert min(crop_vote.crop(image, [200, 200, 300, 300]).size) >= 1  # fully outside


def test_the_clearest_sightings_are_the_highest_scoring_frames_on_disk():
    with tempfile.TemporaryDirectory() as folder:
        images = Path(folder)
        for i in range(1, 8):                      # frame_00008 is not on disk
            Image.new("RGB", (8, 8)).save(images / f"frame_{i:05d}.jpg")
        track = {"frames": {f"frame_{i:05d}.jpg": {"box": [0, 0, 10, 10], "score": i / 10}
                            for i in range(1, 9)}}
        picked = crop_vote.clearest_sightings(track, images, count=3)
    assert [frame for frame, _ in picked] == ["frame_00007.jpg", "frame_00006.jpg", "frame_00005.jpg"]


def test_the_second_opinion_reports_each_tracks_election():
    with tempfile.TemporaryDirectory() as folder:
        images = Path(folder)
        for i in (1, 2):
            Image.new("RGB", (64, 48), (120, 110, 100)).save(images / f"frame_{i:05d}.jpg")
        tracks = [
            {"id": 1, "label": "wardrobe", "votes": {"wardrobe": 3.7, "door": 2.4},
             "frames": {"frame_00001.jpg": {"box": [4, 4, 40, 40], "score": 0.9},
                        "frame_00002.jpg": {"box": [6, 4, 42, 40], "score": 0.8}}},
            {"id": 2, "label": "bed", "votes": {"bed": 5.0},
             "frames": {"frame_00009.jpg": {"box": [0, 0, 10, 10], "score": 0.9}}},  # not on disk
        ]
        rows = crop_vote.second_opinion(tracks, images, NAMES,
                                        FakeScorer([0.05, 0.9, 0.05]))
        assert len(rows) == 1                      # the frameless track is skipped
        assert rows[0]["crops"] == 2
        assert rows[0]["elected"] == "door" and rows[0]["changed"]
        assert json.dumps(rows[0])                 # the report row is plain JSON


if __name__ == "__main__":
    failures = 0
    for name, test in sorted(globals().items()):
        if name.startswith("test_"):
            try:
                test()
                print(f"ok  {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL {name}: {exc}")
    sys.exit(1 if failures else 0)
