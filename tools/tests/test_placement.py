"""Placing furniture by its masks in the keyframes.

    python3 tools/tests/test_placement.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "pipeline"))
import placement as pl  # noqa: E402

UNITS = 9.0


def look_at(position, target):
    """A COLMAP pose (R, t) for a camera at `position` looking at `target`, image up = +z."""
    forward = np.asarray(target, float) - np.asarray(position, float)
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)                     # COLMAP: +y is down the image, +z forward
    R = np.stack([right, down, forward])                # rows: camera axes in world
    t = -R @ np.asarray(position, float)
    return R, t


def view(position, target):
    R, t = look_at(position, target)
    return {"R": R, "t": t, "world": np.eye(3), "fx": 1000.0, "fy": 1000.0, "cx": 540.0, "cy": 960.0,
            "width": 1080, "height": 1920}


def room():
    def wall(x=None, y=None):
        if x is not None:
            return {"kind": "wall", "normal": [-1.0, 0, 0], "center": [x, 9.0, 0.0], "axis_a": [0, 1.0, 0],
                    "axis_b": [0, 0, 1.0], "half_a": 13.0, "half_b": 11.5, "points": 1000}
        return {"kind": "wall", "normal": [0, -1.0, 0], "center": [-7.0, y, 0.0], "axis_a": [1.0, 0, 0],
                "axis_b": [0, 0, 1.0], "half_a": 13.5, "half_b": 11.5, "points": 1000}
    return {"planes": [wall(x=-20.0), wall(y=22.0), wall(x=6.0), wall(y=-4.0)], "boxes": [],
            "room": {"center": [-7.0, 9.0]}, "room_level": {"floor_z": -11.5, "height": 23.0}, "cameras": []}


def test_the_hull_of_a_square_is_its_corners():
    pts = np.array([[0, 0], [2, 0], [2, 2], [0, 2], [1, 1], [1, 0.5]], float)
    hull = pl.convex_hull(pts)
    assert len(hull) == 4 and {tuple(p) for p in hull} == {(0, 0), (2, 0), (2, 2), (0, 2)}


def test_a_box_in_front_of_the_lens_has_a_silhouette_and_one_behind_has_none():
    v = view([0, -30, 0], [0, 0, 0])
    sil = pl.silhouette(np.array([-5, -5, -5.0]), np.array([5, 5, 5.0]), v, (240, 135))
    assert 0 < sil.mean() < 0.5
    rows, cols = np.nonzero(sil)
    assert abs(cols.mean() - 540 / pl.GRID) < 3 and abs(rows.mean() - 960 / pl.GRID) < 3   # centred
    assert not pl.silhouette(np.array([-5, -50, -5.0]), np.array([5, -40, 5.0]), v, (240, 135)).any()   # behind it


def test_the_search_recovers_a_wardrobe_from_its_own_silhouettes():
    shapes = room()
    # The truth: a 1.0 x 0.6 x 2.0 m wardrobe against W0 (x = -20), 1 m along it from the W3 corner (y = -4).
    lo = np.array([-20.0, -4.0 + 1.0 * UNITS, -11.5])
    hi = np.array([-20.0 + 0.6 * UNITS, -4.0 + 2.0 * UNITS, -11.5 + 2.0 * UNITS])
    centre = (lo + hi) / 2
    views = {f"frame_{k:05d}.jpg": view(pos, centre) for k, pos in enumerate(
        [[-5.0, 2.0, 1.0], [-2.0, 12.0, 1.0], [-8.0, 20.0, 2.0]], 1)}
    frames = {n: {"mask": pl.silhouette(lo, hi, v, (240, 135)), **v} for n, v in views.items()}
    unseen = {"frame_00009.jpg": view([0.0, 9.0, 1.0], [-7.0, 9.0 + 8.0, 0.0])}   # looks at W0 elsewhere: no wardrobe there
    evidence = {"label": "wardrobe", "frames": frames, "unseen": unseen}
    box = pl.search(Path("."), shapes, "wardrobe", evidence, UNITS, log=lambda *_: None)
    assert box is not None and box["placement"]["wall"] == 0
    assert np.abs(np.array(box["min"]) - lo).max() < 0.2 * UNITS
    assert np.abs(np.array(box["max"]) - hi).max() < 0.2 * UNITS
    assert box["placement"]["score"] > 0.7


def test_a_few_false_detections_do_not_sink_a_piece_but_a_few_good_frames_do_not_carry_one():
    bed = [0.0, 0.0, 0.62, 0.66, 0.65, 0.52, 0.41, 0.43, 0.22, 0.16, 0.0]     # the pan's bed: 3 false of 11
    door = [0.0, 0.05, 0.36, 0.72, 0.0]                                        # the door read as a wardrobe
    assert pl.trimmed_mean(bed) > pl.ACCEPT_SCORE > pl.trimmed_mean(door)
    assert pl.trimmed_mean([0.9]) == 0.9 and pl.trimmed_mean([]) == 0.0


def test_nothing_is_placed_without_evidence_or_where_the_phone_was():
    shapes = room()
    assert pl.search(Path("."), shapes, "wardrobe", {"label": "wardrobe", "frames": {}, "unseen": {}}, UNITS,
                     log=lambda *_: None) is None
    # a camera standing on every spot along W0 keeps candidates off that wall
    shapes["cameras"] = [[-20.0 + 0.3 * UNITS, y] for y in np.arange(-4.0, 22.0, 0.5)]
    keys = {k[0] for k, _, _ in pl.candidates(shapes, "wardrobe", UNITS)}
    assert 0 not in keys and keys <= {1, 2, 3}


if __name__ == "__main__":
    for name, test in sorted(globals().items()):
        if name.startswith("test_"):
            test()
            print("ok", name)
