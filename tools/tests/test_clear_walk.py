"""Furniture boxes give way to the walk: the phone was carried there.

    python3 tools/tests/test_clear_walk.py      (or pytest tools/tests)
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import classify_shapes as cs  # noqa: E402

UNIT = 2.0
ROOM = {"center": [0.0, 0.0], "axis_u": [1.0, 0.0], "axis_v": [0.0, 1.0], "half_u": 2.0 * UNIT, "half_v": 2.0 * UNIT}
SIZE = lambda d: f"{d / UNIT:.2f} m"   # noqa: E731


def box(lo, hi, label="wardrobe"):
    return {"min": [lo[0] * UNIT, lo[1] * UNIT, 0.0], "max": [hi[0] * UNIT, hi[1] * UNIT, 2.0 * UNIT], "label": label}


def walk(*points):
    return [[x * UNIT, y * UNIT] for x, y in points]


def test_a_wardrobe_box_run_out_over_the_walk_is_cut_back_to_it():
    # A wardrobe 0.6 m deep against the wall at y = -2, whose points ran on to y = -0.5,
    # over the walk along y = -0.9.
    b = box((-1.5, -2.0), (1.5, -0.5))
    notes = cs.clear_the_walk([b], walk((-1, -0.9), (0, -0.9), (1, -0.9)), ROOM, SIZE, UNIT)
    assert notes and abs(b["max"][1] / UNIT - (-0.6)) < 1e-9, (b, notes)   # the walk now 0.3 m over its front
    assert b["min"] == [-1.5 * UNIT, -2.0 * UNIT, 0.0]                        # its back stays on the wall


def test_a_hand_over_the_edge_of_a_bed_is_allowed():
    b = box((-1.0, -1.0), (1.0, 1.0), "bed")
    assert cs.clear_the_walk([b], walk((-0.8, 0.0), (-0.75, 0.5)), ROOM, SIZE, UNIT) == []
    assert b["min"][0] == -1.0 * UNIT


def test_a_walk_through_the_middle_is_left_for_the_review():
    b = box((-1.0, -1.0), (1.0, 1.0), "bed")
    notes = cs.clear_the_walk([b], walk((-0.65, -0.65), (0.65, 0.65), (-0.65, 0.65), (0.65, -0.65)), ROOM, SIZE, UNIT,
                              names=["B3"])
    assert notes == ["the walk runs through the middle of B3 bed; left as it is"], notes
    assert b["max"][0] == 1.0 * UNIT


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print("ok", name)
