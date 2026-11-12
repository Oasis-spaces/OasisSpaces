#!/usr/bin/env python3
"""A piece the review confirmed where it was measured (hold_place) keeps its size
through the finish: the walk does not cut it, and a wall it reaches past moves
out instead of trimming it. Without this the bedroom's bed lost 0.05 of its fit
every time the room was finished again after the review."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import classify_shapes as C  # noqa: E402

ROOM = {"center": [0.0, 0.0], "axis_u": [1.0, 0.0], "axis_v": [0.0, 1.0], "half_u": 2.0, "half_v": 2.0}


def bed(held):
    b = {"min": [-1.0, -1.0, 0.0], "max": [1.0, 1.0, 0.5], "label": "bed"}
    if held:
        b["hold_place"] = True
    return b


def main():
    size = lambda d: f"{d:.2f}"
    cams = [[0.6, 0.0], [0.7, 0.1]]          # the walk passes over the bed's edge
    free, held = bed(False), bed(True)
    C.clear_the_walk([free, held], cams, ROOM, size, units_per_metre=1.0)
    ok = free["max"][0] < 1.0 and held["max"] == [1.0, 1.0, 0.5]
    print(("ok   " if free["max"][0] < 1.0 else "FAIL ") + "an unconfirmed bed is cut back from the walk")
    print(("ok   " if held["max"] == [1.0, 1.0, 0.5] else "FAIL ") + "a confirmed bed keeps its size")
    print("ok" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
