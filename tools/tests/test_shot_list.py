"""The shot list: each gap in the capture becomes one concrete input to take.

    python3 tools/tests/test_shot_list.py
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import shot_list  # noqa: E402


def test_never_faced_bearings_cluster_into_arcs_across_the_wrap():
    assert shot_list.arcs([]) == []
    assert shot_list.arcs([0, 30, 60]) == [(0.0, 90.0)]
    assert shot_list.arcs([-180, -150, 150]) == [(150.0, 240.0)]     # one arc over the wrap
    assert shot_list.arcs([0, 90]) == [(0.0, 30.0), (90.0, 120.0)]   # a gap keeps them apart


def space_with(capture: dict, shapes: dict | None = None, units: float | None = None) -> Path:
    folder = Path(tempfile.mkdtemp())
    (folder / "capture-map.json").write_text(json.dumps(capture))
    if shapes is not None:
        (folder / "shapes.json").write_text(json.dumps(shapes))
    if units is not None:
        (folder / "densify.json").write_text(json.dumps({"colmap_units_per_metre": units}))
    return folder


def test_each_gap_becomes_one_cited_shot_and_a_drawn_plan():
    wall = {"kind": "wall", "normal": [0, 1, 0], "center": [10.0, 14.0, 10.0],
            "axis_a": [1, 0, 0], "half_a": 9.0, "points": 2000}
    inferred = dict(wall, source="inferred", center=[0.0, 7.0, 10.0], axis_a=[0, 1, 0], half_a=7.0)
    space = space_with(
        {"directions_never_faced_deg": [0, 30], "directions_faced": "10 of 12",
         "floor_never_seen": 0.5},
        {"planes": [wall, inferred], "cameras": [[4.0, 4.0], [10.0, 5.0], [16.0, 4.0]]},
        units=9.0)
    shots = shot_list.build(space, {"W0": {"support": 0.1, "body": 0.1, "occluded": False}},
                            log=lambda *_: None)
    takes = " | ".join(s["take"] for s in shots)
    assert "pan across the shaded arc (0 to 60 degrees" in takes
    assert "wall W0 (2.0 m)" in takes and "wall W1 (1.56 m)" in takes
    assert "unseen floor" in takes
    assert all(s["evidence"] for s in shots)
    assert [s["shot"] for s in shots] == [1, 2, 3, 4]
    assert json.loads((space / "shots.json").read_text())["shots"]
    assert (space / "shots.png").stat().st_size > 1000


def test_a_mostly_unfaced_room_asks_for_one_loop_and_a_covered_one_for_nothing():
    space = space_with({"directions_never_faced_deg": [-180, -150, -120, -90, 0, 30, 60, 90],
                        "directions_faced": "4 of 12", "floor_never_seen": 0.1})
    shots = shot_list.build(space, log=lambda *_: None)
    assert len(shots) == 1 and "full loop" in shots[0]["take"]
    covered = space_with({"directions_never_faced_deg": [], "directions_faced": "12 of 12",
                          "floor_never_seen": 0.05})
    assert shot_list.build(covered, log=lambda *_: None) == []
    # An occluded wall is hidden, not unfilmed: no pass for it.
    assert shot_list.wall_passes(
        {"planes": [{"kind": "wall", "normal": [0, 1, 0], "center": [0, 0, 0],
                     "axis_a": [1, 0, 0], "half_a": 5.0, "build": True}]},
        {"W0": {"support": 0.0, "occluded": True}}, None) == []


if __name__ == "__main__":
    for name, test in sorted(globals().items()):
        if name.startswith("test_"):
            test()
            print("ok", name)
