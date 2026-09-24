"""The structure review adding furniture the frames show but the points never boxed.

    python3 tools/tests/test_structure_add.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "pipeline"))
import agent  # noqa: E402

UNITS = 9.0                                              # solve units per metre


def wall(x=None, y=None):
    """A wall along the line x = ... or y = ..., 26 units long, 23 high, centred like the pan room."""
    if x is not None:
        return {"kind": "wall", "label": "wall", "normal": [-1.0, 0.0, 0.0], "center": [x, 9.0, 0.0],
                "axis_a": [0.0, 1.0, 0.0], "axis_b": [0.0, 0.0, 1.0], "half_a": 13.0, "half_b": 11.5, "points": 1000}
    return {"kind": "wall", "label": "wall", "normal": [0.0, -1.0, 0.0], "center": [-7.0, y, 0.0],
            "axis_a": [1.0, 0.0, 0.0], "axis_b": [0.0, 0.0, 1.0], "half_a": 13.5, "half_b": 11.5, "points": 1000}


def room():
    return {"planes": [wall(x=-20.0), wall(y=22.0), wall(x=6.0), wall(y=-4.0),
                       {"kind": "floor_or_ceiling", "label": "floor", "normal": [0, 0, 1.0], "center": [-7.0, 9.0, -11.5],
                        "axis_a": [1.0, 0, 0], "axis_b": [0, 1.0, 0], "half_a": 13.5, "half_b": 13.0, "points": 5000}],
            "boxes": [], "room": {"center": [-7.0, 9.0]}, "room_level": {"floor_z": -11.5, "height": 23.0}}


def close(a, b, tol=0.05):
    return all(abs(x - y) < tol for x, y in zip(a, b))


def test_a_wardrobe_stands_in_the_corner_its_back_on_the_wall():
    box = agent.place_added_box(room(), {"label": "wardrobe", "against": "W0", "from_corner_with": "W3",
                                         "offset_m": 0.3, "width_m": 1.0, "depth_m": 0.6, "height_m": 2.0,
                                         "why": "the tall wooden almirah"}, UNITS)
    # W0 is the line x = -20 (normal -x, so the room is toward +x); its corner with W3 (y = -4) is at
    # (-20, -4); 0.3 m along W0 from there the wardrobe starts, 1 m wide, 0.6 m deep into the room.
    assert close(box["min"], [-20.0, -4.0 + 0.3 * UNITS, -11.5])
    assert close(box["max"], [-20.0 + 0.6 * UNITS, -4.0 + 1.3 * UNITS, -11.5 + 2.0 * UNITS])
    assert box["label"] == "wardrobe" and box["build"] and box["source"] == "claude"
    assert "almirah" in box["reason"]


def test_from_the_other_corner_it_runs_the_other_way():
    box = agent.place_added_box(room(), {"label": "wardrobe", "against": "W0", "from_corner_with": "W1",
                                         "offset_m": 0.3, "width_m": 1.0, "depth_m": 0.6, "height_m": 2.0}, UNITS)
    assert close([box["min"][1], box["max"][1]], [22.0 - 1.3 * UNITS, 22.0 - 0.3 * UNITS])


def test_without_a_corner_it_is_centred_on_the_wall():
    box = agent.place_added_box(room(), {"label": "table", "against": "W1", "from_corner_with": None,
                                         "width_m": 1.2, "depth_m": 0.6, "height_m": 0.75}, UNITS)
    assert close([box["min"][0], box["max"][0]], [-7.0 - 0.6 * UNITS, -7.0 + 0.6 * UNITS])
    assert close([box["min"][1], box["max"][1]], [22.0 - 0.6 * UNITS, 22.0])    # its back on W1, into the room


def test_sizes_are_kept_within_reason_and_the_room():
    box = agent.place_added_box(room(), {"label": "wardrobe", "against": "W0", "from_corner_with": "W3",
                                         "offset_m": 0.0, "width_m": 10.0, "depth_m": 0.1, "height_m": 5.0}, UNITS)
    assert close([box["max"][1] - box["min"][1]], [26.0])                        # no wider than the wall
    assert close([box["max"][0] - box["min"][0]], [0.2 * UNITS])                 # no shallower than 0.2 m
    assert close([box["max"][2] - box["min"][2]], [23.0])                        # no taller than the room


def test_a_box_the_phone_stood_in_or_against_is_refused():
    shapes = room()
    item = {"label": "wardrobe", "against": "W0", "from_corner_with": "W3", "offset_m": 0.3,
            "width_m": 1.0, "depth_m": 0.6, "height_m": 2.0}
    shapes["cameras"] = [[-15.0, 0.0]]                               # inside where the wardrobe would stand
    assert "phone stood" in agent.place_added_box(shapes, item, UNITS)
    shapes["cameras"] = [[-20.0 + 0.6 * UNITS + 0.05 * UNITS, 0.0]]  # 5 cm in front of its face
    assert "phone stood" in agent.place_added_box(shapes, item, UNITS)
    shapes["cameras"] = [[-20.0 + 0.6 * UNITS + 0.3 * UNITS, 0.0]]   # 30 cm away: fine
    assert isinstance(agent.place_added_box(shapes, item, UNITS), dict)


def test_what_cannot_be_placed_says_why():
    assert "not a furniture type" in agent.place_added_box(room(), {"label": "sofa", "against": "W0"}, UNITS)
    assert "not a built wall" in agent.place_added_box(
        room(), {"label": "bed", "against": "W4", "width_m": 1, "depth_m": 2, "height_m": 0.5}, UNITS)   # the floor
    assert "needed" in agent.place_added_box(room(), {"label": "bed", "against": "W0"}, UNITS)


def test_the_review_adds_at_most_three_and_records_each():
    stub = agent.Agent.__new__(agent.Agent)
    stub.densify_metrics = lambda: {"colmap_units_per_metre": UNITS}
    shapes = room()
    item = {"label": "wardrobe", "against": "W0", "from_corner_with": "W3", "offset_m": 0.3,
            "width_m": 1.0, "depth_m": 0.6, "height_m": 2.0, "why": "seen in frames 1-8"}
    applied = agent.Agent.apply_structure_review(stub, shapes, {"add_boxes": [item] * 4})
    assert len(shapes["boxes"]) == 3
    assert applied[:3] == [f"added B{n} wardrobe against W0, from its corner with W3" for n in range(3)]
    assert "at most 3" in applied[3]


def test_without_a_measured_scale_nothing_is_added():
    stub = agent.Agent.__new__(agent.Agent)
    stub.densify_metrics = lambda: {}
    shapes = room()
    applied = agent.Agent.apply_structure_review(
        stub, shapes, {"add_boxes": [{"label": "bed", "against": "W0", "width_m": 1.4, "depth_m": 2, "height_m": 0.5}]})
    assert shapes["boxes"] == [] and "not measured" in applied[0]


if __name__ == "__main__":
    for name, test in sorted(globals().items()):
        if name.startswith("test_"):
            test()
            print("ok", name)
