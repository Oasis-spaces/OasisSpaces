"""The phone's placed objects as candidates for stage 3.

    python3 tools/tests/test_phone_objects.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import phone_objects as po  # noqa: E402

UNITS = 9.0
LOG = """pan-add: 42 frames, 277 ms a frame on this Mac, depth on 42, fitted on 34; 180 observations
placed 2 objects; room yaw 1.52
  B5 bed -> bed footprint IoU 0.81, centre off 0.06 m, size 1.72x0.70x1.54 vs truth 1.41x0.68x1.68
placed:
  o6 bed x -0.78..+0.85  y 0.00..0.70  z -1.52..+0.29  (1.63 x 0.70 x 1.80)  top layer 0.60
  o1 washing machine x -0.19..+0.77  y 0.00..1.70  z +1.31..+1.56  (0.96 x 1.70 x 0.25)  top layer 1.70
  1 placed objects match nothing measured: wardrobe 0.2x1.0
"""


def room():
    def wall(x=None, y=None):
        if x is not None:
            return {"kind": "wall", "normal": [-1.0, 0, 0], "center": [x, 9.0, 0.0], "axis_a": [0, 1.0, 0],
                    "axis_b": [0, 0, 1.0], "half_a": 13.0, "half_b": 11.5, "points": 1000}
        return {"kind": "wall", "normal": [0, -1.0, 0], "center": [-7.0, y, 0.0], "axis_a": [1.0, 0, 0],
                "axis_b": [0, 0, 1.0], "half_a": 13.5, "half_b": 11.5, "points": 1000}
    return {"planes": [wall(x=-20.0), wall(y=22.0), wall(x=6.0), wall(y=-4.0)], "boxes": [],
            "room": {"center": [-7.0, 9.0]}, "room_level": {"floor_z": -11.5, "height": 23.0}, "cameras": []}


def test_the_placed_lines_are_read_with_labels_of_more_than_one_word():
    placed = po.parse_placed(LOG)
    assert [p["label"] for p in placed] == ["bed", "washing machine"]
    assert placed[1]["min"] == [-0.19, 0.0, 1.31] and placed[1]["max"] == [0.77, 1.7, 1.56]
    assert po.parse_placed("no objects here") == []


def test_the_phones_frame_goes_back_into_the_rooms():
    shapes = room()
    # the export's own formula, forwards: phone = (x - ox, z - floor, -(y - oy)) / units
    scene_lo, scene_hi = np.array([-16.0, -4.0, -11.5]), np.array([-7.0, -1.3, 3.8])
    corners = np.array([[x, y, z] for x in (scene_lo[0], scene_hi[0]) for y in (scene_lo[1], scene_hi[1])
                        for z in (scene_lo[2], scene_hi[2])])
    phone = np.stack([corners[:, 0] + 7.0, corners[:, 2] + 11.5, -(corners[:, 1] - 9.0)], 1) / UNITS
    lo, hi = po.to_scene(phone.min(axis=0), phone.max(axis=0), shapes, UNITS)
    assert np.allclose(lo, scene_lo) and np.allclose(hi, scene_hi)


def test_an_object_coinciding_with_a_built_box_is_that_box():
    shapes = room()
    shapes["boxes"] = [{"min": [-16.0, -4.0, -11.5], "max": [-7.0, 5.0, -5.0], "build": True, "label": "bed"}]
    bed = {"id": "o1", "label": "bed", "min": [(-15.5 + 7) / UNITS, 0.0, -(4.5 - 9) / UNITS],
           "max": [(-7.5 + 7) / UNITS, 0.7, -(-4.0 - 9) / UNITS]}
    objects = po.in_room([bed], shapes, UNITS)
    assert objects[0]["matches"] == "B0" and objects[0]["size_m"][2] == 0.7


def test_a_phone_object_stands_against_its_nearest_wall_and_stops_short_of_the_walk():
    shapes = room()
    # a thin thing seen along the wall y = -4 (W3): 1 m wide, 1.7 m tall, 0.25 m deep as the phone saw it
    obj = {"label": "wardrobe", "min": [-12.0, -4.0, -11.5], "max": [-3.0, -4.0 + 0.25 * UNITS, -11.5 + 1.7 * UNITS]}
    box = po.candidate(shapes, obj, UNITS)
    assert box["source"] == "phone" and "W3" in box["reason"]
    assert np.allclose([box["min"][0], box["max"][0]], [-12.0, -3.0])              # its extent along the wall
    assert np.allclose([box["min"][1], box["max"][1]], [-4.0, -4.0 + 0.25 * UNITS])  # back on the wall, its own depth
    assert np.isclose(box["max"][2] - box["min"][2], 1.7 * UNITS)
    # the phone passed 0.2 m in front of the wall there: the box stops short of it
    shapes["cameras"] = [[-8.0, -4.0 + 0.2 * UNITS]]
    box = po.candidate(shapes, obj, UNITS)
    assert np.isclose(box["max"][1] - box["min"][1], (0.2 - po.CLEAR_M) * UNITS)
    # and where the phone walked right along the wall, nothing can stand
    shapes["cameras"] = [[-8.0, -4.0 + 0.1 * UNITS]]
    assert "walked" in po.candidate(shapes, obj, UNITS)


def test_a_piece_the_modelled_wall_cuts_through_keeps_its_own_depth():
    # the phone saw it 0.3 m deep, starting 0.1 m behind where the wall was modelled
    obj = {"label": "wardrobe", "min": [-12.0, -4.0 - 0.1 * UNITS, -11.5], "max": [-3.0, -4.0 + 0.2 * UNITS, 3.8]}
    box = po.candidate(room(), obj, UNITS)
    assert np.isclose(box["max"][1] - box["min"][1], 0.3 * UNITS) and np.isclose(box["min"][1], -4.0)


def test_the_usual_depth_is_assumed_only_as_far_as_the_walk_allows():
    shapes = room()
    obj = {"label": "wardrobe", "min": [-12.0, -4.0, -11.5], "max": [-3.0, -4.0 + 0.13 * UNITS, 3.8]}
    deep = po.candidate(shapes, obj, UNITS, typical_depth_m=0.55)
    assert np.isclose(deep["max"][1] - deep["min"][1], 0.55 * UNITS) and "usual one" in deep["reason"]
    shapes["cameras"] = [[-8.0, -4.0 + 0.33 * UNITS]]                     # the phone passed 0.33 m from the wall
    limited = po.candidate(shapes, obj, UNITS, typical_depth_m=0.55)
    assert np.isclose(limited["max"][1] - limited["min"][1], (0.33 - po.CLEAR_M) * UNITS)
    plain = po.candidate(room(), obj, UNITS)                              # no type given: what it measured
    assert np.isclose(plain["max"][1] - plain["min"][1], 0.2 * UNITS) and "usual one" not in plain["reason"]


def camera(position, target):
    """A view (pipeline/placement.py's) at `position` looking at `target`, image up = +z."""
    forward = np.asarray(target, float) - np.asarray(position, float)
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    R = np.stack([right, np.cross(forward, right), forward])
    return {"R": R, "t": -R @ np.asarray(position, float), "world": np.eye(3), "fx": 1000.0, "fy": 1000.0,
            "cx": 540.0, "cy": 960.0, "width": 1080, "height": 1920}


def test_the_assumed_depth_stays_out_of_frames_that_look_past_the_piece():
    shapes = room()
    obj = {"label": "wardrobe", "min": [-12.0, -4.0, -11.5], "max": [-3.0, -4.0 + 0.13 * UNITS, 3.8]}
    stand = np.array([2.0, -4.0 + 0.35 * UNITS, 1.0])                 # 0.35 m from the wall, east of the piece
    looks_at_it = camera(stand, [-7.0, -4.0, 1.0])
    looks_past = camera(stand, stand + [-np.cos(np.radians(40)), np.sin(np.radians(40)), 0.0])
    # a frame that already has the piece in view does not limit how deep it may be
    full = po.candidate(shapes, obj, UNITS, typical_depth_m=0.55, views={"frame_00003.jpg": looks_at_it})
    assert np.isclose((full["max"][1] - full["min"][1]) / UNITS, 0.55)
    # a frame that looks past it does: the depth is cut back until that frame no longer shows the piece
    views = {"frame_00003.jpg": looks_at_it, "frame_00012.jpg": looks_past}
    box = po.candidate(shapes, obj, UNITS, typical_depth_m=0.55, views=views)
    depth = (box["max"][1] - box["min"][1]) / UNITS
    assert 0.2 <= depth < 0.55 and "frames that look past it" in box["reason"]
    assert "frame_00012.jpg" not in po.frames_showing(box["min"], box["max"], views)
    assert "frame_00003.jpg" in po.frames_showing(box["min"], box["max"], views)


def test_the_frames_that_saw_an_object_or_named_its_kind_are_listed():
    jsonl = "\n".join([
        '{"frame": "frame_00003.jpg", "instances": [{"name": "wardrobe", "label": "wardrobe", "track": "o1"}]}',
        '{"frame": "frame_00004.jpg", "instances": [{"name": "curtain", "label": "wardrobe", "track": "o1"}]}',
        '{"frame": "frame_00009.jpg", "instances": [{"name": "wardrobe", "label": "wardrobe", "track": null}]}',
        '{"frame": "frame_00012.jpg", "instances": [{"name": "bed", "label": "bed", "track": "o6"}]}'])
    seen = po.frames_seen(jsonl, [{"id": "o1", "label": "wardrobe"}, {"id": "o6", "label": "bed"}])
    assert seen == {"o1": ["frame_00003.jpg", "frame_00004.jpg", "frame_00009.jpg"], "o6": ["frame_00012.jpg"]}


def test_an_object_in_the_middle_of_the_room_is_not_snapped_to_a_wall():
    obj = {"label": "table", "min": [-10.0, 6.0, -11.5], "max": [-4.0, 12.0, -5.0]}
    assert "no built wall" in po.candidate(room(), obj, UNITS)


if __name__ == "__main__":
    for name, test in sorted(globals().items()):
        if name.startswith("test_"):
            test()
            print("ok", name)
