"""The structure review driven by measured problems: what it is shown, and its
two bounded actions (resize a side, split a stacked box), each kept only when
the room's score goes up.

    python3 tools/tests/test_measured_review.py      (or pytest tools/tests)
"""

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(ROOT / "tools"))
import agent as ag  # noqa: E402

UNIT = 2.0


def room(boxes, cameras):
    return {"planes": [], "boxes": boxes, "cameras": cameras,
            "room_level": {"floor_z": 0.0, "height": 2.6 * UNIT}}


def box(lo, hi, label="wardrobe", detected="wardrobe"):
    return {"min": [v * UNIT for v in lo], "max": [v * UNIT for v in hi], "label": label,
            "detected": detected, "points": 1000, "build": True}


def agent(space: Path, scorer):
    a = object.__new__(ag.Agent)
    a.space = space
    a.densify_metrics = lambda: {"colmap_units_per_metre": UNIT}
    a.measure_room = scorer
    a.phone_matches = lambda: {}
    return a


def test_the_walk_through_a_wardrobe_is_named_with_how_far_each_side_must_move():
    with tempfile.TemporaryDirectory() as tmp:
        space = Path(tmp)
        (space / "objects.json").write_text(json.dumps({"objects": [{"name": "wardrobe", "role": "storage"}]}))
        shapes = room([box((0, 0, 0), (2, 1.2, 2.4))], [[1.0 * UNIT, 1.0 * UNIT], [1.5 * UNIT, 0.9 * UNIT]])
        problems = agent(space, None).measured_problems(shapes, {"pieces": {"desk": {"box": None, "frames": 6}}})
        walk = [p for p in problems if p["problem"] == "walk"][0]
        assert walk["box"] == "B0" and walk["cameras_inside"] == 2
        assert walk["clear_by_m"]["top"] == 0.3, walk            # 1.2 - 0.9: the front comes in 0.3 m
        assert {"problem": "no box", "object": "desk", "frames": 6} in problems


def test_a_resize_that_raises_the_score_is_kept_and_one_that_lowers_it_is_undone():
    with tempfile.TemporaryDirectory() as tmp:
        shapes = room([box((0, 0, 0), (2, 1.2, 2.4)), box((3, 0, 0), (4, 1, 1), "bed", "bed")], [])
        # the score rises only while the wardrobe gets shallower
        a = agent(Path(tmp), lambda s: {"score": -s["boxes"][0]["max"][1] - s["boxes"][1]["max"][0] / 10})
        applied = a.apply_structure_review(shapes, {"resize_boxes": [
            {"id": "B0", "side": "top", "by_m": 0.5, "why": "the front ran over the walk"},
            {"id": "B1", "side": "right", "by_m": -0.4, "why": "longer"},
            {"id": "B0", "side": "left", "by_m": 5.0, "why": "far too much"}]})
        assert abs(shapes["boxes"][0]["max"][1] - 0.7 * UNIT) < 1e-9, applied
        assert shapes["boxes"][1]["max"][0] == 4 * UNIT, applied                      # rolled back
        assert any(t.startswith("rolled back resizing B1") for t in applied), applied
        # a resize is bounded: at most RESIZE_MAX_SHARE of the side comes off
        assert abs(shapes["boxes"][0]["min"][0] - ag.RESIZE_MAX_SHARE * 2 * UNIT) < 1e-9, applied


def test_a_loft_cabinet_merged_with_its_wardrobe_is_split_and_stays_on_the_wall():
    with tempfile.TemporaryDirectory() as tmp:
        shapes = room([box((0, 0, 0), (2, 0.6, 2.6))], [])
        a = agent(Path(tmp), lambda s: {"score": len([b for b in s["boxes"] if b.get("build", True)])})
        applied = a.apply_structure_review(shapes, {"split_boxes": [
            {"id": "B0", "at_height_m": 2.0, "lower": "wardrobe", "upper": "wardrobe", "why": "loft above"},
            {"id": "B0", "at_height_m": 0.1, "lower": "block", "upper": "wardrobe", "why": "too low"}]})
        lower, upper = shapes["boxes"]
        assert lower["max"][2] == upper["min"][2] == 2.0 * UNIT, applied
        assert upper["mounted"] and upper["source"] == "split" and lower.get("mounted") is None
        assert any("leaves a part under" in t for t in applied), applied


def test_every_question_carries_the_record_and_folds_its_answer_back():
    with tempfile.TemporaryDirectory() as tmp:
        a = agent(Path(tmp), None)
        asked = []

        class Advisor:
            def ask_json(self, prompt, images, max_tokens):
                asked.append(prompt)
                return {"keep": True, "record": {"objects": [{"name": "desk", "boxes": ["B1"]}],
                                                  "doubts": ["a second headboard"]}}
        a.advisor = Advisor()
        a.remember("densify", "removed 'television' from the object list")
        assert a.ask_json("Is this a room?")["keep"]
        assert "removed 'television'" in asked[0] and asked[0].endswith("Is this a room?")
        a.ask_json("And now?")
        assert '"desk"' in asked[1] and "a second headboard" in asked[1]
        # the record says B1 is the desk: the box gets that identity, and the score matches it
        shapes = room([box((0, 0, 0), (1, 1, 2)), box((2, 0, 0), (3, 1, 0.8), "table", "chest of drawers")], [])
        assert a.apply_identities(shapes) == ["B1 is the desk"]
        import room_score
        from semantics import Vocabulary
        vocabulary = Vocabulary([{"name": "desk", "role": "furniture", "build_as": "table"},
                                 {"name": "chest of drawers", "role": "storage", "build_as": "wardrobe"}], "test")
        assert [i for i, _ in room_score.boxes_of(shapes, "desk", vocabulary)] == [1]
        assert 1 not in [i for i, _ in room_score.boxes_of(shapes, "chest of drawers", vocabulary)]   # no longer the detector's name
        a.forget_boxes()
        assert a.record()["objects"] == [{"name": "desk"}]


def test_a_furniture_front_dropped_in_front_of_a_built_wall_is_not_a_missing_wall():
    import room_score
    wall = lambda x, half=1.5, build=True: {"kind": "wall", "axis_a": [0.0, 1.0, 0.0], "center": [x * UNIT, 0.0, 1.0],  # noqa: E731
                                            "half_a": half * UNIT, "build": build}
    shapes = {"planes": [wall(-1.76, build=False), wall(-2.13), wall(-3.6, build=False)]}
    assert room_score.stood_in_for(shapes, 0, UNIT)          # 0.37 m in front of the real wall
    assert not room_score.stood_in_for(shapes, 2, UNIT)      # 1.5 m away: nothing stands in for it


def test_the_review_can_hold_a_box_at_its_height_or_its_place():
    import classify_shapes as cs
    with tempfile.TemporaryDirectory() as tmp:
        shapes = room([box((0, 0, 0.7), (1.2, 0.5, 0.8), "table", "desk"), box((1, 1, 0), (2.5, 3, 0.5), "bed", "bed")], [])
        a = agent(Path(tmp), lambda s: {"score": 0.0})
        a.prefinish_boxes = [box((0, 0, 0.7), (1.2, 0.5, 0.8), "table", "desk"), box((1, 1, 0), (2.5, 3, 0.5), "bed", "bed")]
        shapes["boxes"][0]["min"][2] = 0.0                          # the finish stood the desk on the floor
        shapes["boxes"][1]["min"][1] += 0.4 * UNIT                  # and moved the bed to the wall
        applied = a.apply_structure_review(shapes, {"hold_boxes": [
            {"id": "B0", "hold": "height", "why": "wall-mounted desk"},
            {"id": "B1", "hold": "place", "why": "the bed stands away from the wall"},
            {"id": "B1", "hold": "sideways", "why": "?"}]})
        assert shapes["boxes"][0]["mounted"] and shapes["boxes"][1]["hold_place"], applied
        assert shapes["boxes"][0]["min"][2] == 0.7 * UNIT and shapes["boxes"][1]["min"][1] == 1 * UNIT   # put back
        assert any(t.startswith("ignored holding B1") for t in applied)
        # the finish faces the held bed but does not move it up to the wall
        roomdict = {"center": [2 * UNIT, 2 * UNIT], "axis_u": [1.0, 0.0], "axis_v": [0.0, 1.0],
                    "half_u": 2 * UNIT, "half_v": 2 * UNIT}
        bed = shapes["boxes"][1]
        before = [list(bed["min"]), list(bed["max"])]
        cs.orient_furniture([bed], roomdict, lambda d: f"{d:.1f}", UNIT)
        assert [bed["min"], bed["max"]] == before and bed.get("facing")


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print("ok", name)
