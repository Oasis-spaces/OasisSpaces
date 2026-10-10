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
    stub.phone_matches = lambda: {}
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
    stub.phone_matches = lambda: {}
    shapes = room()
    applied = agent.Agent.apply_structure_review(
        stub, shapes, {"add_boxes": [{"label": "bed", "against": "W0", "width_m": 1.4, "depth_m": 2, "height_m": 0.5}]})
    assert shapes["boxes"] == [] and "not measured" in applied[0]


def test_dropping_every_wardrobe_box_asks_for_the_wardrobe_but_a_built_bed_does_not():
    shapes = room()
    shapes["boxes"] = [{"detected": "wardrobe", "label": "wardrobe", "build": False, "min": [0] * 3, "max": [1] * 3},
                       {"detected": "wardrobe", "label": "wardrobe", "build": False, "min": [0] * 3, "max": [1] * 3},
                       {"detected": "bed", "label": "bed", "build": False, "min": [0] * 3, "max": [1] * 3},
                       {"detected": "bed", "label": "bed", "build": True, "min": [0] * 3, "max": [1] * 3}]
    verdict = {"drop_boxes": [{"id": "B0", "why": "the room door"}, {"id": "B1", "why": "its open door leaf"},
                              {"id": "B2", "why": "a shelf"}]}
    asked = agent.dropped_pieces(shapes, verdict, ["dropped B0", "dropped B1", "dropped B2"])
    assert asked == {"wardrobe": [("B0", "the room door"), ("B1", "its open door leaf")]}
    assert agent.dropped_pieces(shapes, verdict, ["kept B0: never dropped"]) == {}      # only what was really dropped


class Answers:
    """A stand-in advisor that answers each question from a list."""
    available = True

    def __init__(self, *answers):
        self.answers, self.asked = list(answers), []

    def ask_json(self, prompt, images, max_tokens=0):
        self.asked.append(prompt)
        return self.answers.pop(0) if self.answers else None


def stub(advisor):
    import tempfile
    stub = agent.Agent.__new__(agent.Agent)
    stub.space = Path(tempfile.mkdtemp())
    stub.advisor = advisor
    stub.densify_metrics = lambda: {"colmap_units_per_metre": UNITS}
    stub.picked_frames = lambda purpose, count: None
    stub.room_frames = lambda count=3: []
    stub.chosen = None
    stub.place_by_masks = lambda shapes, label: ([], 0)
    stub._phone_objects = []                       # tests never launch the phone simulator
    stub._all_views = {}                           # nor read a solve
    return stub


def test_the_answer_places_the_piece_and_a_refused_spot_gets_one_more_try():
    shapes = room()
    shapes["cameras"] = [[-15.0, 0.0]]
    on_the_walk = {"against": "W0", "from_corner_with": "W3", "offset_m": 0.3, "width_m": 1.0,
                   "depth_m": 0.6, "height_m": 2.0, "why": "next to its open door"}
    in_the_corner = {**on_the_walk, "against": "W1", "from_corner_with": "W0", "offset_m": 0.0}
    advisor = Answers({"add": on_the_walk}, {"add": in_the_corner})
    applied = agent.Agent.ask_where_piece_stands(stub(advisor), shapes, "wardrobe", [("B1", "its open door leaf")])
    assert len(advisor.asked) == 2 and "refused" in advisor.asked[1]
    assert applied[0].startswith("ignored a wardrobe against W0") and "phone stood" in applied[0]
    assert applied[1].startswith("asked where the wardrobe stands: added B0 against W1, from its corner with W0")
    assert shapes["boxes"][0]["label"] == "wardrobe" and shapes["boxes"][0]["source"] == "claude"


def test_masks_place_a_dropped_piece_at_once_and_anything_else_waits_for_the_finished_room():
    box = {"min": [-20.0, 0.0, -11.5], "max": [-14.6, 9.0, 6.5], "label": "wardrobe", "build": True,
           "source": "masks", "reason": "placed by its masks in 4 keyframe(s), score 0.46, against W0"}
    advisor = Answers({"add": {"against": "W1", "width_m": 1, "depth_m": 0.6, "height_m": 2}})
    agent_ = stub(advisor)
    agent_.place_by_masks = lambda shapes, label: ([box], 4)
    shapes = room()
    applied = agent.Agent.place_dropped_piece(agent_, shapes, "wardrobe", [("B1", "its door leaf")])
    assert shapes["boxes"] == [box] and applied == ["added B0 wardrobe: " + box["reason"]]
    assert not getattr(agent_, "_pending_pieces", None)
    agent_.place_by_masks = lambda shapes, label: ([], 5)                    # masks exist but fit no box
    shapes = room()
    assert agent.Agent.place_dropped_piece(agent_, shapes, "wardrobe", [("B1", "its door leaf")]) == []
    assert shapes["boxes"] == [] and agent_._pending_pieces["wardrobe"]["evidence_frames"] == 5
    assert advisor.asked == []                                               # nobody is asked mid-review


def settled(pending, phone=(), answers=()):
    """settle_pieces on a room on disk; returns (boxes, what was recorded, what Claude was asked)."""
    import json
    advisor = Answers(*answers)
    agent_ = stub(advisor)
    (agent_.space / "shapes.json").write_text(json.dumps(room()))
    agent_._pending_pieces = pending
    agent_.phone_objects = lambda: list(phone)
    recorded, ran = [], []
    agent_.judged = lambda stage, verdict, note: recorded.append(note)
    agent_.run = lambda command, stage, note: ran.append(note) or (True, "")
    agent.Agent.settle_pieces(agent_)
    return json.loads((agent_.space / "shapes.json").read_text())["boxes"], recorded, advisor.asked, ran


def test_on_the_finished_room_the_phone_is_heard_first_and_claude_only_where_nothing_saw_the_piece():
    wardrobe = {"label": "wardrobe", "min": [-9.0, -4.0, -11.5], "max": [-3.0, -4.0 + 0.13 * UNITS, -11.5 + 1.7 * UNITS],
                "size_m": [0.67, 0.13, 1.7], "matches": None}
    pending = {"wardrobe": {"dropped": [("B1", "its door leaf")], "evidence_frames": 5}}
    # the phone placed one: it is added, the room is finished again, nobody else is asked
    boxes, recorded, asked, ran = settled(dict(pending), phone=[wardrobe])
    assert len(boxes) == 1 and boxes[0]["source"] == "phone" and asked == [] and len(ran) == 1
    assert recorded[0].startswith("added B0 wardrobe")
    # the phone placed none and the keyframes' masks fit no box: nothing is guessed
    boxes, recorded, asked, ran = settled(dict(pending))
    assert boxes == [] and "support no box" in recorded[0] and asked == [] and ran == []
    # no keyframe saw it and the phone placed none: Claude is asked
    answer = {"add": {"against": "W1", "from_corner_with": "W0", "offset_m": 0.0, "width_m": 1.0, "depth_m": 0.6,
                      "height_m": 2.0, "why": "by the door"}}
    boxes, recorded, asked, ran = settled({"wardrobe": {"dropped": [("B1", "x")], "evidence_frames": 0}}, answers=[answer])
    assert len(boxes) == 1 and boxes[0]["source"] == "claude" and len(asked) == 1 and len(ran) == 1
    # nothing pending and nothing from the phone: the room is left as it is
    boxes, recorded, asked, ran = settled({})
    assert boxes == [] and recorded == [] and ran == []


def test_no_whole_piece_means_nothing_is_added():
    shapes = room()
    applied = agent.Agent.ask_where_piece_stands(stub(Answers({"none": True, "why": "only a door"})),
                                                 shapes, "wardrobe", [("B0", "the room door")])
    assert shapes["boxes"] == [] and applied == ["no wardrobe added: only a door"]


def test_a_misplacement_claim_the_masks_contradict_becomes_minor():
    agent_ = stub(Answers())
    agent_.box_mask_score = lambda label: {"bed": (0.65, 8), "wardrobe": (0.27, 2)}.get(label)
    agent_.phone_corroborates = lambda label: None
    verdict = {"plausible": False, "problems": [
        {"what": "misplaced: bed, set far back in the room", "severity": "structural"},
        {"what": "misplaced: wardrobe, standing in the doorway", "severity": "structural"},
        {"what": "misplaced: lamp, floating", "severity": "structural"},            # no masks: untouched
        {"what": "pillow off-centre", "severity": "minor"}]}
    notes = agent.Agent.overrule_misplacements(agent_, verdict)
    assert notes == ["bed stays, score 0.65 in 8 frames"]
    assert [p["severity"] for p in verdict["problems"]] == ["minor", "structural", "structural", "minor"]
    assert "overruled" in verdict["problems"][0]["what"] and verdict["plausible"] is False
    only_bed = {"plausible": False, "problems": [{"what": "misplaced: bed, too far", "severity": "structural"}]}
    agent.Agent.overrule_misplacements(agent_, only_bed)
    assert only_bed["plausible"] is True                                             # nothing structural is left


def test_the_phones_piece_grows_a_measured_fragment_or_is_added_and_non_furniture_is_ignored():
    def phone(label, x0, x1, depth):
        return {"label": label, "min": [x0, -4.0, -11.5], "max": [x1, -4.0 + depth * UNITS, -11.5 + 1.7 * UNITS],
                "size_m": [round((x1 - x0) / UNITS, 2), depth, 1.7], "matches": None}
    agent_ = stub(Answers())
    low = phone("cabinet", 1.0, 4.0, 0.3)
    low["size_m"][2] = 0.6                                                           # a bedside cabinet: not taken on
    agreed = phone("wardrobe", -18.0, -15.0, 0.5)
    agreed["matches"] = "B7"                                                         # coincides with a measured box: left alone
    agent_._phone_objects = [phone("wardrobe", -9.0, -3.0, 0.13), phone("fridge", 0.0, 4.0, 0.3), low, agreed]
    # the review kept the open door leaf as the wardrobe: 0.2 m deep, beside and overlapping the phone's
    shapes = room()
    shapes["boxes"] = [{"label": "wardrobe", "build": True, "min": [-13.0, -4.0, -11.5], "max": [-6.0, -4.0 + 0.2 * UNITS, 3.5]}]
    notes = agent.Agent.phone_second_opinion(agent_, shapes)
    assert notes == ["grew B0 wardrobe to the phone's wardrobe"] and len(shapes["boxes"]) == 1
    grown = shapes["boxes"][0]
    assert grown["min"][0] == -13.0 and grown["max"][0] == -3.0                       # both extents along the wall
    assert abs((grown["max"][1] - grown["min"][1]) - 0.55 * UNITS) < 1e-6             # the usual depth, no walk in the way
    # the review dropped it: the phone's piece is added
    shapes = room()
    notes = agent.Agent.phone_second_opinion(agent_, shapes)
    assert len(shapes["boxes"]) == 1 and shapes["boxes"][0]["source"] == "phone" and notes[0].startswith("added B0 wardrobe")


def test_a_piece_the_phone_also_places_is_not_misplaced_and_is_not_dropped():
    agent_ = stub(Answers())
    agent_.box_mask_score = lambda label: (0.12, 3)                         # its masks are too mixed to say
    agent_.phone_corroborates = lambda label: "wardrobe 1.18 x 0.6 x 2.46 m on B0" if label == "wardrobe" else None
    verdict = {"plausible": False, "problems": [
        {"what": "misplaced: wardrobe, a full-height box where the frame shows a wall cupboard", "severity": "structural"},
        {"what": "misplaced: desk, floating", "severity": "structural"}]}
    notes = agent.Agent.overrule_misplacements(agent_, verdict)
    assert notes == ["wardrobe stays, the phone places a wardrobe 1.18 x 0.6 x 2.46 m on B0"]
    assert [p["severity"] for p in verdict["problems"]] == ["minor", "structural"]
    # and a review may not drop the measured box the phone vouches for, but may drop a guess
    shapes = room()
    shapes["boxes"] = [{"label": "wardrobe", "detected": "wardrobe", "source": "front", "build": True, "points": 900,
                        "min": [0] * 3, "max": [1] * 3},
                       {"label": "wardrobe", "detected": "wardrobe", "source": "claude", "build": True, "points": 0,
                        "min": [0] * 3, "max": [1] * 3},
                       {"label": "bed", "detected": "bed", "source": "detected", "build": True, "points": 5000,
                        "min": [0] * 3, "max": [1] * 3}]
    agent_.phone_matches = lambda: {"B0": {"label": "wardrobe"}, "B1": {"label": "wardrobe"}}
    applied = agent.Agent.apply_structure_review(agent_, shapes, {"drop_boxes": [{"id": "B0", "why": "x"}, {"id": "B1", "why": "y"}]})
    assert applied == ["kept B0: the phone's own detector places a wardrobe on the same spot", "dropped B1"]
    assert shapes["boxes"][0]["build"] and not shapes["boxes"][1]["build"]


def test_an_edit_that_lowers_the_measured_room_score_is_rolled_back():
    agent_ = stub(Answers())
    agent_.phone_matches = lambda: {}
    # a measurement that likes built boxes: each built one is worth 0.3
    agent_.measure_room = lambda shapes=None: {"score": round(0.3 * sum(1 for b in shapes["boxes"] if b.get("build", True)), 3)}
    shapes = room()
    shapes["boxes"] = [{"label": "bed", "detected": "bed", "source": "detected", "build": True, "points": 5000,
                        "min": [-16.0, -4.0, -11.5], "max": [-7.0, 5.0, -6.0]},
                       {"label": "wardrobe", "detected": "wardrobe", "source": "claude", "build": True, "points": 0,
                        "min": [-20.0, 10.0, -11.5], "max": [-14.0, 15.0, 7.0]}]
    item = {"label": "table", "against": "W1", "from_corner_with": "W2", "offset_m": 0.3,
            "width_m": 1.2, "depth_m": 0.6, "height_m": 0.75, "why": "the desk"}
    applied = agent.Agent.apply_structure_review(
        agent_, shapes, {"drop_boxes": [{"id": "B1", "why": "a guess"}], "add_boxes": [item]})
    assert applied == ["dropped B1", "rolled back drop_boxes: it lowered the measured room score from 0.60 to 0.30",
                       "added B2 table against W1, from its corner with W2"]
    assert all(b["build"] for b in shapes["boxes"]) and len(shapes["boxes"]) == 3   # the drop undone, the add kept
    # without a measurement, the review's word stands
    agent_.measure_room = lambda shapes=None: (_ for _ in ()).throw(FileNotFoundError("no densify.json"))
    shapes["boxes"] = shapes["boxes"][:2]
    applied = agent.Agent.apply_structure_review(agent_, shapes, {"drop_boxes": [{"id": "B1", "why": "a guess"}]})
    assert applied == ["dropped B1"] and not shapes["boxes"][1]["build"]


if __name__ == "__main__":
    for name, test in sorted(globals().items()):
        if name.startswith("test_"):
            test()
            print("ok", name)
