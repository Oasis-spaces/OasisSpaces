"""The object list's handling of glimpses: role "elsewhere" is recorded, never searched.

    python3 tools/tests/test_object_naming.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "pipeline"))
from agent import Agent  # noqa: E402
from semantics import Vocabulary  # noqa: E402

LISTED = [
    {"name": "sofa", "role": "furniture", "build_as": "seat"},
    {"name": "bathtub", "role": "elsewhere", "build_as": None},
    {"name": "door", "role": "fixture", "build_as": None},
    {"name": "toilet", "role": "elsewhere", "build_as": None},
]


def test_an_elsewhere_name_never_reaches_the_vocabulary():
    vocabulary = Vocabulary([o for o in LISTED if o.get("role") != "elsewhere"], "claude")
    assert vocabulary.names == ["sofa", "door"]
    # And the role itself is unknown to the vocabulary, so a stray one is
    # dropped rather than searched.
    assert Vocabulary(LISTED, "claude").names == ["sofa", "door"]


def test_the_review_leaves_an_elsewhere_add_out_and_says_so():
    vocabulary = Vocabulary([{"name": "sofa", "role": "furniture", "build_as": "seat"}], "claude")
    revised, changes = Agent.revise_objects(
        None, vocabulary,
        {"add": [{"name": "bathtub", "role": "elsewhere"},
                 {"name": "curtain", "role": "hanging"}]})
    assert "curtain" in revised.names and "bathtub" not in revised.names
    assert "left out 'bathtub' (elsewhere, not in this room)" in changes
    assert any(c.startswith("added 'curtain'") for c in changes)
    # An elsewhere note alone does not change the list, so densify's rerun
    # is not triggered by it.
    same, changes = Agent.revise_objects(
        None, vocabulary, {"add": [{"name": "bathtub", "role": "elsewhere"}]})
    assert same.names == vocabulary.names and changes == []


def test_an_empty_room_keeps_its_empty_list_not_the_default_furniture():
    import json
    import tempfile

    from semantics import planned_vocabulary

    assert Vocabulary.from_json({"source": "claude", "objects": []}).names == []
    assert Vocabulary.from_json({"objects": []}).source == "default"     # nobody looked: the default list
    assert Vocabulary.from_json(None).source == "default"
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "objects.json").write_text(json.dumps({"source": "claude", "objects": []}))
        searched = planned_vocabulary(Path(tmp)).names
        assert "bed" not in searched and "sofa" not in searched
        assert "window" in searched and "mirror" in searched                # depth is wrong on these anyway


class Answers:
    available, reason, calls = True, None, 0

    def __init__(self, reply):
        self.reply = reply

    def ask_json(self, *args, **kwargs):
        return self.reply


def naming_agent(tmp: Path, reply) -> Agent:
    a = Agent.__new__(Agent)
    a.space, a.chosen, a.judgements = tmp, {}, []
    (tmp / "workspace" / "images").mkdir(parents=True)
    (tmp / "workspace" / "images" / "frame_00001.jpg").write_bytes(b"")
    a.advisor = Answers(reply)
    return a


def test_naming_an_unfurnished_room_writes_what_was_named():
    import json
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        a = naming_agent(Path(tmp), {"objects": [{"name": "window", "role": "unreliable", "build_as": None},
                                                 {"name": "door", "role": "fixture", "build_as": None}],
                                     "room": "an empty room"})
        a.name_objects()
        written = json.loads((Path(tmp) / "objects.json").read_text())
        assert [o["name"] for o in written["objects"]] == ["window", "door"]
    with tempfile.TemporaryDirectory() as tmp:
        a = naming_agent(Path(tmp), {"objects": "a sofa and a bed"})          # junk: no list
        a.name_objects()
        assert not (Path(tmp) / "objects.json").exists()                     # the default list is used


def test_an_unfurnished_room_passes_stage_3_but_unbuilt_furniture_still_warns():
    import json
    import tempfile

    wall = {"kind": "wall", "build": True}
    with tempfile.TemporaryDirectory() as tmp:
        space = Path(tmp)
        (space / "shapes.json").write_text(json.dumps({"planes": [wall, wall, {"kind": "floor"}], "boxes": []}))
        a = Agent.__new__(Agent)
        a.space, a.judgements, a.use_claude = space, [], False
        (space / "densify.json").write_text(json.dumps({"objects": {"source": "claude", "objects": []}}))
        status, why = a.gate("shapes")
        assert status == "pass" and "no furniture was named" in why
        (space / "densify.json").write_text(json.dumps({"objects": {"source": "claude", "objects": [
            {"name": "sofa", "role": "furniture", "build_as": "seat"}]}}))
        assert a.gate("shapes") == ("warn", "no furniture was recognised")


if __name__ == "__main__":
    for name, test in sorted(globals().items()):
        if name.startswith("test_"):
            test()
            print("ok", name)
