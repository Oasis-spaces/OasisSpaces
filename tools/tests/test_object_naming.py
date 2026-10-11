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


if __name__ == "__main__":
    for name, test in sorted(globals().items()):
        if name.startswith("test_"):
            test()
            print("ok", name)
