"""The claude CLI's JSON output, read even when a notice follows the answer.

    python3 tools/tests/test_advisor_parse.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "pipeline"))
from advisor import first_json  # noqa: E402

ANSWER = {"type": "result", "is_error": False, "result": "{\"plausible\": true}", "duration_ms": 5000}
LIMIT = {"type": "result", "is_error": True, "result": "You've hit your session limit · resets 4:30pm"}


def test_one_document_is_read_as_before():
    assert first_json(json.dumps(ANSWER)) == ANSWER
    assert first_json("") == {} and first_json("   \n") == {}
    assert first_json("[1, 2]") == {}                                  # not an object: nothing to read


def test_an_answer_followed_by_a_notice_is_still_the_answer():
    assert first_json(json.dumps(ANSWER) + "\n" + json.dumps(LIMIT))["result"] == ANSWER["result"]


def test_an_empty_answer_followed_by_the_limit_notice_reports_the_limit():
    empty = dict(ANSWER, result="")
    out = first_json(json.dumps(empty) + "\n" + json.dumps(LIMIT) + "\n")
    assert out["is_error"] and "session limit" in out["result"]
    # a trailing fragment that is not JSON does not hide the first document
    assert first_json(json.dumps(empty) + "\nnot json")["result"] == ""


if __name__ == "__main__":
    for name, test in sorted(globals().items()):
        if name.startswith("test_"):
            test()
            print("ok", name)
