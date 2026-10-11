"""Findings on the capture provided: detections with evidence, not tips.

    python3 tools/tests/test_capture_findings.py
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "pipeline"))
import agent as agent_module  # noqa: E402
from agent import Agent, CAPTURE_ADVICE_PROMPT  # noqa: E402


def bare_agent(space: Path | None = None) -> Agent:
    a = Agent.__new__(Agent)
    a.advice = []
    a.stages = []
    if space is not None:
        a.space = space
    return a


def test_every_finding_carries_its_evidence_and_the_input_to_take():
    a = bare_agent()
    a.capture_advice({"rejected_global": True, "models": 2, "fraction": 0.6,
                      "frames": 12, "total": 20},
                     {"scale_spread": 0.2, "points_dropped_unreliable": 9259},
                     {"objects": []}, {"wall_rms_pct": 3.0})
    assert all(f["detection"] and f["evidence"] for f in a.advice)
    placed = next(f for f in a.advice if "frames were placed" in f["detection"])
    assert "12 of 20" in placed["evidence"] and placed["take"]
    # No rejected model, full placement, quiet depth: nothing detected.
    quiet = bare_agent()
    quiet.capture_advice({"rejected_global": False, "fraction": 1.0}, {}, None, {})
    assert quiet.advice == []


def test_the_prompt_takes_exactly_the_evidence_the_run_measured():
    filled = CAPTURE_ADVICE_PROMPT.format(
        recon="{}", dense="{}", shapes="{}", capture="{}", walls="{}",
        sampling="{}", stopped="\"the run finished\"", limit=4)
    assert "diagnose THIS capture" in filled and "abstain" in filled
    assert "{recon}" not in filled and "{limit}" not in filled


def test_the_last_measured_wall_table_and_the_sampling_are_read_back():
    with tempfile.TemporaryDirectory() as tmp:
        space = Path(tmp)
        a = bare_agent(space)
        a.stages = [{"stage": "shapes", "metrics": {"walls": {"W0": {"support": 0.9}}}},
                    {"stage": "shapes", "metrics": {"walls": {"W0": {"support": 0.2}}}},
                    {"stage": "splat", "metrics": {}}]
        assert a.measured_walls() == {"W0": {"support": 0.2}}      # the last one wins
        assert bare_agent(space).sampling_metrics() == {}          # nothing extracted yet
        grid = space / "workspace" / "video-grid"
        grid.mkdir(parents=True)
        (grid / "frames.json").write_text(json.dumps(
            {"fps": 6.0, "frames": {"frame_00001.jpg": 0, "frame_00002.jpg": 3}}))
        sampling = a.sampling_metrics()
        assert sampling["frames_kept"] == 2 and sampling["candidates_fps"] == 6.0
        assert "not the person" in sampling["note"]
        assert a.capture_map_metrics() == {}                       # no capture map on disk


def test_a_malformed_advisor_reply_never_loses_the_report():
    from agent import findings_of
    good = {"detection": "d", "evidence": "e", "take": "t"}
    assert findings_of({"findings": [good, {"detection": "x"}, "junk", 3]}) == [good]
    assert findings_of({"findings": good}) == [good]       # a single finding sent bare
    assert findings_of({"findings": {"nested": "junk"}}) == []
    assert findings_of({"findings": "words"}) == []
    assert findings_of(None) == [] and findings_of({}) == []


def test_the_default_retry_is_denser_sampling_on_a_thin_video_solve():
    from agent import default_retry
    action, why = default_retry({"total": 12, "fraction": 0.17}, video=True)
    assert action == "retry_denser" and "12 frames" in why
    action, _ = default_retry({"total": 120, "fraction": 0.4}, video=True)
    assert action == "retry_aliked"
    action, _ = default_retry({"total": 12, "fraction": 0.17}, video=False)
    assert action == "retry_aliked"                 # photos cannot be resampled


if __name__ == "__main__":
    for name, test in sorted(globals().items()):
        if name.startswith("test_"):
            test()
            print("ok", name)
