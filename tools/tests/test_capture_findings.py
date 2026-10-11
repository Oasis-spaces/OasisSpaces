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
    a.space = space if space is not None else Path(tempfile.gettempdir()) / "no-such-space"
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
    # A run that died before COLMAP made any model counted no frames: nothing to report.
    crashed = bare_agent()
    crashed.capture_advice({"models": 0, "frames": 0, "total": 0, "fraction": 0.0}, {}, None, {})
    assert crashed.advice == []


def test_the_prompt_takes_exactly_the_evidence_the_run_measured():
    filled = CAPTURE_ADVICE_PROMPT.format(
        capture="a set of photos", recon="{}", dense='"not run"', shapes='"not run"', coverage="{}",
        walls='"not run"', sampling="{}", stopped="\"the run finished\"", limit=4)
    assert "diagnose THIS capture" in filled and "abstain" in filled
    assert "{recon}" not in filled and "{limit}" not in filled and "{coverage}" not in filled
    assert "a set of photos" in filled and "not that the camera did not move" in filled
    assert "only guessed they are left out" in filled        # metres off a guessed scale are not evidence


def test_an_unfurnished_room_is_not_told_its_furniture_was_missed():
    with tempfile.TemporaryDirectory() as tmp:
        space = Path(tmp)
        (space / "densify.json").write_text(json.dumps({"objects": {"source": "claude", "objects": [
            {"name": "window", "role": "unreliable", "build_as": None}]}}))
        a = bare_agent(space)
        a.capture_advice({"fraction": 1.0}, {}, {"objects": [], "planes": 5}, {})
        assert not any("furniture" in f["detection"] for f in a.advice)
        # A furnished room whose furniture never became a box still hears it.
        (space / "densify.json").write_text(json.dumps({"objects": {"source": "claude", "objects": [
            {"name": "bed", "role": "furniture", "build_as": "bed"}]}}))
        a = bare_agent(space)
        a.capture_advice({"fraction": 1.0}, {}, {"objects": [], "planes": 5}, {})
        assert any("No furniture" in f["detection"] for f in a.advice)


def test_a_stage_never_reached_reads_not_run_not_empty():
    """The advisor once read the empty depth and wall records of a run that
    stopped at stage 1 as "nothing in this footage was solid enough to fit a
    single wall"; a stage that never ran is marked so."""
    asked = []

    class Recorder:
        available, reason, backend, calls = True, None, "test", 0

        def ask_json(self, prompt, images=(), max_tokens=1024):
            asked.append(prompt)
            return {"findings": [], "abstain": True, "why": "thin"}

    with tempfile.TemporaryDirectory() as tmp:
        a = bare_agent(Path(tmp))
        a.name, a.source, a.do_splat, a.splat_steps = "probe", Path(tmp) / "clip.mp4", False, []
        a.gates, a.judgements, a.use_claude = {"reconstruct": {"status": "stop", "why": "no points"}}, [], True
        a.advisor = Recorder()
        a.finish(0.0, ["reconstruct"], stopped_at="reconstruct")
        assert 'depth: "not run"' in asked[0] and 'structure: "not run"' in asked[0]
        assert "a phone video" in asked[0]                   # named by its extension, though not on disk


def test_a_photo_set_is_never_offered_denser_sampling():
    asked = []

    class Recorder:
        available, reason, backend, calls = True, None, "test", 0

        def ask_json(self, prompt, images=(), max_tokens=1024):
            asked.append(prompt)
            return {"action": "accept", "why": "fine", "capture_problems": []}

    with tempfile.TemporaryDirectory() as tmp:
        photos = Path(tmp) / "photos"
        photos.mkdir()
        a = bare_agent(Path(tmp) / "space")
        a.source, a.judgements, a.advisor = photos, [], Recorder()
        a.draw_capture_map = lambda out=None: None          # no solve on disk to map
        a.capture_verdict({"models": 1, "frames": 5, "total": 20, "fraction": 0.25, "path_jump": None})
        assert "retry_denser" not in asked[0] and "the person's own photos" in asked[0]
        assert "cannot add views the capture never took" in asked[0]
        a.source = Path(tmp) / "walk.mp4"
        a.capture_verdict({"models": 1, "frames": 5, "total": 20, "fraction": 0.25, "path_jump": None})
        assert "retry_denser" in asked[1]


def test_the_capture_is_named_as_what_it_is():
    from agent import capture_phrase
    with tempfile.TemporaryDirectory() as tmp:
        assert capture_phrase(tmp) == "a set of photos"
        assert capture_phrase(Path(tmp) / "walk.MOV") == "a phone video"     # elsewhere, not on disk here
        assert capture_phrase(Path(tmp) / "mystery") == "a phone video or a set of photos"
        assert capture_phrase(None) == "a phone video or a set of photos"


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


def test_a_guessed_scale_hides_the_sizes_and_floor_shares_measured_on_it():
    """Before densify the map's scale is a guess that put one room 95 m high;
    the advisor cited such numbers though told not to, so it never sees them."""
    with tempfile.TemporaryDirectory() as tmp:
        space = Path(tmp)
        record = {"frames": 30, "path_m": 11.21, "footprint_m2": 98.4, "size_m": [124.5, 194.1],
                  "height_m": 95.2, "floor_well_seen": 0.0, "floor_never_seen": 1.0,
                  "directions_faced": "1 of 12", "directions_never_faced_deg": [0, 30],
                  "scale_from": "the phone held 1.4 m up (guessed)"}
        (space / "capture-map.json").write_text(json.dumps(record))
        read = bare_agent(space).capture_map_metrics()
        assert not set(agent_module.SCALED_MAP_FIELDS) & set(read)
        assert read["directions_faced"] == "1 of 12" and read["frames"] == 30 and "guessed" in read["left_out"]
        # Measured by densify, every number is read as it is.
        (space / "capture-map.json").write_text(json.dumps(dict(record, scale_from="densify's measurement")))
        assert bare_agent(space).capture_map_metrics()["height_m"] == 95.2


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
