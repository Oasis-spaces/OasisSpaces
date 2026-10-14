"""tools/check_phone_capture.py on synthetic recordings: what the app writes, read back right.

    python3 tools/tests/test_check_phone_capture.py      (or pytest tools/tests)
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import check_phone_capture as cpc  # noqa: E402


def column_major(m: np.ndarray) -> list:
    return [float(v) for v in m.T.reshape(-1)]


def pose(x: float, yaw_deg: float = 0.0) -> np.ndarray:
    a = np.radians(yaw_deg)
    m = np.eye(4)
    m[:3, :3] = [[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]]
    m[0, 3] = x
    return m


def recording(frames=90, jump_at=None, exposure_drift=False, imu=True, moved=(), planes=True, anchors=True):
    """frames at 30 fps walking 1 cm a frame along x, as Recorder writes them."""
    records, keyframes = [], []
    rng = np.random.default_rng(0)
    x = 0.0
    for k in range(frames):
        x += 0.5 if k == jump_at else 0.01
        t = 100.0 + k / 30
        record = {"t": t, "frame": k, "transform": column_major(pose(x)), "intrinsics": [1500, 1500, 960, 720],
                  "size": [1920, 1440], "tracking": "normal",
                  "exposure": 0.01 * (1 + (0.1 if exposure_drift and k > frames // 2 else 0))}
        if k == jump_at:
            record["jump"] = True
        if imu:
            # three or four samples at 100 Hz before this frame, about 1 g of gravity on y
            start = t - 1 / 30
            record["imu"] = [[s, *rng.normal(0, 0.01, 3), 0.0, -1.0 + rng.normal(0, 0.01), 0.0]
                             for s in np.arange(start + 0.01, t + 1e-9, 0.01)]
        records.append(record)
        if k % cpc.ANCHOR_EVERY == 0:
            placed = pose(x)
            final = pose(x + 0.03, 1.0) if k in moved else placed
            keyframes.append({"frame": k, "placed": column_major(placed), "final": column_major(final),
                              "updates": 2 if k in moved else 0})
    capture = {"device": "iPhone14,5", "system": "18.6", "app": "Oasis Capture 0.1", "framesDropped": 0}
    if anchors:
        capture["keyframeAnchors"] = keyframes
    if planes:
        capture["planeClasses"] = [
            {"alignment": "vertical", "classification": "wall", "extent": [3.0, 2.0], "removed": False},
            {"alignment": "vertical", "classification": "none (undetermined)", "extent": [1.0, 2.0], "removed": False},
            {"alignment": "horizontal", "classification": "floor", "extent": [4.0, 3.0], "removed": False},
            {"alignment": "vertical", "classification": "wall", "extent": [9.0, 9.0], "removed": True},
        ]
    return records, capture


def run(records, capture) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        (folder / "frames.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
        (folder / "capture.json").write_text(json.dumps(capture))
        return cpc.check(folder)


def test_a_clean_recording_passes_every_check():
    report = run(*recording())
    for name in ("frames", "exposure", "imu", "jumps", "keyframe_anchors", "plane_classes"):
        assert report[name]["status"] == "ok", (name, report[name])
    assert report["frames"]["fps"] == 30.0
    assert 90 <= report["imu"]["rate_hz"] <= 110
    assert report["keyframe_anchors"]["anchors"] == report["keyframe_anchors"]["expected"] == 6
    assert report["keyframe_anchors"]["moved"] == 0


def test_moved_anchors_report_how_far_arkit_corrected_them():
    report = run(*recording(moved=(15, 45)))["keyframe_anchors"]
    assert report["moved"] == 2 and report["updates_total"] == 4
    assert abs(report["shift_cm_max"] - 3.0) < 1e-6
    assert abs(report["turn_deg_max"] - 1.0) < 1e-3


def test_a_jump_is_flagged_and_a_missing_flag_is_caught():
    records, capture = recording(jump_at=40)
    assert run(records, capture)["jumps"]["status"] == "ok"
    assert run(records, capture)["jumps"]["jumps"] == [40]
    del records[40]["jump"]
    assert run(records, capture)["jumps"]["status"] == "fail"


def test_exposure_drift_is_a_warning():
    report = run(*recording(exposure_drift=True))["exposure"]
    assert report["status"] == "warn" and report["share_off"] > 0.4


def test_an_anchor_placed_away_from_its_frame_fails():
    records, capture = recording()
    capture["keyframeAnchors"][2]["placed"] = column_major(pose(5.0))
    assert run(records, capture)["keyframe_anchors"]["status"] == "fail"


def test_out_of_order_imu_fails_and_old_recordings_report_missing_fields():
    records, capture = recording()
    records[10]["imu"] = list(reversed(records[10]["imu"]))
    assert run(records, capture)["imu"]["status"] == "fail"
    old = run(*recording(imu=False, planes=False, anchors=False))
    assert old["imu"]["status"] == old["plane_classes"]["status"] == old["keyframe_anchors"]["status"] == "fail"


def test_plane_hit_rate_counts_live_planes_by_area():
    report = run(*recording())["plane_classes"]
    assert report["planes"] == 3 and report["removed"] == 1
    assert report["vertical_classified_share_by_area"] == 0.75      # 6 m2 of wall of 8 m2 vertical
    assert report["by_class"]["vertical wall"] == {"planes": 1, "area_m2": 6.0}


def test_a_missing_keyframe_is_named_and_an_unmoved_anchor_turns_exactly_zero():
    records, capture = recording()
    del capture["keyframeAnchors"][3]
    report = run(records, capture)["keyframe_anchors"]
    assert report["status"] == "warn" and report["missing"] == [45]
    assert report["turn_deg_max"] == 0.0 and report["shift_cm_max"] == 0.0


def test_a_sample_after_its_frame_is_caught():
    records, capture = recording()
    records[20]["imu"].append([records[20]["t"] + 0.004, 0, 0, 0, 0, -1, 0])
    report = run(records, capture)["imu"]
    assert report["status"] == "warn" and "later than the frame" in report["note"]


def test_an_unfinished_recording_is_reported_not_a_crash():
    records, _ = recording()
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        (folder / "frames.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records) + '{"t": 1, "fra')
        report = cpc.check(folder)
    assert report["recording"]["status"] == "fail" and "never finished" in report["recording"]["note"]
    assert report["frames"]["frames"] == len(records)


if __name__ == "__main__":
    for name, test in sorted(globals().items()):
        if name.startswith("test_"):
            test()
            print("ok", name)
