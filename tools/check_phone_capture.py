#!/usr/bin/env python3
"""Check a recording from Oasis Capture: did the new recording fields come out right,
and what do they say? For the first iPhone 13 run (docs/on-device-plan-2026-10.md,
measurements 1, 2 and 7).

    python3 tools/check_phone_capture.py <capture folder> [--json]

The folder holds frames.jsonl and capture.json as the app wrote them. Copy one off
the phone with
    xcrun devicectl device copy from --device <id> --domain-type appDataContainer \\
        --domain-identifier com.oasisspaces.capture --source Documents/Captures/<Capture-...> \\
        --destination <folder>

It reports:
  * frames: how many, over how long, tracking states, frames dropped;
  * exposure: whether it held still while recording (the exposure lock);
  * imu: the motion samples on each frame, their rate, gaps and order;
  * jumps: frames whose pose leapt more than 15 cm from the last;
  * keyframe anchors: how many, whether each one's placed pose matches its frame, how
    many ARKit moved, and how far (the correction the pose refinement can use);
  * plane classes: ARKit's raw classification of every plane, so the wall, floor,
    door and window hit rates can be read.
Each check ends in ok, warn or fail. Recordings made before these fields existed
report them missing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ANCHOR_EVERY = 15           # Recorder.anchorEvery
JUMP_M = 0.15               # Recorder.jumpMetres
IMU_HZ = 100                # Recorder's motion rate
EXPOSURE_TOLERANCE = 0.02   # an exposure within 2% of the recording's typical one counts as held


def matrix(values) -> np.ndarray:
    """A camera-to-world 4x4 from the column-major list the app writes."""
    return np.array(values, dtype=float).reshape(4, 4).T


def angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    """The rotation between two poses, from the chord between their rotation matrices:
    exactly 0 for equal ones and well conditioned for small turns (the arccos of the
    trace is not, with rotations written in single precision)."""
    chord = float(np.linalg.norm(a[:3, :3] - b[:3, :3]))
    return float(np.degrees(2 * np.arcsin(min(1.0, chord / (2 * np.sqrt(2))))))


def frames_report(records: list[dict], capture: dict) -> dict:
    t = np.array([r["t"] for r in records], dtype=float)
    states: dict[str, int] = {}
    for r in records:
        states[r.get("tracking", "?")] = states.get(r.get("tracking", "?"), 0) + 1
    seconds = float(t[-1] - t[0]) if len(t) > 1 else 0.0
    normal = states.get("normal", 0) / max(1, len(records))
    return {
        "frames": len(records), "seconds": round(seconds, 2),
        "fps": round((len(records) - 1) / seconds, 1) if seconds > 0 else None,
        "dropped": capture.get("framesDropped"), "tracking": states,
        "status": "ok" if normal >= 0.9 else "warn",
        "note": f"{normal:.0%} of frames tracked normally",
    }


def exposure_report(records: list[dict]) -> dict:
    values = np.array([r["exposure"] for r in records if "exposure" in r], dtype=float)
    if not len(values):
        return {"status": "fail", "note": "no exposure on any frame"}
    typical = float(np.median(values))
    off = float(np.mean(np.abs(values - typical) > EXPOSURE_TOLERANCE * typical))
    return {
        "typical_s": typical, "min_s": float(values.min()), "max_s": float(values.max()),
        "share_off": round(off, 4),
        "status": "ok" if off <= 0.01 else "warn",
        "note": "held" if off <= 0.01 else f"{off:.0%} of frames differ from the typical exposure: the lock did not hold",
    }


def imu_report(records: list[dict]) -> dict:
    with_imu = [r for r in records if r.get("imu")]
    if not with_imu:
        return {"status": "fail", "note": "no frame carries imu samples"}
    samples = np.array([s for r in with_imu for s in r["imu"]], dtype=float)
    if samples.ndim != 2 or samples.shape[1] != 7:
        return {"status": "fail", "note": f"imu samples should be [t, gx, gy, gz, ax, ay, az]; got shape {samples.shape}"}
    st = samples[:, 0]
    gaps = np.diff(st)
    seconds = float(st[-1] - st[0]) if len(st) > 1 else 0.0
    rate = (len(st) - 1) / seconds if seconds > 0 else 0.0
    # Recorder cuts the samples at each frame's capture time (same clock), so none
    # on a frame should be later than it.
    late = sum(1 for r in with_imu for s in r["imu"] if s[0] > r["t"] + 1e-6)
    gravity = np.linalg.norm(samples[:, 4:7], axis=1)
    problems = []
    if np.any(gaps <= 0):
        problems.append(f"{int(np.sum(gaps <= 0))} samples out of order or repeated")
    if rate < 0.8 * IMU_HZ:
        problems.append(f"rate {rate:.0f} Hz, expected about {IMU_HZ}")
    if len(gaps) and gaps.max() > 0.1:
        problems.append(f"longest gap {gaps.max() * 1000:.0f} ms")
    if late:
        problems.append(f"{late} samples later than the frame they ride on (the cut at capture time failed)")
    if not 0.8 < float(np.median(gravity)) < 1.2:
        problems.append(f"median acceleration {np.median(gravity):.2f} g, expected about 1 g at rest")
    return {
        "frames_with_imu": len(with_imu), "frames": len(records), "samples": int(len(st)),
        "rate_hz": round(rate, 1), "max_gap_ms": round(float(gaps.max()) * 1000, 1) if len(gaps) else None,
        "median_g": round(float(np.median(gravity)), 3),
        "status": "ok" if not problems else ("fail" if any("order" in p for p in problems) else "warn"),
        "note": "; ".join(problems) or "continuous, in order, about 1 g",
    }


def jump_report(records: list[dict]) -> dict:
    flagged = [r["frame"] for r in records if r.get("jump")]
    # Recompute from the poses, to check the flag itself.
    positions = np.array([matrix(r["transform"])[:3, 3] for r in records])
    steps = np.linalg.norm(np.diff(positions, axis=0), axis=1) if len(positions) > 1 else np.zeros(0)
    expected = [records[i + 1]["frame"] for i in np.flatnonzero(steps > JUMP_M)]
    agree = sorted(flagged) == sorted(expected)
    return {
        "jumps": flagged, "expected_from_poses": expected,
        "largest_step_m": round(float(steps.max()), 3) if len(steps) else None,
        "status": "ok" if agree else "fail",
        "note": (f"{len(flagged)} jump(s)" if agree
                 else "the jump flags disagree with the poses (frames dropped between them, or a bug)"),
    }


def anchors_report(records: list[dict], capture: dict) -> dict:
    anchors = capture.get("keyframeAnchors")
    if anchors is None:
        return {"status": "fail", "note": "capture.json has no keyframeAnchors"}
    # Recorder leaves an anchor on every written frame whose index is a multiple of ANCHOR_EVERY.
    expected_frames = {r["frame"] for r in records if r["frame"] % ANCHOR_EVERY == 0}
    expected = len(expected_frames)
    missing = sorted(expected_frames - {a["frame"] for a in anchors})
    by_frame = {r["frame"]: r for r in records}
    mismatched, moves, shifts, turns = [], [], [], []
    for a in anchors:
        placed, final = matrix(a["placed"]), matrix(a["final"])
        frame = by_frame.get(a["frame"])
        if frame is None or not np.allclose(matrix(frame["transform"]), placed, atol=1e-5):
            mismatched.append(a["frame"])
        moves.append(int(a.get("updates", 0)))
        shifts.append(float(np.linalg.norm(final[:3, 3] - placed[:3, 3])))
        turns.append(angle_deg(placed, final))
    moved = sum(1 for m in moves if m > 0)
    problems = []
    if missing:
        problems.append(f"no anchor for keyframes {missing[:8]}{' ...' if len(missing) > 8 else ''}")
    not_seen = [a["frame"] for a in anchors if a.get("atStop") is False]
    if mismatched:
        problems.append(f"{len(mismatched)} anchors placed away from their frame's pose")
    shifts_cm = np.array(shifts) * 100 if shifts else np.zeros(1)
    return {
        "anchors": len(anchors), "expected": expected, "missing": missing, "moved": moved,
        "not_in_last_frame": not_seen,
        "updates_total": int(sum(moves)),
        "shift_cm_median": round(float(np.median(shifts_cm)), 2), "shift_cm_max": round(float(shifts_cm.max()), 2),
        "turn_deg_median": round(float(np.median(turns)), 3) if turns else None,
        "turn_deg_max": round(float(max(turns)), 3) if turns else None,
        "status": "fail" if mismatched else ("warn" if problems else "ok"),
        "note": "; ".join(problems) or (
            f"ARKit moved {moved} of {len(anchors)} anchors (median {np.median(shifts_cm):.1f} cm, "
            f"max {shifts_cm.max():.1f} cm)" if moved else
            "ARKit moved none of the anchors: no correction to read back from this recording"),
    }


def planes_report(capture: dict) -> dict:
    planes = capture.get("planeClasses")
    if planes is None:
        return {"status": "fail", "note": "capture.json has no planeClasses"}
    live = [p for p in planes if not p.get("removed")]
    table: dict[str, dict[str, float]] = {}
    for p in live:
        key = f"{p['alignment']} {p['classification']}"
        area = float(np.prod(p.get("extent", [0, 0])))
        row = table.setdefault(key, {"planes": 0, "area_m2": 0.0})
        row["planes"] += 1
        row["area_m2"] = round(row["area_m2"] + area, 3)
    vertical = [p for p in live if p["alignment"] == "vertical"]
    classified = [p for p in vertical if not p["classification"].startswith(("none", "unsupported", "other"))]
    area = lambda ps: float(sum(np.prod(p.get("extent", [0, 0])) for p in ps))
    share = area(classified) / area(vertical) if area(vertical) > 0 else None
    return {
        "planes": len(live), "removed": len(planes) - len(live), "by_class": table,
        "vertical_classified_share_by_area": round(share, 3) if share is not None else None,
        "status": "ok" if live else "warn",
        "note": (f"{len(classified)} of {len(vertical)} vertical planes classified "
                 f"({share:.0%} of their area)" if vertical else "no vertical planes"),
    }


def check(folder: Path) -> dict:
    records, broken = [], 0
    for line in (folder / "frames.jsonl").read_text().splitlines():
        if line.strip():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                broken += 1             # a line cut short when the app was killed mid-recording
    finished = (folder / "capture.json").exists()
    capture = json.loads((folder / "capture.json").read_text()) if finished else {}
    if not records:
        return {"frames": {"status": "fail", "note": "frames.jsonl is empty"}}
    report = {
        "device": capture.get("device"), "system": capture.get("system"), "app": capture.get("app"),
        "frames": frames_report(records, capture),
        "exposure": exposure_report(records),
        "imu": imu_report(records),
        "jumps": jump_report(records),
        "keyframe_anchors": anchors_report(records, capture),
        "plane_classes": planes_report(capture),
    }
    if not finished or broken:
        report["recording"] = {"status": "fail", "note": (
            "no capture.json: the recording was never finished (the app was closed or killed while recording)"
            if not finished else f"{broken} unreadable line(s) in frames.jsonl")}
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("capture", type=Path, help="folder with frames.jsonl and capture.json")
    parser.add_argument("--json", action="store_true", help="print the full report as JSON")
    args = parser.parse_args()
    report = check(args.capture)
    if args.json:
        print(json.dumps(report, indent=1))
    else:
        print(f"{args.capture.name}: {report.get('device')} on iOS {report.get('system')}")
        for name, part in report.items():
            if isinstance(part, dict) and "status" in part:
                print(f"  {part['status']:4s}  {name.replace('_', ' ')}: {part['note']}")
    return 1 if any(isinstance(p, dict) and p.get("status") == "fail" for p in report.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
