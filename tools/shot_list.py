#!/usr/bin/env python3
"""The inputs still to be taken for a space: a shot list computed from the gaps.

The capture map records which bearings the camera never faced and how much of
the floor was never seen; shapes.json records which walls had to be inferred;
the measured wall table (room_score) records which built walls have no
standing support. Each gap becomes one concrete shot — what to film, from
where, moving how — written to shots.json and drawn, numbered, on shots.png
over the room's outline and the filmed walk. Every shot cites the
measurement that asks for it; when nothing is missing, the list is empty.

    python3 tools/shot_list.py spaces/<name>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

DIRECTION_STEP = 30       # the capture map's bearing bins
LOOP_MISSING = 8          # this many missing bins or more: one loop, not many pans
LOW_SUPPORT = 0.3         # a built wall measured below this asks for its own pass
FLOOR_UNSEEN = 0.3        # more floor than this never seen asks for a downward sweep
IMAGE_W = 900


def arcs(degrees: list[float], step: int = DIRECTION_STEP) -> list[tuple[float, float]]:
    """Contiguous never-faced arcs (start, end) in degrees, joined across the
    +/-180 wrap: [-180, -150, 150] is one arc from 150 to -120."""
    if not degrees:
        return []
    bins = sorted({(float(d) + 180.0) % 360.0 for d in degrees})
    runs, run = [], [bins[0]]
    for d in bins[1:]:
        if d - run[-1] <= step:
            run.append(d)
        else:
            runs.append(run)
            run = [d]
    runs.append(run)
    if len(runs) > 1 and (runs[0][0] + 360.0) - runs[-1][-1] <= step:
        runs[0] = runs.pop() + [d + 360.0 for d in runs[0]]
    return [(run[0] - 180.0, run[-1] + step - 180.0) for run in runs]


def wall_passes(shapes: dict, walls_measured: dict, units: float | None) -> list[dict]:
    """One pass per wall the capture never established: inferred walls, and
    built measured walls whose standing support is low (not occluded ones —
    those are hidden, not unfilmed)."""
    passes = []
    metre = (lambda v: round(v / units, 2)) if units else (lambda v: round(v, 2))
    for i, p in enumerate(shapes.get("planes") or []):
        if p.get("kind") != "wall" or not p.get("build", True):
            continue
        measured = walls_measured.get(f"W{i}") or {}
        inferred = p.get("source") == "inferred"
        weak = bool(measured) and not measured.get("occluded") \
            and measured.get("support", 1.0) < LOW_SUPPORT
        if not inferred and not weak:
            continue
        length = metre(2 * float(p["half_a"]))
        passes.append({
            "take": f"one slow pass along wall W{i} ({length} m), its full height in frame, "
                    "ending past each corner",
            "evidence": ("the wall was inferred, never measured" if inferred else
                         f"wall table: support {measured.get('support')}, "
                         f"body {measured.get('body')}"),
            "wall": f"W{i}", "center": p.get("center"), "axis": p.get("axis_a"),
            "half": p.get("half_a")})
    return passes


def build(space: Path, walls_measured: dict | None = None, log=print) -> list[dict]:
    """shots.json and shots.png for the space; the shots, in filming order."""
    space = Path(space)
    try:
        capture = json.loads((space / "capture-map.json").read_text())
    except (OSError, ValueError):
        return []
    try:
        shapes = json.loads((space / "shapes.json").read_text())
    except (OSError, ValueError):
        shapes = {}
    try:
        units = json.loads((space / "densify.json").read_text()).get("colmap_units_per_metre")
    except (OSError, ValueError):
        units = None

    shots = []
    missing = capture.get("directions_never_faced_deg") or []
    if len(missing) >= LOOP_MISSING:
        shots.append({"take": "one slow, full loop of the room, ending where it starts, "
                              "facing each wall in turn",
                      "evidence": f"directions faced: {capture.get('directions_faced')} "
                                  f"({len(missing)} bearings never faced)"})
    else:
        for start, end in arcs(missing):
            shots.append({"take": f"from the middle of the filmed walk, a slow pan across "
                                  f"the shaded arc ({start:.0f} to {end:.0f} degrees on the map)",
                          "evidence": "directions_never_faced_deg on the capture map",
                          "arc": [start, end]})
    shots += wall_passes(shapes, walls_measured or {}, units)
    if (capture.get("floor_never_seen") or 0) > FLOOR_UNSEEN:
        shots.append({"take": "a sweep across the unseen floor, camera tilted down, "
                              "keeping each wall's base in frame as it passes",
                      "evidence": f"floor never seen: {capture['floor_never_seen']:.0%}"})
    for n, shot in enumerate(shots, 1):
        shot["shot"] = n
    (space / "shots.json").write_text(json.dumps(
        {"space": space.name, "shots": shots}, indent=1) + "\n")
    draw(space, capture, shapes, shots)
    log(f"    inputs to take: {len(shots)} shot(s) -> shots.json, shots.png")
    return shots


def draw(space: Path, capture: dict, shapes: dict, shots: list[dict]) -> None:
    """The room's outline, the filmed walk, and each shot drawn where it
    happens: wedges for pans, arrows along walls, numbered."""
    cameras = np.asarray(shapes.get("cameras") or [[0.0, 0.0]], float)
    walls = [np.asarray(p["center"], float)[:2] for p in shapes.get("planes") or []
             if p.get("kind") == "wall"]
    pts = np.vstack([cameras] + ([np.array(walls)] if walls else []))
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    span = float(max((hi - lo).max(), 1e-6))
    margin, height = 60, int(IMAGE_W * 0.7)

    def at(xy):
        p = (np.asarray(xy, float)[:2] - lo) / span
        return (margin + p[0] * (IMAGE_W - 2 * margin),
                height - margin - p[1] * (height - 2 * margin))

    board = Image.new("RGB", (IMAGE_W, height), (250, 249, 246))
    art = ImageDraw.Draw(board)
    font = ImageFont.load_default(size=16)
    if len(cameras) > 1:
        art.line([at(c) for c in cameras], fill=(70, 100, 200), width=3)
    centre = at(np.median(cameras, axis=0))
    for shot in shots:
        tag = str(shot["shot"])
        if "arc" in shot:
            start, end = shot["arc"]
            radius = min(IMAGE_W, height) * 0.22
            box = [centre[0] - radius, centre[1] - radius, centre[0] + radius, centre[1] + radius]
            # Image angles run clockwise from +x; bearings counter-clockwise.
            art.pieslice(box, -end, -start, fill=(244, 180, 160), outline=(200, 80, 60))
            mid = np.radians((start + end) / 2)
            art.text((centre[0] + radius * 0.6 * np.cos(mid),
                      centre[1] - radius * 0.6 * np.sin(mid)), tag, fill=(120, 30, 20), font=font)
        elif shot.get("center") is not None and shot.get("axis") is not None:
            c, a, half = (np.asarray(shot["center"], float), np.asarray(shot["axis"], float),
                          float(shot["half"]))
            p0, p1 = at(c[:2] - a[:2] * half), at(c[:2] + a[:2] * half)
            art.line([p0, p1], fill=(200, 80, 60), width=5)
            art.text(((p0[0] + p1[0]) / 2 + 6, (p0[1] + p1[1]) / 2 - 20), tag,
                     fill=(120, 30, 20), font=font)
        else:
            art.text((margin, 40 + 20 * shot["shot"]), f"{tag}: {shot['take'][:90]}",
                     fill=(120, 30, 20), font=font)
    art.text((margin, 14), f"{space.name}: the inputs still to be taken "
             f"({len(shots)} shot(s); the blue line is the filmed walk)",
             fill=(60, 60, 60), font=font)
    board.save(space / "shots.png")


def main() -> int:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    shots = build(Path(sys.argv[1]))
    for shot in shots:
        print(f"  {shot['shot']}. {shot['take']}")
        print(f"     evidence: {shot['evidence']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
