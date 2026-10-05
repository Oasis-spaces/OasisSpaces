#!/usr/bin/env python3
"""What the phone's own perception places in a processed space, as a second
opinion for stage 3.

Oasis Capture's detector, depth and tracker (apps/OasisCapture, run on the Mac
by its `phonesim` tool over the space's frames and camera poses) are a
different model with a different vocabulary from the Mac pipeline's, run on
every frame. On the pan it names and places the wardrobe the Mac's detector
called "hanging cloth": a piece seen at a grazing angle, with a mirror front
that left no points, which no Mac-side method could place.

    python3 tools/phone_objects.py spaces/<name>

exports the space for the simulator (tools/phone_sim_export.py), runs it, and
writes spaces/<name>/phone-objects.json: every object the phone placed, as a
box in the room's own frame (shapes.json's units), with the measured box it
coincides with, if any. `candidate()` turns one into a box the room can take:
its back on the nearest wall, never where the phone itself was.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = ROOT / "apps" / "OasisCapture" / "Packages" / "CaptureRules"
SNAP_M = 0.6               # a phone object this near a wall stands against it
DEPTH_M = (0.2, 0.8)       # a snapped box is at least and at most this deep
CLEAR_M = 0.03             # how far in front of a box the phone must have been
MATCH_IOU = 0.3            # a phone object overlapping a built box this much is that box
UNSEEN_SHARE = 0.03        # a piece filling more of a frame than this is in that frame's view
DEPTH_STEP_M = 0.05

PLACED = re.compile(r"^\s+(o\d+) (.+?) x ([+-]?[\d.]+)\.\.([+-]?[\d.]+)\s+y ([+-]?[\d.]+)\.\.([+-]?[\d.]+)"
                    r"\s+z ([+-]?[\d.]+)\.\.([+-]?[\d.]+)", re.M)


def parse_placed(text: str) -> list[dict]:
    """The simulator's `placed:` lines: id, label and box (its frame: metres,
    x right, y up, z towards the viewer, origin on the floor at the room's centre)."""
    block = text.split("\nplaced:", 1)[1] if "\nplaced:" in text else ""
    found = []
    for m in PLACED.finditer(block):
        x0, x1, y0, y1, z0, z1 = (float(v) for v in m.groups()[2:])
        found.append({"id": m.group(1), "label": m.group(2).strip(), "min": [x0, y0, z0], "max": [x1, y1, z1]})
    return found


def to_scene(lo, hi, shapes: dict, units: float) -> tuple[np.ndarray, np.ndarray]:
    """A box in the simulator's frame as (min, max) in the room's frame, the
    inverse of tools/phone_sim_export.py: scene = origin + (x, -z, y) * units."""
    origin = np.array([*shapes["room"]["center"][:2], shapes["room_level"]["floor_z"]], float)
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])], float)
    scene = origin + np.stack([corners[:, 0], -corners[:, 2], corners[:, 1]], 1) * units
    return scene.min(axis=0), scene.max(axis=0)


def plan_iou(a_lo, a_hi, b_lo, b_hi) -> float:
    w = max(0.0, min(a_hi[0], b_hi[0]) - max(a_lo[0], b_lo[0]))
    d = max(0.0, min(a_hi[1], b_hi[1]) - max(a_lo[1], b_lo[1]))
    union = (a_hi[0] - a_lo[0]) * (a_hi[1] - a_lo[1]) + (b_hi[0] - b_lo[0]) * (b_hi[1] - b_lo[1]) - w * d
    return w * d / union if union > 0 else 0.0


def in_room(placed: list[dict], shapes: dict, units: float) -> list[dict]:
    """The phone's objects in the room's frame, each with the built box it coincides with."""
    built = [(f"B{i}", b) for i, b in enumerate(shapes["boxes"]) if b.get("build", True)]
    objects = []
    for p in placed:
        lo, hi = to_scene(p["min"], p["max"], shapes, units)
        match = max(((plan_iou(lo, hi, b["min"], b["max"]), ident) for ident, b in built), default=(0.0, None))
        objects.append({"id": p["id"], "label": p["label"], "min": lo.round(4).tolist(), "max": hi.round(4).tolist(),
                        "size_m": [round(float(v) / units, 2) for v in hi - lo],
                        "matches": match[1] if match[0] >= MATCH_IOU else None})
    return objects


def frames_showing(lo, hi, views: dict) -> set[str]:
    """The frames that have the box in view (it fills more than UNSEEN_SHARE of the picture)."""
    sys.path.insert(0, str(ROOT / "pipeline"))
    import placement

    return {name for name, view in views.items()
            if float(placement.silhouette(np.asarray(lo, float), np.asarray(hi, float), view,
                                          (view["height"] // placement.GRID, view["width"] // placement.GRID)).mean())
            > UNSEEN_SHARE}


def shows_where_unseen(lo, hi, views: dict, seen: set[str]) -> tuple[str, float] | None:
    """The frame, among those not in `seen`, that the box fills most, when
    that is more than UNSEEN_SHARE of the picture."""
    sys.path.insert(0, str(ROOT / "pipeline"))
    import placement

    worst = None
    for name, view in views.items():
        if name in seen:
            continue
        share = float(placement.silhouette(np.asarray(lo, float), np.asarray(hi, float), view,
                                           (view["height"] // placement.GRID, view["width"] // placement.GRID)).mean())
        if share > UNSEEN_SHARE and (worst is None or share > worst[1]):
            worst = (name, share)
    return worst


def frames_seen(jsonl: str, objects: list[dict]) -> dict[str, list[str]]:
    """For each placed object, the frames in which the phone matched a
    detection to it or named something of its label at all (a held-back
    sighting is still not a frame that saw nothing)."""
    seen: dict[str, set[str]] = {o["id"]: set() for o in objects}
    labels = {o["id"]: o["label"] for o in objects}
    for line in jsonl.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        for inst in row.get("instances") or []:
            for ident, label in labels.items():
                if inst.get("track") == ident or label in (inst.get("name"), inst.get("label")):
                    seen[ident].add(row["frame"])
    return {ident: sorted(names) for ident, names in seen.items()}


def candidate(shapes: dict, obj: dict, units: float, label: str | None = None,
              typical_depth_m: float | None = None, views: dict | None = None) -> dict | str:
    """A phone object as a box for the room: its back on the nearest built
    wall, its extent along that wall and its height the phone's. Its depth is
    what the phone measured or, when given and larger, the furniture type's
    typical depth (the phone sees a cupboard's front, not how deep it is),
    kept within DEPTH_M, short of wherever the phone itself was and, given
    the frames' cameras (`views`), out of every frame that does not already
    have the piece in view at the depth the phone measured: the phone looked
    through the space a too-deep box would fill. (Not "frames in which the
    phone detected none": a detector misses a cupboard it is standing next
    to.) Returns the box record, or why it cannot stand there."""
    lo, hi = np.array(obj["min"], float), np.array(obj["max"], float)
    centre = (lo[:2] + hi[:2]) / 2
    room_centre = np.array((shapes.get("room") or {}).get("center", centre)[:2], float)
    best = None
    for i, p in enumerate(shapes["planes"]):
        if p["kind"] != "wall" or not p.get("build", True):
            continue
        c = np.array(p["center"][:2], float)
        a = np.array(p["axis_a"][:2], float)
        a /= np.linalg.norm(a)
        n = np.array(p["normal"][:2], float)
        n /= np.linalg.norm(n)
        if np.dot(room_centre - c, n) < 0:
            n = -n
        distance = abs(float(np.dot(centre - c, n)))
        along = float(np.dot(centre - c, a))
        if abs(along) <= p["half_a"] and (best is None or distance < best[0]):
            best = (distance, i, c, a, n, p)
    if best is None or best[0] > SNAP_M * units:
        return "it stands against no built wall"
    _, i, c, a, n, p = best
    corners = np.array([[x, y] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])])
    ts = (corners - c) @ a
    t0, t1 = max(-p["half_a"], float(ts.min())), min(p["half_a"], float(ts.max()))
    if t1 - t0 < 0.2 * units:
        return "too little of it lies along the wall"
    # As deep as the phone measured it, back to front, even where the modelled wall
    # cuts through it (the pan's south wall was fitted to this wardrobe's own front).
    across = (corners - c) @ n
    measured = max(across.max(), across.max() - across.min())
    depth = float(np.clip(max(measured, (typical_depth_m or 0.0) * units), DEPTH_M[0] * units, DEPTH_M[1] * units))
    assumed = bool(typical_depth_m) and depth > measured + 1e-9
    for x, y in shapes.get("cameras") or []:                # never where the phone was
        t, d = float(np.dot([x, y] - c, a)), float(np.dot([x, y] - c, n))
        if t0 <= t <= t1 and 0 <= d <= depth:
            depth = d - CLEAR_M * units
    if depth < 0.15 * units:
        return "the phone walked where it would stand"
    level = shapes.get("room_level") or {}
    floor_z = level.get("floor_z", p["center"][2] - p["half_b"])
    height = min(float(hi[2] - lo[2]), level.get("height", 2 * p["half_b"]))

    def footprint(d):
        pts = [c + t * a + k * d * n for t in (t0, t1) for k in (0.0, 1.0)]
        return [float(q[0]) for q in pts], [float(q[1]) for q in pts]

    if views:
        floor = min(depth, max(measured, DEPTH_M[0] * units))
        xs, ys = footprint(floor)
        in_view = frames_showing([min(xs), min(ys), floor_z], [max(xs), max(ys), floor_z + height], views)
        while depth > floor + 1e-9:
            xs, ys = footprint(depth)
            if shows_where_unseen([min(xs), min(ys), floor_z], [max(xs), max(ys), floor_z + height],
                                  views, in_view) is None:
                break
            depth = max(floor, depth - DEPTH_STEP_M * units)
        assumed = bool(typical_depth_m) and depth > measured + 1e-9
    xs, ys = footprint(depth)
    name = label or obj["label"]
    return {"min": [min(xs), min(ys), floor_z], "max": [max(xs), max(ys), floor_z + height], "points": 0,
            "source": "phone", "detected": name, "label": name, "build": True, "color": [190, 185, 175],
            "reason": (f"the phone's own detector placed a {obj['label']} here ({(t1 - t0) / units:.2f} m wide, "
                       f"{height / units:.2f} m tall); stood against W{i}, {depth / units:.2f} m deep"
                       + (f" (it measured {measured / units:.2f} m of front; the depth is the usual one for the type, "
                          "as far as the walk and the frames that look past it allow)" if assumed else ""))}


def run(space: Path, log=print) -> list[dict]:
    """Export the space, run the simulator, write and return phone-objects.json's objects.
    Returns [] where the simulator cannot run (no Swift, no package: any machine but a Mac)."""
    space = Path(space).resolve()
    swift = shutil.which("swift")
    if not swift or not (PACKAGE / "Package.swift").exists():
        return []
    exported = subprocess.run([sys.executable, str(ROOT / "tools/phone_sim_export.py"), str(space)],
                              capture_output=True, text=True)
    if exported.returncode != 0:
        raise RuntimeError(f"phone_sim_export failed: {exported.stderr[-400:]}")
    dump = space / "phone-sim" / "dump"
    shutil.rmtree(dump, ignore_errors=True)
    result = subprocess.run([swift, "run", "-c", "release", "phonesim", str(space), "--dump", str(dump)], cwd=PACKAGE,
                            capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"phonesim failed: {(result.stderr or result.stdout)[-600:]}")
    shapes = json.loads((space / "shapes.json").read_text())
    units = json.loads((space / "densify.json").read_text())["colmap_units_per_metre"]
    objects = in_room(parse_placed(result.stdout), shapes, units)
    sightings = dump / "frames.jsonl"
    if sightings.exists():                                    # which frames saw each object, or named its kind
        seen = frames_seen(sightings.read_text(), objects)
        for o in objects:
            o["frames_seen"] = seen.get(o["id"], [])
        shutil.copy2(sightings, space / "phone-sim" / "frames.jsonl")
    shutil.rmtree(dump, ignore_errors=True)                   # its pictures are not needed here
    (space / "phone-objects.json").write_text(json.dumps({"objects": objects}, indent=1) + "\n")
    log(f"the phone's perception placed {len(objects)} object(s): "
        + ", ".join(f"{o['label']} {o['size_m'][0]}x{o['size_m'][1]}x{o['size_m'][2]} m"
                    + (f" (= {o['matches']})" if o["matches"] else " (nothing measured there)") for o in objects))
    return objects


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("space", type=Path)
    args = parser.parse_args()
    if not run(args.space):
        raise SystemExit("nothing placed, or the simulator cannot run here (it needs Swift and the app's package)")
