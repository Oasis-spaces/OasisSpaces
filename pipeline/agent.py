#!/usr/bin/env python3
"""Run the whole capture pipeline, and adapt when a stage disappoints.

Each stage already knows how to do its job. What a person adds is the
judgement in between: noticing that COLMAP fragmented the capture, that the
keyframes disagree about scale, or that nothing was detected in the room, and
changing the settings before running the next stage. This agent encodes that
judgement, so a capture can be processed end to end unattended.

For every stage it runs the existing script, reads the numbers that stage
produced, then decides: accept, retry differently, or carry on and warn. Every
decision, with the evidence behind it, lands in <space>/agent-report.json,
along with capture advice for the next shoot.

The thresholds come from measurements on this project's own captures:
  - a coherent video model keeps consecutive-frame camera jumps under ~0.1 of
    the scene's size (a glued-together model jumped 442x);
  - keyframes of a sound model agree on metric scale within ~6% (a broken one
    disagreed by 366%).

After every stage a gate checks the output: pass, warn (questionable) or stop
(wrong). The run carries on only on a pass, unless --accept-warnings, so a bad
reconstruction never costs the time of densify and splat training.

Usage:
    python3 pipeline/agent.py <video | photo folder> --name kitchen
    python3 pipeline/agent.py walkthrough.mov --name loft --no-splat

One stage at a time, checking each before starting the next:
    python3 pipeline/agent.py room.mov --name room --stage reconstruct
    python3 pipeline/agent.py room.mov --name room --stage densify
    python3 pipeline/agent.py room.mov --name room --stage shapes
    python3 pipeline/agent.py room.mov --name room --stage splat
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(ROOT / "tools"))
from advisor import Advisor  # noqa: E402
from evaluate_space import evaluate, ply_vertex_count  # noqa: E402
from plan_image import draw_plan  # noqa: E402
from object_frames import draw_object_frames  # noqa: E402
from semantics import MAX_OBJECTS, ROLES, Vocabulary, clean_name, room_vocabulary  # noqa: E402
from densify import read_cameras_bin, read_images_bin  # noqa: E402
from pointcloud import space_model_dir  # noqa: E402
import numpy as np  # noqa: E402
import placement  # noqa: E402
import room_score  # noqa: E402
from reconstruct import (  # noqa: E402
    MAX_PATH_JUMP, camera_path_jump, mean_reprojection, registered_images, solved_models,
)

STAGES = ["reconstruct", "densify", "shapes", "splat"]
# Claude's judgements are filed under what was judged; this is the stage each
# belongs to, so re-running a later stage keeps the earlier stages' judgements.
JUDGEMENT_STAGE = {"objects": "densify", "structure": "shapes", "blender": "shapes",
                   "start view": "splat", "fill": "splat", "choose": "splat",
                   "splat training": "splat", "scene": "splat", "frames": "densify"}

# Tools that live in different places per machine: set BLENDER / OPENSPLAT to
# override (the Colab notebook installs both under /opt and /content).
BLENDER = (os.environ.get("BLENDER") or shutil.which("blender")
           or "/Applications/Blender.app/Contents/MacOS/Blender")
OPENSPLAT = os.environ.get("OPENSPLAT") or str(ROOT / "tools/opensplat")

# A capture is in good shape when most frames land in one model; below the
# partial mark, later stages would only rebuild a fragment of the room.
MIN_REGISTERED_FRACTION = 0.60
GAP_MIN_FRAMES = 3               # this many frames left out of the main model are worth placing with MapAnything
GAP_REPROJ_SLACK_PX = 0.3        # the joined solve may be this much looser than COLMAP's own
PARTIAL_REGISTERED_FRACTION = 0.35
# Keyframes of a sound model agree on metric scale within ~6%; past the good
# mark the result is questionable, past the max the cameras are inconsistent.
GOOD_SCALE_SPREAD = 0.15
MAX_SCALE_SPREAD = 0.30
MIN_DENSE_POINTS = 500_000
MIN_SPLAT_GAUSSIANS = 20_000
# A kept fill may change pixels that were already solid in its review renders
# by at most this much on average (0-255): it should only cover gaps.
FILL_MAX_SOLID_CHANGE = 6.0
# ...and may change at most this share of them by more than FILL_DAMAGE_STEP:
# blotches on a curtain barely move the average but are damage all the same.
FILL_DAMAGE_STEP = 40
FILL_MAX_DAMAGED_SHARE = 0.005
FILL_EDGE_PX = 12       # pixels this close to an original gap are allowed to change
FILL_MIN_BLOBS = 500    # smaller fills are not worth a review (and are not kept)

# Stage 4 trains two splats and keeps the better one (see choose_training): a
# quick one, and a long one at twice the resolution, sharper where the video
# saw clearly but able to overfit the frames it trained on. Claude compares
# them at real video frames neither was trained on.
# The long run (about 40 minutes on a Colab T4, 2.5 hours on an 8 GB Mac) is
# off unless --long-splat on: on both test videos it scored worse than the
# quick run at the views it did not train on.
SPLAT_RUNS = {"quick": (10000, 4), "long": (30000, 2), "spirula": (15000, 4)}   # steps, image downscale
# Spirula Studio is a plug-in trainer (pipeline/splat_spirula.py): normal and
# depth supervision, exposure correction, MCMC densification, on Apple Silicon
# too. Off unless --spirula on: on the walkthrough its default recipe lost to
# the quick splat (new-view SSIM 0.711 vs 0.721, Claude 1 of 3 votes), and it
# adds about 45 minutes on an 8 GB Mac. Its depth supervision has not been
# tested on its own (see README).
SPLAT_LABELS = {"quick": "quick ({steps} steps, 1/{downscale} resolution)",
                "long": "long ({steps} steps, 1/{downscale} resolution)",
                "spirula": "Spirula ({steps} steps, 1/{downscale} resolution, normals + exposure)"}
# Stage 4 in steps that can each run on their own (a Colab session can die at
# any time): every step keeps its results in the space, and the next one
# picks them up. The long training also saves every LONG_CHECKPOINT_STEPS and
# resumes from the newest save.
SPLAT_STEPS = ["train-quick", "train-long", "train-spirula", "choose-training", "fill", "choose-best", "scene"]
LONG_CHECKPOINT_STEPS = 10000
# OpenSplat keeps every training image on the GPU as 32-bit floats; past this
# it reads them from memory each step instead, so an 8 GB Mac is not swamped.
MAX_GPU_IMAGE_CACHE_GB = 1.5

# Before stage 2, Claude names what the room holds (see name_objects).
OBJECT_NAMING_FRAMES = 10
ROOM_FRAMES = 6                  # how many frames of the room itself Claude picks
FRAME_CHOICES = 36               # candidates it picks them from
FRAME_CHOICE_PROMPT = """You are choosing which frames of a phone video of a room the rest of a 3D pipeline will look at.

The attached contact sheet holds {count} frames taken evenly across the video, each numbered in its corner. They are small here; the ones you choose are sent full size to every later question. Nothing later sees any frame you do not choose, so a thing shown in no chosen frame is invisible to the pipeline from now on.

Choose for two purposes.

"room": {room} frames that together show the room itself best: its walls, floor, corners and how it is laid out. Prefer frames looking into the room from far enough back that walls meet and the floor is visible, steady and sharp, well lit, with the camera roughly level. Avoid close-ups of one object, frames looking mostly at a wall, a floor or a ceiling, the corridor or another room, blurred frames, and frames where a hand, a door or a curtain covers most of the view. Spread them around the room rather than three of the same corner.

"objects": {objects} frames that together show as much of the room's contents as possible: every piece of furniture, every storage unit, every mirror, window and screen, and any large loose clutter. Judge them as a set, not one by one: a frame earns its place by showing something the others do not. Closer frames are welcome here, and so is a frame that shows one important object well if nothing else shows it.

A frame may serve both purposes. Where two frames show the same thing, keep the sharper, better lit one.

Fields: {{"room": [numbers, best first], "objects": [numbers, best first], "why": one sentence on what the objects set covers}}"""

OBJECT_NAMING_PROMPT = """You are preparing object detection for a 3D reconstruction of a room, filmed on a phone by an ordinary person. The {count} attached frames are spread across the whole video.

Name every kind of thing in the room that the reconstruction needs to know about. An open-vocabulary detector (GroundingDINO) will search every keyframe for exactly your names, so anything you leave out is never found, and anything you name that is not there can mislabel something else. Each name's role tells the pipeline what to do with the points the detector labels.

Naming rules:
- Short, plain English nouns a general detector knows, 1 to 3 words, no commas: "wardrobe" for an almirah, "sofa" for a couch, "shelf" for a wall rack. The plainest common word wins.
- One name per kind of thing, even if there are several (one "chair" for four chairs).
- Never give one object names in two different roles: a cupboard front that looks like a door is "wardrobe", not also "door". Two storage names for the same piece are fine (a "wardrobe" with a "chest of drawers" base).
- Only things you can actually see in these frames. Leave out walls, floor, ceiling, skirting, beams, tiles, light switches and anything smaller than a shoebox, unless it is glass or a screen.
- Always include every mirror, window, glass door or glass panel, television and monitor you see: depth on them is wrong and those points must be removed.
- Most important first, at most {limit} names.

Roles:
- "furniture": a piece of furniture or an appliance standing on the floor. build_as: "bed"; "seat" for anything sat on (sofa, chair, stool, bench, pouffe); "table" for tables, desks, dressing tables, counters; "block" for anything else solid (fridge, washing machine, air cooler, trunk, bin, stacked boxes).
- "storage": wardrobes, almirahs, cupboards, cabinets, shelves, bookcases, chests of drawers, sideboards, TV units standing on the floor. build_as "wardrobe" (it is built from the floor up). A tall closed unit that could be storage or an appliance (a grey steel almirah can look like a fridge) is storage: storage names are grouped even when the detector sees the unit in parts, so a wrong appliance name splits it.
- "on_furniture": decor that sits on other furniture and belongs in a model of the room (table lamp, potted plant, vase, pillow). build_as "block".
- "loose": belongings that do not belong in a model of the room, wherever they lie (clothes, bags, laptops, keyboards, towels, shoes, toys, bottles). build_as null.
- "floor_covering": rugs, mats, carpets lying on the floor. build_as "block".
- "unreliable": mirrors, windows, glass doors and panels, televisions, monitors. build_as null.
- "hanging": curtains, blinds, clothes or towels hanging in front of a wall. build_as null.
- "fixture": part of the room itself: doors, door frames, pinboards, pictures, air conditioners, ceiling fans, radiators, and shelves or cabinets mounted on a wall that do not reach the floor. build_as null.

Fields: {{"objects": [{{"name": string, "role": string, "build_as": string or null}}], "room": one sentence describing the room}}"""
# Claude picks the view a splat opens at from this many rendered options.
START_VIEW_CHOICES = 6
START_VIEW_TILE = (384, 240)       # the viewer's 16:10 shape
START_VIEW_PROMPT = """You choose the opening view of a 3D capture of a room (a Gaussian splat made from a phone video):
the first thing a person sees when they open it, before they move. The first image shows {count} candidate views,
labelled {letters}, rendered from the reconstruction. The other images are frames of the real room from the video.

Pick the view that best shows this room to someone seeing it for the first time:
- the room reads as a room: some floor, walls and the main furniture, rather than a close-up of one surface
- what is in view is sharp and solid, like the real frames: no smears, streaks, fog, floating specks, holes or black areas
- not staring at a blank wall, into a corner, into a curtain, or pressed against furniture
Prefer the clean view over the wide one if the wide one shows reconstruction damage.
Reply with a single JSON object and nothing else:
{{"best": letter, "ranking": [letters best first], "why": one sentence}}"""

# Files that make up a built room, kept aside while stage 3 is reviewed again.
STRUCTURE_FILES = ["shapes.json", "room.blend", "room-render.png", "room-render-plan.png", "room-views.png",
                   "plan-reviewed.png"]
RECHECK_NOTE = """

This is a second look. The room built from the first review was rendered and checked, and the check found these
structural problems: {problems}. The last {count} image(s) are that render from an angle, the room from straight
above, and, when there are three, a sheet of rows, each a video frame beside the room rendered from that frame's own
camera (grey shapes are walls, floor and furniture boxes). Use the same actions to fix them where the images show what
is wrong: drop a box that is not a free-standing object of its own (part of the bed, a panel, a sliver of a cupboard)
or that duplicates another, drop a wall that stands free or duplicates a wall, add a piece of furniture the frames
show but the room lacks, or drop a misplaced piece and add it again where the rows show it stands. Leave alone what
the check did not flag."""

ADD_BOXES_MAX = 3                # boxes Claude may add from the frames in one structure review
FURNITURE_TYPES = {"bed", "seat", "table", "wardrobe", "block"}
ADDED_SIZE_M = {"width": (0.3, 4.0), "depth": (0.2, 2.0), "height": (0.2, 3.0)}   # sanity limits, metres
CAMERA_CLEARANCE_M = 0.1         # an added box this close to where the phone was cannot be there
ASKABLE_PIECES = {"bed", "wardrobe", "table", "seat"}   # a review that drops every box of one of these is asked where it stands
PHONE_ADDS_M = {"wardrobe": 1.4, "bed": 0.3}     # the phone's second opinion adds only large furniture: family -> least height
FRAGMENT_DEPTH_M = 0.25          # a measured box thinner than this, against a wall, may be a piece's fragment (an open door leaf)
SLAB_M = 0.2                     # a built piece thinner than this that lies mostly inside another is that piece's surface ...
SLAB_INSIDE = 0.5                # ... when this share of it is inside (repair_overlaps, after the finish step)


def dropped_pieces(shapes: dict, verdict: dict, applied: list[str]) -> dict[str, list[tuple[str, str]]]:
    """Furniture types the review has just dropped every box of, as fragments,
    with nothing of that type left built: {label: [(id, why), ...]}."""
    boxes = shapes["boxes"]
    built = {b.get("label") for b in boxes if b.get("build", True)}
    found: dict[str, list[tuple[str, str]]] = {}
    for item in verdict.get("drop_boxes") or []:
        ident = str(item.get("id", ""))
        if f"dropped {ident}" not in applied:
            continue
        box = boxes[int(ident[1:])]
        label = box.get("detected") or box.get("label")
        if label in ASKABLE_PIECES and label not in built:
            found.setdefault(label, []).append((ident, str(item.get("why", ""))))
    return found


def place_added_box(shapes: dict, item: dict, units: float) -> dict | str:
    """A box Claude adds from the frames, placed in the measured room: its back
    on the wall `against`, its near side `offset_m` along that wall from the
    corner it shares with `from_corner_with` (centred on the wall without one),
    standing on the floor. Sizes are clamped to ADDED_SIZE_M, the wall's length
    and the room's height. Returns the box record, or why it cannot be placed."""
    import numpy as np

    label = item.get("label")
    if label not in FURNITURE_TYPES:
        return f"'{label}' is not a furniture type"
    planes = shapes["planes"]

    def wall(ident):
        text = str(ident or "")
        i = int(text[1:]) if text[:1] == "W" and text[1:].isdigit() else None
        if i is None or i >= len(planes) or planes[i]["kind"] != "wall" or not planes[i].get("build", True):
            return None, None
        return i, planes[i]

    i, w = wall(item.get("against"))
    if w is None:
        return f"{item.get('against')} is not a built wall"
    try:
        wanted = {k: float(item[f"{k}_m"]) * units for k in ("width", "depth", "height")}
    except (KeyError, TypeError, ValueError):
        return "width_m, depth_m and height_m are needed"
    level = shapes.get("room_level") or {}
    floor_z = level.get("floor_z", w["center"][2] - w["half_b"])
    room_height = level.get("height", 2 * w["half_b"])
    caps = {"width": 2 * w["half_a"], "depth": float("inf"), "height": room_height}
    size = {k: max(lo * units, min(hi * units, caps[k], v))
            for k, v in wanted.items() for lo, hi in [ADDED_SIZE_M[k]]}
    c = np.array(w["center"][:2], dtype=float)
    a = np.array(w["axis_a"][:2], dtype=float)
    a /= np.linalg.norm(a)
    n = np.array(w["normal"][:2], dtype=float)
    n /= np.linalg.norm(n)
    room_centre = np.array((shapes.get("room") or {}).get("center", c)[:2], dtype=float)
    if np.dot(room_centre - c, n) < 0:
        n = -n                                       # into the room
    t0 = -size["width"] / 2                          # centred on the wall, unless a corner is named
    j, other = wall(item.get("from_corner_with"))
    if other is not None and j != i:
        b = np.array(other["axis_a"][:2], dtype=float)
        matrix = np.array([a, -b]).T
        if abs(np.linalg.det(matrix)) > 1e-6:
            t_corner = float(np.linalg.solve(matrix, np.array(other["center"][:2]) - c)[0])
            direction = 1.0 if t_corner < 0 else -1.0    # from that corner toward the wall's middle
            near = t_corner + direction * max(0.0, float(item.get("offset_m") or 0.0)) * units
            t0 = near if direction > 0 else near - size["width"]
    t0 = max(-w["half_a"], min(w["half_a"] - size["width"], t0))
    corners = [c + t * a + k * size["depth"] * n for t in (t0, t0 + size["width"]) for k in (0.0, 1.0)]
    xs, ys = [float(q[0]) for q in corners], [float(q[1]) for q in corners]
    margin = CAMERA_CLEARANCE_M * units
    for k, (x, y) in enumerate(shapes.get("cameras") or [], 1):
        if min(xs) - margin <= x <= max(xs) + margin and min(ys) - margin <= y <= max(ys) + margin:
            return f"the phone stood inside it or against it (camera {k}); nothing solid stands on the walk"
    return {"min": [min(xs), min(ys), floor_z], "max": [max(xs), max(ys), floor_z + size["height"]],
            "points": 0, "source": "claude", "detected": label, "label": label, "build": True,
            "color": [190, 185, 175], "reason": f"Claude: added from the frames: {item.get('why', '')}"}


# After detection, Claude checks the boxes against the list (see review_labels).
LABEL_REVIEW_PROMPT = """You check object detection for a 3D reconstruction of a room filmed on a phone.
An open-vocabulary detector (GroundingDINO) searched {keyframes} for exactly the names in this list; each name has a
role that tells the pipeline what to do with the points it labels:
{listing}
The first image shows {shown} of those frames with every detection box drawn and named (name and confidence). The other
images are plain frames from the same video.

Revise the list so the detector finds what is really in the room, and nothing else:
- "add": objects clearly visible in the frames that no box covers and that matter to the room model: furniture, storage,
  glass or screens, curtains, and loose belongings big enough to confuse the furniture boxes (a heap of clothes, a bag on
  the floor). Skip anything smaller than a shoebox that is not glass or a screen. Same roles and build_as as the list.
- "rename": a name whose object is visible but was never or rarely boxed, where a plainer, more common English word would
  help the detector ("almirah" -> "wardrobe", "monitor" -> "computer screen").
- "remove": a name whose boxes land on something else (a "chest of drawers" box on the side of a bed, a "wardrobe" box
  on a wall shelf) and whose own object is not in the room. Removing it frees those points for the right name.
Change nothing when the detections already match the room. Mirrors, windows and screens are always searched for.
At most {limit} names in total.
Fields: {{"objects_seen": array of short phrases, "missed": array of objects no box covers, "wrong": array of
"name: what its boxes are really on", "add": [{{"name": string, "role": string, "build_as": string or null}}],
"rename": [{{"from": string, "to": string}}], "remove": [{{"name": string, "why": string}}], "why": one sentence}}"""

FILL_REVIEW_PROMPT = """You review an automatic fill in a 3D room reconstruction (a Gaussian splat) made from a phone video.
The reconstruction has gaps where the camera never saw a surface clearly: black holes, see-through patches or smears on
floors, walls and furniture. A fill adds new surface there, continuing the texture around it.

Each image is one filled surface ({surfaces}): on the left a real video frame, then the reconstruction rendered from
exactly that camera before the fill and after it. Decide for each surface whether to keep its fill.

Some filled areas were never seen clearly by any video frame (seen_by_video false): for those the left image is only the
nearest video frame, from a different angle, to show what the room's floor and walls look like, and the before/after
renders come from a nearby viewpoint. Judge those by whether the holes are now covered by believable surface that matches
the room, without new artefacts.

Keep it only if the after image is better: holes or see-through patches are now covered by surface that looks like its
surroundings in the video frame, and nothing that looked right before looks worse.
Reject it if anything got worse, even if some gaps were covered: blotches, streaks or stains that are not in the video frame;
a patch whose colour, brightness or texture visibly differs from its surroundings; hard edges or a pasted-on look; a door,
window, doorway, shelf opening, curtain or any object painted over, darkened or partly erased; or if you see no real
difference (then the fill only adds risk).
Differences that are the same in before and after are not the fill's doing.

Reply with a single JSON object and nothing else:
{{"surfaces": [{{"id": "S1", "keep": true or false, "why": "one sentence naming what you saw"}}]}}"""
# Walls this rough (as a share of the room's diagonal) mean weak geometry.
NOISY_WALL_RMS_PCT = 1.5
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def structural_problems(verdict: dict | None) -> list:
    """The render check's problems graded structural."""
    return [p for p in (verdict or {}).get("problems") or []
            if isinstance(p, dict) and p.get("severity") == "structural"]


def problem_text(problem) -> str:
    """A render-check problem as text: a plain phrase, or {what, severity}."""
    if isinstance(problem, dict):
        return f"{problem.get('what', '?')} ({problem.get('severity', 'unrated')})"
    return str(problem)


def keep_filled(before: dict, after: dict | None) -> tuple[bool, str]:
    """Is the solve with the lost frames placed better than COLMAP's own?
    More frames in one model, no looser than COLMAP's (reprojection within
    GAP_REPROJ_SLACK_PX), and no camera leaping across the room between two
    frames (reconstruct.MAX_PATH_JUMP)."""
    if not after or not after.get("frames"):
        return False, "the joined solve did not come out"
    if after["frames"] <= before["frames"]:
        return False, f"it placed {after['frames']} frames, not more than COLMAP's {before['frames']}"
    if after.get("models", 1) > 1 and after["frames"] < before["total"] * 0.95:
        return False, f"it is still {after['models']} pieces"
    if (before.get("reprojection") is not None and after.get("reprojection") is not None
            and after["reprojection"] > before["reprojection"] + GAP_REPROJ_SLACK_PX):
        return False, (f"its reprojection error is {after['reprojection']:.2f} px against COLMAP's "
                       f"{before['reprojection']:.2f}")
    if (after.get("path_jump") or 0) > MAX_PATH_JUMP:
        return False, f"a camera leaps {after['path_jump']:.2f}x the scene's size between two frames"
    return True, (f"{after['frames']} of {after['total']} frames in one model against COLMAP's "
                  f"{before['frames']}, reprojection {after.get('reprojection') or 0:.2f} px against "
                  f"{before.get('reprojection') or 0:.2f}")


def root_cause(output: str) -> str:
    """The failing step's final error line. A step that says only
    "see log: <file>" (COLMAP writes to its own log) is followed there, so
    the gate's why names the real error — a rejected flag or a denied
    download, not just "did not complete"."""
    lines = [line.strip() for line in output.strip().splitlines() if line.strip()]
    if not lines:
        return ""
    last = lines[-1]
    if "see log: " in last:
        try:
            logged = Path(last.rsplit("see log: ", 1)[1]).read_text(errors="replace")
        except OSError:
            return last
        # COLMAP colours its log lines even into a file, and the log is
        # appended to across steps and runs: strip the colours and read only
        # the failed step's own section (after the last "=== command" header).
        logged = re.sub(r"\x1b\[[0-9;]*m", "", logged).strip().splitlines()
        for i in range(len(logged) - 1, -1, -1):
            if logged[i].startswith("=== "):
                logged = logged[i + 1:]
                break
        section = [line.strip() for line in logged if line.strip()]
        # glog errors and fatals first; otherwise any line that says error,
        # but not an info line mentioning one in passing.
        errors = ([line for line in section if line[:2] in ("E2", "F2")]
                  or [line for line in section
                      if "error" in line.lower() and not line.startswith("I2")])
        if errors:
            return f"{last.split(', see log')[0]}: {errors[-1]}"
    return last


WALK_PROMPT = """Two maps of the same phone video of a room, each drawn from a camera solve. Both show the room from above: the measured points in their colours, the floor shaded by how often it was seen, and the walk from start (green) to end (red) with a tick where the camera looked.

Map 1 is COLMAP's own solve: it could place {before} of the video's {total} frames in one piece. Map 2 adds the frames it lost, placed by MapAnything from COLMAP's cameras and then triangulated and adjusted together with them: {after} frames. Numbers: {facts}.

The measurements say map 2 is at least as tight as map 1. What they cannot see is whether the frames it added are in the right place. Judge the walk: a person walked through this room once with a phone, so the path should be one continuous, unhurried line with no teleports, no loops through furniture or walls, no stretch that doubles back on itself for no reason, and the added part should continue the walk where map 1's ends or has a gap. The room itself should keep its shape.

Reply as JSON: {{"keep": true if map 2 is a plausible walk and room, false if the added frames are clearly misplaced, "why": "one or two sentences on what you saw"}}"""


class Agent:
    def __init__(self, source: Path, name: str, fps: float, do_splat: bool,
                 allow_retry: bool = True, use_claude: bool = True,
                 long_splat: bool | None = None, trained_elsewhere: bool = False,
                 retrain: bool = False, splat_steps: list[str] | None = None,
                 spirula: bool | None = None):
        self.source = source
        self.name = name
        self.fps = fps
        self.do_splat = do_splat
        self.cuda = shutil.which("nvidia-smi") is not None
        # The long run measured worse than the quick one on both test videos
        # (Sep 2026), so it only runs when asked for.
        self.long_splat = bool(long_splat)
        from splat_spirula import find_binary
        self.spirula_requested = bool(spirula)
        self.spirula = self.spirula_requested and find_binary() is not None
        self.trained_elsewhere = trained_elsewhere
        self.retrain = retrain
        self.splat_steps = splat_steps or list(SPLAT_STEPS)
        self.current_step: str | None = None   # the stage-4 step recording decisions
        self.chosen: dict | None = None        # the frames Claude picked (choose_frames)
        self.allow_retry = allow_retry
        self.space = ROOT / "spaces" / name
        self.log_path = self.space / "agent.log"
        self.stages: list[dict] = []
        self.advice: list[str] = []
        self.judgements: list[dict] = []
        self.gates: dict[str, dict] = {}
        self.last_error = ""                   # the failing step's root cause (run)
        self.use_claude = use_claude
        self.advisor = Advisor(enabled=use_claude)

    def sample_frames(self, count: int = 3) -> list[Path]:
        """Frames spread across the capture, for Claude to look at."""
        images = sorted((self.space / "workspace" / "images").glob("*.jpg"))
        if not images or count <= 0:
            return []
        if count == 1:
            return [images[len(images) // 2]]
        if len(images) <= count:
            return images
        step = (len(images) - 1) / (count - 1)
        return [images[round(i * step)] for i in range(count)]

    def room_frames(self, count: int = 3) -> list[Path]:
        """Frames of the room: Claude's own picks (choose_frames), else the
        ones whose cameras look most directly at the room's centre, spread
        across the capture. A video can start and end somewhere else (a pan
        from the corridor), so the first and last frames may not show the room."""
        import numpy as np

        picked = self.picked_frames("room", count)
        if picked:
            return picked

        try:
            shapes = json.loads((self.space / "shapes.json").read_text())
            model = space_model_dir(self.space)
            room = shapes["room"]
            if model is None:
                return self.sample_frames(count)
        except (OSError, KeyError, ValueError):
            return self.sample_frames(count)
        walls = [p for p in shapes["planes"] if p["kind"] == "wall" and p.get("build", True)]
        height = float(np.median([w["center"][2] for w in walls])) if walls else 0.0
        world = np.array(shapes["world"])
        target = world.T @ np.array([room["center"][0], room["center"][1], height])
        cameras = read_cameras_bin(model / "cameras.bin")
        scored = []
        for info in read_images_bin(model / "images.bin").values():
            local = info["R"] @ target + info["t"]
            if local[2] <= 0:
                continue  # the room centre is behind this camera
            cam = cameras[info["camera_id"]]
            fx = cam["params"][0]
            in_view = abs(local[0] / local[2]) < cam["width"] / (2 * fx)
            angle = float(np.degrees(np.arccos(local[2] / np.linalg.norm(local))))
            scored.append((not in_view, angle, info["name"]))
        images = self.space / "workspace" / "images"
        names = sorted(p.name for p in images.glob("*.jpg"))
        if not scored or not names:
            return self.sample_frames(count)
        spacing = max(1, len(names) // (2 * count))
        chosen: list[str] = []
        for _, _, name in sorted(scored):
            index = names.index(name) if name in names else -1
            if index >= 0 and all(abs(index - names.index(c)) >= spacing for c in chosen):
                chosen.append(name)
            if len(chosen) == count:
                break
        return [images / name for name in chosen] or self.sample_frames(count)

    def contact_sheet(self, names: list[str], out: Path, across: int = 6, tile: int = 300) -> Path:
        """The candidate frames as one numbered picture."""
        from PIL import Image, ImageDraw, ImageFont

        images = self.space / "workspace" / "images"
        shots = [Image.open(images / name).convert("RGB") for name in names]
        wide = max(s.width / s.height for s in shots) >= 1
        cell = (tile, round(tile / 1.4)) if wide else (round(tile / 1.4), tile)
        down = -(-len(shots) // across)
        page = Image.new("RGB", (across * (cell[0] + 6) + 6, down * (cell[1] + 6) + 6), "white")
        draw = ImageDraw.Draw(page)
        font = ImageFont.load_default(size=22)
        for n, shot in enumerate(shots):
            shot.thumbnail(cell)
            x, y = 6 + (n % across) * (cell[0] + 6), 6 + (n // across) * (cell[1] + 6)
            page.paste(shot, (x, y))
            draw.rectangle([x, y, x + 34, y + 26], fill=(0, 0, 0))
            draw.text((x + 6, y + 3), str(n + 1), fill=(255, 210, 90), font=font)
        page.save(out)
        return out

    def choose_frames(self) -> dict | None:
        """Claude picks the frames every later question is asked with.

        Which frames are sent decides what can be judged at all: an object in
        no chosen frame cannot be named, and a room shown only from a corner
        cannot be checked. A rule can only spread frames evenly or point them
        at the room's middle, so the choice is Claude's, once, from a contact
        sheet of the whole video; every later question then draws on it."""
        images = sorted(p.name for p in (self.space / "workspace" / "images").glob("*.jpg"))
        if not images:
            return None
        if not self.advisor.available:
            return None
        step = max(1, round(len(images) / FRAME_CHOICES))
        candidates = images[::step][:FRAME_CHOICES]
        sheet = self.contact_sheet(candidates, self.space / "frame-choices.png")
        verdict = self.advisor.ask_json(
            FRAME_CHOICE_PROMPT.format(count=len(candidates), room=ROOM_FRAMES, objects=OBJECT_NAMING_FRAMES),
            [sheet], max_tokens=800)
        if not verdict:
            print("    claude (frames): no usable answer; frames are picked by the old rules")
            return None
        picked = {}
        for purpose in ("room", "objects"):
            numbers = [n for n in (verdict.get(purpose) or []) if isinstance(n, int) and 1 <= n <= len(candidates)]
            picked[purpose] = [candidates[n - 1] for n in dict.fromkeys(numbers)]
        if not picked["room"] and not picked["objects"]:
            return None
        self.chosen = picked
        (self.space / "frames-chosen.json").write_text(json.dumps(
            {"candidates": candidates, **picked, "why": verdict.get("why")}, indent=1) + "\n")
        self.judged("frames", {"room": picked["room"], "objects": picked["objects"],
                               "why": verdict.get("why")},
                    f"{len(picked['room'])} frame(s) of the room, {len(picked['objects'])} of its contents"
                    + (f": {verdict.get('why')}" if verdict.get("why") else ""))
        return picked

    def picked_frames(self, purpose: str, count: int) -> list[Path] | None:
        """Claude's frames for this purpose, if it chose any."""
        if not self.chosen:
            found = self.space / "frames-chosen.json"
            self.chosen = json.loads(found.read_text()) if found.exists() else {}
        names = (self.chosen or {}).get(purpose) or []
        images = self.space / "workspace" / "images"
        kept = [images / name for name in names[:count] if (images / name).exists()]
        return kept or None

    def safe(self, call, *args, default=None):
        """A judgement is an optional extra: never let one end the run."""
        try:
            return call(*args)
        except Exception as exc:
            print(f"    claude check skipped ({type(exc).__name__}: {exc})")
            return default

    def judged(self, stage: str, verdict: dict, shown: str) -> None:
        entry = {"stage": stage, **verdict}
        if JUDGEMENT_STAGE.get(stage, stage) == "splat" and self.current_step:
            entry["step"] = self.current_step
        self.judgements.append(entry)
        print(f"    claude ({stage}): {shown}")

    # ---------------------------------------------------------------- stages
    def run(self, args: list[str], stage: str, note: str = "") -> tuple[bool, str]:
        """Run one stage, tee its output to the space's log."""
        printable = " ".join(a.replace(str(ROOT) + "/", "") for a in args[1:])
        print(f"\n=== {stage}{': ' + note if note else ''}\n    $ {printable}")
        started = time.time()
        result = subprocess.run(args, text=True, capture_output=True, errors="replace")
        seconds = round(time.time() - started, 1)
        self.space.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "a") as log:
            log.write(f"\n=== {stage} ({seconds}s): {' '.join(args)}\n")
            log.write(result.stdout + result.stderr)
        if result.returncode != 0:
            self.last_error = root_cause(result.stdout + result.stderr)
            print(f"    failed after {seconds}s; see {self.log_path}")
            if self.last_error:
                print(f"    {self.last_error}")
        else:
            self.last_error = ""
            print(f"    done in {seconds}s")
        return result.returncode == 0, result.stdout

    def log(self, text: str) -> None:
        """A line for the space's log only (steps that run in this process)."""
        self.space.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "a") as log:
            log.write(text + "\n")

    def decide(self, stage: str, action: str, why: str, metrics: dict | None = None,
               seconds: float | None = None) -> None:
        entry = {"stage": stage, "action": action, "why": why}
        if stage == "splat" and self.current_step:
            entry["step"] = self.current_step
        if metrics:
            entry["metrics"] = metrics
        if seconds is not None:
            entry["seconds"] = seconds
        self.stages.append(entry)
        print(f"    -> {action}: {why}")
        if action != "stop":
            # The failure is judged and tolerated (warn, skip, retry...):
            # a later sub-step's stop must not wear this error.
            self.last_error = ""

    # -------------------------------------------------------- claude's calls
    def capture_verdict(self, metrics: dict) -> dict | None:
        """Claude looks at the capture and picks how to retry the solve."""
        if not self.advisor.available:
            return None
        prompt = (
            "You are running a photogrammetry pipeline on a phone video of a room. "
            "COLMAP solved the camera positions using SIFT features and a global "
            f"mapper, and reported: {json.dumps(metrics)}. A 'model' is one connected "
            "reconstruction, so more than one means the capture broke into pieces.\n"
            "The attached frames come from the start, middle and end of the capture.\n"
            "Choose exactly one action:\n"
            '  "accept": good enough to carry on;\n'
            '  "retry_aliked": redo with learned features (ALIKED), which find far '
            "more on textureless surfaces but take about 2.5 times as long to "
            "extract;\n"
            '  "retry_incremental": redo with the incremental mapper, slower, but it '
            "never joins pieces that do not belong together.\n"
            'Fields: {"action": string, "why": one short sentence, '
            '"capture_problems": array of short phrases}')
        drawn = self.safe(self.draw_capture_map)
        images = self.sample_frames(3) + ([self.space / "capture-map.png"] if drawn else [])
        if drawn:
            prompt += ("\nThe last image is a map of the solve from above: the walk, the floor shaded by how "
                       "often it was seen (red never), and a compass of the directions the camera faced.")
        verdict = self.advisor.ask_json(prompt, images)
        if verdict:
            problems = ", ".join(verdict.get("capture_problems") or [])
            self.judged("reconstruct", verdict,
                        f"{verdict.get('action')} - {verdict.get('why', '')}"
                        + (f" [{problems}]" if problems else ""))
        return verdict

    def draw_capture_map(self, out: Path | None = None) -> dict:
        """capture-map.png/json from the current solve (tools/capture_map.py):
        the dense cloud once densify has run, else the solve's own points, and
        the scale measured or, before densify, guessed from the phone's height."""
        from capture_map import build

        dense = (self.space / "cloud-dense.ply").exists()
        record = build(self.space, source="colmap", points_from="dense" if dense else "sparse",
                       out=out, log=lambda text: print("    " + text))
        return record

    def mapanything_ready(self) -> bool:
        import importlib.util

        return self.cuda and importlib.util.find_spec("mapanything") is not None

    def fill_gaps(self) -> None:
        """Place the frames COLMAP could not, and join its pieces into one.

        COLMAP's global mapper places most of a walked video precisely and
        loses the frames around a sharp turn or along a plain wall, or splits
        the walk in two. MapAnything, given COLMAP's lens and the cameras
        COLMAP did place, places the rest in the same frame
        (pipeline/mapanything_solve.py --guide); COLMAP then triangulates its
        own matches around all of them and adjusts (reconstruct.py --mapper
        priors). Kept only if it measures better (keep_filled) and Claude,
        comparing the two walks, sees nothing misplaced. Measured on the
        walkthrough (Sep 2026): 183 of 186 frames in one model against
        167 + a 14-frame piece, 70,286 points against 38,757, 0.78 px against
        0.87."""
        first = self.reconstruction_metrics()
        lost = first["total"] - first["frames"]
        if lost < GAP_MIN_FRAMES and first["models"] <= 1:
            return
        if not self.mapanything_ready():
            self.decide("reconstruct", "skip",
                        f"{lost} of {first['total']} frames are not in the main model ({first['models']} "
                        "piece(s)); placing them with MapAnything needs a CUDA GPU with map-anything "
                        "(stage 1 on Colab has both)")
            return
        workspace = self.space / "workspace"
        best = max(solved_models(workspace / "sparse"), key=lambda m: (m / "points3D.bin").stat().st_size)
        first["reprojection"] = mean_reprojection(best)
        # COLMAP's solve, kept to go back to; its main model also guides MapAnything.
        kept = self.space / "workspace-colmap"
        shutil.rmtree(kept, ignore_errors=True)
        kept.mkdir()
        for name in ("sparse", "dropped-frames.json"):
            if (workspace / name).is_dir():
                shutil.copytree(workspace / name, kept / name)
            elif (workspace / name).exists():
                shutil.copy2(workspace / name, kept / name)
        if (self.space / "cloud.ply").exists():
            shutil.copy2(self.space / "cloud.ply", kept / "cloud.ply")
        guide = kept / "sparse" / best.name
        self.safe(self.draw_capture_map, self.space / "capture-map-colmap")
        started = time.time()
        ok, _ = self.run([sys.executable, str(ROOT / "pipeline/mapanything_solve.py"), str(self.space),
                          "--guide", str(guide)], "reconstruct", f"MapAnything places the {lost} frames COLMAP lost")
        if ok:
            ok, _ = self.run([sys.executable, str(ROOT / "pipeline/reconstruct.py"), str(self.source),
                              "--name", self.name, "--fps", str(self.fps), "--mapper", "priors"],
                             "reconstruct", "COLMAP triangulates around its cameras and MapAnything's")
        after = self.reconstruction_metrics() if ok else None
        if after and after.get("frames"):
            joined = max(solved_models(workspace / "sparse"), key=lambda m: (m / "points3D.bin").stat().st_size)
            after["reprojection"] = mean_reprojection(joined)
        keep, why = keep_filled(first, after)
        if keep:
            self.safe(self.draw_capture_map)
            verdict = self.safe(self.walk_verdict, first, after, why)
            if verdict and verdict.get("keep") is False:
                keep, why = False, f"Claude: {verdict.get('why', 'the added frames look misplaced')}"
        seconds = round(time.time() - started, 1)
        if keep:
            self.decide("reconstruct", "accept", f"frames COLMAP lost placed with MapAnything: {why}", after, seconds)
            shutil.rmtree(kept, ignore_errors=True)
            return
        self.decide("reconstruct", "revert", f"keeping COLMAP's own solve: {why}", after, seconds)
        for name in ("sparse", "dropped-frames.json"):
            target = workspace / name
            if target.is_dir():
                shutil.rmtree(target)
            elif target.exists():
                target.unlink()
            if (kept / name).exists():
                shutil.move(str(kept / name), str(target))
        if (kept / "cloud.ply").exists():
            shutil.move(str(kept / "cloud.ply"), str(self.space / "cloud.ply"))
        shutil.rmtree(kept, ignore_errors=True)

    def walk_verdict(self, before: dict, after: dict, facts: str) -> dict | None:
        """Claude compares the walk of COLMAP's solve with the joined one."""
        maps = [self.space / "capture-map-colmap.png", self.space / "capture-map.png"]
        if not self.advisor.available or not all(m.exists() for m in maps):
            return None
        verdict = self.advisor.ask_json(WALK_PROMPT.format(before=before["frames"], total=before["total"],
                                                           after=after["frames"], facts=facts), maps)
        if verdict:
            self.judged("reconstruct", {"walk_keep": verdict.get("keep"), "why": verdict.get("why")},
                        f"{'keep' if verdict.get('keep') else 'revert'} the joined walk - {verdict.get('why', '')}")
        return verdict

    def name_objects(self) -> None:
        """Claude names what is in this room before stage 2, so the detector
        searches for the room's own objects instead of a fixed list, and each
        name carries its role (see ROLES in pipeline/semantics.py). Written to
        objects.json, which densify.py reads; an existing list is kept, so a
        re-run labels the cloud the same way."""
        path = self.space / "objects.json"
        if path.exists():
            print(f"    object list: keeping {path.name} "
                  f"({len(json.loads(path.read_text()).get('objects', []))} names)")
            return
        if not self.advisor.available:
            print("    object list: Claude unavailable, the detector uses the default list")
            return
        frames = self.picked_frames("objects", OBJECT_NAMING_FRAMES) or self.sample_frames(OBJECT_NAMING_FRAMES)
        verdict = self.advisor.ask_json(
            OBJECT_NAMING_PROMPT.format(count=len(frames), limit=MAX_OBJECTS - 4),
            frames, max_tokens=2000)
        vocabulary = Vocabulary((verdict or {}).get("objects") or [], "claude")
        if not vocabulary.furniture:
            print("    object list: no usable answer, the detector uses the default list"
                  + (f" ({self.advisor.reason})" if self.advisor.reason else ""))
            return
        path.write_text(json.dumps({**vocabulary.to_json(), "room": verdict.get("room"),
                                    "frames": [f.name for f in frames]}, indent=1) + "\n")
        self.judged("objects", {"objects": vocabulary.objects, "room": verdict.get("room")},
                    f"{len(vocabulary.names)} names: " + ", ".join(
                        f"{o['name']} ({o['role']})" for o in vocabulary.objects[:12])
                    + (" ..." if len(vocabulary.names) > 12 else ""))

    def review_labels(self) -> bool:
        """Claude checks what the detector found, on keyframes with every
        detection box drawn, against the room's object list, and revises the
        list: names to add for objects that were never boxed, plainer words for
        names the detector does not understand, and names to remove that land
        on something else. Returns True when the list changed, so densify runs
        again with it (once)."""
        if not self.advisor.available:
            return False
        from object_frames import draw_detections

        dense = json.loads((self.space / "densify.json").read_text())
        vocabulary = room_vocabulary(self.space)
        sheet = self.space / "detections.png"
        frames = self.safe(draw_detections, self.space, sheet, default=[]) or []
        if not frames:
            return False
        found_in = {}
        for dets in (dense.get("detections") or {}).values():
            for name in {d["label"] for d in dets}:
                found_in[name] = found_in.get(name, 0) + 1
        listing = [{"name": o["name"], "role": o["role"],
                    "frames_found_in": found_in.get(o["name"], 0),
                    "points_labelled": (dense.get("labels") or {}).get(o["name"], 0)}
                   for o in vocabulary.objects]
        # With tracking (densify.json "tracks"), each thing the detector found
        # was carried through the video, so the counts cover every frame.
        searched = (f"{dense['tracks']['frames']} frames (each object it found on about twenty of them was "
                    "tracked through the rest, so a box is a tracked outline's rectangle)"
                    if dense.get("tracks") else f"{len(dense.get('detections') or {})} keyframes")
        verdict = self.advisor.ask_json(LABEL_REVIEW_PROMPT.format(
            keyframes=searched, shown=len(frames),
            listing=json.dumps(listing), limit=MAX_OBJECTS - 4),
            [sheet] + (self.picked_frames("objects", 4) or self.sample_frames(4)), max_tokens=2000)
        if not verdict:
            return False
        revised, changes = self.revise_objects(vocabulary, verdict)
        self.judged("densify", {**verdict, "applied": changes},
                    ("; ".join(changes) if changes else "the object list stands")
                    + (f"; missed {', '.join(map(str, verdict.get('missed', [])[:4]))}"
                       if verdict.get("missed") else ""))
        if not changes:
            return False
        path = self.space / "objects.json"
        previous = json.loads(path.read_text()) if path.exists() else {"objects": vocabulary.objects}
        revisions = previous.pop("revisions", [])
        revisions.append({"objects": previous.get("objects"), "changes": changes,
                          "why": verdict.get("why")})
        path.write_text(json.dumps({**previous, **revised.to_json(), "revisions": revisions},
                                   indent=1) + "\n")
        return True

    def revise_objects(self, vocabulary, verdict: dict):
        """Apply Claude's add / rename / remove to the object list, within
        limits: names are cleaned like any other, roles must be known, and the
        list stays under the detector's size."""
        objects = [dict(o) for o in vocabulary.objects]
        by_name = {o["name"]: o for o in objects}
        changes = []
        for item in verdict.get("remove") or []:
            name = clean_name(item.get("name", "") if isinstance(item, dict) else item)
            if name in by_name:
                objects.remove(by_name.pop(name))
                changes.append(f"removed '{name}'")
        for item in verdict.get("rename") or []:
            old, new = clean_name(item.get("from", "")), clean_name(item.get("to", ""))
            if old in by_name and new and new not in by_name:
                by_name[new] = by_name.pop(old)
                by_name[new]["name"] = new
                changes.append(f"renamed '{old}' to '{new}'")
        for item in verdict.get("add") or []:
            name = clean_name(item.get("name", ""))
            if name and name not in by_name and item.get("role") in ROLES:
                entry = {"name": name, "role": item["role"], "build_as": item.get("build_as")}
                objects.append(entry)
                by_name[name] = entry
                changes.append(f"added '{name}' ({item['role']})")
        revised = Vocabulary(objects, "claude")
        dropped = [o["name"] for o in objects if o["name"] not in revised.names]
        if dropped:
            changes.append("left out " + ", ".join(dropped) + " (limits)")
        return revised, (changes if revised.objects != vocabulary.objects else [])

    def structure_review(self, feedback: list | None = None) -> None:
        """Claude decides what each measured candidate is. It can drop a wall or
        a box, relabel a box, or add a box for furniture the frames show that the
        points never boxed (placed against a measured wall, place_added_box), but
        never move or resize measured geometry, so the room keeps it. With `feedback` (the structural problems
        the render check found in the room built from an earlier review), it
        looks again with those problems and the renders in front of it."""
        if not self.advisor.available:
            return
        shapes_path = self.space / "shapes.json"
        shapes = json.loads(shapes_path.read_text())
        plan = draw_plan(self.space, self.space / "plan-candidates.png")
        units = self.densify_metrics().get("colmap_units_per_metre")
        size = (lambda v: round(v / units, 2)) if units else (lambda v: round(v, 2))
        candidates = []
        for i, p in enumerate(shapes["planes"]):
            tag = "W" if p["kind"] == "wall" else "F"
            candidates.append({"id": f"{tag}{i}", "label": p.get("label", p["kind"]),
                               "size": [size(2 * p["half_a"]), size(2 * p["half_b"])],
                               "points": p["points"],
                               **({"surface_behind": size(p["behind"]["depth"])}
                                  if p.get("behind") else {})})
        for i, b in enumerate(shapes["boxes"]):
            candidates.append({"id": f"B{i}", "label": b.get("label"),
                               "detected_as": b.get("detected"),
                               "size": [size(b["max"][k] - b["min"][k]) for k in range(3)],
                               "points": b["points"], "built": b.get("build", True),
                               "check": b.get("reason")})
        sheet = self.space / "object-frames.png"
        picked = self.safe(draw_object_frames, self.space, sheet, default={}) or {}
        prompt = (
            "You are reviewing the room structure our 3D pipeline measured from a phone "
            "video. The first image is a floor plan seen from above: darker areas are "
            "dense reconstructed points, the blue outline is the floor, red numbered "
            "lines (W) are wall candidates, green numbered rectangles (B) are furniture "
            "boxes that will be built, grey ones were rejected by our checks, and the "
            "small blue dots joined by a line are where the phone was as it filmed, so "
            "nothing solid stands there. "
            + ("The second image shows, for each box, the two video frames where it is "
               "seen best, cropped around it, with the box's outline drawn in green, its "
               "B number and size: judge each box from its own crops, since an object at "
               "the side of the room may not appear in any other frame. A door sits flush "
               "in its wall, in a frame, and swings; a cupboard, almirah or wardrobe stands "
               "in front of a wall with depth, and often handles, drawers, shelves or a "
               "decorated top. Filmed side-on, a cupboard's front can look like a door, "
               "so check both crops. " if picked else "")
            + "The remaining images are frames looking into the room.\n"
            f"Candidates (sizes in {'metres' if units else 'scene units'}): "
            f"{json.dumps(candidates)}\n"
            "Decide what each candidate is. You cannot move or resize anything; the "
            "measurements stay. You may drop a wall that duplicates another wall or is "
            "not a wall (for example a wardrobe front), drop a box that is not a real "
            "object, or relabel a built box as one of: bed, seat, table, wardrobe, block. "
            "Boxes are built as free-standing furniture, so drop any box that is part of "
            "the room or of another object: a door, a door frame or jamb, a curtain, a "
            "wall patch, or a fragment of a bigger piece of furniture. Relabel only boxes "
            "that are a whole piece of furniture of their own; 'block' is for real "
            "furniture that fits no other type, not for leftovers. "
            "Rejected boxes stay rejected. "
            "A wall with surface_behind has another flat surface that far behind it, "
            "outside the room: the wall may be the front of built-in furniture (the "
            "doors of a fitted wardrobe or cupboard) with the room's real wall behind. "
            "If the frames show built-in furniture along that wall, list it in "
            "furniture_fronts: the wall moves back and the space in front of it becomes "
            "a wardrobe. If it is simply the wall, leave it. "
            "The frames may also show a whole piece of furniture that has no box of its "
            "own: a cupboard whose only box is its open door, a bed or table the points "
            "missed. Add it in add_boxes: its type (bed, seat, table, wardrobe, block), "
            "the wall it stands against (W id), the wall it meets at its nearest corner "
            "(W id, or null when it stands away from the corners), offset_m from that "
            "corner to its near side, and its width_m along the wall, depth_m into the "
            "room and height_m, judged from the frames and from what such furniture "
            "usually measures (a wardrobe is about 0.6 m deep and 2 m tall). Add only "
            "what the frames show clearly and no box already covers; when a box covers "
            f"part of it, drop that box and add the whole piece. At most {ADD_BOXES_MAX}. "
            "Only act where the images make you confident.\n"
            'Fields: {"drop_walls": [{"id": string, "why": string}], '
            '"furniture_fronts": [{"id": string, "why": string}], '
            '"drop_boxes": [{"id": string, "why": string}], '
            '"relabel_boxes": [{"id": string, "label": string, "why": string}], '
            '"add_boxes": [{"label": string, "against": "W<n>", "from_corner_with": "W<n>" or null, '
            '"offset_m": number, "width_m": number, "depth_m": number, "height_m": number, "why": string}], '
            '"notes": one sentence}')
        images = [plan] + ([sheet] if picked else []) + self.room_frames(3)
        if feedback:
            renders = [r for r in (self.space / "room-render.png", self.space / "room-render-plan.png",
                                   self.space / "room-views.png") if r.exists()]
            prompt += RECHECK_NOTE.format(count=len(renders), problems=json.dumps(feedback))
            images += renders
        verdict = self.advisor.ask_json(prompt, images, max_tokens=2048)
        if not verdict:
            print(f"    claude (structure): no usable answer"
                  + (f" ({self.advisor.reason})" if self.advisor.reason else ""))
            return
        applied = self.apply_structure_review(shapes, verdict)
        # On disk as reviewed before any piece is placed, so nothing downstream reads the candidates.
        shapes_path.write_text(json.dumps(shapes, indent=1) + "\n")
        for label, dropped in dropped_pieces(shapes, verdict, applied).items():
            applied += self.safe(self.place_dropped_piece, shapes, label, dropped, default=[]) or []
        shapes_path.write_text(json.dumps(shapes, indent=1) + "\n")
        draw_plan(self.space, self.space / "plan-reviewed.png")
        self.judged("structure", {**verdict, "applied": applied},
                    "; ".join(applied) or "no changes")

    def place_by_masks(self, shapes: dict, label: str) -> tuple[list[dict], int]:
        """Fit boxes for `label` to its SAM masks in the keyframes
        (pipeline/placement.py), one per instance. Returns (boxes, evidence frames)."""
        units = self.densify_metrics().get("colmap_units_per_metre")
        evidence = placement.mask_evidence(self.space, label, log=lambda text: print("    " + text))
        found = placement.instances(self.space, shapes, label, evidence, units, log=lambda text: print("    " + text))
        sheet = self.space / f"placement-{label.replace(' ', '-')}.png"
        self.safe(placement.evidence_sheet, evidence, found[0] if found else None, sheet, self.space)
        return found, len(evidence["frames"])

    def phone_objects(self) -> list[dict]:
        """What the phone's own perception places in this space
        (tools/phone_objects.py), once per run; [] where it cannot run."""
        if getattr(self, "_phone_objects", None) is None:
            import phone_objects

            self._phone_objects = self.safe(phone_objects.run, self.space,
                                            lambda text: print("    " + text), default=[]) or []
        return self._phone_objects

    def all_views(self) -> dict:
        """Pose and lens of every registered frame (placement.views_of), once per run."""
        if getattr(self, "_all_views", None) is None:
            images = sorted(p.name for p in (self.space / "workspace" / "images").glob("*.jpg"))
            self._all_views = placement.views_of(self.space, images)
        return self._all_views

    def phone_second_opinion(self, shapes: dict, limit: int = 2) -> list[str]:
        """Large furniture the phone placed that the room does not have as
        such. Where it overlaps a thin measured box of the same family the two
        are one piece seen two ways (the pan's wardrobe: the Mac measured its
        open door leaf, the phone its body), so that box grows to cover both;
        where nothing of the family stands there, it is added. A phone object
        that coincides with a measured box changes nothing: the measurement
        stands, and agreement is not a reason to move it. Small things the
        phone names (a low cabinet, a basket, a TV) are left out: its boxes are
        coarse and its labels noisy, so only pieces a room cannot do without
        are taken on its word. The outcome then does not hang on whether the
        review kept or dropped a fragment."""
        import phone_objects

        units = self.densify_metrics().get("colmap_units_per_metre")
        if not units:
            return []
        notes = []
        for obj in self.phone_objects():
            family = placement.furniture_family(self.space, obj["label"])
            if family not in PHONE_ADDS_M or obj.get("matches") or len(notes) >= limit:
                continue
            if obj["size_m"][2] < PHONE_ADDS_M[family]:
                continue                                              # a low cabinet is not a wardrobe
            box = phone_objects.candidate(shapes, obj, units, label=family,
                                          typical_depth_m=placement.TYPICAL_M[family][1], views=self.all_views())
            if isinstance(box, str):
                continue
            built = [(i, b) for i, b in enumerate(shapes["boxes"]) if b.get("build", True)]
            overlaps = [(phone_objects.plan_iou(box["min"], box["max"], b["min"], b["max"]), i, b) for i, b in built]
            # Only a thin measured box grows: a box Claude or the phone added is a guess already,
            # and a full-depth measured piece is not a fragment.
            def thin(b):
                return min(b["max"][0] - b["min"][0], b["max"][1] - b["min"][1]) < FRAGMENT_DEPTH_M * units

            same = [(v, i, b) for v, i, b in overlaps
                    if v > 0.02 and placement.furniture_family(self.space, b.get("label") or "") == family
                    and b.get("source") not in ("claude", "phone", "masks") and thin(b)]
            if same:
                _, i, target = max(same, key=lambda item: item[0])
                grown_min = [min(target["min"][k], box["min"][k]) for k in range(3)]
                grown_max = [max(target["max"][k], box["max"][k]) for k in range(3)]
                if grown_min != target["min"] or grown_max != target["max"]:
                    target["min"], target["max"] = grown_min, grown_max
                    target["reason"] = (str(target.get("reason", "")) + "; grown to what the phone's detector "
                                        f"placed as a {obj['label']} ({obj['size_m'][0]} m wide)").lstrip("; ")
                    notes.append(f"grew B{i} {target.get('label')} to the phone's {obj['label']}")
            elif not any(v > 0.3 for v, _, _ in overlaps):          # nothing else already stands there
                shapes["boxes"].append(box)
                notes.append(f"added B{len(shapes['boxes']) - 1} {family}: {box['reason']}")
        return notes

    def settle_pieces(self) -> None:
        """On the finished room: the phone's second opinion on large furniture
        (its tracker needs closed walls and a floor to ground what it sees,
        which the candidates mid-review are not: asked then, it reported the
        pan's wardrobe 1.14 m tall, asked now 1.70), then, for a furniture
        type the review dropped whole that still has nothing built, Claude's
        answer where no keyframe detected it. The room is finished again
        around whatever was added."""
        pending = getattr(self, "_pending_pieces", None) or {}
        self._pending_pieces = {}
        self._phone_objects = None                        # the room has changed: the phone is asked afresh
        shapes_path = self.space / "shapes.json"
        shapes = json.loads(shapes_path.read_text())
        before = len(shapes["boxes"])
        # What the second opinion adds is measured like the review's edits: a
        # piece the phone places over one already measured (its thin front
        # matched no box, the box it stands in does) is undone.
        measured = self.safe(self.measure_room, shapes)
        snapshot = copy.deepcopy(shapes["boxes"])
        notes = self.safe(self.repair_overlaps, shapes, default=[]) or []
        notes += self.safe(self.phone_second_opinion, shapes, default=[]) or []
        for label, piece in pending.items():
            family = placement.furniture_family(self.space, label)
            if any(b.get("build", True) and placement.furniture_family(self.space, b.get("label") or "") == family
                   for b in shapes["boxes"]):
                continue
            if piece["evidence_frames"] >= placement.MIN_FRAMES:
                # The keyframes that saw one agree on no box and the phone placed none:
                # the piece is not guessed into the room.
                notes.append(f"no {label} added: its masks in {piece['evidence_frames']} keyframe(s) "
                             "support no box against a wall")
            else:
                notes += self.safe(self.ask_where_piece_stands, shapes, label, piece["dropped"], default=[]) or []
        if not notes:
            return
        if measured is not None and shapes["boxes"] != snapshot:
            after = self.safe(self.measure_room, shapes)
            if after is not None and room_score.compare(measured, after) == "worse":
                shapes["boxes"] = snapshot
                notes.append("rolled back what the second opinion added: it lowered the measured room "
                             f"score from {measured['score']:.2f} to {after['score']:.2f}")
        shapes_path.write_text(json.dumps(shapes, indent=1) + "\n")
        self.judged("structure", {"second_opinion": notes, "applied": notes}, "; ".join(notes))
        if len(shapes["boxes"]) != before or any(note.startswith("grew") for note in notes):
            self.run([sys.executable, str(ROOT / "tools/classify_shapes.py"), str(self.space), "--finish-only"],
                     "finish", "around what the second opinion added")

    def repair_candidates(self) -> None:
        """Before the review: a thin slab lying mostly inside another candidate
        is its surface (repair_overlaps), so it is not shown to the review as a
        piece, and the review is not tempted to drop the piece it lies in for
        their overlap (the pan's cupboard went that way, for a 0.14 m "bed"
        slab 87% inside it)."""
        path = self.space / "shapes.json"
        shapes = json.loads(path.read_text())
        notes = self.repair_overlaps(shapes)
        if notes:
            path.write_text(json.dumps(shapes, indent=1) + "\n")
            self.judged("structure", {"repairs": notes, "applied": notes}, "; ".join(notes))

    def repair_overlaps(self, shapes: dict) -> list[str]:
        """Once the finish step has stood every piece on the floor, a thin
        piece lying mostly inside another is that piece's surface, not
        furniture: the front of a bed's storage base measured as a "chest of
        drawers" (the pan, 0.07 m thick, 71% inside the bed once the bed
        reached the floor). It is not built. Bounded: only pieces thinner
        than SLAB_M, only when SLAB_INSIDE of them lies inside another."""
        units = self.densify_metrics().get("colmap_units_per_metre")
        if not units:
            return []
        notes = []
        pieces = room_score.built_pieces(shapes)
        for i, a in pieces:
            lo, hi = np.array(a["min"], float), np.array(a["max"], float)
            thickness = float(min(hi[0] - lo[0], hi[1] - lo[1])) / units
            if thickness > SLAB_M or not a.get("build", True):
                continue
            volume = float(np.prod(hi - lo))
            for j, b in pieces:
                if j == i or not b.get("build", True):
                    continue
                inner = np.maximum(np.minimum(hi, np.array(b["max"], float)) - np.maximum(lo, np.array(b["min"], float)), 0.0)
                share = float(np.prod(inner)) / volume if volume > 0 else 0.0
                if share >= SLAB_INSIDE:
                    a["build"] = False
                    a["reason"] = f"a {thickness:.2f} m thin slab lying {share:.0%} inside B{j}: its surface, not a piece"
                    notes.append(f"dropped B{i} ({a.get('detected') or a.get('label')}): a {thickness:.2f} m thin slab "
                                 f"lying {share:.0%} inside B{j} ({b.get('label')}), its surface rather than a piece")
                    break
        return notes

    def place_dropped_piece(self, shapes: dict, label: str, dropped: list[tuple[str, str]]) -> list[str]:
        """The review dropped every '{label}' box as a fragment (a door leaf, a
        shelf) and nothing of that type is built, so the room would lose a
        piece the frames show. Its own masks in the keyframes place it where
        they can (place_by_masks); otherwise it waits for settle_pieces, on the
        finished room."""
        applied = []
        units = self.densify_metrics().get("colmap_units_per_metre")
        if not units:
            return applied
        by_masks, evidence_frames = self.safe(self.place_by_masks, shapes, label, default=([], 0)) or ([], 0)
        if by_masks:
            for box in by_masks:
                shapes["boxes"].append(box)
                applied.append(f"added B{len(shapes['boxes']) - 1} {label}: {box['reason']}")
            return applied
        if getattr(self, "_pending_pieces", None) is None:
            self._pending_pieces = {}
        self._pending_pieces[label] = {"dropped": dropped, "evidence_frames": evidence_frames}
        return applied

    def ask_where_piece_stands(self, shapes: dict, label: str, dropped: list[tuple[str, str]]) -> list[str]:
        """Claude says where a piece the review dropped whole stands, and it is
        added there (place_added_box); a refused spot gets one more try. The
        last resort: only where no keyframe detected the piece and the phone
        placed none."""
        applied = []
        units = self.densify_metrics().get("colmap_units_per_metre")
        if not units or not self.advisor.available:
            return applied
        crops = self.space / "object-frames.png"
        plan = self.space / "plan-candidates.png"
        images = ([plan] if plan.exists() else []) + ([crops] if crops.exists() else []) \
            + (self.picked_frames("objects", 3) or []) + self.room_frames(2)
        reasons = "; ".join(f"{ident}: {why}" for ident, why in dropped)
        prompt = (
            f"Our review of this room dropped these boxes labelled '{label}' as fragments, not the "
            f"whole piece: {reasons}. No {label} is built now. If the frames show that this room "
            f"really has a {label}, say where the whole piece stands so it can be built.\n"
            "The first image is the floor plan from above: red numbered lines are the walls (W), "
            "the blue dots joined by a line are where the phone was, and nothing solid stands on "
            "that walk. "
            + ("The next image shows the review's boxes outlined in the two frames where each was "
               "seen best, the dropped ones among them. " if crops.exists() else "")
            + "The remaining images are frames of the room. An open door lying flat along a wall "
            "belongs to a body standing next to it, usually in the corner it swings from.\n"
            "Give the wall the piece stands against (W id), the wall it meets at its nearest corner "
            "(W id, or null when it stands away from the corners), offset_m from that corner to its "
            "near side, and its width_m along the wall, depth_m into the room and height_m, judged "
            "from the frames and from what such furniture usually measures.\n"
            'Fields: {"add": {"against": "W<n>", "from_corner_with": "W<n>" or null, "offset_m": number, '
            '"width_m": number, "depth_m": number, "height_m": number, "why": string}} '
            f'or {{"none": true, "why": string}} when the frames do not show a whole {label}.')
        for attempt in range(2):
            verdict = self.advisor.ask_json(prompt, images, max_tokens=800)
            if not verdict:
                applied.append(f"asked where the {label} stands: no usable answer")
                break
            if verdict.get("none") or not isinstance(verdict.get("add"), dict):
                applied.append(f"no {label} added: {verdict.get('why', 'Claude sees no whole piece')}")
                break
            item = {**verdict["add"], "label": label}
            placed = place_added_box(shapes, item, units)
            if isinstance(placed, str):
                applied.append(f"ignored a {label} against {item.get('against')}: {placed}")
                prompt += f"\n\nThat spot was refused: {placed}. Give another, or none."
                continue
            shapes["boxes"].append(placed)
            corner = item.get("from_corner_with")
            applied.append(f"asked where the {label} stands: added B{len(shapes['boxes']) - 1} against "
                           f"{item.get('against')}" + (f", from its corner with {corner}" if corner else "")
                           + f" ({item.get('why', '')})")
            break
        return applied

    def apply_structure_review(self, shapes: dict, verdict: dict) -> list[str]:
        """Apply Claude's decisions within fixed limits; report what happened.
        Each group of edits that touches the furniture is measured
        (room_score.py) and undone when it lowers the room's score: Claude
        names what looks wrong, the measurements decide."""
        applied = []
        measured = self.safe(self.measure_room, shapes)

        def checkpoint() -> dict:
            return {"planes": copy.deepcopy(shapes["planes"]), "boxes": copy.deepcopy(shapes["boxes"])}

        def settle(group: str, snapshot: dict) -> bool:
            """Undo the group's edits when they lowered the measured score. True when undone."""
            nonlocal measured
            changed = shapes["planes"] != snapshot["planes"] or shapes["boxes"] != snapshot["boxes"]
            if measured is None or not changed:
                return False
            after = self.safe(self.measure_room, shapes)
            if after is None:
                return False
            if room_score.compare(measured, after) == "worse":
                shapes["planes"], shapes["boxes"] = snapshot["planes"], snapshot["boxes"]
                applied.append(f"rolled back {group}: it lowered the measured room score "
                               f"from {measured['score']:.2f} to {after['score']:.2f}")
                return True
            measured = after
            return False

        walls = {i: p for i, p in enumerate(shapes["planes"])
                 if p["kind"] == "wall" and p.get("build", True)}
        fronted = set()
        snapshot = checkpoint()
        for item in verdict.get("furniture_fronts") or []:
            ident = str(item.get("id", ""))
            if not ident:
                continue  # an empty placeholder entry, not a decision
            i = int(ident[1:]) if ident[:1] == "W" and ident[1:].isdigit() else None
            if i not in walls or not walls[i].get("behind"):
                applied.append(f"ignored {ident}: not a wall with a surface behind it")
                continue
            applied.append(self.build_front(shapes, i, item.get("why", "")))
            fronted.add(i)
        if settle("furniture_fronts", snapshot):
            walls = {i: p for i, p in enumerate(shapes["planes"])
                     if p["kind"] == "wall" and p.get("build", True)}
            fronted.clear()
        biggest = max(walls, key=lambda i: walls[i]["points"]) if walls else None
        can_drop = len(walls) // 2  # never remove more than half the walls
        for item in verdict.get("drop_walls") or []:
            ident = str(item.get("id", ""))
            i = int(ident[1:]) if ident[:1] == "W" and ident[1:].isdigit() else None
            if i in fronted:
                applied.append(f"kept {ident}: it was moved back as a furniture front")
            elif i not in walls:
                applied.append(f"ignored {ident}: not a wall candidate")
            elif i == biggest:
                applied.append(f"kept {ident}: the largest wall is never dropped")
            elif can_drop <= 0:
                applied.append(f"kept {ident}: at most half the walls can be dropped")
            else:
                walls[i]["build"] = False
                walls[i]["review"] = f"Claude: {item.get('why', '')}"
                can_drop -= 1
                applied.append(f"dropped {ident}")
        boxes = shapes["boxes"]
        snapshot = checkpoint()
        vouched = self.safe(self.phone_matches, default={}) or {}
        detected = [i for i, b in enumerate(boxes) if b.get("detected") and b.get("build", True)]
        main_object = max(detected, key=lambda i: boxes[i]["points"]) if detected else None
        for item in verdict.get("drop_boxes") or []:
            ident = str(item.get("id", ""))
            i = int(ident[1:]) if ident[:1] == "B" and ident[1:].isdigit() else None
            if i is None or i >= len(boxes) or not boxes[i].get("build", True):
                applied.append(f"ignored {ident}: not a built box")
                continue
            if i == main_object:
                applied.append(f"kept {ident}: the room's largest detected object "
                               f"({boxes[i]['detected']}) is never dropped")
                continue
            if ident in vouched and boxes[i].get("source") not in ("claude", "phone", "masks"):
                # Two independent measurements agree on it: one opinion does not remove it.
                applied.append(f"kept {ident}: the phone's own detector places a "
                               f"{vouched[ident]['label']} on the same spot")
                continue
            boxes[i]["build"] = False
            boxes[i]["reason"] = f"Claude: {item.get('why', '')}"
            applied.append(f"dropped {ident}")
            # One box at a time: dropping the first of two boxes measured on
            # one spot removes their overlap (kept); dropping the second loses
            # a piece the frames show (undone), whatever the review said.
            if settle(f"dropping {ident}", snapshot):
                boxes = shapes["boxes"]
            snapshot = checkpoint()
        for item in verdict.get("relabel_boxes") or []:
            ident, new = str(item.get("id", "")), item.get("label")
            i = int(ident[1:]) if ident[:1] == "B" and ident[1:].isdigit() else None
            if i is None or i >= len(boxes) or not boxes[i].get("build", True):
                applied.append(f"ignored {ident}: not a built box")
            elif new not in {"bed", "seat", "table", "wardrobe", "block"}:
                applied.append(f"ignored {ident}: '{new}' is not a furniture type")
            elif new != boxes[i].get("label"):
                applied.append(f"relabelled {ident} {boxes[i].get('label')} -> {new}")
                boxes[i]["label"] = new
                boxes[i]["reason"] = f"Claude: {item.get('why', '')}"
                if settle(f"relabelling {ident}", snapshot):
                    boxes = shapes["boxes"]
                snapshot = checkpoint()
        units = self.densify_metrics().get("colmap_units_per_metre")
        for n, item in enumerate(verdict.get("add_boxes") or []):
            what = f"an added {item.get('label')}"
            if n >= ADD_BOXES_MAX:
                applied.append(f"ignored {what}: at most {ADD_BOXES_MAX} boxes can be added")
            elif not units:
                applied.append(f"ignored {what}: the room's scale is not measured")
            elif isinstance(placed := place_added_box(shapes, item, units), str):
                applied.append(f"ignored {what}: {placed}")
            else:
                boxes.append(placed)
                corner = item.get("from_corner_with")
                applied.append(f"added B{len(boxes) - 1} {placed['label']} against {item.get('against')}"
                               + (f", from its corner with {corner}" if corner else ""))
                if settle(f"adding B{len(boxes) - 1}", snapshot):
                    boxes = shapes["boxes"]
                snapshot = checkpoint()
        return applied

    def build_front(self, shapes: dict, i: int, why: str) -> str:
        """Wall i is the front of built-in furniture: move the wall back to the
        surface behind it, and fill the space between with a wardrobe box that
        stands on the floor and covers the measured stretch of that surface."""
        plane = shapes["planes"][i]
        behind = plane.pop("behind")
        c, a, normal = plane["center"], plane["axis_a"], plane["normal"]
        out = behind["outward"]
        # The wall was squared up after shapes.py measured it: step back along
        # its current normal, turned to point away from the room.
        sign = 1.0 if normal[0] * out[0] + normal[1] * out[1] >= 0 else -1.0
        step = [sign * normal[0], sign * normal[1]]
        depth = behind["depth"]
        ts = [max(-plane["half_a"], min(plane["half_a"],
                                        (x - c[0]) * a[0] + (y - c[1]) * a[1]))
              for x, y in behind["span_xy"]]
        corners = [(c[0] + t * a[0] + k * depth * step[0], c[1] + t * a[1] + k * depth * step[1])
                   for t in ts for k in (0.0, 1.0)]
        level = shapes.get("room_level") or {}
        floor_z = level.get("floor_z", c[2] - plane["half_b"])
        top = max(behind["top"], floor_z + 0.5 * plane["half_b"])
        shapes["boxes"].append({
            "min": [min(x for x, _ in corners), min(y for _, y in corners), floor_z],
            "max": [max(x for x, _ in corners), max(y for _, y in corners), top],
            "points": behind["points"], "source": "front", "detected": "wardrobe",
            "label": "wardrobe", "build": True, "color": plane["color"],
            "reason": f"Claude: built-in furniture in front of W{i}: {why}",
        })
        plane["center"] = [c[0] + depth * step[0], c[1] + depth * step[1], c[2]]
        plane["review"] = f"Claude: furniture front, moved back to the wall behind: {why}"
        units = self.densify_metrics().get("colmap_units_per_metre")
        shown = f"{depth / units:.2f} m" if units else f"{depth:.2f} units"
        return (f"W{i} is a furniture front: moved it back {shown} and built "
                f"B{len(shapes['boxes']) - 1} wardrobe in front of it")

    def frames_showing_added_pieces(self, limit: int = 2) -> list[str]:
        """For each piece the review added rather than measured (by its masks,
        by the phone, by Claude), the frame that shows most of it: the room
        frames may all look elsewhere, and a piece judged without a frame that
        sees it is judged from an arbitrary angle."""
        shapes = json.loads((self.space / "shapes.json").read_text())
        added = [b for b in shapes["boxes"] if b.get("build", True) and b.get("source") in ("masks", "phone", "claude")]
        if not added:
            return []
        images = sorted(p.name for p in (self.space / "workspace" / "images").glob("*.jpg"))
        views = placement.views_of(self.space, images)
        chosen = []
        for box in added[:limit]:
            lo, hi = np.array(box["min"]), np.array(box["max"])
            corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])

            def seen(name: str) -> float:
                """The share of the frame the piece fills, for a frame that has all of it in front of
                the lens and is not filled by it (a camera standing against the piece shows nothing)."""
                view = views[name]
                ahead = ((view["R"] @ (corners @ view["world"]).T).T + view["t"])[:, 2]
                if ahead.min() <= 0.3 * (hi - lo).max():
                    return 0.0
                share = float(placement.silhouette(lo, hi, view, (view["height"] // placement.GRID,
                                                                  view["width"] // placement.GRID)).mean())
                return share if share <= 0.6 else 0.0

            best = max(views, default=None, key=seen)
            if best is not None and seen(best) > 0.01 and best not in chosen:
                chosen.append(best)
        return chosen

    def view_sheet(self) -> Path | None:
        """The built room rendered from three of the video's own cameras, each
        beside its frame (tools/room_views.py): a misplaced piece shows as an
        offset against the frame, not a hunch from another angle."""
        from room_views import pairs_sheet, render_views

        names = [p.name for p in self.room_frames(3)]
        for name in self.safe(self.frames_showing_added_pieces, default=[]) or []:
            if name not in names:
                names.append(name)
        pairs = render_views(self.space, names, self.space / "room-views", blender=BLENDER)
        return pairs_sheet(pairs, self.space / "room-views.png") if pairs else None

    def render_verdict(self) -> None:
        """Claude looks at the built room, from an angle and from above, next to
        a photo of the real one, and from the video's own cameras beside their
        frames (view_sheet)."""
        render = self.space / "room-render.png"
        plan = self.space / "room-render-plan.png"
        if not self.advisor.available or not render.exists():
            return
        sheet = self.safe(self.view_sheet, default=None)
        prompt = (
            f"{'Four' if sheet else 'Three'} images. The first is a perspective render of a parametric room "
            "our pipeline built from a phone video: grey shapes are walls, floor "
            "and furniture boxes. The second is the same room seen from straight "
            "above, like a floor plan. The third is a frame from the video.\n"
            + ("The fourth is a sheet of rows: on the left a video frame, on the right "
               "the room rendered from that frame's own camera position and lens, so "
               "whatever the video shows should appear in the render in the same place "
               "and at the same size, apart from what the render lacks by design "
               "(real colours, doors, windows, curtains, small objects). In these "
               "renders the floor is sand-coloured, walls pale grey and furniture "
               "blue-grey, so the line where a wall meets the floor is where the tones "
               "change. Use the rows to "
               "judge placement: a piece of furniture standing where its row's frame "
               "shows open floor, bare wall or something else, or absent from where the "
               "frame shows it, or far bigger or smaller, is misplaced; a wall likewise. "
               "Name such a problem 'misplaced: <what>, <how>'.\n" if sheet else "")
            + "How the render is drawn, so do not report these as problems: the walls "
            "between the camera and the room are hidden on purpose so the inside "
            "shows (the plan view has every wall); walls in a light beige were not "
            "seen and only close the room; furniture is drawn from simple parts, so "
            "a bed is a wide low base with a slightly smaller mattress on top and a "
            "pillow, and a wardrobe is a tall box with doors.\n"
            "Judge only what was reconstructed. Walls the camera never faced cannot "
            "appear, so a capture filmed from one spot gives a partial room; that "
            "alone does not make it implausible. Ask whether what is there sits "
            "plausibly: walls upright and meeting sensibly, furniture on the floor "
            "at a believable size and place, nothing floating or cutting through "
            "walls. List what is missing separately from what is wrong.\n"
            "Grade every problem. structural: the room itself is wrong - a wall "
            "standing free or ending in open floor, walls not meeting, the floor or "
            "ceiling running past the walls, the room far too big or small for the "
            "video, furniture floating, sunk into the floor or cutting through a wall"
            + (", or a piece the rows show standing where the video shows none, or far "
               "from where the video shows it" if sheet else "") + ". "
            "minor: everything else, such as furniture proportions or placement "
            "details. A room with any structural problem is not plausible.\n"
            'Fields: {"plausible": boolean, "problems": array of {"what": short '
            'phrase, "severity": "structural" or "minor"} (things that are wrong), '
            '"missing": array of short phrases (not captured), "advice": array of '
            'short instructions for the next capture}')
        images = [render] + ([plan] if plan.exists() else []) + self.room_frames(1) + ([sheet] if sheet else [])
        verdict = self.advisor.ask_json(prompt, images)
        if not verdict:
            print(f"    claude (blender): no usable answer"
                  + (f" ({self.advisor.reason})" if self.advisor.reason else ""))
            return None
        overruled = self.safe(self.overrule_misplacements, verdict, default=[]) or []
        if overruled:
            print("    claude (blender): overruled by the frames' masks: " + "; ".join(overruled))
        self.judged("blender", verdict,
                    ("plausible room" if verdict.get("plausible") else "not a plausible room")
                    + (f" - wrong: {', '.join(problem_text(p) for p in verdict['problems'][:3])}"
                       if verdict.get("problems") else "")
                    + (f"; missing: {', '.join(verdict.get('missing', [])[:3])}"
                       if verdict.get("missing") else ""))
        return verdict

    def phone_matches(self) -> dict[str, dict]:
        """Built boxes the phone's own perception placed an object on in this
        run (phone-objects.json): {box id: the phone's object}."""
        path = self.space / "phone-objects.json"
        if not path.exists():
            return {}
        return {o["matches"]: o for o in json.loads(path.read_text()).get("objects", []) if o.get("matches")}

    def phone_corroborates(self, label: str) -> str | None:
        """What the phone placed on a built box labelled `label`, if it placed
        something of the same furniture family there."""
        family = placement.furniture_family(self.space, label)
        if family is None:
            return None
        shapes = json.loads((self.space / "shapes.json").read_text())
        for ident, obj in self.phone_matches().items():
            box = shapes["boxes"][int(ident[1:])]
            if (box.get("build", True) and label in (box.get("label"), box.get("detected"))
                    and placement.furniture_family(self.space, obj["label"]) == family):
                return f"{obj['label']} {obj['size_m'][0]} x {obj['size_m'][1]} x {obj['size_m'][2]} m on {ident}"
        return None

    def evidence_for(self, label: str) -> dict:
        """A name's outlines in the frames (placement.mask_evidence), loaded once per run."""
        if getattr(self, "_evidence", None) is None:
            self._evidence = room_score.Evidence(self.space, log=lambda text: print("    " + text))
        return self._evidence(label)

    def measure_room(self, shapes: dict | None = None) -> dict:
        """The measured room score (room_score.py) of `shapes`, or of shapes.json
        as it stands: how well every built piece covers its outlines in the
        frames, less collisions and pieces the walk went through."""
        if shapes is None:
            shapes = json.loads((self.space / "shapes.json").read_text())
        result = room_score.room_score(self.space, shapes, self.evidence_for, room_vocabulary(self.space))
        print("    " + room_score.describe(result))
        return result

    def box_mask_score(self, label: str) -> tuple[float, int] | None:
        """How well the built box(es) labelled `label` agree with the label's
        own masks in the keyframes (pipeline/placement.py): (best score,
        agreeing frames), or None when no keyframe detected it."""
        shapes = json.loads((self.space / "shapes.json").read_text())
        boxes = [b for b in shapes["boxes"] if b.get("build", True)
                 and label in (b.get("label"), b.get("detected"))]
        if not boxes:
            return None
        evidence = self.evidence_for(label)
        if len(evidence["frames"]) < placement.MIN_FRAMES:
            return None
        best = None
        for b in boxes:
            total, per_frame = placement.score(np.array(b["min"]), np.array(b["max"]), evidence)
            agreeing = sum(1 for v in per_frame.values() if v >= 0.2)
            if best is None or total > best[0]:
                best = (total, agreeing)
        return best

    def overrule_misplacements(self, verdict: dict) -> list[str]:
        """A structural 'misplaced: <piece>' claim is checked against
        measurements: where the piece's own masks agree with its box, or the
        phone's own detector places the same kind of piece on the same box,
        the claim becomes minor. The same rows have drawn opposite verdicts on
        the same bed, and a judge has had a measured wardrobe replaced by a
        guess; measurements do not change their mind. (Frontier models asked
        which of two reconstructions is better agree with the true geometric
        metric 45.8% of the time: docs/research-weaknesses-2026-10.md.)"""
        notes = []
        for problem in verdict.get("problems") or []:
            if not isinstance(problem, dict) or problem.get("severity") != "structural":
                continue
            what = str(problem.get("what", ""))
            if not what.lower().startswith("misplaced:"):
                continue
            label = what.split(":", 1)[1].split(",")[0].strip().lower()
            scored = self.safe(self.box_mask_score, label, default=None)
            if scored and scored[0] >= placement.ACCEPT_SCORE and scored[1] >= placement.MIN_FRAMES:
                total, agreeing = scored
                problem["severity"] = "minor"
                problem["what"] = f"{what} (overruled: its masks in {agreeing} keyframes agree with the box, score {total:.2f})"
                notes.append(f"{label} stays, score {total:.2f} in {agreeing} frames")
                continue
            # A second, independent measurement of the same piece on the same spot.
            seen = self.safe(self.phone_corroborates, label, default=None)
            if seen:
                problem["severity"] = "minor"
                problem["what"] = f"{what} (overruled: the phone's own detector places a {seen})"
                notes.append(f"{label} stays, the phone places a {seen}")
        if notes and not structural_problems(verdict):
            verdict["plausible"] = True
        return notes

    def recheck_structure(self, first: dict, first_score: dict | None = None) -> None:
        """The render check found the built room structurally wrong: run the
        structure review again with those problems, rebuild, check again, and
        keep the second room when its measured score (room_score.py) is
        higher, the first when it is lower; only when the two measure the
        same does the judge decide, by which has fewer structural problems
        (the first on a tie). The first room is kept in structure-first/."""
        problems = structural_problems(first)
        kept = self.space / "structure-first"
        kept.mkdir(exist_ok=True)
        saved = [name for name in STRUCTURE_FILES if (self.space / name).exists()]
        for name in saved:
            shutil.copy2(self.space / name, kept / name)
        self.decide("shapes", "recheck",
                    "Claude's render check found structural problems ("
                    + ", ".join(problem_text(p) for p in problems[:3])
                    + "); reviewing the structure again with them")
        self.structure_review(feedback=problems)
        self.run([sys.executable, str(ROOT / "tools/classify_shapes.py"),
                  str(self.space), "--finish-only"], "finish", "after the second review")
        self.safe(self.settle_pieces)
        self.build_room()
        second = self.safe(self.render_verdict)
        second_score = self.safe(self.measure_room) if first_score else None
        if second_score:
            self.decide("shapes", "score", room_score.describe(second_score), second_score)
            outcome = room_score.compare(first_score, second_score)
            if outcome == "better":
                self.decide("shapes", "accept",
                            f"the second review raised the measured room score from "
                            f"{first_score['score']:.2f} to {second_score['score']:.2f}")
                return
            if outcome == "worse":
                for name in saved:
                    shutil.copy2(kept / name, self.space / name)
                self.judgements.append({"stage": "blender", **first, "restored": True})
                self.decide("shapes", "revert",
                            f"the second review lowered the measured room score from "
                            f"{first_score['score']:.2f} to {second_score['score']:.2f}; kept the first room")
                return
            # measured the same: what follows is the judge's call
        if second is not None and len(structural_problems(second)) < len(problems):
            self.decide("shapes", "accept",
                        f"the second review left {len(structural_problems(second))} structural "
                        f"problem(s), down from {len(problems)}")
            return
        if second is None and self.judgements and self.judgements[-1].get("stage") == "structure" \
                and self.judgements[-1].get("applied"):
            # No answer from the second check: the first room is known to be wrong and the
            # second review changed something, so the second room is the better bet.
            self.decide("shapes", "accept", "the second check gave no answer; keeping the second room, "
                        "since the first was found structurally wrong and the review changed it")
            return
        for name in saved:
            shutil.copy2(kept / name, self.space / name)
        # The gate reads the latest render check: that is the first room's again.
        self.judgements.append({"stage": "blender", **first, "restored": True})
        self.decide("shapes", "revert",
                    "the second review did not reduce the structural problems"
                    + ("" if second is not None else " (no answer from the render check)")
                    + "; kept the first room")

    def build_room(self) -> None:
        self.run([BLENDER, "--background",
                  "--python", str(ROOT / "tools/blender_room.py"), "--",
                  str(self.space / "shapes.json"), str(self.space / "room.blend"),
                  str(self.space / "room-render.png")], "blender", "build the room")

    # ------------------------------------------------------------ inspection
    def reconstruction_metrics(self) -> dict:
        sparse = self.space / "workspace" / "sparse"
        images = self.space / "workspace" / "images"
        if not sparse.exists():
            return {"models": 0, "frames": 0, "total": 0, "fraction": 0.0}
        models = solved_models(sparse)
        if not models:
            return {"models": 0, "frames": 0, "total": 0, "fraction": 0.0}
        best = max(models, key=lambda m: (m / "points3D.bin").stat().st_size)
        total = sum(1 for p in images.iterdir()
                    if p.suffix.lower() in IMAGE_EXTENSIONS) if images.exists() else 0
        frames = registered_images(best)
        return {
            "models": len(models),
            "frames": frames,
            "total": total,
            "fraction": round(frames / total, 3) if total else 0.0,
            "path_jump": round(camera_path_jump(best), 3),
            # Misplaced frames reconstruct.py removed (see detour_frames).
            "dropped_frames": (json.loads(dropped.read_text()).get(best.name, [])
                               if (dropped := self.space / "workspace"
                                   / "dropped-frames.json").exists() else []),
            "rejected_global": (self.space / "workspace" / "sparse-global-rejected").exists(),
        }

    def densify_metrics(self) -> dict:
        """densify.json without its bulky records (every detection box, the
        full object list), which belong to later stages, not to reports."""
        path = self.space / "densify.json"
        if not path.exists():
            return {}
        meta = json.loads(path.read_text())
        meta.pop("detections", None)
        objects = meta.pop("objects", None)
        if objects:
            meta["object_list"] = f"{objects.get('source')}, {len(objects.get('objects', []))} names"
        return meta

    def shape_metrics(self) -> dict:
        path = self.space / "shapes.json"
        if not path.exists():
            return {}
        shapes = json.loads(path.read_text())
        boxes = shapes.get("boxes", [])
        return {
            "planes": len(shapes.get("planes", [])),
            "walls": sum(1 for p in shapes.get("planes", [])
                         if p["kind"] == "wall" and p.get("build", True)),
            "inferred_walls": sum(1 for p in shapes.get("planes", [])
                                  if p.get("source") == "inferred"),
            "boxes": len(boxes),
            "detected_boxes": sum(1 for b in boxes if b.get("source") == "detected"),
            "built_boxes": sum(1 for b in boxes if b.get("build", True)),
            "objects": sorted({b["detected"] for b in boxes if b.get("detected")}),
        }

    # ----------------------------------------------------------------- steps
    def step_reconstruct(self) -> bool:
        """Stage 1: COLMAP places the cameras; frames it could not place are
        then placed with MapAnything where a GPU allows (fill_gaps); a map of
        what the capture covered is drawn from the result."""
        if not self.solve_with_colmap():
            return False
        self.safe(self.fill_gaps)
        self.safe(self.draw_capture_map)
        return True

    def solve_with_colmap(self) -> bool:
        started = time.time()
        ok, _ = self.run(
            [sys.executable, str(ROOT / "pipeline/reconstruct.py"), str(self.source),
             "--name", self.name, "--fps", str(self.fps)],
            "reconstruct", "SIFT features, global mapper")
        if not ok:
            self.decide("reconstruct", "stop", "COLMAP could not build any model"
                        + (f" — {self.last_error}" if self.last_error else ""))
            return False
        first = self.reconstruction_metrics()
        first_seconds = round(time.time() - started, 1)

        good = (first["fraction"] >= MIN_REGISTERED_FRACTION and first["models"] <= 2)
        if good or not self.allow_retry:
            self.decide("reconstruct", "accept",
                        f"{first['frames']}/{first['total']} frames in the best of "
                        f"{first['models']} model(s)", first, first_seconds)
            return True

        # Weak solve. Claude looks at the capture itself and picks the retry;
        # the measurement still has the final say on an obviously broken model.
        action = "retry_aliked"
        why = ("only "
               f"{first['fraction']:.0%} of frames placed; learned features grip "
               "low-texture walls better")
        verdict = self.safe(self.capture_verdict, first)
        if verdict and verdict.get("action") in {"accept", "retry_aliked", "retry_incremental"}:
            action = verdict["action"]
            why = f"Claude: {str(verdict.get('why', '')).strip()}"
            if action == "accept" and first["fraction"] < 0.35:
                action = "retry_aliked"
                why = (f"Claude wanted to accept, but only {first['fraction']:.0%} of "
                       "frames were placed, so the measurement wins")
        if action == "accept":
            self.decide("reconstruct", "accept", why, first, first_seconds)
            return True
        retry_args = (["--features", "aliked"] if action == "retry_aliked"
                      else ["--mapper", "incremental"])
        self.decide("reconstruct", action, why, first, first_seconds)
        workspace = self.space / "workspace"
        kept = self.space / "workspace-sift"
        shutil.rmtree(kept, ignore_errors=True)
        shutil.move(str(workspace), str(kept))
        kept_cloud = self.space / "cloud-sift.ply"
        if (self.space / "cloud.ply").exists():
            shutil.copy2(self.space / "cloud.ply", kept_cloud)

        ok, _ = self.run(
            [sys.executable, str(ROOT / "pipeline/reconstruct.py"), str(self.source),
             "--name", self.name, "--fps", str(self.fps), *retry_args],
            "reconstruct", " ".join(retry_args))
        second = self.reconstruction_metrics() if ok else {"frames": -1, "models": 0}

        if ok and second["frames"] >= first["frames"]:
            self.decide("reconstruct", "accept",
                        f"the retry registered {second['frames']} frames vs "
                        f"{first['frames']} on the first attempt", second)
            shutil.rmtree(kept, ignore_errors=True)
            kept_cloud.unlink(missing_ok=True)
        else:
            self.decide("reconstruct", "revert",
                        f"the retry registered {second.get('frames')} frames vs "
                        f"{first['frames']}; keeping the first attempt", second)
            # Keep the failed retry's COLMAP log: it is the only record of why.
            if (workspace / "colmap.log").exists():
                shutil.copy2(workspace / "colmap.log", self.space / "retry-failed-colmap.log")
            shutil.rmtree(workspace, ignore_errors=True)
            shutil.move(str(kept), str(workspace))
            if kept_cloud.exists():
                shutil.move(str(kept_cloud), str(self.space / "cloud.ply"))
        return True

    def step_densify(self) -> bool:
        # Which frames every later question is asked with, chosen once here:
        # stage 1 has just placed the cameras, and naming the objects is the
        # first question whose answer depends on seeing the whole room.
        if self.chosen is None:
            self.safe(self.choose_frames)
        self.safe(self.name_objects)
        ok, _ = self.run([sys.executable, str(ROOT / "pipeline/densify.py"),
                          str(self.space)], "densify", "MoGe-2 + object detection and outlines")
        if not ok:
            self.decide("densify", "stop", "densify failed"
                        + (f" — {self.last_error}" if self.last_error else ""))
            return False
        metrics = self.densify_metrics()
        spread = metrics.get("scale_spread")
        if spread is not None and spread > MAX_SCALE_SPREAD and self.allow_retry:
            # The keyframes disagree about metres: the camera model is stitched
            # from pieces at different scales. Incremental mapping is slower but
            # keeps each piece honest.
            self.decide("densify", "retry",
                        f"keyframes disagree on scale by {spread:.0%}; rebuilding "
                        f"the cameras with the incremental mapper", metrics)
            ok, _ = self.run(
                [sys.executable, str(ROOT / "pipeline/reconstruct.py"), str(self.source),
                 "--name", self.name, "--fps", str(self.fps), "--mapper", "incremental"],
                "reconstruct", "incremental mapper")
            if ok:
                ok, _ = self.run([sys.executable, str(ROOT / "pipeline/densify.py"),
                                  str(self.space)], "densify", "after remapping")
                metrics = self.densify_metrics()
        if self.safe(self.review_labels, default=False):
            # Claude revised the object list: label the cloud again with it.
            ok, _ = self.run([sys.executable, str(ROOT / "pipeline/densify.py"),
                              str(self.space)], "densify", "again, with the revised object list")
            if not ok:
                self.decide("densify", "stop", "densify failed with the revised object list"
                            + (f" — {self.last_error}" if self.last_error else ""))
                return False
            metrics = self.densify_metrics()
            labels = json.loads((self.space / "densify.json").read_text()).get("labels", {})
            self.decide("densify", "relabel",
                        "Claude revised the object list after seeing the detections; "
                        "labelled again: " + ", ".join(f"{k} {v:,}" for k, v in
                                                       sorted(labels.items(), key=lambda kv: -kv[1])[:8]),
                        metrics)
        self.decide("densify", "accept",
                    f"1 m = {metrics.get('colmap_units_per_metre', float('nan')):.3f} units, "
                    f"keyframes agree within {metrics.get('scale_spread', 0):.0%}, "
                    f"{metrics.get('points', 0):,} points", metrics)
        # The capture map again, now from the dense cloud and in measured metres.
        self.safe(self.draw_capture_map)
        return True

    def step_shapes(self) -> bool:
        # The phone's second opinion names boxes by index: one from an earlier run would
        # vouch for the wrong boxes.
        (self.space / "phone-objects.json").unlink(missing_ok=True)
        ok, _ = self.run([sys.executable, str(ROOT / "pipeline/shapes.py"),
                          str(self.space)], "shapes", "planes and labelled boxes")
        if not ok:
            self.decide("shapes", "stop", "shape detection failed"
                        + (f" — {self.last_error}" if self.last_error else ""))
            return False
        self.run([sys.executable, str(ROOT / "tools/classify_shapes.py"),
                  str(self.space), "--no-finish"], "classify", "label and sanity-check boxes")
        self.safe(self.repair_candidates)
        self.safe(self.structure_review)
        # Finish only what survived the review, so a box Claude dropped cannot
        # have pushed a wall out first.
        self.run([sys.executable, str(ROOT / "tools/classify_shapes.py"),
                  str(self.space), "--finish-only"], "finish",
                 "stand furniture on the floor, keep it inside the walls, close the room")
        self.safe(self.settle_pieces)
        measured = self.safe(self.measure_room)
        if measured:
            self.decide("shapes", "score", room_score.describe(measured), measured)
        metrics = self.shape_metrics()
        self.decide("shapes", "accept",
                    f"{metrics.get('walls', 0)} walls"
                    + (f" ({metrics['inferred_walls']} inferred)"
                       if metrics.get("inferred_walls") else "")
                    + f", {metrics.get('built_boxes', 0)} of "
                    f"{metrics.get('boxes', 0)} boxes worth building"
                    + (f"; found {', '.join(metrics['objects'])}" if metrics.get("objects") else ""),
                    metrics)
        self.build_room()
        verdict = self.safe(self.render_verdict)
        if verdict and structural_problems(verdict):
            self.safe(self.recheck_structure, verdict, measured)
        return True

    def step_splat(self) -> bool:
        """Stage 4, one step after another (SPLAT_STEPS): only the steps in
        self.splat_steps run, each picking up what the earlier ones left."""
        steps = {"train-quick": self.train_quick, "train-long": self.train_long,
                 "train-spirula": self.train_spirula,
                 "choose-training": self.pick_training, "fill": self.fill_step,
                 "choose-best": self.best_step, "scene": self.scene_step}
        for name in SPLAT_STEPS:
            if name not in self.splat_steps:
                continue
            self.current_step = name
            print(f"\n--- stage 4 step: {name}")
            if not steps[name]():
                self.current_step = None
                return False
        self.current_step = None
        return True

    def trained_splat(self, run_name: str) -> dict | None:
        """The run's splat when it exists and was trained from the current
        dense cloud (a dense cloud made later makes it stale)."""
        steps, downscale = SPLAT_RUNS[run_name]
        out = self.space / f"splat-{run_name}.ply"
        dense = self.space / "cloud-dense.ply"
        if not out.exists() or (dense.exists() and out.stat().st_mtime < dense.stat().st_mtime):
            return None
        return {"label": SPLAT_LABELS[run_name].format(steps=steps, downscale=downscale),
                "space": self.space, "ply": out}

    def seed(self) -> bool:
        """The OpenSplat project (cameras, frames, dense seed points); rebuilt
        each time, it takes seconds and a new Colab session has none."""
        ok, _ = self.run([sys.executable, str(ROOT / "pipeline/splat_seed.py"),
                          str(self.space)], "splat seed", "dense cloud as starting points")
        if not ok:
            self.decide("splat", "stop", "could not build the splat project")
        return ok

    def train(self, run_name: str, resume: Path | None = None, save_every: int = -1) -> bool:
        steps, downscale = SPLAT_RUNS[run_name]
        out = self.space / f"splat-{run_name}.ply"
        args = [OPENSPLAT, str(self.space / "splat-project"), "-n", str(steps),
                "-d", str(downscale), "-o", str(out)]
        if save_every > 0:
            args += ["-s", str(save_every)]
        if resume:
            args += ["--resume", str(resume)]
        if not self.cuda and self.image_cache_gb(downscale) > MAX_GPU_IMAGE_CACHE_GB:
            args.append("--no-gpu-cache")
        ran, _ = self.run(args, "splat", f"{run_name}: {steps} steps at 1/{downscale} resolution"
                          + (f", resuming from {resume.name}" if resume else ""))
        return ran and out.exists()

    def train_quick(self) -> bool:
        if self.trained_elsewhere or (not self.retrain and self.trained_splat("quick")):
            if self.trained_splat("quick"):
                self.decide("splat", "reuse", "splat-quick.ply is already trained from this dense cloud")
                return True
            self.decide("splat", "stop", "no splat-quick.ply trained from this dense cloud")
            return False
        if not self.seed():
            return False
        ok = self.train("quick")
        self.decide("splat", "accept" if ok else "stop",
                    "trained the quick splat" if ok else "OpenSplat failed; see the log")
        return ok

    def train_long(self) -> bool:
        """The long training, resumable: OpenSplat saves splat-long_<step>.ply
        every LONG_CHECKPOINT_STEPS, and a new attempt resumes from the newest
        save trained from this dense cloud."""
        if self.trained_elsewhere or not self.long_splat:
            self.decide("splat", "skip", "no long training here"
                        + ("" if self.trained_elsewhere else " (--long-splat on to train it)"))
            return True
        if not self.retrain and self.trained_splat("long"):
            self.decide("splat", "reuse", "splat-long.ply is already trained from this dense cloud")
            return True
        dense = self.space / "cloud-dense.ply"
        saves = sorted((p for p in self.space.glob("splat-long_*.ply")
                        if p.stem.rsplit("_", 1)[1].isdigit()
                        and (not dense.exists() or p.stat().st_mtime > dense.stat().st_mtime)),
                       key=lambda p: int(p.stem.rsplit("_", 1)[1]))
        resume = saves[-1] if saves and not self.retrain else None
        if not self.seed():
            return False
        ok = self.train("long", resume=resume, save_every=LONG_CHECKPOINT_STEPS)
        self.decide("splat", "accept" if ok else "warn",
                    "trained the long splat" if ok else
                    "the long training failed; the quick splat stays (see the log)")
        return True  # a failed long run leaves the quick splat to carry on with

    def train_spirula(self) -> bool:
        """Spirula Studio's training of the same cameras and seed
        (pipeline/splat_spirula.py); a failure leaves the other splats to carry on."""
        from splat_spirula import DEFAULTS, find_binary

        if self.trained_elsewhere or not self.spirula:
            self.decide("splat", "skip", "no Spirula training here"
                        + ("" if self.trained_elsewhere else
                           " (--spirula on to train it)" if not self.spirula_requested else
                           " (Spirula Studio is not installed; see pipeline/splat_spirula.py)"))
            return True
        if not self.retrain and self.trained_splat("spirula"):
            self.decide("splat", "reuse", "splat-spirula.ply is already trained from this dense cloud")
            return True
        if not self.seed():
            return False
        steps, downscale = SPLAT_RUNS["spirula"]
        ok, _ = self.run([sys.executable, str(ROOT / "pipeline/splat_spirula.py"), str(self.space),
                          "--iters", str(steps), "--divisor", str(downscale),
                          "--cap", str(DEFAULTS["cap"]), "--depth-weight", str(DEFAULTS["depth_weight"]),
                          "--floaters", DEFAULTS["floaters"]],
                         "splat", f"Spirula: {steps} steps at 1/{downscale} resolution")
        ok = ok and (self.space / "splat-spirula.ply").exists()
        self.decide("splat", "accept" if ok else "warn",
                    "trained the Spirula splat" if ok else
                    "the Spirula training failed; the other splats stay (see the log)")
        return True

    def pick_training(self) -> bool:
        """Claude chooses between the trained splats; the choice becomes
        splat.ply, exported for the viewer with Claude's opening view."""
        trained = [t for t in (self.trained_splat(run) for run in SPLAT_RUNS) if t]
        if not trained:
            self.decide("splat", "stop", "no splat trained from this dense cloud: run train-quick first")
            return False
        best = trained[0]
        if len(trained) > 1:
            best = self.safe(self.choose_training, trained, default=trained[0])
        shutil.copyfile(best["ply"], self.space / "splat.ply")
        self.decide("splat", "accept", f"splat.ply is the {best['label']} splat"
                    + ("" if len(trained) > 1 else " (the only one trained)"))
        # The viewer loads the compact copy; the .ply stays the full result.
        self.run([sys.executable, str(ROOT / "pipeline/splat_export.py"),
                  str(self.space / "splat.ply")], "splat export", "compact copy for the viewer")
        self.safe(self.choose_start_view, self.space / "splat.ply")
        return True

    def fill_step(self) -> bool:
        if not (self.space / "splat.ply").exists():
            self.decide("splat", "stop", "no splat.ply yet: run choose-training first")
            return False
        self.safe(self.fill_surfaces)
        fill = self.space / "splat-filled.fill.json"
        if fill.exists() and json.loads(fill.read_text()).get("kept"):
            self.safe(self.choose_start_view, self.space / "splat-filled.ply")
        return True

    def best_step(self) -> bool:
        if (self.space / "splat.ply").exists():
            self.safe(self.choose_best)
        return True

    def scene_step(self) -> bool:
        """The room as a mixed scene (tools/mixed_scene.py): its surfaces as
        textured meshes, the scanned furniture as movable pieces, a clean model
        beside each. Claude looks at every texture against what was filmed and
        every piece against a frame of the real thing, and decides what to keep,
        what to paint plain and which pieces to show as models. An extra: it
        never ends the run, and what came of it is in the report either way."""
        missing = [f for f in ("splat.ply", "shapes.json") if not (self.space / f).exists()]
        if missing:
            self.decide("splat", "skip", f"no scene: {' and '.join(missing)} missing "
                        "(the scene is cut from the chosen splat along stage 3's room model)")
            return True
        started = time.time()
        try:
            self.build_scene(started)
        except (Exception, SystemExit) as exc:
            self.decide("splat", "skip", f"no scene: the build failed ({type(exc).__name__}: {exc})",
                        seconds=round(time.time() - started, 1))
        return True

    def build_scene(self, started: float) -> None:
        from mixed_scene import build, claude_review
        from surface_fill import LAMA_PATH

        if not LAMA_PATH.exists():
            self.decide("splat", "skip", f"no scene: LaMa weights not found at {LAMA_PATH}")
            return
        seen = {}
        ask = claude_review(self.advisor, self.room_frames(2), log=lambda text: print("   " + text))

        def review(*sheets):
            seen["verdict"] = ask(*sheets)
            return seen["verdict"]

        out = build(self.space, 0.005, "splat.ply", log=lambda text: print("   " + text),
                    review=review if self.advisor.available else None)
        verdict = seen.get("verdict")
        if verdict:
            changed = [f"{name}: {v.get('use')}" for name, v in (verdict.get("surfaces") or {}).items() if v.get("use") != "keep"]
            changed += [f"{ident}: {v.get('use')}" for ident, v in (verdict.get("pieces") or {}).items() if v.get("use") != "scan"]
            changed += [f"{name}: {v.get('use')}" for name, v in (verdict.get("faces") or {}).items() if v.get("use") != "keep"]
            self.judged("scene", {"surfaces": verdict.get("surfaces"), "pieces": verdict.get("pieces"),
                                  "faces": verdict.get("faces"), "summary": verdict.get("summary")},
                        f"{verdict.get('summary', '')}" + (f" ({'; '.join(changed)})" if changed else ""))
        pieces = json.loads((out / "scene.json").read_text())["pieces"]
        movable = [p for p in pieces if p["movable"]]
        as_model = [p["label"] for p in movable if p.get("show") == "model"]
        photographed = sum(len(p.get("photographed", [])) for p in movable)
        fixed = sum(p["count"] for p in pieces if not p["movable"])
        self.decide("splat", "accept",
                    f"scene built: {len(movable)} movable piece(s) ({', '.join(p['label'] for p in movable) or 'none'})"
                    + (f", shown as clean models: {', '.join(as_model)}" if as_model else "")
                    + (f", {photographed} model side(s) photographed from the video" if photographed else "")
                    + f"; {fixed:,} Gaussians left fixed; "
                    + ("reviewed by Claude" if verdict else "not reviewed: every surface and piece kept as built")
                    + f" ({(out / 'scene.json').relative_to(self.space)})",
                    metrics={"movable": len(movable), "as_model": len(as_model), "photographed_sides": photographed,
                             "fixed_gaussians": fixed,
                             "reviewed": bool(verdict)},
                    seconds=round(time.time() - started, 1))

    def image_cache_gb(self, downscale: int) -> float:
        """What OpenSplat's GPU copy of the training images would take."""
        from PIL import Image

        images = sorted((self.space / "workspace" / "images").glob("*.jpg"))
        if not images:
            return 0.0
        width, height = Image.open(images[0]).size
        return len(images) * (width // downscale) * (height // downscale) * 3 * 4 / 1e9

    def choose_training(self, trained: list[dict]) -> dict:
        """Claude compares the trained splats at trained and new views
        (tools/splat_choose.py), in context: the video's currently published
        best splat is shown beside them as a reference (Claude judged every
        splat of a video together reliably, and one pair alone less so). Its
        pick among this run's splats stands unless it measures clearly worse at
        the new views (splat_choose.guarded); the choice becomes splat.ply."""
        from splat_choose import guarded, judge, published_best

        reference = self.safe(published_best, self.source)
        # A published best from this same space is one of this run's own splats.
        shown = trained + ([reference] if reference and reference["space"].resolve() != self.space.resolve()
                           else [])
        record = judge(self.source, shown, self.space / "training-compare",
                       log=lambda text: print("   " + text), advisor=self.advisor)
        guarded(record, {t["label"] for t in trained})
        (self.space / "splat-training.json").write_text(json.dumps(record, indent=1) + "\n")
        verdict = record.get("claude") or {}
        self.judged("splat training", {"ranking": verdict.get("ranking"),
                                       "reasons": verdict.get("reasons"),
                                       "candidates": record["candidates"],
                                       "overruled": record.get("overruled")},
                    f"kept {record['chosen']}"
                    + (f" ({record['overruled']})" if record.get("overruled") else
                       f" (decided by {record['decided_by']}"
                       + (f"; {verdict['agreement']}" if verdict.get("agreement") else "") + ")")
                    + (f"; best of all shown: {record['best']}" if len(shown) > len(trained) else ""))
        return next(t for t in trained if t["label"] == record["chosen"])

    def choose_start_view(self, ply: Path) -> None:
        """Claude picks the view a splat opens at: the best-scoring start
        views from different capture positions are rendered side by side, next
        to frames of the real room, and the chosen one is written to the
        splat's .view.json (the choice and the options to .view-choice.json)."""
        from PIL import Image, ImageDraw, ImageFont
        from splat_export import start_views
        from splat_render import render
        from splat_tools import read_splat

        if not self.advisor.available or not ply.exists():
            return
        views = start_views(self.space, ply, START_VIEW_CHOICES)
        if len(views) < 2:
            return
        arr, _ = read_splat(ply)
        letters = "ABCDEFGH"[:len(views)]
        cols = 3
        rows = (len(views) + cols - 1) // cols
        w, h = START_VIEW_TILE
        sheet = Image.new("RGB", (cols * (w + 8), rows * (h + 30)), "white")
        draw = ImageDraw.Draw(sheet)
        font = ImageFont.load_default(size=20)
        for n, (letter, view) in enumerate(zip(letters, views)):
            image = render(arr, view["viewMatrix"], w, h, view["fovY"])
            x, y = (n % cols) * (w + 8), (n // cols) * (h + 30)
            sheet.paste(Image.fromarray(image), (x, y + 30))
            draw.text((x + 4, y + 4), letter, fill="black", font=font)
        sheet_path = self.space / f"{ply.stem}-start-views.png"
        sheet.save(sheet_path)
        verdict = self.advisor.ask_json(START_VIEW_PROMPT.format(count=len(views), letters=", ".join(letters)),
                                        [sheet_path] + self.room_frames(2), max_tokens=1000)
        pick = verdict.get("best") if verdict else None
        chosen = views[letters.index(pick)] if pick in letters else views[0]
        view_path = ply.with_name(ply.stem + ".view.json")
        view_path.write_text(json.dumps({**chosen, "chosenBy": "claude" if pick in letters else "score"}) + "\n")
        ply.with_name(ply.stem + ".view-choice.json").write_text(json.dumps(
            {"options": {letter: {k: v[k] for k in ("frame", "turn", "backMetres", "coverage",
                                                    "blurShare", "score")}
                         for letter, v in zip(letters, views)},
             "claude": verdict, "chosen": pick if pick in letters else "A"}, indent=1) + "\n")
        if verdict:
            self.judged("start view", {"splat": ply.name, **verdict},
                        f"{ply.name} opens at {pick} (near {chosen['frame']}): {verdict.get('why', '')}")

    def choose_best(self) -> None:
        """Compare every finished splat of this video (this run's trained and
        filled splats, and earlier runs), let Claude pick, and publish the best
        to splats/<video>/best.splat (tools/splat_choose.py)."""
        from splat_choose import choose

        record = choose(self.source, log=lambda text: print("   " + text), advisor=self.advisor)
        if not record:
            return
        if record.get("claude"):
            self.judged("choose", {"ranking": record["claude"].get("ranking"),
                                   "reasons": record["claude"].get("reasons"),
                                   "candidates": record["candidates"]},
                        f"best splat of {self.source.name}: {record['best']}")
        if record.get("kept_earlier"):
            self.decide("splat", "keep",
                        f"Claude did not decide this time, so {record['best']} stays the best splat of "
                        f"{self.source.name} as Claude chose earlier (the numbers alone would pick "
                        f"{record['numbers_now']})", {"best_splat": record["best"]})
            return
        self.decide("splat", "accept", f"published {record['best']} as the best splat of "
                    f"{self.source.name} (decided by {record['decided_by']}"
                    + (f"; {record['claude']['agreement']}" if (record.get("claude") or {}).get("agreement") else "")
                    + ")",
                    {"best_splat": record["best"]})

    def fill_surfaces(self) -> None:
        """Fill floor, wall and flat furniture faces the splat has no blobs for
        (tools/surface_fill.py), judged surface by surface: each filled surface
        is rendered before and after from a video frame that looks at it,
        beside that real frame; Claude keeps or rejects each one, and a
        measured damage check can veto. Only kept surfaces go into
        splat-filled; choose_best then decides between it and the trained splat."""
        import numpy as np
        from splat_edit import Room, save
        from splat_tools import read_splat
        from surface_fill import LAMA_PATH, fill_room

        record_path = self.space / "splat-filled.fill.json"
        if not LAMA_PATH.exists():
            self.decide("splat", "skip", f"no fill: LaMa weights not found at {LAMA_PATH}")
            return
        if not (self.space / "shapes.json").exists():
            self.decide("splat", "skip", "no fill: stage 3's room model is needed to know the surfaces")
            return
        print("\n=== fill: floor, walls and flat furniture faces the splat is missing")
        started = time.time()
        room = Room(self.space)
        original, trailing = read_splat(self.space / "splat.ply")
        pieces: list = []
        fill_room(room, original, floor=True, walls=True, objects=True, pieces=pieces,
                  log=lambda text: self.log(text))
        reviewed = [p for p in pieces if p["blobs"] is not None and len(p["blobs"]) >= FILL_MIN_BLOBS]
        print(f"    {len(pieces)} surfaces looked at, {len(reviewed)} with a fill worth reviewing "
              f"({time.time() - started:.0f}s)")
        sheets, rows = [], []
        for n, piece in enumerate(reviewed, 1):
            row = self.review_surface(n, piece, original, room)
            if row:
                rows.append(row)
                sheets.append(row["sheet"])

        verdicts = {}
        if rows and self.advisor.available:
            answer = self.advisor.ask_json(FILL_REVIEW_PROMPT.format(
                surfaces=json.dumps([{"id": r["id"], "surface": r["surface"],
                                      "seen_by_video": not r["unseen"]} for r in rows])),
                sheets, max_tokens=2000)
            for item in (answer or {}).get("surfaces") or []:
                verdicts[str(item.get("id"))] = item
        kept_pieces = []
        by_surface = {p["surface"]: p for p in reviewed}
        for row in rows:
            piece = by_surface[row["surface"]]
            claude = verdicts.get(row["id"])
            numbers_ok = row["gaps_after"] <= row["gaps_before"] and row["damaged"] <= FILL_MAX_DAMAGED_SHARE
            # Numbers cannot see blotches or pasted-on patches, which is what the
            # review is for. Without Claude's verdict a fill is kept only when the
            # run was started without Claude on purpose (--no-claude), never
            # because Claude could not be reached (a dropped connection kept fills
            # Claude had rejected).
            keep = numbers_ok and (bool(claude.get("keep")) if claude else not self.use_claude)
            row.update({"keep": keep, "numbers_ok": numbers_ok,
                        "claude": (claude or {}).get("why"), "sheet": str(row["sheet"])})
            if keep:
                kept_pieces.append(piece)
        for row in rows:
            print(f"    {row['id']} {row['surface']}: {'keep' if row['keep'] else 'reject'} - "
                  f"gaps {row['gaps_before']:.1%} -> {row['gaps_after']:.1%}, damage {row['damaged']:.2%}"
                  + (f"; claude: {row['claude']}" if row["claude"] else ""))
        if verdicts:
            self.judged("fill", {"surfaces": [{k: r[k] for k in ("id", "surface", "keep", "claude")}
                                              for r in rows]},
                        f"kept {len(kept_pieces)} of {len(rows)} filled surfaces")

        record = {"surfaces": [{k: v for k, v in r.items()} for r in rows], "kept": bool(kept_pieces)}
        if kept_pieces:
            remove = np.zeros(len(original), bool)
            for piece in kept_pieces:
                remove |= piece["remove"]
            result = np.concatenate([original[~remove], *[p["blobs"] for p in kept_pieces]])
            save(self.space, result, trailing, None, room, name="splat-filled")
            record["added"] = int(sum(len(p["blobs"]) for p in kept_pieces))
            record["removed"] = int(remove.sum())   # haze and remnants the kept fills replace
        record_path.write_text(json.dumps(record, indent=1) + "\n")
        self.decide("splat", "accept" if kept_pieces else "skip",
                    (f"kept the fill on {', '.join(p['surface'] for p in kept_pieces)}"
                     if kept_pieces else "kept no fill: no surface was clearly better")
                    + ("" if verdicts else
                       " (not reviewed by Claude)" if not self.use_claude else
                       f" (Claude could not be reached: {self.advisor.reason}; unreviewed fills are not kept)"),
                    {"fill_surfaces_kept": [p["surface"] for p in kept_pieces]})

    def review_surface(self, n: int, piece: dict, original, room) -> dict | None:
        """Render one filled surface before and after from the video frame that
        looks most squarely at it, beside that frame; measure the change."""
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont
        from scipy.ndimage import distance_transform_edt
        from splat_choose import TILE_H, TILE_W, camera_views
        from splat_export import model_dir
        from splat_render import render

        world = np.array(room.shapes["world"])
        normal = world.T @ np.asarray(piece["normal"])          # scene -> splat frame
        # Look at where the fill went (its solid blobs), not the middle of the whole
        # surface, which for a floor sits under the bed.
        blobs = piece["blobs"]
        solid = blobs["opacity"] > 1.0
        chosen = blobs[solid] if solid.sum() > 50 else blobs
        centre = np.median(np.stack([chosen["x"], chosen["y"], chosen["z"]], axis=1), axis=0).astype(float)
        model = model_dir(self.space)
        cameras = read_cameras_bin(model / "cameras.bin")
        best, best_score = None, -1.0
        for info in read_images_bin(model / "images.bin").values():
            cam = cameras[info["camera_id"]]
            local = info["R"] @ centre + info["t"]
            if local[2] <= 0.5 * room.metre:
                continue
            u = cam["params"][0] * local[0] / local[2] / (cam["width"] / 2)
            v = cam["params"][1] * local[1] / local[2] / (cam["height"] / 2)
            if abs(u) > 0.8 or abs(v) > 0.8:
                continue                                    # not well inside this frame
            position = -info["R"].T @ info["t"]
            ray = (centre - position) / np.linalg.norm(centre - position)
            facing = float(-ray @ normal)
            if facing < 0.15:
                continue
            score = facing * (1 - 0.5 * max(abs(u), abs(v))) / max(local[2] / room.metre, 1.0) ** 0.5
            if score > best_score:
                best, best_score = info, score
        unseen = best is None
        if unseen:
            # No frame looks at it (a fill goes where the video saw nothing clearly):
            # look at it from the nearest point of the camera path, 1-3 m away.
            from splat_export import look_matrix, view_json

            infos = list(read_images_bin(model / "images.bin").values())
            dist = [np.linalg.norm(-i["R"].T @ i["t"] - centre) / room.metre for i in infos]
            options = [(d, i) for d, i in zip(dist, infos) if 1.0 <= d <= 3.0]
            if not options:
                print(f"    {piece['surface']}: nowhere on the camera path to see it from; not reviewed")
                return None
            _, best = min(options, key=lambda o: o[0])
            position = -best["R"].T @ best["t"]
            view = view_json(look_matrix(position, centre - position, world[2]), position)["viewMatrix"]
            fov = 60.0
        else:
            view, fov = camera_views(self.space)[best["name"]]
        after = np.concatenate([original[~piece["remove"]], piece["blobs"]])
        img_b, cov_b = render(original, view, TILE_W, TILE_H, fov, with_coverage=True)
        img_a, cov_a = render(after, view, TILE_W, TILE_H, fov, with_coverage=True)
        solid = (cov_b > 0.9) & (distance_transform_edt(cov_b >= 0.5) > FILL_EDGE_PX)
        diff = np.abs(img_a.astype(int) - img_b.astype(int)).max(axis=-1)
        frame = Image.open(self.space / "workspace" / "images" / best["name"]).convert("RGB") \
            .resize((TILE_W, TILE_H))
        sheet = Image.new("RGB", (3 * (TILE_W + 8), TILE_H + 26), "white")
        draw = ImageDraw.Draw(sheet)
        font = ImageFont.load_default(size=15)
        first = "nearest video frame (other angle)" if unseen else "video"
        for k, (image, label) in enumerate(((frame, first), (Image.fromarray(img_b), "before"),
                                            (Image.fromarray(img_a), "after"))):
            sheet.paste(image, (k * (TILE_W + 8), 26))
            draw.text((k * (TILE_W + 8) + 4, 4), f"S{n} {label}" if k == 0 else label,
                      fill="black", font=font)
        path = self.space / f"fill-surface-{n}.png"
        sheet.save(path)
        return {"id": f"S{n}", "surface": piece["surface"], "frame": best["name"], "unseen": unseen,
                "blobs": len(piece["blobs"]), "sheet": path,
                "gaps_before": round(float((cov_b < 0.5).mean()), 4),
                "gaps_after": round(float((cov_a < 0.5).mean()), 4),
                "damaged": round(float((diff[solid] > FILL_DAMAGE_STEP).mean()) if solid.any() else 0.0, 4)}

    # ---------------------------------------------------------------- advice
    def capture_advice(self, recon: dict, dense: dict, shapes: dict, row: dict) -> None:
        if recon.get("rejected_global"):
            self.advice.append(
                "The global solve joined pieces that do not belong together, so the "
                "incremental mapper was used. Walk a closed loop and return to where "
                "you started, so the pieces have overlap to connect through.")
        if recon.get("fraction", 1) < 0.9:
            self.advice.append(
                f"Only {recon.get('fraction', 0):.0%} of frames were placed. Walk "
                "slower with more overlap between frames, and never pan in place.")
        if (dense.get("scale_spread") or 0) > 0.15:
            self.advice.append(
                f"Keyframes disagreed about scale by {dense['scale_spread']:.0%}, which "
                "means thin geometry. More parallax (move sideways, not just forward) helps.")
        if dense.get("points_dropped_unreliable"):
            self.advice.append(
                f"{dense['points_dropped_unreliable']:,} points were dropped on mirrors, "
                "windows and screens. Close curtains and avoid filming into mirrors.")
        if shapes and not shapes.get("objects"):
            self.advice.append(
                "No furniture was recognised. Film each piece of furniture from closer, "
                "with the lights on.")
        if (row.get("wall_rms_pct") or 0) > NOISY_WALL_RMS_PCT:
            self.advice.append(
                f"Walls came out rough ({row['wall_rms_pct']:.2f}% of the room's size). "
                "Blank walls need texture: film along them, and add a few pieces of "
                "painter's tape or sticky notes.")

    # ------------------------------------------------------------------- run
    # ----------------------------------------------------------------- gates
    def gate(self, stage: str) -> tuple[str, str]:
        """Is this stage's output right? Returns (pass | warn | stop, why)."""
        if stage == "reconstruct":
            m = self.reconstruction_metrics()
            if m["models"] == 0:
                return "stop", "no camera model was built"
            if m["fraction"] >= MIN_REGISTERED_FRACTION:
                return "pass", (f"{m['frames']}/{m['total']} frames placed in "
                                + ("one model" if m["models"] == 1 else
                                   f"the best of {m['models']} models"))
            if m["fraction"] >= PARTIAL_REGISTERED_FRACTION:
                return "warn", (f"only {m['fraction']:.0%} of frames placed: the model is "
                                "coherent but covers part of the room")
            return "stop", (f"only {m['fraction']:.0%} of frames placed, so later stages "
                            "would rebuild a fragment; re-shoot instead")
        if stage == "densify":
            d = self.densify_metrics()
            if not d:
                return "stop", "no dense cloud was written"
            spread = d.get("scale_spread") or 0.0
            if spread > MAX_SCALE_SPREAD:
                return "stop", (f"keyframes disagree on scale by {spread:.0%}, so the "
                                "cameras are inconsistent")
            if d.get("points", 0) < MIN_DENSE_POINTS:
                return "stop", f"only {d.get('points', 0):,} dense points"
            summary = (f"1 m = {d.get('colmap_units_per_metre', 0):.3f} units, scale "
                       f"agreement {spread:.0%}, {d['points']:,} points")
            if d.get("depth_model") == "moge" and "labels" not in d:
                return "warn", ("object detection did not run, so the cloud has no "
                                "labels and mirrors/windows were not filtered; " + summary)
            if d.get("depth_model") == "moge" and d.get("outlines") == "boxes":
                return "warn", ("objects were labelled by whole detection rectangles, "
                                "not their outlines, so furniture boxes take in the floor "
                                "and walls around them; " + summary)
            if spread > GOOD_SCALE_SPREAD:
                return "warn", "keyframes only loosely agree on scale: " + summary
            return "pass", summary
        if stage == "shapes":
            s = self.shape_metrics()
            if not s or s["planes"] < 2:
                return "stop", "fewer than two planes found, so there is no room shell"
            verdict = next((j for j in reversed(self.judgements)
                            if j["stage"] == "blender"), None)
            # A check that never ran is not a pass.
            unchecked = [name for stage_name, name in (("structure", "structure review"),
                                                      ("blender", "render check"))
                         if not any(j["stage"] == stage_name for j in self.judgements)]
            if self.use_claude and unchecked:
                return "warn", ("Claude's " + " and ".join(unchecked) + " gave no answer"
                                + (f" ({self.advisor.reason})" if self.advisor.reason else "")
                                + ", so the built room was not checked")
            if verdict is not None:
                structural = [p for p in verdict.get("problems") or []
                              if isinstance(p, dict) and p.get("severity") == "structural"]
                # Claude may call a room plausible and still list a free-standing
                # wall; a structural problem warns whatever the yes/no says.
                if verdict.get("plausible") is False or structural:
                    shown = structural or verdict.get("problems") or []
                    return "warn", ("Claude found the built room wrong: "
                                    + ", ".join(problem_text(p) for p in shown[:3]))
            if s["walls"] < 2:
                return "warn", f"only {s['walls']} wall(s) found"
            if not s["objects"]:
                return "warn", "no furniture was recognised"
            return "pass", (f"{s['walls']} walls, {s['built_boxes']} boxes to build "
                            f"({', '.join(s['objects'])})")
        if stage == "splat":
            count = ply_vertex_count(self.space / "splat.ply")
            done = ", ".join(self.splat_steps)
            if not count:
                trained = [r for r in SPLAT_RUNS if self.trained_splat(r)]
                if trained and "choose-training" not in self.splat_steps:
                    return "pass", (f"steps {done} done: trained {', '.join(trained)}; "
                                    "next: choose-training")
                return "stop", "no splat was written"
            if count < MIN_SPLAT_GAUSSIANS:
                return "warn", f"only {count:,} gaussians"
            fill = self.space / "splat-filled.fill.json"
            kept = fill.exists() and json.loads(fill.read_text()).get("kept")
            return "pass", (f"{count:,} gaussians"
                            + ("; gaps filled (view splat-filled.splat)" if kept else ""))
        raise ValueError(stage)

    def missing_prerequisite(self, stage: str) -> str | None:
        if stage in ("densify", "splat") and self.reconstruction_metrics()["models"] == 0:
            return "no camera model yet: run the reconstruct stage first"
        if stage in ("shapes", "splat") and not (self.space / "cloud-dense.ply").exists():
            return "no dense cloud yet: run the densify stage first"
        return None

    def run_stage(self, stage: str) -> str:
        missing = self.missing_prerequisite(stage)
        if missing:
            status, why = "stop", missing
        else:
            # A step may shrug off a helper's failure; don't let that error
            # label a later stage that stops without running anything.
            self.last_error = ""
            step = {"reconstruct": self.step_reconstruct, "densify": self.step_densify,
                    "shapes": self.step_shapes, "splat": self.step_splat}[stage]
            status, why = (self.gate(stage) if step() else
                           ("stop", f"{stage} did not complete"
                            + (f" — {self.last_error}" if self.last_error else "")))
        self.gates[stage] = {"status": status, "why": why}
        print(f"\n*** {stage}: {status.upper()} - {why}")
        return status

    # ------------------------------------------------------------------- run
    def load_earlier_stages(self, first: str) -> None:
        """Keep the record of stages before `first`; later ones are now stale.
        When stage 4 starts at a later step, the records of its earlier steps
        are kept too."""
        path = self.space / "agent-report.json"
        if not path.exists() or first == STAGES[0]:
            return
        earlier = set(STAGES[:STAGES.index(first)])
        earlier_steps = (set(SPLAT_STEPS[:SPLAT_STEPS.index(self.splat_steps[0])])
                         if first == "splat" else set())

        def kept(entry, stage):
            return stage in earlier or (stage == "splat" and entry.get("step") in earlier_steps)

        previous = json.loads(path.read_text())
        self.stages = [e for e in previous.get("decisions", []) if kept(e, e["stage"])]
        self.judgements = [j for j in previous.get("judgements", [])
                           if kept(j, JUDGEMENT_STAGE.get(j["stage"], j["stage"]))]
        self.gates = {k: v for k, v in previous.get("gates", {}).items() if k in earlier}

    def go(self, stages: list[str], accept_warnings: bool = False) -> int:
        started = time.time()
        self.load_earlier_stages(stages[0])
        print(f"Agent: {self.source.name} -> spaces/{self.name}, "
              f"stage(s): {', '.join(stages)}")
        for stage in stages:
            status = self.run_stage(stage)
            if status == "stop" or (status == "warn" and not accept_warnings):
                return self.finish(started, stages, stopped_at=stage)
        return self.finish(started, stages)

    def finish(self, started: float, stages: list[str], stopped_at: str | None = None) -> int:
        recon, dense, shapes = (self.reconstruction_metrics(), self.densify_metrics(),
                                self.shape_metrics())
        row = evaluate(self.space) if (self.space / "workspace").exists() else {}
        last = "shapes" if not self.do_splat else "splat"
        # Stage 4 run step by step is over only after its last step; advice after
        # every step would ask Claude the same question again each time.
        splat_done = stages[-1] == "splat" and self.splat_steps[-1] == SPLAT_STEPS[-1]
        run_is_over = stopped_at is not None or (stages[-1] == last and last != "splat") or splat_done

        self.advice = []
        if run_is_over:
            self.capture_advice(recon, dense, shapes, row)
            for j in self.judgements:
                if j["stage"] == "densify" and j.get("missed") and not j.get("applied"):
                    self.advice.append(
                        "The detector missed " + ", ".join(map(str, j["missed"][:6]))
                        + ", so they stay leftover points instead of furniture.")
                if j["stage"] == "blender" and j.get("plausible") is False:
                    self.advice += [a for a in (j.get("advice") or [])[:3] if isinstance(a, str)]
            if self.advisor.available:
                written = self.safe(self.advisor.ask,
                    "A room capture was processed by our 3D pipeline. Measurements:\n"
                    f"cameras {json.dumps(recon)}\ndepth {json.dumps(dense)}\n"
                    f"structure {json.dumps(shapes)}\n"
                    "Write at most four short, concrete instructions for re-shooting this "
                    "room so the reconstruction comes out better. One instruction per "
                    "line, no numbering, no preamble.")
                for line in (written or "").splitlines():
                    line = line.strip("-* \t")
                    if len(line) > 15:
                        self.advice.append(line)

        report = {
            "space": self.name,
            "source": str(self.source),
            "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
            "minutes_this_run": round((time.time() - started) / 60, 1),
            "stopped_at": stopped_at,
            "gates": self.gates,
            "claude": {"backend": self.advisor.backend, "calls": self.advisor.calls,
                       "note": self.advisor.reason},
            "decisions": self.stages,
            "judgements": self.judgements,
            "evaluation": row,
            "capture_advice": self.advice,
        }
        self.space.mkdir(parents=True, exist_ok=True)  # a stage run before its prerequisites
        (self.space / "agent-report.json").write_text(json.dumps(report, indent=2) + "\n")

        print(f"\n{'=' * 70}\nAgent report for {self.name} "
              f"({report['minutes_this_run']} min this run)")
        print(f"  claude: {self.advisor.backend}, {self.advisor.calls} call(s) this run"
              + (f" - {self.advisor.reason}" if self.advisor.reason else ""))
        for stage in STAGES:
            if stage in self.gates:
                g = self.gates[stage]
                print(f"  {stage:12s} {g['status'].upper():5s} {g['why']}")
        if stopped_at:
            g = self.gates[stopped_at]
            print(f"  stopped at {stopped_at} ({g['status']}); nothing after it was run")
        else:
            following = STAGES[STAGES.index(stages[-1]) + 1:] if stages[-1] in STAGES else []
            if following and (following[0] != "splat" or self.do_splat):
                print(f"  next: --stage {following[0]}")
        if row and run_is_over:
            rms = row.get("wall_rms_pct")
            print(f"  result: {row.get('frames')}/{row.get('total')} frames, "
                  f"{row.get('dense') or 0:,} dense points, walls "
                  f"{f'{rms:.2f}% rms' if rms is not None else 'not measured'}, "
                  f"splat {row.get('splat') or 0:,} gaussians")
        if self.advice:
            print("  capture advice:")
            for item in self.advice:
                print(f"    - {item}")
        print(f"  report: {self.space / 'agent-report.json'}")
        if stopped_at:
            return 1 if self.gates[stopped_at]["status"] == "stop" else 3
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", help="video file or folder of photos")
    parser.add_argument("--name", required=True, help="space name")
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--stage", choices=STAGES,
                        help="run only this stage, check its output, and stop "
                             "(run the stages one by one this way)")
    parser.add_argument("--accept-warnings", action="store_true",
                        help="carry on past a stage whose output is questionable")
    parser.add_argument("--no-splat", action="store_true",
                        help="stop after the Blender room (splat training is the slow part)")
    parser.add_argument("--no-retry", action="store_true",
                        help="run each stage once, never retry with other settings")
    parser.add_argument("--no-claude", action="store_true",
                        help="decide from the measurements alone, without asking Claude")
    parser.add_argument("--trained-elsewhere", action="store_true",
                        help="stage 4 uses splat-quick.ply / splat-long.ply already in the space "
                             "(trained on a Colab GPU) instead of training")
    parser.add_argument("--splat-steps", nargs="+", choices=SPLAT_STEPS, default=None,
                        help="with --stage splat: run only these steps of stage 4, in order "
                             "(" + ", ".join(SPLAT_STEPS) + ")")
    parser.add_argument("--retrain", action="store_true",
                        help="train the splats again even if ones trained from the current "
                             "dense cloud are already in the space")
    parser.add_argument("--spirula", choices=["on", "off"], default="off",
                        help="also train with Spirula Studio and let Claude judge it; off by "
                             "default, as it lost to the quick splat on the walkthrough and adds "
                             "about 45 minutes on an 8 GB Mac")
    parser.add_argument("--long-splat", choices=["on", "off"], default="off",
                        help="also train a long splat (30,000 steps at half resolution) for Claude "
                             "to choose from; off by default, as it scored worse on both test videos")
    args = parser.parse_args()

    source = Path(args.source).expanduser()
    if not source.exists():
        sys.exit(f"Source not found: {source}")
    agent = Agent(source, args.name, args.fps, not args.no_splat,
                  allow_retry=not args.no_retry, use_claude=not args.no_claude,
                  long_splat=args.long_splat == "on",
                  trained_elsewhere=args.trained_elsewhere, retrain=args.retrain,
                  splat_steps=args.splat_steps,
                  spirula=args.spirula == "on")
    if args.stage:
        stages = [args.stage]
    else:
        stages = STAGES if not args.no_splat else STAGES[:-1]
    return agent.go(stages, accept_warnings=args.accept_warnings)


if __name__ == "__main__":
    raise SystemExit(main())
