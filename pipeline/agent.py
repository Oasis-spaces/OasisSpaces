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
import json
import os
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
from reconstruct import (  # noqa: E402
    camera_path_jump, registered_images, solved_models,
)

STAGES = ["reconstruct", "densify", "shapes", "splat"]
# Claude's judgements are filed under what was judged; this is the stage each
# belongs to, so re-running a later stage keeps the earlier stages' judgements.
JUDGEMENT_STAGE = {"objects": "densify", "structure": "shapes", "blender": "shapes",
                   "start view": "splat", "fill": "splat", "choose": "splat",
                   "splat training": "splat"}

# Tools that live in different places per machine: set BLENDER / OPENSPLAT to
# override (the Colab notebook installs both under /opt and /content).
BLENDER = (os.environ.get("BLENDER") or shutil.which("blender")
           or "/Applications/Blender.app/Contents/MacOS/Blender")
OPENSPLAT = os.environ.get("OPENSPLAT") or str(ROOT / "tools/opensplat")

# A capture is in good shape when most frames land in one model; below the
# partial mark, later stages would only rebuild a fragment of the room.
MIN_REGISTERED_FRACTION = 0.60
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
SPLAT_RUNS = {"quick": (10000, 4), "long": (30000, 2)}   # steps, image downscale
# Stage 4 in steps that can each run on their own (a Colab session can die at
# any time): every step keeps its results in the space, and the next one
# picks them up. The long training also saves every LONG_CHECKPOINT_STEPS and
# resumes from the newest save.
SPLAT_STEPS = ["train-quick", "train-long", "choose-training", "fill", "choose-best"]
LONG_CHECKPOINT_STEPS = 10000
# OpenSplat keeps every training image on the GPU as 32-bit floats; past this
# it reads them from memory each step instead, so an 8 GB Mac is not swamped.
MAX_GPU_IMAGE_CACHE_GB = 1.5

# Before stage 2, Claude names what the room holds (see name_objects).
OBJECT_NAMING_FRAMES = 10
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
STRUCTURE_FILES = ["shapes.json", "room.blend", "room-render.png", "room-render-plan.png",
                   "plan-reviewed.png"]
RECHECK_NOTE = """

This is a second look. The room built from the first review was rendered and checked, and the check found these
structural problems: {problems}. The last {count} image(s) are that render, from an angle and from straight above
(grey shapes are walls, floor and furniture boxes). Use the same actions to fix them where the images show what is
wrong: drop a box that is not a free-standing object of its own (part of the bed, a panel, a sliver of a cupboard) or
that duplicates another, drop a wall that stands free or duplicates a wall. Leave alone what the check did not flag."""

# After detection, Claude checks the boxes against the list (see review_labels).
LABEL_REVIEW_PROMPT = """You check object detection for a 3D reconstruction of a room filmed on a phone.
An open-vocabulary detector (GroundingDINO) searched {keyframes} keyframes for exactly the names in this list; each name has a
role that tells the pipeline what to do with the points it labels:
{listing}
The first image shows {shown} of the keyframes with every detection box drawn and named (name and confidence). The other
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


class Agent:
    def __init__(self, source: Path, name: str, fps: float, do_splat: bool,
                 allow_retry: bool = True, use_claude: bool = True,
                 long_splat: bool | None = None, trained_elsewhere: bool = False,
                 retrain: bool = False, splat_steps: list[str] | None = None):
        self.source = source
        self.name = name
        self.fps = fps
        self.do_splat = do_splat
        self.cuda = shutil.which("nvidia-smi") is not None
        # The long run measured worse than the quick one on both test videos
        # (Sep 2026), so it only runs when asked for.
        self.long_splat = bool(long_splat)
        self.trained_elsewhere = trained_elsewhere
        self.retrain = retrain
        self.splat_steps = splat_steps or list(SPLAT_STEPS)
        self.current_step: str | None = None   # the stage-4 step recording decisions
        self.allow_retry = allow_retry
        self.space = ROOT / "spaces" / name
        self.log_path = self.space / "agent.log"
        self.stages: list[dict] = []
        self.advice: list[str] = []
        self.judgements: list[dict] = []
        self.gates: dict[str, dict] = {}
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
        """Frames whose cameras look most directly at the room's centre, spread
        across the capture. A video can start and end somewhere else (a pan
        from the corridor), so the first and last frames may not show the room."""
        import numpy as np

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
        result = subprocess.run(args, text=True, capture_output=True)
        seconds = round(time.time() - started, 1)
        self.space.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "a") as log:
            log.write(f"\n=== {stage} ({seconds}s): {' '.join(args)}\n")
            log.write(result.stdout + result.stderr)
        if result.returncode != 0:
            print(f"    failed after {seconds}s; see {self.log_path}")
            print("   ", (result.stderr or result.stdout).strip().splitlines()[-1:])
        else:
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
        verdict = self.advisor.ask_json(prompt, self.sample_frames(3))
        if verdict:
            problems = ", ".join(verdict.get("capture_problems") or [])
            self.judged("reconstruct", verdict,
                        f"{verdict.get('action')} - {verdict.get('why', '')}"
                        + (f" [{problems}]" if problems else ""))
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
        frames = self.sample_frames(OBJECT_NAMING_FRAMES)
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
                    "keyframes_found_in": found_in.get(o["name"], 0),
                    "points_labelled": (dense.get("labels") or {}).get(o["name"], 0)}
                   for o in vocabulary.objects]
        verdict = self.advisor.ask_json(LABEL_REVIEW_PROMPT.format(
            keyframes=len(dense.get("detections") or {}), shown=len(frames),
            listing=json.dumps(listing), limit=MAX_OBJECTS - 4),
            [sheet] + self.sample_frames(4), max_tokens=2000)
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
        a box, or relabel a box, but never move or resize anything, so the room
        keeps the measured geometry. With `feedback` (the structural problems
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
            "boxes that will be built, grey ones were rejected by our checks. "
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
            "Only act where the images make you confident.\n"
            'Fields: {"drop_walls": [{"id": string, "why": string}], '
            '"furniture_fronts": [{"id": string, "why": string}], '
            '"drop_boxes": [{"id": string, "why": string}], '
            '"relabel_boxes": [{"id": string, "label": string, "why": string}], '
            '"notes": one sentence}')
        images = [plan] + ([sheet] if picked else []) + self.room_frames(3)
        if feedback:
            renders = [r for r in (self.space / "room-render.png", self.space / "room-render-plan.png")
                       if r.exists()]
            prompt += RECHECK_NOTE.format(count=len(renders), problems=json.dumps(feedback))
            images += renders
        verdict = self.advisor.ask_json(prompt, images, max_tokens=2048)
        if not verdict:
            print(f"    claude (structure): no usable answer"
                  + (f" ({self.advisor.reason})" if self.advisor.reason else ""))
            return
        applied = self.apply_structure_review(shapes, verdict)
        shapes_path.write_text(json.dumps(shapes, indent=1) + "\n")
        draw_plan(self.space, self.space / "plan-reviewed.png")
        self.judged("structure", {**verdict, "applied": applied},
                    "; ".join(applied) or "no changes")

    def apply_structure_review(self, shapes: dict, verdict: dict) -> list[str]:
        """Apply Claude's decisions within fixed limits; report what happened."""
        applied = []
        walls = {i: p for i, p in enumerate(shapes["planes"])
                 if p["kind"] == "wall" and p.get("build", True)}
        fronted = set()
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
            boxes[i]["build"] = False
            boxes[i]["reason"] = f"Claude: {item.get('why', '')}"
            applied.append(f"dropped {ident}")
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

    def render_verdict(self) -> None:
        """Claude looks at the built room, from an angle and from above, next to
        a photo of the real one."""
        render = self.space / "room-render.png"
        plan = self.space / "room-render-plan.png"
        if not self.advisor.available or not render.exists():
            return
        prompt = (
            "Three images. The first is a perspective render of a parametric room "
            "our pipeline built from a phone video: grey shapes are walls, floor "
            "and furniture boxes. The second is the same room seen from straight "
            "above, like a floor plan. The third is a frame from the video.\n"
            "How the render is drawn, so do not report these as problems: the walls "
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
            "video, furniture floating, sunk into the floor or cutting through a wall. "
            "minor: everything else, such as furniture proportions or placement "
            "details. A room with any structural problem is not plausible.\n"
            'Fields: {"plausible": boolean, "problems": array of {"what": short '
            'phrase, "severity": "structural" or "minor"} (things that are wrong), '
            '"missing": array of short phrases (not captured), "advice": array of '
            'short instructions for the next capture}')
        images = [render] + ([plan] if plan.exists() else []) + self.room_frames(1)
        verdict = self.advisor.ask_json(prompt, images)
        if not verdict:
            print(f"    claude (blender): no usable answer"
                  + (f" ({self.advisor.reason})" if self.advisor.reason else ""))
            return None
        self.judged("blender", verdict,
                    ("plausible room" if verdict.get("plausible") else "not a plausible room")
                    + (f" - wrong: {', '.join(problem_text(p) for p in verdict['problems'][:3])}"
                       if verdict.get("problems") else "")
                    + (f"; missing: {', '.join(verdict.get('missing', [])[:3])}"
                       if verdict.get("missing") else ""))
        return verdict

    def recheck_structure(self, first: dict) -> None:
        """The render check found the built room structurally wrong: run the
        structure review again with those problems, rebuild, check again, and
        keep whichever room has fewer structural problems (the first one on a
        tie). The first room is kept in structure-first/."""
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
        self.build_room()
        second = self.safe(self.render_verdict)
        if second is not None and len(structural_problems(second)) < len(problems):
            self.decide("shapes", "accept",
                        f"the second review left {len(structural_problems(second))} structural "
                        f"problem(s), down from {len(problems)}")
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
        started = time.time()
        ok, _ = self.run(
            [sys.executable, str(ROOT / "pipeline/reconstruct.py"), str(self.source),
             "--name", self.name, "--fps", str(self.fps)],
            "reconstruct", "SIFT features, global mapper")
        if not ok:
            self.decide("reconstruct", "stop", "COLMAP could not build any model")
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
        self.safe(self.name_objects)
        ok, _ = self.run([sys.executable, str(ROOT / "pipeline/densify.py"),
                          str(self.space)], "densify", "MoGe-2 + object detection and outlines")
        if not ok:
            self.decide("densify", "stop", "densify failed")
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
                self.decide("densify", "stop", "densify failed with the revised object list")
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
        return True

    def step_shapes(self) -> bool:
        ok, _ = self.run([sys.executable, str(ROOT / "pipeline/shapes.py"),
                          str(self.space)], "shapes", "planes and labelled boxes")
        if not ok:
            self.decide("shapes", "stop", "shape detection failed")
            return False
        self.run([sys.executable, str(ROOT / "tools/classify_shapes.py"),
                  str(self.space), "--no-finish"], "classify", "label and sanity-check boxes")
        self.safe(self.structure_review)
        # Finish only what survived the review, so a box Claude dropped cannot
        # have pushed a wall out first.
        self.run([sys.executable, str(ROOT / "tools/classify_shapes.py"),
                  str(self.space), "--finish-only"], "finish",
                 "stand furniture on the floor, keep it inside the walls, close the room")
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
            self.safe(self.recheck_structure, verdict)
        return True

    def step_splat(self) -> bool:
        """Stage 4, one step after another (SPLAT_STEPS): only the steps in
        self.splat_steps run, each picking up what the earlier ones left."""
        steps = {"train-quick": self.train_quick, "train-long": self.train_long,
                 "choose-training": self.pick_training, "fill": self.fill_step,
                 "choose-best": self.best_step}
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
        return {"label": f"{run_name} ({steps} steps, 1/{downscale} resolution)",
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

    def pick_training(self) -> bool:
        """Claude chooses between the trained splats; the choice becomes
        splat.ply, exported for the viewer with Claude's opening view."""
        trained = [t for t in (self.trained_splat("quick"), self.trained_splat("long")) if t]
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
            keep = numbers_ok and (bool(claude.get("keep")) if claude else not self.advisor.available)
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
                    + ("" if verdicts else " (not reviewed by Claude)"),
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
                return "pass", f"{m['frames']}/{m['total']} frames placed in one model"
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
            step = {"reconstruct": self.step_reconstruct, "densify": self.step_densify,
                    "shapes": self.step_shapes, "splat": self.step_splat}[stage]
            status, why = self.gate(stage) if step() else ("stop", f"{stage} did not complete")
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
                  splat_steps=args.splat_steps)
    if args.stage:
        stages = [args.stage]
    else:
        stages = STAGES if not args.no_splat else STAGES[:-1]
    return agent.go(stages, accept_warnings=args.accept_warnings)


if __name__ == "__main__":
    raise SystemExit(main())
