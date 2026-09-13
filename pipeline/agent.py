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
from densify import read_cameras_bin, read_images_bin  # noqa: E402
from reconstruct import (  # noqa: E402
    camera_path_jump, registered_images, solved_models,
)

STAGES = ["reconstruct", "densify", "shapes", "splat"]

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
# Walls this rough (as a share of the room's diagonal) mean weak geometry.
NOISY_WALL_RMS_PCT = 1.5
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


class Agent:
    def __init__(self, source: Path, name: str, fps: float, do_splat: bool,
                 allow_retry: bool = True, use_claude: bool = True):
        self.source = source
        self.name = name
        self.fps = fps
        self.do_splat = do_splat
        self.allow_retry = allow_retry
        self.space = ROOT / "spaces" / name
        self.log_path = self.space / "agent.log"
        self.stages: list[dict] = []
        self.advice: list[str] = []
        self.judgements: list[dict] = []
        self.gates: dict[str, dict] = {}
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
            model = Path(self.densify_metrics()["model_dir"])
            room = shapes["room"]
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
        self.judgements.append({"stage": stage, **verdict})
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

    def decide(self, stage: str, action: str, why: str, metrics: dict | None = None,
               seconds: float | None = None) -> None:
        entry = {"stage": stage, "action": action, "why": why}
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

    def label_verdict(self, dense: dict) -> None:
        """Claude checks what the detector named against what is in the frame."""
        if not self.advisor.available:
            return
        found = ", ".join(dense.get("labels", {})) or "nothing"
        prompt = (
            "Attached is one frame from an indoor capture. An open-vocabulary detector "
            f"labelled the reconstructed 3D points as: {found}.\n"
            "List the furniture and fixtures you can actually see in this frame, then "
            "say which of those labels look wrong, and which visible objects the "
            "detector missed.\n"
            'Fields: {"objects": array, "wrong": array, "missing": array}')
        verdict = self.advisor.ask_json(prompt, self.sample_frames(1))
        if not verdict:
            return
        self.judged("densify", verdict,
                    f"sees {', '.join(verdict.get('objects', [])[:6]) or 'nothing'}"
                    + (f"; missed {', '.join(verdict.get('missing', [])[:4])}"
                       if verdict.get("missing") else ""))

    def structure_review(self) -> None:
        """Claude decides what each measured candidate is. It can drop a wall or
        a box, or relabel a box, but never move or resize anything, so the room
        keeps the measured geometry."""
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
                               "points": p["points"]})
        for i, b in enumerate(shapes["boxes"]):
            candidates.append({"id": f"B{i}", "label": b.get("label"),
                               "detected_as": b.get("detected"),
                               "size": [size(b["max"][k] - b["min"][k]) for k in range(3)],
                               "points": b["points"], "built": b.get("build", True),
                               "check": b.get("reason")})
        prompt = (
            "You are reviewing the room structure our 3D pipeline measured from a phone "
            "video. The first image is a floor plan seen from above: darker areas are "
            "dense reconstructed points, the blue outline is the floor, red numbered "
            "lines (W) are wall candidates, green numbered rectangles (B) are furniture "
            "boxes that will be built, grey ones were rejected by our checks. The other "
            "images are frames from the video.\n"
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
            "Rejected boxes stay rejected. Only act where the images make you confident.\n"
            'Fields: {"drop_walls": [{"id": string, "why": string}], '
            '"drop_boxes": [{"id": string, "why": string}], '
            '"relabel_boxes": [{"id": string, "label": string, "why": string}], '
            '"notes": one sentence}')
        verdict = self.advisor.ask_json(prompt, [plan, *self.room_frames(3)],
                                        max_tokens=2048)
        if not verdict:
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
        biggest = max(walls, key=lambda i: walls[i]["points"]) if walls else None
        can_drop = len(walls) // 2  # never remove more than half the walls
        for item in verdict.get("drop_walls") or []:
            ident = str(item.get("id", ""))
            i = int(ident[1:]) if ident[:1] == "W" and ident[1:].isdigit() else None
            if i not in walls:
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
            'Fields: {"plausible": boolean, "problems": array of short phrases '
            '(things that are wrong), "missing": array of short phrases (not '
            'captured), "advice": array of short instructions for the next capture}')
        images = [render] + ([plan] if plan.exists() else []) + self.room_frames(1)
        verdict = self.advisor.ask_json(prompt, images)
        if not verdict:
            return
        self.judged("blender", verdict,
                    ("plausible room" if verdict.get("plausible") else "not a plausible room")
                    + (f" - wrong: {', '.join(verdict.get('problems', [])[:3])}"
                       if verdict.get("problems") else "")
                    + (f"; missing: {', '.join(verdict.get('missing', [])[:3])}"
                       if verdict.get("missing") else ""))

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
        path = self.space / "densify.json"
        return json.loads(path.read_text()) if path.exists() else {}

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
        self.decide("densify", "accept",
                    f"1 m = {metrics.get('colmap_units_per_metre', float('nan')):.3f} units, "
                    f"keyframes agree within {metrics.get('scale_spread', 0):.0%}, "
                    f"{metrics.get('points', 0):,} points", metrics)
        self.safe(self.label_verdict, metrics)
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
        self.run(["/Applications/Blender.app/Contents/MacOS/Blender", "--background",
                  "--python", str(ROOT / "tools/blender_room.py"), "--",
                  str(self.space / "shapes.json"), str(self.space / "room.blend"),
                  str(self.space / "room-render.png")], "blender", "build the room")
        self.safe(self.render_verdict)
        return True

    def step_splat(self) -> bool:
        ok, _ = self.run([sys.executable, str(ROOT / "pipeline/splat_seed.py"),
                          str(self.space)], "splat seed", "dense cloud as starting points")
        if not ok:
            self.decide("splat", "skip", "could not build the splat project")
            return False
        ok, _ = self.run([str(ROOT / "tools/opensplat"), str(self.space / "splat-project"),
                          "-n", "10000", "-d", "4",
                          "-o", str(self.space / "splat.ply")], "splat", "10000 steps")
        self.decide("splat", "accept" if ok else "skip",
                    "trained" if ok else "OpenSplat failed; see the log")
        return ok

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
            if verdict is not None and verdict.get("plausible") is False:
                return "warn", ("Claude judged the built room implausible: "
                                + ", ".join(verdict.get("problems", [])[:3]))
            if s["walls"] < 2:
                return "warn", f"only {s['walls']} wall(s) found"
            if not s["objects"]:
                return "warn", "no furniture was recognised"
            return "pass", (f"{s['walls']} walls, {s['built_boxes']} boxes to build "
                            f"({', '.join(s['objects'])})")
        if stage == "splat":
            count = ply_vertex_count(self.space / "splat.ply")
            if not count:
                return "stop", "no splat was written"
            if count < MIN_SPLAT_GAUSSIANS:
                return "warn", f"only {count:,} gaussians"
            return "pass", f"{count:,} gaussians"
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
        """Keep the record of stages before `first`; later ones are now stale."""
        path = self.space / "agent-report.json"
        if not path.exists() or first == STAGES[0]:
            return
        earlier = set(STAGES[:STAGES.index(first)])
        previous = json.loads(path.read_text())
        self.stages = [e for e in previous.get("decisions", []) if e["stage"] in earlier]
        self.judgements = [j for j in previous.get("judgements", [])
                           if j["stage"] in earlier]
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
        run_is_over = stopped_at is not None or stages[-1] in (last, "splat")

        self.advice = []
        if run_is_over:
            self.capture_advice(recon, dense, shapes, row)
            for j in self.judgements:
                if j["stage"] == "densify" and j.get("missing"):
                    self.advice.append(
                        "The detector missed " + ", ".join(map(str, j["missing"][:6]))
                        + ". Adding those words to VOCABULARY in pipeline/semantics.py "
                        "would let them become furniture instead of leftover points.")
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
    args = parser.parse_args()

    source = Path(args.source).expanduser()
    if not source.exists():
        sys.exit(f"Source not found: {source}")
    agent = Agent(source, args.name, args.fps, not args.no_splat,
                  allow_retry=not args.no_retry, use_claude=not args.no_claude)
    if args.stage:
        stages = [args.stage]
    else:
        stages = STAGES if not args.no_splat else STAGES[:-1]
    return agent.go(stages, accept_warnings=args.accept_warnings)


if __name__ == "__main__":
    raise SystemExit(main())
