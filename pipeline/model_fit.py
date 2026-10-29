#!/usr/bin/env python3
"""A generated furniture model, turned and sized to its piece by what the
video saw, and kept only when it matches the frames as well as the box does.

tools/object_mesh.py turns each piece's frames into a textured mesh with
TRELLIS. What comes back is the object alone, upright (y up), at an arbitrary
scale and facing an arbitrary way, and sometimes not the object at all: on the
walkthrough the single-view meshes came out as flat slabs with clutter fused
in. So a generated model is never trusted on sight. It is placed the way
placement.py places a box, by silhouettes through the real cameras:

  1. its height is the piece's measured height (floor to top is what stage 3
     measures best), so its scale follows;
  2. it is turned about the vertical in YAW_STEP steps, and the turn whose
     silhouettes in the job's frames best match the piece's masks (the masks
     object_models.py cut from the scene's own piece) is kept;
  3. around that turn, finer turns, a scale within SCALE_TRY and a shift of
     up to SHIFT_TRY of the box's width and depth are tried the same way.

Its agreement (mean IoU of the silhouettes over the frames) is then set beside
the agreement of the plain box model the scene already has, measured the same
way. The generated model is used only when it agrees at least MIN_IOU and no
worse than the box by more than BOX_MARGIN: measured, not judged, like the
room score.

What is kept goes into the scene: scene/generated/<id>.glb, and in scene.json
the piece's "generated": the file, the model's own bottom centre ("origin"),
"scale", "turnDegrees" and "offset" from the piece's anchor (in the viewer:
offset + turn(scale * (vertex - origin))), and the "fit". A piece Claude asked
to show as its clean model shows the generated one when there is one.
Everything tried, kept or not, is in object-models/fits.json.

Usage:
    python3 pipeline/model_fit.py spaces/<name>        (after object_mesh.py)
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

WORK_WIDTH = 320          # silhouettes are compared at this width
SAMPLES = 30000           # points spread over the model's surface to draw its silhouette
YAW_STEP = 15             # degrees, the first search
YAW_FINE = (-10, -5, 0, 5, 10)
SCALE_TRY = (0.85, 1.0, 1.15)
SHIFT_TRY = 0.1           # of the box's width and depth
MIN_IOU = 0.5             # a model agreeing less than this with the frames is not the object
BOX_MARGIN = 0.03         # ... and it may agree this much less than the plain box, no more
TO_VIEWER = np.array([[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]])   # tools/mixed_scene.py's


# ------------------------------------------------------------------ geometry
def turn(degrees: float) -> np.ndarray:
    """three.js's rotation.y: x' = x cos + z sin, z' = -x sin + z cos."""
    a = np.radians(degrees)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def surface_points(vertices: np.ndarray, faces: np.ndarray, count: int = SAMPLES, seed: int = 0) -> np.ndarray:
    """Points spread over the triangles by area: dense enough that their
    projection, closed up, is the silhouette."""
    rng = np.random.default_rng(seed)
    a, b, c = (vertices[faces[:, k]] for k in range(3))
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    if area.sum() <= 0:
        return vertices.copy()
    pick = rng.choice(len(faces), size=count, p=area / area.sum())
    u, v = rng.random(count), rng.random(count)
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    return a[pick] + u[:, None] * (b[pick] - a[pick]) + v[:, None] * (c[pick] - a[pick])


def box_points(lo, hi, count: int = SAMPLES, seed: int = 0) -> np.ndarray:
    """Points over the six faces of a box, by area."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    faces = np.array([f for a, b, c, d in quads for f in ((a, b, c), (a, c, d))])
    return surface_points(corners, faces, count, seed)


def viewer_to_solve(points: np.ndarray, solve_frame: dict) -> np.ndarray:
    """The inverse of scene.json's frame: viewer metres -> camera-solve coordinates
    (viewer = ((solve @ world.T - origin) / unitsPerMetre) @ toViewer.T)."""
    to_viewer = np.asarray(solve_frame.get("toViewer", TO_VIEWER), float)
    scene = (np.asarray(points, float) @ to_viewer) * float(solve_frame["unitsPerMetre"]) \
        + np.asarray(solve_frame["origin"], float)
    return scene @ np.asarray(solve_frame["world"], float)


# ------------------------------------------------------------------ silhouettes
class View:
    """One of the job's frames: its camera, at WORK_WIDTH, and the piece's mask there."""

    def __init__(self, camera: dict, mask: np.ndarray):
        self.scale = WORK_WIDTH / camera["width"]
        self.width, self.height = WORK_WIDTH, max(1, round(camera["height"] * self.scale))
        self.R, self.t = np.asarray(camera["R"], float), np.asarray(camera["t"], float)
        fx, fy, cx, cy = (float(v) for v in camera["params"][:4])
        self.fx, self.fy, self.cx, self.cy = fx * self.scale, fy * self.scale, cx * self.scale, cy * self.scale
        self.mask = mask

    @classmethod
    def from_job(cls, folder: Path, view: dict, n: int) -> "View":
        from PIL import Image

        camera = view["camera"]
        scale = WORK_WIDTH / camera["width"]
        size = (WORK_WIDTH, max(1, round(camera["height"] * scale)))
        mask = np.asarray(Image.open(folder / f"view-{n}.png").convert("L").resize(size, Image.BILINEAR)) > 127
        return cls(camera, mask)

    def silhouette(self, points_solve: np.ndarray) -> np.ndarray:
        """The points' outline in this frame: projected, closed up, small holes filled."""
        from scipy.ndimage import binary_closing, binary_dilation, binary_fill_holes

        local = points_solve @ self.R.T + self.t
        ahead = local[:, 2] > 1e-6
        local = local[ahead]
        u = (self.fx * local[:, 0] / local[:, 2] + self.cx).astype(int)
        v = (self.fy * local[:, 1] / local[:, 2] + self.cy).astype(int)
        on = (u >= 0) & (v >= 0) & (u < self.width) & (v < self.height)
        mask = np.zeros((self.height, self.width), bool)
        mask[v[on], u[on]] = True
        return binary_fill_holes(binary_closing(binary_dilation(mask, iterations=1), iterations=2))

    def iou(self, points_solve: np.ndarray) -> float:
        drawn = self.silhouette(points_solve)
        union = (drawn | self.mask).sum()
        return float((drawn & self.mask).sum() / union) if union else 0.0


def agreement(points_viewer: np.ndarray, views: list[View], solve_frame: dict) -> float:
    """Mean silhouette IoU over the frames."""
    solve = viewer_to_solve(points_viewer, solve_frame)
    return float(np.mean([view.iou(solve) for view in views])) if views else 0.0


# ------------------------------------------------------------------ the fit
def place(points_model: np.ndarray, origin, scale: float, degrees: float, offset) -> np.ndarray:
    """Model coordinates -> the piece's frame (relative to its anchor), as the viewer does it."""
    return (scale * (points_model - np.asarray(origin, float))) @ turn(degrees).T + np.asarray(offset, float)


def fit(vertices: np.ndarray, faces: np.ndarray, piece: dict, views: list[View], solve_frame: dict) -> dict:
    """The generated model's best placement in its piece, and how it compares
    with the plain box. `piece` is its scene.json entry (anchor and box in
    viewer metres)."""
    anchor = np.asarray(piece["anchor"], float)
    lo, hi = np.asarray(piece["box"]["min"], float), np.asarray(piece["box"]["max"], float)
    size = hi - lo
    model_lo, model_hi = vertices.min(axis=0), vertices.max(axis=0)
    origin = np.array([(model_lo[0] + model_hi[0]) / 2, model_lo[1], (model_lo[2] + model_hi[2]) / 2])
    height = model_hi[1] - model_lo[1]
    if height <= 0:
        raise ValueError("the model is flat")
    base_scale = size[1] / height
    centre = np.array([(lo[0] + hi[0]) / 2, lo[1], (lo[2] + hi[2]) / 2])
    samples = surface_points(vertices, faces)

    def score(scale, degrees, offset):
        return agreement(place(samples, origin, scale, degrees, offset) + anchor, views, solve_frame)

    best = max(((score(base_scale, d, centre), d) for d in range(0, 360, YAW_STEP)))
    yaw = best[1]
    trials = [(score(base_scale * k, (yaw + d) % 360, centre + np.array([dx * size[0], 0, dz * size[2]])),
               base_scale * k, (yaw + d) % 360, centre + np.array([dx * size[0], 0, dz * size[2]]))
              for d in YAW_FINE for k in SCALE_TRY
              for dx in (-SHIFT_TRY, 0, SHIFT_TRY) for dz in (-SHIFT_TRY, 0, SHIFT_TRY)]
    iou, scale, degrees, offset = max(trials, key=lambda t: t[0])
    box_iou = agreement(box_points(lo, hi) + anchor, views, solve_frame)
    placed = place(np.array([model_lo, model_hi]), origin, scale, degrees, offset)
    footprint = np.abs(turn(degrees)[[0, 2]][:, [0, 2]]) @ (model_hi - model_lo)[[0, 2]] * scale   # the turned box's
    keep = iou >= MIN_IOU and iou >= box_iou - BOX_MARGIN
    why = (f"agrees {iou:.2f} with the frames against the box's {box_iou:.2f}"
           + ("" if keep else f" (needs at least {MIN_IOU:.2f} and no worse than the box by {BOX_MARGIN:.2f})"))
    return {"keep": bool(keep), "why": why, "iou": round(iou, 3), "boxIou": round(box_iou, 3),
            "views": len(views), "origin": origin.round(5).tolist(), "scale": round(float(scale), 5),
            "turnDegrees": round(float(degrees), 1), "offset": offset.round(4).tolist(),
            "footprintM": footprint.round(3).tolist(), "boxFootprintM": size[[0, 2]].round(3).tolist(),
            "heightM": round(float(placed[1, 1] - placed[0, 1]), 3)}


# ------------------------------------------------------------------ the scene
def attach(space: Path, log=print) -> dict:
    """Fit every generated model (object-models/<id>/model.glb) to its piece and
    put the ones that agree into the scene. Returns {id: fit}."""
    from mesh_render import read_glb

    scene_dir = space / "scene"
    manifest_path = scene_dir / "scene.json"
    manifest = json.loads(manifest_path.read_text())
    solve_frame = manifest.get("solveFrame")
    if not solve_frame:
        raise SystemExit(f"{manifest_path} has no solveFrame: build the scene again (tools/mixed_scene.py)")
    out = scene_dir / "generated"
    shutil.rmtree(out, ignore_errors=True)
    fits = {}
    for piece in manifest["pieces"]:
        piece.pop("generated", None)
        if piece.get("show") == "generated":
            piece["show"] = "model"
        folder = space / "object-models" / piece["id"]
        if not piece.get("movable") or not (folder / "model.glb").exists() or not (folder / "job.json").exists():
            continue
        job = json.loads((folder / "job.json").read_text())
        views = [View.from_job(folder, view, n) for n, view in enumerate(job["views"])
                 if (folder / f"view-{n}.png").exists()]
        if not views:
            continue
        try:
            mesh = read_glb(folder / "model.glb")
            result = fit(mesh.vertices, mesh.faces, piece, views, solve_frame)
        except (ValueError, KeyError) as exc:
            log(f"  {piece['id']} {piece['label']}: could not fit its model ({exc})")
            fits[piece["id"]] = {"keep": False, "why": f"could not fit: {exc}"}
            continue
        fits[piece["id"]] = result
        log(f"  {piece['id']} {piece['label']}: {'kept' if result['keep'] else 'not kept'}, {result['why']}; "
            f"turned {result['turnDegrees']:.0f} deg, {result['footprintM'][0]:.2f} x {result['footprintM'][1]:.2f} m "
            f"(box {result['boxFootprintM'][0]:.2f} x {result['boxFootprintM'][1]:.2f})")
        if not result["keep"]:
            continue
        out.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(folder / "model.glb", out / f"{piece['id']}.glb")
        piece["generated"] = {"file": f"generated/{piece['id']}.glb",
                              **{k: result[k] for k in ("origin", "scale", "turnDegrees", "offset")},
                              "fit": {k: result[k] for k in ("iou", "boxIou", "views")}}
        if piece.get("show") == "model":
            piece["show"] = "generated"
    manifest_path.write_text(json.dumps(manifest, indent=1) + "\n")
    (space / "object-models").mkdir(exist_ok=True)
    (space / "object-models" / "fits.json").write_text(json.dumps(fits, indent=1) + "\n")
    return fits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("space", type=Path)
    args = parser.parse_args()
    fits = attach(args.space.resolve())
    kept = [k for k, v in fits.items() if v.get("keep")]
    print(f"{len(kept)} of {len(fits)} generated model(s) kept: {', '.join(kept) or 'none'}")


if __name__ == "__main__":
    main()
