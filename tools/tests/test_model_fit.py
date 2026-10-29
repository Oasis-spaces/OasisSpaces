"""Fitting a generated model to its piece: turn, size and the box comparison.

A chair is made of boxes in a model frame of its own (another scale, facing
another way), placed in a room whose camera solve is turned, scaled and moved
against the viewer's frame, and drawn through four cameras as the masks
object_models.py would write. model_fit.attach must find the chair's turn and
size from those masks alone and keep it; a flat slab, what TRELLIS made of the
walkthrough's pieces, must lose to the box.

    python3 tools/tests/test_model_fit.py      (or pytest tools/tests)
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(ROOT / "tools"))
import mixed_scene as ms  # noqa: E402
import model_fit as mf  # noqa: E402

WIDTH, HEIGHT, FOCAL = 1280, 960, 1000.0
T = mf.TO_VIEWER


def rotation(axis, degrees):
    a = np.radians(degrees)
    x, y, z = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


SOLVE = {"origin": [3.0, -1.0, 0.7], "unitsPerMetre": 2.5, "world": rotation((0.3, 0.2, 1.0), 35).tolist(),
         "toViewer": T.tolist()}


def chair_parts(unit=1.0):
    """A chair in its own frame, y up, facing +z: seat, a back along -z, four legs."""
    box = lambda lo, hi: (np.array(lo, np.float32) * unit, np.array(hi, np.float32) * unit)   # noqa: E731
    parts = [box([0, 0.40, 0], [0.5, 0.48, 0.5]), box([0, 0.48, 0], [0.5, 1.0, 0.08])]
    for x in (0, 0.44):
        for z in (0, 0.44):
            parts.append(box([x, 0, z], [x + 0.06, 0.40, z + 0.06]))
    return parts


def mesh_of(parts):
    vertices, faces = [], []
    for lo, hi in parts:
        positions, _normals, indices = ms.cuboid(lo, hi)
        faces.append(indices.reshape(-1, 3) + sum(len(v) for v in vertices))
        vertices.append(positions)
    return np.vstack(vertices).astype(float), np.vstack(faces)


def camera(eye, target):
    """A camera in the solve frame looking from `eye` at `target` (viewer metres)."""
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    forward = (target - eye) / np.linalg.norm(target - eye)
    right = np.cross(forward, [0, 1, 0])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    Rv = np.array([right, down, forward])                     # viewer -> camera axes
    W, o, m = np.array(SOLVE["world"]), np.array(SOLVE["origin"]), SOLVE["unitsPerMetre"]
    # viewer = T (W solve - o) / m, so camera = Rv (viewer - eye) is, times m,
    # Rv T W solve - Rv (T o + m eye): the same pixels.
    return {"width": WIDTH, "height": HEIGHT, "params": [FOCAL, FOCAL, WIDTH / 2, HEIGHT / 2, 0, 0, 0, 0],
            "R": (Rv @ T @ W).tolist(), "t": (-Rv @ (T @ o + m * eye)).tolist()}


def mask_of(vertices_viewer, faces, cam):
    """The triangles drawn as polygons at full resolution: not the fit's own way of drawing."""
    solve = mf.viewer_to_solve(vertices_viewer, SOLVE)
    local = solve @ np.array(cam["R"]).T + np.array(cam["t"])
    u = FOCAL * local[:, 0] / local[:, 2] + WIDTH / 2
    v = FOCAL * local[:, 1] / local[:, 2] + HEIGHT / 2
    image = Image.new("L", (WIDTH, HEIGHT), 0)
    draw = ImageDraw.Draw(image)
    for tri in faces:
        draw.polygon([(u[k], v[k]) for k in tri], fill=255)
    return image


def make_space(folder: Path, model_parts, true_turn=120.0, anchor=(0.4, 0.0, -0.3)):
    """A space with one piece, B0, the chair as it really stands, and `model_parts`
    as its generated model."""
    vertices, faces = mesh_of(chair_parts())
    # The chair as it stands: 0.9 m tall, turned true_turn, its feet's centre at the anchor.
    true = mf.place(vertices, [0.25, 0, 0.25], 0.9, true_turn, [0, 0, 0]) + np.array(anchor)
    lo, hi = true.min(axis=0), true.max(axis=0)
    centre = np.array([(lo[0] + hi[0]) / 2, 0, (lo[2] + hi[2]) / 2])
    piece = {"id": "B0", "label": "chair", "movable": True, "show": "model", "anchor": centre.round(4).tolist(),
             "box": {"min": (lo - centre).round(4).tolist(), "max": (hi - centre).round(4).tolist()}}
    (folder / "scene").mkdir(parents=True)
    (folder / "scene" / "scene.json").write_text(json.dumps({"space": "fit", "solveFrame": SOLVE, "pieces": [piece]}))
    job_dir = folder / "object-models" / "B0"
    job_dir.mkdir(parents=True)
    views = []
    for n, angle in enumerate((20, 75, 140, 200)):
        a = np.radians(angle)
        cam = camera(centre + [2.4 * np.sin(a), 1.5, 2.4 * np.cos(a)], centre + [0, 0.45, 0])
        mask_of(true, faces, cam).save(job_dir / f"view-{n}.png")
        views.append({"frame": f"frame_{n}.jpg", "camera": cam})
    (job_dir / "job.json").write_text(json.dumps({"id": "B0", "label": "chair", "views": views}))
    ms.write_glb(job_dir / "model.glb", ms.model_meshes([(lo, hi, "wood", (120, 90, 60)) for lo, hi in model_parts]))
    return true


def test_the_chair_is_found_turned_and_sized():
    # The generated model: the same chair at a third of the size, already turned 90 degrees in its own frame.
    parts = [(np.array([lo[2], lo[1], -hi[0]]) * 0.37, np.array([hi[2], hi[1], -lo[0]]) * 0.37)
             for lo, hi in chair_parts()]
    with tempfile.TemporaryDirectory() as tmp:
        space = Path(tmp)
        true = make_space(space, parts)
        fits = mf.attach(space, log=lambda *_: None)
        result = fits["B0"]
        assert result["keep"], result
        assert result["iou"] > 0.8, result
        manifest = json.loads((space / "scene" / "scene.json").read_text())
        piece = manifest["pieces"][0]
        assert piece["show"] == "generated" and (space / "scene" / piece["generated"]["file"]).exists()
        # Where the model's own corners land is where the chair's do.
        model_vertices, _ = mesh_of([(np.asarray(lo, float), np.asarray(hi, float)) for lo, hi in parts])
        g = piece["generated"]
        placed = mf.place(model_vertices, g["origin"], g["scale"], g["turnDegrees"], g["offset"]) + piece["anchor"]
        assert np.abs(placed.min(axis=0) - true.min(axis=0)).max() < 0.08, (placed.min(axis=0), true.min(axis=0))
        assert np.abs(placed.max(axis=0) - true.max(axis=0)).max() < 0.08, (placed.max(axis=0), true.max(axis=0))
        assert abs(result["heightM"] - 0.9) < 0.15, result


def test_a_flat_slab_loses_to_the_box():
    slab = [(np.array([0, 0, 0], float), np.array([1.0, 0.08, 0.9]))]
    with tempfile.TemporaryDirectory() as tmp:
        space = Path(tmp)
        make_space(space, slab)
        result = mf.attach(space, log=lambda *_: None)["B0"]
        assert not result["keep"], result
        piece = json.loads((space / "scene" / "scene.json").read_text())["pieces"][0]
        assert "generated" not in piece and piece["show"] == "model"
        assert not (space / "scene" / "generated").exists()


def test_viewer_and_solve_frames_invert():
    points = np.random.default_rng(3).normal(size=(50, 3))
    solve = mf.viewer_to_solve(points, SOLVE)
    W, o, m = np.array(SOLVE["world"]), np.array(SOLVE["origin"]), SOLVE["unitsPerMetre"]
    back = ((solve @ W.T - o) / m) @ T.T                     # scene.json's own formula
    assert np.allclose(back, points)


def test_turn_matches_threejs():
    # three.js: rotation.y = 90 deg takes +x to -z.
    assert np.allclose(mf.turn(90) @ [1, 0, 0], [0, 0, -1])


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print("ok", name)
