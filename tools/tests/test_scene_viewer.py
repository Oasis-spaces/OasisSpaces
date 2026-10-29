"""The scene viewer in a real browser: added objects, colours, layouts.

A small scene is made here (a textured floor and wall, a "bed" with its splat
and clean model, "everything else"), scene-viewer/ is served beside it, and
tools/tests/scene_viewer_check.mjs drives it in headless Chromium with
Playwright. Skipped when node or its playwright package is not installed.

    python3 tools/tests/test_scene_viewer.py      (or pytest tools/tests)
"""

import io
import json
import shutil
import subprocess
import sys
import tempfile
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import mixed_scene as ms  # noqa: E402

CHECK = Path(__file__).with_name("scene_viewer_check.mjs")


def jpeg(rgb) -> bytes:
    out = io.BytesIO()
    pixels = np.zeros((16, 16, 3), np.uint8) + np.array(rgb, np.uint8)
    pixels[::4] //= 2                                   # stripes, so the texture is not flat
    Image.fromarray(pixels).save(out, "JPEG")
    return out.getvalue()


def flat(name, corners, normal, rgb):
    positions = np.array(corners, np.float32)
    return {"name": name, "jpeg": jpeg(rgb), "roughness": 0.9,
            "quad": (positions, np.tile(np.array(normal, np.float32), (4, 1)),
                     np.array([[0, 0], [1, 0], [0, 1], [1, 1]], np.float32), np.array([0, 2, 1, 2, 3, 1], np.uint16))}


def splat(path: Path, lo, hi, rgb, count=4000, seed=0):
    rng = np.random.default_rng(seed)
    out = np.zeros(count, dtype=[("position", "<f4", 3), ("scale", "<f4", 3), ("rgba", "u1", 4), ("rotation", "u1", 4)])
    out["position"] = rng.uniform(lo, hi, (count, 3))
    out["scale"] = 0.02
    shade = rng.uniform(0.8, 1.0, (count, 1))
    out["rgba"][:, :3] = np.clip(np.array(rgb) * shade, 0, 255)
    out["rgba"][:, 3] = 230
    out["rotation"] = [255, 128, 128, 128]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(out.tobytes())


def cuboid_glb(path: Path, lo, hi, rgb):
    ms.write_glb(path, ms.model_meshes([(np.array(lo, np.float32), np.array(hi, np.float32), "wood", rgb)]))


def make_scene(folder: Path) -> None:
    scene = folder / "scene"
    (scene / "models").mkdir(parents=True)
    ms.write_glb(scene / "shell.glb", [
        flat("floor", [[-2, 0, -1.5], [2, 0, -1.5], [-2, 0, 1.5], [2, 0, 1.5]], [0, 1, 0], (150, 120, 90)),
        flat("wall-0", [[-2, 0, -1.5], [2, 0, -1.5], [-2, 2.5, -1.5], [2, 2.5, -1.5]], [0, 0, 1], (220, 215, 200)),
    ])
    # The bed: 1.4 x 0.5 x 2.0 m, its anchor on the floor at its middle.
    splat(scene / "pieces" / "B0.splat", [-0.7, 0, -1.0], [0.7, 0.5, 1.0], (180, 60, 50))
    cuboid_glb(scene / "models" / "B0.glb", [-0.7, 0, -1.0], [0.7, 0.5, 1.0], (180, 60, 50))
    # A pillow on it, carried by it.
    splat(scene / "pieces" / "B1.splat", [-0.3, 0, -0.15], [0.3, 0.15, 0.15], (240, 240, 235), count=500, seed=1)
    splat(scene / "pieces" / "rest.splat", [-1.8, 0, -1.4], [-1.5, 1.0, -1.2], (90, 90, 90), count=500, seed=2)
    manifest = {
        "space": "test-room", "units": "metres", "up": "y",
        "room": {"width": 4.0, "depth": 3.0, "height": 2.5}, "shell": "shell.glb",
        "pieces": [
            {"id": "B0", "file": "pieces/B0.splat", "label": "bed", "movable": True, "model": "models/B0.glb",
             "show": "scan", "anchor": [0.5, 0, 0.2], "box": {"min": [-0.7, 0, -1.0], "max": [0.7, 0.5, 1.0]}},
            {"id": "B1", "file": "pieces/B1.splat", "label": "pillow", "movable": True, "on": "B0",
             "anchor": [0.5, 0.5, -0.5], "box": {"min": [-0.3, 0, -0.15], "max": [0.3, 0.15, 0.15]}},
            {"id": "rest", "file": "pieces/rest.splat", "label": "everything else", "movable": False,
             "anchor": [0, 0, 0], "box": {"min": [-1.8, 0, -1.4], "max": [-1.5, 1.0, -1.2]}},
        ],
    }
    (scene / "scene.json").write_text(json.dumps(manifest))
    # Objects to add: a 0.5 x 0.9 x 0.5 m chair-sized block in metres, and one made in centimetres.
    cuboid_glb(folder / "chair.glb", [0, 0, 0], [0.5, 0.9, 0.5], (40, 140, 60))
    cuboid_glb(folder / "lamp-cm.glb", [-15, 0, -15], [15, 160, 15], (230, 200, 90))


def playwright_ready() -> bool:
    node = shutil.which("node")
    return bool(node) and subprocess.run([node, "-e", "require('playwright')"], capture_output=True).returncode == 0


def test_scene_viewer_in_a_browser():
    if not playwright_ready():
        print("skipped: node with the playwright package is not installed")
        return
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        make_scene(folder)
        (folder / "scene-viewer").symlink_to(ROOT / "scene-viewer")
        handler = partial(type("Quiet", (SimpleHTTPRequestHandler,), {"log_message": lambda *a: None}),
                          directory=str(folder))
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            base = f"http://127.0.0.1:{server.server_address[1]}"
            result = subprocess.run(["node", str(CHECK), base, str(folder)], capture_output=True, text=True,
                                    timeout=180)
        finally:
            server.shutdown()
        print(result.stdout, end="")
        assert result.returncode == 0, result.stdout + result.stderr


if __name__ == "__main__":
    test_scene_viewer_in_a_browser()
    print("ok")
