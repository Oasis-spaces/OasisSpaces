"""The built room seen from the video's own cameras, each beside its frame.

A render from an arbitrary angle next to a frame from another cannot say
whether a piece of furniture stands where the video shows it. Rendered from
the frame's own camera position and lens, the built room should line up with
the frame: what is off shows as an offset, not a hunch. COLMAP gives every
frame's pose; shapes.json records the rotation (`world`) that took the solve
into the room's frame, with no translation or scale, so a camera goes into the
Blender scene by that rotation alone.

    render_views(space, ["frame_00019.jpg", ...], out_dir, blender=...)
    pairs_sheet(pairs, out_png)
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))

RENDER_WIDTH = 540         # pixels across each render (the frames are 2160 wide)
TILE_WIDTH = 360           # each picture on the sheet


def blender_camera_matrix(world: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """A COLMAP pose (x_cam = R x + t) as a Blender camera-to-world matrix in
    the room's frame. COLMAP's camera looks along +Z with +Y down the image;
    Blender's looks along -Z with +Y up, so the Y and Z axes flip."""
    world = np.asarray(world, dtype=float)
    R, t = np.asarray(R, dtype=float), np.asarray(t, dtype=float).reshape(3)
    centre = world @ (-R.T @ t)
    axes = world @ R.T @ np.diag([1.0, -1.0, -1.0])
    matrix = np.eye(4)
    matrix[:3, :3], matrix[:3, 3] = axes, centre
    return matrix


def camera_views(space: Path, names: list[str], render_width: int = RENDER_WIDTH) -> list[dict]:
    """The views for these frames (COLMAP image names), for blender_views.py.
    Frames the solve did not place are left out."""
    from densify import read_cameras_bin, read_images_bin
    from pointcloud import space_model_dir

    space = Path(space)
    model = space_model_dir(space)
    if model is None:
        return []
    world = np.array(json.loads((space / "shapes.json").read_text())["world"], dtype=float)
    cameras = read_cameras_bin(model / "cameras.bin")
    images = {v["name"]: v for v in read_images_bin(model / "images.bin").values()}
    views = []
    for name in names:
        info = images.get(name)
        if info is None:
            continue
        camera = cameras[info["camera_id"]]
        fx, fy, cx, cy = camera["params"][:4]         # OPENCV and PINHOLE: fx, fy, cx, cy, ...
        views.append({"name": Path(name).stem,
                      "matrix": blender_camera_matrix(world, info["R"], info["t"]).tolist(),
                      "fx": fx, "fy": fy, "cx": cx, "cy": cy,
                      "width": camera["width"], "height": camera["height"],
                      "render_width": render_width})
    return views


def render_views(space: Path, names: list[str], out_dir: Path,
                 blender: str | None = None) -> list[tuple[str, Path, Path]]:
    """Render the built room (room.blend) from these frames' cameras. Returns
    (name, frame, render) for each frame rendered."""
    space, out_dir = Path(space), Path(out_dir)
    blender = blender or os.environ.get("BLENDER") or shutil.which("blender")
    blend = space / "room.blend"
    views = camera_views(space, names)
    if not views or not blend.exists() or not blender:
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    views_path = out_dir / "views.json"
    views_path.write_text(json.dumps(views, indent=1))
    result = subprocess.run([blender, "--background", "--python", str(ROOT / "tools/blender_views.py"),
                             "--", str(blend), str(views_path), str(out_dir)],
                            capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"blender_views failed: {result.stderr[-800:] or result.stdout[-800:]}")
    pairs = []
    for view in views:
        frame = space / "workspace" / "images" / f"{view['name']}.jpg"
        render = out_dir / f"{view['name']}.png"
        if frame.exists() and render.exists():
            pairs.append((view["name"], frame, render))
    return pairs


def pairs_sheet(pairs: list[tuple[str, Path, Path]], out: Path, tile_width: int = TILE_WIDTH) -> Path:
    """One sheet: each row a frame (left) and the room rendered from that
    frame's camera (right), the row named after the frame."""
    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.load_default(size=16)
    tiles = []
    for name, frame, render in pairs:
        row = []
        for path in (frame, render):
            image = Image.open(path).convert("RGB")
            height = max(1, round(image.height * tile_width / image.width))
            row.append(image.resize((tile_width, height)))
        tiles.append((name, row))
    gap, top = 10, 26
    height = sum(max(im.height for im in row) + top + gap for _, row in tiles) + gap
    sheet = Image.new("RGB", (2 * tile_width + 3 * gap, height), "white")
    draw = ImageDraw.Draw(sheet)
    y = gap
    for name, row in tiles:
        draw.text((gap, y), f"{name}: the video frame (left), the built room from its camera (right)",
                  fill=(30, 30, 30), font=font)
        y += top
        for n, image in enumerate(row):
            sheet.paste(image, (gap + n * (tile_width + gap), y))
        y += max(im.height for im in row) + gap
    out = Path(out)
    sheet.save(out)
    return out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("space", type=Path)
    parser.add_argument("frames", nargs="+", help="frame names, e.g. frame_00019.jpg")
    parser.add_argument("--out", type=Path, help="sheet to write (default <space>/room-views.png)")
    args = parser.parse_args()
    found = render_views(args.space, args.frames, args.space / "room-views")
    if not found:
        raise SystemExit("nothing rendered: is the room built (room.blend), and are these frames in the solve?")
    print("wrote", pairs_sheet(found, args.out or args.space / "room-views.png"))
