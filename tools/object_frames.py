#!/usr/bin/env python3
"""For each furniture box in shapes.json, the video frame that shows it best.

Claude's structure review (pipeline/agent.py) otherwise sees only frames that
look at the middle of the room, so an object off to one side is judged from the
plan alone: the pan video's cupboard, filmed side-on in its first frames, was
dropped as "no built-in wardrobe is visible in any frame". This projects each
box into every solved frame, prefers frames where the object detector found
that kind of object over the box (densify.json "detections"), and tiles a crop
of each chosen frame, with the box's outline drawn and its B<i> number, into
one sheet.

Usage:
    python3 tools/object_frames.py spaces/<name> [--out object-frames.png]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
from densify import read_cameras_bin, read_images_bin  # noqa: E402
from semantics import room_vocabulary  # noqa: E402

TILE_W, TILE_H = 300, 400
MAX_TILES = 12


def box_corners(box: dict) -> np.ndarray:
    lo, hi = np.array(box["min"]), np.array(box["max"])
    return np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])
                     for z in (lo[2], hi[2])])


def best_frames(space: Path, boxes: dict[int, dict], per_box: int = 2) -> dict[int, list[dict]]:
    """{box index: [{"frame", "rect" (x0, y0, x1, y1 in pixels), "detected"}]},
    the best frames first, at least a tenth of the video apart."""
    shapes = json.loads((space / "shapes.json").read_text())
    meta = json.loads((space / "densify.json").read_text())
    model = Path(meta["model_dir"])
    if not (model / "images.bin").exists():
        model = space / "workspace" / "sparse" / model.name
    world = np.array(shapes["world"])
    cameras = read_cameras_bin(model / "cameras.bin")
    detections = meta.get("detections", {})
    vocabulary = room_vocabulary(space)
    chosen = {}
    infos = sorted(read_images_bin(model / "images.bin").values(), key=lambda v: v["name"])
    order = {info["name"]: n for n, info in enumerate(infos)}
    for i, box in boxes.items():
        corners = box_corners(box) @ world  # scene -> camera-solve frame (world is a rotation)
        scored = []
        for info in infos:
            cam = cameras[info["camera_id"]]
            fx, fy, cx, cy = cam["params"][:4]
            W, H = cam["width"], cam["height"]
            local = corners @ info["R"].T + info["t"]
            if np.any(local[:, 2] <= 0.05 * np.abs(local).max()):
                continue  # part of the box is behind the camera
            us = fx * local[:, 0] / local[:, 2] + cx
            vs = fy * local[:, 1] / local[:, 2] + cy
            x0, x1, y0, y1 = us.min(), us.max(), vs.min(), vs.max()
            cut_w = max(0.0, min(x1, W) - max(x0, 0)) / max(x1 - x0, 1e-6)
            cut_h = max(0.0, min(y1, H) - max(y0, 0)) / max(y1 - y0, 1e-6)
            inside = cut_w * cut_h
            visible = (max(0.0, min(x1, W) - max(x0, 0)) * max(0.0, min(y1, H) - max(y0, 0))
                       / (W * H))
            if inside < 0.4 and visible < 0.25:
                continue  # mostly out of the frame (a big object may never fit, though)
            # Large in view, and as much of it in the frame as possible.
            score = (0.5 + 0.5 * inside) * min(visible, 0.5) / 0.5
            detected = False
            for det in detections.get(info["name"], []):
                if not vocabulary.same_kind(det["label"], box.get("detected")):
                    continue
                dx0, dy0, dx1, dy1 = det["box"]
                overlap = (max(0.0, min(x1, dx1) - max(x0, dx0))
                           * max(0.0, min(y1, dy1) - max(y0, dy0)))
                if overlap > 0.3 * min((dx1 - dx0) * (dy1 - dy0), (x1 - x0) * (y1 - y0)):
                    score, detected = score * (2 + det["score"]), True
                    break
            scored.append({"frame": info["name"], "score": score, "detected": detected,
                           "rect": [float(max(x0, 0)), float(max(y0, 0)),
                                    float(min(x1, W)), float(min(y1, H))]})
        picks = []
        for cand in sorted(scored, key=lambda c: -c["score"]):
            if all(abs(order[cand["frame"]] - order[p["frame"]]) >= max(2, len(infos) // 10)
                   for p in picks):
                picks.append(cand)
            if len(picks) == per_box:
                break
        if picks:
            chosen[i] = picks
    return chosen


def draw_object_frames(space: Path, out: Path, include=None) -> dict[int, dict]:
    """Tile the best two frames for each box into `out`. `include` limits the
    boxes (indexes); by default every box not already outside the room.
    Returns best_frames' result for the tiles drawn."""
    space = Path(space)
    shapes = json.loads((space / "shapes.json").read_text())
    # Boxes that will be built: the checks' rejections stay rejected, so crops
    # of them would only take room on the sheet.
    boxes = {i: b for i, b in enumerate(shapes["boxes"])
             if (include is None or i in include)
             and "outside the room" not in (b.get("reason") or "")
             and b.get("build", True)}
    # The largest objects first, when there are more than fit on the sheet: a
    # cupboard seen at the edge of the video has few points but matters more
    # than a pillow.
    volume = lambda b: float(np.prod([b["max"][k] - b["min"][k] for k in range(3)]))
    boxes = dict(sorted(boxes.items(), key=lambda kv: -volume(kv[1]))[:MAX_TILES // 2])
    chosen = best_frames(space, boxes)
    if not chosen:
        return {}
    metre = json.loads((space / "densify.json").read_text()).get("colmap_units_per_metre")
    tiles = [(i, pick) for i, picks in sorted(chosen.items()) for pick in picks]
    cols = min(4, len(tiles))
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * TILE_W, rows * TILE_H), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=18)
    for n, (i, pick) in enumerate(tiles):
        img = Image.open(space / "workspace" / "images" / pick["frame"]).convert("RGB")
        x0, y0, x1, y1 = pick["rect"]
        # Crop around the box with room to see what surrounds it, keeping the tile's shape.
        cxm, cym = (x0 + x1) / 2, (y0 + y1) / 2
        half_h = max((y1 - y0) * 0.8, (x1 - x0) * 0.8 * TILE_H / TILE_W, img.height * 0.25)
        half_w = half_h * TILE_W / TILE_H
        half_w, half_h = min(half_w, img.width / 2), min(half_h, img.height / 2)
        # Slide the window back inside the photo rather than padding it with black.
        cxm = min(max(cxm, half_w), img.width - half_w)
        cym = min(max(cym, half_h), img.height - half_h)
        crop = [cxm - half_w, cym - half_h, cxm + half_w, cym + half_h]
        tile = img.crop([round(v) for v in crop]).resize((TILE_W, TILE_H))
        k = TILE_W / (crop[2] - crop[0])
        k_y = TILE_H / (crop[3] - crop[1])
        tx, ty = (n % cols) * TILE_W, (n // cols) * TILE_H
        sheet.paste(tile, (tx, ty))
        draw.rectangle([tx + (x0 - crop[0]) * k, ty + (y0 - crop[1]) * k_y,
                        tx + (x1 - crop[0]) * k, ty + (y1 - crop[1]) * k_y],
                       outline=(0, 230, 90), width=4)
        box = shapes["boxes"][i]
        text = f"B{i} {box.get('detected') or box.get('label') or 'box'}"
        if metre:
            dims = [(box["max"][k] - box["min"][k]) / metre for k in range(3)]
            text += f"  {dims[0]:.1f}x{dims[1]:.1f}x{dims[2]:.1f} m"
        label_box = draw.textbbox((tx + 8, ty + 8), text, font=font)
        draw.rectangle([label_box[0] - 4, label_box[1] - 3, label_box[2] + 4, label_box[3] + 3],
                       fill="white")
        draw.text((tx + 8, ty + 8), text, fill="black", font=font)
        draw.text((tx + 8, ty + TILE_H - 26), pick["frame"], fill="white", font=font,
                  stroke_width=2, stroke_fill="black")
    sheet.save(out)
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("space")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    space = Path(args.space)
    out = Path(args.out) if args.out else space / "object-frames.png"
    for i, picks in sorted(draw_object_frames(space, out).items()):
        print(f"B{i}: " + ", ".join(p["frame"] + (" (detected there)" if p["detected"] else "")
                                    for p in picks))
    print(out)


if __name__ == "__main__":
    main()
