#!/usr/bin/env python3
"""Draw a space's floor plan with every structure candidate numbered.

A top-down view of the dense cloud (darker = more points), with the floor
outline in blue, walls as red lines numbered W<i> (thin orange when inferred
for a side nobody filmed), and furniture boxes as green
rectangles numbered B<i> (grey when a check rejected them). The numbers are
indexes into shapes.json, so agent.py can show this to Claude and apply its
decisions by number; it is also a quick visual check for a person. A scale bar
is drawn in metres when densify.json recorded a metric scale.

Usage:
    python3 tools/plan_image.py spaces/<name> [--out plan-candidates.png]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
from pointcloud import load_ply  # noqa: E402

SIZE = 1100
MARGIN = 70


def draw_plan(space: Path, out: Path) -> Path:
    space = Path(space)
    shapes = json.loads((space / "shapes.json").read_text())
    meta_path = space / "densify.json"
    units_per_metre = (json.loads(meta_path.read_text()).get("colmap_units_per_metre")
                       if meta_path.exists() else None)
    world = np.array(shapes["world"])

    cloud = load_ply(space / "cloud-dense.ply")
    pts = cloud.points
    if len(pts) > 400_000:
        pts = pts[np.random.default_rng(0).choice(len(pts), 400_000, replace=False)]
    xy = (pts.astype(np.float64) @ world.T)[:, :2]

    lo = np.percentile(xy, 1, axis=0)
    hi = np.percentile(xy, 99, axis=0)
    span = float(max(hi - lo))
    scale = (SIZE - 2 * MARGIN) / span
    center = (lo + hi) / 2

    def px(p):
        """Scene x, y -> image pixel (y up in the scene is up in the image)."""
        return (SIZE / 2 + (p[0] - center[0]) * scale,
                SIZE / 2 - (p[1] - center[1]) * scale)

    # Point density as the background.
    bins = SIZE // 4
    hist, _, _ = np.histogram2d(
        xy[:, 0], xy[:, 1], bins=bins,
        range=[[center[0] - SIZE / 2 / scale, center[0] + SIZE / 2 / scale],
               [center[1] - SIZE / 2 / scale, center[1] + SIZE / 2 / scale]])
    density = np.log1p(hist.T[::-1])
    density = (255 - 200 * density / max(density.max(), 1e-9)).astype(np.uint8)
    img = Image.fromarray(density, "L").resize((SIZE, SIZE), Image.NEAREST).convert("RGB")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=22)
    small = ImageFont.load_default(size=16)

    def label(text, at, colour):
        x, y = at
        box = draw.textbbox((x, y), text, font=font, anchor="mm")
        draw.rectangle([box[0] - 3, box[1] - 2, box[2] + 3, box[3] + 2], fill="white")
        draw.text((x, y), text, fill=colour, font=font, anchor="mm")

    for i, plane in enumerate(shapes["planes"]):
        c, a, b = (np.array(plane[k]) for k in ("center", "axis_a", "axis_b"))
        ha, hb = plane["half_a"], plane["half_b"]
        dropped = plane.get("build") is False
        if plane.get("label") == "floor" or (plane["kind"] == "floor_or_ceiling"
                                             and plane.get("label") != "ceiling"):
            corners = [c + sa * ha * a + sb * hb * b
                       for sa, sb in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
            draw.polygon([px(p) for p in corners], outline=(40, 90, 220), width=4)
            label(f"F{i}", px(c + ha * a * 0.8), (40, 90, 220))
        elif plane["kind"] == "wall":
            ends = [px(c - ha * a), px(c + ha * a)]
            inferred = plane.get("source") == "inferred"
            colour = ((170, 170, 170) if dropped else (230, 140, 30) if inferred
                      else (210, 40, 40))
            draw.line(ends, fill=colour, width=3 if inferred else 6)
            mid = ((ends[0][0] + ends[1][0]) / 2, (ends[0][1] + ends[1][1]) / 2)
            label(f"W{i} inferred" if inferred else f"W{i}", mid, colour)

    for i, box in enumerate(shapes["boxes"]):
        (x0, y0), (x1, y1) = px(box["min"]), px(box["max"])
        built = box.get("build", True)
        colour = (30, 150, 60) if built else (160, 160, 160)
        draw.rectangle([min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)],
                       outline=colour, width=4 if built else 2)
        name = box.get("detected") or box.get("label") or "box"
        label(f"B{i} {name}", ((x0 + x1) / 2, (y0 + y1) / 2), colour)

    if units_per_metre:
        bar = units_per_metre * scale
        y = SIZE - 30
        draw.line([(MARGIN, y), (MARGIN + bar, y)], fill="black", width=5)
        draw.text((MARGIN + bar + 10, y), "1 m", fill="black", font=small, anchor="lm")
    draw.text((MARGIN, 25), "Floor plan, seen from above: darker = more 3D points",
              fill="black", font=small, anchor="lm")
    img.save(out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("space")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    space = Path(args.space)
    out = Path(args.out) if args.out else space / "plan-candidates.png"
    print(draw_plan(space, out))


if __name__ == "__main__":
    main()
