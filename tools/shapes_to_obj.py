#!/usr/bin/env python3
"""shapes.json -> room-model.obj/.mtl (opens in Blender, Preview, anything).

Usage: python3 tools/shapes_to_obj.py spaces/<name>
"""

import json
import sys
from pathlib import Path

if len(sys.argv) < 2:
    sys.exit(__doc__)
space = Path(sys.argv[1])
shapes = json.loads((space / "shapes.json").read_text())

obj_lines = [f"mtllib room-model.mtl"]
mtl_lines = []
vertex_count = 0


def add_material(name, rgb):
    mtl_lines.append(
        f"newmtl {name}\nKd {rgb[0]/255:.3f} {rgb[1]/255:.3f} {rgb[2]/255:.3f}\n")


def add_quad(name, corners):
    global vertex_count
    obj_lines.append(f"o {name}\nusemtl {name}")
    for c in corners:
        obj_lines.append(f"v {c[0]:.4f} {c[1]:.4f} {c[2]:.4f}")
    base = vertex_count
    obj_lines.append(f"f {base+1} {base+2} {base+3} {base+4}")
    vertex_count += 4


wall_i = floor_i = 0
for p in shapes["planes"]:
    import numpy as np
    a, b = np.array(p["axis_a"]), np.array(p["axis_b"])
    c = np.array(p["center"])
    ha, hb = p["half_a"], p["half_b"]
    corners = [c - a * ha - b * hb, c + a * ha - b * hb,
               c + a * ha + b * hb, c - a * ha + b * hb]
    if p["kind"] == "floor_or_ceiling":
        floor_i += 1
        name = f"Floor_{floor_i}"
    else:
        wall_i += 1
        name = f"Wall_{wall_i}"
    add_material(name, p["color"])
    add_quad(name, corners)

for i, box in enumerate(shapes["boxes"], 1):
    import numpy as np
    lo, hi = np.array(box["min"]), np.array(box["max"])
    name = f"Furniture_{i}"
    add_material(name, box["color"])
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    quads = [
        [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)],
        [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)],
        [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)],
        [(x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)],
        [(x0, y0, z0), (x0, y1, z0), (x0, y1, z1), (x0, y0, z1)],
        [(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)],
    ]
    for j, q in enumerate(quads):
        add_quad(name if j == 0 else f"{name}", q)

(space / "room-model.obj").write_text("\n".join(obj_lines) + "\n")
(space / "room-model.mtl").write_text("\n".join(mtl_lines))
print(f"wrote {space}/room-model.obj ({vertex_count} vertices)")
