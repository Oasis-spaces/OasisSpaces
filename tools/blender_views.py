"""Render a built room from the video's own cameras. Run by Blender:

    blender --background --python tools/blender_views.py -- \
        spaces/<name>/room.blend views.json out_dir/

views.json (tools/room_views.py writes it) lists one view per frame:
  {"name": "frame_00019", "matrix": 4x4 camera-to-world in Blender's
   convention (looking along -Z, +Y up), in the room's own coordinates,
   "fx", "fy", "cx", "cy", "width", "height": the frame's calibration,
   "render_width": pixels}
Each is rendered to out_dir/<name>.png at render_width, keeping the frame's
aspect, with the same look as blender_room.py's preview. The principal point
is taken as the frame's centre (COLMAP's single-camera solves keep it there
to within a few pixels); lens distortion is not applied.
"""
import json
import os
import sys
from pathlib import Path

import bpy
from mathutils import Matrix

argv = sys.argv[sys.argv.index("--") + 1:]
blend_path, views_path, out_dir = argv[0], argv[1], Path(argv[2])
out_dir.mkdir(parents=True, exist_ok=True)

bpy.ops.wm.open_mainfile(filepath=blend_path)
scene = bpy.context.scene
# The same engine choice as blender_room.py: Workbench where there is a
# display, Cycles on the CPU where there is none (Colab).
if os.environ.get("OASIS_RENDER_ENGINE", "").upper() == "CYCLES":
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 16
    world = bpy.data.worlds.get("World") or bpy.data.worlds.new("World")
    scene.world = world
    world.color = (0.8, 0.8, 0.8)
else:
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.display.shading.color_type = "MATERIAL"
    scene.display.shading.show_object_outline = True
scene.render.resolution_percentage = 100

# Flat grey floor against flat grey walls hid their junction in these renders
# (Claude read a measured wall as "built too close"), so each kind gets its own
# tone: sand floor, pale walls, blue-grey furniture. The .blend keeps the
# measured colours; this is render-only.
TONES = {"floor": (0.80, 0.72, 0.56, 1.0), "wall": (0.86, 0.86, 0.88, 1.0), "furniture": (0.50, 0.56, 0.72, 1.0)}
for obj in bpy.data.objects:
    if obj.type != "MESH":
        continue
    kind = "floor" if obj.name.startswith("Floor") else "wall" if obj.name.startswith("Wall") else "furniture"
    mat = bpy.data.materials.new(f"view-{kind}")
    mat.use_nodes = True
    mat.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = TONES[kind]
    mat.diffuse_color = TONES[kind]
    obj.data.materials.clear()
    obj.data.materials.append(mat)

views = json.loads(Path(views_path).read_text())
for view in views:
    data = bpy.data.cameras.new(view["name"])
    camera = bpy.data.objects.new(view["name"], data)
    scene.collection.objects.link(camera)
    camera.matrix_world = Matrix(view["matrix"])
    # The horizontal field of view is the frame's: sensor width is arbitrary,
    # the lens follows from it and the calibrated focal length.
    data.sensor_fit = "HORIZONTAL"
    data.sensor_width = 36.0
    data.lens = view["fx"] * 36.0 / view["width"]
    data.clip_start = 0.05
    data.clip_end = 1.0e5
    width = int(view.get("render_width", 540))
    scene.render.resolution_x = width
    scene.render.resolution_y = max(1, round(width * view["height"] / view["width"]))
    scene.camera = camera
    scene.render.filepath = str(out_dir / f"{view['name']}.png")
    bpy.ops.render.render(write_still=True)
    print(f"rendered {scene.render.filepath}")
