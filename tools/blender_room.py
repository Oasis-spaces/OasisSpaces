"""Regenerate a detected room as an editable Blender scene.

Run headless via Blender's bundled Python:

    blender --background --python tools/blender_room.py -- \
        spaces/<name>/shapes.json spaces/<name>/room.blend [render.png]

Reads shapes.json from pipeline/shapes.py and creates one named, colored
mesh object per detected shape (Floor, Wall_1..N, Furniture_1..N), saves a
.blend, and optionally renders a preview.
"""

import json
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Vector

argv = sys.argv[sys.argv.index("--") + 1:]
shapes_path, blend_path = argv[0], argv[1]
render_path = argv[2] if len(argv) > 2 else None

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    import furniture_library
except ImportError:
    furniture_library = None

shapes = json.loads(open(shapes_path).read())

# clean default scene
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete()


def material(name, rgb):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    color = (rgb[0] / 255, rgb[1] / 255, rgb[2] / 255, 1.0)
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = color
    bsdf.inputs["Roughness"].default_value = 0.85
    mat.diffuse_color = color  # Workbench/viewport uses this one
    return mat


counters = {}


def next_name(base):
    counters[base] = counters.get(base, 0) + 1
    return base if counters[base] == 1 else f"{base}_{counters[base]}"


for plane in shapes["planes"]:
    label = plane.get("label") or (
        "floor" if plane["kind"] == "floor_or_ceiling" else "wall")
    if label == "ceiling":
        continue  # keep renders and editing unobstructed
    if plane.get("build") is False:
        continue  # dropped in review: a duplicate, or not a wall at all
    a = Vector(plane["axis_a"])
    b = Vector(plane["axis_b"])
    n = Vector(plane["normal"])
    center = Vector(plane["center"])
    # Inferred walls close sides nobody filmed; the name (and their light
    # colour from shapes.json) keeps them apart from measured walls.
    name = next_name("Wall_inferred" if plane.get("source") == "inferred"
                     else label.capitalize())
    bpy.ops.mesh.primitive_plane_add(size=2)
    obj = bpy.context.active_object
    obj.name = name
    obj.matrix_world = Matrix((
        (a.x * plane["half_a"], b.x * plane["half_b"], n.x, center.x),
        (a.y * plane["half_a"], b.y * plane["half_b"], n.y, center.y),
        (a.z * plane["half_a"], b.z * plane["half_b"], n.z, center.z),
        (0, 0, 0, 1),
    ))
    obj.data.materials.append(material(name, plane["color"]))

for i, box in enumerate(shapes["boxes"], 1):
    if not box.get("build", True):
        continue  # classifier rejected it: scan debris, not furniture
    lo, hi = Vector(box["min"]), Vector(box["max"])
    label = box.get("label", "block")
    name = next_name(label.capitalize())
    if furniture_library is not None:
        furniture_library.build(label, name, list(lo), list(hi), box["color"])
    else:
        bpy.ops.mesh.primitive_cube_add(size=1)
        obj = bpy.context.active_object
        obj.name = name
        obj.location = (lo + hi) / 2
        obj.scale = hi - lo
        obj.data.materials.append(material(name, box["color"]))

# camera + light framing the WHOLE scene bound
corners = []
for obj in bpy.data.objects:
    if obj.type == "MESH":
        corners += [obj.matrix_world @ Vector(c) for c in obj.bound_box]
lo = Vector((min(c[i] for c in corners) for i in range(3)))
hi = Vector((max(c[i] for c in corners) for i in range(3)))
center = (lo + hi) / 2
span = (hi - lo).length

# shapes.json coordinates are already Z-up (pipeline/shapes.py transforms
# them), so frame the views with the scene's own axes. shapes["up"] and
# shapes["world"] describe the camera solve's frame, not this one.
up = Vector((0.0, 0.0, 1.0))
side = Vector((1.0, 0.0, 0.0))
back = Vector((0.0, 1.0, 0.0))
bpy.ops.object.light_add(type="SUN", location=center + up * span)
bpy.context.active_object.data.energy = 3

# Give walls a little thickness so they still show from straight above.
for obj in bpy.data.objects:
    if obj.type == "MESH" and obj.name.startswith("Wall"):
        obj.modifiers.new("Thickness", "SOLIDIFY").thickness = span * 0.01


def look_at(camera, position, target, image_up):
    forward = (target - position).normalized()
    cam_up = (image_up - forward * image_up.dot(forward)).normalized()
    right = forward.cross(cam_up).normalized()
    # a camera looks along its -Z with +Y as image-up
    camera.matrix_world = Matrix((
        (right.x, cam_up.x, -forward.x, position.x),
        (right.y, cam_up.y, -forward.y, position.y),
        (right.z, cam_up.z, -forward.z, position.z),
        (0, 0, 0, 1),
    ))


cam_pos = center + (side * 0.9 + back * -0.9 + up * 0.7) * span
bpy.ops.object.camera_add(location=cam_pos)
cam = bpy.context.active_object
cam.name = "Camera_Perspective"
look_at(cam, cam_pos, center, up)
cam.data.clip_end = span * 40

# A second camera straight above, orthographic: the floor plan.
RES_X, RES_Y = 1280, 900
bpy.ops.object.camera_add(location=center + up * span)
plan = bpy.context.active_object
plan.name = "Camera_Plan"
look_at(plan, center + up * span, center, back)
plan.data.type = "ORTHO"
plan.data.ortho_scale = max(hi.x - lo.x, (hi.y - lo.y) * RES_X / RES_Y) * 1.1
plan.data.clip_end = span * 40
bpy.context.scene.camera = cam

bpy.ops.wm.save_as_mainfile(filepath=blend_path)
print(f"saved {blend_path}")

if render_path:
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.display.shading.color_type = "MATERIAL"
    scene.display.shading.show_object_outline = True
    scene.render.resolution_x = RES_X
    scene.render.resolution_y = RES_Y
    plan_path = str(Path(render_path).with_name(Path(render_path).stem + "-plan.png"))
    # Cutaway for the angled view, like a dollhouse: hide the walls between the
    # camera and the room's centre so the furniture inside shows. Render-only;
    # the saved .blend keeps every wall.
    toward_camera = cam_pos - center
    toward_camera.z = 0.0
    near_walls = [o for o in bpy.data.objects
                  if o.type == "MESH" and o.name.startswith("Wall")
                  and (o.matrix_world.translation - center).dot(toward_camera) > 0]
    for camera, path, hidden in ((cam, render_path, near_walls), (plan, plan_path, [])):
        for wall in near_walls:
            wall.hide_render = wall in hidden
        scene.camera = camera
        scene.render.filepath = path
        bpy.ops.render.render(write_still=True)
        print(f"rendered {path}")
