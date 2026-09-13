"""Parametric furniture builders for detected bounding boxes.

Importable inside Blender (headless or interactive).  Each builder receives
the world-space (Z-up) min/max corners of a detected box and fills that
volume with a plausible piece of furniture built from cube primitives.  All
parts are parented under a new Empty named after the piece, so the whole
item moves/rotates/scales as one object in the editor.

Public contract:

    LIBRARY: dict[str, callable]      label -> builder(name, lo, hi, color)
    build(label, name, lo, hi, color) dispatch, falls back to "block"

`lo`/`hi` are [x, y, z] world coordinates; `color` is [r, g, b] 0-255.
Every builder returns the parent Empty.
"""

import zlib
from random import Random

import bpy
from mathutils import Vector

# Proportions are always fractions of the detected bounds — never absolute
# units — so the same builder works for a doll-house scan or a warehouse.

_EPS = 1e-6


# --------------------------------------------------------------------------
# shared helpers
# --------------------------------------------------------------------------

def _material(name, color):
    """New material showing `color` in both Cycles/EEVEE and Workbench."""
    rgba = (
        max(0.0, min(1.0, color[0] / 255.0)),
        max(0.0, min(1.0, color[1] / 255.0)),
        max(0.0, min(1.0, color[2] / 255.0)),
        1.0,
    )
    mat = bpy.data.materials.new(name)
    if mat.node_tree is None:  # use_nodes is deprecated once trees are default
        mat.use_nodes = True
    bsdf = next(
        (n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
    if bsdf is None:
        bsdf = mat.node_tree.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = rgba
    bsdf.inputs["Roughness"].default_value = 0.8
    mat.diffuse_color = rgba  # Workbench MATERIAL color mode reads this
    return mat


def _box(name, center, size, mat):
    """One cube spanning `size` around `center`, with material `mat`."""
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=tuple(center))
    obj = bpy.context.active_object
    obj.name = name
    obj.scale = (max(size[0], _EPS), max(size[1], _EPS), max(size[2], _EPS))
    obj.data.materials.append(mat)
    return obj


def _finish(name, lo, hi, parts):
    """Create the parent Empty at the footprint center and adopt `parts`."""
    cx, cy = (lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2
    bpy.ops.object.empty_add(type="PLAIN_AXES", location=(cx, cy, lo[2]))
    empty = bpy.context.active_object
    empty.name = name
    bpy.context.view_layer.update()  # settle matrix_world before parenting
    inv = empty.matrix_world.inverted()
    for part in parts:
        part.parent = empty
        part.matrix_parent_inverse = inv
    return empty


def _dims(lo, hi):
    lo = Vector(lo)
    hi = Vector(hi)
    return lo, hi, Vector((max(hi[i] - lo[i], _EPS) for i in range(3)))


# --------------------------------------------------------------------------
# builders
# --------------------------------------------------------------------------

def build_block(name, lo, hi, color):
    """Plain box filling the bounds — the universal fallback."""
    lo, hi, d = _dims(lo, hi)
    mat = _material(name, color)
    part = _box(f"{name}.body", (lo + hi) / 2, d, mat)
    return _finish(name, lo, hi, [part])


def _facing(facing):
    """"+x" / "-y" style direction -> (axis index, +1 or -1), or None."""
    if not facing or len(facing) != 2 or facing[1] not in "xy" or facing[0] not in "+-":
        return None
    return (0 if facing[1] == "x" else 1), (1 if facing[0] == "+" else -1)


def build_bed(name, lo, hi, color, facing=None):
    """Low base, slightly inset mattress, one pillow near the head end. With
    `facing` ("+x", "-y", ...) the head is at that end; without it, at the
    `hi` end of the longer side."""
    lo, hi, d = _dims(lo, hi)
    mat = _material(name, color)
    cx, cy = (lo.x + hi.x) / 2, (lo.y + hi.y) / 2

    # the longer horizontal side is the bed's length; head at its `hi` end
    length_axis = 0 if d.x >= d.y else 1
    head_sign = 1
    if _facing(facing):
        length_axis, head_sign = _facing(facing)
    width_axis = 1 - length_axis
    length, width = d[length_axis], d[width_axis]

    base_h = 0.40 * d.z
    mattress_h = 0.45 * d.z
    pillow_h = 0.15 * d.z
    inset = 0.05 * width

    parts = [_box(
        f"{name}.base",
        (cx, cy, lo.z + base_h / 2),
        (d.x, d.y, base_h),
        mat,
    )]

    m_size = [0.0, 0.0, mattress_h]
    m_size[length_axis] = length - 2 * inset
    m_size[width_axis] = width - 2 * inset
    parts.append(_box(
        f"{name}.mattress",
        (cx, cy, lo.z + base_h + mattress_h / 2),
        m_size,
        mat,
    ))

    p_size = [0.0, 0.0, pillow_h]
    p_size[length_axis] = 0.18 * length
    p_size[width_axis] = 0.55 * width
    p_center = [cx, cy, lo.z + base_h + mattress_h + pillow_h / 2]
    head_end = hi[length_axis] if head_sign > 0 else lo[length_axis]
    p_center[length_axis] = head_end - head_sign * (
        inset + 0.06 * length + p_size[length_axis] / 2)
    parts.append(_box(f"{name}.pillow", p_center, p_size, mat))

    return _finish(name, lo, hi, parts)


def build_seat(name, lo, hi, color):
    """Solid base with a thinner, slightly inset cushion on top."""
    lo, hi, d = _dims(lo, hi)
    mat = _material(name, color)
    cx, cy = (lo.x + hi.x) / 2, (lo.y + hi.y) / 2

    base_h = 0.65 * d.z
    cushion_h = d.z - base_h
    inset = 0.06 * min(d.x, d.y)

    parts = [
        _box(
            f"{name}.base",
            (cx, cy, lo.z + base_h / 2),
            (d.x, d.y, base_h),
            mat,
        ),
        _box(
            f"{name}.cushion",
            (cx, cy, lo.z + base_h + cushion_h / 2),
            (d.x - 2 * inset, d.y - 2 * inset, cushion_h),
            mat,
        ),
    ]
    return _finish(name, lo, hi, parts)


def build_table(name, lo, hi, color):
    """Thin top slab carried by four corner legs."""
    lo, hi, d = _dims(lo, hi)
    mat = _material(name, color)
    cx, cy = (lo.x + hi.x) / 2, (lo.y + hi.y) / 2

    top_t = 0.08 * d.z
    leg = 0.12 * min(d.x, d.y)
    leg_h = d.z - top_t

    parts = [_box(
        f"{name}.top",
        (cx, cy, hi.z - top_t / 2),
        (d.x, d.y, top_t),
        mat,
    )]
    for i, (sx, sy) in enumerate(((-1, -1), (1, -1), (-1, 1), (1, 1)), 1):
        parts.append(_box(
            f"{name}.leg{i}",
            (cx + sx * (d.x - leg) / 2, cy + sy * (d.y - leg) / 2,
             lo.z + leg_h / 2),
            (leg, leg, leg_h),
            mat,
        ))
    return _finish(name, lo, hi, parts)


def build_wardrobe(name, lo, hi, color, facing=None):
    """Tall body with two front door panels split by a thin inset gap,
    plus small handles either side of the split. With `facing` ("+x", ...)
    the doors face that way; without it, the `hi` end of the shorter side."""
    lo, hi, d = _dims(lo, hi)
    mat = _material(name, color)

    # the shorter horizontal side is the depth; the front faces its hi end
    depth_axis = 0 if d.x <= d.y else 1
    front_sign = 1
    if _facing(facing):
        depth_axis, front_sign = _facing(facing)
    width_axis = 1 - depth_axis
    depth, width = d[depth_axis], d[width_axis]

    door_t = 0.06 * depth
    gap = 0.015 * width
    margin = 0.02 * width  # door inset from the sides / top / bottom
    door_w = (width - 2 * margin - gap) / 2
    door_h = d.z - 2 * margin
    mid_w = (lo[width_axis] + hi[width_axis]) / 2
    mid_d = (lo[depth_axis] + hi[depth_axis]) / 2 - front_sign * door_t / 2
    front = hi[depth_axis] if front_sign > 0 else lo[depth_axis]

    body_center = [0.0, 0.0, (lo.z + hi.z) / 2]
    body_size = [0.0, 0.0, d.z]
    body_center[width_axis] = mid_w
    body_center[depth_axis] = mid_d
    body_size[width_axis] = width
    body_size[depth_axis] = depth - door_t
    parts = [_box(f"{name}.body", body_center, body_size, mat)]

    door_front = front - front_sign * door_t / 2
    handle_s = min(0.035 * width, 0.5 * door_t + 0.02 * width)
    handle_z = lo.z + 0.52 * d.z
    for side, tag in ((-1, "left"), (1, "right")):
        door_center = [0.0, 0.0, lo.z + margin + door_h / 2]
        door_size = [0.0, 0.0, door_h]
        door_center[width_axis] = mid_w + side * (gap / 2 + door_w / 2)
        door_center[depth_axis] = door_front
        door_size[width_axis] = door_w
        door_size[depth_axis] = door_t
        parts.append(_box(f"{name}.door_{tag}", door_center, door_size, mat))

        handle_center = [0.0, 0.0, handle_z]
        handle_size = [0.0, 0.0, 3 * handle_s]
        handle_center[width_axis] = mid_w + side * (gap / 2 + 1.5 * handle_s)
        handle_center[depth_axis] = front + front_sign * handle_s / 2
        handle_size[width_axis] = handle_s
        handle_size[depth_axis] = handle_s
        parts.append(_box(
            f"{name}.handle_{tag}", handle_center, handle_size, mat))

    return _finish(name, lo, hi, parts)


def build_clutter(name, lo, hi, color):
    """A few small irregular boxes scattered on the bottom of the bounds —
    deliberately low visual weight for unidentified point blobs."""
    lo, hi, d = _dims(lo, hi)
    mat = _material(name, color)
    rng = Random(zlib.crc32(name.encode("utf-8")))  # stable per name

    parts = []
    for i in range(rng.randint(5, 7)):
        size = (
            d.x * rng.uniform(0.08, 0.20),
            d.y * rng.uniform(0.08, 0.20),
            d.z * rng.uniform(0.15, 0.45),
        )
        center = (
            rng.uniform(lo.x + size[0] / 2, hi.x - size[0] / 2),
            rng.uniform(lo.y + size[1] / 2, hi.y - size[1] / 2),
            lo.z + size[2] / 2,  # rest on the bottom of the bounds
        )
        parts.append(_box(f"{name}.piece{i + 1}", center, size, mat))
    return _finish(name, lo, hi, parts)


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------

LIBRARY = {
    "bed": build_bed,
    "seat": build_seat,
    "table": build_table,
    "wardrobe": build_wardrobe,
    "clutter": build_clutter,
    "block": build_block,
}


def build(label, name, lo, hi, color, facing=None):
    """Build furniture for a detected box; unknown labels become a block.
    `facing` ("+x", "-y", ...) turns beds (head end) and wardrobes (doors).

    Returns the parent Empty holding all created mesh parts.
    """
    builder = LIBRARY.get(label, build_block)
    if builder in (build_bed, build_wardrobe):
        return builder(name, lo, hi, color, facing=facing)
    return builder(name, lo, hi, color)
