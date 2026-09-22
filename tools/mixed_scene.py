#!/usr/bin/env python3
"""Build a room as a mixed scene: a clean textured shell, and the furniture as
it was filmed, each piece on its own.

A splat of a phone video is soft where the room is plain (walls, floor) and
cannot be edited: moving a bed leaves a hole, because the floor under it was
never filmed. This turns a processed space into something that can:

  shell.glb       the floor, walls and ceiling stage 3 measured, as flat
                  meshes textured with the video's own frames. Every frame is
                  projected onto each surface, keeping only views nothing
                  blocks (tools/surface_fill.py), and what no frame saw (behind
                  the wardrobe, under the bed) is continued by an inpainting
                  model from the surface around it. So the background behind
                  every piece of furniture already exists.
  pieces/*.splat  the trained splat, cut up: one file per measured object (its
                  own Gaussians, as filmed), one for the ceiling's fittings,
                  one for the rest (clutter, curtains, things on the walls).
                  The Gaussians that were the walls and floor are dropped: the
                  shell replaces them.
  models/*.glb    a clean stand-in for every piece: simple furniture of the
                  piece's kind, at its measured size, in the scan's own
                  colours, with its back to the wall the piece stands against.
                  The viewer can show either; a poor scan starts as its model.
  scene.json      what is where: each piece's label, box and anchor, in metres,
                  y up, the room's centre on the floor as origin (three.js
                  conventions), so a viewer can select, move, turn and hide
                  pieces. scene-viewer/ is that viewer.

    python3 tools/mixed_scene.py spaces/<name> [--cell 0.005] [--splat splat.ply] [--claude]

With --claude (and always in the pipeline, stage 4's `scene` step) Claude looks
at every surface's texture beside what was actually filmed, and at every piece
beside a frame of the real thing, and decides: keep a texture, keep only its
filmed part, or paint the surface plain; show a piece as scanned, as its clean
model, or not at all (review-surfaces.png, review-pieces.png, review.json).

Needs stage 4's splat, stage 3's shapes.json, the frames in workspace/images
and LaMa's weights (~/.cache/oasisspaces/big-lama.pt), like fill-room.
"""

from __future__ import annotations

import argparse
import io
import json
import struct
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(ROOT / "tools"))

SH_C0 = 0.28209479177387814
SHELL_BAND_M = 0.08         # Gaussians this close to a wall, the floor or the ceiling are that surface
WALL_BAND_M = 0.30          # ...and out to here when they are the wall's own colour (soft paint-coloured blobs)
FITTING_BAND_M = 0.45       # things hanging this far under the ceiling are its fittings (a fan, a lamp)
OUTSIDE_M = 0.12            # beyond the room by this much: seen through a door or window, or a floater
MIN_ALPHA = 0.05            # fainter Gaussians are dropped
MIN_PIECE = 200             # a measured object with fewer Gaussians than this is left in the rest
NESTED = 0.8                # a box with this much of its volume inside a larger one is part of that piece
HAZE_SUPPORT_M = 0.08       # a soft Gaussian with no dense-cloud surface this close is haze, not a thing
HAZE_ALPHA = 0.6            # ...soft meaning fainter than this,
HAZE_SCALE_M = 0.05         # ...or larger than this
PANEL_CELL_M = 0.0025       # texels of a photographed furniture face: a pattern on a door needs finer than paint
PANEL_SHARPNESS = 4.0       # its photograph is decided by the closest, most frontal frames (surface_fill.photograph)
SPECKLE_GAP = 30            # a filmed texel this far (RGB) from its neighbourhood's median is a speckle
PANEL_LIFT_M = 0.004        # a photographed face sits this far in front of its model's side
THING_REACH_M = 0.05        # a leftover Gaussian looks this far for measured points...
THING_NEIGHBOURS = 8        # ...at up to this many of them,
THING_MIN = 4               # needs this many,
THING_SHARE = 0.5           # and this share of them detected as some object, to be a thing and not paint
FRONT_CLEAR_M = 0.15        # what stands this far in front of a face hides it (a bed before a wardrobe)...
FRONT_CELL_M = 0.02         # ...counted on cells this size,
FRONT_POINTS = 8            # with this many measured points in one
PANEL_MIN_FILMED = 0.35     # a face filmed less than this stays the model's plain colour
TURNED_DEGREES = 55         # the review also shows each scan from this far round, and from above
SEEN_WEIGHT = 0.02          # a texel counts as filmed above this (as in surface_fill)
ROUGHNESS = {"floor": 0.45, "ceiling": 1.0, "wall": 0.92}

# scene (x, y along the walls, z up) -> viewer (x, y up, z towards the viewer): a rotation.
TO_VIEWER = np.array([[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]])


# ------------------------------------------------------------------ frames
class Frame:
    """Splat (camera-solve) coordinates to the viewer's: metres, y up, the
    room's centre on the floor at the origin."""

    def __init__(self, room, floor_height: float):
        self.metre = room.metre
        self.origin = np.array([room.centre[0], room.centre[1], floor_height])
        self.world = room.world                      # scene = world @ splat
        self.rotation = TO_VIEWER @ room.world       # viewer = rotation @ splat (then scale, shift)
        assert np.linalg.det(self.rotation) > 0.99, "the room's frame is not a rotation"

    def scene_to_viewer(self, scene: np.ndarray) -> np.ndarray:
        return ((np.asarray(scene, float) - self.origin) / self.metre) @ TO_VIEWER.T

    def direction(self, scene_vector: np.ndarray) -> np.ndarray:
        return TO_VIEWER @ np.asarray(scene_vector, float)


def quaternion_of(R: np.ndarray) -> np.ndarray:
    """(w, x, y, z) of a rotation matrix."""
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        q = np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = np.sqrt(R[i, i] - R[j, j] - R[k, k] + 1.0) * 2
        q = np.zeros(4)
        q[1 + i] = 0.25 * s
        q[0] = (R[k, j] - R[j, k]) / s
        q[1 + j] = (R[j, i] + R[i, j]) / s
        q[1 + k] = (R[k, i] + R[i, k]) / s
    return q / np.linalg.norm(q)


def multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Quaternion a times each row of b, (w, x, y, z)."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return np.stack([aw * bw - ax * bx - ay * by - az * bz,
                     aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw], axis=1)


# ------------------------------------------------------------------ pieces
def write_piece(path: Path, arr: np.ndarray, frame: Frame, scene: np.ndarray, anchor: np.ndarray) -> int:
    """The Gaussians as a .splat file (32 bytes each: position, scale, colour
    and opacity, rotation) in the viewer's frame, relative to `anchor`, the
    largest and most solid first."""
    log_scales = np.stack([arr["scale_0"], arr["scale_1"], arr["scale_2"]], axis=1).astype(np.float64)
    alpha = 1 / (1 + np.exp(-arr["opacity"].astype(np.float64)))
    order = np.argsort(-np.exp(log_scales.sum(axis=1)) * alpha)
    out = np.zeros(len(arr), dtype=[("position", "<f4", 3), ("scale", "<f4", 3),
                                    ("rgba", "u1", 4), ("rotation", "u1", 4)])
    out["position"] = frame.scene_to_viewer(scene) - anchor
    out["scale"] = np.exp(log_scales) / frame.metre
    rgb = 0.5 + SH_C0 * np.stack([arr["f_dc_0"], arr["f_dc_1"], arr["f_dc_2"]], axis=1)
    out["rgba"] = np.clip(np.column_stack([rgb, alpha]) * 255, 0, 255).astype(np.uint8)
    quat = np.stack([arr["rot_0"], arr["rot_1"], arr["rot_2"], arr["rot_3"]], axis=1).astype(np.float64)
    quat /= np.maximum(np.linalg.norm(quat, axis=1, keepdims=True), 1e-12)
    quat = multiply(quaternion_of(frame.rotation), quat)
    out["rotation"] = np.clip(quat * 128 + 128, 0, 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(out[order].tobytes())
    return len(out)


def nested_in(los, his) -> dict:
    """{box: the larger box it lies inside} for boxes with NESTED of their
    volume inside a larger one. Stage 3 measures every box from the floor up,
    so a pillow on a bed is a column through the mattress: cut on its own it
    would take a core out of the bed, and protected from the bed it would stay
    behind in mid-air when the bed moves. It goes with the bed instead."""
    volume = np.prod(his - los, axis=1)
    inside = {}
    for k in np.argsort(volume):                     # each box looks for the largest box holding it
        for big in np.argsort(-volume):
            if volume[big] <= volume[k] or big == k:
                break
            shared = np.prod(np.clip(np.minimum(his[k], his[big]) - np.maximum(los[k], los[big]), 0, None))
            if volume[k] > 0 and shared / volume[k] >= NESTED:
                inside[int(k)] = int(big)
                break
    # A box inside a box that is itself inside a third goes with the outermost.
    for k in inside:
        while inside[k] in inside:
            inside[k] = inside[inside[k]]
    return inside


def claims(los, his, points, margin) -> np.ndarray:
    """The box each point belongs to (-1: none). Inside one box: that box;
    inside several: the smallest (a basket half under a desk keeps its body);
    outside them all: the nearest box within `margin`. A piece takes nothing
    another box claims, so a bed pushed against a wardrobe, its margin and the
    parts growing up out of it leave the wardrobe's doors on the wardrobe."""
    owner = np.full(len(points), -1)
    volume = np.prod(his - los, axis=1)
    for k in np.argsort(-volume):                    # largest first: a smaller box overwrites
        owner[np.all((points >= los[k]) & (points <= his[k]), axis=1)] = k
    free = np.flatnonzero(owner < 0)
    if len(free) and len(los):
        gap = np.stack([np.linalg.norm(np.maximum(np.maximum(lo - points[free], points[free] - hi), 0), axis=1)
                        for lo, hi in zip(los, his)])
        nearest = gap.argmin(axis=0)
        close = gap[nearest, np.arange(len(free))] <= margin
        owner[free[close]] = nearest[close]
    return owner


def cut_pieces(room, arr, scene, colours, alpha, floor_height, walls, log, detected=None):
    """Which Gaussians are which: {piece id: mask}. The room's surfaces are
    found first, so a piece never takes the wall it stands against with it:
    a Gaussian in a wall's band counts as the wall when it is the wall's
    colour (a headboard or a shelf on the wall is not). Then the objects
    (stage 3's built boxes, each within what it claims, largest first), then
    what is left.

    `detected(positions) -> mask` says which positions sit on something the
    detector outlined (on_detected_object). With it, what a piece takes from
    outside its measured box (the margin, a headboard growing up out of it)
    and everything left over must be on a detected object: the rest is wall
    paint hanging in the room, which a moved bed would drag along as a smear.
    Inside a box nothing is asked: outlines cover only part of a wardrobe."""
    from splat_edit import ABOVE_COLOUR_GAP, REMOVE_MARGIN_M, object_blobs, typical

    m = room.metre
    level = room.shapes["room_level"]
    ceiling = room.floor_z + level["height"]
    rel = scene[:, :2] - room.centre
    outside = (np.any(np.abs(rel) > room.half + OUTSIDE_M * m, axis=1)
               | (scene[:, 2] < floor_height - OUTSIDE_M * m) | (scene[:, 2] > ceiling + OUTSIDE_M * m))
    shell = np.abs(scene[:, 2] - floor_height) < SHELL_BAND_M * m
    shell |= scene[:, 2] > ceiling - SHELL_BAND_M * m
    for surface in walls:
        depth = (scene - surface.origin) @ surface.normal
        along = (scene - surface.origin) @ surface.u
        band = np.flatnonzero((np.abs(depth) < WALL_BAND_M * m) & (along > -0.1 * m)
                              & (along < surface.cols * surface.cell + 0.1 * m))
        if len(band) < 50:
            continue
        paint = np.median(colours[typical(band, colours)], axis=0)
        like = np.linalg.norm(colours[band] - paint, axis=1) < ABOVE_COLOUR_GAP
        close = np.abs(depth[band]) < SHELL_BAND_M * m
        shell[band[like | close]] = True
    shell |= outside | (alpha < MIN_ALPHA)

    boxes = [(i, b) for i, b in enumerate(room.shapes["boxes"]) if b.get("build", True)]
    volume = lambda b: float(np.prod(np.array(b["max"]) - np.array(b["min"])))
    boxes.sort(key=lambda ib: -volume(ib[1]))
    name = lambda ib: f"B{ib[0]} {ib[1].get('detected') or ib[1].get('label')}"
    corners = lambda which: np.array([b[which] for _, b in boxes], float).reshape(-1, 3)
    for k, big in sorted(nested_in(corners("min"), corners("max")).items()):
        log(f"  {name(boxes[k])}: inside {name(boxes[big])}, goes with it")
        boxes[k] = None
    boxes = [ib for ib in boxes if ib]
    # A box claiming too little to be a piece claims nothing: its few Gaussians
    # go to the box around or beside it, not to the rest.
    while boxes:
        owner = claims(corners("min"), corners("max"), scene, REMOVE_MARGIN_M * m)
        held = np.bincount(owner[(owner >= 0) & ~shell], minlength=len(boxes))
        if held.min() >= MIN_PIECE:
            break
        log(f"  {name(boxes[int(held.argmin())])}: only {int(held.min())} Gaussians, not a piece of its own")
        boxes.pop(int(held.argmin()))
    taken = np.zeros(len(arr), bool)
    paint = np.zeros(len(arr), bool)
    pieces = {}
    for k, (i, box) in enumerate(boxes):
        protected = shell | ((owner >= 0) & (owner != k))
        mask, _, _ = object_blobs(room, scene, colours, box, protected | taken, taken | shell)
        mask &= ~taken & ~shell
        if detected is not None:
            beyond = np.flatnonzero(mask & ~np.all((scene >= np.array(box["min"]) - 0.02 * m)
                                                   & (scene <= np.array(box["max"]) + 0.02 * m), axis=1))
            paint[beyond[~detected(scene[beyond])]] = True
            mask &= ~paint
        if mask.sum() < MIN_PIECE:
            log(f"  B{i} {box.get('label')}: only {int(mask.sum())} Gaussians, left in the rest")
            continue
        pieces[f"B{i}"] = mask
        taken |= mask
        log(f"  B{i} {box.get('detected') or box.get('label')}: {int(mask.sum()):,} Gaussians")

    left = ~taken & ~shell
    fittings = left & (scene[:, 2] > ceiling - FITTING_BAND_M * m)
    pieces["ceiling-fittings"] = fittings
    pieces["rest"] = left & ~fittings
    if detected is not None:
        around = int(paint.sum())
        rest = np.flatnonzero(pieces["rest"] & ~paint)
        paint[rest[~detected(scene[rest])]] = True
        pieces["rest"] &= ~paint
        log(f"  dropped {int(paint.sum()):,} Gaussians on no detected object (paint hanging off the walls): "
            f"{around:,} around the pieces, {int(paint.sum()) - around:,} from what was left")
    log(f"  dropped {int((shell & ~outside).sum()):,} Gaussians that were walls, floor or ceiling, "
        f"{int(outside.sum()):,} outside the room; {int(pieces['rest'].sum()):,} left as the rest")
    return pieces


# ------------------------------------------------------------------ models
# Which of splat_edit's simple furniture stands in for a label (first match wins).
MODEL_KINDS = [("bed", "bed"), ("mattress", "bed"), ("crib", "bed"), ("sofa", "sofa"), ("couch", "sofa"),
               ("armchair", "armchair"), ("chair", "chair"), ("stool", "chair"), ("bench", "sofa"),
               ("desk", "desk"), ("table", "table"), ("nightstand", "box"), ("wardrobe", "wardrobe"),
               ("cabinet", "wardrobe"), ("cupboard", "wardrobe"), ("dresser", "wardrobe"),
               ("drawers", "wardrobe"), ("shelf", "wardrobe"), ("bookshelf", "wardrobe"), ("fridge", "wardrobe")]
# (roughness, metallic) of the parts' materials; colours come from the scan.
FINISH = {"wood": (0.5, 0.0), "dark wood": (0.55, 0.0), "fabric": (0.95, 0.0), "cushion": (0.95, 0.0),
          "linen": (0.9, 0.0), "metal": (0.35, 1.0)}


def model_kind(label: str) -> str:
    label = (label or "").lower()
    return next((kind for word, kind in MODEL_KINDS if word in label), "box")


def model_parts(room, box: dict, label: str, colours: np.ndarray, heights: np.ndarray, frame: Frame, anchor: np.ndarray):
    """The clean stand-in for a measured object: [(min, max, finish, rgb)] boxes
    in the viewer's frame relative to `anchor`. The piece's depth runs along
    the box's longer or shorter side as its kind has it (a bed is deeper than
    wide, a wardrobe wider than deep), its back to the nearer wall."""
    from splat_edit import CUSHION, DARK_WOOD, FABRIC, LINEN, METAL, PIECES, WOOD

    default, parts = PIECES[model_kind(label)]
    lo, hi = np.array(box["min"], float), np.array(box["max"], float)
    size = hi - lo
    deep_is_long = default[1] > default[0]
    depth_axis = int(np.argmax(size[:2])) if deep_is_long else int(np.argmin(size[:2]))
    width_axis = 1 - depth_axis
    room_lo, room_hi = room.centre - room.half, room.centre + room.half
    back_at_hi = (room_hi[depth_axis] - hi[depth_axis]) <= (lo[depth_axis] - room_lo[depth_axis])

    # The scan's own colours: the whole piece, and its top (the bedding, a table's top).
    body = np.median(colours, axis=0) if len(colours) else np.array(WOOD, float)
    top = heights >= np.percentile(heights, 70) if len(heights) else np.zeros(0, bool)
    upper = np.median(colours[top], axis=0) if top.any() else body
    palette = {WOOD: ("wood", body), DARK_WOOD: ("dark wood", body * 0.72), FABRIC: ("fabric", body),
               CUSHION: ("cushion", np.minimum(body * 1.12, 255)), LINEN: ("linen", upper),
               METAL: ("metal", np.array(METAL, float))}
    out = []
    for cx, cy, z0, w, d, h, colour in parts:
        finish, rgb = palette[colour]
        part_lo, part_hi = lo.copy(), hi.copy()
        part_lo[width_axis] = lo[width_axis] + (cx - w / 2) * size[width_axis]
        part_hi[width_axis] = lo[width_axis] + (cx + w / 2) * size[width_axis]
        near, far = cy - d / 2, cy + d / 2                       # along the depth, the back at 1
        if back_at_hi:
            part_lo[depth_axis], part_hi[depth_axis] = lo[depth_axis] + near * size[depth_axis], lo[depth_axis] + far * size[depth_axis]
        else:
            part_lo[depth_axis], part_hi[depth_axis] = hi[depth_axis] - far * size[depth_axis], hi[depth_axis] - near * size[depth_axis]
        part_lo[2], part_hi[2] = lo[2] + z0 * size[2], lo[2] + min(1.6, z0 + h) * size[2]
        corners = frame.scene_to_viewer(np.array([part_lo, part_hi])) - anchor
        out.append((corners.min(axis=0), corners.max(axis=0), finish, np.clip(rgb, 0, 255)))
    return out


def cuboid(lo: np.ndarray, hi: np.ndarray):
    """24 vertices (4 a face, so the faces shade flat), their normals and 36 indices."""
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    faces = [((1, 0, 0), [(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)]),
             ((-1, 0, 0), [(x0, y0, z1), (x0, y1, z1), (x0, y1, z0), (x0, y0, z0)]),
             ((0, 1, 0), [(x0, y1, z0), (x0, y1, z1), (x1, y1, z1), (x1, y1, z0)]),
             ((0, -1, 0), [(x0, y0, z1), (x0, y0, z0), (x1, y0, z0), (x1, y0, z1)]),
             ((0, 0, 1), [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]),
             ((0, 0, -1), [(x1, y0, z0), (x0, y0, z0), (x0, y1, z0), (x1, y1, z0)])]
    positions, normals, indices = [], [], []
    for normal, quad_corners in faces:
        base = len(positions)
        positions += quad_corners
        normals += [normal] * 4
        indices += [base, base + 1, base + 2, base, base + 2, base + 3]
    return np.array(positions, np.float32), np.array(normals, np.float32), np.array(indices, np.uint16)


def model_meshes(parts) -> list:
    """The parts as glTF meshes (one each), coloured, not textured."""
    meshes = []
    for n, (lo, hi, finish, rgb) in enumerate(parts):
        positions, normals, indices = cuboid(lo, hi)
        roughness, metallic = FINISH[finish]
        linear = (np.asarray(rgb, float) / 255.0) ** 2.2             # glTF colours are linear
        meshes.append({"name": f"{finish} {n}", "quad": (positions, normals, None, indices),
                       "color": [*linear.round(4).tolist(), 1.0], "roughness": roughness, "metallic": metallic})
    return meshes


# ------------------------------------------------------------------- shell
def ceiling_surface(room, cell: float):
    from surface_fill import Surface

    lo = room.centre - room.half
    height = room.floor_z + room.shapes["room_level"]["height"]
    # u along x, v along -y: seen from below, a right-handed pair with the normal pointing down.
    origin = np.array([lo[0], lo[1] + 2 * room.half[1], height])
    return Surface("ceiling", origin, np.array([1.0, 0, 0]), np.array([0, -1.0, 0]), np.array([0, 0, -1.0]),
                   int(round(2 * room.half[0] / cell)), int(round(2 * room.half[1] / cell)), cell)


def hanging_mask(room, surface, dense) -> np.ndarray:
    """Where something hangs under the ceiling (a fan, a lamp): not the ceiling's own texture."""
    from scipy.ndimage import binary_closing, binary_dilation
    from surface_fill import raster

    m = room.metre
    depth = (dense - surface.origin) @ surface.normal
    below = dense[(depth > 0.06 * m) & (depth < FITTING_BAND_M * m)]
    mask = raster(surface, below, surface.rows, surface.cols, surface.cell) >= 2
    return binary_closing(binary_dilation(mask, iterations=6), iterations=3)


def texture(surface, photo, seen, blocked, log, plain=None) -> np.ndarray:
    """The surface's texture: the filmed photo where it was filmed cleanly,
    continued by inpainting everywhere else. A surface nobody filmed is
    `plain` (the colour of the walls that were filmed)."""
    from surface_fill import inpaint

    known = (seen > SEEN_WEIGHT) & ~blocked
    share = float(known.mean())
    if share < 0.03:
        colour = plain if plain is not None else np.array([200.0] * 3)
        log(f"  {surface.name}: hardly filmed ({share:.0%}); painted like the other walls {colour.astype(int).tolist()}")
        return np.broadcast_to(colour, photo.shape).copy()
    if share > 0.999:
        return photo
    log(f"  {surface.name}: {share:.0%} filmed cleanly, the rest continued from it")
    return inpaint(np.where(known[..., None], photo, 128), ~known, log)


def panel_surface(surface, box: dict, lift: float):
    """The photographed face moved onto its box's side, `lift` in front of it.
    surface_fill.object_surfaces puts a face at the depth where the splat has
    it, a few centimetres in or out of stage 3's box; the clean model fills
    the box exactly, so its panel belongs on the box's side, never behind it."""
    from dataclasses import replace

    axis = int(np.argmax(np.abs(surface.normal)))
    side = np.array(box["max" if surface.normal[axis] > 0 else "min"], float)[axis]
    origin = np.array(surface.origin, float)
    origin[axis] = side + np.sign(surface.normal[axis]) * lift
    return replace(surface, origin=origin)


def on_detected_object(tree, labels: np.ndarray, positions: np.ndarray, reach: float) -> np.ndarray:
    """Which `positions` sit on something stage 2's detector outlined (the
    dense cloud's points carry the object names Claude chose for the room;
    0 is unlabelled: walls, floor, ceiling, whatever nobody named). What is
    left of a splat once the furniture is cut out is mostly wall paint that
    training left hanging up to half a metre into the room, in colours too
    far from the wall's to be taken for it; by opacity and size it is like any
    other Gaussian, but no detected object is under it. A curtain, an air
    conditioner or a backpack has one."""
    if not len(positions):
        return np.zeros(0, bool)
    distance, index = tree.query(positions, k=THING_NEIGHBOURS, distance_upper_bound=reach, workers=-1)
    found = np.isfinite(distance)
    named = found & (labels[np.minimum(index, len(labels) - 1)] > 0)
    return (found.sum(axis=1) >= THING_MIN) & (named.sum(axis=1) >= THING_SHARE * found.sum(axis=1))


def despeckle(photo: np.ndarray, known: np.ndarray, size: int = 7) -> np.ndarray:
    """The photo with its speckles smoothed away: single filmed texels unlike
    everything round them, where one frame's depth test let a sliver of
    something nearer through. Only those texels change (to their
    neighbourhood's median); the rest keeps its sharpness."""
    from scipy.ndimage import median_filter, uniform_filter

    median = median_filter(photo, size=(size, size, 1))
    surrounded = uniform_filter(known.astype(float), size) > 0.6         # a median next to a hole is half black
    speckle = known & surrounded & (np.abs(photo - median).sum(axis=2) > SPECKLE_GAP)
    return np.where(speckle[..., None], median, photo)


def standing_in_front(room, face, points: np.ndarray) -> np.ndarray:
    """Texels of a furniture face that something else stands in front of.
    surface_fill.object_masks asks for two points in a 5 mm texel from 6 cm
    out, which suits deciding where a splat may be filled; but the dense
    cloud's depth is rough by several centimetres, so on a photograph it
    blanks the doors themselves (the walkthrough's wardrobe: 71% filmed down
    to 36%). Each frame's own depth test already keeps other things off the
    photograph; this only adds what clearly stands before the face."""
    from scipy.ndimage import binary_dilation, binary_opening
    from surface_fill import raster

    m = room.metre
    k = max(1, int(round(FRONT_CELL_M * m / face.cell)))
    rows, cols = -(-face.rows // k), -(-face.cols // k)
    depth = (points - face.origin) @ face.normal
    before = points[(depth > FRONT_CLEAR_M * m) & (depth < 0.6 * m)]
    coarse = raster(face, before, rows, cols, face.cell * k) >= FRONT_POINTS
    coarse = binary_dilation(binary_opening(coarse, iterations=1), iterations=2)
    return np.kron(coarse, np.ones((k, k), bool))[:face.rows, :face.cols]


def face_box(surface) -> str:
    """'object-B13-y-' -> 'B13'."""
    return surface.name.split("-")[1]


def quad(surface, frame: Frame):
    """The surface's rectangle in the viewer's frame: positions, normal, texture
    coordinates and triangles facing the room. Texel (row, col) of the texture
    sits at origin + col * u + row * v, so the image needs no flip."""
    width, height = surface.cols * surface.cell, surface.rows * surface.cell
    corners = np.array([surface.origin, surface.origin + width * surface.u,
                        surface.origin + height * surface.v,
                        surface.origin + width * surface.u + height * surface.v])
    positions = frame.scene_to_viewer(corners).astype(np.float32)
    normal = frame.direction(surface.normal)
    uvs = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], np.float32)
    facing = np.cross(positions[1] - positions[0], positions[2] - positions[0]) @ normal
    indices = np.array([0, 1, 2, 2, 1, 3] if facing > 0 else [0, 2, 1, 2, 3, 1], np.uint16)
    return positions, np.tile(normal.astype(np.float32), (4, 1)), uvs, indices


def write_glb(path: Path, meshes: list) -> None:
    """A binary glTF 2.0 file: one textured quad per surface, the images inside."""
    blob = bytearray()
    views, accessors, images, textures, materials, gltf_meshes, nodes = [], [], [], [], [], [], []

    def view(data: bytes, target: int | None = None) -> int:
        while len(blob) % 4:
            blob.append(0)
        entry = {"buffer": 0, "byteOffset": len(blob), "byteLength": len(data)}
        if target:
            entry["target"] = target
        blob.extend(data)
        views.append(entry)
        return len(views) - 1

    def accessor(array: np.ndarray, kind: str, component: int, target: int) -> int:
        entry = {"bufferView": view(array.tobytes(), target), "componentType": component,
                 "count": len(array), "type": kind}
        if kind == "VEC3" and component == 5126:
            entry["min"], entry["max"] = array.min(axis=0).tolist(), array.max(axis=0).tolist()
        accessors.append(entry)
        return len(accessors) - 1

    for mesh in meshes:
        positions, normals, uvs, indices = mesh["quad"]
        surface = {"metallicFactor": mesh.get("metallic", 0.0), "roughnessFactor": mesh["roughness"]}
        if mesh.get("jpeg"):
            images.append({"bufferView": view(mesh["jpeg"]), "mimeType": "image/jpeg", "name": mesh["name"]})
            textures.append({"source": len(images) - 1, "sampler": 0})
            surface["baseColorTexture"] = {"index": len(textures) - 1}
        else:
            surface["baseColorFactor"] = mesh["color"]
        materials.append({"name": mesh["name"], "doubleSided": False, "pbrMetallicRoughness": surface})
        attributes = {"POSITION": accessor(positions, "VEC3", 5126, 34962),
                      "NORMAL": accessor(normals, "VEC3", 5126, 34962)}
        if uvs is not None:
            attributes["TEXCOORD_0"] = accessor(uvs, "VEC2", 5126, 34962)
        gltf_meshes.append({"name": mesh["name"], "primitives": [{
            "attributes": attributes, "indices": accessor(indices, "SCALAR", 5123, 34963),
            "material": len(materials) - 1}]})
        nodes.append({"name": mesh["name"], "mesh": len(gltf_meshes) - 1})
    while len(blob) % 4:
        blob.append(0)
    document = {"asset": {"version": "2.0", "generator": "OasisSpaces mixed_scene.py"},
                "scene": 0, "scenes": [{"nodes": list(range(len(nodes)))}], "nodes": nodes,
                "meshes": gltf_meshes, "materials": materials,
                "accessors": accessors, "bufferViews": views, "buffers": [{"byteLength": len(blob)}]}
    if images:
        document.update(textures=textures, images=images,
                        samplers=[{"magFilter": 9729, "minFilter": 9987, "wrapS": 33071, "wrapT": 33071}])
    text = json.dumps(document, separators=(",", ":")).encode()
    text += b" " * ((-len(text)) % 4)
    with open(path, "wb") as f:
        f.write(struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(text) + 8 + len(blob)))
        f.write(struct.pack("<II", len(text), 0x4E4F534A))
        f.write(text)
        f.write(struct.pack("<II", len(blob), 0x004E4942))
        f.write(bytes(blob))


# ------------------------------------------------------------------ review
REVIEW_PROMPT = """You are the last check on a 3D scene built from one phone video of a room, before people open it.

WHAT THE SCENE IS. The room's flat surfaces (floor, walls, ceiling) have become textured meshes; each piece of furniture is a separate object that a person can click, drag across the floor, turn, hide, or swap between its scan and a clean model. So every decision below changes what someone can do in the room, not just how it looks. The bar is a room that looks smooth and polished from any angle and behaves sensibly when things are moved, not a faithful copy of a messy video.

WHAT YOU ARE GIVEN. Sheet 1 the surfaces, sheet 2 the pieces, sheet 3 the room with every piece named, {faces_sheet}then frames of the real room. Judge only from these; where they do not show you something, say so in the reason rather than guessing.

SHEET 1, SURFACES. One row per surface: on the left what the video actually filmed of it (magenta = never filmed, or hidden behind furniture), on the right the finished texture, where an inpainting model continued the filmed part into the magenta. Surfaces: {surfaces}.
For each surface choose "use":
- "keep": the finished texture is believable everywhere. Paint continues as paint, tiles as tiles; real things on the wall (a door, a window, a curtain, a switch, a board, a picture) may stay, and so may honest wear.
- "filmed": the filmed part is right but the continued part is not. The filmed part stays and the rest becomes plain paint, blended over a few centimetres. Choose this when the continuation invented objects, smeared furniture across the wall, or broke into seams and blotches.
- "plain": even the filmed part is wrong for a clean surface, so the whole surface becomes flat paint in the room's own colour. Choose this when furniture or clutter is printed flat onto the surface, when it is heavily blurred, or when almost nothing was filmed and the texture is a guess.
A surface is the background behind everything else, so a wrong texture is worse than a plain one: when you are undecided between "keep" and "filmed", choose "filmed"; between "filmed" and "plain", prefer the one that leaves no invented object visible.

SHEET 2, PIECES. One row per piece: the scan from where the video saw it best; the same scan from round the side and above, an angle nobody filmed; then the video frame. Pieces: {pieces}.
People move and turn these pieces and look at the room from above, so the second picture is what they will mostly see. A scan is a cloud of soft blobs that only looks right from where it was filmed. Every piece also has a clean simple model of its kind, built to its measured size in its own colours.
For each piece choose "use":
- "scan": the scan is recognisably that object and still clean from the unfilmed angle, with no haze, streaks or smears hanging off it.
- "model": it is a real piece of furniture, but from the unfilmed angle the scan is hazy, streaked, full of holes, or a shapeless cloud. The clean model is shown instead. Prefer this for flat-sided furniture (wardrobes, cabinets, desks, tables, chests) whenever the scan is not crisp: smooth and polished beats faithful-but-foggy.
- "drop": it is not a separate real object at all: part of a wall or floor, a duplicate of another piece, a fragment, or empty space. It disappears from the room, so do not use "drop" on a real object merely because its scan is poor; that is what "model" is for.
Be fair to soft things: a bed under bedding, a pile of clothes or a cushion has no flat sides, so a simple model would lose what it is. Keep those as "scan" if one can tell what they are.

SHEET 3, THE ROOM, AND WHAT RESTS ON WHAT. Sheet 3 shows the room from above and from where it was filmed, with each piece named where it stands. Use it with the frames to work out how the pieces depend on each other, and fill in "relations" for every piece in the list.
This decides what happens when a person edits the room. A piece that rests on another is carried by it: drag the bed and its pillows go along, hide the desk and what stood on it goes too. A piece that rests on the floor stays where it is when anything else moves.
For each piece give {{"on": "<id of the piece it rests on>" or "floor" or "wall", "why": "..."}}:
- "<id>": it sits on top of that piece, or is tucked into it, so it should travel with it. A pillow or cushion on a bed, a laptop or monitor on a desk, a lamp or books on a table, a basket on a shelf, a cushion on a chair.
- "floor": it stands on the floor on its own, even if it touches another piece. A bedside table beside a bed, a chair pushed under a desk, a basket on the floor next to a wardrobe: none of these should follow the other piece.
- "wall": it hangs on or is fixed to a wall and does not stand on the floor.
The same kind of object can rest on different things in different rooms, so decide from the pictures, not from what is usual: a pillow can be on a bed, on a chair or on the floor. Where two pieces overlap, ask which one would fall if the other were taken away. If you cannot see what it rests on, answer "floor", which changes nothing.
{faces_ask}
Reply as JSON: {{"surfaces": {{"<name>": {{"use": "keep|filmed|plain", "why": "..."}}, ...}}, "pieces": {{"<id>": {{"use": "scan|model|drop", "why": "..."}}, ...}}, "relations": {{"<id>": {{"on": "<id>|floor|wall", "why": "..."}}, ...}}, "faces": {{"<name>": {{"use": "keep|plain", "why": "..."}}, ...}}, "summary": "one sentence on how the room will look and what moves with what"}}"""

FACES_NOTE = ("sheet 4 the sides of those models that the video filmed flat-on, as photographs to put on the model "
              "(left: what was filmed, magenta = never seen or hidden; right: finished, the magenta continued by an "
              "inpainting model; faces: {faces}),")
FACES_ASK = """
For each face on sheet 4 choose:
- "keep": the finished photograph is believable as that side of the furniture (doors, drawers, handles, a table top), including where it was continued.
- "plain": it is not (other objects printed flat onto it, smears, a ghost of what stood in front, heavy blur): leave that side of the model in its plain colour.
"""


def claude_review(advisor, frames=(), log=print):
    """A review callback for build(): Claude's verdicts, or None when Claude is not reachable."""
    def review(surface_sheet: Path, piece_sheet: Path, surfaces: list, pieces: list,
               face_sheet: Path | None = None, faces: list = (), layout_sheet: Path | None = None):
        if advisor is None or not advisor.available:
            return None
        listed = ", ".join(f"{f['name']} = the {f['side']} of {f['piece']} ({f['filmed']:.0%} filmed)" for f in faces)
        with_faces = bool(faces and face_sheet)
        prompt = REVIEW_PROMPT.format(
            surfaces=", ".join(f"{s['name']} ({s['filmed']:.0%} filmed)" for s in surfaces),
            pieces=", ".join(f"{p['id']} = {p['label']} ({p['size']})" for p in pieces) or "none",
            faces_sheet=(FACES_NOTE.format(faces=listed) + " ") if with_faces else "",
            faces_ask=FACES_ASK if with_faces else "")
        images = ([surface_sheet] + ([piece_sheet] if pieces else [])
                  + ([layout_sheet] if layout_sheet else []) + ([face_sheet] if with_faces else [])
                  + list(frames))
        verdict = advisor.ask_json(prompt, images, max_tokens=3500)
        if verdict:
            log(f"  Claude: {verdict.get('summary', '')}")
        return verdict
    return review


def fit(image: np.ndarray, width: int, height: int) -> Image.Image:
    picture = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8))
    picture.thumbnail((width, height))
    return picture


def sheet(rows: list, out: Path, tile=(420, 300)) -> Path:
    """rows of (title, image, image, ...) as one labelled picture."""
    from PIL import ImageDraw, ImageFont

    font = ImageFont.load_default(size=18)
    w, h = tile
    across = max([len(row) - 1 for row in rows] + [1])
    page = Image.new("RGB", (across * (w + 10) + 10, max(1, len(rows)) * (h + 34) + 10), "white")
    draw = ImageDraw.Draw(page)
    for n, (title, *pictures) in enumerate(rows):
        y = 10 + n * (h + 34)
        draw.text((10, y), title, fill="black", font=font)
        for k, picture in enumerate(pictures):
            small = fit(picture, w, h)
            page.paste(small, (10 + k * (w + 10), y + 24))
    page.save(out)
    return out


def turned_view(room, view: dict, target: np.ndarray) -> list:
    """The view matrix of `view`'s camera carried TURNED_DEGREES round `target`
    (scene frame) and up to look down on it: an angle nobody filmed, which is
    how a piece is seen once the room is looked at from above or it is turned."""
    from splat_export import look_matrix, view_json

    V = np.array(view["viewMatrix"], float).reshape(4, 4).T
    position = room.to_scene((-V[:3, :3].T @ V[:3, 3])[None])[0]
    arm = position - target
    a = np.radians(TURNED_DEGREES)
    turn = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    reach = max(np.linalg.norm(arm[:2]), 1.2 * room.metre)
    flat = turn @ (arm[:2] / max(np.linalg.norm(arm[:2]), 1e-9)) * reach
    moved = np.array([target[0] + flat[0], target[1] + flat[1], target[2] + 0.9 * reach])
    p = room.to_splat(moved[None])[0]
    return view_json(look_matrix(p, room.to_splat(target[None])[0] - p, room.world[2]), p)["viewMatrix"]


def pixels_of(points_s, room, view, width: int, height: int, fov_y: float = 55.0):
    """Where scene-frame points land in a render made with `view` (the same
    convention as splat_render.render)."""
    V = np.array(view, float).reshape(4, 4).T
    cam = room.to_splat(np.asarray(points_s, float)) @ V[:3, :3].T + V[:3, 3]
    focal = height / 2 / np.tan(np.radians(fov_y) / 2)
    ahead = cam[:, 2] > 1e-6
    u = focal * cam[:, 0] / np.where(ahead, cam[:, 2], 1) + width / 2
    v = focal * cam[:, 1] / np.where(ahead, cam[:, 2], 1) + height / 2
    return u, v, ahead


def camera_at(room, position, target, up) -> list:
    """A viewer camera at `position` (scene frame) looking at `target`."""
    from splat_export import look_matrix, view_json

    eye = room.to_splat(np.asarray(position, float)[None])[0]
    return view_json(look_matrix(eye, room.to_splat(np.asarray(target, float)[None])[0] - eye, up),
                     eye)["viewMatrix"]


def layout_row(room, arr, scene, masks, log, size=(760, 570)) -> list:
    """For the review: the room with every piece named where it stands, from
    above and from the doorway, so that what rests on what can be judged. The
    pieces alone tell Claude what each one is; only this tells it where they
    are in relation to each other."""
    from PIL import ImageDraw, ImageFont
    from splat_export import VIEW_FOV_Y
    from splat_render import render

    m = room.metre
    boxes = room.shapes["boxes"]
    named = {ident: mask for ident, mask in masks.items() if ident.startswith("B") and mask.sum()}
    if not named:
        return []
    width, height = size
    centre = np.array([room.centre[0], room.centre[1], room.floor_z + 1.1 * m])
    # High enough that the whole room fits, looking straight down.
    reach = np.tan(np.radians(VIEW_FOV_Y / 2))
    up = max(room.half[1], room.half[0] * height / width) / reach * 1.2
    above = np.array([room.centre[0], room.centre[1], room.floor_z + up])
    views = [("from above", camera_at(room, above, centre, room.world[1]))]
    # From where the person stood, at eye height: what they see when they open the room.
    path = np.array(room.shapes.get("cameras") or [], float)
    if len(path):
        stood = np.median(path, axis=0)
        views.append(("from where it was filmed",
                      camera_at(room, np.array([stood[0], stood[1], room.floor_z + 1.5 * m]),
                                centre, room.world[2])))
    shown = np.zeros(len(arr), bool)
    for mask in named.values():
        shown |= mask
    shown |= masks.get("rest", np.zeros(len(arr), bool))
    font = ImageFont.load_default(size=19)
    images = []
    for name, view in views:
        picture = Image.fromarray(render(arr[shown], view, width, height))
        draw = ImageDraw.Draw(picture)
        for ident, mask in named.items():
            label = boxes[int(ident[1:])]
            middle = scene[mask].mean(axis=0)
            u, v, ahead = pixels_of(middle[None], room, view, width, height)
            if not ahead[0] or not (0 <= u[0] < width and 0 <= v[0] < height):
                continue
            text = f"{ident} {label.get('detected') or label.get('label')}"
            box = draw.textbbox((u[0], v[0]), text, font=font, anchor="mm")
            draw.rectangle([box[0] - 4, box[1] - 3, box[2] + 4, box[3] + 3], fill=(0, 0, 0))
            draw.text((u[0], v[0]), text, fill=(255, 210, 90), font=font, anchor="mm")
        images.append(picture)
    log(f"  layout: the room with {len(named)} named piece(s), {len(images)} view(s)")
    return [("the room with every piece named   (" + " | ".join(n for n, _ in views) + ")", *images)]


def piece_rows(room, arr, scene, masks, log) -> list:
    """For the review: each movable piece alone, from the camera that saw it
    best and from an angle nobody filmed, beside the best frame."""
    from object_frames import best_frames
    from splat_edit import camera_looking_at
    from splat_render import render

    rows = []
    boxes = room.shapes["boxes"]
    for ident, mask in masks.items():
        if not ident.startswith("B") or mask.sum() == 0:
            continue
        index = int(ident[1:])
        box = boxes[index]
        target = (np.array(box["min"]) + np.array(box["max"])) / 2
        try:
            picks = best_frames(room.space, {index: box}, per_box=1)
            view = camera_looking_at(room, target, index)
        except Exception as error:                       # a space moved here without its camera model
            log(f"  {ident}: no review view ({error})")
            continue
        if not view:
            continue
        scan = render(arr[mask], view["viewMatrix"], 480, 360, view.get("fovY", 55.0))
        turned = render(arr[mask], turned_view(room, view, target), 480, 360, view.get("fovY", 55.0))
        frame_name = picks[index][0]["frame"] if picks.get(index) else None
        photo = (np.asarray(Image.open(room.space / "workspace" / "images" / frame_name).convert("RGB"))
                 if frame_name else np.full((360, 480, 3), 230, np.uint8))
        rows.append((f"{ident}  {box.get('detected') or box.get('label')}   (scan as filmed | the scan from an unfilmed angle "
                     f"| the video, {frame_name})", scan, turned, photo))
    return rows


def mark_stacked(pieces: list, relations: dict | None = None, log=print) -> None:
    """What rests on what, as "on": the id of the piece that carries it. A
    piece that rests on another travels with it in the viewer, so dragging a
    bed takes its pillows and hiding a desk hides what stood on it.

    Claude decides this from the room (the "relations" of its review), because
    it depends on the room and not on the kind of thing: a pillow can be on a
    bed, on a chair or on the floor, and a chair tucked under a desk still
    stands on the floor. Geometry fills in only for pieces Claude did not
    answer for, and nothing may end up carrying itself in a circle."""
    movable = [p for p in pieces if p.get("movable")]
    here = {p["id"] for p in movable}
    answered = set()
    for p in movable:
        said = (relations or {}).get(p["id"]) or {}
        on = said.get("on")
        if not on:
            continue
        answered.add(p["id"])                       # "floor" and "wall" are answers too
        if on not in here or on == p["id"]:
            continue
        if carries(relations, here, on, p["id"]):
            log(f"  {p['id']}: cannot rest on {on}, which rests on it; left on the floor")
            continue
        p["on"], p["onWhy"] = on, said.get("why")
        log(f"  {p['id']} {p['label']}: rests on {on} ({said.get('why') or ''})")

    volume = lambda p: float(np.prod(np.array(p["box"]["max"]) - np.array(p["box"]["min"])))
    for p in [q for q in movable if q["id"] not in answered]:
        hosts = []
        for q in movable:
            if p is q or volume(p) >= volume(q) or p["box"]["min"][1] <= 0.2:
                continue
            a, qa = np.array(p["anchor"]), np.array(q["anchor"])
            lo, hi = qa[[0, 2]] + np.array(q["box"]["min"])[[0, 2]], qa[[0, 2]] + np.array(q["box"]["max"])[[0, 2]]
            if np.all(a[[0, 2]] >= lo) and np.all(a[[0, 2]] <= hi):
                hosts.append(q)
        if hosts:
            p["on"] = min(hosts, key=volume)["id"]          # the smallest thing it stands on


def carries(relations: dict | None, here: set, parent: str, child: str) -> bool:
    """Does `parent` already rest on `child`, directly or through others?"""
    seen = set()
    while parent in here and parent not in seen:
        seen.add(parent)
        if parent == child:
            return True
        parent = ((relations or {}).get(parent) or {}).get("on")
    return parent == child


# ------------------------------------------------------------------- build
def build(space: Path, cell_m: float, splat_name: str, log=print, review=None) -> Path:
    from dataclasses import replace

    import surface_fill
    from pointcloud import load_ply
    from scipy.ndimage import gaussian_filter
    from scipy.spatial import cKDTree
    from splat_edit import Room
    from splat_tools import read_splat
    from surface_fill import blob_arrays, floor_masks, object_surfaces, photograph, wall_masks, wall_surfaces

    if not surface_fill.LAMA_PATH.exists():
        sys.exit(f"LaMa weights not found at {surface_fill.LAMA_PATH} (see the README's fill-room)")
    surface_fill.PHOTO_CELL_M = cell_m
    room = Room(space)
    m = room.metre
    out = space / "scene"
    for folder in ("pieces", "textures", "models"):
        (out / folder).mkdir(parents=True, exist_ok=True)

    arr, _ = read_splat(space / splat_name)
    log(f"{space.name}: {len(arr):,} Gaussians in {splat_name}; the room is "
        f"{2 * room.half[0] / m:.2f} x {2 * room.half[1] / m:.2f} m")
    cloud = load_ply(space / "cloud-dense.ply")
    dense = room.to_scene(cloud.points.astype(np.float64))
    floor, floor_blocked, arr, _ = floor_masks(room, arr, dense, log)      # also clears haze over the floor
    floor_height = float(floor.origin[2])
    scene, colours, alpha, scale = blob_arrays(room, arr)
    # Haze: training leaves soft, oversized Gaussians hanging in the air along plain walls.
    # In a splat they blur into the wall; on a moved piece they would come along as a smear.
    sampled = np.random.default_rng(0).choice(len(dense), min(len(dense), 2_000_000), replace=False)
    measured = cKDTree(dense[sampled])
    support = measured.query(scene, workers=-1)[0]
    haze = (support > HAZE_SUPPORT_M * m) & ((alpha < HAZE_ALPHA) | (scale > HAZE_SCALE_M * m))
    log(f"  {int(haze.sum()):,} of {len(arr):,} Gaussians are haze (soft, with no measured surface near)")
    arr, scene, colours, alpha, scale = arr[~haze], scene[~haze], colours[~haze], alpha[~haze], scale[~haze]
    walls = wall_surfaces(room, scene, alpha, log)
    ceiling = ceiling_surface(room, floor.cell)
    frame = Frame(room, floor_height)

    log("cutting the splat into pieces")
    detected = None
    if cloud.labels is not None and (cloud.labels > 0).any():        # a cloud from before stage 2 named its points has none
        detected = lambda positions: on_detected_object(measured, cloud.labels[sampled], positions, THING_REACH_M * m)
    masks = cut_pieces(room, arr, scene, colours, alpha, floor_height, walls, log, detected)

    log("photographing the surfaces from the frames")
    front = np.vstack([dense, scene[alpha > 0.5]])
    surfaces = [floor, *walls, ceiling]
    blocked = [floor_blocked]
    for wall in walls:
        standing, hidden, _ = wall_masks(room, wall, front, walls)
        blocked.append(standing | hidden)
    blocked.append(hanging_mask(room, ceiling, front))
    solid = np.stack([arr["x"], arr["y"], arr["z"]], axis=1)[alpha > 0.3].astype(np.float64)
    # The furniture's flat sides that face the room (a wardrobe's doors, a table's top) are
    # photographed in the same pass: a scan is haze from any angle it was not filmed from, a
    # photograph on the clean model's side is sharp from all of them.
    # They get their own pass: finer texels from the full-size frames, the best frames deciding,
    # because a door's pattern smears when every frame is averaged in; paint does not.
    finer = max(1, int(round(cell_m / PANEL_CELL_M)))
    faces = [replace(f, cols=f.cols * finer, rows=f.rows * finer, cell=f.cell / finer)
             for f in object_surfaces(room, scene, alpha, log) if masks.get(face_box(f), np.zeros(1)).sum()]
    face_blocked = [standing_in_front(room, face, front) for face in faces]
    photos = photograph(space, room, surfaces, log, occluders=solid, frame_step=1)
    face_photos = (photograph(space, room, faces, log, occluders=solid, frame_step=1,
                              sharpness=PANEL_SHARPNESS, shrink=1) if faces else [])

    # The paint of the walls that were filmed, for a wall that was not.
    knowns = [(seen > SEEN_WEIGHT) & ~block for block, (_p, seen, _t) in zip(blocked, photos)]
    paints = [np.median(photo[known], axis=0) for surface, known, (photo, _s, _t) in zip(surfaces, knowns, photos)
              if surface.name.startswith("wall") and known.mean() > 0.15]
    paint = np.mean(paints, axis=0) if paints else None
    finished = [texture(surface, photo, seen, block, log, plain=paint)
                for surface, block, (photo, seen, _t) in zip(surfaces, blocked, photos)]

    # ---- the review: Claude (or whoever `review` is) sees what was made, and decides
    upright = lambda surface, image: image[::-1] if surface.name.startswith("wall") else image
    rows = []
    for surface, known, (photo, _s, _t), final in zip(surfaces, knowns, photos, finished):
        filmed = np.where(known[..., None], photo, np.array([255.0, 0, 200]))
        rows.append((f"{surface.name}   (filmed {known.mean():.0%} | finished)", upright(surface, filmed), upright(surface, final)))
    surface_sheet = sheet(rows, out / "review-surfaces.png")
    piece_sheet = sheet(piece_rows(room, arr, scene, masks, log), out / "review-pieces.png", tile=(480, 360))
    layout = layout_row(room, arr, scene, masks, log)
    layout_sheet = sheet(layout, out / "review-layout.png", tile=(760, 570)) if layout else None
    boxes = room.shapes["boxes"]
    # The faces filmed well enough, finished like the walls: what stood in front is continued.
    panels, rows = {}, []
    for face, block, (photo, seen, _t) in zip(faces, face_blocked, face_photos):
        known = (seen > SEEN_WEIGHT) & ~block
        photo = despeckle(photo, known)
        name = face.name.removeprefix("object-")
        if known.mean() < PANEL_MIN_FILMED:
            log(f"  {name}: only {known.mean():.0%} of it filmed; that side of the model stays plain")
            continue
        final = texture(face, photo, seen, block, log)
        stand = (lambda image: image[::-1]) if face.normal[2] == 0 else (lambda image: image)
        rows.append((f"{name}   (filmed {known.mean():.0%} | finished)",
                     stand(np.where(known[..., None], photo, np.array([255.0, 0, 200]))), stand(final)))
        box = boxes[int(face_box(face)[1:])]
        panels[name] = {"face": face, "image": final, "filmed": float(known.mean()), "piece": face_box(face),
                        "asked": {"name": name, "filmed": float(known.mean()),
                                  "side": "top" if face.normal[2] else "side facing the room",
                                  "piece": f"{face_box(face)} {box.get('detected') or box.get('label')}"}}
    face_sheet = sheet(rows, out / "review-faces.png") if rows else None
    size_of = lambda b: " x ".join(f"{v:.1f}" for v in (np.array(b["max"]) - np.array(b["min"])) / m) + " m"
    asked = [{"id": ident, "label": boxes[int(ident[1:])].get("detected") or boxes[int(ident[1:])].get("label"),
              "size": size_of(boxes[int(ident[1:])])} for ident, mask in masks.items() if ident.startswith("B") and mask.sum()]
    verdict = review(surface_sheet, piece_sheet,
                     [{"name": s.name, "filmed": float(k.mean())} for s, k in zip(surfaces, knowns)], asked,
                     face_sheet, [panel["asked"] for panel in panels.values()], layout_sheet) if review else None
    surface_use = {name: (v or {}).get("use", "keep") for name, v in ((verdict or {}).get("surfaces") or {}).items()}
    piece_use = {ident: (v or {}).get("use", "scan") for ident, v in ((verdict or {}).get("pieces") or {}).items()}
    face_use = {name: (v or {}).get("use", "keep") for name, v in ((verdict or {}).get("faces") or {}).items()}
    for name in [n for n in panels if face_use.get(n, "keep") != "keep"]:
        log(f"  {name}: plain ({((verdict or {}).get('faces') or {}).get(name, {}).get('why', '')})")
        del panels[name]

    meshes = []
    for surface, known, (photo, _s, _t), final in zip(surfaces, knowns, photos, finished):
        use = surface_use.get(surface.name, "keep")
        flat = np.median(photo[known], axis=0) if known.mean() > 0.05 else (paint if paint is not None else np.array([200.0] * 3))
        if use == "plain":
            image = np.broadcast_to(flat, final.shape).copy()
        elif use == "filmed":
            # The filmed part, fading into plain paint over a few centimetres.
            weight = np.clip(gaussian_filter(known.astype(float), 0.03 / cell_m) * 2 - 1, 0, 1)[..., None]
            image = weight * photo + (1 - weight) * flat
        else:
            image = final
        if use != "keep":
            log(f"  {surface.name}: {use} ({((verdict or {}).get('surfaces') or {}).get(surface.name, {}).get('why', '')})")
        image = np.clip(image, 0, 255).astype(np.uint8)
        jpeg = io.BytesIO()
        Image.fromarray(image).save(jpeg, "JPEG", quality=92)
        (out / "textures" / f"{surface.name}.jpg").write_bytes(jpeg.getvalue())
        kind = "wall" if surface.name.startswith("wall") else surface.name
        meshes.append({"name": surface.name, "quad": quad(surface, frame), "jpeg": jpeg.getvalue(),
                       "roughness": ROUGHNESS[kind]})
    write_glb(out / "shell.glb", meshes)
    log(f"wrote shell.glb: {len(meshes)} surfaces, {(out / 'shell.glb').stat().st_size / 1e6:.1f} MB")

    pieces = []
    for old in (list((out / "pieces").glob("*.splat")) + list((out / "models").glob("*.glb"))
                + list((out / "textures").glob("B*.jpg"))):
        old.unlink()
    for ident, mask in masks.items():
        if mask.sum() == 0:
            continue
        use = piece_use.get(ident, "scan")
        if use == "drop":
            log(f"  {ident}: dropped ({((verdict or {}).get('pieces') or {}).get(ident, {}).get('why', '')})")
            continue
        points = frame.scene_to_viewer(scene[mask])
        lo, hi = np.percentile(points, 1, axis=0), np.percentile(points, 99, axis=0)
        entry = {"id": ident, "file": f"pieces/{ident}.splat", "count": int(mask.sum())}
        if ident.startswith("B"):
            box = boxes[int(ident[1:])]
            corners = frame.scene_to_viewer(np.array([box["min"], box["max"]]))
            blo, bhi = corners.min(axis=0), corners.max(axis=0)
            bhi[1] = max(bhi[1], hi[1])                   # a headboard taller than the measured box
            anchor = np.array([(blo[0] + bhi[0]) / 2, 0.0, (blo[2] + bhi[2]) / 2])
            label = box.get("detected") or box.get("label")
            parts = model_parts(room, box, label, colours[mask], scene[mask][:, 2], frame, anchor)
            photographed = []
            for name, panel in panels.items():
                if panel["piece"] != ident:
                    continue
                jpeg = io.BytesIO()
                Image.fromarray(np.clip(panel["image"], 0, 255).astype(np.uint8)).save(jpeg, "JPEG", quality=92)
                (out / "textures" / f"{name}.jpg").write_bytes(jpeg.getvalue())
                positions, normals, uvs, indices = quad(panel_surface(panel["face"], box, PANEL_LIFT_M * m), frame)
                photographed.append({"name": f"photo {name}", "jpeg": jpeg.getvalue(), "roughness": 0.8,
                                     "quad": (positions - anchor.astype(np.float32), normals, uvs, indices)})
            write_glb(out / "models" / f"{ident}.glb", model_meshes(parts) + photographed)
            if photographed:
                entry["photographed"] = [mesh["name"].removeprefix("photo ") for mesh in photographed]
                log(f"  {ident} {label}: its model carries {len(photographed)} photographed side(s)")
            entry.update(label=label, movable=True, model=f"models/{ident}.glb", modelKind=model_kind(label),
                         show="model" if use == "model" else "scan",
                         box={"min": (blo - anchor).round(3).tolist(), "max": (bhi - anchor).round(3).tolist()})
            why = ((verdict or {}).get("pieces") or {}).get(ident, {}).get("why")
            if why:
                entry["why"] = why
            if use == "model":
                log(f"  {ident} {label}: shown as its clean model ({why or ''})")
        else:
            anchor = np.zeros(3)
            entry.update(label={"rest": "everything else", "ceiling-fittings": "ceiling fittings"}[ident],
                         movable=False, box={"min": lo.round(3).tolist(), "max": hi.round(3).tolist()})
        entry["anchor"] = anchor.round(3).tolist()
        write_piece(out / entry["file"], arr[mask], frame, scene[mask], anchor)
        pieces.append(entry)
    mark_stacked(pieces, (verdict or {}).get("relations"), log)

    # Where the person stood while filming, for a view from inside: the middle of the walked
    # path, at eye height, looking at the room's centre.
    path = np.array(room.shapes.get("cameras") or [[room.centre[0], room.centre[1]]], float)
    stood = np.median(path, axis=0)
    eye = frame.scene_to_viewer(np.array([stood[0], stood[1], floor_height + 1.5 * m]))
    manifest = {
        "space": space.name, "units": "metres", "up": "y",
        # viewer = ((solve @ world.T - origin) / unitsPerMetre) @ toViewer.T
        "solveFrame": {"origin": frame.origin.round(6).tolist(), "unitsPerMetre": round(float(m), 6),
                       "world": np.asarray(room.world).round(9).tolist(), "toViewer": TO_VIEWER.tolist()},
        "inside": {"position": eye.round(3).tolist(), "target": [0.0, 1.1, 0.0]},
        "room": {"width": round(2 * room.half[0] / m, 3), "depth": round(2 * room.half[1] / m, 3),
                 "height": round(room.shapes["room_level"]["height"] / m, 3)},
        "shell": "shell.glb", "source": splat_name, "texelMetres": cell_m, "pieces": pieces,
        "reviewedBy": "claude" if verdict else None, "summary": (verdict or {}).get("summary"),
    }
    (out / "scene.json").write_text(json.dumps(manifest, indent=1) + "\n")
    (out / "review.json").write_text(json.dumps({"verdict": verdict, "surfaces": surface_use, "pieces": piece_use,
                                                 "faces": {name: "keep" for name in panels}}, indent=1) + "\n")
    log(f"wrote {out / 'scene.json'}: {len(pieces)} pieces "
        f"({', '.join(p['label'] + (' [model]' if p.get('show') == 'model' else '') for p in pieces if p['movable'])})")
    log(f"view: http://localhost:8734/scene-viewer/index.html?scene=../spaces/{space.name}/scene/scene.json")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("space", type=Path)
    parser.add_argument("--cell", type=float, default=0.005, help="metres per texel of the shell's textures")
    parser.add_argument("--splat", default="splat.ply", help="the trained splat to cut up")
    parser.add_argument("--claude", action="store_true", help="let Claude review the textures and the pieces")
    args = parser.parse_args()
    review = None
    if args.claude:
        from advisor import Advisor

        review = claude_review(Advisor())
    build(args.space.resolve(), args.cell, args.splat, review=review)


if __name__ == "__main__":
    main()
