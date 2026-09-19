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
HAZE_SUPPORT_M = 0.08       # a soft Gaussian with no dense-cloud surface this close is haze, not a thing
HAZE_ALPHA = 0.6            # ...soft meaning fainter than this,
HAZE_SCALE_M = 0.05         # ...or larger than this
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


def cut_pieces(room, arr, scene, colours, alpha, floor_height, walls, log):
    """Which Gaussians are which: {piece id: mask}. The room's surfaces are
    found first, so a piece never takes the wall it stands against with it:
    a Gaussian in a wall's band counts as the wall when it is the wall's
    colour (a headboard or a shelf on the wall is not). Then the objects
    (stage 3's built boxes, largest first, so a pillow does not take the bed's
    blanket), then what is left."""
    from splat_edit import ABOVE_COLOUR_GAP, KEEP_OTHER_M, object_blobs, typical

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
    taken = np.zeros(len(arr), bool)
    pieces = {}
    for i, box in boxes:
        protected = shell.copy()
        for j, other in boxes:
            if j != i:
                lo = np.array(other["min"]) + KEEP_OTHER_M * m
                hi = np.array(other["max"]) - KEEP_OTHER_M * m
                protected |= np.all((scene >= lo) & (scene <= hi), axis=1)
        mask, _, _ = object_blobs(room, scene, colours, box, protected | taken, taken | shell)
        mask &= ~taken & ~shell
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
REVIEW_PROMPT = """You are checking a 3D scene made from a phone video of a room, before people see it. The room's flat surfaces (floor, walls, ceiling) became textured meshes; each piece of furniture is kept as it was scanned, as a movable piece. Two sheets are attached, then frames of the real room.

Sheet 1, surfaces. One row per surface: on the left what the video actually filmed of it (magenta = never filmed, or hidden behind furniture), on the right the finished texture, where an inpainting model continued the filmed part into the magenta. Surfaces: {surfaces}.
For each surface choose:
- "keep": the finished texture is believable everywhere (paint continues as paint, tiles as tiles; real things on the wall such as a door, a window, a curtain, a switch or a board may stay).
- "filmed": the filmed part is good but the continued part is not (smears, ghosts of furniture, invented objects, blotches): keep the filmed part and paint the rest plain.
- "plain": even the filmed part is wrong for a clean surface (furniture or clutter printed flat onto it, heavy blur, patchwork): paint the whole surface plain.

Sheet 2, pieces. One row per piece: on the left the scanned piece alone, seen from where the video saw it best; on the right that frame of the video. Pieces: {pieces}.
For each piece choose:
- "scan": the scan is recognisably that object; show it as filmed.
- "model": it is a real piece of furniture but the scan is too broken to show (mostly holes, smears, or a shapeless cloud); show a clean simple model of it instead.
- "drop": it is not a separate real object (part of a wall, a duplicate of another piece, empty space).
Be fair to soft scans: furniture covered in clothes or bedding is still "scan" if one can tell what it is.

Reply as JSON: {{"surfaces": {{"<name>": {{"use": "keep|filmed|plain", "why": "..."}}, ...}}, "pieces": {{"<id>": {{"use": "scan|model|drop", "why": "..."}}, ...}}, "summary": "one sentence on how the scene will look"}}"""


def claude_review(advisor, frames=(), log=print):
    """A review callback for build(): Claude's verdicts, or None when Claude is not reachable."""
    def review(surface_sheet: Path, piece_sheet: Path, surfaces: list, pieces: list):
        if advisor is None or not advisor.available:
            return None
        prompt = REVIEW_PROMPT.format(
            surfaces=", ".join(f"{s['name']} ({s['filmed']:.0%} filmed)" for s in surfaces),
            pieces=", ".join(f"{p['id']} = {p['label']} ({p['size']})" for p in pieces) or "none")
        images = [surface_sheet] + ([piece_sheet] if pieces else []) + list(frames)
        verdict = advisor.ask_json(prompt, images, max_tokens=2500)
        if verdict:
            log(f"  Claude: {verdict.get('summary', '')}")
        return verdict
    return review


def fit(image: np.ndarray, width: int, height: int) -> Image.Image:
    picture = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8))
    picture.thumbnail((width, height))
    return picture


def sheet(rows: list, out: Path, tile=(420, 300)) -> Path:
    """rows of (title, left image, right image) as one labelled picture."""
    from PIL import ImageDraw, ImageFont

    font = ImageFont.load_default(size=18)
    w, h = tile
    page = Image.new("RGB", (2 * w + 30, max(1, len(rows)) * (h + 34) + 10), "white")
    draw = ImageDraw.Draw(page)
    for n, (title, left, right) in enumerate(rows):
        y = 10 + n * (h + 34)
        draw.text((10, y), title, fill="black", font=font)
        for k, picture in enumerate((left, right)):
            small = fit(picture, w, h)
            page.paste(small, (10 + k * (w + 10), y + 24))
    page.save(out)
    return out


def piece_rows(room, arr, scene, masks, log) -> list:
    """For the review: each movable piece alone, from the camera that saw it best, beside that frame."""
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
        frame_name = picks[index][0]["frame"] if picks.get(index) else None
        photo = (np.asarray(Image.open(room.space / "workspace" / "images" / frame_name).convert("RGB"))
                 if frame_name else np.full((360, 480, 3), 230, np.uint8))
        rows.append((f"{ident}  {box.get('detected') or box.get('label')}   (scan | the video, {frame_name})", scan, photo))
    return rows


# ------------------------------------------------------------------- build
def build(space: Path, cell_m: float, splat_name: str, log=print, review=None) -> Path:
    import surface_fill
    from pointcloud import load_ply
    from scipy.ndimage import gaussian_filter
    from scipy.spatial import cKDTree
    from splat_edit import Room
    from splat_tools import read_splat
    from surface_fill import blob_arrays, floor_masks, photograph, wall_masks, wall_surfaces

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
    dense = room.to_scene(load_ply(space / "cloud-dense.ply").points.astype(np.float64))
    floor, floor_blocked, arr, _ = floor_masks(room, arr, dense, log)      # also clears haze over the floor
    floor_height = float(floor.origin[2])
    scene, colours, alpha, scale = blob_arrays(room, arr)
    # Haze: training leaves soft, oversized Gaussians hanging in the air along plain walls.
    # In a splat they blur into the wall; on a moved piece they would come along as a smear.
    sample = dense[np.random.default_rng(0).choice(len(dense), min(len(dense), 2_000_000), replace=False)]
    support = cKDTree(sample).query(scene, workers=-1)[0]
    haze = (support > HAZE_SUPPORT_M * m) & ((alpha < HAZE_ALPHA) | (scale > HAZE_SCALE_M * m))
    log(f"  {int(haze.sum()):,} of {len(arr):,} Gaussians are haze (soft, with no measured surface near)")
    arr, scene, colours, alpha, scale = arr[~haze], scene[~haze], colours[~haze], alpha[~haze], scale[~haze]
    walls = wall_surfaces(room, scene, alpha, log)
    ceiling = ceiling_surface(room, floor.cell)
    frame = Frame(room, floor_height)

    log("cutting the splat into pieces")
    masks = cut_pieces(room, arr, scene, colours, alpha, floor_height, walls, log)

    log("photographing the surfaces from the frames")
    front = np.vstack([dense, scene[alpha > 0.5]])
    surfaces = [floor, *walls, ceiling]
    blocked = [floor_blocked]
    for wall in walls:
        standing, hidden, _ = wall_masks(room, wall, front, walls)
        blocked.append(standing | hidden)
    blocked.append(hanging_mask(room, ceiling, front))
    solid = np.stack([arr["x"], arr["y"], arr["z"]], axis=1)[alpha > 0.3].astype(np.float64)
    photos = photograph(space, room, surfaces, log, occluders=solid, frame_step=1)

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
    boxes = room.shapes["boxes"]
    size_of = lambda b: " x ".join(f"{v:.1f}" for v in (np.array(b["max"]) - np.array(b["min"])) / m) + " m"
    asked = [{"id": ident, "label": boxes[int(ident[1:])].get("detected") or boxes[int(ident[1:])].get("label"),
              "size": size_of(boxes[int(ident[1:])])} for ident, mask in masks.items() if ident.startswith("B") and mask.sum()]
    verdict = review(surface_sheet, piece_sheet,
                     [{"name": s.name, "filmed": float(k.mean())} for s, k in zip(surfaces, knowns)], asked) if review else None
    surface_use = {name: (v or {}).get("use", "keep") for name, v in ((verdict or {}).get("surfaces") or {}).items()}
    piece_use = {ident: (v or {}).get("use", "scan") for ident, v in ((verdict or {}).get("pieces") or {}).items()}

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
    for old in list((out / "pieces").glob("*.splat")) + list((out / "models").glob("*.glb")):
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
            write_glb(out / "models" / f"{ident}.glb", model_meshes(parts))
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
    # A thing standing on another moves with it (pillows on the bed).
    movable = [p for p in pieces if p["movable"]]
    for p in movable:
        for q in movable:
            if p is q:
                continue
            a, qa = np.array(p["anchor"]), np.array(q["anchor"])
            within = np.all(a[[0, 2]] >= qa[[0, 2]] + np.array(q["box"]["min"])[[0, 2]]) and \
                np.all(a[[0, 2]] <= qa[[0, 2]] + np.array(q["box"]["max"])[[0, 2]])
            smaller = np.prod(np.array(p["box"]["max"]) - p["box"]["min"]) < np.prod(np.array(q["box"]["max"]) - q["box"]["min"])
            if within and smaller and p["box"]["min"][1] > 0.2:
                p["on"] = q["id"]

    # Where the person stood while filming, for a view from inside: the middle of the walked
    # path, at eye height, looking at the room's centre.
    path = np.array(room.shapes.get("cameras") or [[room.centre[0], room.centre[1]]], float)
    stood = np.median(path, axis=0)
    eye = frame.scene_to_viewer(np.array([stood[0], stood[1], floor_height + 1.5 * m]))
    manifest = {
        "space": space.name, "units": "metres", "up": "y",
        "inside": {"position": eye.round(3).tolist(), "target": [0.0, 1.1, 0.0]},
        "room": {"width": round(2 * room.half[0] / m, 3), "depth": round(2 * room.half[1] / m, 3),
                 "height": round(room.shapes["room_level"]["height"] / m, 3)},
        "shell": "shell.glb", "source": splat_name, "texelMetres": cell_m, "pieces": pieces,
        "reviewedBy": "claude" if verdict else None, "summary": (verdict or {}).get("summary"),
    }
    (out / "scene.json").write_text(json.dumps(manifest, indent=1) + "\n")
    (out / "review.json").write_text(json.dumps({"verdict": verdict, "surfaces": surface_use, "pieces": piece_use}, indent=1) + "\n")
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
