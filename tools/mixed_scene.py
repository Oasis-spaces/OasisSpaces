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
  scene.json      what is where: each piece's label, box and anchor, in metres,
                  y up, the room's centre on the floor as origin (three.js
                  conventions), so a viewer can select, move, turn and hide
                  pieces. scene-viewer/ is that viewer.

    python3 tools/mixed_scene.py spaces/<name> [--cell 0.005] [--splat splat.ply]

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
        images.append({"bufferView": view(mesh["jpeg"]), "mimeType": "image/jpeg", "name": mesh["name"]})
        textures.append({"source": len(images) - 1, "sampler": 0})
        materials.append({"name": mesh["name"], "doubleSided": False,
                          "pbrMetallicRoughness": {"baseColorTexture": {"index": len(textures) - 1},
                                                   "metallicFactor": 0.0, "roughnessFactor": mesh["roughness"]}})
        gltf_meshes.append({"name": mesh["name"], "primitives": [{
            "attributes": {"POSITION": accessor(positions, "VEC3", 5126, 34962),
                           "NORMAL": accessor(normals, "VEC3", 5126, 34962),
                           "TEXCOORD_0": accessor(uvs, "VEC2", 5126, 34962)},
            "indices": accessor(indices, "SCALAR", 5123, 34963), "material": len(materials) - 1}]})
        nodes.append({"name": mesh["name"], "mesh": len(gltf_meshes) - 1})
    while len(blob) % 4:
        blob.append(0)
    document = {"asset": {"version": "2.0", "generator": "OasisSpaces mixed_scene.py"},
                "scene": 0, "scenes": [{"nodes": list(range(len(nodes)))}], "nodes": nodes,
                "meshes": gltf_meshes, "materials": materials, "textures": textures, "images": images,
                "samplers": [{"magFilter": 9729, "minFilter": 9987, "wrapS": 33071, "wrapT": 33071}],
                "accessors": accessors, "bufferViews": views, "buffers": [{"byteLength": len(blob)}]}
    text = json.dumps(document, separators=(",", ":")).encode()
    text += b" " * ((-len(text)) % 4)
    with open(path, "wb") as f:
        f.write(struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(text) + 8 + len(blob)))
        f.write(struct.pack("<II", len(text), 0x4E4F534A))
        f.write(text)
        f.write(struct.pack("<II", len(blob), 0x004E4942))
        f.write(bytes(blob))


# ------------------------------------------------------------------- build
def build(space: Path, cell_m: float, splat_name: str, log=print) -> Path:
    import surface_fill
    from pointcloud import load_ply
    from splat_edit import Room, splat_colours
    from splat_tools import read_splat
    from surface_fill import blob_arrays, floor_masks, photograph, wall_masks, wall_surfaces

    if not surface_fill.LAMA_PATH.exists():
        sys.exit(f"LaMa weights not found at {surface_fill.LAMA_PATH} (see the README's fill-room)")
    surface_fill.PHOTO_CELL_M = cell_m
    room = Room(space)
    m = room.metre
    out = space / "scene"
    (out / "pieces").mkdir(parents=True, exist_ok=True)
    (out / "textures").mkdir(exist_ok=True)

    arr, _ = read_splat(space / splat_name)
    log(f"{space.name}: {len(arr):,} Gaussians in {splat_name}; the room is "
        f"{2 * room.half[0] / m:.2f} x {2 * room.half[1] / m:.2f} m")
    dense = room.to_scene(load_ply(space / "cloud-dense.ply").points.astype(np.float64))
    floor, floor_blocked, arr, _ = floor_masks(room, arr, dense, log)      # also clears haze over the floor
    floor_height = float(floor.origin[2])
    scene, colours, alpha, scale = blob_arrays(room, arr)
    # Haze: training leaves soft, oversized Gaussians hanging in the air along plain walls.
    # In a splat they blur into the wall; on a moved piece they would come along as a smear.
    from scipy.spatial import cKDTree

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
    paints = [np.median(photo[(seen > SEEN_WEIGHT) & ~block], axis=0)
              for surface, block, (photo, seen, _t) in zip(surfaces, blocked, photos)
              if surface.name.startswith("wall") and ((seen > SEEN_WEIGHT) & ~block).mean() > 0.15]
    paint = np.mean(paints, axis=0) if paints else None
    meshes = []
    for surface, block, (photo, seen, _through) in zip(surfaces, blocked, photos):
        image = np.clip(texture(surface, photo, seen, block, log, plain=paint), 0, 255).astype(np.uint8)
        jpeg = io.BytesIO()
        Image.fromarray(image).save(jpeg, "JPEG", quality=92)
        (out / "textures" / f"{surface.name}.jpg").write_bytes(jpeg.getvalue())
        kind = "wall" if surface.name.startswith("wall") else surface.name
        meshes.append({"name": surface.name, "quad": quad(surface, frame), "jpeg": jpeg.getvalue(),
                       "roughness": ROUGHNESS[kind]})
    write_glb(out / "shell.glb", meshes)
    log(f"wrote shell.glb: {len(meshes)} surfaces, {(out / 'shell.glb').stat().st_size / 1e6:.1f} MB")

    pieces = []
    boxes = room.shapes["boxes"]
    for ident, mask in masks.items():
        if mask.sum() == 0:
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
            entry.update(label=box.get("detected") or box.get("label"), movable=True,
                         box={"min": (blo - anchor).round(3).tolist(), "max": (bhi - anchor).round(3).tolist()})
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
    }
    (out / "scene.json").write_text(json.dumps(manifest, indent=1) + "\n")
    log(f"wrote {out / 'scene.json'}: {len(pieces)} pieces "
        f"({', '.join(p['label'] for p in pieces if p['movable'])})")
    log(f"view: http://localhost:8734/scene-viewer/index.html?scene=../spaces/{space.name}/scene/scene.json")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("space", type=Path)
    parser.add_argument("--cell", type=float, default=0.005, help="metres per texel of the shell's textures")
    parser.add_argument("--splat", default="splat.ply", help="the trained splat to cut up")
    args = parser.parse_args()
    build(args.space.resolve(), args.cell, args.splat)


if __name__ == "__main__":
    main()
