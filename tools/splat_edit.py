#!/usr/bin/env python3
"""Edit a room's splat using what stage 3 knows about the room.

A splat is a list of small coloured blobs (Gaussians) with no idea what a bed
is. shapes.json (stage 3) does know: every object's box, the floor height and
the walls, in the same space as the splat and at a known metric scale. This
uses it to take objects out of the splat and to put new furniture in, without
re-running any stage.

    objects  list the room's objects and size, with positions in metres
    remove   delete an object's Gaussians; the floor and wall it hid, which no
             frame ever saw, are patched with texture sampled from around it
    add      build a piece of furniture from simple parts, as Gaussians, and
             place it on the floor at a position in metres (or where a removed
             object stood, or backed against the nearest wall)
    look     print a viewer link that opens looking at an object

Each edit reads splat-edited.ply if there is one (else splat.ply) and writes
splat-edited.ply, its .splat for the viewer and a starting camera, so edits
chain; --fresh starts again from splat.ply.

Positions are metres from the room's centre: x along the room's length as
the floor plan draws it (left to right), y across it (bottom to top).

Usage:
    python3 tools/splat_edit.py objects spaces/<name>
    python3 tools/splat_edit.py remove spaces/<name> B1 [B4 ...] [--no-patch]
    python3 tools/splat_edit.py add spaces/<name> sofa --at B1 --against-wall
    python3 tools/splat_edit.py add spaces/<name> table --size 1.2 0.7 0.75 --at 0.4 -0.8 --turn 90
    python3 tools/splat_edit.py look spaces/<name> B1
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(ROOT / "tools"))
from splat_tools import SH_C0, SPLAT_DTYPE, read_splat, write_splat  # noqa: E402
from splat_export import (look_matrix, ply_to_splat, start_view,  # noqa: E402
                          view_json)

REMOVE_MARGIN_M = 0.10     # also take blobs this far outside the object's box
                           # (a blanket hangs over the edge of a bed's box)...
KEEP_FLOOR_M = 0.03        # ...but never within this height of the floor,
KEEP_WALL_M = 0.05         # ...this close to the room's walls,
KEEP_OTHER_M = 0.02        # ...or inside another piece of furniture
PATCH_SPACING_M = 0.02
FLOOR_SOURCE_M = 1.5       # floor texture comes from open floor within this distance
ABOVE_M = 0.35             # parts sticking out above the box (a headboard) are
ABOVE_COLOUR_GAP = 60      # cleared up to this height when unlike the wall above
WALL_TOUCH_M = 0.15        # an object this close to a wall hid part of it

# Furniture from simple parts, each (x, y, z, width, depth, height, colour) as
# fractions of the piece's size, in a frame where x runs along its width, y
# along its depth with the back at +y, and z up from the floor.
WOOD, DARK_WOOD = (150, 112, 78), (105, 78, 56)
FABRIC, CUSHION = (118, 126, 142), (140, 148, 164)
LINEN, METAL = (232, 230, 224), (70, 70, 74)
PIECES = {
    "sofa": ((1.8, 0.85, 0.80), [
        (0.5, 0.45, 0.0, 1.0, 0.90, 0.45, FABRIC),        # base
        (0.5, 0.92, 0.0, 1.0, 0.16, 1.00, FABRIC),        # back
        (0.05, 0.45, 0.0, 0.10, 0.90, 0.68, FABRIC),      # arms
        (0.95, 0.45, 0.0, 0.10, 0.90, 0.68, FABRIC),
        (0.5, 0.42, 0.45, 0.80, 0.70, 0.12, CUSHION),     # seat cushion
    ]),
    "armchair": ((0.85, 0.85, 0.85), [
        (0.5, 0.45, 0.0, 1.0, 0.90, 0.45, FABRIC),
        (0.5, 0.92, 0.0, 1.0, 0.16, 1.00, FABRIC),
        (0.08, 0.45, 0.0, 0.16, 0.90, 0.68, FABRIC),
        (0.92, 0.45, 0.0, 0.16, 0.90, 0.68, FABRIC),
        (0.5, 0.42, 0.45, 0.68, 0.70, 0.12, CUSHION),
    ]),
    "bed": ((1.6, 2.0, 0.55), [
        (0.5, 0.5, 0.0, 1.0, 1.0, 0.45, WOOD),             # frame
        (0.5, 0.97, 0.0, 1.0, 0.06, 1.60, DARK_WOOD),      # headboard
        (0.5, 0.48, 0.45, 0.94, 0.92, 0.35, LINEN),        # mattress
        (0.5, 0.86, 0.80, 0.70, 0.12, 0.18, LINEN),        # pillows
    ]),
    "table": ((1.2, 0.7, 0.75), [
        (0.5, 0.5, 0.93, 1.0, 1.0, 0.07, WOOD),            # top
        (0.05, 0.08, 0.0, 0.05, 0.08, 0.93, DARK_WOOD),    # legs
        (0.95, 0.08, 0.0, 0.05, 0.08, 0.93, DARK_WOOD),
        (0.05, 0.92, 0.0, 0.05, 0.08, 0.93, DARK_WOOD),
        (0.95, 0.92, 0.0, 0.05, 0.08, 0.93, DARK_WOOD),
    ]),
    "chair": ((0.45, 0.50, 0.90), [
        (0.5, 0.5, 0.48, 1.0, 1.0, 0.05, WOOD),            # seat
        (0.5, 0.95, 0.48, 1.0, 0.08, 0.52, WOOD),          # back
        (0.08, 0.08, 0.0, 0.10, 0.10, 0.48, DARK_WOOD),
        (0.92, 0.08, 0.0, 0.10, 0.10, 0.48, DARK_WOOD),
        (0.08, 0.92, 0.0, 0.10, 0.10, 0.48, DARK_WOOD),
        (0.92, 0.92, 0.0, 0.10, 0.10, 0.48, DARK_WOOD),
    ]),
    "wardrobe": ((1.2, 0.6, 2.0), [
        (0.5, 0.5, 0.0, 1.0, 1.0, 1.0, WOOD),
        (0.5, 0.0, 0.03, 0.01, 0.02, 0.94, DARK_WOOD),     # door split
        (0.46, 0.0, 0.48, 0.02, 0.03, 0.08, METAL),        # handles
        (0.54, 0.0, 0.48, 0.02, 0.03, 0.08, METAL),
    ]),
    "desk": ((1.2, 0.6, 0.75), [
        (0.5, 0.5, 0.93, 1.0, 1.0, 0.07, WOOD),
        (0.03, 0.5, 0.0, 0.05, 0.95, 0.93, DARK_WOOD),     # side panels
        (0.97, 0.5, 0.0, 0.05, 0.95, 0.93, DARK_WOOD),
    ]),
    "box": ((0.5, 0.5, 0.5), [(0.5, 0.5, 0.0, 1.0, 1.0, 1.0, WOOD)]),
}


# ----------------------------------------------------------------- the room
class Room:
    """shapes.json in convenient form. Stage 3 turns its scene so the walls run
    along x and y; the splat stays in the camera solve's frame, and
    scene = world @ splat for any point."""

    def __init__(self, space: Path):
        self.space = Path(space)
        self.shapes = json.loads((self.space / "shapes.json").read_text())
        meta = json.loads((self.space / "densify.json").read_text())
        self.metre = meta["colmap_units_per_metre"]
        self.world = np.array(self.shapes["world"])
        room = self.shapes["room"]
        self.centre = np.array(room["center"])
        self.half = np.array([room["half_u"], room["half_v"]])
        self.floor_z = self.shapes["room_level"]["floor_z"]

    def box(self, ident: str) -> tuple[int, dict]:
        if not (ident[:1].upper() == "B" and ident[1:].isdigit()):
            sys.exit(f"{ident}: name objects as B<number>; see the objects command")
        i = int(ident[1:])
        if i >= len(self.shapes["boxes"]):
            sys.exit(f"{ident}: the room has only {len(self.shapes['boxes'])} boxes")
        return i, self.shapes["boxes"][i]

    def to_scene(self, xyz: np.ndarray) -> np.ndarray:
        return xyz @ self.world.T

    def to_splat(self, xyz: np.ndarray) -> np.ndarray:
        return xyz @ self.world

    def metres_xy(self, scene_xy) -> np.ndarray:
        return (np.asarray(scene_xy) - self.centre) / self.metre

    def scene_xy(self, metres_xy) -> np.ndarray:
        return self.centre + np.asarray(metres_xy) * self.metre

    def walls(self):
        """Built walls as (axis the wall runs along, fixed coordinate across
        it, span along it), in scene units."""
        out = []
        for p in self.shapes["planes"]:
            if p.get("label") != "wall" or p.get("build") is False:
                continue
            a = np.array(p["axis_a"])
            along = 0 if abs(a[0]) > abs(a[1]) else 1
            c = np.array(p["center"])
            out.append((along, c[1 - along],
                        (c[along] - p["half_a"], c[along] + p["half_a"])))
        return out


# ------------------------------------------------------------ gaussian maths
def quaternion(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> (w, x, y, z)."""
    w = math.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = math.copysign(math.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def discs(centres: np.ndarray, normal: np.ndarray, colours: np.ndarray, spacing: float,
          room: Room, opacity: float = 0.97) -> np.ndarray:
    """Flat Gaussians lying in a surface with this scene-frame normal, one per
    centre, written in the splat's frame."""
    n = normal / np.linalg.norm(normal)
    helper = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    t1 = np.cross(helper, n)
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(n, t1)
    local_to_scene = np.column_stack([t1, t2, n])  # a disc's thin axis is its local z
    q = quaternion(room.world.T @ local_to_scene)
    arr = np.zeros(len(centres), dtype=SPLAT_DTYPE)
    xyz = room.to_splat(centres)
    arr["x"], arr["y"], arr["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    f_dc = (np.clip(colours, 0, 255) / 255.0 - 0.5) / SH_C0
    arr["f_dc_0"], arr["f_dc_1"], arr["f_dc_2"] = f_dc[:, 0], f_dc[:, 1], f_dc[:, 2]
    arr["opacity"] = math.log(opacity / (1 - opacity))
    arr["scale_0"] = arr["scale_1"] = math.log(spacing * 0.75)
    arr["scale_2"] = math.log(spacing * 0.08)
    arr["rot_0"], arr["rot_1"], arr["rot_2"], arr["rot_3"] = q
    return arr


def splat_colours(arr: np.ndarray) -> np.ndarray:
    rgb = 0.5 + SH_C0 * np.stack([arr["f_dc_0"], arr["f_dc_1"], arr["f_dc_2"]], axis=1)
    return np.clip(rgb * 255, 0, 255)


def typical(indices: np.ndarray, colours: np.ndarray, keep: float = 0.6) -> np.ndarray:
    """The share of `indices` whose colour is closest to their median: the
    surface itself, without a blanket edge, a shadow or a sock lying on it."""
    if len(indices) == 0:
        return indices
    distance = np.linalg.norm(colours[indices] - np.median(colours[indices], axis=0), axis=1)
    return indices[distance <= np.quantile(distance, keep)]


def grid(lo: float, hi: float, spacing: float) -> np.ndarray:
    count = max(1, int(round((hi - lo) / spacing)))
    return lo + (np.arange(count) + 0.5) * (hi - lo) / count


# ------------------------------------------------------------------ editing
def load(space: Path, fresh: bool):
    src = space / "splat-edited.ply"
    if fresh or not src.exists():
        src = space / "splat.ply"
    arr, trailing = read_splat(src)
    print(f"editing {src.name}: {len(arr):,} gaussians")
    return arr, trailing


def save(space: Path, arr: np.ndarray, trailing: bytes, look_at: np.ndarray | None,
         room: Room, box_index: int | None = None) -> None:
    out = space / "splat-edited.ply"
    write_splat(out, arr, trailing)
    ply_to_splat(out, out.with_suffix(".splat"))
    view = (camera_looking_at(room, look_at, box_index) if look_at is not None
            else start_view(space, out))
    if view:
        out.with_name("splat-edited.view.json").write_text(json.dumps(view) + "\n")
    print(f"wrote {out} ({len(arr):,} gaussians), {out.with_suffix('.splat').name} "
          "and a starting camera")
    print("view: http://localhost:8734/splat-viewer/index.html?url=../spaces/"
          f"{space.name}/splat-edited.splat")


def remove(room: Room, arr: np.ndarray, idents: list[str], patch: bool):
    scene = room.to_scene(np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype(np.float64))
    colours = splat_colours(arr)
    m = room.metre
    lo_room = room.centre - room.half + KEEP_WALL_M * m
    hi_room = room.centre + room.half - KEEP_WALL_M * m
    removing = {room.box(ident)[0] for ident in idents}
    protected = np.zeros(len(arr), bool)
    for i, other in enumerate(room.shapes["boxes"]):
        if i in removing or not other.get("build", True):
            continue
        o_lo = np.array(other["min"]) + KEEP_OTHER_M * m
        o_hi = np.array(other["max"]) - KEEP_OTHER_M * m
        protected |= np.all((scene >= o_lo) & (scene <= o_hi), axis=1)
    drop = np.zeros(len(arr), bool)
    patches, centres = [], []
    for ident in idents:
        _, box = room.box(ident)
        full_lo = np.array(box["min"]) - REMOVE_MARGIN_M * m
        full_hi = np.array(box["max"]) + REMOVE_MARGIN_M * m
        full_lo[2] = max(full_lo[2], room.floor_z + KEEP_FLOOR_M * m)
        lo, hi = full_lo.copy(), full_hi.copy()
        lo[:2], hi[:2] = np.maximum(lo[:2], lo_room), np.minimum(hi[:2], hi_room)
        inside = np.all((scene >= lo) & (scene <= hi), axis=1) & ~protected
        # Against a wall, and just above the box, take only what does not look
        # like the wall: the back of a headboard, a lamp taller than the box.
        inside |= unlike_wall(room, scene, colours, full_lo, full_hi, protected | inside)
        drop |= inside
        centres.append((lo + hi) / 2)
        print(f"{ident} {box.get('label')}: removing {int(inside.sum()):,} gaussians")
        if patch:
            patches.append(patch_floor(room, scene, colours, lo, hi, drop | protected))
            patches += patch_walls(room, scene, colours, lo, hi, inside)
    kept = arr[~drop]
    added = [p for p in patches if p is not None and len(p)]
    if added:
        print(f"patched the hidden floor and wall with {sum(len(p) for p in added):,} gaussians")
        kept = np.concatenate([kept, *added])
    return kept, np.mean(centres, axis=0)


def unlike_wall(room: Room, scene, colours, lo, hi, taken) -> np.ndarray:
    """Blobs in the box reaching into the wall band, or up to ABOVE_M over its
    top, whose colour is unlike the wall above the object."""
    m = room.metre
    footprint = np.all((scene[:, :2] >= lo[:2]) & (scene[:, :2] <= hi[:2]), axis=1)
    higher = np.flatnonzero(footprint & (scene[:, 2] > hi[2] + ABOVE_M * m)
                            & (scene[:, 2] <= hi[2] + 0.9 * m))
    if len(higher) < 50:
        return np.zeros(len(scene), bool)
    wall = np.median(colours[typical(higher, colours)], axis=0)
    region = footprint & (scene[:, 2] >= lo[2]) & (scene[:, 2] <= hi[2] + ABOVE_M * m) & ~taken
    idx = np.flatnonzero(region)
    out = np.zeros(len(scene), bool)
    out[idx] = np.linalg.norm(colours[idx] - wall, axis=1) > ABOVE_COLOUR_GAP
    return out


def patch_floor(room: Room, scene, colours, lo, hi, taken):
    """Cover the floor under a removed object with texture from the open floor
    nearby (right beside a bed is often shadow or clothes)."""
    m = room.metre
    near_floor = np.abs(scene[:, 2] - room.floor_z) < 0.05 * m
    open_floor = near_floor & ~taken
    for other in room.shapes["boxes"]:
        if other.get("build", True):
            o_lo, o_hi = np.array(other["min"][:2]), np.array(other["max"][:2])
            open_floor &= ~np.all((scene[:, :2] >= o_lo - 0.1 * m)
                                  & (scene[:, :2] <= o_hi + 0.1 * m), axis=1)
    dx = np.maximum(np.maximum(lo[0] - scene[:, 0], scene[:, 0] - hi[0]), 0)
    dy = np.maximum(np.maximum(lo[1] - scene[:, 1], scene[:, 1] - hi[1]), 0)
    source = np.flatnonzero(open_floor & (np.hypot(dx, dy) > 0.1 * m)
                            & (np.hypot(dx, dy) < FLOOR_SOURCE_M * m))
    if len(source) < 50:
        print("  floor patch skipped: too little open floor near the object")
        return None
    source = typical(source, colours, keep=0.5)
    height = float(np.median(scene[source, 2]))
    s = PATCH_SPACING_M * m
    xs, ys = np.meshgrid(grid(lo[0], hi[0], s), grid(lo[1], hi[1], s))
    centres = np.column_stack([xs.ravel(), ys.ravel(), np.full(xs.size, height)])
    pick = np.random.default_rng(1).choice(source, len(centres))
    return discs(centres, np.array([0.0, 0.0, 1.0]), colours[pick], s, room)


def patch_walls(room: Room, scene, colours, lo, hi, removed):
    """Cover the stretch of wall an object stood against, up to its top, with
    texture from the same wall beside it."""
    m = room.metre
    out = []
    for along, offset, (w_lo, w_hi) in room.walls():
        across = 1 - along
        gap = min(abs(offset - lo[across]), abs(offset - hi[across]))
        if gap > WALL_TOUCH_M * m + REMOVE_MARGIN_M * m:
            continue
        a_lo, a_hi = max(lo[along], w_lo), min(hi[along], w_hi)
        if a_hi - a_lo < 0.1 * m:
            continue
        on_wall = np.abs(scene[:, across] - offset) < 0.06 * m
        # The wall just above the hidden stretch is most likely the same paint.
        over = (scene[:, along] > a_lo) & (scene[:, along] < a_hi) & \
               (scene[:, 2] > hi[2] + 0.02 * m) & (scene[:, 2] < hi[2] + 0.45 * m)
        source = np.flatnonzero(on_wall & over & ~removed)
        if len(source) < 50:
            beside = ((scene[:, along] < a_lo) & (scene[:, along] > a_lo - 0.6 * m)) | \
                     ((scene[:, along] > a_hi) & (scene[:, along] < a_hi + 0.6 * m))
            low = (scene[:, 2] > room.floor_z + 0.1 * m) & (scene[:, 2] < hi[2] + 0.1 * m)
            source = np.flatnonzero(on_wall & beside & low & ~removed)
        if len(source) < 50:
            continue
        source = typical(source, colours)
        depth = float(np.median(scene[source, across]))
        s = PATCH_SPACING_M * m
        ts, zs = np.meshgrid(grid(a_lo, a_hi, s), grid(room.floor_z + 0.02 * m, hi[2], s))
        centres = np.zeros((ts.size, 3))
        centres[:, along], centres[:, across], centres[:, 2] = ts.ravel(), depth, zs.ravel()
        normal = np.zeros(3)
        normal[across] = 1.0 if offset < room.centre[across] else -1.0
        pick = np.random.default_rng(2).choice(source, len(centres))
        out.append(discs(centres, normal, colours[pick], s, room))
    return out


def build_piece(room: Room, kind: str, size, at_xy, turn_deg: float, colour=None):
    """Gaussians for a piece of furniture standing on the floor: its bottom at
    the floor, centred on at_xy (scene units), turned turn_deg about the
    vertical (0 = its back towards +y)."""
    m = room.metre
    default_size, parts = PIECES[kind]
    W, D, H = (np.array(size) if size else np.array(default_size)) * m
    a = math.radians(turn_deg)
    rot = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
    s = PATCH_SPACING_M * m * 0.75
    light = np.array([0.35, -0.45, 0.82])
    light /= np.linalg.norm(light)
    pieces = []
    for fx, fy, fz, fw, fd, fh, rgb in parts:
        if colour is not None and rgb in (FABRIC, WOOD):
            rgb = colour
        size3 = np.array([fw * W, fd * D, fh * H])
        mid = np.array([(fx - 0.5) * W, (fy - 0.5) * D, fz * H + size3[2] / 2])
        for axis in range(3):
            for sign in (-1.0, 1.0):
                if axis == 2 and sign < 0 and fz == 0:
                    continue  # the underside sits on the floor
                others = [k for k in range(3) if k != axis]
                u = grid(-size3[others[0]] / 2, size3[others[0]] / 2, s)
                v = grid(-size3[others[1]] / 2, size3[others[1]] / 2, s)
                uu, vv = np.meshgrid(u, v)
                local = np.zeros((uu.size, 3))
                local[:, others[0]], local[:, others[1]] = uu.ravel(), vv.ravel()
                local[:, axis] = sign * size3[axis] / 2
                normal = np.zeros(3)
                normal[axis] = sign
                normal = rot @ normal
                centres = (local + mid) @ rot.T + np.array([at_xy[0], at_xy[1], room.floor_z])
                shade = 0.72 + 0.28 * max(0.0, float(normal @ light))
                jitter = np.random.default_rng(len(pieces)).normal(0, 4, (len(centres), 3))
                cols = np.array(rgb, float) * shade + jitter
                pieces.append(discs(centres, normal, cols, s, room, opacity=0.99))
    return np.concatenate(pieces), np.array([W, D, H])


def against_wall(room: Room, at_xy, dims, turn_deg: float):
    """Turn and slide a piece so its back is flush with the nearest wall."""
    best = None
    for along, offset, (w_lo, w_hi) in room.walls():
        across = 1 - along
        if not (w_lo <= at_xy[along] <= w_hi):
            continue
        distance = abs(at_xy[across] - offset)
        if best is None or distance < best[0]:
            best = (distance, along, offset)
    if best is None:
        return at_xy, turn_deg
    _, along, offset = best
    across = 1 - along
    toward = 1.0 if offset > at_xy[across] else -1.0
    # Back (+y in the piece's frame) faces the wall.
    turn = {(1, 1.0): 0.0, (1, -1.0): 180.0, (0, 1.0): -90.0, (0, -1.0): 90.0}[(across, toward)]
    at = np.array(at_xy, float)
    at[across] = offset - toward * dims[1] / 2
    return at, turn


def camera_looking_at(room: Room, target: np.ndarray, box_index: int | None = None) -> dict | None:
    """A viewer camera looking at `target` (scene frame): from the capture
    position that saw that object best, where the splat is sharpest, or else
    2.2 m away at eye height where the room has the most space."""
    m = room.metre
    if box_index is not None:
        from object_frames import best_frames
        from densify import read_images_bin
        from splat_export import model_dir

        picks = best_frames(room.space, {box_index: room.shapes["boxes"][box_index]}, per_box=1)
        model = model_dir(room.space)
        if picks and model is not None:
            info = next(v for v in read_images_bin(model / "images.bin").values()
                        if v["name"] == picks[box_index][0]["frame"])
            position = -info["R"].T @ info["t"]
            look = room.to_splat(target[None])[0] - position
            return view_json(look_matrix(position, look, room.world[2]), position)
    best = None
    for angle in np.radians(np.arange(0, 360, 15)):
        direction = np.array([math.cos(angle), math.sin(angle), 0.0])
        position = target.copy()
        position[:2] = target[:2] + direction[:2] * 2.2 * m
        position[2] = room.floor_z + 1.5 * m
        inside = np.all(np.abs(position[:2] - room.centre) < room.half - 0.25 * m)
        room_space = -np.max(np.abs(position[:2] - room.centre) - room.half)
        if inside and (best is None or room_space > best[0]):
            best = (room_space, position)
    if best is None:
        return None
    position = room.to_splat(best[1][None])[0]
    look = room.to_splat(target[None])[0] - position
    up = room.world[2]
    return view_json(look_matrix(position, look, up), position)


# ----------------------------------------------------------------- commands
def cmd_objects(room: Room, _args) -> None:
    size = lambda v: round(v / room.metre, 2)
    print(f"room {2 * size(room.half[0])} x {2 * size(room.half[1])} m "
          "(positions: metres from its centre, x left to right and y bottom to top "
          "on the floor plan)")
    for i, b in enumerate(room.shapes["boxes"]):
        lo, hi = np.array(b["min"]), np.array(b["max"])
        at = room.metres_xy((lo[:2] + hi[:2]) / 2)
        state = "built" if b.get("build", True) else "not built"
        print(f"  B{i:<2} {b.get('label', '?'):9} {state:9} "
              f"{size(hi[0] - lo[0])} x {size(hi[1] - lo[1])} x {size(hi[2] - lo[2])} m "
              f"at ({at[0]:+.2f}, {at[1]:+.2f})  {(b.get('reason') or '')[:48]}")


def cmd_remove(room: Room, args) -> None:
    arr, trailing = load(room.space, args.fresh)
    kept, centre = remove(room, arr, args.objects, patch=not args.no_patch)
    save(room.space, kept, trailing, centre, room, room.box(args.objects[0])[0])


def cmd_add(room: Room, args) -> None:
    arr, trailing = load(room.space, args.fresh)
    box_index = None
    if len(args.at) == 1:
        box_index, box = room.box(args.at[0])
        at = (np.array(box["min"][:2]) + np.array(box["max"][:2])) / 2
    elif len(args.at) == 2:
        at = room.scene_xy([float(v) for v in args.at])
    else:
        sys.exit("--at takes an object (B1) or two numbers (x y metres)")
    kind = args.kind
    dims = (np.array(args.size) if args.size else np.array(PIECES[kind][0])) * room.metre
    turn = args.turn
    if args.against_wall:
        at, turn = against_wall(room, at, dims, turn)
    piece, dims = build_piece(room, kind, args.size, at, turn,
                              tuple(args.colour) if args.colour else None)
    centre = np.array([at[0], at[1], room.floor_z + dims[2] / 2])
    where = room.metres_xy(at)
    print(f"added a {kind}, {dims[0] / room.metre:.2f} x {dims[1] / room.metre:.2f} x "
          f"{dims[2] / room.metre:.2f} m, at ({where[0]:+.2f}, {where[1]:+.2f}) turned "
          f"{turn:.0f} deg: {len(piece):,} gaussians")
    save(room.space, np.concatenate([arr, piece]), trailing, centre, room, box_index)


def cmd_look(room: Room, args) -> None:
    index, box = room.box(args.object)
    target = (np.array(box["min"]) + np.array(box["max"])) / 2
    view = camera_looking_at(room, target, index)
    if view is None:
        sys.exit("no spot inside the room to look from")
    print(f"http://localhost:8734/splat-viewer/index.html?url=../spaces/{room.space.name}/"
          f"{args.splat}&fov={view['fovY']:g}#{json.dumps(view['viewMatrix']).replace(' ', '')}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("objects", help="list the room's objects")
    p.add_argument("space")
    p = sub.add_parser("remove", help="take objects out of the splat")
    p.add_argument("space")
    p.add_argument("objects", nargs="+", help="B<number> from the objects list")
    p.add_argument("--no-patch", action="store_true",
                   help="leave the hidden floor and wall empty")
    p.add_argument("--fresh", action="store_true", help="start from splat.ply")
    p = sub.add_parser("add", help="put furniture into the splat")
    p.add_argument("space")
    p.add_argument("kind", choices=sorted(PIECES))
    p.add_argument("--at", nargs="+", required=True,
                   help="B<number> (where that object stands) or x y in metres")
    p.add_argument("--size", nargs=3, type=float, metavar=("WIDTH", "DEPTH", "HEIGHT"),
                   help="metres (default: a typical size)")
    p.add_argument("--turn", type=float, default=0.0,
                   help="degrees about the vertical; 0 puts its back towards +y")
    p.add_argument("--against-wall", action="store_true",
                   help="turn and slide it back against the nearest wall")
    p.add_argument("--colour", nargs=3, type=int, metavar=("R", "G", "B"))
    p.add_argument("--fresh", action="store_true", help="start from splat.ply")
    p = sub.add_parser("look", help="print a viewer link looking at an object")
    p.add_argument("space")
    p.add_argument("object")
    p.add_argument("--splat", default="splat.splat")
    args = parser.parse_args()
    room = Room(Path(args.space))
    {"objects": cmd_objects, "remove": cmd_remove, "add": cmd_add,
     "look": cmd_look}[args.command](room, args)


if __name__ == "__main__":
    main()
