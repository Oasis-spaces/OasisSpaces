#!/usr/bin/env python3
"""Classify detected shapes in a space's shapes.json.

Usage:
    python3 tools/classify_shapes.py spaces/<name> [--no-finish | --finish-only]

Reads spaces/<name>/shapes.json, assigns a "label" to every plane and box
(rewriting the file in place), and prints a table of the decisions. Then it
finishes the room (see finish_room): furniture stands on the floor, walls move
out so furniture does not cut through them, and sides nobody filmed get an
inferred wall. pipeline/agent.py labels with --no-finish, lets Claude review,
and finishes with --finish-only.

Plane labels:
    floor    large horizontal plane at the room's floor level
    ceiling  horizontal plane clearly above the floor by ~room height
    wall     vertical plane
    surface  other horizontal plane (e.g. a table top)
    slanted  anything else

Box labels (all thresholds are RELATIVE to room size derived from the floor
plane and wall heights, so the classifier works regardless of scan units):
    seat, table, bed, wardrobe, clutter, block (fallback)

Coordinates in shapes.json are already in a Z-up frame (up = +Z).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median

# --------------------------------------------------------------------------
# Thresholds. Angles are absolute (orientation is unit-free); everything else
# is a fraction of room height (H) or floor area (A).
# --------------------------------------------------------------------------
HORIZONTAL_MAX_TILT_DEG = 25.0   # normal within this angle of +Z  -> horizontal
VERTICAL_MIN_TILT_DEG = 65.0     # normal at least this far from +Z -> vertical

FLOOR_AREA_MIN_FRAC = 0.40       # floor candidate: area >= this frac of largest horizontal plane
FLOOR_LEVEL_TOL_FRAC = 0.25      # floor candidate: at most this * H above the wall-bottom estimate...
FLOOR_BELOW_TOL_FRAC = 0.50      # ...or this * H below it: furniture often hides the lower walls
CEILING_MIN_HEIGHT_FRAC = 0.65   # ceiling: at least this * H above floor level

SEAT_MAX_FOOTPRINT_FRAC = 0.06   # seat: footprint <= 6% of floor area
SEAT_MAX_HEIGHT_FRAC = 0.25      # seat/table height band: < ~25% of room height
TABLE_MAX_HEIGHT_FRAC = 0.35     # table/bed may be slightly taller
TABLE_MIN_ELONGATION = 1.60      # elongated footprint => table
TABLE_MAX_FOOTPRINT_FRAC = 0.15  # a table still has a bounded footprint
BED_MIN_FOOTPRINT_FRAC = 0.15    # bed: footprint > ~15% of floor area
WARDROBE_MIN_HEIGHT_FRAC = 0.55  # wardrobe: taller than ~55% of room height
WARDROBE_MAX_FOOTPRINT_FRAC = 0.20
WARDROBE_MIN_FOOTPRINT_FRAC = 0.002
WARDROBE_WALL_CLEARANCE_FRAC = 0.10  # of sqrt(floor area)

NEAR_FLOOR_MAX_GAP_FRAC = 0.20   # bottom within this * H above floor counts as "on the floor"
NEAR_FLOOR_MIN_GAP_FRAC = -0.25  # tolerate the box sinking slightly below the floor fit
FLOATING_MIN_GAP_FRAC = 0.35     # bottom higher than this * H above floor -> floating clutter
LOW_DENSITY_MAX_REL = 0.10       # box surface density below 10% of scene density -> clutter
OVERSIZE_HEIGHT_FRAC = 1.10      # taller than the room itself -> not furniture
OVERSIZE_FOOTPRINT_FRAC = 0.70   # covers most of the floor -> not furniture
# A leftover cluster nobody recognised (source "geometry") that is nearly room
# height AND covers a large share of the floor is room shell (walls, ceiling),
# not furniture: a wardrobe is tall but narrow, a bed is broad but low.
LEFTOVER_MAX_HEIGHT_FRAC = 0.80
LEFTOVER_MAX_FOOTPRINT_FRAC = 0.20


# --------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------
def plane_tilt_deg(plane: dict) -> float:
    """Angle between the plane normal and the up axis (+Z), folded to [0, 90]."""
    nz = abs(plane["normal"][2])
    return math.degrees(math.acos(max(-1.0, min(1.0, nz))))


def plane_area(plane: dict) -> float:
    return 4.0 * plane["half_a"] * plane["half_b"]


def plane_z_extent(plane: dict) -> float:
    """Half-extent of the plane patch along Z."""
    return (abs(plane["axis_a"][2]) * plane["half_a"]
            + abs(plane["axis_b"][2]) * plane["half_b"])


def box_size(box: dict) -> tuple[float, float, float]:
    return tuple(box["max"][i] - box["min"][i] for i in range(3))


def box_surface_area(box: dict) -> float:
    sx, sy, sz = box_size(box)
    return 2.0 * (sx * sy + sx * sz + sy * sz)


def box_center(box: dict) -> tuple[float, float, float]:
    return tuple((box["min"][i] + box["max"][i]) / 2.0 for i in range(3))


# --------------------------------------------------------------------------
# Room context
# --------------------------------------------------------------------------
@dataclass
class RoomContext:
    floor_z: float
    room_height: float
    floor_area: float
    ref_density: float | None       # scene points-per-area reference (median over planes)
    floor_source: str
    height_source: str
    walls: list[dict] = field(default_factory=list)
    surfaces: list[dict] = field(default_factory=list)
    room: dict | None = None  # footprint the walls enclose (see room_footprint)


# Furniture this far beyond the walls' footprint (as a share of its larger
# half-size) stands in a neighbouring space seen through a door, not this room.
OUTSIDE_ROOM_MARGIN_FRAC = 0.10


def room_footprint(walls: list[dict]) -> dict | None:
    """The rectangle the walls enclose, lined up with the room's main wall
    direction (wall directions folded to 90 degrees, weighted by points)."""
    if len(walls) < 2:
        return None
    sx = sy = 0.0
    for w in walls:
        angle = math.atan2(w["axis_a"][1], w["axis_a"][0])
        sx += w["points"] * math.cos(4 * angle)
        sy += w["points"] * math.sin(4 * angle)
    angle = math.atan2(sy, sx) / 4
    u = (math.cos(angle), math.sin(angle))
    v = (-u[1], u[0])
    us, vs = [], []
    for w in walls:
        for sign in (-1, 1):
            x = w["center"][0] + sign * w["half_a"] * w["axis_a"][0]
            y = w["center"][1] + sign * w["half_a"] * w["axis_a"][1]
            us.append(x * u[0] + y * u[1])
            vs.append(x * v[0] + y * v[1])
    cu, cv = (min(us) + max(us)) / 2, (min(vs) + max(vs)) / 2
    return {"center": [cu * u[0] + cv * v[0], cu * u[1] + cv * v[1]],
            "axis_u": list(u), "axis_v": list(v),
            "half_u": (max(us) - min(us)) / 2, "half_v": (max(vs) - min(vs)) / 2}


def outside_room(box: dict, room: dict) -> bool:
    x, y, _ = box_center(box)
    du = (x - room["center"][0]) * room["axis_u"][0] + (y - room["center"][1]) * room["axis_u"][1]
    dv = (x - room["center"][0]) * room["axis_v"][0] + (y - room["center"][1]) * room["axis_v"][1]
    margin = OUTSIDE_ROOM_MARGIN_FRAC * max(room["half_u"], room["half_v"])
    return abs(du) > room["half_u"] + margin or abs(dv) > room["half_v"] + margin


def fit_plane_to_room(plane: dict, room: dict) -> None:
    """Make a floor or ceiling a horizontal rectangle the size of the room
    footprint, at its measured height (the camera may only have seen the floor
    of a neighbouring space, so its own extent means little)."""
    u, v = room["axis_u"], room["axis_v"]
    x, y = room["center"]
    plane.update({"normal": [0.0, 0.0, 1.0], "center": [x, y, plane["center"][2]],
                  "axis_a": [u[0], u[1], 0.0], "axis_b": [v[0], v[1], 0.0],
                  "half_a": room["half_u"], "half_b": room["half_v"]})


# Walls within this angle of a room axis are squared up to it; a wall end is
# moved to meet a perpendicular wall lying within this share of the room side.
WALL_SNAP_DEGREES = 15.0
CORNER_JOIN_FRAC = 0.40


def to_room(room: dict, point) -> tuple[float, float]:
    """Scene x, y -> offsets along the room's u and v axes from its centre."""
    dx, dy = point[0] - room["center"][0], point[1] - room["center"][1]
    u, v = room["axis_u"], room["axis_v"]
    return dx * u[0] + dy * u[1], dx * v[0] + dy * v[1]


def from_room(room: dict, du: float, dv: float) -> tuple[float, float]:
    u, v = room["axis_u"], room["axis_v"]
    return (room["center"][0] + du * u[0] + dv * v[0],
            room["center"][1] + du * u[1] + dv * v[1])


def room_walls(planes: list[dict], room: dict) -> list[dict]:
    """Walls lying along the room's axes, in room coordinates: the axis each
    runs along, its offset across the room, and where its ends are. Walls at
    an angle to the room are left out, so they stay as measured."""
    u, v = room["axis_u"], room["axis_v"]
    cos_snap = math.cos(math.radians(WALL_SNAP_DEGREES))
    walls = []
    for plane in planes:
        a = plane["axis_a"]
        on_u = abs(a[0] * u[0] + a[1] * u[1])
        on_v = abs(a[0] * v[0] + a[1] * v[1])
        if max(on_u, on_v) < cos_snap:
            continue
        along = "u" if on_u >= on_v else "v"
        du, dv = to_room(room, plane["center"])
        position, offset = (du, dv) if along == "u" else (dv, du)
        walls.append({"plane": plane, "along": along, "offset": offset,
                      "lo": position - plane["half_a"], "hi": position + plane["half_a"]})
    return walls


def join_corners(walls: list[dict], room: dict) -> None:
    """Move each wall end to the nearest perpendicular wall, if one is close."""
    for w in walls:
        side = 2 * (room["half_u"] if w["along"] == "u" else room["half_v"])
        crossing = [c["offset"] for c in walls if c["along"] != w["along"]]
        for end in ("lo", "hi"):
            near = [c for c in crossing if abs(c - w[end]) <= CORNER_JOIN_FRAC * side]
            if near:
                w[end] = min(near, key=lambda c: abs(c - w[end]))


def place_walls(walls: list[dict], room: dict, floor_z: float, height: float) -> None:
    """Write room-coordinate walls back to their planes: exactly vertical,
    along the room's axes, from the floor up to the room height."""
    u, v = room["axis_u"], room["axis_v"]
    for w in walls:
        plane = w["plane"]
        axis, across = (u, v) if w["along"] == "u" else (v, u)
        mid = (w["lo"] + w["hi"]) / 2
        x, y = from_room(room, *((mid, w["offset"]) if w["along"] == "u"
                                 else (w["offset"], mid)))
        normal = [across[0], across[1], 0.0]
        if normal[0] * plane["normal"][0] + normal[1] * plane["normal"][1] < 0:
            normal = [-normal[0], -normal[1], 0.0]
        plane.update({
            "normal": normal, "center": [x, y, floor_z + height / 2],
            "axis_a": [axis[0], axis[1], 0.0], "axis_b": [0.0, 0.0, 1.0],
            "half_a": max((w["hi"] - w["lo"]) / 2, 1e-6), "half_b": height / 2,
        })


def square_up_room(planes: list[dict], labels: list[str], ctx: RoomContext) -> int:
    """Regularise walls the way room scanners do: exactly vertical, along the
    room's axes, spanning floor to room height, with ends meeting the nearest
    perpendicular wall. Positions across the room stay as measured, and gaps in
    the middle of a wall (doorways) are left open. Returns walls changed."""
    walls = room_walls([p for p, lab in zip(planes, labels) if lab == "wall"], ctx.room)
    join_corners(walls, ctx.room)
    place_walls(walls, ctx.room, ctx.floor_z, ctx.room_height)
    return len(walls)


# --------------------------------------------------------------------------
# Finishing: once it is settled which walls and boxes get built (by the checks
# here, then by Claude's review in pipeline/agent.py), make the room hold
# together: furniture stands on the floor, no wall cuts through furniture, and
# every side of the room has a wall.
# --------------------------------------------------------------------------
FLOOR_STANDING = {"bed", "seat", "table", "wardrobe", "block"}
# Detected things that may sit on other furniture: extended to the floor only
# when they already start near it (a floor lamp, not a bedside lamp).
MAY_SIT_ON_FURNITURE = {"pillow", "lamp", "potted plant"}
# A wall this close to a side of the room (share of the room's width) is on
# that side: it closes the side, and it moves out to contain furniture.
WALL_ON_SIDE_FRAC = 0.10
# Furniture poking through a wall moves the wall out by at most this share of
# the room's width; anything left over is trimmed off the furniture instead.
MAX_WALL_PUSH_FRAC = 0.10
INFERRED_WALL_COLOR = [232, 218, 190]


def box_in_room(box: dict, room: dict) -> tuple[float, float, float, float]:
    """A box's extent in room coordinates: u_lo, u_hi, v_lo, v_hi."""
    corners = [to_room(room, (x, y)) for x in (box["min"][0], box["max"][0])
               for y in (box["min"][1], box["max"][1])]
    us, vs = [c[0] for c in corners], [c[1] for c in corners]
    return min(us), max(us), min(vs), max(vs)


def trim_box(box: dict, room: dict, wall: dict, sign: float) -> bool:
    """Cut the part of a box beyond a wall off. Boxes are aligned with the
    scene axes, so this only works when the room is too (shapes.py turns the
    scene to the walls); returns False otherwise."""
    across = room["axis_v"] if wall["along"] == "u" else room["axis_u"]
    k = 0 if abs(across[0]) > 0.999 else 1 if abs(across[1]) > 0.999 else None
    if k is None:
        return False
    limit = room["center"][k] + wall["offset"] * across[k]
    if across[k] * sign > 0:
        box["max"][k] = min(box["max"][k], limit)
    else:
        box["min"][k] = max(box["min"][k], limit)
    return True


def contain_furniture(walls: list[dict], boxes: list[dict], room: dict, size) -> list[str]:
    """Move walls on the room's edge out so furniture does not cut through
    them. The measured furniture wins over the measured wall, up to a limit."""
    notes = []
    for w in walls:
        half_across = room["half_v"] if w["along"] == "u" else room["half_u"]
        if abs(w["offset"]) < half_across - WALL_ON_SIDE_FRAC * 2 * half_across:
            continue  # inside the room, not on its edge: nothing to contain
        sign = 1.0 if w["offset"] > 0 else -1.0
        poking = []
        for b in boxes:
            u_lo, u_hi, v_lo, v_hi = box_in_room(b, room)
            along_lo, along_hi, across_lo, across_hi = (
                (u_lo, u_hi, v_lo, v_hi) if w["along"] == "u" else (v_lo, v_hi, u_lo, u_hi))
            if along_hi <= w["lo"] or along_lo >= w["hi"]:
                continue  # beside the wall, for example in a doorway
            beyond = (across_hi if sign > 0 else -across_lo) - abs(w["offset"])
            if beyond > 0:
                poking.append((b, beyond))
        if not poking:
            continue
        push = min(max(d for _, d in poking), MAX_WALL_PUSH_FRAC * 2 * half_across)
        w["offset"] += sign * push
        name = f"W{w['index']}" if "index" in w else "a wall"
        notes.append(f"moved {name} out {size(push)} so "
                     + ", ".join(b.get("label", "furniture") for b, _ in poking)
                     + " stays inside")
        for b, beyond in poking:
            if beyond > push + 1e-9 and trim_box(b, room, w, sign):
                notes.append(f"trimmed {size(beyond - push)} off the {b.get('label')} "
                             f"where it still went through {name}")
    return notes


def close_room(walls: list[dict], boxes: list[dict], room: dict, size) -> list[str]:
    """Give every side of the room a wall. A side no measured wall covers
    (the camera never faced it) gets an inferred wall where the room's walls
    and furniture end; it is marked so it can be drawn as a guess."""
    ends = {"u": [], "v": []}
    for w in walls:
        ends[w["along"]] += [w["lo"], w["hi"]]
        ends["v" if w["along"] == "u" else "u"].append(w["offset"])
    for b in boxes:
        u_lo, u_hi, v_lo, v_hi = box_in_room(b, room)
        ends["u"] += [u_lo, u_hi]
        ends["v"] += [v_lo, v_hi]
    if not ends["u"] or not ends["v"]:
        return []
    u_lo, u_hi, v_lo, v_hi = min(ends["u"]), max(ends["u"]), min(ends["v"]), max(ends["v"])
    notes = []
    # (runs along, offset across, span along, room width across, side name)
    sides = [("v", u_lo, (v_lo, v_hi), u_hi - u_lo, "-u"),
             ("v", u_hi, (v_lo, v_hi), u_hi - u_lo, "+u"),
             ("u", v_lo, (u_lo, u_hi), v_hi - v_lo, "-v"),
             ("u", v_hi, (u_lo, u_hi), v_hi - v_lo, "+v")]
    for along, offset, (lo, hi), width, name in sides:
        if hi - lo <= 0 or any(w["along"] == along
                               and abs(w["offset"] - offset) <= WALL_ON_SIDE_FRAC * width
                               for w in walls):
            continue
        across = room["axis_u"] if along == "v" else room["axis_v"]
        inward = -1.0 if name[0] == "+" else 1.0
        plane = {"kind": "wall", "label": "wall", "source": "inferred", "points": 0,
                 "color": INFERRED_WALL_COLOR, "build": True,
                 "reason": "inferred: no wall was seen on this side, so it closes "
                           "the room where the walls and furniture end",
                 "normal": [inward * across[0], inward * across[1], 0.0]}
        walls.append({"plane": plane, "along": along, "offset": offset, "lo": lo, "hi": hi})
        notes.append(f"added an inferred wall on the {name} side ({size(hi - lo)} long), "
                     "where no wall was seen")
    return notes


def finish_room(data: dict, units_per_metre: float | None = None) -> list[str]:
    """Rest furniture on the floor, keep it inside the walls, and close the
    room with inferred walls. Works on built walls and boxes only, and starts
    from the measured walls each time, so it can be re-run after a review."""
    level = data.get("room_level")
    if not level:
        return ["no floor level recorded; run the classifier first"]
    floor_z, height = level["floor_z"], level["height"]
    size = ((lambda d: f"{100 * d / units_per_metre:.0f} cm") if units_per_metre
            else (lambda d: f"{d:.2f} units"))
    planes, boxes = data.get("planes", []), data.get("boxes", [])
    planes[:] = [p for p in planes if p.get("source") != "inferred"]
    notes = []

    built = [(i, b) for i, b in enumerate(boxes) if b.get("build", True)]
    for i, b in built:
        gap = b["min"][2] - floor_z
        if gap < 0:
            b["min"][2] = floor_z  # sunk into the floor: cut at the floor
        elif (gap > 0 and b.get("label") in FLOOR_STANDING
              and not (b.get("detected") in MAY_SIT_ON_FURNITURE
                       and gap > NEAR_FLOOR_MAX_GAP_FRAC * height)):
            b["min"][2] = floor_z
            notes.append(f"stood B{i} {b.get('label')} on the floor "
                         f"(its bottom was {size(gap)} above it, hidden from the camera)")

    measured = [(i, p) for i, p in enumerate(planes)
                if p.get("label") == "wall" and p.get("build", True)]
    room = room_footprint([p for _, p in measured]) or data.get("room")
    if not room:
        return notes
    walls = room_walls([p for _, p in measured], room)
    index = {id(p): i for i, p in measured}
    for w in walls:
        w["index"] = index[id(w["plane"])]
    furniture = [b for _, b in built]
    join_corners(walls, room)
    notes += contain_furniture(walls, furniture, room, size)
    join_corners(walls, room)
    before = len(walls)
    notes += close_room(walls, furniture, room, size)
    join_corners(walls, room)
    place_walls(walls, room, floor_z, height)
    planes.extend(w["plane"] for w in walls[before:])

    final = room_footprint([p for p in planes
                            if p.get("label") == "wall" and p.get("build", True)]) or room
    for plane in planes:
        if plane.get("label") in ("floor", "ceiling"):
            fit_plane_to_room(plane, final)
    data["room"] = final
    notes.append(f"room {size(2 * final['half_u'])} x {size(2 * final['half_v'])}, "
                 f"{sum(1 for p in planes if p.get('label') == 'wall' and p.get('build', True))} "
                 "walls, floor sized to match")
    return notes


def orientation_class(plane: dict) -> str:
    tilt = plane_tilt_deg(plane)
    if tilt <= HORIZONTAL_MAX_TILT_DEG:
        return "horizontal"
    if tilt >= VERTICAL_MIN_TILT_DEG:
        return "vertical"
    return "slanted"


def classify_planes(planes: list[dict]) -> tuple[list[str], list[str], RoomContext]:
    """Label every plane and derive the room context used for boxes."""
    orient = [orientation_class(p) for p in planes]
    walls = [p for p, o in zip(planes, orient) if o == "vertical"]
    horiz = [(i, p) for i, (p, o) in enumerate(zip(planes, orient)) if o == "horizontal"]

    # Room height and floor level, estimated from the vertical span of walls.
    # Walls are the most reliable cue: horizontal planes may be table tops or
    # duplicate ceiling detections, so the lowest one is NOT necessarily the floor.
    if walls:
        room_height = median(2.0 * plane_z_extent(w) for w in walls)
        floor_est = median(w["center"][2] - plane_z_extent(w) for w in walls)
        height_source = f"median of {len(walls)} wall heights"
    elif horiz:
        zs = [p["center"][2] for _, p in horiz]
        floor_est = min(zs)
        spread = max(zs) - min(zs)
        if spread > 1e-6:
            room_height = spread
            height_source = "horizontal-plane z spread (no walls found)"
        else:
            # All horizontal planes at one level: no vertical cue at all.
            # A room's height is on the order of its floor width, so use that
            # (scale-aware) rather than a degenerate near-zero spread.
            largest = max(plane_area(p) for _, p in horiz)
            room_height = max(math.sqrt(largest), 1.0)
            height_source = "sqrt of largest horizontal plane area (no vertical cue)"
    else:
        floor_est, room_height, height_source = 0.0, 1.0, "default (no planes)"

    # Floor: large horizontal planes sitting near the wall-bottom estimate. The
    # tolerance is lopsided: a bed or wardrobe in front of a wall hides its
    # bottom, so the real floor can lie well below where the walls seem to end,
    # while a horizontal plane above that point is more likely a bed or table top.
    floor_idx: set[int] = set()
    # shapes.py marks the floor and ceiling it found as height levels inside
    # the walls; those are trusted over guessing from horizontal planes.
    level_of = {p.get("level"): i for i, p in enumerate(planes) if p.get("level")}
    if "floor" in level_of:
        floor_idx = {level_of["floor"]}
    elif horiz:
        max_area = max(plane_area(p) for _, p in horiz)
        candidates = [
            (i, p) for i, p in horiz
            if plane_area(p) >= FLOOR_AREA_MIN_FRAC * max_area
            and (not walls
                 or -FLOOR_BELOW_TOL_FRAC * room_height
                 <= p["center"][2] - floor_est
                 <= FLOOR_LEVEL_TOL_FRAC * room_height)
        ]
        if candidates:
            lowest_z = min(p["center"][2] for _, p in candidates)
            floor_idx = {i for i, p in candidates
                         if p["center"][2] <= lowest_z + 0.05 * room_height}

    room = room_footprint(walls)
    if floor_idx:
        floor_z = min(planes[i]["center"][2] for i in floor_idx)
        floor_area = max(plane_area(planes[i]) for i in floor_idx)
        floor_source = ("lowest level inside the walls" if "floor" in level_of
                        else "floor plane")
        if room:
            # The floor gets resized to the walls' footprint, so size the room by it.
            floor_area = 4.0 * room["half_u"] * room["half_v"]
            floor_source += ", sized to the walls' footprint"
        if walls and floor_z < floor_est:
            # The walls were only seen above the furniture, so measure the room
            # from the floor that was actually found up to the wall tops.
            wall_top = median(w["center"][2] + plane_z_extent(w) for w in walls)
            if wall_top - floor_z > room_height:
                room_height = wall_top - floor_z
                height_source += "; measured from the floor plane to the wall tops"
        if "ceiling" in level_of:
            room_height = planes[level_of["ceiling"]]["center"][2] - floor_z
            height_source = "floor level to ceiling level"
    else:
        floor_z = floor_est
        floor_area = max((plane_area(p) for _, p in horiz), default=0.0)
        floor_source = "wall-bottom estimate (no floor plane detected)"
    if floor_area <= 0.0:  # last-resort so ratios stay finite
        floor_area = room_height * room_height
        floor_source += "; area from room height"

    ref_density = None
    densities = [p["points"] / plane_area(p) for p in planes if plane_area(p) > 0]
    if densities:
        ref_density = median(densities)

    labels, reasons = [], []
    for i, (plane, o) in enumerate(zip(planes, orient)):
        tilt = plane_tilt_deg(plane)
        z = plane["center"][2]
        if o == "vertical":
            labels.append("wall")
            reasons.append(f"vertical (tilt {tilt:.1f} deg)")
        elif o == "slanted":
            labels.append("slanted")
            reasons.append(f"tilted {tilt:.1f} deg from up")
        elif i in floor_idx:
            labels.append("floor")
            reasons.append(f"large horizontal at floor level z={z:.2f}")
        elif z >= floor_z + CEILING_MIN_HEIGHT_FRAC * room_height:
            labels.append("ceiling")
            reasons.append(f"horizontal, {(z - floor_z) / room_height:.0%} of room height above floor")
        else:
            labels.append("surface")
            reasons.append(f"horizontal, not floor/ceiling (z={z:.2f})")

    ctx = RoomContext(
        floor_z=floor_z,
        room_height=max(room_height, 1e-9),
        floor_area=floor_area,
        ref_density=ref_density,
        floor_source=floor_source,
        height_source=height_source,
        walls=walls,
        surfaces=[p for p, lab in zip(planes, labels) if lab == "surface"],
        room=room,
    )
    return labels, reasons, ctx


# --------------------------------------------------------------------------
# Box classification
# --------------------------------------------------------------------------
def _wall_clearance(box: dict, wall: dict) -> float:
    """Distance between the box and the wall plane (0 if they touch/intersect)."""
    n = wall["normal"]
    c = box_center(box)
    d = sum((c[k] - wall["center"][k]) * n[k] for k in range(3))
    half_along_n = sum(0.5 * s * abs(n[k]) for k, s in enumerate(box_size(box)))
    return max(0.0, abs(d) - half_along_n)


def _has_surface_above(box: dict, ctx: RoomContext) -> bool:
    """A 'surface' plane hovering just above the box footprint => table pairing."""
    cx, cy, _ = box_center(box)
    sx, sy, _ = box_size(box)
    top = box["max"][2]
    h = ctx.room_height
    for s in ctx.surfaces:
        z = s["center"][2]
        if not (top - 0.05 * h <= z <= top + 0.30 * h):
            continue
        if (abs(s["center"][0] - cx) <= 0.5 * sx + 0.10 * math.sqrt(ctx.floor_area)
                and abs(s["center"][1] - cy) <= 0.5 * sy + 0.10 * math.sqrt(ctx.floor_area)):
            return True
    return False


# Things that only ever lie on the floor: a detection of one up on the bed
# is a mislabel (a blanket seen as a rug).
FLOOR_ONLY = {"rug"}

# Detected object names (pipeline/semantics.py) -> furniture library builders.
DETECTED_TO_LIBRARY = {
    "bed": "bed", "sofa": "seat", "armchair": "seat", "chair": "seat",
    "stool": "seat", "table": "table", "desk": "table", "wardrobe": "wardrobe",
    "cabinet": "wardrobe", "chest of drawers": "wardrobe", "shelf": "wardrobe",
    "bookcase": "wardrobe", "lamp": "block", "rug": "block",
    "potted plant": "block", "pillow": "block",
}


def classify_box(box: dict, ctx: RoomContext) -> tuple[str, str, dict]:
    sx, sy, sz = box_size(box)
    footprint = sx * sy
    fp_frac = footprint / ctx.floor_area
    h_frac = sz / ctx.room_height
    gap_frac = (box["min"][2] - ctx.floor_z) / ctx.room_height
    elong = max(sx, sy) / max(min(sx, sy), 1e-9)
    rel_density = None
    if ctx.ref_density:
        rel_density = (box["points"] / max(box_surface_area(box), 1e-9)) / ctx.ref_density
    metrics = {"fp_frac": fp_frac, "h_frac": h_frac, "gap_frac": gap_frac,
               "elong": elong, "rel_density": rel_density}

    near_floor = NEAR_FLOOR_MIN_GAP_FRAC <= gap_frac <= NEAR_FLOOR_MAX_GAP_FRAC

    # Sanity gates first: scan ghosts and debris, in relative terms.
    if h_frac > OVERSIZE_HEIGHT_FRAC or fp_frac > OVERSIZE_FOOTPRINT_FRAC:
        return ("clutter",
                f"implausibly large ({h_frac:.0%} of room height, "
                f"{fp_frac:.0%} of floor area)", metrics)
    if (box.get("source") == "geometry" and h_frac >= LEFTOVER_MAX_HEIGHT_FRAC
            and fp_frac >= LEFTOVER_MAX_FOOTPRINT_FRAC):
        return ("clutter",
                f"unrecognised and room-sized ({h_frac:.0%} of room height, "
                f"{fp_frac:.0%} of floor): leftover walls, not furniture", metrics)
    if rel_density is not None and rel_density < LOW_DENSITY_MAX_REL:
        return ("clutter",
                f"sparse: {rel_density:.1%} of scene surface density", metrics)
    if gap_frac > FLOATING_MIN_GAP_FRAC:
        return ("clutter",
                f"floating {gap_frac:.0%} of room height above floor", metrics)
    if gap_frac < NEAR_FLOOR_MIN_GAP_FRAC:
        return ("clutter",
                f"detached: {-gap_frac:.0%} of room height below floor level", metrics)

    if (h_frac >= WARDROBE_MIN_HEIGHT_FRAC
            and WARDROBE_MIN_FOOTPRINT_FRAC <= fp_frac <= WARDROBE_MAX_FOOTPRINT_FRAC
            and near_floor):
        limit = WARDROBE_WALL_CLEARANCE_FRAC * math.sqrt(ctx.floor_area)
        if any(_wall_clearance(box, w) <= limit for w in ctx.walls):
            return ("wardrobe",
                    f"tall ({h_frac:.0%} of room height), near a wall", metrics)

    if fp_frac >= BED_MIN_FOOTPRINT_FRAC and h_frac <= TABLE_MAX_HEIGHT_FRAC and near_floor:
        return ("bed",
                f"broad footprint ({fp_frac:.0%} of floor), low ({h_frac:.0%} of room height)",
                metrics)

    if h_frac <= TABLE_MAX_HEIGHT_FRAC and near_floor and fp_frac <= TABLE_MAX_FOOTPRINT_FRAC:
        if _has_surface_above(box, ctx):
            return ("table", "surface plane detected just above it", metrics)
        if elong >= TABLE_MIN_ELONGATION and h_frac <= TABLE_MAX_HEIGHT_FRAC:
            return ("table", f"elongated footprint ({elong:.1f}:1), low, on floor", metrics)

    if fp_frac <= SEAT_MAX_FOOTPRINT_FRAC and h_frac <= SEAT_MAX_HEIGHT_FRAC and near_floor:
        return ("seat",
                f"small footprint ({fp_frac:.1%} of floor), "
                f"low ({h_frac:.0%} of room height), on floor", metrics)

    return ("block", "no furniture rule matched", metrics)


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------
def _print_table(headers: list[str], rows: list[list[str]]) -> None:
    widths = [max(len(h), *(len(r[i]) for r in rows)) if rows else len(h)
              for i, h in enumerate(headers)]
    line = "  ".join(h.ljust(w) for h, w in zip(headers, widths))
    print(line)
    print("-" * len(line))
    for r in rows:
        print("  ".join(c.ljust(w) for c, w in zip(r, widths)))


def report(space: str, planes: list[dict], boxes: list[dict],
           plane_labels: list[str], plane_reasons: list[str],
           box_results: list[tuple[str, str, dict]], ctx: RoomContext) -> None:
    print(f"== {space} ==")
    dens = f"{ctx.ref_density:.0f} pts/area" if ctx.ref_density else "n/a"
    print(f"room context: floor_z={ctx.floor_z:.3f} ({ctx.floor_source}), "
          f"room_height={ctx.room_height:.2f} ({ctx.height_source}), "
          f"floor_area={ctx.floor_area:.2f}, scene_density={dens}")
    print()

    print(f"PLANES ({len(planes)})")
    rows = []
    for i, (p, lab, why) in enumerate(zip(planes, plane_labels, plane_reasons)):
        rows.append([
            str(i), p["kind"], lab, f"{plane_tilt_deg(p):5.1f}",
            f"{p['center'][2]:8.3f}",
            f"{2 * p['half_a']:.2f} x {2 * p['half_b']:.2f}",
            str(p["points"]), why,
        ])
    _print_table(["#", "kind", "label", "tilt", "center_z", "size", "points", "reason"], rows)
    print()

    print(f"BOXES ({len(boxes)})")
    rows = []
    for i, (b, (lab, why, m)) in enumerate(zip(boxes, box_results)):
        sx, sy, sz = box_size(b)
        dens = f"{m['rel_density']:.2f}" if m["rel_density"] is not None else "n/a"
        rows.append([
            str(i), lab, f"{sx:.2f} x {sy:.2f} x {sz:.2f}",
            f"{m['fp_frac']:6.1%}", f"{m['h_frac']:5.0%}", f"{m['gap_frac']:+5.0%}",
            dens, str(b["points"]), why,
        ])
    _print_table(["#", "label", "size", "foot/A", "hgt/H", "gap/H", "dens", "points", "reason"],
                 rows)
    print()


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Label planes and boxes in a space's shapes.json (in place).")
    parser.add_argument("space", help="space directory, e.g. spaces/sample-room")
    parser.add_argument("--no-finish", action="store_true",
                        help="label only; pipeline/agent.py finishes the room after "
                             "Claude's review")
    parser.add_argument("--finish-only", action="store_true",
                        help="only finish an already labelled room (see finish_room)")
    args = parser.parse_args(argv)

    shapes_path = Path(args.space) / "shapes.json"
    if not shapes_path.is_file():
        print(f"error: {shapes_path} not found", file=sys.stderr)
        return 1

    with open(shapes_path, encoding="utf-8") as f:
        data = json.load(f)

    meta_path = Path(args.space) / "densify.json"
    units_per_metre = (json.loads(meta_path.read_text()).get("colmap_units_per_metre")
                       if meta_path.is_file() else None)
    if args.finish_only:
        for note in finish_room(data, units_per_metre):
            print(note)
        with open(shapes_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=1)
            f.write("\n")
        print(f"wrote the finished room to {shapes_path}")
        return 0

    planes = data.get("planes", [])
    boxes = data.get("boxes", [])

    plane_labels, plane_reasons, ctx = classify_planes(planes)
    box_results = []
    for box in boxes:
        label, why, metrics = classify_box(box, ctx)
        if ctx.room and outside_room(box, ctx.room):
            box_results.append(("clutter", "outside the room the walls enclose: part of "
                                "a neighbouring space seen through a door", metrics))
            continue
        detected = box.get("detected")
        if detected:
            # An object detector saw a real thing here, so keep its identity;
            # the sanity gates still decide whether the box is worth building.
            gap = (box["min"][2] - ctx.floor_z) / ctx.room_height
            if label == "clutter":
                why = f"detected {detected}, but {why}"
            elif detected in FLOOR_ONLY and gap > NEAR_FLOOR_MAX_GAP_FRAC:
                label = "clutter"
                why = (f"detected {detected}, but it sits {gap:.0%} of room height "
                       f"above the floor, so it is something else")
            else:
                label = DETECTED_TO_LIBRARY.get(detected, "block")
                why = f"detected as {detected}"
        box_results.append((label, why, metrics))

    for plane, label in zip(planes, plane_labels):
        plane["label"] = label
    if ctx.room:
        squared = square_up_room(planes, plane_labels, ctx)
        # Re-measure the footprint from the squared walls so the floor meets them.
        room = room_footprint([p for p, lab in zip(planes, plane_labels) if lab == "wall"]) \
            or ctx.room
        for plane, label in zip(planes, plane_labels):
            if label in ("floor", "ceiling"):
                fit_plane_to_room(plane, room)
        data["room"] = room
        print(f"squared up {squared} wall(s) to the room's axes and joined their corners; "
              f"floor sized to the {2 * room['half_u']:.2f} x {2 * room['half_v']:.2f} footprint")
    for box, (label, why, _) in zip(boxes, box_results):
        box["label"] = label
        box["reason"] = why
        # "clutter" only comes from a failed sanity gate: scan debris, not
        # furniture, so the Blender room skips it.
        box["build"] = label != "clutter"
    data["room_level"] = {"floor_z": ctx.floor_z, "height": ctx.room_height}
    if not args.no_finish:
        for note in finish_room(data, units_per_metre):
            print(note)

    with open(shapes_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1)
        f.write("\n")

    report(args.space, planes, boxes, plane_labels, plane_reasons, box_results, ctx)
    print(f"wrote labels to {shapes_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
