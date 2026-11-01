#!/usr/bin/env python3
"""The room's outline seen from above: any shape, not just a rectangle.

The room model used to be one rectangle: the box around every wall found.
Real rooms are often not one: an L-shaped living room, a kitchen with a
corridor, a bathroom in two parts. And a scan always sees a little of the next
room through each open door, so a wall glimpsed there stretched the rectangle
over floor that is not this room's. Measured on 20 ARKitScenes rooms (real
homes, LiDAR): their ceilings fill only 42-91% of the box around them
(median 81%), so a rectangle claims a fifth too much floor in a typical room
and up to half in the worst.

This finds the outline from what was actually seen, on a grid in the room's
own axes (the walls' directions):

  1. open floor: cells where floor, ceiling or something low (a bed, a
     counter: floor beneath) was seen, or the phone stood;
     blocked: cells with points at wall height between floor and ceiling
     (walls, tall furniture)
  2. the space: the open cells reachable from where the phone stood without
     crossing a blocked cell
  3. doors: the space is shrunk by half a door's width, which cuts it at
     every doorway into parts, and each cell goes to the part nearest it; the
     parts the phone was in are kept (a room glimpsed through a door is left
     out, a room walked through is in), and so are parts too small to be
     another room (an alcove, a shower)
  4. the outline: traced along cell edges, so its sides already run along the
     room's axes, then cleaned of steps shorter than MIN_STEP_M

The result is a polygon (or a few, for parts joined only through a wall's
gap) in scene units; its bounding box replaces the old wall rectangle, and a
plain rectangular room comes out as four corners, as before.
"""

from __future__ import annotations

import numpy as np

CELL_M = 0.05             # grid cell
LEVEL_BAND_M = 0.06       # floor / ceiling points lie this close to their level
WALL_BAND = (0.35, 0.80)  # blocked: points between these shares of the room height
WALL_MIN_POINTS = 3       # points in a cell's wall band for it to block
CAMERA_RADIUS_M = 0.30    # the phone (and the person) stood on this much floor
DOOR_M = 1.10             # the widest gap between two spaces that still counts as a door (a 0.9 m door scans ~1 m)
MIN_PART_CAMERAS = 0.05   # a part must hold this share of the camera positions to be kept...
MIN_ROOM_M2 = 3.0         # ...unless it is too small to be another room (an alcove, a shower)
MIN_STEP_M = 0.30         # outline steps shorter than this are noise, not an alcove
CLOSE_M = 0.15            # gaps in seen floor this narrow are filled


def room_angle(points_xy: np.ndarray, step_deg: float = 0.5, cell: float = CELL_M) -> float:
    """The angle of the room's axes: the turn (0-90 degrees) at which wall
    points pile up sharpest along both axes."""
    best, best_angle = -1.0, 0.0
    for a in np.radians(np.arange(0, 90, step_deg)):
        c, s = np.cos(a), np.sin(a)
        score = 0.0
        for coord in (points_xy @ [c, s], points_xy @ [-s, c]):
            hist = np.bincount(((coord - coord.min()) / cell).astype(int))
            score += float((hist.astype(float) ** 2).sum())
        if score > best:
            best, best_angle = score, a
    return float(best_angle)


def _disk(radius_cells: float) -> np.ndarray:
    r = max(1, int(np.ceil(radius_cells)))
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    return x * x + y * y <= radius_cells * radius_cells + 1e-9


def _trace(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    """Closed loops of cell-edge corners around each region (outer boundaries
    and holes), in (column, row) corner coordinates."""
    m = np.pad(mask, 1)
    edges: dict[tuple[int, int], list[tuple[int, int]]] = {}
    rows, cols = np.nonzero(m)
    for r, c in zip(rows, cols):
        # Each boundary edge, walked with the inside on the left.
        if not m[r - 1, c]:
            edges.setdefault((c + 1, r), []).append((c, r))
        if not m[r + 1, c]:
            edges.setdefault((c, r + 1), []).append((c + 1, r + 1))
        if not m[r, c - 1]:
            edges.setdefault((c, r), []).append((c, r + 1))
        if not m[r, c + 1]:
            edges.setdefault((c + 1, r + 1), []).append((c + 1, r))
    loops = []
    while edges:
        start = next(iter(edges))
        loop, at = [], start
        while True:
            ends = edges.get(at)
            if not ends:
                break
            nxt = ends.pop()
            if not ends:
                del edges[at]
            loop.append(at)
            at = nxt
            if at == start:
                break
        if len(loop) >= 4:
            loops.append([(x - 1, y - 1) for x, y in loop])
    return loops


def _corners(loop):
    """Only the points where the direction turns."""
    n = len(loop)
    out = []
    for i in range(n):
        a, b, c = loop[i - 1], loop[i], loop[(i + 1) % n]
        if (b[0] - a[0], b[1] - a[1]) != (c[0] - b[0], c[1] - b[1]):
            out.append(b)
    return out


def _area(poly) -> float:
    p = np.asarray(poly, float)
    return 0.5 * float(np.dot(p[:, 0], np.roll(p[:, 1], -1)) - np.dot(p[:, 1], np.roll(p[:, 0], -1)))


def _clean(pts):
    """Drop repeated points and points on a straight run (or a spike back
    along it), until every point is a real corner."""
    pts = list(pts)
    changed = True
    while changed and len(pts) >= 3:
        changed = False
        for i in range(len(pts)):
            a, b, c = pts[i - 1], pts[i], pts[(i + 1) % len(pts)]
            if b == a or (a[0] == b[0] == c[0]) or (a[1] == b[1] == c[1]):
                del pts[i]
                changed = True
                break
    return pts


def simplify(poly, min_step: float) -> list[tuple[float, float]]:
    """Remove steps shorter than min_step from a polygon whose sides run along
    the axes: the shortest side is folded away by moving its shorter
    neighbouring side onto the longer one's line, until none is short."""
    pts = _clean([tuple(map(float, p)) for p in poly])
    original = pts
    while len(pts) > 4:
        n = len(pts)
        lengths = [abs(pts[(i + 1) % n][0] - pts[i][0]) + abs(pts[(i + 1) % n][1] - pts[i][1]) for i in range(n)]
        i = int(np.argmin(lengths))
        if lengths[i] >= min_step:
            break
        k = 0 if pts[i][1] == pts[(i + 1) % n][1] else 1    # the coordinate side i spans
        if lengths[(i - 1) % n] >= lengths[(i + 1) % n]:
            target, moving = pts[i][k], ((i + 1) % n, (i + 2) % n)
        else:
            target, moving = pts[(i + 1) % n][k], ((i - 1) % n, i)
        for j in moving:
            q = list(pts[j])
            q[k] = target
            pts[j] = tuple(q)
        pts = _clean(pts)
    if len(pts) < 4 or abs(_area(pts)) == 0:
        xs, ys = [p[0] for p in original], [p[1] for p in original]
        return [(min(xs), min(ys)), (max(xs), min(ys)), (max(xs), max(ys)), (min(xs), max(ys))]
    return pts


def outline(points: np.ndarray, floor_z: float, height: float, cameras_xy, unit: float = 1.0,
            angle: float | None = None, centre_xy=None) -> dict | None:
    """The room's outline from a z-up cloud in scene units.

    unit: scene units per metre (1.0 for a cloud in metres; without a metric
    scale pass the best guess, the thresholds are all in metres times this).
    angle: the room's axis angle in the xy plane (radians); found from the
    wall points when not given. Returns {"angle", "polygons": [[[x, y], ...]],
    "area", "rect": {center, axis_u, axis_v, half_u, half_v}, "parts",
    "dropped_area"} in scene units, or None when nothing was seen."""
    from scipy import ndimage

    P = np.asarray(points, float)
    cams = np.asarray(cameras_xy, float).reshape(-1, 2)
    cell = CELL_M * unit
    z = P[:, 2] - floor_z
    level = (np.abs(z) < LEVEL_BAND_M * unit) | (np.abs(z - height) < LEVEL_BAND_M * unit)
    wall = (z > WALL_BAND[0] * height) & (z < WALL_BAND[1] * height)
    low = (z > LEVEL_BAND_M * unit) & (z <= WALL_BAND[0] * height)      # beds, counters, tables: floor beneath
    if angle is None:
        angle = room_angle(P[wall, :2] / unit) if wall.sum() > 50 else 0.0
    c, s = np.cos(angle), np.sin(angle)
    u_axis, v_axis = np.array([c, s]), np.array([-s, c])
    to_uv = lambda xy: np.column_stack([xy @ u_axis, xy @ v_axis])   # noqa: E731
    uv, cam_uv = to_uv(P[:, :2]), to_uv(cams) if len(cams) else np.zeros((0, 2))
    # A margin wider than a door: closing the space at its doorways grows it
    # that far before shrinking it back, and must not run off the grid.
    pad = (DOOR_M + 0.5) * unit
    lo = np.vstack([uv[level | wall | low], cam_uv]).min(axis=0) - pad
    hi = np.vstack([uv[level | wall | low], cam_uv]).max(axis=0) + pad
    shape = tuple((np.ceil((hi - lo) / cell)).astype(int)[::-1] + 1)     # rows = v, cols = u

    def cells(q):
        ij = ((q - lo) / cell).astype(int)
        return ij[:, 1], ij[:, 0]

    seen = np.zeros(shape, bool)
    seen[cells(uv[level])] = True
    seen[cells(uv[low])] = True
    blocked = np.zeros(shape, np.int32)
    np.add.at(blocked, cells(uv[wall]), 1)
    blocked = blocked >= WALL_MIN_POINTS
    stood = np.zeros(shape, bool)
    if len(cam_uv):
        stood[cells(cam_uv)] = True
        stood = ndimage.binary_dilation(stood, _disk(CAMERA_RADIUS_M / CELL_M))
    open_ = ndimage.binary_closing(seen, _disk(CLOSE_M / CELL_M)) | stood
    open_ &= ~(blocked & ~stood)
    open_ = ndimage.binary_fill_holes(open_)        # under furniture, behind a table leg

    labels, n = ndimage.label(open_)
    if n == 0:
        return None
    if len(cam_uv):
        r, cc = cells(cam_uv)
        start = set(np.unique(labels[r, cc])) - {0}
    else:
        start = {1 + int(np.argmax(np.bincount(labels.ravel())[1:]))}
    space = np.isin(labels, list(start))

    # Cut at doors; keep the parts the phone was in; grow them back.
    # The cores are the space shrunk by half a door: every doorway closes. (Its
    # opening would not do: grown back from both sides, two rooms meet again on
    # a wall one cell thick.)
    door = _disk(DOOR_M / CELL_M / 2)
    cores = ndimage.binary_erosion(space, door)
    parts, m = ndimage.label(cores)
    keep = np.zeros(m + 1, bool)
    # Every cell (and every camera: a phone in a doorway stands between) belongs to the core nearest it.
    idx = ndimage.distance_transform_edt(~cores, return_distances=False, return_indices=True) if m else None
    if len(cam_uv) and m:
        r, cc = cells(cam_uv)
        owner = parts[idx[0][r, cc], idx[1][r, cc]]
        counts = np.bincount(owner, minlength=m + 1)
        keep[1:] = counts[1:] >= max(1, MIN_PART_CAMERAS * len(cam_uv))
    elif m:
        keep[1 + int(np.argmax(np.bincount(parts.ravel())[1:]))] = True
    if m:
        owner_of = np.where(space, parts[idx[0], idx[1]], 0)
        sizes = np.bincount(owner_of.ravel(), minlength=m + 1) * CELL_M ** 2
        walked = np.isin(owner_of, np.flatnonzero(keep))
        touching = np.unique(owner_of[ndimage.binary_dilation(walked, np.ones((5, 5), bool)) & ~walked])
        for k in touching:
            if k and sizes[k] < MIN_ROOM_M2:      # next to where the phone was, but not a room of its own
                keep[k] = True
    if not keep.any():
        kept = space
    else:
        kept = keep[owner_of] & space
        # the doorways between two kept parts belong to the room too
        kept = ndimage.binary_closing(kept, door) & space
    dropped = (space & ~kept).sum() * CELL_M ** 2

    # Square off noise: features narrower than a step are not the room's shape.
    step = max(1, int(round(MIN_STEP_M / CELL_M)))
    sq = np.ones((step, step), bool)
    kept = ndimage.binary_closing(ndimage.binary_opening(kept, sq), sq)
    kept = ndimage.binary_fill_holes(kept)

    polygons = []
    for loop in _trace(kept):
        corners = _corners(loop)[::-1]              # traced clockwise; holes come out the other way
        if _area(corners) <= 0:                     # a hole
            continue
        poly = simplify(corners, MIN_STEP_M / CELL_M)
        if abs(_area(poly)) * CELL_M ** 2 < 1.0:    # a scrap under a square metre
            continue
        uvp = lo + np.asarray(poly, float) * cell
        xy = uvp[:, :1] * u_axis + uvp[:, 1:] * v_axis
        polygons.append(xy.round(4).tolist())
    if not polygons:
        return None
    all_uv = to_uv(np.vstack([np.asarray(p) for p in polygons]))
    lo_uv, hi_uv = all_uv.min(axis=0), all_uv.max(axis=0)
    mid = (lo_uv + hi_uv) / 2
    area = sum(abs(_area(p)) for p in polygons)
    return {"angle": angle, "polygons": polygons, "area": area,
            "parts": int(keep.sum()), "dropped_area": dropped * unit * unit,
            "rect": {"center": (mid[0] * u_axis + mid[1] * v_axis).round(4).tolist(),
                     "axis_u": u_axis.tolist(), "axis_v": v_axis.tolist(),
                     "half_u": float((hi_uv - lo_uv)[0] / 2), "half_v": float((hi_uv - lo_uv)[1] / 2)}}


def inside(polygons, x: float, y: float) -> bool:
    """Whether a point lies in any of the outline's polygons."""
    hit = False
    for poly in polygons:
        n = len(poly)
        for i in range(n):
            (x0, y0), (x1, y1) = poly[i], poly[(i + 1) % n]
            if (y0 > y) != (y1 > y) and x < x0 + (y - y0) * (x1 - x0) / (y1 - y0):
                hit = not hit
    return hit
