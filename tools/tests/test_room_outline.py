"""The room's outline from above: shapes a rectangle cannot be, and doors.

Rooms are drawn on a 5 cm grid as their floor and ceiling (points on both
levels) and their walls (points up the boundary, with gaps for doors), in
scene units of 2.5 per metre and turned to the axes as stage 3 leaves them.

    python3 tools/tests/test_room_outline.py      (or pytest tools/tests)
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(ROOT / "tools"))
import classify_shapes as cs  # noqa: E402
import room_outline as ro  # noqa: E402

UNIT = 2.5
HEIGHT = 2.6
G = 0.05


def cloud(rooms, doors=(), seen=None, seed=0):
    """Points of rooms, each a list of rectangles (x0, y0, x1, y1 in metres)
    walled around their union. `doors` are (x0, y0, x1, y1) stretches of wall
    left open; `seen` limits the floor and ceiling points to these rectangles
    (what the camera saw)."""
    rng = np.random.default_rng(seed)
    rects = [r for room in rooms for r in room]
    lo = np.min([r[:2] for r in rects], axis=0) - 1
    hi = np.max([r[2:] for r in rects], axis=0) + 1
    xs, ys = np.arange(lo[0], hi[0], G) + G / 2, np.arange(lo[1], hi[1], G) + G / 2
    X, Y = np.meshgrid(xs, ys)
    inside = np.zeros(X.shape, bool)
    for x0, y0, x1, y1 in rects:
        inside |= (X > x0) & (X < x1) & (Y > y0) & (Y < y1)
    visible = inside.copy()
    if seen is not None:
        visible &= np.any([(X > x0) & (X < x1) & (Y > y0) & (Y < y1) for x0, y0, x1, y1 in seen], axis=0)
    pts = []
    for z in (0.0, HEIGHT):
        xy = np.column_stack([X[visible], Y[visible]]) + rng.uniform(-G / 2, G / 2, (visible.sum(), 2))
        pts.append(np.column_stack([xy, np.full(len(xy), z)]))
    edge = np.zeros(X.shape, bool)
    for room in rooms:
        own = np.any([(X > x0) & (X < x1) & (Y > y0) & (Y < y1) for x0, y0, x1, y1 in room], axis=0)
        pad = np.pad(own, 1)
        edge |= own & ~(pad[:-2, 1:-1] & pad[2:, 1:-1] & pad[1:-1, :-2] & pad[1:-1, 2:])
    edge &= visible | (seen is None)
    for x0, y0, x1, y1 in doors:
        edge &= ~((X > x0 - G) & (X < x1 + G) & (Y > y0 - G) & (Y < y1 + G))
    exy = np.column_stack([X[edge], Y[edge]])
    for z in np.linspace(0.05, HEIGHT - 0.05, 20):
        pts.append(np.column_stack([exy + rng.normal(0, 0.01, exy.shape), np.full(len(exy), z)]))
    return np.vstack(pts) * UNIT


def walk(points_m):
    return np.asarray(points_m, float) * UNIT


def area_m2(result):
    return result["area"] / UNIT ** 2


def corners(result):
    return [len(p) for p in result["polygons"]]


L_ROOM = [(0, 0, 6, 3), (0, 3, 3, 5)]          # 6 x 3 plus 3 x 2: 24 m²


def test_a_rectangle_stays_a_rectangle():
    r = ro.outline(cloud([[(0, 0, 5, 4)]]), 0.0, HEIGHT * UNIT, walk([[1, 1], [4, 3], [2, 2]]), unit=UNIT, angle=0.0)
    assert corners(r) == [4], r["polygons"]
    assert abs(area_m2(r) - 20) < 1.0, area_m2(r)
    assert abs(r["rect"]["half_u"] / UNIT - 2.5) < 0.15 and abs(r["rect"]["half_v"] / UNIT - 2.0) < 0.15


def test_an_l_shaped_room_is_an_l_not_its_box():
    r = ro.outline(cloud([L_ROOM]), 0.0, HEIGHT * UNIT, walk([[1, 1], [5, 1.5], [1.5, 4]]), unit=UNIT, angle=0.0)
    assert corners(r) == [6], r["polygons"]
    assert abs(area_m2(r) - 24) < 1.2, area_m2(r)           # the box would be 30


def test_a_room_seen_through_a_door_is_left_out():
    # Room A (5 x 4) and room B beyond its right wall, joined by a 0.9 m door;
    # the phone only ever stood in A, B's floor was seen through the door.
    rects = [[(0, 0, 5, 4)], [(5, 0.5, 8, 3.5)]]
    door = [(4.95, 1.6, 5.05, 2.5)]
    r = ro.outline(cloud(rects, door), 0.0, HEIGHT * UNIT, walk([[1, 1], [3, 3], [2, 2], [4, 2]]),
                   unit=UNIT, angle=0.0)
    assert corners(r) == [4], r["polygons"]
    assert abs(area_m2(r) - 20) < 1.2, area_m2(r)
    assert r["dropped_area"] / UNIT ** 2 > 5


def test_two_rooms_walked_through_are_both_kept():
    rects = [[(0, 0, 5, 4)], [(5, 0.5, 8, 3.5)]]
    door = [(4.95, 1.6, 5.05, 2.5)]
    path = walk([[1, 1], [3, 3], [4.5, 2], [5.5, 2], [6.5, 1.5], [7.5, 3]] * 3)
    r = ro.outline(cloud(rects, door), 0.0, HEIGHT * UNIT, path, unit=UNIT, angle=0.0)
    assert abs(area_m2(r) - 29) < 1.5, area_m2(r)
    assert r["parts"] == 2


def test_floor_under_a_bed_is_still_the_room():
    pts = cloud([[(0, 0, 5, 4)]], seen=[(0, 0, 5, 4)])
    bed = (np.c_[pts[:, 0], pts[:, 1]] / UNIT)
    under = (bed[:, 0] > 3) & (bed[:, 1] > 2) & (pts[:, 2] < UNIT * HEIGHT / 2)
    pts[under & (pts[:, 2] < 0.01), 2] = 0.55 * UNIT          # the floor there is a mattress top
    pts = pts[~(under & (pts[:, 2] < 0.01))]
    r = ro.outline(pts, 0.0, HEIGHT * UNIT, walk([[1, 1], [2, 2]]), unit=UNIT, angle=0.0)
    assert corners(r) == [4] and abs(area_m2(r) - 20) < 1.0, (r["polygons"], area_m2(r))


def test_simplify_folds_away_short_steps_and_keeps_alcoves():
    # a 10 x 10 square with a 1-cell bump (noise) and a 4 x 3 alcove
    poly = [(0, 0), (10, 0), (10, 4), (11, 4), (11, 5), (10, 5), (10, 10), (6, 10), (6, 13), (2, 13), (2, 10), (0, 10)]
    out = ro.simplify(poly, 2)
    assert len(out) == 8, out
    assert ro._area(out) > 0


def test_the_classifier_drops_walls_outside_the_outline_and_closes_along_it():
    outline = {"polygons": [(np.array([[0, 0], [6, 0], [6, 3], [3, 3], [3, 5], [0, 5]], float) * UNIT).tolist()]}
    margin = cs.outline_margin(outline, UNIT)
    wall = lambda c, a, half: {"center": [c[0] * UNIT, c[1] * UNIT, 1.3 * UNIT], "axis_a": a, "half_a": half * UNIT,  # noqa: E731
                               "normal": [-a[1], a[0], 0.0]}
    assert cs.wall_near_outline(wall((3, 0), [1, 0, 0], 3), outline, margin)
    assert not cs.wall_near_outline(wall((9, 1.5), [0, 1, 0], 1.5), outline, margin)   # a wall in the next room
    room = {"center": [3 * UNIT, 2.5 * UNIT], "axis_u": [1.0, 0.0], "axis_v": [0.0, 1.0],
            "half_u": 3 * UNIT, "half_v": 2.5 * UNIT}
    # Measured: the bottom wall with a 0.9 m door in it, and the left wall. The rest was never faced.
    walls = [{"plane": {}, "along": "u", "offset": -2.5 * UNIT, "lo": -3 * UNIT, "hi": -0.5 * UNIT},
             {"plane": {}, "along": "u", "offset": -2.5 * UNIT, "lo": 0.4 * UNIT, "hi": 3 * UNIT},
             {"plane": {}, "along": "v", "offset": -3 * UNIT, "lo": -2.5 * UNIT, "hi": 2.5 * UNIT}]
    notes = cs.close_outline(walls, room, outline, lambda d: f"{d / UNIT:.1f} m", UNIT)
    added = walls[3:]
    assert len(added) == 4, notes                                          # right, the step, its top, the top
    assert all(w["plane"]["source"] == "inferred" for w in added)
    lengths = sorted(round((w["hi"] - w["lo"]) / UNIT, 2) for w in added)
    assert lengths == [2.0, 3.0, 3.0, 3.0], lengths
    # the door's gap in the bottom wall stays open
    assert not any(w["along"] == "u" and abs(w["offset"] + 2.5 * UNIT) < 0.1 for w in added)
    # every inferred wall faces into the room
    for w in added:
        n = w["plane"]["normal"]
        mid = (w["lo"] + w["hi"]) / 2
        x, y = cs.from_room(room, *((mid, w["offset"]) if w["along"] == "u" else (w["offset"], mid)))
        assert cs.inside_outline(outline["polygons"], x + 0.2 * UNIT * n[0], y + 0.2 * UNIT * n[1])


def test_the_floor_mesh_covers_the_outline_only():
    import mixed_scene as ms
    from surface_fill import Surface

    class FakeRoom:
        centre = np.array([3.0, 2.5]) * UNIT
        half = np.array([3.0, 2.5]) * UNIT
        metre = UNIT
        world = np.eye(3)

    frame = ms.Frame(FakeRoom(), 0.0)
    cell = 0.1 * UNIT
    floor = Surface("floor", np.array([0.0, 0.0, 0.0]), np.array([1.0, 0, 0]), np.array([0, 1.0, 0]),
                    np.array([0, 0, 1.0]), 60, 50, cell)
    polygons = [(np.array([[0, 0], [6, 0], [6, 3], [3, 3], [3, 5], [0, 5]], float) * UNIT).tolist()]
    positions, _normals, uvs, indices = ms.outline_mesh(floor, frame, polygons)
    tri = positions[indices.reshape(-1, 3)]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1).sum()
    assert abs(area - 24) < 1e-3, area                              # viewer metres
    assert uvs.min() >= 0 and uvs.max() <= 1
    assert abs(ms.surface_inside(floor, polygons).mean() - 24 / 30) < 0.01


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print("ok", name)
