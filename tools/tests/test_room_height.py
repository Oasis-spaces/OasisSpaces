"""The room's height when the phone never filmed the ceiling, and when it filmed
the ceiling but not the floor.

    python3 tools/tests/test_room_height.py
"""

import contextlib
import io
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import classify_shapes  # noqa: E402

U = 10.0                       # solve units per metre
W, D = 4.4, 3.6                # the room, metres; its floor is at z = 0


def level(z, which):
    """A floor or ceiling level shapes.py found, z metres up."""
    return {"kind": "floor_or_ceiling", "level": which, "normal": [0.0, 0.0, 1.0],
            "center": [W / 2 * U, D / 2 * U, z * U], "axis_a": [1.0, 0.0, 0.0], "axis_b": [0.0, 1.0, 0.0],
            "half_a": W / 2 * U, "half_b": D / 2 * U, "points": 30000}


def wall(along, offset, bottom, top, centre=None, half=None):
    """A vertical wall seen from `bottom` to `top` metres: along x at y = offset, or along y at x = offset."""
    z = (bottom + top) / 2 * U
    if along == "x":
        c, n, a, h = [(centre or W / 2) * U, offset * U, z], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0], half or W / 2
    else:
        c, n, a, h = [offset * U, (centre or D / 2) * U, z], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], half or D / 2
    return {"kind": "wall", "normal": n, "center": c, "axis_a": a, "axis_b": [0.0, 0.0, 1.0],
            "half_a": h * U, "half_b": (top - bottom) / 2 * U, "points": 20000}


def walls_seen(tops, bottoms=(0.5, 0.6, 0.4, 0.7)):
    """The room's four walls, each seen between its bottom and top (furniture hides the lower part,
    the frame's top edge cuts off the upper part)."""
    return [wall("x", 0.0, bottoms[0], tops[0]), wall("x", D, bottoms[1], tops[1]),
            wall("y", 0.0, bottoms[2], tops[2]), wall("y", W, bottoms[3], tops[3])]


def close(a, b, tol=1e-6):
    return abs(a - b) < tol


def test_without_a_ceiling_the_room_is_as_tall_as_its_walls_were_seen_to_reach():
    planes = [level(0.0, "floor"), *walls_seen(tops=(1.75, 1.80, 2.10, 2.20))]
    _, _, ctx = classify_shapes.classify_planes(planes)
    # Each top is a lower bound on the ceiling: the median (1.95 m) is too low.
    # The second-highest top is taken.
    assert close(ctx.room_height, 2.10 * U), ctx.room_height / U
    assert ctx.height_kind == "wall tops"


def test_one_stray_tall_plane_does_not_raise_the_room():
    # A building across the street, seen through a window up to 6 m.
    planes = [level(0.0, "floor"), *walls_seen(tops=(1.75, 1.80, 2.10, 2.20)),
              wall("x", -3.0, 1.0, 6.0, half=0.5)]
    _, _, ctx = classify_shapes.classify_planes(planes)
    assert close(ctx.room_height, 2.20 * U), ctx.room_height / U


def test_with_two_walls_the_higher_top_is_taken():
    # The lower of two tops would undercut their median (here 2.00 m), which
    # the height used to be measured to.
    planes = [level(0.0, "floor"), wall("x", 0.0, 0.5, 1.8), wall("y", 0.0, 0.5, 2.2)]
    _, _, ctx = classify_shapes.classify_planes(planes)
    assert close(ctx.room_height, 2.20 * U), ctx.room_height / U
    assert ctx.height_source.startswith("higher of 2 wall tops"), ctx.height_source
    # With three, the second-highest, which is never below their median either.
    planes.append(wall("x", D, 0.5, 2.0))
    _, _, ctx = classify_shapes.classify_planes(planes)
    assert close(ctx.room_height, 2.00 * U), ctx.room_height / U
    # A stray plane as one of two: the metric cap bounds it.
    planes = [level(0.0, "floor"), wall("x", 0.0, 0.5, 1.8), wall("x", -3.0, 1.0, 6.0, half=0.5)]
    _, _, ctx = classify_shapes.classify_planes(planes, U)
    assert close(ctx.room_height, 3.50 * U) and ctx.height_kind == "prior"


def test_with_a_metric_scale_an_unseen_ceiling_is_kept_between_2_4_and_3_5_m():
    low = [level(0.0, "floor"), *walls_seen(tops=(1.75, 1.80, 2.10, 2.20))]
    _, _, ctx = classify_shapes.classify_planes(low, U)
    assert close(ctx.room_height, 2.40 * U) and ctx.height_kind == "prior"
    assert "2.40 m assumed" in ctx.height_source
    high = [level(0.0, "floor"), *walls_seen(tops=(1.75, 4.0, 4.5, 2.20))]
    _, _, ctx = classify_shapes.classify_planes(high, U)
    assert close(ctx.room_height, 3.50 * U) and ctx.height_kind == "prior"
    seen = [level(0.0, "floor"), *walls_seen(tops=(2.0, 2.6, 2.8, 2.2))]
    _, _, ctx = classify_shapes.classify_planes(seen, U)
    assert close(ctx.room_height, 2.60 * U) and ctx.height_kind == "wall tops"


def test_a_ceiling_level_counts_when_no_floor_level_was_found():
    # A level clip: the ceiling is in view, the floor near the walls is not.
    planes = [level(2.60, "ceiling"),
              *walls_seen(tops=(2.2, 2.3, 2.1, 2.25), bottoms=(0.3, 0.4, 0.35, 0.45))]
    labels, _, ctx = classify_shapes.classify_planes(planes)
    assert close(ctx.floor_z, 0.375 * U)                       # the walls' bottoms stand in for the floor
    assert close(ctx.room_height, (2.60 - 0.375) * U), ctx.room_height / U
    assert ctx.height_kind == "ceiling" and labels[0] == "ceiling"


def test_a_measured_ceiling_is_kept_even_below_the_assumed_minimum():
    planes = [level(0.0, "floor"), level(2.30, "ceiling"), *walls_seen(tops=(2.2, 2.25, 2.1, 2.0))]
    _, _, ctx = classify_shapes.classify_planes(planes, U)
    assert close(ctx.room_height, 2.30 * U) and ctx.height_kind == "ceiling"


def test_a_low_level_without_a_floor_is_a_cabinet_underside_not_the_ceiling():
    # 1.2 and 1.7 m above the walls' bottoms (0.375 m): with the 2.40 m room
    # the height rule would call the first a surface, built as a slab, and
    # the second the ceiling. Both are the same thing, left unbuilt.
    for z in (1.575, 2.075):
        planes = [level(z, "ceiling"),
                  *walls_seen(tops=(1.9, 2.0, 1.8, 1.85), bottoms=(0.3, 0.4, 0.35, 0.45))]
        labels, reasons, ctx = classify_shapes.classify_planes(planes, U)
        assert ctx.height_kind == "prior" and close(ctx.room_height, 2.40 * U)
        assert labels[0] == "surface" and planes[0].get("build") is False, (z, labels[0], reasons[0])
        assert "not the ceiling" in reasons[0]
        assert all(p.get("build", True) for p in planes[1:])


def test_with_a_ceiling_but_no_floor_the_floor_is_at_least_2_4_m_below_it():
    # The walls were seen down to 0.375 m; furniture hides their feet, so the
    # floor is lower still, and the room at least the 2.40 m minimum.
    walls = walls_seen(tops=(2.2, 2.3, 2.1, 2.25), bottoms=(0.3, 0.4, 0.35, 0.45))
    labels, _, ctx = classify_shapes.classify_planes([level(2.60, "ceiling"), *walls], U)
    assert close(ctx.floor_z, 0.20 * U), ctx.floor_z / U
    assert close(ctx.room_height, 2.40 * U) and ctx.height_kind == "prior"
    assert labels[0] == "ceiling" and "2.40 m below the ceiling" in ctx.floor_source
    # A ceiling far enough above the walls' bottoms is left as measured.
    labels, _, ctx = classify_shapes.classify_planes([level(2.90, "ceiling"), *walls], U)
    assert close(ctx.floor_z, 0.375 * U) and close(ctx.room_height, 2.525 * U)
    assert ctx.height_kind == "ceiling" and labels[0] == "ceiling"
    # So is one over a floor level, even below 2.40 m (as above).
    _, _, ctx = classify_shapes.classify_planes([level(0.0, "floor"), level(2.30, "ceiling"), *walls], U)
    assert close(ctx.floor_z, 0.0) and close(ctx.room_height, 2.30 * U)


def test_a_level_turned_down_as_the_ceiling_is_written_as_not_built():
    with tempfile.TemporaryDirectory() as tmp:
        space = Path(tmp)
        (space / "shapes.json").write_text(json.dumps({
            "planes": [level(2.075, "ceiling"),
                       *walls_seen(tops=(1.9, 2.0, 1.8, 1.85), bottoms=(0.3, 0.4, 0.35, 0.45))],
            "boxes": []}))
        (space / "densify.json").write_text(json.dumps({"colmap_units_per_metre": U}))
        with contextlib.redirect_stdout(io.StringIO()):
            assert classify_shapes.main([str(space)]) == 0
        data = json.loads((space / "shapes.json").read_text())
    underside = data["planes"][0]
    assert underside["label"] == "surface" and underside["build"] is False, underside
    assert data["room_level"]["height_source"] == "prior"


def test_the_room_level_says_where_its_height_came_from_and_a_desk_seen_from_above_is_built():
    desk = {"min": [3.0 * U, 2.9 * U, 0.72 * U], "max": [4.2 * U, 3.5 * U, 0.76 * U], "points": 5000,
            "source": "detected", "detected": "desk", "color": [120, 90, 60]}
    with tempfile.TemporaryDirectory() as tmp:
        space = Path(tmp)
        (space / "shapes.json").write_text(json.dumps({
            "planes": [level(0.0, "floor"), *walls_seen(tops=(1.75, 1.80, 2.10, 2.20))], "boxes": [desk]}))
        (space / "densify.json").write_text(json.dumps({"colmap_units_per_metre": U}))
        with contextlib.redirect_stdout(io.StringIO()):
            assert classify_shapes.main([str(space), "--no-finish"]) == 0
        data = json.loads((space / "shapes.json").read_text())
    assert data["room_level"]["height_source"] == "prior"
    assert close(data["room_level"]["height"], 2.40 * U)
    # Only its top was filmed: 0.72 m up is 37% of a 1.95 m room (floating), 30% of 2.40 m.
    box = data["boxes"][0]
    assert box["build"] and box["label"] == "table", box["reason"]
    walls = [p for p in data["planes"] if p["label"] == "wall"]
    assert len(walls) == 4
    for p in walls:                                            # built from the floor to the room height
        assert close(p["center"][2] - p["half_b"], 0.0) and close(p["center"][2] + p["half_b"], 2.40 * U)


if __name__ == "__main__":
    for name, test in sorted(globals().items()):
        if name.startswith("test_"):
            test()
            print("ok", name)
