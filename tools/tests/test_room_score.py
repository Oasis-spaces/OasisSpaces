"""The measured room score and what moves it.

    python3 tools/tests/test_room_score.py
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "pipeline"))
import placement as pl  # noqa: E402
import room_score as rs  # noqa: E402
from semantics import Vocabulary  # noqa: E402

UNITS = 9.0
VOCAB = Vocabulary([{"name": "bed", "role": "furniture", "build_as": "bed"},
                    {"name": "almirah", "role": "storage", "build_as": "wardrobe"},
                    {"name": "wood stove", "role": "furniture", "build_as": "block"},
                    {"name": "pillow", "role": "on_furniture", "build_as": "block"},
                    {"name": "door", "role": "fixture", "build_as": None}])


def look_at(position, target):
    forward = np.asarray(target, float) - np.asarray(position, float)
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    R = np.stack([right, down, forward])
    return R, -R @ np.asarray(position, float)


def view(position, target):
    R, t = look_at(position, target)
    return {"R": R, "t": t, "world": np.eye(3), "fx": 1000.0, "fy": 1000.0, "cx": 540.0, "cy": 960.0,
            "width": 1080, "height": 1920}


def room():
    return {"planes": [], "boxes": [], "room": {"center": [0.0, 0.0]},
            "room_level": {"floor_z": 0.0, "height": 2.6 * UNITS},
            "cameras": [[0.0, -8.0], [3.0, -8.0], [-3.0, -8.0]]}


BED = {"min": [-7.0, 0.0, 0.0], "max": [7.0, 17.0, 4.5], "label": "bed", "detected": "bed", "build": True}
CUPBOARD = {"min": [-20.0, 10.0, 0.0], "max": [-11.0, 15.0, 18.0], "label": "wardrobe", "detected": "almirah", "build": True}


def evidence_from(true_box: dict, positions) -> dict:
    """Outlines that are exactly the true box's silhouettes, from a few cameras."""
    frames = {}
    for n, position in enumerate(positions):
        v = view(position, [(true_box["min"][0] + true_box["max"][0]) / 2, (true_box["min"][1] + true_box["max"][1]) / 2, 2.0])
        shape = (v["height"] // pl.GRID, v["width"] // pl.GRID)
        frames[f"frame_{n:05d}.jpg"] = {"mask": pl.silhouette(np.array(true_box["min"]), np.array(true_box["max"]), v, shape), **v, "window": None}
    return {"label": true_box["detected"], "frames": frames, "unseen": {}}


def evidence(bed_views=((0.0, -20.0, 12.0), (8.0, -18.0, 12.0), (-8.0, -18.0, 12.0)),
             cupboard_views=((-5.0, -10.0, 12.0), (0.0, -12.0, 12.0))):
    table = {"bed": evidence_from(BED, bed_views), "almirah": evidence_from(CUPBOARD, cupboard_views)}
    return lambda label: table.get(label, {"frames": {}, "unseen": {}})


def test_a_room_whose_pieces_cover_their_outlines_scores_high_and_dropping_one_lowers_it():
    shapes = room()
    shapes["boxes"] = [dict(BED), dict(CUPBOARD)]
    full = rs.room_score(Path("."), shapes, evidence(), VOCAB)
    assert full["agreement"] > 0.9 and full["collision"] == 0 and full["walk"] == 0
    assert full["pieces"]["bed"]["box"] == "B0" and full["pieces"]["almirah"]["box"] == "B1"
    without = dict(shapes, boxes=[dict(BED), dict(CUPBOARD, build=False)])
    dropped = rs.room_score(Path("."), without, evidence(), VOCAB)
    assert dropped["pieces"]["almirah"]["agreement"] == 0 and dropped["pieces"]["almirah"]["box"] is None
    assert rs.compare(full, dropped) == "worse" and rs.compare(dropped, full) == "better"
    assert rs.compare(full, dict(full, score=full["score"] + 0.01)) == "same"
    assert "almirah 0.00 (no box)" in rs.describe(dropped) and "bed" in rs.describe(full)


def test_a_misplaced_piece_scores_lower_than_a_measured_one():
    shapes = room()
    shifted = dict(CUPBOARD, min=[-2.0, 10.0, 0.0], max=[7.0, 15.0, 18.0])
    shapes["boxes"] = [dict(BED), shifted]
    moved = rs.room_score(Path("."), shapes, evidence(), VOCAB)
    right = rs.room_score(Path("."), dict(shapes, boxes=[dict(BED), dict(CUPBOARD)]), evidence(), VOCAB)
    assert moved["pieces"]["almirah"]["agreement"] < 0.3 < right["pieces"]["almirah"]["agreement"]
    assert rs.compare(right, moved) == "worse"


def test_overlapping_pieces_and_a_walk_through_a_cupboard_are_penalised():
    shapes = room()
    twin = dict(CUPBOARD, min=[-15.5, 10.0, 0.0], max=[-6.5, 15.0, 18.0])        # half inside the first
    shapes["boxes"] = [dict(CUPBOARD), twin]
    result = rs.room_score(Path("."), shapes, evidence(), VOCAB)
    assert abs(result["collision"] - 0.5) < 1e-6 and "overlap by 50%" in rs.describe(result)
    assert abs(result["score"] - (result["agreement"] - rs.COLLISION_WEIGHT * 0.5)) < 1e-6
    # a camera standing inside the cupboard's footprint: a third of the walk
    shapes = room()
    shapes["boxes"] = [dict(CUPBOARD)]
    shapes["cameras"] = [[-15.0, 12.0], [3.0, -8.0], [-3.0, -8.0]]
    walked = rs.room_score(Path("."), shapes, evidence(), VOCAB)
    assert abs(walked["walk"] - 1 / 3) < 1e-3 and "33% of the walk" in rs.describe(walked)
    # over a bed is fine: nobody is inside a piece that low
    shapes["boxes"] = [dict(BED)]
    shapes["cameras"] = [[0.0, 5.0]]
    assert rs.room_score(Path("."), shapes, evidence(), VOCAB)["walk"] == 0


def test_without_outlines_only_the_penalties_count_and_small_things_are_not_pieces():
    shapes = room()
    shapes["boxes"] = [dict(BED), {"min": [0, 0, 4.5], "max": [3, 3, 6], "label": "block", "detected": "pillow", "build": True}]
    none = rs.room_score(Path("."), shapes, lambda label: {"frames": {}, "unseen": {}}, VOCAB)
    assert none["agreement"] is None and none["score"] == 0 and none["pieces"] == {}
    assert "no piece with outlines" in rs.describe(none)
    # the pillow box is not a piece: it neither collides nor stands for a name
    assert [i for i, _ in rs.built_pieces(shapes)] == [0]
    assert rs.boxes_of(shapes, "pillow", VOCAB) == []


def test_a_piece_standing_where_the_frames_show_the_door_is_penalised():
    door = {"min": [8.0, 15.0, 0.0], "max": [15.0, 15.6, 18.0], "label": "door", "detected": "door"}
    # the door is looked at from beside the bed, not through it (an outline stops where a piece covers it)
    table = {"bed": evidence_from(BED, ((0.0, -20.0, 12.0), (8.0, -18.0, 12.0), (-8.0, -18.0, 12.0))),
             "door": evidence_from(door, ((11.0, 2.0, 12.0), (9.5, 0.0, 12.0), (13.0, 4.0, 12.0)))}
    evidence_for = lambda label: table.get(label, {"frames": {}, "unseen": {}})
    shapes = room()
    in_doorway = dict(CUPBOARD, min=[8.5, 12.0, 0.0], max=[14.5, 15.5, 17.0])   # a "wardrobe" standing in the doorway
    shapes["boxes"] = [dict(BED), in_doorway]
    result = rs.room_score(Path("."), shapes, evidence_for, VOCAB)
    assert result["doorways"]["B1"] > 0.7
    assert "B0" not in result["doorways"] and result["doorway"] == result["doorways"]["B1"]
    assert "B1 stands in a doorway" in rs.describe(result)
    clear = rs.room_score(Path("."), dict(shapes, boxes=[dict(BED), dict(CUPBOARD)]), evidence_for, VOCAB)
    assert clear["doorway"] == 0 and rs.compare(clear, result) == "worse"
    # one frame is enough when most of the piece lies in the door; a slight overlap in one is not
    one = {"door": evidence_from(door, ((11.0, 2.0, 12.0),))}
    filling = dict(CUPBOARD, min=list(door["min"]), max=list(door["max"]))        # exactly in the doorway
    once = rs.room_score(Path("."), dict(shapes, boxes=[dict(BED), filling]), lambda l: one.get(l, {"frames": {}, "unseen": {}}), VOCAB)
    assert once["doorways"]["B1"] > 0.9
    beside = dict(CUPBOARD, min=[13.0, 12.0, 0.0], max=[19.0, 15.5, 17.0])      # mostly past the door's edge
    aside = rs.room_score(Path("."), dict(shapes, boxes=[dict(BED), beside]), lambda l: one.get(l, {"frames": {}, "unseen": {}}), VOCAB)
    assert aside["doorway"] < once["doorway"]


def test_a_tracked_object_the_detector_part_named_a_door_counts_as_one():
    import tempfile

    import tracking

    door = {"min": [8.0, 15.0, 0.0], "max": [15.0, 15.6, 18.0]}
    positions = ((11.0, 2.0, 12.0), (9.5, 0.0, 12.0), (13.0, 4.0, 12.0))
    frames = [f"frame_{n:05d}.jpg" for n in range(3)]
    views = {}
    masks, seen = {}, {}
    for name, position in zip(frames, positions):
        v = view(position, [11.5, 15.3, 2.0])
        views[name] = v
        shape = (v["height"] // pl.GRID, v["width"] // pl.GRID)
        sil = pl.silhouette(np.array(door["min"]), np.array(door["max"]), v, shape)
        from PIL import Image
        small = np.asarray(Image.fromarray(sil.astype(np.uint8) * 255).resize((256, 256), Image.NEAREST)) > 0
        masks[tracking.Tracks.key(7, name)] = np.packbits(small)
        seen[name] = {"box": tracking.mask_box(small, v["width"], v["height"]), "score": 0.95}
    tracks = tracking.Tracks(frames, (1080, 1920), [
        {"id": 7, "label": "almirah", "votes": {"almirah": 3.7, "door": 1.9}, "seed": frames[0], "frames": seen}], masks)
    shapes = room()
    shapes["boxes"] = [dict(BED), dict(CUPBOARD, min=list(door["min"]), max=list(door["max"]))]
    with tempfile.TemporaryDirectory() as tmp:
        space = Path(tmp)
        tracks.save(space / "workspace" / "tracks")
        kept = pl.views_of
        pl.views_of = lambda space_, names: {n: views[n] for n in names if n in views}
        try:
            by_votes = rs.doorways_by_votes(space, shapes, VOCAB)
            result = rs.room_score(space, shapes, evidence(), VOCAB)
        finally:
            pl.views_of = kept
    assert set(by_votes) == {"B1"} and abs(by_votes["B1"] - 1.9 / 5.6) < 0.05      # all of it, by a third of the votes
    assert abs(result["doorways"]["B1"] - by_votes["B1"]) < 1e-3 and "B1 stands in a doorway" in rs.describe(result)
    assert rs.doorways_by_votes(Path("/nonexistent"), shapes, VOCAB) == {}
    # with the room's scale known, a piece thinner than THIN_M in that outline is the door itself: in full
    with tempfile.TemporaryDirectory() as tmp:
        space = Path(tmp)
        tracks.save(space / "workspace" / "tracks")
        (space / "densify.json").write_text(json.dumps({"colmap_units_per_metre": UNITS}))
        kept = pl.views_of
        pl.views_of = lambda space_, names: {n: views[n] for n in names if n in views}
        try:
            thin = rs.doorways_by_votes(space, shapes, VOCAB)                     # 0.6 units = 0.07 m thick
            deep = dict(CUPBOARD, min=[8.0, 15.0, 0.0], max=[15.0, 15.0 + 0.5 * UNITS, 18.0])  # 0.5 m deep
            not_thin = rs.doorways_by_votes(space, dict(shapes, boxes=[dict(BED), deep]), VOCAB)
        finally:
            pl.views_of = kept
    assert thin["B1"] > 0.9 and (not not_thin or not_thin["B1"] < 0.5)


def test_a_block_detected_as_real_furniture_is_a_piece_and_a_pillow_is_not():
    shapes = room()
    stove = {"min": [8.0, 8.0, 0.0], "max": [12.0, 12.0, 8.0], "label": "block",
             "detected": "wood stove", "build": True}
    cushion = {"min": [0.0, 5.0, 4.0], "max": [3.0, 8.0, 5.0], "label": "block",
               "detected": "pillow", "build": True}
    anonymous = {"min": [14.0, 2.0, 0.0], "max": [16.0, 4.0, 2.0], "label": "block", "build": True}
    shapes["boxes"] = [stove, cushion, anonymous, dict(BED)]
    assert [i for i, _ in rs.built_pieces(shapes, VOCAB)] == [0, 3]
    assert [i for i, _ in rs.built_pieces(shapes)] == [3]      # without the vocabulary: as before
    # The stove's box now stands for its name, so the score stops reporting
    # "no box" for a box the room has.
    assert [i for i, _ in rs.boxes_of(shapes, "wood stove", VOCAB)] == [0]


def cloud_room():
    """A synthetic room in the shapes frame: floor 0..20 x 0..14 at z=0, a
    real back wall along y=14, a bogus half-way wall along y=7 (the room
    continues beyond it), and a second room's wall along y=20 seen past the
    real one. UNITS units to the metre, like the other fixtures."""
    rng = np.random.default_rng(3)
    def sheet(n, xr, yr, zr):
        return np.column_stack([rng.uniform(*xr, n), rng.uniform(*yr, n), rng.uniform(*zr, n)])
    floor = sheet(4000, (0, 20), (0, 14), (0.0, 0.05))
    back = sheet(2000, (0, 20), (13.95, 14.05), (0.0, 2.5 * UNITS / 9 * 9 / 4))
    back[:, 2] = rng.uniform(0.0, 20.0, len(back))        # full height, base included
    far = sheet(800, (6, 12), (19.95, 20.05), (0.0, 20.0))  # the next room, via a doorway
    P = np.vstack([floor, back, far])
    def wall(center_y, half_a, points, x=10.0):
        return {"kind": "wall", "normal": [0.0, 1.0, 0.0], "center": [x, center_y, 10.0],
                "axis_a": [1.0, 0.0, 0.0], "half_a": half_a, "points": points}
    planes = [wall(14.0, 10.0, 2000), wall(7.0, 10.0, 1500), wall(20.0, 3.0, 800)]
    cams = np.array([[6.0, 4.0], [10.0, 5.0], [14.0, 4.0]])
    return P, planes, cams


def test_wall_support_tells_a_real_wall_from_a_bogus_and_a_glimpsed_one():
    P, planes, cams = cloud_room()
    real = rs.wall_support(P, planes[0], 0.0, cams, UNITS, [planes[1], planes[2]])
    bogus = rs.wall_support(P, planes[1], 0.0, cams, UNITS, [planes[0], planes[2]])
    glimpsed = rs.wall_support(P, planes[2], 0.0, cams, UNITS, [planes[0], planes[1]])
    assert real["seam"] > 0.8 and real["interior"] < 0.15 and not real["occluded"]
    assert real["support"] > 0.7
    assert bogus["interior"] > 0.3 and bogus["support"] < 0.5 * real["support"]
    assert glimpsed["occluded"] and glimpsed["support"] == 0.0


def test_dropping_a_supported_wall_costs_and_dropping_a_bogus_one_pays(tmp_path=None):
    import tempfile
    P, planes, cams = cloud_room()
    with tempfile.TemporaryDirectory() as tmp:
        space = Path(tmp)
        walls = rs.WallEvidence(space)
        walls._loaded = True                       # the cloud, injected: no PLY on disk
        walls._positions = P
        walls.unit = UNITS
        shapes = {"world": np.eye(3).tolist(), "planes": [dict(p) for p in planes],
                  "room_level": {"floor_z": 0.0, "height": 20.0},
                  "cameras": cams.tolist(), "boxes": []}
        penalty = lambda m, c: rs.WALL_WEIGHT * m + rs.WALL_CUT_WEIGHT * c
        missing, cut, detail = rs.wall_terms(shapes, walls)
        assert missing == 0.0 and cut == detail["W1"]["interior"] > 0.3   # the kept bogus wall cuts the room
        dropped_bogus = json.loads(json.dumps(shapes)); dropped_bogus["planes"][1]["build"] = False
        m2, c2, _ = rs.wall_terms(dropped_bogus, walls)
        assert c2 < cut and penalty(m2, c2) < penalty(missing, cut)   # dropping the bogus wall pays net
        dropped_real = json.loads(json.dumps(shapes)); dropped_real["planes"][0]["build"] = False
        m3, c3, _ = rs.wall_terms(dropped_real, walls)
        assert m3 > 0.6 and penalty(m3, c3) > penalty(missing, cut)   # dropping the supported wall costs net
        assert rs.wall_terms(shapes, None) == (0.0, 0.0, {})


def test_wall_evidence_loads_a_real_cloud_from_disk():
    import tempfile
    from pointcloud import PointCloud, save_ply
    P, planes, cams = cloud_room()
    with tempfile.TemporaryDirectory() as tmp:
        space = Path(tmp)
        cloud = PointCloud(P.astype(np.float32), np.zeros((len(P), 3), np.uint8))
        save_ply(cloud, space / "cloud-dense.ply")
        (space / "densify.json").write_text(json.dumps({"colmap_units_per_metre": UNITS}))
        walls = rs.WallEvidence(space)
        shapes = {"world": np.eye(3).tolist(), "planes": [dict(p) for p in planes],
                  "room_level": {"floor_z": 0.0, "height": 20.0},
                  "cameras": cams.tolist(), "boxes": []}
        measured = walls.measured(shapes)
        assert measured["W0"]["support"] > 0.7 and measured["W2"]["occluded"]
        # A moved neighbour re-measures its dependants on the same evidence:
        # shift the real wall away and the far wall is no longer hidden.
        moved = json.loads(json.dumps(shapes))
        moved["planes"][0]["center"][1] = 40.0
        assert not walls.measured(moved)["W2"]["occluded"]


def test_boxes_stand_for_a_name_by_detection_kind_or_by_type():
    shapes = room()
    shapes["boxes"] = [dict(CUPBOARD, detected="wardrobe"),                       # another storage name: same kind
                       {"min": [0, 0, 0], "max": [5, 5, 18], "label": "wardrobe", "build": True},   # added, undetected
                       dict(BED)]
    assert [i for i, _ in rs.boxes_of(shapes, "almirah", VOCAB)] == [0, 1]
    assert [i for i, _ in rs.boxes_of(shapes, "bed", VOCAB)] == [2]


if __name__ == "__main__":
    for name, test in sorted(globals().items()):
        if name.startswith("test_"):
            test()
            print("ok", name)
