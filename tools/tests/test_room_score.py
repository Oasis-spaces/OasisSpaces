"""The measured room score and what moves it.

    python3 tools/tests/test_room_score.py
"""

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
                    {"name": "pillow", "role": "on_furniture", "build_as": "block"}])


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
