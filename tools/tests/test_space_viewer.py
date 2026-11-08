"""The space viewer's export: a cloud and room model in the viewer's metres.

A 4 x 3 m room (scene units 2.5 per metre, the camera solve turned against the
scene) with a bed: the bed's points must come out as piece 1, in metres with y
up and the room's centre on the floor at the origin, and the bed's box with them.

    python3 tools/tests/test_space_viewer.py      (or pytest tools/tests)
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(ROOT / "tools"))
import space_viewer_export as sve  # noqa: E402
from pointcloud import PointCloud, save_ply  # noqa: E402

U = 2.5


def rz(deg):
    a = np.radians(deg)
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])


WORLD = rz(30)   # scene = WORLD @ solve


def test_the_bed_comes_out_as_a_piece_in_metres_with_y_up():
    rng = np.random.default_rng(0)
    floor = np.column_stack([rng.uniform(-2, 2, 4000), rng.uniform(-1.5, 1.5, 4000), np.zeros(4000)])
    bed = np.column_stack([rng.uniform(0.5, 1.5, 1500), rng.uniform(-1.0, 1.0, 1500), rng.uniform(0.05, 0.5, 1500)])
    scene_m = np.vstack([floor, bed]) + [1.0, 2.0, 0.3]                      # the room's centre at (1, 2), floor at 0.3
    solve = (scene_m * U) @ WORLD                                            # solve = WORLD.T @ scene
    colours = np.vstack([np.tile([150, 140, 120], (4000, 1)), np.tile([200, 50, 50], (1500, 1))]).astype(np.uint8)
    shapes = {"world": WORLD.tolist(), "room_level": {"floor_z": 0.3 * U, "height": 2.6 * U},
              "room": {"center": [1.0 * U, 2.0 * U], "axis_u": [1.0, 0.0], "axis_v": [0.0, 1.0],
                       "half_u": 2.0 * U, "half_v": 1.5 * U},
              "planes": [{"label": "wall", "center": [3.0 * U, 2.0 * U, 1.6 * U], "axis_a": [0.0, 1.0, 0.0],
                          "half_a": 1.5 * U, "build": True}],
              "boxes": [{"label": "bed", "detected": "bed", "identity": "double bed",
                         "min": [1.5 * U, 1.0 * U, 0.3 * U], "max": [2.5 * U, 3.0 * U, 0.8 * U]}],
              "cameras": [[1.0 * U, 2.0 * U], [0.5 * U, 2.0 * U]]}
    with tempfile.TemporaryDirectory() as tmp:
        space = Path(tmp) / "room"
        space.mkdir()
        (space / "shapes.json").write_text(json.dumps(shapes))
        (space / "densify.json").write_text(json.dumps({"colmap_units_per_metre": U}))
        save_ply(PointCloud(solve.astype(np.float32), colours), space / "cloud-dense.ply")
        out = sve.export(space)
        room = json.loads((out / "room.json").read_text())
        raw = (out / "points.bin").read_bytes()
        n = room["count"]
        assert len(raw) == n * 16 and n == 5500
        pts = np.frombuffer(raw, np.float32, n * 3).reshape(n, 3)
        piece = np.frombuffer(raw, np.uint8, n, offset=n * 15)
        assert abs(pts[:, 1].min()) < 0.01 and pts[:, 1].max() < 0.55          # y up, the floor at 0
        bed_pts = pts[piece == 1]
        assert len(bed_pts) == 1500 and bed_pts[:, 0].min() > 0.45               # the bed, and only the bed
        assert np.allclose(np.median(bed_pts[:, [0, 2]], axis=0), [1.0, 0.0], atol=0.05)   # scene y -> viewer -z
        (bed_box,) = room["pieces"]
        assert bed_box["name"] == "double bed" and np.allclose(bed_box["min"], [0.5, 0.0, -1.0], atol=1e-3)
        assert np.allclose(bed_box["max"], [1.5, 0.5, 1.0], atol=1e-3)
        assert room["floorColor"] == [150, 140, 120] and abs(room["height"] - 2.6) < 1e-6
        assert room["walls"][0]["from"] == [2.0, 1.5] and room["walls"][0]["to"] == [2.0, -1.5]
        assert room["scaleSource"] == "video"

        # A person measured the bed's 1.0 m width as really 1.1 m: the space gets its real scale.
        import set_real_scale
        done = set_real_scale.set_scale(space, real_m=1.1, reconstructed_m=1.0)
        assert abs(done["factor"] - 1.1) < 1e-9
        meta = json.loads((space / "densify.json").read_text())
        assert meta["colmap_units_per_metre_video"] == U and abs(meta["colmap_units_per_metre"] - U / 1.1) < 1e-9
        room = json.loads((sve.export(space) / "room.json").read_text())
        assert room["scaleSource"] == "measured"
        bed = room["pieces"][0]
        assert abs((bed["max"][0] - bed["min"][0]) - 1.1) < 1e-3 and abs(room["height"] - 2.86) < 1e-3
        # set again from the same video-scale measurement: the factor is not applied twice
        set_real_scale.set_scale(space, real_m=1.1, reconstructed_m=1.0)
        assert abs(json.loads((space / "densify.json").read_text())["colmap_units_per_metre"] - U / 1.1) < 1e-9


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print("ok", name)
