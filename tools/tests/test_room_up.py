"""Which way is up: the cameras' estimate must not lean with the camera's
pitch, and shapes.py levels it on the room's floor and ceiling.

    python3 tools/tests/test_room_up.py
"""

import contextlib
import io
import json
import struct
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "pipeline"))
import room_up  # noqa: E402
import shapes  # noqa: E402
from pointcloud import PointCloud, save_ply  # noqa: E402

# The solve's frame is not the room's: true up in it is SOLVE @ +Z.
SOLVE = Rotation.from_euler("xyz", [71.0, -18.0, 27.0], degrees=True).as_matrix()
UP = SOLVE @ np.array([0.0, 0.0, 1.0])


def angle(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.degrees(np.arccos(np.clip(a @ b / np.linalg.norm(a) / np.linalg.norm(b), -1, 1))))


def world_to_camera(yaw, pitch, roll=0.0):
    """COLMAP's R (world -> camera; rows x right, y down, z forward) of a
    camera turned `yaw`, looking `pitch` up (negative: down), rolled `roll`, in degrees."""
    a, p, r = np.radians([yaw, pitch, roll])
    forward = np.array([np.cos(p) * np.cos(a), np.cos(p) * np.sin(a), np.sin(p)])
    right = np.array([np.sin(a), -np.cos(a), 0.0])
    down = np.cross(forward, right)
    right, down = np.cos(r) * right + np.sin(r) * down, -np.sin(r) * right + np.cos(r) * down
    return np.stack([right, down, forward]) @ SOLVE.T


def pan(yaws, pitch, roll_sd=0.0, seed=0):
    rng = np.random.default_rng(seed)
    return [world_to_camera(y, pitch, rng.normal(0, roll_sd)) for y in yaws]


def write_model(space: Path, rotations) -> Path:
    """The space's COLMAP model: images.bin with these cameras, no points."""
    model = space / "workspace" / "sparse" / "0"
    model.mkdir(parents=True)
    with open(model / "images.bin", "wb") as f:
        f.write(struct.pack("<Q", len(rotations)))
        for i, R in enumerate(rotations, 1):
            x, y, z, w = Rotation.from_matrix(R).as_quat()
            centre = SOLVE @ np.array([2.0, 0.6 + 0.1 * i, 1.4])
            f.write(struct.pack("<I", i) + struct.pack("<dddd", w, x, y, z) + struct.pack("<ddd", *(-R @ centre)))
            f.write(struct.pack("<I", 1) + f"frame_{i:04d}.jpg".encode() + b"\x00" + struct.pack("<Q", 0))
    (model / "points3D.bin").write_bytes(struct.pack("<Q", 0))
    return model


def room_cloud(normals=True, sloped_ceiling=False, seed=1):
    """A 4 x 5 x 2.6 m room's surfaces, with their inward normals. With
    sloped_ceiling, no floor was filmed and the ceiling is an attic roof
    pitched 33 degrees."""
    rng = np.random.default_rng(seed)
    parts = []

    def sheet(n, lo, hi, normal):
        parts.append((rng.uniform(lo, hi, (n, 3)), np.tile(normal, (n, 1))))

    if not sloped_ceiling:
        sheet(30000, [0, 0, 0], [4, 5, 0], [0, 0, 1.0])
        sheet(20000, [0, 0, 2.6], [4, 5, 2.6], [0, 0, -1.0])
    else:
        slope = np.radians(33)
        roof = rng.uniform([0, 0], [4, 5], (20000, 2))
        points = np.column_stack([roof, 1.2 + roof[:, 0] * np.tan(slope)])
        parts.append((points, np.tile([np.sin(slope), 0, -np.cos(slope)], (20000, 1))))
    sheet(12000, [0, 0, 0], [0, 5, 2.6], [1.0, 0, 0])
    sheet(12000, [4, 0, 0], [4, 5, 2.6], [-1.0, 0, 0])
    sheet(10000, [0, 0, 0], [4, 0, 2.6], [0, 1.0, 0])
    sheet(10000, [0, 5, 0], [4, 5, 2.6], [0, -1.0, 0])
    points = np.vstack([p for p, _ in parts]) + rng.normal(0, 0.02, (sum(len(p) for p, _ in parts), 3))
    nrm = np.vstack([n for _, n in parts])
    return PointCloud((points @ SOLVE.T).astype(np.float32), np.full((len(points), 3), 160, np.uint8),
                      (nrm @ SOLVE.T).astype(np.float32) if normals else None)


def run_shapes(rotations, cloud) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        space = Path(tmp) / "room"
        write_model(space, rotations)
        save_ply(cloud, space / "cloud-dense.ply")
        argv, sys.argv = sys.argv, ["shapes.py", str(space)]
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                shapes.main()
        finally:
            sys.argv = argv
        return json.loads((space / "shapes.json").read_text())


def test_a_pan_filmed_looking_down_does_not_tilt_the_room():
    # A 60-degree pan from one spot, looking 15 degrees down: the mean image
    # up leans 13 degrees toward where the camera looked.
    with tempfile.TemporaryDirectory() as tmp:
        up, how = shapes.estimate_up(write_model(Path(tmp), pan(np.linspace(60, 120, 20), -15, roll_sd=0.5)))
    assert angle(up, UP) < 1.0, angle(up, UP)
    assert "sideways" in how


def test_a_camera_that_never_turned_gives_its_mean_image_up():
    # A walk along the room looking one way: the sideways axes are all one,
    # and their scatter is hand roll, which says nothing about up.
    rotations = pan(np.full(16, 70.0), -20, roll_sd=1.0)
    up, how = room_up.camera_up(np.stack([R.T for R in rotations]))
    image_up = -np.mean([R[1] for R in rotations], axis=0)
    assert angle(up, image_up) < 0.1 and "never turned" in how
    assert 19 < angle(up, UP) < 21          # the pitch: shapes.py levels it on the floor


def test_up_settles_on_the_floor_and_ceiling_normals():
    rng = np.random.default_rng(2)
    floor = np.tile(UP, (4000, 1)) + rng.normal(0, 0.08, (4000, 3))
    walls = np.cross(UP, rng.normal(size=(6000, 3)))
    off = Rotation.from_rotvec(np.cross(UP, [1.0, 0, 0]) / np.linalg.norm(np.cross(UP, [1.0, 0, 0]))
                               * np.radians(22)).apply(UP)
    assert angle(room_up.refine_up(off, np.vstack([floor, -floor[:2000], walls])), UP) < 0.3
    # Too little floor to go on: no correction.
    assert room_up.refine_up(off, np.vstack([floor[:150], walls])) is None


def test_shapes_levels_a_walk_that_never_turned_on_the_floor():
    data = run_shapes(pan(np.full(16, 70.0), -20, roll_sd=1.0), room_cloud())
    assert angle(np.array(data["world"])[2], UP) < 0.5, data["up"]
    assert 19 < data["up_correction_deg"] < 21
    assert "never turned" in data["up_source"] and "cloud's normals" in data["up_source"]


def test_shapes_levels_on_fitted_surfaces_when_the_cloud_has_no_normals():
    data = run_shapes(pan(np.full(16, 70.0), -20, roll_sd=1.0), room_cloud(normals=False))
    assert angle(np.array(data["world"])[2], UP) < 1.0, data["up"]
    assert "fitted" in data["up_source"]


def test_an_attic_roof_does_not_tip_a_level_solve():
    # The cameras turned, so their up is right; the only surface near level is
    # a roof pitched 33 degrees, further than a solve is believed to be off.
    data = run_shapes(pan(np.linspace(0, 300, 24), -10, roll_sd=0.5), room_cloud(sloped_ceiling=True))
    assert angle(np.array(data["world"])[2], UP) < 1.0, data["up"]
    assert data["up_correction_deg"] == 0.0 and "levelled" not in data["up_source"]


if __name__ == "__main__":
    for name, test in sorted(globals().items()):
        if name.startswith("test_"):
            test()
            print("ok", name)
