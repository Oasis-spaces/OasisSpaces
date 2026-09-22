"""The capture map: which way is up, and what a camera can see of the floor.

    python3 tools/tests/test_capture_map.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import capture_map as cm  # noqa: E402


def camera(position, look, up=(0, 0, 1)):
    """cam2world of an OpenCV camera (x right, y down, z forward) at `position` looking along `look`."""
    f = np.asarray(look, float) / np.linalg.norm(look)
    down = -(np.asarray(up, float) - f * (np.asarray(up, float) @ f))
    down /= np.linalg.norm(down)
    right = np.cross(down, f)
    pose = np.eye(4)
    pose[:3, :3] = np.stack([right, down, f], axis=1)
    pose[:3, 3] = position
    return pose


K = np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1]])


def test_up_is_the_way_the_phone_was_held():
    rng = np.random.default_rng(0)
    poses = np.stack([camera([x, 0, 1.5], [np.cos(a), np.sin(a), -0.3]) for x, a in zip(range(8), rng.uniform(0, 6, 8))])
    points = rng.uniform([-2, -2, 0], [2, 2, 2.5], (2000, 3))
    up, right, forward = cm.room_frame(poses, points)
    assert np.allclose(up, [0, 0, 1], atol=0.05)
    assert abs(right @ up) < 1e-9 and abs(forward @ up) < 1e-9


def test_the_floor_behind_a_bed_is_not_seen():
    eye = camera([0, 0, 1.5], [1, 0, -0.6])
    cells = np.array([[1.5, 0, 0.0],        # open floor in front
                      [3.0, 0, 0.0]])       # floor behind the bed
    xs, ys, zs = np.meshgrid(np.linspace(2.0, 2.4, 12), np.linspace(-0.6, 0.6, 30), np.linspace(0.0, 0.6, 12))
    bed = np.stack([xs.ravel(), ys.ravel(), zs.ravel()], axis=1)
    seen = cm.floor_coverage(cells, eye[None], K[None], (640, 480), 5.0, bed, 0.08)
    assert seen.tolist() == [1, 0]
    # with nothing in the way both are seen, and nothing beyond reach is
    seen = cm.floor_coverage(cells, eye[None], K[None], (640, 480), 5.0, np.zeros((0, 3)) + 99, 0.08)
    assert seen.tolist() == [1, 1]
    seen = cm.floor_coverage(cells, eye[None], K[None], (640, 480), 2.5, np.zeros((0, 3)) + 99, 0.08)   # 2.1 m and 3.4 m away
    assert seen.tolist() == [1, 0]


def test_a_cell_behind_the_camera_is_not_seen():
    eye = camera([0, 0, 1.5], [1, 0, -0.6])
    seen = cm.floor_coverage(np.array([[-1.5, 0, 0.0]]), eye[None], K[None], (640, 480), 5.0,
                             np.zeros((0, 3)) + 99, 0.08)
    assert seen.tolist() == [0]


def test_the_floor_is_the_hull_of_an_open_ring_of_walls():
    walls = np.zeros((40, 40), bool)
    walls[5, 5:35] = walls[34, 5:35] = walls[5:35, 5] = True        # three walls; the fourth side open
    floor = cm.hull_mask(walls)
    assert floor[20, 20] and floor[20, 33]                            # the middle, and up to the open side
    assert not floor[2, 2] and not floor[38, 20]                      # nothing outside the walls


if __name__ == "__main__":
    for name, test in sorted(globals().items()):
        if name.startswith("test_"):
            test()
            print("ok", name)
