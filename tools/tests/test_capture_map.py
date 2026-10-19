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


def test_up_stays_upright_when_the_camera_never_turned():
    # A walk along the room looking one way, about 19 degrees down, the hand
    # rolling a degree or so: the sideways axes are all one, and their scatter
    # is roll, which would make the viewing direction look like up.
    rng = np.random.default_rng(3)
    poses = []
    for x in np.linspace(0, 3, 12):
        pose = camera([x, 0, 1.4], [0.0, 1, -0.35])
        right, down, roll = pose[:3, 0].copy(), pose[:3, 1].copy(), np.radians(rng.normal(0, 1.0))
        pose[:3, 0], pose[:3, 1] = np.cos(roll) * right + np.sin(roll) * down, np.cos(roll) * down - np.sin(roll) * right
        poses.append(pose)
    poses = np.stack(poses)
    points = rng.uniform([-2, -1, 0], [2, 5, 2.5], (2000, 3))
    up, _, _ = cm.room_frame(poses, points)
    image_up = -poses[:, :3, 1].mean(axis=0)
    assert np.allclose(up, image_up / np.linalg.norm(image_up), atol=1e-6)   # the cameras' only guess
    assert up @ [0, 0, 1] > np.cos(np.radians(20))                           # upright, leaning by the pitch


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


def test_a_cell_the_lens_bends_into_the_picture_is_seen():
    eye = camera([0, 0, 1.5], [1, 0, -0.6])
    # the floor along the ray (0.66, 0.1, 1): a pinhole puts it at u = 650, past the
    # 640-pixel edge; the solved barrel lens (k1 = -0.2) shows it at u = 621
    ray = eye[:3, :3] @ np.array([0.66, 0.1, 1.0])
    cell = eye[:3, 3] + ray * (-eye[2, 3] / ray[2])
    nothing = np.zeros((0, 3)) + 99
    assert cm.floor_coverage(cell[None], eye[None], K[None], (640, 480), 5.0, nothing, 0.08).tolist() == [0]
    seen = cm.floor_coverage(cell[None], eye[None], K[None], (640, 480), 5.0, nothing, 0.08,
                             np.array([[-0.2, 0.0, 0.0, 0.0]]))
    assert seen.tolist() == [1]


def test_the_scale_is_guessed_from_how_high_the_phone_was_held():
    rng = np.random.default_rng(4)
    units = 2.8                                                      # solve units per metre
    floor = rng.uniform([-2, -2, 0], [2, 2, 0.01], (4000, 3)) * units
    walls = rng.uniform([-2, -2, 0], [2, 2, 2.6], (4000, 3)) * units
    poses = np.stack([camera(np.array([x, 0, 1.4]) * units, [1, 0.2 * x, -0.3]) for x in np.linspace(-1, 1, 9)])
    up = np.array([0, 0, 1.0])
    assert abs(cm.units_from_height(poses, np.vstack([floor, walls]), up) - units) < 0.05 * units


def test_a_stale_recorded_scale_loses_to_the_phones_height():
    rng = np.random.default_rng(5)
    units = 2.8
    points = np.vstack([rng.uniform([-2, -2, 0], [2, 2, 0.01], (3000, 3)),
                        rng.uniform([-2, -2, 0], [2, 2, 2.6], (3000, 3))]) * units
    poses = np.stack([camera(np.array([x, 0, 1.4]) * units, [1, 0.1 * x, -0.3]) for x in np.linspace(-1, 1, 9)])
    up = np.array([0, 0, 1.0])
    guessed = cm.units_from_height(poses, points, up)
    assert 1 / cm.SCALE_DISAGREEMENT < units / guessed < cm.SCALE_DISAGREEMENT      # a real scale is believed
    assert not 1 / cm.SCALE_DISAGREEMENT < 1.0 / guessed < cm.SCALE_DISAGREEMENT    # 1.0, the stale record, is not


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
