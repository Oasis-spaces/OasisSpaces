"""The built room rendered from the video's own cameras.

    python3 tools/tests/test_room_views.py
"""

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import room_views as rv  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def test_a_colmap_camera_becomes_a_blender_camera_looking_the_same_way():
    # A COLMAP camera 5 units back on -Z looking along +Z, with the world frame the identity.
    R, t = np.eye(3), np.array([0.0, 0.0, 5.0])                     # x_cam = x + t: centre at (0, 0, -5)
    m = rv.blender_camera_matrix(np.eye(3), R, t)
    assert np.allclose(m[:3, 3], [0, 0, -5])
    assert np.allclose(-m[:3, 2], [0, 0, 1])                       # Blender looks along its -Z: +Z here
    assert np.allclose(m[:3, 1], [0, -1, 0])                       # image up is COLMAP's -Y
    assert np.allclose(m[:3, 0], [1, 0, 0])


def test_the_rooms_rotation_turns_the_camera_with_it():
    turn = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])   # 90 deg about z
    m = rv.blender_camera_matrix(turn, np.eye(3), np.array([0.0, 0.0, 5.0]))
    assert np.allclose(m[:3, 3], turn @ [0, 0, -5])
    assert np.allclose(-m[:3, 2], turn @ [0, 0, 1])
    assert np.allclose(np.linalg.det(m[:3, :3]), 1.0)              # still a rotation


def test_views_of_a_real_solve_stand_inside_its_room():
    space = ROOT / "spaces" / "pan-sep23"
    if not (space / "shapes.json").exists():
        print("  (skipped: spaces/pan-sep23 is not here)")
        return
    import json
    views = rv.camera_views(space, ["frame_00019.jpg", "frame_00012.jpg", "no_such_frame.jpg"])
    assert [v["name"] for v in views] == ["frame_00019", "frame_00012"]
    room = json.loads((space / "shapes.json").read_text())["room"]
    for v in views:
        x, y = v["matrix"][0][3], v["matrix"][1][3]
        assert abs(x - room["center"][0]) < room["half_u"] * 1.2 and abs(y - room["center"][1]) < room["half_v"] * 1.2
        assert v["width"] == 2160 and v["height"] == 3840 and 2000 < v["fx"] < 5000


def test_the_sheet_puts_each_frame_beside_its_render():
    from PIL import Image

    with tempfile.TemporaryDirectory() as folder:
        folder = Path(folder)
        frame, render = folder / "f.jpg", folder / "r.png"
        Image.new("RGB", (216, 384), (200, 50, 50)).save(frame)
        Image.new("RGB", (54, 96), (50, 50, 200)).save(render)
        out = rv.pairs_sheet([("frame_00001", frame, render), ("frame_00002", frame, render)],
                             folder / "sheet.png", tile_width=100)
        sheet = np.asarray(Image.open(out))
        assert sheet.shape[1] == 2 * 100 + 3 * 10
        # two rows: a red tile on the left, a blue one on the right, both scaled to 100 wide
        # (the frame is a JPEG, so its colour comes back a little off)
        assert np.abs(sheet[60, 50].astype(int) - [200, 50, 50]).max() < 12
        assert np.abs(sheet[60, 160].astype(int) - [50, 50, 200]).max() < 12
        assert sheet.shape[0] > 2 * (178 + 26 + 10)


if __name__ == "__main__":
    for name, test in sorted(globals().items()):
        if name.startswith("test_"):
            test()
            print("ok", name)
