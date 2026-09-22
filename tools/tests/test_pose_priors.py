"""Pose priors: MapAnything's cameras taken back to our frames, and the
frames reconstruct.py --mapper priors keeps.

    python3 tools/tests/test_pose_priors.py
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "pipeline"))
import mapanything_solve as ms  # noqa: E402
import reconstruct  # noqa: E402


def test_a_phone_frame_fits_the_model_as_its_loader_does():
    # 9:16 portrait, the model's 294 x 518: scale to cover (294 x 522), crop 2 rows off the top.
    sx, sy, left, top = ms.crop_of(2160, 3840, 294, 518)
    assert (left, top) == (0, 2)
    assert abs(sx - 294 / 2160) < 1e-9 and abs(sy - 522 / 3840) < 1e-9


def test_the_crop_is_undone_pixel_for_pixel():
    from PIL import Image

    W, H, tw, th = 2160, 3840, 294, 518
    sx, sy, left, top = ms.crop_of(W, H, tw, th)
    scale = max(tw / W, th / H) + 1e-8
    rw, rh = int(np.floor(W * scale)), int(np.floor(H * scale))
    xs, ys = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    for field, axis in ((xs, 0), (ys, 1)):
        seen = np.asarray(Image.fromarray(field, mode="F").resize((rw, rh), Image.LANCZOS))[top:top + th, left:left + tw]
        K = np.array([[300.0, 0, 100.0], [0, 300.0, 200.0], [0, 0, 1]])
        back = ms.to_frame_intrinsics(K, sx, sy, left, top)
        # the principal point of the processed image lands on the frame pixel the loader took it from
        u, v = K[0, 2], K[1, 2]
        expected = seen[int(v), int(u)] + (u - int(u)) / (sx if axis == 0 else sy)
        assert abs((back[0, 2] if axis == 0 else back[1, 2]) - expected) < 0.05
    assert abs(back[0, 0] - 300 / sx) < 1e-9 and abs(back[1, 1] - 300 / sy) < 1e-9


def test_quaternions_turn_back_into_their_rotations():
    from scipy.spatial.transform import Rotation

    for rot in Rotation.random(20, random_state=1):
        R = rot.as_matrix()
        w, x, y, z = ms.quaternion(R)
        assert np.allclose(Rotation.from_quat([x, y, z, w]).as_matrix(), R, atol=1e-9)


def test_a_skipped_frame_sits_between_its_neighbours():
    from scipy.spatial.transform import Rotation

    a, b = np.eye(4), np.eye(4)
    b[:3, :3] = Rotation.from_euler("y", 90, degrees=True).as_matrix()
    b[:3, 3] = [2.0, 0, 0]
    mid = ms.interpolate_pose(a, b, 0.5)
    assert np.allclose(mid[:3, 3], [1.0, 0, 0])
    assert abs(Rotation.from_matrix(mid[:3, :3]).as_euler("xyz", degrees=True)[1] - 45) < 1e-6


def test_points_fuse_one_per_voxel():
    points = np.array([[0.001, 0, 0], [0.005, 0.004, 0], [0.5, 0.5, 0.5]])
    colours = np.array([[0, 0, 0], [200, 100, 50], [10, 10, 10]], float)
    fused, tint, counts = ms.voxel_fuse(points, colours, 0.02)
    assert sorted(counts.tolist()) == [1, 2]
    assert np.allclose(tint[np.argmax(counts)], [100, 50, 25])


def test_a_priors_run_keeps_only_the_frames_the_priors_were_made_from():
    with tempfile.TemporaryDirectory() as folder:
        workspace = Path(folder)
        images = workspace / "images"
        images.mkdir()
        for name in ("frame_00001.jpg", "frame_00002.jpg"):
            (images / name).write_bytes(b"")
        assert not reconstruct.priors_match_frames(workspace, images)            # no priors yet
        (workspace / reconstruct.PRIORS_FILE).write_text(json.dumps({"frames": {"frame_00001.jpg": {}}}))
        assert reconstruct.priors_match_frames(workspace, images)
        (workspace / reconstruct.PRIORS_FILE).write_text(json.dumps({"frames": {"frame_00009.jpg": {}}}))
        assert not reconstruct.priors_match_frames(workspace, images)            # other frames


if __name__ == "__main__":
    for name, test in sorted(globals().items()):
        if name.startswith("test_"):
            test()
            print("ok", name)
