"""Photographing a surface from the frames: the depth buffer that decides what
hides what, on frames whose sides are not a multiple of its block (1080p).

    python3 tools/tests/test_surface_fill_depth.py      (or pytest tools/tests)
"""

import struct
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import surface_fill as sf  # noqa: E402

FOCAL = 1000.0
SIZES = [(1080, 1920), (1920, 1080)]     # 1080p upright and sideways: 1080 is 67.5 blocks of 16


def camera(width, height):
    """A pinhole camera at the origin looking along +z, as COLMAP's model gives it."""
    return ({"R": np.eye(3), "t": np.zeros(3), "camera_id": 1, "name": "frame_00001.png"},
            {"model": 1, "width": width, "height": height, "params": (FOCAL, FOCAL, width / 2, height / 2)})


def wall(width, height, depth, spacing=0.01, margin=1.2):
    """Points on a wall `depth` in front of the camera, reaching past the frame on every side."""
    hx, hy = margin * width / 2 / FOCAL * depth, margin * height / 2 / FOCAL * depth
    xx, yy = np.meshgrid(np.arange(-hx, hx, spacing), np.arange(-hy, hy, spacing))
    return np.stack([xx.ravel(), yy.ravel(), np.full(xx.size, depth)], axis=1)


def at_pixel(cam, u, v, depth):
    """The point `depth` in front of the camera that shows at pixel (u, v)."""
    fx, fy, cx, cy = cam["params"]
    return np.array([[(u - cx) / fx * depth, (v - cy) / fy * depth, depth]])


def lookup(buf, u, v):
    """The buffer's depth at full-resolution pixels, read the way photograph reads it."""
    return buf[(np.asarray(v) / sf.DEPTH_SHRINK).astype(int), (np.asarray(u) / sf.DEPTH_SHRINK).astype(int)]


def test_a_frame_whose_sides_are_not_multiples_of_the_block_has_a_depth_to_its_edges():
    for width, height in SIZES:
        info, cam = camera(width, height)
        # A wall 2 m away; a box 1 m away in the bottom-right corner, inside the last partial
        # block; another at the right edge halfway down.
        corner = at_pixel(cam, width - 3, height - 3, 1.0)
        right = at_pixel(cam, width - 3, height / 2, 1.0)
        buf = sf.depth_buffer(np.vstack([wall(width, height, 2.0), corner, right]), info, cam)
        assert buf.shape == (-(-height // 16), -(-width // 16))
        # Every pixel photograph may look up, to the last row and column, has a depth.
        u, v = np.meshgrid(np.arange(0, width - 1, 3.0), np.arange(0, height - 1, 3.0))
        assert np.isfinite(lookup(buf, u, v)).all()
        assert np.isclose(lookup(buf, width - 1.5, height - 1.5), 1.0)       # the box in the corner
        assert np.isclose(lookup(buf, width - 1.5, height / 2), 1.0)         # the box at the edge
        assert np.isclose(lookup(buf, width - 1.5, 0.0), 2.0)                # the wall beside them
        assert np.isclose(lookup(buf, 0.0, height - 1.5), 2.0)
        # The box at the right edge stays there and does not show at the left edge of the next row.
        assert np.isclose(lookup(buf, 0.0, height / 2 + 16), 2.0)


def test_a_frame_that_sees_none_of_the_points_has_no_depth_anywhere():
    for width, height in SIZES:
        info, cam = camera(width, height)
        for points in (wall(width, height, -2.0), np.zeros((0, 3))):       # all behind the camera; none at all
            buf = sf.depth_buffer(points, info, cam)
            assert buf.shape == (-(-height // 16), -(-width // 16))
            assert np.isinf(buf).all()


def test_points_are_projected_through_the_same_lens_as_the_other_stages():
    import lens

    info, _ = camera(1080, 1920)
    # A wide phone lens with barrel distortion sees farther out than its pinhole would:
    # the rays its corner pixels show are well past 1.1 times the pinhole's half-width.
    wide = {"model": 4, "width": 1080, "height": 1920, "params": (FOCAL, FOCAL, 540.0, 960.0, -0.2, 0.03, 0.0, 0.0)}
    corners_u, corners_v = np.array([0.5, 1079.5, 0.5, 1079.5]), np.array([0.5, 0.5, 1919.5, 1919.5])
    x, y = lens.unproject(lens.from_colmap(wide), corners_u, corners_v)
    assert (np.abs(x) > 1.1 * 540 / FOCAL).all()
    u, v, z = sf.project(np.column_stack([2 * x, 2 * y, np.full(4, 2.0)]), info, wide)   # 2 m out on those rays
    assert np.allclose(u, corners_u, atol=1e-6) and np.allclose(v, corners_v, atol=1e-6) and np.allclose(z, 2.0)
    # A lens whose polynomial turns back keeps a point far outside the view outside the frame.
    folding = {"model": 4, "width": 2160, "height": 3840,
               "params": (4341.0, 4198.4, 1080.0, 1920.0, 0.6271, -1.6139, 0.01728, -0.00291)}
    u, v, _ = sf.project(np.array([[2.0, 0.0, 2.0], [0.0, 3.0, 2.0]]), info, folding)
    assert ((u < 0) | (u >= 2160) | (v < 0) | (v >= 3840)).all()
    # Any point lands where lens.project, the projection the other stages use, puts it.
    rng = np.random.default_rng(0)
    points = np.column_stack([rng.uniform(-3, 3, 1000), rng.uniform(-3, 3, 1000), rng.uniform(0.5, 4, 1000)])
    for cam in (wide, folding):
        u, v, _ = sf.project(points, info, cam)
        lu, lv = lens.project(lens.from_colmap(cam), points[:, 0] / points[:, 2], points[:, 1] / points[:, 2])
        assert np.allclose(u, lu) and np.allclose(v, lv)


def write_model(space, info, cam):
    """A COLMAP model of one frame: cameras.bin, images.bin and an empty points3D.bin."""
    model = space / "workspace" / "sparse" / "0"
    model.mkdir(parents=True)
    with open(model / "cameras.bin", "wb") as f:
        f.write(struct.pack("<Q", 1))
        f.write(struct.pack("<iiQQ", 1, cam["model"], cam["width"], cam["height"]))
        f.write(struct.pack(f"<{len(cam['params'])}d", *cam["params"]))
    with open(model / "images.bin", "wb") as f:
        f.write(struct.pack("<Q", 1))
        f.write(struct.pack("<I", 1) + struct.pack("<dddd", 1.0, 0, 0, 0) + struct.pack("<ddd", *info["t"]))
        f.write(struct.pack("<I", 1) + info["name"].encode() + b"\x00" + struct.pack("<Q", 0))
    (model / "points3D.bin").write_bytes(struct.pack("<Q", 0))


def test_a_wall_is_photographed_to_the_edges_of_a_1080p_frame():
    from PIL import Image
    from pointcloud import PointCloud, save_ply

    colour = (200, 60, 30)
    for width, height in SIZES:
        info, cam = camera(width, height)
        # The wall 2 m in front of the camera, facing it, in 1 cm texels and past the frame on every side.
        hx, hy = 1.2 * width / FOCAL, 1.2 * height / FOCAL
        surface = sf.Surface("wall", np.array([-hx, -hy, 2.0]), np.array([1.0, 0, 0]), np.array([0, 1.0, 0]),
                             np.array([0, 0, -1.0]), int(round(2 * hx / 0.01)), int(round(2 * hy / 0.01)), 0.01)
        with tempfile.TemporaryDirectory() as folder:
            space = Path(folder)
            write_model(space, info, cam)
            (space / "workspace" / "images").mkdir()
            Image.new("RGB", (width, height), colour).save(space / "workspace" / "images" / info["name"])
            dense = wall(width, height, 2.0)
            save_ply(PointCloud(dense.astype(np.float32), np.full((len(dense), 3), 128, np.uint8)),
                     space / "cloud-dense.ply")
            [(photo, seen, through)] = sf.photograph(space, SimpleNamespace(metre=1.0, world=np.eye(3)),
                                                     [surface], log=lambda *a: None)
        u, v, _ = sf.project(surface.points(), info, cam)
        u, v = u.reshape(seen.shape), v.reshape(seen.shape)
        inside = (u >= 0) & (v >= 0) & (u < width - 1) & (v < height - 1)
        last_block = inside & ((u >= width // 16 * 16) | (v >= height // 16 * 16))
        assert last_block.any() and (seen[last_block] > 0).all()            # the frame's partial strip too
        assert (seen[inside] > 0).all() and not (seen[~inside] > 0).any()
        assert np.allclose(photo[inside], colour, atol=0.5)
        assert not through.any()                                           # nothing seen through a solid wall


if __name__ == "__main__":
    for name, test in sorted(globals().items()):
        if name.startswith("test_"):
            test()
            print("ok", name)
