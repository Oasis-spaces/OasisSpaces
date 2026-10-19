"""The solved lens: rays out of the raw, distorted frames, and points back into them.

    python3 tools/tests/test_lens.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "pipeline"))
import densify  # noqa: E402
import lens  # noqa: E402
import mirrors  # noqa: E402
import placement as pl  # noqa: E402

OPENCV = 4
# Stage-1 solves of real clips (fx, fy, cx, cy, k1, k2, p1, p2): a square-pixel phone
# lens; the strongest radial one; and a degenerate one (fy/fx 0.54, p1 0.045) on which
# 20 fixed-point steps still miss by a pixel at the corners.
SQUARE = {"model": OPENCV, "width": 1920, "height": 1080,
          "params": (1082.0, 1086.6, 960.0, 540.0, 0.1399, -0.0196, 0.00675, -0.00133)}
STRONG = {"model": OPENCV, "width": 3840, "height": 2160,
          "params": (3682.9, 4001.9, 1920.0, 1080.0, 0.7292, 0.3009, 0.00400, -0.03241)}
DEGENERATE = {"model": OPENCV, "width": 1080, "height": 1920,
              "params": (894.1, 480.0, 540.0, 960.0, 0.0056, -0.0001, 0.04500, -0.00140)}
FOLDING = {"model": OPENCV, "width": 2160, "height": 3840,          # k2 < 0: the polynomial turns back
           "params": (4341.0, 4198.4, 1080.0, 1920.0, 0.6271, -1.6139, 0.01728, -0.00291)}


def colmap_project(x, y, params):
    """COLMAP's OPENCV projection of the rays (x, y, 1), written out here on its own."""
    fx, fy, cx, cy, k1, k2, p1, p2 = params
    r2 = x * x + y * y
    radial = 1 + k1 * r2 + k2 * r2 * r2
    xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    return fx * xd + cx, fy * yd + cy


def pixel_centres(cam, step):
    us, vs = np.meshgrid(np.r_[np.arange(0, cam["width"], step), cam["width"] - 1],
                         np.r_[np.arange(0, cam["height"], step), cam["height"] - 1])
    return us.ravel() + 0.5, vs.ravel() + 0.5


def test_every_pixel_of_the_solved_lenses_has_a_ray_that_projects_back_onto_it():
    for cam in (SQUARE, STRONG, DEGENERATE, FOLDING):
        u, v = pixel_centres(cam, 16)
        x, y = lens.unproject(lens.from_colmap(cam), u, v)
        assert np.isfinite(x).all()
        pu, pv = colmap_project(x, y, cam["params"])
        assert np.abs(pu - u).max() < 1e-6 and np.abs(pv - v).max() < 1e-6, cam["params"]
        # and lens.project, inside the frame, is COLMAP's projection
        qu, qv = lens.project(lens.from_colmap(cam), x, y)
        assert np.abs(qu - u).max() < 1e-6 and np.abs(qv - v).max() < 1e-6


def test_a_pinhole_camera_is_its_own_rays():
    cam = {"model": 1, "width": 640, "height": 480, "params": (500.0, 510.0, 320.0, 240.0)}
    x, y = lens.unproject(lens.from_colmap(cam), np.array([0.5, 639.5]), np.array([0.5, 479.5]))
    assert np.allclose(x, (np.array([0.5, 639.5]) - 320) / 500) and np.allclose(y, (np.array([0.5, 479.5]) - 240) / 510)
    simple = lens.from_colmap({"model": 2, "width": 640, "height": 480, "params": (500.0, 320.0, 240.0, 0.1)})
    assert (simple["fx"], simple["fy"], simple["cx"], simple["cy"]) == (500.0, 500.0, 320.0, 240.0)
    assert simple["distortion"] == (0.1, 0.0, 0.0, 0.0)


def test_a_point_far_outside_the_view_is_not_folded_back_into_the_picture():
    l = lens.from_colmap(FOLDING)
    # 45 degrees off the axis, three times the half-width of the view: the raw polynomial
    # puts it in the middle of the picture
    raw_u, _ = colmap_project(np.array([1.0]), np.array([0.0]), FOLDING["params"])
    assert 0 < raw_u[0] < FOLDING["width"]
    u, _ = lens.project(l, np.array([1.0]), np.array([0.0]))
    assert u[0] > FOLDING["width"]
    # along a ray outward the projection only ever moves outward
    u, _ = lens.project(l, np.linspace(0, 3, 300), np.zeros(300))
    assert np.all(np.diff(u) > 0)


def test_back_projected_points_lie_on_the_rays_their_pixels_see_through_the_lens():
    from PIL import Image

    img = Image.new("RGB", (SQUARE["width"], SQUARE["height"]))
    info = {"R": np.eye(3), "t": np.zeros(3)}
    pts, _, _, (us, vs) = densify.back_project(info, SQUARE, img, lambda u, v: np.full(len(u), 3.0), 16, 10.0)
    assert len(pts) == len(us) > 1000
    pu, pv = colmap_project(pts[:, 0] / pts[:, 2], pts[:, 1] / pts[:, 2], SQUARE["params"])
    # float32 points 3 m away: a thousandth of a pixel
    assert np.abs(pu - (us + 0.5)).max() < 1e-3 and np.abs(pv - (vs + 0.5)).max() < 1e-3


def test_a_mirror_seen_through_a_distorting_lens_takes_the_depth_of_its_wall():
    cam = SQUARE
    fx, fy, cx, cy, k1, k2, p1, p2 = cam["params"]
    k = 0.25                                                    # the working-size map is a quarter of the frame
    rows, cols = round(cam["height"] * k), round(cam["width"] * k)
    vv, uu = np.mgrid[:rows, :cols]
    # The true ray of each working pixel's centre, by plain fixed-point iteration (enough
    # for this mild lens), not by the lens module.
    xd, yd = ((uu + 0.5) / k - cx) / fx, ((vv + 0.5) / k - cy) / fy
    x, y = xd.copy(), yd.copy()
    for _ in range(200):
        r2 = x * x + y * y
        radial = 1 + k1 * r2 + k2 * r2 * r2
        x = (xd - 2 * p1 * x * y - p2 * (r2 + 2 * x * x)) / radial
        y = (yd - p1 * (r2 + 2 * y * y) - 2 * p2 * x * y) / radial
    n = np.array([0.35, 0.1, -1.0]); n /= np.linalg.norm(n)   # a wall about 3 m away, well turned
    wall = -3.0 / (np.stack([x, y, np.ones_like(x)], -1) @ n)
    depth = wall.copy()
    mask = np.zeros((rows, cols), bool); mask[60:200, 330:440] = True   # off to the side, where the lens bends most
    depth[mask] = 7.0
    det = {"label": "mirror", "box": [330 / k, 60 / k, 439 / k, 199 / k], "mask": mask, "score": 0.9}
    filled_depth, _, filled, records = mirrors.fill_unreliable(
        depth, k, [det], {"mirror"}, (fx, fy, cx, cy), None, (k1, k2, p1, p2))
    assert records[0]["filled"] == mask.sum() and np.array_equal(filled, mask)
    assert np.abs(filled_depth[mask] - wall[mask]).max() < 0.002          # within 2 mm of the true wall


def test_a_silhouette_lands_where_the_lens_shows_the_box():
    cam = {"model": OPENCV, "width": 1080, "height": 1920,
           "params": (1000.0, 1000.0, 540.0, 960.0, 0.3, 0.0, 0.0, 0.0)}
    view = {"R": np.eye(3), "t": np.zeros(3), "world": np.eye(3), **lens.from_colmap(cam)}
    centre = np.array([0.4, 0.7, 1.0]) * 10                      # well off the middle, 10 units out
    sil = pl.silhouette(centre - 0.2, centre + 0.2, view, (1920 // pl.GRID, 1080 // pl.GRID))
    rows, cols = np.nonzero(sil)
    u, v = colmap_project(np.array([0.4]), np.array([0.7]), cam["params"])
    # the pinhole puts it 10 and 17 cells nearer the middle
    assert abs(cols.mean() - u[0] / pl.GRID) < 1.5 and abs(rows.mean() - v[0] / pl.GRID) < 1.5


if __name__ == "__main__":
    for name, test in sorted(globals().items()):
        if name.startswith("test_"):
            test()
            print("ok", name)
