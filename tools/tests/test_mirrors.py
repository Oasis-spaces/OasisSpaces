"""Mirrors and windows filled with the plane of their surroundings.

    python3 tools/tests/test_mirrors.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "pipeline"))
import mirrors  # noqa: E402

ROWS, COLS = 240, 320
FX = FY = 300.0
CX, CY = COLS / 2, ROWS / 2
K = 1.0                                     # the map is at frame size here


def plane_depth(normal, offset):
    """Depth of the plane n.p = offset along every pixel's ray."""
    vv, uu = np.mgrid[:ROWS, :COLS]
    rays = np.stack([(uu - CX) / FX, (vv - CY) / FY, np.ones_like(uu, float)], -1)
    return offset / (rays @ np.asarray(normal, float))


def test_dilation_grows_a_square_by_its_radius():
    m = np.zeros((20, 20), bool)
    m[8:12, 8:12] = True
    grown = mirrors.dilate(m, 3)
    assert grown.sum() == 10 * 10 and grown[5, 5] and not grown[4, 5] and grown[14, 14]
    assert np.array_equal(mirrors.dilate(m, 0), m)


def test_a_mirror_takes_the_depth_of_the_wall_around_it():
    n = np.array([0.1, 0.0, -1.0]); n /= np.linalg.norm(n)
    wall = plane_depth(n, -3.0)                           # a wall about 3 m away, slightly turned
    depth = wall.copy()
    mask = np.zeros((ROWS, COLS), bool); mask[80:160, 120:200] = True
    depth[mask] = 7.0                                     # the reflection: far too deep
    det = {"label": "mirror", "box": [120, 80, 199, 159], "mask": mask, "score": 0.9}
    normals = np.zeros((ROWS, COLS, 3), np.float16)
    filled_depth, normals, filled, records = mirrors.fill_unreliable(
        depth, K, [det], {"mirror"}, (FX, FY, CX, CY), normals)
    assert records[0]["filled"] == mask.sum() and records[0]["why"] is None
    assert np.allclose(filled_depth[mask], wall[mask], atol=0.005)       # within 5 mm of the true wall
    assert np.array_equal(filled, mask) and np.allclose(filled_depth[~mask], wall[~mask])
    assert np.allclose(normals[mask][0].astype(float), -n if n[2] > 0 else n, atol=0.01)
    assert np.allclose(depth[mask], 7.0)                                 # the input was not touched


def test_a_strip_with_some_clutter_still_gives_the_plane():
    n = np.array([0.0, 0.0, -1.0])
    wall = plane_depth(n, -2.5)
    depth = wall.copy()
    mask = np.zeros((ROWS, COLS), bool); mask[60:180, 100:220] = True
    depth[mask] = 0.4
    rng = np.random.default_rng(1)
    ring = mirrors.dilate(mask, 6) & ~mask
    noisy = ring & (rng.random(depth.shape) < 0.25)       # a quarter of the strip is something nearer
    depth[noisy] = 1.2
    filled_depth, _, filled, records = mirrors.fill_unreliable(depth, K, [{"label": "window", "box": [100, 60, 219, 179], "mask": mask}],
                                                               {"window"}, (FX, FY, CX, CY))
    assert records[0]["why"] is None and np.allclose(filled_depth[mask], wall[mask], atol=0.01)


def test_nothing_is_filled_without_flat_surroundings_or_enough_of_them():
    depth = plane_depth([0, 0, -1.0], -2.0)
    mask = np.zeros((ROWS, COLS), bool); mask[100:140, 150:190] = True
    # surroundings that are two planes a metre apart: no single plane fits
    depth[:, :170] = 1.0
    depth[mask] = 5.0
    _, _, filled, records = mirrors.fill_unreliable(depth, K, [{"label": "mirror", "box": [150, 100, 189, 139], "mask": mask}],
                                                    {"mirror"}, (FX, FY, CX, CY))
    assert not filled.any() and "not flat" in records[0]["why"]
    # surroundings with no depth at all
    depth = plane_depth([0, 0, -1.0], -2.0)
    depth[mirrors.dilate(mask, 8)] = 0.0
    depth[mask] = 5.0
    _, _, filled, records = mirrors.fill_unreliable(depth, K, [{"label": "mirror", "box": [150, 100, 189, 139], "mask": mask}],
                                                    {"mirror"}, (FX, FY, CX, CY))
    assert not filled.any() and "pixels of surroundings" in records[0]["why"]


def test_a_window_whose_surroundings_are_the_view_outside_is_left_alone():
    depth = plane_depth([0, 0, -1.0], -2.5)                              # the room, 2.5 m away at most
    mask = np.zeros((ROWS, COLS), bool); mask[80:160, 120:200] = True
    depth[mirrors.dilate(mask, 8)] = 9.0                                 # the strip is already outside
    depth[mask] = 12.0
    _, _, filled, records = mirrors.fill_unreliable(depth, K, [{"label": "window", "box": [120, 80, 199, 159], "mask": mask}],
                                                    {"window"}, (FX, FY, CX, CY))
    assert not filled.any() and "beyond the rest of the frame" in records[0]["why"]


def test_only_unreliable_detections_are_filled_and_a_rectangle_serves_without_an_outline():
    wall = plane_depth([0, 0, -1.0], -3.0)
    depth = wall.copy()
    depth[50:90, 50:90] = 9.0                                            # a screen, as a rectangle only
    depth[150:200, 200:260] = 1.0                                        # a bed: not unreliable, stays
    dets = [{"label": "television", "box": [50, 50, 89, 89]}, {"label": "bed", "box": [200, 150, 259, 199]}]
    filled_depth, _, filled, records = mirrors.fill_unreliable(depth, K, dets, {"television", "mirror"}, (FX, FY, CX, CY))
    assert [r["label"] for r in records] == ["television"]
    assert np.allclose(filled_depth[50:90, 50:90], wall[50:90, 50:90], atol=0.005)
    assert np.allclose(filled_depth[150:200, 200:260], 1.0) and filled[60, 60] and not filled[160, 220]
    # two mirrors side by side: each strip leaves out the other's outline
    depth = wall.copy()
    a = np.zeros((ROWS, COLS), bool); a[100:140, 100:140] = True
    b = np.zeros((ROWS, COLS), bool); b[100:140, 142:182] = True
    depth[a] = 8.0; depth[b] = 0.5
    filled_depth, _, _, records = mirrors.fill_unreliable(
        depth, K, [{"label": "mirror", "box": [100, 100, 139, 139], "mask": a}, {"label": "mirror", "box": [142, 100, 181, 139], "mask": b}],
        {"mirror"}, (FX, FY, CX, CY))
    assert all(r["why"] is None for r in records) and np.allclose(filled_depth[a | b], wall[a | b], atol=0.005)


def test_an_outline_at_a_coarser_raster_is_resized_to_the_map():
    wall = plane_depth([0, 0, -1.0], -3.0)
    depth = wall.copy()
    depth[80:160, 120:200] = 7.0
    small = np.zeros((ROWS // 2, COLS // 2), bool); small[40:80, 60:100] = True   # the outline at half size
    _, _, filled, records = mirrors.fill_unreliable(depth, K, [{"label": "mirror", "box": [120, 80, 199, 159], "mask": small}],
                                                    {"mirror"}, (FX, FY, CX, CY))
    assert filled[100, 150] and not filled[70, 150] and records[0]["filled"] == 80 * 80


if __name__ == "__main__":
    for name, test in sorted(globals().items()):
        if name.startswith("test_"):
            test()
            print("ok", name)
