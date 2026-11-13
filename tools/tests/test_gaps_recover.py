"""pipeline/gaps.py recovers gap colour only from frames that saw it: never
through a piece, never from one frame alone, never as lone specks."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "pipeline"))
import gaps  # noqa: E402


def test_segment_through_box():
    lo, hi = np.array([0.0, 0, 0]), np.array([1.0, 1, 0.5])
    start = np.array([0.5, 0.5, 2.0])
    ends = np.array([[0.5, 0.5, 0.0],      # under the box, seen from above: blocked
                     [2.0, 0.5, 0.0],      # beside it: clear
                     [0.5, 0.5, 0.52]])    # on its top, within the stop: clear
    assert gaps.through_box(start, ends, lo, hi, stop=0.05).tolist() == [True, False, False]


def test_patches_drop_lone_cells():
    m = np.zeros((20, 20), bool)
    m[2:8, 2:8] = True           # a patch
    m[15, 15] = True             # a speck
    m[12, 3] = m[12, 5] = True   # scattered cells
    out = gaps.patches(m, min_cells=9)
    assert out[2:8, 2:8].all() and out.sum() == 36


def _view(centre, look_at, image):
    """A pinhole camera at centre looking at look_at (+z forward), no lens distortion."""
    f = np.asarray(look_at, float) - centre
    z = f / np.linalg.norm(f)
    x = np.cross([0, 0, 1.0], z)
    if np.linalg.norm(x) < 1e-6:
        x = np.array([1.0, 0, 0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    K = np.array([[200.0, 0, 200], [0, 200, 200], [0, 0, 1]])
    return {"name": "v", "R": R, "t": -R @ centre, "K": K, "dist": np.zeros(4), "size": (400, 400),
            "image": image, "centre": np.asarray(centre, float), "mask": None}


def test_recover_needs_two_clear_agreeing_views(monkeypatch):
    monkeypatch.setattr(gaps, "FRAME_REDUCE", 1)
    monkeypatch.setattr(gaps, "DEPTH_REDUCE", 4)
    red = np.zeros((400, 400, 3), np.uint8)
    red[..., 0] = 200
    cells = np.array([[0.0, 0, 0], [3.0, 0, 0]])          # on the floor (z = 0, normal up)
    target = {"cells": cells, "normal": np.array([0, 0, 1.0]), "own": np.zeros(0, bool)}
    views = [_view(np.array([0.0, -1, 2]), [0, 0, 0], red), _view(np.array([0.5, -1, 2]), [0, 0, 0], red)]
    pts, front = np.zeros((0, 3)), np.zeros(0, bool)
    gaps.recover([target], views, pts, front, metre=1.0)
    assert target["ok"].tolist() == [True, False]        # the second cell is out of both frames
    assert target["colours"][0, 0] == 200
    # A box over the first cell (a bed) hides it from both frames.
    target = {"cells": cells, "normal": np.array([0, 0, 1.0]), "own": np.zeros(0, bool)}
    bed = (np.array([-0.5, -0.5, 0.0]), np.array([0.5, 0.5, 0.5]))
    gaps.recover([target], views, pts, front, metre=1.0, boxes=[bed], world=np.eye(3))
    assert not target["ok"].any()
    # One frame alone is not enough.
    target = {"cells": cells, "normal": np.array([0, 0, 1.0]), "own": np.zeros(0, bool)}
    gaps.recover([target], views[:1], pts, front, metre=1.0)
    assert not target["ok"].any()


def test_pattern_recognition_finds_tiles_and_ignores_plain_surfaces():
    rows, cols = np.indices((120, 120))
    seen = np.ones((120, 120), bool)
    tiles = 128 + 60 * (((rows // 15) + (cols // 15)) % 2)          # 30 cell repeat in both directions
    pu, pv, strength = gaps.period(tiles.astype(float), seen, 0.02)
    assert pu in range(29, 32) and pv in range(29, 32) and strength > 0.5
    rng = np.random.default_rng(0)
    plain = 200 + rng.normal(0, 4, (120, 120))                      # paint: low contrast, no repeat
    assert gaps.period(plain, seen, 0.02)[:2] == (None, None)
    noisy = 128 + rng.normal(0, 40, (120, 120))                     # clutter: contrast, but no repeat
    assert gaps.period(noisy, seen, 0.02)[:2] == (None, None)


def test_splat_depth_decides_what_a_frame_sees(monkeypatch):
    """With the trained splat's depth map, a frame sees a cell only where the
    splat's surface is that cell: nearer means something is in the way."""
    monkeypatch.setattr(gaps, "FRAME_REDUCE", 1)
    red = np.zeros((400, 400, 3), np.uint8)
    red[..., 0] = 200
    cells = np.array([[0.0, 0, 0]])
    views = [_view(np.array([0.0, -1, 2]), [0, 0, 0], red), _view(np.array([0.5, -1, 2]), [0, 0, 0], red)]
    pts, front = np.zeros((0, 3)), np.zeros(0, bool)

    def with_map(depth_value):
        for v in views:
            z = (v["R"] @ cells[0] + v["t"])[2]
            v["splat_depth"] = np.full((50, 50), depth_value if depth_value is not None else z, np.float16)
            v["splat_alpha"] = np.full((50, 50), 255, np.uint8)
            v["splat_downscale"] = 8
        target = {"cells": cells, "normal": np.array([0, 0, 1.0]), "own": np.zeros(0, bool)}
        gaps.recover([target], views, pts, front, metre=1.0)
        return bool(target["ok"][0])

    assert with_map(None)            # the splat's surface is at the cell: seen
    assert not with_map(0.5)         # something half a metre from the camera stands in front
    assert not with_map(9.0)         # the splat has nothing there: the frame shows what lies beyond
