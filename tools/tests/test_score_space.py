"""Scoring a space against measured truth: alignment, scale and the room's parts.

A 5 x 4 x 2.6 m room with a sofa is built twice: once as the truth (a camera
path at 30 Hz, the surfaces as points, the sofa as a labelled box, turned and
moved in a frame of its own) and once as a space would hold it (a camera
solve turned against the scene, a believed metre 10% too long, walls, levels,
a room rectangle and boxes in scene units, frames half a second behind the
truth's clock). score_space must find the clock offset and the 10% and,
everything else being right, put the walls, floor and sofa on the truth.

    python3 tools/tests/test_score_space.py      (or pytest tools/tests)
"""

import json
import struct
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import score_space as ss  # noqa: E402

UNITS = 3.0              # the solve's true units per metre
BELIEVED = UNITS * 1.1   # what densify.json says: the space thinks a metre is 10% longer
HALF = (2.5, 2.0)
HEIGHT = 2.6
CLOCK = 0.5              # the space's video starts this many seconds after the truth's path


def rz(deg):
    a = np.radians(deg)
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])


TRUTH_TURN, TRUTH_SHIFT = rz(20), np.array([1.0, 2.0, 0.0])
WORLD = rz(35) @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])   # scene = WORLD @ solve


def to_truth(r):
    return np.asarray(r, float) @ TRUTH_TURN.T + TRUTH_SHIFT


def path(tau):
    """The camera in room metres at the truth's time tau."""
    tau = np.asarray(tau, float)
    return np.column_stack([1.5 * np.cos(0.4 * tau), 1.2 * np.sin(0.4 * tau), 1.4 + 0.1 * np.sin(tau)])


SOFA = {"centre": [1.0, -1.0, 0.4], "size": [2.0, 0.9, 0.8]}
CHAIR = {"centre": [-1.5, 1.2, 0.45], "size": [0.5, 0.5, 0.9]}   # in the truth, never found by the space


def room_surface(seed=0):
    rng = np.random.default_rng(seed)
    hx, hy = HALF
    out = []
    for n, lo, hi in ((60000, [-hx, -hy, 0], [hx, hy, 0]), (60000, [-hx, -hy, HEIGHT], [hx, hy, HEIGHT])):
        out.append(rng.uniform(lo, hi, (n, 3)))
    for x in (-hx, hx):
        out.append(np.column_stack([np.full(30000, x), rng.uniform(-hy, hy, 30000), rng.uniform(0, HEIGHT, 30000)]))
    for y in (-hy, hy):
        out.append(np.column_stack([rng.uniform(-hx, hx, 30000), np.full(30000, y), rng.uniform(0, HEIGHT, 30000)]))
    out.append(ss.box_surface(SOFA["centre"], SOFA["size"], np.eye(3), per_face=2000))
    return np.vstack(out)


def write_truth(folder: Path):
    folder.mkdir(parents=True)
    taus = np.arange(0, 20, 1 / 30)
    boxes = [{"label": label, "centre": to_truth([b["centre"]])[0].tolist(), "size": b["size"],
              "axes": TRUTH_TURN.T.tolist()} for label, b in (("sofa", SOFA), ("chair", CHAIR))]
    (folder / "truth.json").write_text(json.dumps({
        "source": "synthetic", "boxes": boxes,
        "cameras": [{"t": float(t), "centre": c.tolist()} for t, c in zip(taus, to_truth(path(taus)))]}))
    np.save(folder / "surface.npy", to_truth(room_surface()).astype(np.float32))


def write_space(space: Path):
    hx, hy = HALF
    s = lambda v: (np.asarray(v, float) * UNITS).tolist()           # noqa: E731  room metres -> scene units
    planes = []
    for axis, sign in ((0, -1), (0, 1), (1, -1), (1, 1)):
        centre = [0.0, 0.0, HEIGHT / 2]
        centre[axis] = sign * HALF[axis]
        along = [0.0, 1.0, 0.0] if axis == 0 else [1.0, 0.0, 0.0]
        planes.append({"label": "wall", "center": s(centre), "axis_a": along, "axis_b": [0.0, 0.0, 1.0],
                       "half_a": HALF[1 - axis] * UNITS, "half_b": HEIGHT / 2 * UNITS, "source": "observed"})
    boxes = [{"label": "sofa", "min": s(np.subtract(SOFA["centre"], np.divide(SOFA["size"], 2))),
              "max": s(np.add(SOFA["centre"], np.divide(SOFA["size"], 2)))},
             {"label": "lamp", "min": s([2.0, 1.5, 0.0]), "max": s([2.3, 1.8, 1.5])}]
    shapes = {"world": WORLD.tolist(), "planes": planes, "boxes": boxes,
              "room": {"center": [0.0, 0.0], "axis_u": [1.0, 0.0], "axis_v": [0.0, 1.0],
                       "half_u": hx * UNITS, "half_v": hy * UNITS},
              "room_level": {"floor_z": 0.0, "height": HEIGHT * UNITS, "height_source": "measured"}}
    space.mkdir(parents=True)
    (space / "shapes.json").write_text(json.dumps(shapes))
    (space / "densify.json").write_text(json.dumps({"colmap_units_per_metre": BELIEVED}))

    fps = 6.0
    times = np.arange(0, 19, 0.5)
    centres = (path(times + CLOCK) * UNITS) @ WORLD     # solve = WORLD.T @ scene, as rows
    model = space / "workspace" / "sparse" / "0"
    model.mkdir(parents=True)
    frames = {}
    with open(model / "images.bin", "wb") as f:
        f.write(struct.pack("<Q", len(times)))
        for i, (t, c) in enumerate(zip(times, centres)):
            name = f"frame_{i + 1:05d}.jpg"
            frames[name] = int(round(t * fps))
            f.write(struct.pack("<I", i + 1) + struct.pack("<dddd", 1, 0, 0, 0) + struct.pack("<ddd", *(-c)))
            f.write(struct.pack("<I", 1) + name.encode() + b"\x00" + struct.pack("<Q", 0))
    (model / "points3D.bin").write_bytes(struct.pack("<Q", 0))
    grid = space / "workspace" / "video-grid"
    grid.mkdir(parents=True)
    (grid / "frames.json").write_text(json.dumps({"fps": fps, "frames": frames}))


def test_a_space_with_a_long_metre_scores_ten_percent_big_and_otherwise_true():
    with tempfile.TemporaryDirectory() as tmp:
        space, truth = Path(tmp) / "space", Path(tmp) / "truth"
        write_truth(truth)
        write_space(space)
        r = ss.score(space, truth)
        assert abs(r["scale_error"] - 0.10) < 0.005, r["scale_error"]
        assert abs(r["clock_offset_s"] - CLOCK) < 0.06 and r["path_rmse_m"] < 0.01, r
        assert abs(r["floor_error_m"]) < 0.03 and abs(r["ceiling_error_m"]) < 0.03, r
        assert abs(r["height_m"] - HEIGHT) < 0.01 and r["tilt_deg"] < 0.5, r
        assert len(r["walls"]) == 4 and r["wall_error_m"] < 0.02, r["walls"]
        assert r["footprint"]["iou"] > 0.9, r["footprint"]
        assert abs(r["footprint"]["true_area_m2"] - 20) < 1.5, r["footprint"]
        assert r["coverage"] > 0.9, r["coverage"]
        f = r["furniture"]
        assert f["recall"] == 0.5 and f["precision"] == 0.5, f
        assert f["missed"] == ["chair"] and f["unmatched_space_boxes"] == ["lamp"], f
        sofa = f["matched"][0]
        assert sofa["iou"] > 0.85 and sofa["centre_error_m"] < 0.03, sofa
        assert max(abs(e) for e in sofa["size_error_m"]) < 0.03, sofa
        assert r["room_type"] == "living room"
        assert "scale +10%" in ss.summary(r)


def test_umeyama_recovers_a_similarity():
    rng = np.random.default_rng(4)
    src = rng.normal(size=(40, 3))
    R = rz(70) @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    s, R2, t = ss.umeyama(src, 2.5 * src @ R.T + [1, -2, 3])
    assert abs(s - 2.5) < 1e-9 and np.allclose(R2, R) and np.allclose(t, [1, -2, 3])


def test_oriented_boxes_overlap_by_their_footprint_and_height():
    a = {"centre": [0, 0, 0.5], "size": [2, 1, 1], "axes": np.eye(3).tolist()}
    turned = {"centre": [0, 0, 0.5], "size": [1, 2, 1], "axes": rz(90).T.tolist()}   # the same box, its sides swapped
    assert ss.oriented_iou(a, turned) > 0.95
    half_up = {"centre": [0, 0, 1.0], "size": [2, 1, 1], "axes": np.eye(3).tolist()}
    assert abs(ss.oriented_iou(a, half_up) - 1 / 3) < 0.03


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print("ok", name)
