"""Placing furniture by its masks in the keyframes.

    python3 tools/tests/test_placement.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "pipeline"))
import placement as pl  # noqa: E402

UNITS = 9.0


def look_at(position, target):
    """A COLMAP pose (R, t) for a camera at `position` looking at `target`, image up = +z."""
    forward = np.asarray(target, float) - np.asarray(position, float)
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)                     # COLMAP: +y is down the image, +z forward
    R = np.stack([right, down, forward])                # rows: camera axes in world
    t = -R @ np.asarray(position, float)
    return R, t


def view(position, target):
    R, t = look_at(position, target)
    return {"R": R, "t": t, "world": np.eye(3), "fx": 1000.0, "fy": 1000.0, "cx": 540.0, "cy": 960.0,
            "width": 1080, "height": 1920}


def room():
    def wall(x=None, y=None):
        if x is not None:
            return {"kind": "wall", "normal": [-1.0, 0, 0], "center": [x, 9.0, 0.0], "axis_a": [0, 1.0, 0],
                    "axis_b": [0, 0, 1.0], "half_a": 13.0, "half_b": 11.5, "points": 1000}
        return {"kind": "wall", "normal": [0, -1.0, 0], "center": [-7.0, y, 0.0], "axis_a": [1.0, 0, 0],
                "axis_b": [0, 0, 1.0], "half_a": 13.5, "half_b": 11.5, "points": 1000}
    return {"planes": [wall(x=-20.0), wall(y=22.0), wall(x=6.0), wall(y=-4.0)], "boxes": [],
            "room": {"center": [-7.0, 9.0]}, "room_level": {"floor_z": -11.5, "height": 23.0}, "cameras": []}


def test_the_hull_of_a_square_is_its_corners():
    pts = np.array([[0, 0], [2, 0], [2, 2], [0, 2], [1, 1], [1, 0.5]], float)
    hull = pl.convex_hull(pts)
    assert len(hull) == 4 and {tuple(p) for p in hull} == {(0, 0), (2, 0), (2, 2), (0, 2)}


def test_a_box_in_front_of_the_lens_has_a_silhouette_and_one_behind_has_none():
    v = view([0, -30, 0], [0, 0, 0])
    sil = pl.silhouette(np.array([-5, -5, -5.0]), np.array([5, 5, 5.0]), v, (240, 135))
    assert 0 < sil.mean() < 0.5
    rows, cols = np.nonzero(sil)
    assert abs(cols.mean() - 540 / pl.GRID) < 3 and abs(rows.mean() - 960 / pl.GRID) < 3   # centred
    assert not pl.silhouette(np.array([-5, -50, -5.0]), np.array([5, -40, 5.0]), v, (240, 135)).any()   # behind it


def test_the_search_recovers_a_wardrobe_from_its_own_silhouettes():
    shapes = room()
    # The truth: a 1.0 x 0.6 x 2.0 m wardrobe against W0 (x = -20), 1 m along it from the W3 corner (y = -4).
    lo = np.array([-20.0, -4.0 + 1.0 * UNITS, -11.5])
    hi = np.array([-20.0 + 0.6 * UNITS, -4.0 + 2.0 * UNITS, -11.5 + 2.0 * UNITS])
    centre = (lo + hi) / 2
    views = {f"frame_{k:05d}.jpg": view(pos, centre) for k, pos in enumerate(
        [[-5.0, 2.0, 1.0], [-2.0, 12.0, 1.0], [-8.0, 20.0, 2.0]], 1)}
    frames = {n: {"mask": pl.silhouette(lo, hi, v, (240, 135)), **v} for n, v in views.items()}
    unseen = {"frame_00009.jpg": view([0.0, 9.0, 1.0], [-7.0, 9.0 + 8.0, 0.0])}   # looks at W0 elsewhere: no wardrobe there
    evidence = {"label": "wardrobe", "frames": frames, "unseen": unseen}
    box = pl.search(Path("."), shapes, "wardrobe", evidence, UNITS, log=lambda *_: None)
    assert box is not None and box["placement"]["wall"] == 0
    assert np.abs(np.array(box["min"]) - lo).max() < 0.2 * UNITS
    assert np.abs(np.array(box["max"]) - hi).max() < 0.2 * UNITS
    assert box["placement"]["score"] > 0.7


def test_a_few_false_detections_do_not_sink_a_piece_but_a_few_good_frames_do_not_carry_one():
    bed = [0.0, 0.0, 0.62, 0.66, 0.65, 0.52, 0.41, 0.43, 0.22, 0.16, 0.0]     # the pan's bed: 3 false of 11
    door = [0.0, 0.05, 0.36, 0.72, 0.0]                                        # the door read as a wardrobe
    assert pl.trimmed_mean(bed) > pl.ACCEPT_SCORE > pl.trimmed_mean(door)
    assert pl.trimmed_mean([0.9]) == 0.9 and pl.trimmed_mean([]) == 0.0


def test_loose_things_on_a_piece_join_its_evidence_and_others_do_not():
    bed = [{"label": "bed", "score": 0.5, "box": [100, 1000, 2000, 3800]}]
    frame = bed + [{"label": "backpack", "score": 0.6, "box": [500, 2000, 900, 2600]},      # on the bed
                   {"label": "backpack", "score": 0.6, "box": [1800, 2200, 2400, 2800]},    # half off it
                   {"label": "curtain", "score": 0.7, "box": [600, 1200, 1000, 3000]},      # hanging: never
                   {"label": "pillow", "score": 0.2, "box": [300, 1100, 700, 1400]}]        # too weak a detection
    joined = pl.with_what_lies_on_it(bed, frame, {"backpack", "pillow"})
    assert [d["label"] for d in joined] == ["bed", "backpack"] and joined[1]["box"][0] == 500


def test_a_detection_at_the_frames_edge_is_scored_inside_its_rectangle():
    v = view([0, -30, 0], [0, 0, 0])
    lo, hi = np.array([-5, -5, -5.0]), np.array([5, 5, 5.0])
    full = pl.silhouette(lo, hi, v, (240, 135))
    rows, cols = np.nonzero(full)
    # a sliver: only the left third of the piece shows, the detector's rectangle ends at the frame's left edge
    cut = cols.min() + (cols.max() - cols.min()) // 3
    sliver = full.copy()
    sliver[:, cut:] = False
    box = [0, int(rows.min() * pl.GRID), int(cut * pl.GRID), int(rows.max() * pl.GRID)]
    window = pl.scoring_window([{"box": box}], 1080, 1920)
    assert window is not None and window[0] == 0
    assert pl.scoring_window([{"box": [300, 500, 700, 900]}], 1080, 1920) is None             # clear of the edges
    whole = {"label": "x", "frames": {"a": {"mask": sliver, **v, "window": None}}, "unseen": {}}
    windowed = {"label": "x", "frames": {"a": {"mask": sliver, **v, "window": window}}, "unseen": {}}
    assert pl.score(lo, hi, whole)[0] < 0.5 < pl.score(lo, hi, windowed)[0]


def test_two_cupboards_under_one_label_come_out_as_two_instances():
    shapes = room()
    one_lo, one_hi = np.array([-20.0, 5.0, -11.5]), np.array([-14.6, 14.0, 6.5])          # on W0
    two_lo, two_hi = np.array([-10.0, 22.0 - 5.4, -11.5]), np.array([-1.0, 22.0, 6.5])    # on W1 (y = 22)
    def views_at(centre, positions):
        return {f"frame_{k:05d}.jpg": view(pos, centre) for k, pos in positions}
    v1 = views_at((one_lo + one_hi) / 2, [(1, [-5.0, 2.0, 1.0]), (2, [-2.0, 12.0, 1.0]), (3, [-12.0, 14.0, 2.0])])
    v2 = views_at((two_lo + two_hi) / 2, [(4, [-6.0, 2.0, 1.0]), (5, [2.0, 8.0, 1.0]), (6, [-14.0, 10.0, 2.0]), (7, [-5.0, 5.0, 3.0])])
    frames = {n: {"mask": pl.silhouette(one_lo, one_hi, v, (240, 135)), **v, "window": None} for n, v in v1.items()}
    frames.update({n: {"mask": pl.silhouette(two_lo, two_hi, v, (240, 135)), **v, "window": None} for n, v in v2.items()})
    found = pl.instances(Path("."), shapes, "wardrobe", {"label": "wardrobe", "frames": frames, "unseen": {}}, UNITS,
                         log=lambda *_: None)
    assert len(found) == 2
    walls = sorted(b["placement"]["wall"] for b in found)
    assert walls == [0, 1]
    for box in found:
        truth = (one_lo, one_hi) if box["placement"]["wall"] == 0 else (two_lo, two_hi)
        assert np.abs(np.array(box["min"]) - truth[0]).max() < 0.25 * UNITS
        assert np.abs(np.array(box["max"]) - truth[1]).max() < 0.25 * UNITS


def test_sizes_the_masks_cannot_tell_apart_go_to_the_typical_size():
    assert pl.size_prior((0, 0, 1.5 * UNITS, 1.9 * UNITS, 0.5 * UNITS), "bed", UNITS) < 1e-9
    assert pl.size_prior((0, 0, 0.9 * UNITS, 0.9 * UNITS, 0.4 * UNITS), "bed", UNITS) > 0.2


def test_nothing_is_placed_without_evidence_or_where_the_phone_was():
    shapes = room()
    assert pl.search(Path("."), shapes, "wardrobe", {"label": "wardrobe", "frames": {}, "unseen": {}}, UNITS,
                     log=lambda *_: None) is None
    # a camera standing on every spot along W0 keeps candidates off that wall
    shapes["cameras"] = [[-20.0 + 0.3 * UNITS, y] for y in np.arange(-4.0, 22.0, 0.5)]
    keys = {k[0] for k, _, _ in pl.candidates(shapes, "wardrobe", UNITS)}
    assert 0 not in keys and keys <= {1, 2, 3}


def test_with_tracks_the_evidence_is_every_frame_the_object_shows_in_with_what_lies_on_it():
    import json
    import tempfile

    import tracking

    frames = [f"frame_{i:05d}.jpg" for i in range(1, 101)]
    big = np.zeros((256, 256), bool); big[100:200, 60:180] = True        # the bed
    small = np.zeros((256, 256), bool); small[110:130, 80:110] = True     # a blanket on it
    stray = np.zeros((256, 256), bool); stray[10:30, 10:30] = True        # a blanket elsewhere
    W, H = 1080, 1920
    box = lambda m: tracking.mask_box(m, W, H)
    tracks = tracking.Tracks(frames, (W, H), [
        {"id": 0, "label": "bed", "votes": {"bed": 3.0}, "seed": frames[0],
         "frames": {f: {"box": box(big), "score": 0.95} for f in frames[:90]}},
        {"id": 1, "label": "blanket", "votes": {"blanket": 1.0}, "seed": frames[0],
         "frames": {f: {"box": box(small), "score": 0.9} for f in frames[:90]}},
        {"id": 2, "label": "blanket", "votes": {"blanket": 1.0}, "seed": frames[0],
         "frames": {f: {"box": box(stray), "score": 0.9} for f in frames[:90]}},
    ], {**{tracking.Tracks.key(0, f): np.packbits(big) for f in frames[:90]},
        **{tracking.Tracks.key(1, f): np.packbits(small) for f in frames[:90]},
        **{tracking.Tracks.key(2, f): np.packbits(stray) for f in frames[:90]}})
    with tempfile.TemporaryDirectory() as tmp:
        space = Path(tmp)
        tracks.save(space / "workspace" / "tracks")
        (space / "densify.json").write_text(json.dumps({"detections": tracks.all_detections()}))
        (space / "objects.json").write_text(json.dumps({"objects": [{"name": "blanket", "role": "loose"}]}))
        views = {f: view([0, 0, 0], [0, 1, 0]) for f in frames}
        kept = pl.views_of
        pl.views_of = lambda space, names: {n: views[n] for n in names if n in views}
        try:
            evidence = pl.mask_evidence(space, "bed", log=lambda *_: None)
        finally:
            pl.views_of = kept
    assert len(evidence["frames"]) == pl.MAX_EVIDENCE_FRAMES              # 90 frames, evenly thinned
    assert frames[0] in evidence["frames"] and frames[89] in evidence["frames"]
    assert evidence["unseen"] == {}                                        # no frame shows nothing of it
    mask = evidence["frames"][frames[0]]["mask"]
    assert mask.shape == (H // pl.GRID, W // pl.GRID)
    assert mask[int(115 * mask.shape[0] / 256), int(90 * mask.shape[1] / 256)]     # the blanket on the bed counts
    assert not mask[int(20 * mask.shape[0] / 256), int(20 * mask.shape[1] / 256)]  # the stray one does not
    assert pl.spaced(list("abcdefgh"), 3) == ["a", "e", "h"] and pl.spaced(list("ab"), 3) == ["a", "b"]


if __name__ == "__main__":
    for name, test in sorted(globals().items()):
        if name.startswith("test_"):
            test()
            print("ok", name)
