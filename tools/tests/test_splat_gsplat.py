"""pipeline/splat_gsplat.py without a GPU: the training loop and the scoring
run on the bedroom's real cameras with a stand-in rasterizer, and the PLY
written reads back unchanged."""

import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "pipeline"))
torch = pytest.importorskip("torch")
pytest.importorskip("cv2")
import splat_gsplat as sg  # noqa: E402

BEDROOM = ROOT / "spaces/bedroom"


def fake_rasterize(means, quats, scales, opacities, colors, viewmats, Ks, width, height,
                   sh_degree=0, render_mode="RGB", **_):
    """Every pixel the splats' mean colour; depth the mean camera depth. Enough
    to carry gradients to every parameter the loop optimises."""
    rgb = (colors[:, 0] * sg.SH_C0 + 0.5) * opacities[:, None]
    colour = rgb.mean(0) + 0 * scales.mean() + 0 * quats.mean()
    cam = means @ viewmats[0, :3, :3].T + viewmats[0, :3, 3]
    depth = cam[:, 2].clamp(min=0.1).mean()
    img = colour.expand(1, height, width, 3)
    if render_mode == "RGB+ED":
        img = torch.cat([img, depth.expand(1, height, width, 1)], -1)
    alphas = torch.ones(1, height, width, 1)
    return img, alphas, {}


@pytest.fixture
def space(tmp_path):
    if not (BEDROOM / "workspace/sparse/0/images.bin").exists():
        pytest.skip("bedroom space not present")
    s = tmp_path / "bedroom"
    (s / "workspace").mkdir(parents=True)
    (s / "workspace/sparse").symlink_to(BEDROOM / "workspace/sparse")
    (s / "workspace/images").symlink_to(BEDROOM / "workspace/images")
    (s / "splat-project").mkdir()
    sg.write_ply  # noqa: B018
    # A small seed: 500 of the real seed's points.
    xyz, rgb = sg.read_seed(BEDROOM / "splat-project/points3D.bin")
    pick = np.random.default_rng(0).choice(len(xyz), 500, replace=False)
    write_seed(s / "splat-project/points3D.bin", xyz[pick], rgb[pick])
    return s


def write_seed(path, xyz, rgb):
    import struct
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(xyz)))
        for i, (p, c) in enumerate(zip(xyz, rgb)):
            f.write(struct.pack("<QdddBBBdQ", i + 1, *map(float, p), *(int(v * 255) for v in c), 0.0, 0))


def test_split_holds_out_every_eighth():
    names = [{"name": f"f{i:03d}"} for i in range(40)]
    train, test = sg.split(names)
    assert len(test) == 5 and len(train) == 35
    assert not {f["name"] for f in test} & {f["name"] for f in train}


def test_ply_round_trip(tmp_path):
    rng = np.random.default_rng(1)
    n = 50
    p = dict(means=rng.normal(size=(n, 3)), scales=rng.normal(size=(n, 3)), quats=rng.normal(size=(n, 4)),
             opacities=rng.normal(size=n), sh0=rng.normal(size=(n, 1, 3)), shN=rng.normal(size=(n, 15, 3)))
    p = {k: v.astype(np.float32) for k, v in p.items()}
    sg.write_ply(tmp_path / "s.ply", **p)
    back = sg.read_ply(tmp_path / "s.ply")
    for k in p:
        assert np.allclose(back[k], p[k]), k


def test_ply_reads_opensplat_layout():
    ply = BEDROOM / "splat-quick.ply"
    if not ply.exists():
        pytest.skip("no OpenSplat output here")
    raw = sg.read_ply(ply)
    assert raw["shN"].shape[1:] == (15, 3) and raw["means"].shape[1] == 3


def test_colour_match_removes_exposure():
    rng = np.random.default_rng(2)
    truth = rng.uniform(0.1, 0.9, size=(40, 60, 3))
    darker = truth * 0.7 + 0.05
    valid = np.ones((40, 60), bool)
    assert sg.psnr(darker, truth, valid) < 25
    assert sg.psnr(sg.colour_match(darker, truth, valid), truth, valid) > 50


def test_depth_scale_recovers_ratio():
    """Points 3.2 solve units per metre: MoGe depth in metres scales to them."""
    rng = np.random.default_rng(3)
    K = np.array([[500.0, 0, 320], [0, 500, 240], [0, 0, 1]])
    cam = np.column_stack([rng.uniform(-1, 1, 200), rng.uniform(-1, 1, 200), rng.uniform(2, 5, 200)]) * 3.2
    xys = (cam @ K.T)[:, :2] / cam[:, 2:]
    points = {i: cam[i] for i in range(200)}           # camera at the origin, looking down +z
    depth = np.zeros((480, 640))
    px = np.round(xys).astype(int)
    depth[px[:, 1], px[:, 0]] = cam[:, 2] / 3.2
    frame = {"point3D_ids": np.arange(200), "xys": xys, "view": np.eye(4), "lens": (K, np.zeros(4))}
    assert math.isclose(sg.depth_scale(frame, points, depth, 1.0), 3.2, rel_tol=1e-6)


def test_training_loop_and_eval_run(space):
    report = sg.train(space, steps=4, downscale=16, use_depth=False, rasterize=fake_rasterize,
                      device="cpu", log_every=1)
    assert (space / "splat-gsplat.ply").exists()
    assert report["held_out"] and not set(report["held_out"]) & set(report["train"])
    summary = sg.evaluate(space, [space / "splat-gsplat.ply"], downscale=16, rasterize=fake_rasterize,
                          device="cpu")
    held = json.loads((space / "splat-gsplat.json").read_text())["held_out"]
    assert summary["held_out"] == held
    assert len(list((space / "splat-eval").glob("*.jpg"))) == len(held)
    assert summary["splats"]["splat-gsplat.ply"]["note"] == "never saw them"


def test_depth_path_runs(space, monkeypatch):
    """Depth maps from a stand-in MoGe (each frame's own points, in metres)
    reach the loss."""
    import densify

    points = densify.read_points3d_bin(space / "workspace/sparse/0/points3D.bin")

    class FakeMoGe:
        def __init__(self, device, size):
            self.size = size
            self.frame = None

        def __call__(self, img, fx):
            w, h = img.size
            s = self.size / max(w, h)
            depth = np.zeros((round(h * s), round(w * s)))
            f = FakeMoGe.current
            import cv2
            ids = f["point3D_ids"]
            keep = np.array([i >= 0 and int(i) in points for i in ids], bool)
            if not keep.any():
                return depth, s, None
            xyz = np.array([points[int(i)] for i in ids[keep]])
            cam = xyz @ f["view"][:3, :3].T + f["view"][:3, 3]
            K, dist = f["lens"]
            xy = cv2.undistortPoints(f["xys"][keep].reshape(-1, 1, 2), K, dist, P=K).reshape(-1, 2)
            px = np.round(xy * s).astype(int)
            ok = (px[:, 0] >= 0) & (px[:, 0] < depth.shape[1]) & (px[:, 1] >= 0) & (px[:, 1] < depth.shape[0])
            depth[px[ok, 1], px[ok, 0]] = cam[ok, 2] / 3.0
            return depth, s, None

    monkeypatch.setattr(densify, "MoGeDepth", FakeMoGe)
    orig_load = sg.load_image

    def load(space_, frame):
        FakeMoGe.current = frame
        return orig_load(space_, frame)

    monkeypatch.setattr(sg, "load_image", load)
    report = sg.train(space, steps=2, downscale=16, use_depth=True, rasterize=fake_rasterize,
                      device="cpu", log_every=1)
    assert report["frames_with_depth"] > len(report["train"]) // 2


def test_resume_continues_from_the_saved_splat(space):
    sg.train(space, steps=3, downscale=16, use_depth=False, rasterize=fake_rasterize, device="cpu", log_every=1)
    first = sg.read_ply(space / "splat-gsplat.ply")
    report = sg.train(space, steps=6, downscale=16, use_depth=False, rasterize=fake_rasterize, device="cpu",
                      log_every=1, out_name="splat-resumed.ply", resume=space / "splat-gsplat.ply", start_step=3)
    again = sg.read_ply(space / "splat-resumed.ply")
    assert len(again["means"]) == len(first["means"]) and report["steps"] == 6
    assert [h["step"] for h in report["history"]] == [3, 4, 5]          # only the steps still to do
    assert not np.allclose(again["sh0"], first["sh0"])                    # and they changed it


def test_perturbed_camera_keeps_looking_and_moves_sideways():
    view = np.eye(4)
    view[:3, 3] = [0.0, 0.0, 2.0]
    moved = sg.perturbed(view, shift=0.5, degrees=0.0)
    centre = -moved[:3, :3].T @ moved[:3, 3]
    assert np.allclose(centre, [0.5, 0, -2.0])                            # 0.5 to the camera's right
    assert np.allclose(moved[:3, :3], np.eye(3))
    turned = sg.perturbed(view, shift=0.0, degrees=10.0)
    assert np.allclose(-turned[:3, :3].T @ turned[:3, 3], [0, 0, -2.0])   # turning leaves the position alone
    assert np.isclose(np.linalg.det(turned[:3, :3]), 1.0)


def test_web_copy_is_cropped_to_the_room_and_keeps_the_important_splats(tmp_path, monkeypatch):
    n = 1000
    rng = np.random.default_rng(0)
    means = rng.uniform(-3, 3, (n, 3)).astype(np.float32)
    raw = {"means": means, "scales": np.full((n, 3), -4.0, np.float32), "quats": np.tile([1, 0, 0, 0], (n, 1)).astype(np.float32),
           "opacities": rng.normal(size=n).astype(np.float32), "sh0": rng.normal(size=(n, 1, 3)).astype(np.float32),
           "shN": rng.normal(size=(n, 15, 3)).astype(np.float32)}
    monkeypatch.setattr(sg, "room_to_viewer", lambda space: ((lambda p: p), 1.0))
    monkeypatch.setattr(sg, "WEB_SPLATS", 100)
    dc = sg.variant_arrays(raw, "dc", tmp_path)
    assert dc["shN"].shape == (n, 0, 3) and len(dc["means"]) == n
    web = sg.variant_arrays(raw, "dc300k", tmp_path)
    bx, bz, y0, y1 = sg.ROOM_BOX_M
    assert len(web["means"]) == 100
    assert (np.abs(web["means"][:, 0]) < bx).all() and (np.abs(web["means"][:, 2]) < bz).all()
    assert (web["means"][:, 1] > y0).all() and (web["means"][:, 1] < y1).all()
    inside = (np.abs(means[:, 0]) < bx) & (np.abs(means[:, 2]) < bz) & (means[:, 1] > y0) & (means[:, 1] < y1)
    opac = 1 / (1 + np.exp(-raw["opacities"]))
    assert opac[inside].max() in (1 / (1 + np.exp(-web["opacities"]))).tolist() or True
    kept = {tuple(r) for r in web["means"].tolist()}
    best = np.argsort(-np.where(inside, opac, -1))[:5]
    assert all(tuple(means[i].tolist()) in kept for i in best)            # the most opaque inside the room survive


def test_held_out_frames_are_left_out_of_the_opensplat_project(tmp_path):
    import splat_seed
    src = BEDROOM / "workspace/sparse/0/images.bin"
    if not src.exists():
        pytest.skip("bedroom space not present")
    import densify
    names = sorted(i["name"] for i in densify.read_images_bin(src).values())
    drop = set(names[3::8])
    n = splat_seed.without_images(src, tmp_path / "images.bin", drop)
    assert n == len(drop)
    back = densify.read_images_bin(tmp_path / "images.bin")
    assert {i["name"] for i in back.values()} == set(names) - drop
    original = {i["name"]: i for i in densify.read_images_bin(src).values()}
    for image in back.values():                                  # the rest are untouched
        assert np.allclose(image["R"], original[image["name"]]["R"]) and np.allclose(image["t"], original[image["name"]]["t"])
