#!/usr/bin/env python3
"""Train a space's splat with gsplat, and score splats on frames none of them
should have learned from.

Why a second trainer: OpenSplat's longer, sharper run came out *worse* than
its quick one on the bedroom (milky smears and streaks on plain walls). That
is what a phone's auto exposure and white balance do to a splat: each frame
is a little brighter or warmer than the last, and the splat paints those
differences into the room as view-dependent haze. Training longer at higher
resolution only fits them better. This trainer adds what OpenSplat lacks:

  - exposure: every training frame gets its own learned colour correction
    (3x3 matrix and offset) applied to the render before the loss, so the room
    keeps one colour and the frame's exposure is absorbed there;
  - depth: MoGe-2 depth for every training frame, put in the solve's units
    with the frame's own COLMAP points, pulls splats onto real surfaces
    (inverse-depth loss, strong early, fading out);
  - resolution: frames undistorted with the solved lens and trained at full
    resolution by default;
  - budget: gsplat's MCMC strategy, which keeps a fixed number of splats and
    moves dead ones to where the loss is, instead of growing floaters.

Every HOLD_OUT_EVERY-th frame is kept out of training. `eval` renders each
splat it is given from those frames' cameras and scores it against the
frame (PSNR and SSIM after one colour match per frame, so a splat is not
blamed for the frame's exposure), and writes the frame beside each render.

Usage (needs a CUDA GPU and gsplat):
    python3 pipeline/splat_gsplat.py train spaces/<name> [--steps 30000 --downscale 1]
    python3 pipeline/splat_gsplat.py eval spaces/<name> splat-quick.ply splat-long.ply splat-gsplat.ply
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from densify import read_cameras_bin, read_images_bin, read_points3d_bin  # noqa: E402

HOLD_OUT_EVERY = 8           # every 8th frame (by name) is a test frame
STEPS = 30000
DOWNSCALE = 1                # full resolution
CAP = 1_500_000              # splats kept by MCMC
SH_DEGREE = 3
SH_EVERY = 1000              # one more SH degree every this many steps
SSIM_WEIGHT = 0.2
DEPTH_WEIGHT = (0.3, 0.02)   # inverse-depth loss weight at the start and the end
DEPTH_SIZE = 1024            # MoGe's working size (long side) and the depth maps kept
EXPOSURE_LR = 1e-3
EXPOSURE_REG = 1e-2          # pull towards no correction, so the room keeps the colour
OPACITY_REG = 0.01           # MCMC's regularisers
SCALE_REG = 0.01
EVAL_DOWNSCALE = 2
SAVE_AT = (15000, 22000)
BUDGET_RATE_SHARE = 0.85      # share of the probed rate the rest of the run is planned at
BUDGET_PROBE = 300            # steps timed before fitting the run to its time budget
SH_C0 = 0.28209479177387814


# ----------------------------------------------------------------------- cameras

def frames(space: Path, downscale: int):
    """Solved frames, sorted by name: name, world-to-camera 4x4, undistorted
    pinhole K at the downscale, the size, and the lens (K, distortion) to
    undistort with."""
    model = space / "workspace/sparse/0"
    cameras = read_cameras_bin(model / "cameras.bin")
    images = read_images_bin(model / "images.bin")
    out = []
    for image in sorted(images.values(), key=lambda i: i["name"]):
        cam = cameras[image["camera_id"]]
        K, dist = lens(cam)
        view = np.eye(4)
        view[:3, :3], view[:3, 3] = image["R"], image["t"]
        w, h = cam["width"] // downscale, cam["height"] // downscale
        Ks = K.copy()
        Ks[:2] *= np.array([[w / cam["width"]], [h / cam["height"]]])
        out.append({"name": image["name"], "view": view, "K": Ks, "size": (w, h),
                    "lens": (K, dist), "full": (cam["width"], cam["height"]),
                    "xys": image["xys"], "point3D_ids": image["point3D_ids"]})
    return out


def lens(cam: dict) -> tuple[np.ndarray, np.ndarray]:
    """K and OpenCV distortion for COLMAP's camera models."""
    p, model = cam["params"], cam["model"]
    if model == 0:                       # SIMPLE_PINHOLE f cx cy
        f, cx, cy, d = p[0], p[1], p[2], []
        fx = fy = f
    elif model == 1:                     # PINHOLE fx fy cx cy
        fx, fy, cx, cy, d = *p[:4], []
    elif model == 2:                     # SIMPLE_RADIAL f cx cy k
        fx = fy = p[0]; cx, cy, d = p[1], p[2], [p[3], 0, 0, 0]
    elif model == 3:                     # RADIAL f cx cy k1 k2
        fx = fy = p[0]; cx, cy, d = p[1], p[2], [p[3], p[4], 0, 0]
    elif model == 4:                     # OPENCV fx fy cx cy k1 k2 p1 p2
        fx, fy, cx, cy, d = *p[:4], list(p[4:8])
    else:
        raise SystemExit(f"camera model {model} is not handled; solve with OPENCV or simpler")
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
    return K, np.array(d + [0] * (4 - len(d)), float)


def split(all_frames: list[dict]) -> tuple[list[dict], list[dict]]:
    test = [f for i, f in enumerate(all_frames) if i % HOLD_OUT_EVERY == HOLD_OUT_EVERY // 2]
    train = [f for f in all_frames if f not in test]
    return train, test


def load_image(space: Path, frame: dict):
    """The frame undistorted to its pinhole K and resized; with a mask of the
    pixels the undistortion had image for."""
    import cv2

    img = cv2.imread(str(space / "workspace/images" / frame["name"]), cv2.IMREAD_COLOR)[:, :, ::-1]
    K, dist = frame["lens"]
    if np.any(dist):
        img = cv2.undistort(img, K, dist)
        valid = cv2.undistort(np.full(img.shape[:2], 255, np.uint8), K, dist) > 250
    else:
        valid = np.ones(img.shape[:2], bool)
    w, h = frame["size"]
    if (img.shape[1], img.shape[0]) != (w, h):
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        valid = cv2.resize(valid.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
    return np.ascontiguousarray(img), valid


# --------------------------------------------------------------------- splat io

def read_seed(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """xyz and rgb (0..1) of a COLMAP points3D.bin (splat_seed.py writes one
    from the dense cloud)."""
    xyz, rgb = [], []
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            _, x, y, z, r, g, b, _, track = struct.unpack("<QdddBBBdQ", f.read(51))
            f.read(8 * track)
            xyz.append((x, y, z))
            rgb.append((r, g, b))
    return np.array(xyz, np.float32), np.array(rgb, np.float32) / 255.0


def ply_fields(sh_rest: int) -> list[str]:
    return (["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
            + [f"f_rest_{i}" for i in range(sh_rest * 3)]
            + ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"])


def write_ply(path: Path, means, scales, quats, opacities, sh0, shN, comment="gsplat") -> None:
    """The same layout OpenSplat writes (3DGS's): raw parameters, SH rest
    channel-major, so every other tool reads it unchanged."""
    n, rest = len(means), shN.shape[1]
    data = np.concatenate([
        means, np.zeros((n, 3), np.float32), sh0.reshape(n, 3),
        shN.transpose(0, 2, 1).reshape(n, rest * 3),
        opacities.reshape(n, 1), scales, quats], axis=1).astype("<f4")
    header = (f"ply\nformat binary_little_endian 1.0\ncomment {comment}\nelement vertex {n}\n"
              + "".join(f"property float {k}\n" for k in ply_fields(rest)) + "end_header\n")
    path.write_bytes(header.encode() + data.tobytes())


def read_ply(path: Path) -> dict:
    """Raw parameters of a 3DGS-layout PLY (OpenSplat's or ours)."""
    with open(path, "rb") as f:
        names, n = [], 0
        while (line := f.readline().decode().strip()) != "end_header":
            if line.startswith("element vertex"):
                n = int(line.split()[-1])
            elif line.startswith("property"):
                names.append(line.split()[-1])
        data = np.frombuffer(f.read(n * 4 * len(names)), "<f4").reshape(n, len(names))
    col = {k: data[:, i] for i, k in enumerate(names)}
    rest = sorted((k for k in names if k.startswith("f_rest_")), key=lambda k: int(k[7:]))
    shN = (np.stack([col[k] for k in rest], 1).reshape(n, 3, len(rest) // 3).transpose(0, 2, 1)
           if rest else np.zeros((n, 0, 3), np.float32))
    return {"means": np.stack([col["x"], col["y"], col["z"]], 1),
            "scales": np.stack([col[f"scale_{i}"] for i in range(3)], 1),
            "quats": np.stack([col[f"rot_{i}"] for i in range(4)], 1),
            "opacities": col["opacity"],
            "sh0": np.stack([col[f"f_dc_{i}"] for i in range(3)], 1)[:, None, :],
            "shN": np.ascontiguousarray(shN)}


# ------------------------------------------------------------------------ depth

def frame_depths(space: Path, train: list[dict], device) -> dict[str, np.ndarray]:
    """MoGe-2 depth for each training frame, in the solve's units: each map
    is scaled by the median ratio of its frame's own COLMAP points' depth to
    MoGe's at those pixels. Frames with too few points, or whose points
    disagree with MoGe, get no depth."""
    from PIL import Image
    from densify import MoGeDepth

    points = read_points3d_bin(space / "workspace/sparse/0/points3D.bin")
    moge = MoGeDepth(device, DEPTH_SIZE)
    out, kept = {}, 0
    for frame in train:
        img, _ = load_image(space, {**frame, "size": frame["full"]})
        K = frame["lens"][0]
        depth, scale, _ = moge(Image.fromarray(img), K[0, 0])
        ratio = depth_scale(frame, points, depth, scale)
        if ratio is not None:
            out[frame["name"]] = (depth * ratio).astype(np.float16)
            kept += 1
    print(f"depth: {kept} of {len(train)} training frames have MoGe depth in solve units", flush=True)
    del moge
    return out


def depth_scale(frame: dict, points: dict, depth: np.ndarray, scale: float) -> float | None:
    """Solve units per metre for this frame's MoGe depth, from its own points."""
    import cv2

    ids = frame["point3D_ids"]
    keep = np.array([i >= 0 and int(i) in points for i in ids], bool)
    if keep.sum() < 30:
        return None
    xyz = np.array([points[int(i)] for i in ids[keep]])
    cam = xyz @ frame["view"][:3, :3].T + frame["view"][:3, 3]
    K, dist = frame["lens"]
    # The observations are in the distorted frame; MoGe saw the undistorted one.
    xys = cv2.undistortPoints(frame["xys"][keep].reshape(-1, 1, 2), K, dist, P=K).reshape(-1, 2) \
        if np.any(dist) else frame["xys"][keep]
    px = np.round(xys * scale).astype(int)
    h, w = depth.shape
    ok = (cam[:, 2] > 0) & (px[:, 0] >= 0) & (px[:, 0] < w) & (px[:, 1] >= 0) & (px[:, 1] < h)
    d = np.zeros(len(px))
    d[ok] = depth[px[ok, 1], px[ok, 0]]
    ok &= d > 0
    if ok.sum() < 30:
        return None
    ratios = cam[ok, 2] / d[ok]
    ratio = float(np.median(ratios))
    spread = float(np.median(np.abs(np.log(ratios / ratio))))
    return ratio if spread < math.log(1.25) else None


# ----------------------------------------------------------------------- losses

def ssim(a, b, window: int = 11):
    """Mean SSIM of two [1,3,H,W] images in 0..1."""
    import torch
    import torch.nn.functional as F

    sigma = 1.5
    x = torch.arange(window, dtype=a.dtype, device=a.device) - window // 2
    g = torch.exp(-x ** 2 / (2 * sigma ** 2))
    g = (g / g.sum())
    kernel = (g[:, None] @ g[None, :]).expand(3, 1, window, window).contiguous()
    blur = lambda t: F.conv2d(t, kernel, padding=window // 2, groups=3)  # noqa: E731
    mu_a, mu_b = blur(a), blur(b)
    va = blur(a * a) - mu_a ** 2
    vb = blur(b * b) - mu_b ** 2
    cov = blur(a * b) - mu_a * mu_b
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    s = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2))
    return s.mean()


def colour_match(render: np.ndarray, frame: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """The render with one 3x4 colour transform fitted to the frame (least
    squares over valid pixels): exposure and white balance are the frame's,
    not the splat's, so scoring compares the room, not the camera's mood."""
    a = render[valid].reshape(-1, 3)
    b = frame[valid].reshape(-1, 3)
    if len(a) > 400_000:
        pick = np.random.default_rng(0).choice(len(a), 400_000, replace=False)
        a, b = a[pick], b[pick]
    A = np.column_stack([a, np.ones(len(a))])
    M, *_ = np.linalg.lstsq(A, b, rcond=None)
    out = np.column_stack([render.reshape(-1, 3), np.ones(render.shape[0] * render.shape[1])]) @ M
    return np.clip(out.reshape(render.shape), 0, 1)


def psnr(a: np.ndarray, b: np.ndarray, valid: np.ndarray) -> float:
    mse = float(np.mean((a[valid] - b[valid]) ** 2))
    return 10 * math.log10(1.0 / max(mse, 1e-10))


# --------------------------------------------------------------------- training

def scene_scale(views: list[np.ndarray]) -> float:
    centres = np.array([-v[:3, :3].T @ v[:3, 3] for v in views])
    return float(1.1 * np.linalg.norm(centres - centres.mean(0), axis=1).max())


def init_params(xyz: np.ndarray, rgb: np.ndarray, device):
    import torch
    from scipy.spatial import cKDTree

    dist, _ = cKDTree(xyz).query(xyz, k=4)
    spacing = np.sqrt((dist[:, 1:] ** 2).mean(1)).clip(1e-7)
    n = len(xyz)
    t = lambda a: torch.nn.Parameter(torch.tensor(np.asarray(a, np.float32), device=device))  # noqa: E731
    quats = np.zeros((n, 4), np.float32)
    quats[:, 0] = 1
    shN = np.zeros((n, (SH_DEGREE + 1) ** 2 - 1, 3), np.float32)
    return torch.nn.ParameterDict({
        "means": t(xyz), "scales": t(np.log(spacing)[:, None].repeat(3, 1)), "quats": t(quats),
        "opacities": t(np.full(n, math.log(0.1 / 0.9))), "sh0": t(((rgb - 0.5) / SH_C0)[:, None, :]),
        "shN": t(shN)})


def train(space: Path, steps: int = STEPS, downscale: int = DOWNSCALE, cap: int = CAP,
          use_depth: bool = True, use_exposure: bool = True, out_name: str = "splat-gsplat.ply",
          rasterize=None, device=None, log_every: int = 500, minutes: float | None = None,
          resume: Path | None = None, start_step: int = 0) -> dict:
    """Train and write spaces/<name>/<out_name> and splat-gsplat.json.
    With `minutes`, the step count is cut after BUDGET_PROBE steps to what
    fits the time (a GPU session that ends mid-run returns nothing).
    With `resume` (a splat this trainer wrote) and the `start_step` it reached,
    training continues from those splats towards `steps`, on the same schedule
    as one uninterrupted run: learning rates, SH degree and the depth weight
    are functions of the global step. Per-frame exposure starts again from none.
    `rasterize` replaces gsplat's (tests run a stand-in on the CPU)."""
    import torch
    import torch.nn.functional as F

    device = device or "cuda"
    if rasterize is None:
        from gsplat import rasterization as rasterize
        from gsplat.strategy import MCMCStrategy
        strategy = MCMCStrategy(cap_max=cap, refine_start_iter=max(500, start_step),
                                refine_stop_iter=int(steps * 0.85), refine_every=100)
    else:
        strategy = None

    all_frames = frames(space, downscale)
    train_frames, test_frames = split(all_frames)
    print(f"{len(train_frames)} training frames, {len(test_frames)} held out; "
          f"{train_frames[0]['size'][0]}x{train_frames[0]['size'][1]} each", flush=True)
    t0 = time.time()
    images = [load_image(space, f) for f in train_frames]
    print(f"frames loaded and undistorted in {time.time() - t0:.0f} s", flush=True)
    depths = frame_depths(space, train_frames, device) if use_depth else {}

    if resume:
        raw = read_ply(resume)
        params = torch.nn.ParameterDict({k: torch.nn.Parameter(torch.tensor(v, dtype=torch.float32, device=device))
                                         for k, v in raw.items()})
        print(f"resuming from {resume.name}: {len(params['means']):,} splats at step {start_step}", flush=True)
    else:
        xyz, rgb = read_seed(space / "splat-project/points3D.bin")
        params = init_params(xyz, rgb, device)
    scale = scene_scale([f["view"] for f in all_frames])
    lrs = {"means": 1.6e-4 * scale, "scales": 5e-3, "quats": 1e-3, "opacities": 5e-2,
           "sh0": 2.5e-3, "shN": 2.5e-3 / 20}
    optimizers = {k: torch.optim.Adam([{"params": params[k], "lr": lr, "name": k}], eps=1e-15)
                  for k, lr in lrs.items()}
    exposure = torch.nn.Parameter(torch.eye(3, 4, device=device).repeat(len(train_frames), 1, 1))
    exposure_opt = torch.optim.Adam([exposure], lr=EXPOSURE_LR)
    identity = torch.eye(3, 4, device=device)
    if strategy:
        strategy.check_sanity(params, optimizers)
        state = strategy.initialize_state()

    views = torch.tensor(np.stack([f["view"] for f in train_frames]), dtype=torch.float32, device=device)
    Ks = torch.tensor(np.stack([f["K"] for f in train_frames]), dtype=torch.float32, device=device)
    rng = np.random.default_rng(0)
    order: list[int] = []
    history = []
    t0 = time.time()
    requested = steps
    step = start_step
    while step < steps:
        if minutes and step == start_step + BUDGET_PROBE:
            rate = BUDGET_PROBE / (time.time() - t0)
            # densification gets busier as splats grow: allow for most of the rate
            fits = int(rate * BUDGET_RATE_SHARE * (minutes * 60 - (time.time() - t0))) + step
            if fits < steps:
                steps = max(fits, step + 1000)
                if strategy:
                    strategy.refine_stop_iter = int(steps * 0.85)
                print(f"time budget {minutes:.0f} min: {steps} steps instead of {requested}", flush=True)
        if not order:
            order = list(rng.permutation(len(train_frames)))
        i = order.pop()
        w, h = train_frames[i]["size"]
        img, valid = images[i]
        target = torch.from_numpy(img).to(device, non_blocking=True).float() / 255.0
        mask = torch.from_numpy(valid).to(device)
        sh_degree = min(step // SH_EVERY, SH_DEGREE)
        colours = torch.cat([params["sh0"], params["shN"]], 1)
        renders, alphas, info = rasterize(
            means=params["means"], quats=params["quats"], scales=torch.exp(params["scales"]),
            opacities=torch.sigmoid(params["opacities"]), colors=colours,
            viewmats=views[i:i + 1], Ks=Ks[i:i + 1], width=w, height=h, sh_degree=sh_degree,
            render_mode="RGB+ED", rasterize_mode="antialiased", packed=False)
        rgb_out, depth_out = renders[0, ..., :3], renders[0, ..., 3]
        if use_exposure:
            M = exposure[i]
            rgb_out = rgb_out @ M[:, :3].T + M[:, 3]
        diff = (rgb_out - target).abs()[mask]
        l1 = diff.mean()
        m = mask[None, None].float()
        loss_ssim = 1 - ssim((rgb_out.permute(2, 0, 1)[None] * m), (target.permute(2, 0, 1)[None] * m))
        loss = (1 - SSIM_WEIGHT) * l1 + SSIM_WEIGHT * loss_ssim
        name = train_frames[i]["name"]
        depth_loss = torch.zeros((), device=device)
        if name in depths:
            mono = torch.from_numpy(depths[name].astype(np.float32)).to(device)
            rendered = F.interpolate(depth_out[None, None], size=mono.shape, mode="area")[0, 0]
            alpha_small = F.interpolate(alphas[0, ..., 0][None, None], size=mono.shape, mode="area")[0, 0]
            ok = (mono > 0) & (alpha_small > 0.5) & (rendered > 1e-4)
            if ok.any():
                depth_loss = (1 / rendered[ok] - 1 / mono[ok]).abs().mean() * scale
                frac = step / steps
                weight = DEPTH_WEIGHT[0] * (DEPTH_WEIGHT[1] / DEPTH_WEIGHT[0]) ** frac
                loss = loss + weight * depth_loss
        if use_exposure:
            loss = loss + EXPOSURE_REG * (exposure[i] - identity).abs().mean()
        if strategy:
            loss = loss + OPACITY_REG * torch.sigmoid(params["opacities"]).mean() \
                + SCALE_REG * torch.exp(params["scales"]).mean()
            if hasattr(strategy, "step_pre_backward"):
                strategy.step_pre_backward(params, optimizers, state, step, info)
        loss.backward()
        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
        exposure_opt.step()
        exposure_opt.zero_grad(set_to_none=True)
        for group in optimizers["means"].param_groups:
            group["lr"] = lrs["means"] * 0.01 ** ((step + 1) / steps)
        if strategy:
            strategy.step_post_backward(params, optimizers, state, step, info,
                                        lr=optimizers["means"].param_groups[0]["lr"])
        if step % log_every == 0 or step == steps - 1:
            rate = (step + 1 - start_step) / max(time.time() - t0, 1e-6)
            history.append({"step": step, "l1": round(float(l1.detach()), 4), "depth": round(float(depth_loss.detach()), 4),
                            "splats": len(params["means"])})
            print(f"step {step}/{steps}  l1 {history[-1]['l1']:.4f}  depth {history[-1]['depth']:.4f}  "
                  f"splats {len(params['means'])}  {rate:.2f} steps/s  "
                  f"~{(steps - step) / rate / 60:.0f} min left", flush=True)
        if step + 1 in SAVE_AT and step + 1 < steps:
            save(params, space / out_name.replace(".ply", f"_{step + 1}.ply"))
        step += 1

    save(params, space / out_name)
    report = {"trainer": "gsplat", "steps": steps, "steps_requested": requested, "downscale": downscale, "cap": cap,
              "depth": use_depth, "exposure": use_exposure,
              "frames_with_depth": len(depths), "train": [f["name"] for f in train_frames],
              "held_out": [f["name"] for f in test_frames], "splats": len(params["means"]),
              "minutes": round((time.time() - t0) / 60, 1), "history": history,
              "exposure_spread": float((exposure.detach() - identity).abs().mean())}
    (space / "splat-gsplat.json").write_text(json.dumps(report, indent=1))
    return report


def save(params, path: Path) -> None:
    p = {k: v.detach().cpu().numpy() for k, v in params.items()}
    write_ply(path, p["means"], p["scales"], p["quats"], p["opacities"], p["sh0"], p["shN"])
    print(f"wrote {path.name}: {len(p['means'])} splats", flush=True)


# ------------------------------------------------------------------------- eval

def evaluate(space: Path, plys: list, downscale: int = EVAL_DOWNSCALE, rasterize=None,
             device=None) -> dict:
    """Score each splat on the held-out frames; write splat-eval.json and
    splat-eval/<frame>.jpg (the frame, then each splat's render)."""
    import cv2
    import torch

    device = device or "cuda"
    if rasterize is None:
        from gsplat import rasterization as rasterize
    all_frames = frames(space, downscale)
    _, test = split(all_frames)
    trained_on = {}
    report_path = space / "splat-gsplat.json"
    if report_path.exists():
        held = set(json.loads(report_path.read_text()).get("held_out", []))
        test = [f for f in all_frames if f["name"] in held] or test
    out_dir = space / "splat-eval"
    out_dir.mkdir(exist_ok=True)
    entries = [(p if isinstance(p, tuple) else (p, None)) for p in plys]
    names = [path.name + (f":{variant}" if variant else "") for path, variant in entries]
    results = {n: {"psnr": [], "ssim": [], "psnr_raw": []} for n in names}
    loaded = {}
    for (p, variant), name in zip(entries, names):
        raw = variant_arrays(read_ply(p), variant, space)
        t = lambda a: torch.tensor(a, dtype=torch.float32, device=device)  # noqa: E731
        degree = int(round(math.sqrt(raw["shN"].shape[1] + 1))) - 1
        loaded[name] = (t(raw["means"]), t(raw["quats"]), torch.exp(t(raw["scales"])),
                          torch.sigmoid(t(raw["opacities"])),
                          torch.cat([t(raw["sh0"]), t(raw["shN"])], 1), degree)
        trained_on[name] = ("never saw them" if "gsplat" in p.name or "heldout" in p.name
                            else "held-out frames were in its training")
    for frame in test:
        img, valid = load_image(space, frame)
        truth = img.astype(np.float32) / 255
        w, h = frame["size"]
        tiles = [img]
        for name in names:
            means, quats, scales, opac, colours, degree = loaded[name]
            with torch.no_grad():
                renders, _, _ = rasterize(
                    means=means, quats=quats, scales=scales, opacities=opac, colors=colours,
                    viewmats=torch.tensor(frame["view"][None], dtype=torch.float32, device=device),
                    Ks=torch.tensor(frame["K"][None], dtype=torch.float32, device=device),
                    width=w, height=h, sh_degree=degree, rasterize_mode="antialiased")
            render = renders[0, ..., :3].clamp(0, 1).cpu().numpy()
            matched = colour_match(render, truth, valid).astype(np.float32)
            r = results[name]
            r["psnr_raw"].append(psnr(render, truth, valid))
            r["psnr"].append(psnr(matched, truth, valid))
            r["ssim"].append(float(ssim(torch.tensor(matched).permute(2, 0, 1)[None],
                                        torch.tensor(truth).permute(2, 0, 1)[None])))
            tiles.append((render * 255).astype(np.uint8))
        strip = np.concatenate(tiles, axis=1)
        labels = ["video frame"] + [n.removesuffix(".ply").replace(".ply", "") for n in names]
        for k, label in enumerate(labels):
            cv2.putText(strip, label, (k * w + 20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 0), 3)
        small = cv2.resize(strip, (strip.shape[1] // 2, strip.shape[0] // 2), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out_dir / f"{Path(frame['name']).stem}.jpg"), small[:, :, ::-1],
                    [cv2.IMWRITE_JPEG_QUALITY, 85])
    summary = {"held_out": [f["name"] for f in test], "downscale": downscale,
               "scored_after": "one 3x4 colour match per frame (psnr_raw: without it)",
               "splats": {name: {"psnr": round(float(np.mean(r["psnr"])), 2),
                                 "psnr_raw": round(float(np.mean(r["psnr_raw"])), 2),
                                 "ssim": round(float(np.mean(r["ssim"])), 3),
                                 "per_frame_psnr": [round(x, 2) for x in r["psnr"]],
                                 "note": trained_on[name]}
                          for name, r in results.items()}}
    (space / "splat-eval.json").write_text(json.dumps(summary, indent=1))
    for name, s in summary["splats"].items():
        print(f"{name}: PSNR {s['psnr']} (raw {s['psnr_raw']}), SSIM {s['ssim']} - {s['note']}", flush=True)
    return summary


# ------------------------------------------------------- web-viewer stand-ins, novel views
WEB_SPLATS = 300_000         # what a web page can carry of a splat
ROOM_BOX_M = (1.7, 1.95, -0.15, 2.85)   # |x|, |z| and the y range of the room in the viewer's frame, metres


def room_to_viewer(space: Path):
    """The map from the solve's frame to the viewer's (metres, y up, floor 0, room centred)."""
    sf = json.loads((space / "scene" / "scene.json").read_text())["solveFrame"]
    W, T, o, m = np.array(sf["world"]), np.array(sf["toViewer"]), np.array(sf["origin"]), sf["unitsPerMetre"]
    return (lambda p: (((p @ W.T) - o) / m) @ T.T), m


def variant_arrays(raw: dict, variant: str | None, space: Path) -> dict:
    """The splat as a web page would carry it. `dc`: flat colour only (no
    view-dependent shine). `dc300k`: also cropped to the room and reduced to the
    WEB_SPLATS most important splats (opacity times volume), like the page's copy."""
    if not variant:
        return raw
    out = dict(raw)
    out["shN"] = np.zeros((len(raw["means"]), 0, 3), np.float32)
    if variant == "dc":
        return out
    if variant != "dc300k":
        raise SystemExit(f"unknown variant {variant}")
    to_viewer, _ = room_to_viewer(space)
    v = to_viewer(raw["means"].astype(np.float64))
    bx, bz, y0, y1 = ROOM_BOX_M
    inside = (np.abs(v[:, 0]) < bx) & (np.abs(v[:, 2]) < bz) & (v[:, 1] > y0) & (v[:, 1] < y1)
    importance = 1 / (1 + np.exp(-raw["opacities"])) * np.exp(raw["scales"]).prod(1)
    importance[~inside] = -1
    keep = np.argsort(-importance)[:WEB_SPLATS]
    keep = keep[importance[keep] >= 0]
    return {k: val[keep] for k, val in out.items()}


def perturbed(view: np.ndarray, shift: float, degrees: float) -> np.ndarray:
    """A camera near this one: turned about its own vertical axis and moved sideways."""
    a = math.radians(degrees)
    Ry = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
    R = Ry @ view[:3, :3]
    centre = -view[:3, :3].T @ view[:3, 3] + view[:3, :3].T @ np.array([shift, 0, 0])
    out = np.eye(4)
    out[:3, :3], out[:3, 3] = R, -R @ centre
    return out


def novel(space: Path, variants: list[tuple[Path, str | None]], downscale: int = EVAL_DOWNSCALE,
          rasterize=None, device=None, shift_m: float = 0.25, degrees: float = 9.0) -> None:
    """Renders of the splat from cameras beside the held-out frames (moved
    sideways and turned): views nobody trained on or filmed. One strip per frame:
    the nearest filmed frame, then each variant. splat-novel/<frame>.jpg."""
    import cv2
    import torch

    device = device or "cuda"
    if rasterize is None:
        from gsplat import rasterization as rasterize
    _, m = room_to_viewer(space)
    held = set(json.loads((space / "splat-gsplat.json").read_text()).get("held_out", []))
    shown = [f for f in frames(space, downscale) if f["name"] in held]
    out_dir = space / "splat-novel"
    out_dir.mkdir(exist_ok=True)
    t = lambda a: torch.tensor(a, dtype=torch.float32, device=device)  # noqa: E731
    loaded = []
    for path, variant in variants:
        raw = variant_arrays(read_ply(path), variant, space)
        degree = int(round(math.sqrt(raw["shN"].shape[1] + 1))) - 1
        loaded.append((f"{path.stem}{':' + variant if variant else ''} ({len(raw['means']):,})",
                       t(raw["means"]), t(raw["quats"]), torch.exp(t(raw["scales"])),
                       torch.sigmoid(t(raw["opacities"])), torch.cat([t(raw["sh0"]), t(raw["shN"])], 1), degree))
    for i, frame in enumerate(shown):
        img, _ = load_image(space, frame)
        w, h = frame["size"]
        view = perturbed(frame["view"], (1 if i % 2 == 0 else -1) * shift_m * m, degrees * (1 if i % 2 == 0 else -1))
        tiles = [img]
        for label, means, quats, scales, opac, colours, degree in loaded:
            with torch.no_grad():
                renders, _, _ = rasterize(means=means, quats=quats, scales=scales, opacities=opac, colors=colours,
                                          viewmats=t(view[None]), Ks=t(frame["K"][None]), width=w, height=h,
                                          sh_degree=degree, rasterize_mode="antialiased")
            tiles.append((renders[0, ..., :3].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8))
        strip = np.concatenate(tiles, axis=1)
        for k, label in enumerate(["filmed frame (nearby)"] + [x[0] for x in loaded]):
            cv2.putText(strip, label, (k * w + 12, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
        small = cv2.resize(strip, (strip.shape[1] // 2, strip.shape[0] // 2), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out_dir / f"{Path(frame['name']).stem}.jpg"), small[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"wrote {len(shown)} novel-view strips to {out_dir}", flush=True)


DEPTH_MAP_DOWNSCALE = 8      # depth and coverage maps for gap recovery, per solved frame


def depth_maps(space: Path, ply: Path, downscale: int = DEPTH_MAP_DOWNSCALE, rasterize=None, device=None) -> Path:
    """What the trained splat shows from every solved camera: the expected depth
    (solve units) and the opacity it has gathered, per pixel of the undistorted
    frame at 1/downscale. Gap recovery uses them to know what each frame really
    sees (walls, clutter and all) instead of guessing from the keyframes' cloud.
    Written to splat-depths.npz: <name>/depth (float16), <name>/alpha (uint8),
    and the downscale."""
    import torch

    device = device or "cuda"
    if rasterize is None:
        from gsplat import rasterization as rasterize
    raw = read_ply(ply)
    t = lambda a: torch.tensor(a, dtype=torch.float32, device=device)  # noqa: E731
    means, quats, scales = t(raw["means"]), t(raw["quats"]), torch.exp(t(raw["scales"]))
    opac, colours = torch.sigmoid(t(raw["opacities"])), t(raw["sh0"])
    out = {"downscale": np.array(downscale)}
    for frame in frames(space, downscale):
        w, h = frame["size"]
        with torch.no_grad():
            renders, alphas, _ = rasterize(means=means, quats=quats, scales=scales, opacities=opac, colors=colours,
                                           viewmats=t(frame["view"][None]), Ks=t(frame["K"][None]), width=w, height=h,
                                           sh_degree=0, render_mode="RGB+ED", rasterize_mode="antialiased")
        out[f"{frame['name']}/depth"] = renders[0, ..., 3].cpu().numpy().astype(np.float16)
        out[f"{frame['name']}/alpha"] = (alphas[0, ..., 0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    path = space / "splat-depths.npz"
    np.savez_compressed(path, **out)
    print(f"wrote {path.name}: depth and coverage for {(len(out) - 1) // 2} frames from {ply.name}", flush=True)
    return path


def parse_variants(space: Path, names: list[str]) -> list[tuple[Path, str | None]]:
    out = []
    for name in names:
        file, _, variant = name.partition(":")
        path = Path(file)
        out.append((path if path.is_absolute() else space / path, variant or None))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("space", type=Path)
    t.add_argument("--steps", type=int, default=STEPS)
    t.add_argument("--downscale", type=int, default=DOWNSCALE)
    t.add_argument("--cap", type=int, default=CAP)
    t.add_argument("--no-depth", action="store_true")
    t.add_argument("--no-exposure", action="store_true")
    t.add_argument("--output", default="splat-gsplat.ply")
    t.add_argument("--minutes", type=float, help="time budget for the training loop")
    t.add_argument("--resume", type=Path, help="continue from a splat this trainer wrote")
    t.add_argument("--start-step", type=int, default=0, help="the step that splat had reached")
    e = sub.add_parser("eval")
    e.add_argument("space", type=Path)
    e.add_argument("plys", nargs="+")
    e.add_argument("--downscale", type=int, default=EVAL_DOWNSCALE)
    d = sub.add_parser("depths", help="depth and coverage per solved frame, for gap recovery")
    d.add_argument("space", type=Path)
    d.add_argument("ply")
    n = sub.add_parser("novel", help="renders beside the held-out frames, for looking at")
    n.add_argument("space", type=Path)
    n.add_argument("plys", nargs="+", help="splat files, each optionally :dc or :dc300k")
    n.add_argument("--downscale", type=int, default=EVAL_DOWNSCALE)
    args = ap.parse_args()
    if args.cmd == "train":
        train(args.space, args.steps, args.downscale, args.cap, not args.no_depth, not args.no_exposure,
              args.output, minutes=args.minutes,
              resume=(args.resume if args.resume is None or args.resume.is_absolute() else args.space / args.resume),
              start_step=args.start_step)
    elif args.cmd == "depths":
        depth_maps(args.space, args.space / args.ply)
    elif args.cmd == "eval":
        evaluate(args.space, parse_variants(args.space, args.plys), args.downscale)
    else:
        novel(args.space, parse_variants(args.space, args.plys), args.downscale)


if __name__ == "__main__":
    main()
