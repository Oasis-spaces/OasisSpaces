#!/usr/bin/env python3
"""Place every frame of a space with MapAnything, on a GPU.

COLMAP's mapper is the step that breaks on a phone video of a room: after a
turn, or along a plain wall, it cannot register the next frames and the
capture splits into pieces (the walkthrough split in two around frame 112 and
placed 90% of its frames). MapAnything (facebookresearch/map-anything; the
Apache-2.0 checkpoint facebook/map-anything-apache) predicts, in one pass over
all the frames together, every camera's pose and intrinsics and a metric
depth map for each frame. It has nothing to register, so it cannot split.

Its cameras become priors, not the answer: reconstruct.py --mapper priors
triangulates COLMAP's own SIFT matches with the cameras where MapAnything put
them, adjusts, and triangulates again, so everything downstream gets the
multi-view tracks and precision it gets from a COLMAP solve.

Writes, in <space>/workspace/:
  - pose-priors.json      every frame's cam_from_world and the shared camera
                          (the format reconstruct.read_priors documents)
  - mapanything/views.npz each frame's cam2world, intrinsics, confidence and
                          depth at the model's resolution
  - mapanything/points.ply the confident points of all frames, fused, in colour
  - mapanything/solve.json what ran: model, views, seconds, GPU, memory

Guided by COLMAP (--guide <model dir>). On its own MapAnything guessed the
walkthrough's lens 30% too wide and its cameras landed 1.6 m from COLMAP's
(Sep 2026). But COLMAP had solved most of that video precisely and lost only
the frames around a turn and at the end. So the guided mode keeps COLMAP's
cameras and asks MapAnything only for the rest: it is given COLMAP's lens for
every frame and COLMAP's poses for the frames COLMAP placed, and it places the
others in the same frame (its own example does this with COLMAP outputs).
Some placed frames are held back unposed, to measure how far off it puts a
frame COLMAP knows (mapanything/guided.json).

Usage (on the GPU machine, after reconstruct.py --frames-only):
    python pipeline/mapanything_solve.py spaces/<name> [--stride 1]
    python pipeline/mapanything_solve.py spaces/<name> --guide spaces/<name>/workspace/sparse/0
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

MODEL = "facebook/map-anything-apache"      # Apache-2.0; facebook/map-anything is CC-BY-NC
POINT_VOXEL_M = 0.02
CONF_PERCENTILE = 30                        # points below this confidence (per frame) are not fused
MAX_POINTS_PER_VIEW = 60_000
GUIDED_VIEWS = 94                           # what fitted a T4 (16 GB) in half precision
HOLDOUT_EVERY = 12                          # one placed frame in this many goes in unposed, as a check
MIN_VIEWS = 24                              # below this, out of memory is an error, not a reason to thin out


def quaternion(R: np.ndarray) -> list[float]:
    """(w, x, y, z) of a rotation matrix, COLMAP's order."""
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        return [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    i = int(np.argmax(np.diag(R)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = math.sqrt(R[i, i] - R[j, j] - R[k, k] + 1.0) * 2
    q = [0.0] * 4
    q[1 + i] = 0.25 * s
    q[0] = (R[k, j] - R[j, k]) / s
    q[1 + j] = (R[j, i] + R[i, j]) / s
    q[1 + k] = (R[k, i] + R[i, k]) / s
    return q


def crop_of(width: int, height: int, target_w: int, target_h: int):
    """How MapAnything's loader fits a frame to its resolution (utils/cropping:
    scale to cover the target, floor the size, crop the middle): per-axis
    scale and the crop's top-left, to take its intrinsics back to the frame."""
    scale = max(target_w / width, target_h / height) + 1e-8
    resized_w, resized_h = math.floor(width * scale), math.floor(height * scale)
    return (resized_w / width, resized_h / height,
            (resized_w - target_w) // 2, (resized_h - target_h) // 2)


def to_frame_intrinsics(K: np.ndarray, sx: float, sy: float, left: int, top: int) -> np.ndarray:
    """Intrinsics of the model's cropped, resized image -> the full frame's, in
    the pixel-centre convention (a pixel's centre at its integer index), where
    a processed pixel u came from frame pixel (u + 0.5 + left) / sx - 0.5
    (checked against the loader's resize and crop in tools/tests). The priors
    only use the focal lengths, which no convention changes."""
    out = np.eye(3)
    out[0, 0], out[1, 1] = K[0, 0] / sx, K[1, 1] / sy
    out[0, 2] = (K[0, 2] + 0.5 + left) / sx - 0.5
    out[1, 2] = (K[1, 2] + 0.5 + top) / sy - 0.5
    return out


def interpolate_pose(a: np.ndarray, b: np.ndarray, w: float) -> np.ndarray:
    """cam2world between two poses: slerp of the rotation, lerp of the position.
    A frame skipped by --stride gets this as its prior; the adjustment in
    reconstruct.py --mapper priors then moves it to where its matches say."""
    from scipy.spatial.transform import Rotation, Slerp

    rotations = Rotation.from_matrix(np.stack([a[:3, :3], b[:3, :3]]))
    out = np.eye(4)
    out[:3, :3] = Slerp([0, 1], rotations)([w]).as_matrix()[0]
    out[:3, 3] = (1 - w) * a[:3, 3] + w * b[:3, 3]
    return out


def similarity(src: np.ndarray, dst: np.ndarray):
    """(scale, R, t) with dst ~ scale * R @ src + t, least squares (Umeyama)."""
    mu_s, mu_d = src.mean(axis=0), dst.mean(axis=0)
    a, b = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(b.T @ a / len(src))
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    R = U @ D @ Vt
    scale = float(np.trace(np.diag(S) @ D) / (a ** 2).sum(axis=1).mean())
    return scale, R, mu_d - scale * R @ mu_s


def read_guide(model: Path):
    """A COLMAP model's camera (the pipeline's single OPENCV camera) and every
    placed frame's cam2world."""
    from densify import read_cameras_bin, read_images_bin

    cameras = read_cameras_bin(model / "cameras.bin")
    if len(cameras) != 1:
        raise SystemExit(f"{model} has {len(cameras)} cameras; the guide needs the pipeline's one camera")
    cam = next(iter(cameras.values()))
    params = [float(v) for v in cam["params"]]
    camera = {"width": cam["width"], "height": cam["height"], "fx": params[0], "fy": params[1],
              "cx": params[2], "cy": params[3],
              **dict(zip(("k1", "k2", "p1", "p2"), params[4:8] if len(params) >= 8 else [0.0] * 4))}
    poses = {}
    for info in read_images_bin(model / "images.bin").values():
        pose = np.eye(4)
        pose[:3, :3] = info["R"].T
        pose[:3, 3] = -info["R"].T @ info["t"]
        poses[info["name"]] = pose
    return camera, poses


def pick_guided(images: list, known: dict, budget: int, holdout_every: int):
    """The frames to give the model: every frame COLMAP did not place, a few it
    did place held back unposed as a check, and as many posed ones, spread
    along the walk, as the budget leaves. The first is posed: MapAnything
    takes its frame from the first view."""
    unknown = [p for p in images if p.name not in known]
    placed = [p for p in images if p.name in known]
    held = placed[holdout_every // 2::holdout_every] if holdout_every else []
    rest = [p for p in placed if p not in held]
    room = max(2, min(len(rest), budget - len(unknown) - len(held)))
    anchors = [rest[i] for i in sorted(set(np.linspace(0, len(rest) - 1, room).round().astype(int)))]
    chosen = sorted(set(anchors) | set(unknown) | set(held), key=lambda p: p.name)
    first = next(p for p in chosen if p in anchors)
    chosen.remove(first)
    return [first] + chosen, {p.name for p in anchors}, {p.name for p in held}


def voxel_fuse(points: np.ndarray, colours: np.ndarray, voxel: float):
    """One point per voxel: the mean position and colour of what fell in it."""
    keys = np.floor(points / voxel).astype(np.int64)
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.ravel()
    fused = np.zeros((len(counts), 3))
    tint = np.zeros((len(counts), 3))
    np.add.at(fused, inverse, points)
    np.add.at(tint, inverse, colours)
    return fused / counts[:, None], tint / counts[:, None], counts


def write_ply(path: Path, points: np.ndarray, colours: np.ndarray) -> None:
    header = (f"ply\nformat binary_little_endian 1.0\nelement vertex {len(points)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    rows = np.zeros(len(points), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                        ("r", "u1"), ("g", "u1"), ("b", "u1")])
    rows["x"], rows["y"], rows["z"] = points.T
    rows["r"], rows["g"], rows["b"] = np.clip(colours, 0, 255).astype(np.uint8).T
    path.write_bytes(header.encode() + rows.tobytes())


def solve(space: Path, stride: int = 1, log=print) -> dict:
    import torch
    from mapanything.models import MapAnything
    from mapanything.utils.image import load_images
    from PIL import Image

    workspace = space / "workspace"
    images = sorted((workspace / "images").glob("*.jpg"))
    if len(images) < 2:
        raise SystemExit(f"no frames in {workspace / 'images'}: run reconstruct.py --frames-only first")
    placed = images[::stride]
    if placed[-1] != images[-1]:
        placed.append(images[-1])               # the last frame anchors the tail of the walk
    width, height = Image.open(images[0]).size
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Turing (a T4) has no native bfloat16: half precision there.
    amp = "bf16" if device == "cuda" and torch.cuda.get_device_capability()[0] >= 8 else "fp16"
    log(f"{len(placed)} of {len(images)} frames into {MODEL} on "
        f"{torch.cuda.get_device_name(0) if device == 'cuda' else 'cpu'} ({amp})")
    started = time.time()
    model = MapAnything.from_pretrained(MODEL).to(device).eval()
    while True:
        views = load_images([str(p) for p in placed])
        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        try:
            with torch.no_grad():
                # One view at a time through the dense heads, as MapAnything's own COLMAP export does.
                outputs = model.infer(views, memory_efficient_inference=True, minibatch_size=1,
                                      use_amp=True, amp_dtype=amp, apply_mask=True, mask_edges=True)
            break
        except torch.OutOfMemoryError:
            if len(placed) <= MIN_VIEWS:
                raise
            # All frames attend to each other, so memory grows with their number:
            # give the model every other frame and interpolate the rest.
            stride *= 2
            placed = images[::stride] + ([images[-1]] if (len(images) - 1) % stride else [])
            del views
            log(f"  out of GPU memory; one frame in {stride} instead ({len(placed)} of {len(images)})")
    seconds = round(time.time() - started)
    peak = round(torch.cuda.max_memory_allocated() / 1e9, 1) if device == "cuda" else None

    target_h, target_w = (int(v) for v in views[0]["true_shape"][0])
    sx, sy, left, top = crop_of(width, height, target_w, target_h)
    cam2world, intrinsics, confidence, depths = [], [], [], []
    points, colours = [], []
    rng = np.random.default_rng(0)
    for out in outputs:
        pose = out["camera_poses"][0].float().cpu().numpy().astype(np.float64)
        K = out["intrinsics"][0].float().cpu().numpy().astype(np.float64)
        conf = out["conf"][0].float().cpu().numpy()
        mask = out["mask"][0].squeeze(-1).cpu().numpy().astype(bool)
        cam2world.append(pose)
        intrinsics.append(to_frame_intrinsics(K, sx, sy, left, top))
        confidence.append(conf.astype(np.float16))
        depths.append(out["depth_z"][0].squeeze(-1).float().cpu().numpy().astype(np.float16))
        keep = mask & (conf >= np.percentile(conf[mask], CONF_PERCENTILE) if mask.any() else mask)
        index = np.flatnonzero(keep.ravel())
        if len(index) > MAX_POINTS_PER_VIEW:
            index = rng.choice(index, MAX_POINTS_PER_VIEW, replace=False)
        pts = out["pts3d"][0].float().cpu().numpy().reshape(-1, 3)[index]
        rgb = out["img_no_norm"][0].float().cpu().numpy().reshape(-1, 3)[index]
        points.append(pts)
        colours.append(rgb * 255 if rgb.max() <= 1.0 else rgb)
    cam2world = np.stack(cam2world)
    intrinsics = np.stack(intrinsics)

    # Every frame's pose: the placed ones from the model, the ones --stride
    # skipped from their placed neighbours.
    placed_names = [p.name for p in placed]
    order = {name: n for n, name in enumerate(placed_names)}
    poses = {}
    for n, image in enumerate(images):
        if image.name in order:
            poses[image.name] = cam2world[order[image.name]]
            continue
        before = max(i for i, p in enumerate(images[:n]) if p.name in order)
        after = min(i for i, p in enumerate(images) if i > n and p.name in order)
        poses[image.name] = interpolate_pose(cam2world[order[images[before].name]],
                                             cam2world[order[images[after].name]],
                                             (n - before) / (after - before))

    # One camera for the whole video: the median focal length; the principal
    # point at the frame's centre (a phone's is within a few pixels of it, and
    # the adjustment keeps it fixed).
    fx, fy = float(np.median(intrinsics[:, 0, 0])), float(np.median(intrinsics[:, 1, 1]))
    frames = {}
    for name, pose in poses.items():
        R = pose[:3, :3].T                          # world -> camera
        t = -R @ pose[:3, 3]
        frames[name] = {"qvec": [round(v, 10) for v in quaternion(R)], "tvec": [round(float(v), 8) for v in t]}
    priors = {"source": f"mapanything ({MODEL})", "metric": True,
              "camera": {"width": width, "height": height, "fx": round(fx, 3), "fy": round(fy, 3),
                         "cx": width / 2, "cy": height / 2},
              "frames": frames}
    (workspace / "pose-priors.json").write_text(json.dumps(priors, indent=1) + "\n")

    out_dir = workspace / "mapanything"
    out_dir.mkdir(exist_ok=True)
    fused, tint, counts = voxel_fuse(np.vstack(points), np.vstack(colours), POINT_VOXEL_M)
    write_ply(out_dir / "points.ply", fused, tint)
    np.savez_compressed(out_dir / "views.npz", names=np.array(placed_names), cam2world=cam2world,
                        intrinsics=intrinsics, confidence=np.stack(confidence), depth=np.stack(depths),
                        crop=np.array([sx, sy, left, top], float), frame_size=np.array([width, height]))
    positions = cam2world[:, :3, 3]
    record = {"model": MODEL, "views": len(placed), "frames": len(images), "stride": stride,
              "seconds": seconds, "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
              "amp": amp, "peak_gpu_gb": peak, "model_size": [target_w, target_h],
              "focal_px": [round(fx, 1), round(fy, 1)],
              "focal_spread": round(float(np.std(intrinsics[:, 0, 0]) / fx), 4),
              # where the model puts the principal point, from the frame's centre (either convention)
              "principal_offset_px": [round(float(np.median(intrinsics[:, 0, 2])) - (width - 1) / 2, 1),
                                      round(float(np.median(intrinsics[:, 1, 2])) - (height - 1) / 2, 1)],
              "points": int(len(fused)),
              "path_m": round(float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum()), 2),
              "extent_m": np.round(np.percentile(fused, 98, axis=0) - np.percentile(fused, 2, axis=0), 2).tolist()}
    (out_dir / "solve.json").write_text(json.dumps(record, indent=1) + "\n")
    log(f"placed {len(images)} frames in {seconds}s (peak {peak} GB); {len(fused):,} points, "
        f"a {record['path_m']} m walk; focal {fx:.0f}px")
    return record


def solve_guided(space: Path, guide: Path, budget: int = GUIDED_VIEWS,
                 holdout_every: int = HOLDOUT_EVERY, log=print) -> dict:
    import torch
    from mapanything.models import MapAnything
    from mapanything.utils.image import preprocess_inputs
    from PIL import Image

    workspace = space / "workspace"
    images = sorted((workspace / "images").glob("*.jpg"))
    camera, known = read_guide(guide)
    missing = [p.name for p in images if p.name not in known]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = "bf16" if device == "cuda" and torch.cuda.get_device_capability()[0] >= 8 else "fp16"
    K = np.array([[camera["fx"], 0, camera["cx"]], [0, camera["fy"], camera["cy"]], [0, 0, 1]], np.float32)
    model = MapAnything.from_pretrained(MODEL).to(device).eval()
    started = time.time()
    while True:
        chosen, anchors, held = pick_guided(images, known, budget, holdout_every)
        log(f"{len(chosen)} frames into {MODEL} ({amp}): {len(anchors)} with COLMAP's poses, "
            f"{len(held)} of COLMAP's held back as a check, {len(missing)} COLMAP did not place")
        raw = []
        for path in chosen:
            view = {"img": torch.from_numpy(np.asarray(Image.open(path).convert("RGB"))),
                    "intrinsics": torch.from_numpy(K)}
            if path.name in anchors:
                view["camera_poses"] = torch.from_numpy(known[path.name].astype(np.float32))
                view["is_metric_scale"] = torch.tensor([False])      # COLMAP's units, not metres
            raw.append(view)
        views = preprocess_inputs(raw)
        del raw
        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        try:
            with torch.no_grad():
                outputs = model.infer(views, memory_efficient_inference=True, minibatch_size=1,
                                      use_amp=True, amp_dtype=amp, apply_mask=True, mask_edges=True)
            break
        except torch.OutOfMemoryError:
            del views
            if budget <= MIN_VIEWS + len(missing):
                raise
            budget = int(budget * 0.8)
            log(f"  out of GPU memory; {budget} frames instead")
    seconds = round(time.time() - started)
    peak = round(torch.cuda.max_memory_allocated() / 1e9, 1) if device == "cuda" else None

    predicted = {path.name: out["camera_poses"][0].float().cpu().numpy().astype(np.float64)
                 for path, out in zip(chosen, outputs)}
    # Its frame onto COLMAP's, through the frames it was given poses for (identity if it kept them).
    names = sorted(anchors)
    scale, R, t = similarity(np.array([predicted[n][:3, 3] for n in names]),
                             np.array([known[n][:3, 3] for n in names]))

    def into_colmap(pose):
        out = np.eye(4)
        out[:3, :3] = R @ pose[:3, :3]
        out[:3, 3] = scale * R @ pose[:3, 3] + t
        return out

    placed = {n: into_colmap(p) for n, p in predicted.items()}
    # Metres: densify's measured scale, else MapAnything's own, since its poses are metric
    # and the fit maps its metres onto COLMAP's units (walkthrough: 3.02 vs densify's 2.84).
    units = scale
    meta = space / "densify.json"
    if meta.exists():
        units = float(json.loads(meta.read_text()).get("colmap_units_per_metre") or scale)

    def angle(a, b):
        return float(np.degrees(np.arccos(np.clip((np.trace(a[:3, :3].T @ b[:3, :3]) - 1) / 2, -1, 1))))

    anchor_err = [np.linalg.norm(placed[n][:3, 3] - known[n][:3, 3]) / units for n in names]
    check = {n: {"position_m": round(float(np.linalg.norm(placed[n][:3, 3] - known[n][:3, 3]) / units), 3),
                 "rotation_deg": round(angle(placed[n], known[n]), 2)} for n in sorted(held)}

    # The priors: COLMAP's cameras where it has them, MapAnything's for the rest.
    frames = {}
    for path in images:
        pose = known.get(path.name)
        if pose is None:
            pose = placed[path.name]
        Rw = pose[:3, :3].T
        frames[path.name] = {"qvec": [round(v, 10) for v in quaternion(Rw)],
                             "tvec": [round(float(v), 8) for v in -Rw @ pose[:3, 3]]}
    priors = {"source": f"COLMAP ({guide.name}), MapAnything ({MODEL}) for {len(missing)} frames it lost",
              "metric": False, "camera": camera, "frames": frames,
              "from_mapanything": missing}
    (workspace / "pose-priors.json").write_text(json.dumps(priors, indent=1) + "\n")

    out_dir = workspace / "mapanything"
    out_dir.mkdir(exist_ok=True)
    points, colours = [], []
    for out in outputs:
        mask = out["mask"][0].squeeze(-1).cpu().numpy().astype(bool)
        conf = out["conf"][0].float().cpu().numpy()
        keep = mask & (conf >= np.percentile(conf[mask], CONF_PERCENTILE)) if mask.any() else mask
        index = np.flatnonzero(keep.ravel())[:MAX_POINTS_PER_VIEW]
        pts = out["pts3d"][0].float().cpu().numpy().reshape(-1, 3)[index]
        rgb = out["img_no_norm"][0].float().cpu().numpy().reshape(-1, 3)[index]
        points.append((scale * (R @ pts.T)).T + t)
        colours.append(rgb * 255 if rgb.max() <= 1.0 else rgb)
    fused, tint, _ = voxel_fuse(np.vstack(points), np.vstack(colours), POINT_VOXEL_M * units)
    write_ply(out_dir / "points.ply", fused, tint)
    chosen_names = [p.name for p in chosen]
    np.savez_compressed(out_dir / "views.npz", names=np.array(chosen_names),
                        cam2world=np.stack([placed[n] for n in chosen_names]),
                        intrinsics=np.stack([K.astype(np.float64)] * len(chosen_names)),
                        frame_size=np.array([camera["width"], camera["height"]]),
                        role=np.array(["anchor" if n in anchors else "check" if n in held else "placed"
                                       for n in chosen_names]))
    held_err = [c["position_m"] for c in check.values()]
    record = {"model": MODEL, "guide": str(guide), "seconds": seconds, "peak_gpu_gb": peak, "amp": amp,
              "views": len(chosen), "anchors": len(anchors), "checks": len(held), "placed": missing,
              "units_per_metre": units, "fit_scale": round(scale, 4),
              "anchor_error_m": {"median": round(float(np.median(anchor_err)), 3),
                                 "max": round(float(np.max(anchor_err)), 3)},
              "check_error_m": {"median": round(float(np.median(held_err)), 3) if held_err else None,
                                "max": round(float(np.max(held_err)), 3) if held_err else None},
              "check_rotation_deg": {"median": round(float(np.median([c["rotation_deg"] for c in check.values()])), 2)
                                     if check else None},
              "checks_by_frame": check}
    (out_dir / "guided.json").write_text(json.dumps(record, indent=1) + "\n")
    log(f"placed {len(missing)} frames COLMAP lost in {seconds}s (peak {peak} GB); frames COLMAP knows, "
        f"given unposed, land {record['check_error_m']['median']} m from it (median, max "
        f"{record['check_error_m']['max']} m), turned {record['check_rotation_deg']['median']} deg")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("space", type=Path)
    parser.add_argument("--stride", type=int, default=1,
                        help="give the model every Nth frame (less memory); the others get "
                             "interpolated priors that the adjustment then corrects")
    parser.add_argument("--guide", type=Path,
                        help="a COLMAP model (e.g. workspace/sparse/0): keep its cameras and lens, "
                             "and ask MapAnything only for the frames it did not place")
    parser.add_argument("--views", type=int, default=GUIDED_VIEWS, help="frames to give the model when guided")
    args = parser.parse_args()
    if args.guide:
        solve_guided(args.space.resolve(), args.guide.resolve(), args.views)
    else:
        solve(args.space.resolve(), max(1, args.stride))


if __name__ == "__main__":
    main()
