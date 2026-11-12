#!/usr/bin/env python3
"""Neural dense reconstruction: monocular depth fused over solved cameras.

COLMAP's sparse stage gives camera poses but few points. This stage runs a
monocular depth network on selected keyframes, scales each frame's depth to
the sparse points visible in it, back-projects every pixel into world space,
and merges the result into one dense colored cloud.

Depth models (--depth-model):
    moge  MoGe-2 (default, MIT): metric depth from one image. Each frame needs
          only a single scale to COLMAP units, so a frame that sees too few
          sparse points still fuses, using the median scale of the others.
    da2   Depth Anything V2 Small: relative inverse depth with a per-frame
          scale + shift fit (the previous default; frames without enough
          sparse points are skipped).

Usage:
    python3 pipeline/densify.py spaces/<name> [--keyframes 12] [--stride 4]
    python3 pipeline/densify.py spaces/<name> --depth-model da2

Writes <space>/cloud-dense.ply and <space>/densify.json (the COLMAP model
used and, for MoGe, how many COLMAP units make one metre).

Needs: torch, plus MoGe (pip install --no-deps from the pinned pre-v3 commit
925b8ed, with utils3d) or transformers for da2. Weights download on first run.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from pointcloud import PointCloud, remove_outliers, save_ply, trim_far_points
from semantics import (SEGMENTER_ID, ClaudeDetector, Detector, Segmenter, planned_vocabulary,
                       default_device, pixel_labels)
from tracking import PROMPT_FRAMES, TRACKERS, merge_detections, prompt_frames, track
from mirrors import fill_unreliable, mask_at
import lens

MOGE_CHECKPOINT = "Ruicheng/moge-2-vitl-normal"
# MoGe-3 (Jul 2026): much finer point-map detail; needs MoGe's current code and
# FlexGEMM (Triton), so a CUDA machine (Linux), not a Mac.
MOGE3_CHECKPOINT = "Ruicheng/moge-3-vitl"
# Depth Anything 3, Base (Apache-2.0): all keyframes in one pass with the solve's
# cameras, so their depths agree by construction. Its depth is in the solve's
# units; metres come from an earlier MoGe reading of the same space.
DA3_CHECKPOINT = "depth-anything/DA3-BASE"
DA3_RESOLUTION = 504
DA3_GROUP = 8                 # keyframes the network sees together
MIN_ANCHORS = 12
# Scale. One monocular depth map is right up to a scale, and every keyframe's
# scale is fitted to the solve's points it sees. Along a walk the fits drift a
# little (the solve's own scale drifts), so a keyframe takes the scale of its
# stretch of the walk: its own fit blended with its neighbours', weighted by how
# many points each fit rests on. A keyframe whose own fit is far from its
# neighbours' has depth that does not match the room (a close-up, a blur, a
# reflection taken for space) and is left out rather than forced onto a scale.
SCALE_WINDOW = 2              # keyframes either side that a scale is smoothed with
SCALE_OFF = float(np.log(1.25))  # own fit this far from the neighbours': bad depth
# Free space. A point another keyframe looks straight through (it lies well in
# front of the surface that keyframe sees at the same spot) is a floater: depth
# bleeding at an edge, a reflection, a moving thing. Checked against the nearest
# keyframes along the walk.
# Labels. Each keyframe names its own points from its own outlines, so one wrong
# outline mislabels a patch. Where several keyframes see the same small cube of
# space, the name most of them give it wins.
VOTE_VOXEL_M = 0.04            # side of the cube that is voted on, metres
VOTE_SHARE = 0.6               # an unlabelled point takes a name this share of keyframes give
SEE_THROUGH = 0.85            # nearer than this share of a neighbour's surface depth
CHECK_NEIGHBOURS = 2          # keyframes either side checked


def quat_to_rot(qw, qx, qy, qz):
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
        [2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
        [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)],
    ])


def read_cameras_bin(path):
    cameras = {}
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            cam_id, model, width, height = struct.unpack("<iiQQ", f.read(24))
            num_params = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8, 6: 12}.get(model, 4)
            params = struct.unpack(f"<{num_params}d", f.read(8 * num_params))
            cameras[cam_id] = {"model": model, "width": width, "height": height,
                               "params": params}
    return cameras


def read_images_bin(path):
    """Full parse: pose, camera id, name, and 2D->3D observations."""
    images = {}
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            image_id = struct.unpack("<I", f.read(4))[0]
            qw, qx, qy, qz = struct.unpack("<dddd", f.read(32))
            t = np.array(struct.unpack("<ddd", f.read(24)))
            cam_id = struct.unpack("<I", f.read(4))[0]
            name = b""
            while (ch := f.read(1)) != b"\x00":
                name += ch
            npts = struct.unpack("<Q", f.read(8))[0]
            raw = np.frombuffer(f.read(24 * npts), dtype=np.uint8)
            xys = raw.view("<f8").reshape(npts, 3)[:, :2] if npts else np.zeros((0, 2))
            p3d = raw.view("<i8").reshape(npts, 3)[:, 2] if npts else np.zeros(0, np.int64)
            images[image_id] = {
                "R": quat_to_rot(qw, qx, qy, qz), "t": t, "camera_id": cam_id,
                "name": name.decode(), "xys": xys, "point3D_ids": p3d,
            }
    return images


def read_points3d_bin(path):
    points = {}
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            pid = struct.unpack("<Q", f.read(8))[0]
            xyz = np.array(struct.unpack("<ddd", f.read(24)))
            f.read(3 + 8)  # rgb + error
            track_len = struct.unpack("<Q", f.read(8))[0]
            f.read(8 * track_len)
            points[pid] = xyz
    return points


MIN_KEYFRAMES = 12
MAX_KEYFRAMES = 30
FRAMES_PER_KEYFRAME = 7


def pick_keyframes(images, count):
    """Evenly spread keyframes, preferring frames with many 3D observations."""
    ordered = sorted(images.items(), key=lambda kv: kv[1]["name"])
    if len(ordered) <= count:
        return [k for k, _ in ordered]
    bins = np.array_split(np.arange(len(ordered)), count)
    chosen = []
    for b in bins:
        best = max(b, key=lambda i: (ordered[i][1]["point3D_ids"] >= 0).sum())
        chosen.append(ordered[best][0])
    return chosen


def frame_anchors(info, points3d, W, H):
    """Pixel positions and COLMAP depths of the sparse points a frame sees."""
    valid = info["point3D_ids"] >= 0
    xy = info["xys"][valid]
    ids = info["point3D_ids"][valid]
    known = np.array([p in points3d for p in ids], dtype=bool)
    if not known.any():
        return np.zeros(0, int), np.zeros(0, int), np.zeros(0)
    world = np.array([points3d[p] for p in ids[known]])
    cam_space = (info["R"] @ world.T).T + info["t"]
    in_front = cam_space[:, 2] > 0.05
    xy, depth = xy[known][in_front], cam_space[in_front, 2]
    px = np.clip(xy[:, 0].astype(int), 0, W - 1)
    py = np.clip(xy[:, 1].astype(int), 0, H - 1)
    return px, py, depth


def fit_scale_shift(pred_inv, sparse_depth):
    """Robust fit: sparse_inverse_depth ~ a * predicted_inverse + b."""
    target = 1.0 / np.maximum(sparse_depth, 1e-6)
    a, b = 1.0, 0.0
    mask = np.ones(len(target), bool)
    for _ in range(4):
        A = np.column_stack([pred_inv[mask], np.ones(mask.sum())])
        (a, b), *_ = np.linalg.lstsq(A, target[mask], rcond=None)
        residual = np.abs(a * pred_inv + b - target)
        cutoff = 2.5 * np.median(residual) + 1e-9
        mask = residual < cutoff
        if mask.sum() < 8:
            break
    return a, b


def fit_scale(pred_depth, sparse_depth, with_support=False):
    """Robust single scale for metric predictions: sparse ~ s * predicted.
    Returns None when too few anchors land on valid predicted depth; with
    with_support, (s, number of anchors that agree with it)."""
    valid = np.isfinite(pred_depth) & (pred_depth > 0)
    if valid.sum() < 8:
        return (None, 0) if with_support else None
    ratio = sparse_depth[valid] / pred_depth[valid]
    s = float(np.median(ratio))
    keep = np.ones(len(ratio), bool)
    for _ in range(3):
        keep = np.abs(np.log(ratio / s)) < 0.25
        if keep.sum() < 8:
            break
        s = float(np.median(ratio[keep]))
    return (s, int(keep.sum())) if with_support else s


def weighted_median(values, weights) -> float:
    order = np.argsort(values)
    v, w = np.asarray(values, float)[order], np.asarray(weights, float)[order]
    return float(v[np.searchsorted(np.cumsum(w), 0.5 * w.sum())])


def settle_scales(fits, window=SCALE_WINDOW, off=SCALE_OFF):
    """Each keyframe's scale from its own fit and its stretch of the walk.

    fits: (scale or None, supporting anchors) per keyframe, in walk order.
    Returns (scales, metre, left_out): a scale per keyframe (None for one left
    out), the support-weighted median scale (COLMAP units per metre), and the
    indices of keyframes whose own fit disagrees with their neighbours'."""
    logs = np.array([np.log(s) if s else np.nan for s, _ in fits])
    support = np.array([n if s else 0 for s, n in fits], float)
    fitted = np.isfinite(logs)
    if not fitted.any():
        return [None] * len(fits), None, []
    metre_log = weighted_median(logs[fitted], support[fitted])
    scales, left_out = [], []
    for i in range(len(fits)):
        near = [j for j in range(max(0, i - window), min(len(fits), i + window + 1)) if j != i and fitted[j]]
        # Neighbours that themselves sit far off are not trusted as a reference.
        near = [j for j in near if abs(logs[j] - metre_log) < 2 * off] or near
        reference = weighted_median(logs[near], support[near]) if near else metre_log
        if not fitted[i]:
            scales.append(float(np.exp(reference)))
            continue
        if abs(logs[i] - reference) > off:
            scales.append(None)
            left_out.append(i)
            continue
        weight_near = 0.5 * support[near].sum() / max(1, len(near)) if near else 0.0
        blended = (support[i] * logs[i] + weight_near * reference) / (support[i] + weight_near)
        scales.append(float(np.exp(blended)))
    return scales, float(np.exp(metre_log)), left_out


def vote_labels(points, labels, frame_of, voxel):
    """Labels decided by the keyframes that see each small cube of space.

    In a cube that several keyframes see, every labelled point takes the name
    most of those keyframes give there (each keyframe one vote), and an
    unlabelled point takes it too when at least VOTE_SHARE of them give it.
    Returns the new labels and how many points changed."""
    labels = labels.copy()
    cells = np.floor(points / voxel).astype(np.int64)
    _, cell = np.unique(cells, axis=0, return_inverse=True)
    cell = cell.ravel()
    # One vote per (cube, keyframe, label).
    votes = np.unique(np.column_stack([cell, frame_of, labels]), axis=0)
    frames_per_cell = np.bincount(np.unique(votes[:, :2], axis=0)[:, 0], minlength=cell.max() + 1)
    named = votes[votes[:, 2] > 0]
    if not len(named):
        return labels, 0
    pair, count = np.unique(named[:, [0, 2]], axis=0, return_counts=True)
    order = np.lexsort((-count, pair[:, 0]))
    pair, count = pair[order], count[order]
    first = np.r_[True, pair[1:, 0] != pair[:-1, 0]]
    winner = np.zeros(cell.max() + 1, np.int64)
    support = np.zeros(cell.max() + 1, np.int64)
    winner[pair[first, 0]] = pair[first, 1]
    support[pair[first, 0]] = count[first]
    contested = frames_per_cell[cell] >= 2
    new = labels.copy()
    labelled = (labels > 0) & contested & (winner[cell] > 0)
    new[labelled] = winner[cell][labelled]
    unlabelled = (labels == 0) & contested & (support[cell] >= VOTE_SHARE * frames_per_cell[cell])
    new[unlabelled] = winner[cell][unlabelled]
    return new.astype(labels.dtype), int((new != labels).sum())


def seen_through(world, others):
    """Which world points a neighbouring keyframe looks straight through.

    others: (info, cam, depth map at working size, k, scale) per neighbour; the
    depth map is in the depth network's units, so scale * depth is the solve's."""
    hit = np.zeros(len(world), bool)
    for info, cam, depth, k, scale in others:
        cam_pts = (info["R"] @ world.T).T + info["t"]
        z = cam_pts[:, 2]
        front = z > 0.05
        if not front.any():
            continue
        u, v = lens.project(lens.from_colmap(cam), cam_pts[front, 0] / z[front], cam_pts[front, 1] / z[front])
        rows, cols = depth.shape
        r, c = (v * k).astype(int), (u * k).astype(int)
        inside = np.isfinite(u) & (r >= 0) & (r < rows) & (c >= 0) & (c < cols)
        idx = np.flatnonzero(front)[inside]
        surface = scale * depth[r[inside], c[inside]]
        ok = np.isfinite(surface) & (surface > 0)
        hit[idx[ok]] |= z[idx[ok]] < SEE_THROUGH * surface[ok]
    return hit


def depth_edges(depth, threshold=0.05):
    """Pixels on depth discontinuities, where monocular depth smears the
    foreground into the background ("flying pixels")."""
    from scipy.ndimage import maximum_filter

    log_d = np.log(np.maximum(depth, 1e-6))
    grad = np.maximum(
        np.abs(np.diff(log_d, axis=1, append=log_d[:, -1:])),
        np.abs(np.diff(log_d, axis=0, append=log_d[-1:, :])),
    )
    return maximum_filter(grad > threshold, size=3)


class MoGeDepth:
    """MoGe-2 metric depth, predicted at a reduced working resolution."""

    def __init__(self, device, work_size):
        import torch
        from moge.model.v2 import MoGeModel

        self.torch = torch
        self.model = MoGeModel.from_pretrained(MOGE_CHECKPOINT).to(device).eval()
        self.device = device
        self.work_size = work_size
        self.fp16 = device != "cpu"

    def __call__(self, img, fx):
        """Depth in metres (0 where invalid), the working-size scale, and
        camera-space normals (None if the checkpoint predicts none)."""
        from PIL import Image

        W, H = img.size
        scale = min(1.0, self.work_size / max(W, H))
        small = img.convert("RGB")
        if scale < 1.0:
            small = small.resize((round(W * scale), round(H * scale)), Image.BILINEAR)
        # Field of view does not change with resizing; pass COLMAP's so the
        # predicted depth matches the solved camera.
        fov_x = float(np.degrees(2 * np.arctan(W / (2 * fx))))
        tensor = self.torch.from_numpy(
            np.asarray(small, dtype=np.float32) / 255.0
        ).permute(2, 0, 1).to(self.device)
        with self.torch.inference_mode():
            try:
                out = self.model.infer(tensor, fov_x=fov_x, use_fp16=self.fp16)
            except RuntimeError:
                if not self.fp16:
                    raise
                self.fp16 = False  # some MPS builds lack fp16 kernels
                out = self.model.infer(tensor, fov_x=fov_x, use_fp16=False)
        depth = out["depth"].float().cpu().numpy()
        depth[~np.isfinite(depth)] = 0.0
        depth[depth_edges(depth)] = 0.0
        # Same OpenCV camera axes as the depth; float16 keeps the maps small.
        normal = out["normal"].float().cpu().numpy().astype(np.float16) \
            if "normal" in out else None
        return depth, scale, normal


class MoGe3Depth(MoGeDepth):
    """MoGe-3: the same interface as MoGe-2, sharper."""

    def __init__(self, device, work_size):
        import torch
        from moge.model.v3 import MoGeModel

        self.torch = torch
        self.model = MoGeModel.from_pretrained(MOGE3_CHECKPOINT).to(device).eval()
        self.device = device
        self.work_size = work_size
        self.fp16 = device != "cpu"


class DA3Depth:
    """Depth Anything 3 over all keyframes at once, given the solve's cameras.

    prepare() runs the network on the keyframes with their solved poses and
    intrinsics; calling it then answers one keyframe at a time like MoGeDepth,
    with depth in the solve's own units (so each keyframe's scale fit is about 1)."""

    def __init__(self, device):
        from depth_anything_3.api import DepthAnything3

        self.model = DepthAnything3.from_pretrained(DA3_CHECKPOINT).to(device=device).eval()
        self.depth: dict[str, np.ndarray] = {}

    def prepare(self, paths, infos, cams):
        extrinsics, intrinsics = [], []
        for info, cam in zip(infos, cams):
            w2c = np.eye(4)
            w2c[:3, :3], w2c[:3, 3] = info["R"], info["t"]
            extrinsics.append(w2c)
            fx, fy, cx, cy = cam["params"][:4]
            intrinsics.append(np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], float))
        # All keyframes at once attend across every frame (13 GB at 24 frames on a
        # T4): they go in groups, each with its solved cameras, so every group's
        # depth is in the solve's units and the groups agree.
        depths = []
        for start in range(0, len(paths), DA3_GROUP):
            part = slice(start, start + DA3_GROUP)
            prediction = self.model.inference([str(p) for p in paths[part]],
                                              extrinsics=np.array(extrinsics[part], np.float32),
                                              intrinsics=np.array(intrinsics[part], np.float32),
                                              align_to_input_ext_scale=True, process_res=DA3_RESOLUTION)
            depths += list(prediction.depth)
        for path, depth in zip(paths, depths):
            depth = np.asarray(depth, np.float32)
            depth[~np.isfinite(depth)] = 0.0
            depth[depth_edges(depth)] = 0.0
            self.depth[Path(path).name] = depth

    def __call__(self, img, fx):
        depth = self.depth[Path(img.filename).name]
        return depth, depth.shape[1] / img.size[0], None


def release_model_memory(torch) -> None:
    """Give a freed model's memory back before the next model loads."""
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()


def back_project(info, cam, img, z_at, stride, max_depth, n_at=None):
    """World-space points, colours, (with n_at) normals, and the frame pixels
    they came from, for a pixel grid of one frame; and the share of that grid
    the lens has no ray for."""
    W, H = img.size
    us, vs = np.meshgrid(np.arange(0, W, stride), np.arange(0, H, stride))
    us, vs = us.ravel(), vs.ravel()
    # The frames are the raw ones, still bent by the lens stage 1 solved, so
    # each pixel's ray (through its centre) has that distortion taken out.
    x, y = lens.unproject(lens.from_colmap(cam), us + 0.5, vs + 0.5)
    # A solve whose polynomial folds inside the frame has no ray past the
    # fold (lens.py): those pixels are dropped, and counted so it shows.
    has_ray = np.isfinite(x)
    no_ray = float(1 - has_ray.mean()) if len(us) else 0.0
    z = z_at(us, vs)
    ok = has_ray & np.isfinite(z) & (z > 0.05) & (z < max_depth)
    us, vs, z, x, y = us[ok], vs[ok], z[ok], x[ok], y[ok]
    rays = np.column_stack([x, y, np.ones(len(us))])
    cam_pts = rays * z[:, None]
    world_pts = (info["R"].T @ (cam_pts - info["t"]).T).T
    rgb = np.asarray(img.convert("RGB"))[vs, us]
    normals = None
    if n_at is not None:
        cam_normals = n_at(us, vs).astype(np.float64)
        normals = (info["R"].T @ cam_normals.T).T.astype(np.float32)
    return world_pts.astype(np.float32), rgb.astype(np.uint8), normals, (us, vs), no_ray


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("space", help="space folder (e.g. spaces/first-test)")
    parser.add_argument("--model-dir", default=None,
                        help="COLMAP model dir (default: workspace/sparse/<best>)")
    parser.add_argument("--keyframes", type=int, default=None,
                        help="frames to fuse (default: one per ~7 registered frames, "
                             f"{MIN_KEYFRAMES} to {MAX_KEYFRAMES})")
    parser.add_argument("--stride", type=int, default=None,
                        help="back-project every Nth pixel of the full frame (default 4 "
                             "for 12 keyframes, wider with more so the cloud keeps its size)")
    parser.add_argument("--depth-model", choices=["moge", "moge3", "da3", "da2"], default="moge",
                        help="monocular depth network (default moge)")
    parser.add_argument("--work-size", type=int, default=1280,
                        help="longest side, in pixels, that MoGe predicts at (default 1280)")
    parser.add_argument("--detector", choices=["auto", "claude", "dino"], default="auto",
                        help="who marks the room's objects on the keyframes: claude (the room's own understanding; "
                             "SAM cuts the outlines), dino (GroundingDINO, matching the object list's words), auto "
                             "(Claude when reachable, GroundingDINO otherwise)")
    parser.add_argument("--marks-only", action="store_true",
                        help="only have Claude mark the objects on the frames a GPU run with tracking will use, "
                             "kept in workspace/claude-marks/marks.json for that run (which then needs no Claude)")
    parser.add_argument("--no-semantics", action="store_true",
                        help="skip object detection: no labels on the cloud, and no "
                             "filtering of mirrors, windows and screens")
    parser.add_argument("--no-outlines", action="store_true",
                        help="label whole detection rectangles instead of cutting "
                             "each one to the object's outline with SAM 2.1")
    parser.add_argument("--track", choices=["auto", "on", "off"], default="auto",
                        help="carry each detected object through every frame with SAM 2.1's "
                             "video model (tracking.py), so it keeps one name and has an outline "
                             "in every frame; auto = on CUDA only (it is slow on a Mac's GPU)")
    parser.add_argument("--tracker", choices=TRACKERS, default="auto",
                        help="what tracks: sam3 finds and tracks the room's names in one model "
                             "(facebook/sam3, gated: needs access and a Hugging Face login); sam2 "
                             "tracks what GroundingDINO found; auto = SAM 3 when its weights load")
    parser.add_argument("--prompt-frames", type=int, default=None,
                        help="about how many frames the detector runs on when tracking "
                             f"(default {PROMPT_FRAMES})")
    parser.add_argument("--no-mirror-planes", action="store_true",
                        help="drop the points on mirrors, windows and screens instead of giving "
                             "them the depth of the plane their surroundings lie in (mirrors.py)")
    parser.add_argument("--output", default=None,
                        help="output PLY (default <space>/cloud-dense.ply)")
    args = parser.parse_args()

    space = Path(args.space)
    workspace = space / "workspace"
    if args.model_dir:
        model_dir = Path(args.model_dir)
    else:
        # COLMAP 4's global mapper also leaves a project.ini here.
        candidates = sorted(d for d in (workspace / "sparse").iterdir()
                            if (d / "points3D.bin").exists())
        model_dir = max(
            candidates,
            key=lambda d: len(read_points3d_bin(d / "points3D.bin")),
        )
    print(f"Using model {model_dir}")

    cameras = read_cameras_bin(model_dir / "cameras.bin")
    images = read_images_bin(model_dir / "images.bin")
    points3d = read_points3d_bin(model_dir / "points3D.bin")
    # A short pan needs few keyframes; a walk through several rooms needs more
    # viewpoints. More keyframes sample every Nth pixel more sparsely, so the
    # fused cloud (and memory on an 8 GB Mac) stays about the same size.
    keyframes = args.keyframes or int(np.clip(round(len(images) / FRAMES_PER_KEYFRAME),
                                              MIN_KEYFRAMES, MAX_KEYFRAMES))
    args.stride = args.stride or max(4, round(4 * np.sqrt(keyframes / MIN_KEYFRAMES)))
    keys = pick_keyframes(images, keyframes)
    print(f"{len(images)} registered frames; densifying {len(keys)} keyframes, "
          f"every {args.stride}th pixel")
    if args.marks_only:
        # The frames the detector is asked about with tracking on (a GPU run).
        key_names = sorted(images[k]["name"] for k in keys)
        all_names = sorted(p.name for p in (workspace / "images").glob("*.jpg"))
        prompt_names = prompt_frames(all_names, key_names, args.prompt_frames or PROMPT_FRAMES)
        detector = ClaudeDetector(planned_vocabulary(space), work_dir=workspace / "claude-marks")
        print(f"Claude marks {len(prompt_names)} frames for a run with tracking")
        detector.detect_many([workspace / "images" / n for n in prompt_names])
        print(f"Marks kept in {workspace / 'claude-marks' / 'marks.json'}")
        return

    import torch
    from PIL import Image

    device = default_device()
    metre_from = None
    if args.depth_model == "moge":
        predictor = MoGeDepth(device, args.work_size)
        print(f"MoGe-2 ({MOGE_CHECKPOINT}) on {device}")
    elif args.depth_model == "moge3":
        predictor = MoGe3Depth(device, args.work_size)
        print(f"MoGe-3 ({MOGE3_CHECKPOINT}) on {device}")
    elif args.depth_model == "da3":
        earlier = space / "densify.json"
        metre_from = json.loads(earlier.read_text()).get("colmap_units_per_metre") if earlier.exists() else None
        if not metre_from:
            sys.exit("Depth Anything 3 gives depth in the solve's units: run densify with MoGe first, "
                     "so the space knows its metres (densify.json).")
        predictor = DA3Depth(device)
        ordered = sorted(keys, key=lambda k: images[k]["name"])
        predictor.prepare([workspace / "images" / images[k]["name"] for k in ordered],
                          [images[k] for k in ordered], [cameras[images[k]["camera_id"]] for k in ordered])
        print(f"Depth Anything 3 ({DA3_CHECKPOINT}) on {device}, {len(ordered)} keyframes with their solved cameras; "
              f"metres from the earlier MoGe reading (1 m = {metre_from:.4f} units)")
    else:
        from transformers import pipeline as hf_pipeline
        predictor = hf_pipeline(
            "depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf",
            device=device,
        )
        print(f"Depth Anything V2 (small) on {device}")

    use_detector = args.depth_model != "da2" and not args.no_semantics
    label_index, label_names, unreliable_ids, yielding = None, None, np.zeros(0, int), set()

    all_pts, all_cols, all_norms, all_labels = [], [], [], []
    dropped_unreliable, filled_on_planes, fills = 0, 0, {}
    without_ray = []                     # per keyframe, the share of its pixels the lens has no ray for
    meta = {"model_dir": str(model_dir.resolve()), "depth_model": args.depth_model,
            "keyframes": len(keys)}

    if args.depth_model == "da2":
        for i, image_id in enumerate(keys):
            info = images[image_id]
            cam = cameras[info["camera_id"]]
            img = Image.open(workspace / "images" / info["name"])
            W, H = img.size
            px, py, depth_true = frame_anchors(info, points3d, W, H)
            if len(px) < MIN_ANCHORS:
                print(f"  {info['name']}: too few anchors, skipped")
                continue
            pred = predictor(img)["predicted_depth"].squeeze().float().cpu().numpy()
            pred = np.array(Image.fromarray(pred).resize((W, H), Image.BILINEAR))
            # Depth Anything outputs relative INVERSE depth (bigger = closer)
            pred_inv = np.maximum(pred, 1e-6)
            a, b = fit_scale_shift(pred_inv[py, px], depth_true)
            pts, cols, _, _, no_ray = back_project(
                info, cam, img,
                lambda us, vs: 1.0 / np.maximum(a * pred_inv[vs, us] + b, 1e-6),
                args.stride, np.percentile(depth_true, 98) * 3,
            )
            without_ray.append(no_ray)
            all_pts.append(pts)
            all_cols.append(cols)
            print(f"  [{i+1}/{len(keys)}] {info['name']}: "
                  f"{len(pts):,} points (scale fit on {len(px)} anchors)")
    else:
        # Pass 1: metric depth per keyframe, and its scale to COLMAP units.
        frames = []
        for i, image_id in enumerate(keys):
            info = images[image_id]
            cam = cameras[info["camera_id"]]
            img = Image.open(workspace / "images" / info["name"])
            W, H = img.size
            px, py, depth_true = frame_anchors(info, points3d, W, H)
            depth, k, normal = predictor(img, cam["params"][0])
            s = None
            if len(px) >= MIN_ANCHORS:
                sy = np.minimum((py * k).astype(int), depth.shape[0] - 1)
                sx = np.minimum((px * k).astype(int), depth.shape[1] - 1)
                s, support = fit_scale(depth[sy, sx], depth_true, with_support=True)
            else:
                support = 0
            frames.append({
                "id": image_id, "depth": depth, "k": k, "s": s, "support": support, "normal": normal,
                "detections": [],
                "cap": np.percentile(depth_true, 98) * 3 if len(px) >= MIN_ANCHORS else None,
            })
            note = f"scale {s:.3f}" if s is not None else "no scale fit"
            print(f"  [{i+1}/{len(keys)}] {info['name']}: {note} ({len(px)} anchors)")

        # One model in memory at a time (8 GB Macs): free MoGe before the
        # detector loads, and the detector before back-projection.
        del predictor
        release_model_memory(torch)
        tracking_on = args.track == "on" or (args.track == "auto" and device == "cuda")
        if use_detector:
            try:
                vocabulary = planned_vocabulary(space)
                detector = None
                if args.detector in ("auto", "claude"):
                    try:
                        detector = ClaudeDetector(vocabulary, work_dir=workspace / "claude-marks")
                    except Exception as exc:
                        if args.detector == "claude":
                            raise
                        print(f"Claude cannot mark the objects ({exc}); GroundingDINO instead")
                if detector is None:
                    detector = Detector(device=device, vocabulary=vocabulary)
                label_names = ["unlabelled"] + vocabulary.names
                label_index = {name: i for i, name in enumerate(label_names)}
                unreliable_ids = np.array([label_index[name] for name in vocabulary.unreliable],
                                          dtype=int)
                # Names of the room itself: they give way to furniture outlines (pixel_labels).
                yielding = set(vocabulary.with_role("fixture", "hanging"))
                meta["objects"] = vocabulary.to_json()
                print(f"Object detection: {'Claude' if detector.device == 'claude' else 'GroundingDINO-tiny'} "
                      f"on {detector.device}, "
                      f"{len(vocabulary.names)} names ({vocabulary.source} list): "
                      + ", ".join(vocabulary.names))
                # With tracking, the detector also names things on frames in
                # between the keyframes, so the tracker has more to go on.
                key_names = sorted(images[f["id"]]["name"] for f in frames)
                all_names = sorted(p.name for p in (workspace / "images").glob("*.jpg"))
                prompt_names = (prompt_frames(all_names, key_names, args.prompt_frames or PROMPT_FRAMES)
                                if tracking_on else key_names)
                prompts = {}
                if isinstance(detector, ClaudeDetector):
                    prompts = detector.detect_many([workspace / "images" / n for n in prompt_names])
                    meta["detector"] = {"by": "claude", "usage": detector.advisor.usage_total()}
                else:
                    for name in prompt_names:
                        prompts[name] = detector.detect(Image.open(workspace / "images" / name))
                        seen = ", ".join(sorted({d["label"] for d in prompts[name]}))
                        print(f"  {name}: {seen or 'nothing detected'}")
                    meta["detector"] = {"by": "groundingdino"}
                for f in frames:
                    f["detections"] = prompts.get(images[f["id"]]["name"], [])
                # Kept so stage 3 can show Claude the frame where each object
                # was actually seen (agent.py object_frames).
                meta["detections"] = {
                    name: [{"label": d["label"], "score": d["score"],
                            "box": [round(v) for v in d["box"]]} for d in dets]
                    for name, dets in prompts.items() if dets}
                del detector
                release_model_memory(torch)
            except Exception as exc:  # transformers or weights missing
                label_index = None
                print(f"Object detection unavailable ({exc}); continuing without labels")
        tracks = None
        if label_index is not None and tracking_on and any(prompts.values()):
            try:
                # The room's names (pieces, surfaces, openings) are tracked;
                # the rest keep their per-keyframe detections, outlined below.
                tracks, rest = track(workspace / "images", all_names, prompts, vocabulary,
                                     args.tracker, device, log=print)
                release_model_memory(torch)
                tracks.save(workspace / "tracks")
                # The keyframes are labelled by the tracked outlines, named by
                # the detector's votes over the whole video; and every frame
                # an object shows in is recorded for stage 3.
                for f in frames:
                    name = images[f["id"]]["name"]
                    f["detections"] = tracks.detections(name, work_size=1024) + list(rest.get(name, []))
                meta["detections"] = merge_detections(tracks.all_detections(), rest)
                meta["tracks"] = tracks.summary()
                meta["outlines"] = tracks.model + " (video)"
                counts = tracks.summary()["labels"]
                print(f"  {len(tracks.tracks)} objects: "
                      + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
            except Exception as exc:  # weights missing, or the model failed
                tracks = None
                print(f"Object tracking unavailable ({exc}); outlining the keyframes' detections instead")
        elif label_index is not None and not tracking_on and args.track == "auto":
            print(f"Object tracking: off on {device} (about a second an object and frame here; "
                  "--track on forces it)")
        if label_index is not None:
            # A rectangle around a bed also holds floor and curtain; cut each
            # one down to the object's own outline before labelling points
            # (with tracks, only the untracked names still need one).
            if tracks is None:
                meta["outlines"] = "boxes"
            todo = {f["id"]: [d for d in f["detections"] if "mask" not in d] for f in frames}
            if not args.no_outlines and any(todo.values()):
                try:
                    segmenter = Segmenter(device=device)
                    print(f"Object outlines: SAM 2.1 hiera-tiny on {segmenter.device}")
                    for f in frames:
                        if not todo[f["id"]]:
                            continue
                        name = images[f["id"]]["name"]
                        done = segmenter.outline(Image.open(workspace / "images" / name), todo[f["id"]])
                        print(f"  {name}: {done}/{len(todo[f['id']])} outlined")
                    del segmenter
                    release_model_memory(torch)
                    meta["outlines"] = (SEGMENTER_ID if tracks is None
                                        else f"{tracks.model} (video), {SEGMENTER_ID} (image)")
                except Exception as exc:  # weights missing, or the model failed
                    for dets in todo.values():
                        for det in dets:
                            det.pop("mask", None)
                    print(f"Object outlines unavailable ({exc}); "
                          "labelling whole detection rectangles")

        fitted = np.array([f["s"] for f in frames if f["s"] is not None])
        if len(fitted) == 0:
            sys.exit("No keyframe sees enough sparse points to fix the scale; "
                     "check the COLMAP model (reconstruct.py) and retry.")
        # Keyframes in walk order, each scaled with its stretch of the walk.
        frames.sort(key=lambda f: images[f["id"]]["name"])
        settled, s_med, left_out = settle_scales([(f["s"], f["support"]) for f in frames])
        depth_unit = s_med
        if metre_from:
            # The network's unit is the solve's; a metre is what MoGe measured before.
            s_med = metre_from
        for f, s in zip(frames, settled):
            f["scale"] = s
        spread = float(np.subtract(*np.percentile(fitted, [75, 25])) / depth_unit)
        caps = [f["cap"] for f in frames if f["cap"] is not None]
        default_cap = float(np.median(caps)) if caps else np.inf
        print(f"Scale: 1 m = {s_med:.4f} COLMAP units "
              f"(interquartile spread {spread:.0%} over {len(fitted)} frames)")
        if spread > 0.3:
            print("Warning: keyframes disagree on scale by more than 30%, so the "
                  "camera model is probably inconsistent (pieces joined at "
                  "different scales). Rerun reconstruct.py with --mapper incremental.")

        if left_out:
            print("Left out (depth disagrees with its neighbours' scale): " + ", ".join(
                f"{images[frames[i]['id']]['name']} ({frames[i]['s']:.2f} against about "
                f"{np.exp(np.median([np.log(f['scale']) for f in frames if f['scale']])):.2f})" for i in left_out))
        on_median = sum(1 for f in frames if f["s"] is None)

        # Pass 2a: mirrors, windows and screens stood on their planes first, so
        # the free-space check below compares against the corrected surfaces.
        for f in frames:
            f["filled_map"], f["fill_records"] = None, []
            if f["scale"] is None or label_index is None or args.no_mirror_planes or not f["detections"]:
                continue
            camera = lens.from_colmap(cameras[images[f["id"]]["camera_id"]])
            f["depth"], f["normal"], f["filled_map"], f["fill_records"] = fill_unreliable(
                f["depth"], f["k"], f["detections"], set(vocabulary.unreliable),
                (camera["fx"], camera["fy"], camera["cx"], camera["cy"]), f["normal"], camera["distortion"])
        # What a keyframe is checked against: its depth, without the mirrors,
        # windows and screens that could not be stood on a plane (their depth
        # is the reflection or the view outside, which every wall is in front of).
        for f in frames:
            f["check_depth"] = f["depth"]
            if label_index is None or f["scale"] is None:
                continue
            doubtful = [d for d in f["detections"] if d["label"] in vocabulary.unreliable]
            if doubtful:
                f["check_depth"] = f["depth"].copy()
                for d in doubtful:
                    m = mask_at(d, f["depth"].shape, f["k"])
                    if f["filled_map"] is not None:
                        m &= ~f["filled_map"]
                    f["check_depth"][m] = np.nan
        kept = [f for f in frames if f["scale"] is not None]
        seen_through_total = 0

        # Pass 2b: back-project.
        for at_frame, f in enumerate(kept):
            info = images[f["id"]]
            cam = cameras[info["camera_id"]]
            img = Image.open(workspace / "images" / info["name"])
            s = f["scale"]
            depth, k, normal = f["depth"], f["k"], f["normal"]
            # A mirror's depth is its reflection's, a window's the view
            # outside: give each the plane of what surrounds it (the wall,
            # the wardrobe it is a door of) instead of dropping it.
            filled_map, records = f["filled_map"], f["fill_records"]
            if records:
                for r in records:
                    fills.setdefault(r["label"], []).append(r)
                done = ", ".join(f"{r['label']} {r['filled']:,} px at {r['plane_depth_m']} m"
                                 for r in records if r["filled"])
                left = ", ".join(f"{r['label']} ({r['why']})" for r in records if not r["filled"])
                if done or left:
                    print(f"  {info['name']}: filled {done or 'nothing'}"
                          + (f"; left {left}" if left else ""))
            rows, cols = depth.shape

            def pixel(us, vs, k=k, rows=rows, cols=cols):
                """Full-frame pixel -> index into the working-size maps."""
                return (np.minimum((vs * k).astype(int), rows - 1),
                        np.minimum((us * k).astype(int), cols - 1))

            pts, rgb, nrm, (us, vs), no_ray = back_project(
                info, cam, img,
                lambda us, vs: s * depth[pixel(us, vs)],
                args.stride, f["cap"] if f["cap"] is not None else default_cap,
                n_at=(lambda us, vs: normal[pixel(us, vs)]) if normal is not None else None,
            )
            without_ray.append(no_ray)
            # Points the neighbouring keyframes look straight through are floaters.
            others = [(images[g["id"]], cameras[images[g["id"]]["camera_id"]], g["check_depth"], g["k"], g["scale"])
                      for g in kept[max(0, at_frame - CHECK_NEIGHBOURS):at_frame + CHECK_NEIGHBOURS + 1] if g is not f]
            through = seen_through(pts.astype(np.float64), others) if others else np.zeros(len(pts), bool)
            if through.any():
                keep_pts = ~through
                pts, rgb, us, vs = pts[keep_pts], rgb[keep_pts], us[keep_pts], vs[keep_pts]
                if nrm is not None:
                    nrm = nrm[keep_pts]
                seen_through_total += int(through.sum())
            labels = None
            if label_index is not None:
                labels = pixel_labels(us, vs, f["detections"], label_index,
                                      yielding, set(vocabulary.with_role("furniture", "storage")))
                on_unreliable = np.isin(labels, unreliable_ids)
                if filled_map is not None and on_unreliable.any():
                    # A filled outline stands where its surface is, so its
                    # points take the name of what it sits on: the wardrobe
                    # whose door it is, or nothing (the wall).
                    at = np.flatnonzero(on_unreliable)
                    at = at[filled_map[pixel(us[at], vs[at])]]
                    if len(at):
                        solid = [d for d in f["detections"] if d["label"] not in vocabulary.unreliable]
                        labels[at] = pixel_labels(us[at], vs[at], solid, label_index)
                        on_unreliable[at] = False
                        filled_on_planes += len(at)
                # Mirrors, windows and screens that could not be filled: the
                # depth there is a reflection or the view outside, so drop them.
                keep = ~on_unreliable
                dropped_unreliable += int((~keep).sum())
                pts, rgb, labels = pts[keep], rgb[keep], labels[keep]
                if nrm is not None:
                    nrm = nrm[keep]
            all_pts.append(pts)
            all_cols.append(rgb)
            all_norms.append(nrm)
            all_labels.append(labels)
        print(f"Fused {len(kept)} keyframes ({on_median} without their own scale fit, "
              f"{len(left_out)} left out); {seen_through_total:,} points another keyframe sees through dropped")
        meta.update({"colmap_units_per_metre": s_med, "scale_spread": spread,
                     "scale_inconsistent": spread > 0.3,
                     "frames_on_median_scale": on_median,
                     "scale_method": "support-weighted, smoothed along the walk",
                     "frames_left_out": [images[frames[i]["id"]]["name"] for i in left_out],
                     "points_seen_through": seen_through_total})

    meta["pixels_without_ray"] = round(float(np.mean(without_ray)), 4) if without_ray else 0.0
    if meta["pixels_without_ray"] > 0:
        print(f"Warning: the solved lens folds inside the frame, so {np.mean(without_ray):.2%} of the "
              f"keyframes' pixels have no ray and were left out (worst frame {max(without_ray):.1%}). "
              "The camera solve is suspect: check the COLMAP model (reconstruct.py), or pass "
              "another with --model-dir.")

    if not all_pts:
        sys.exit("No keyframe could be densified.")
    normals = np.vstack(all_norms) \
        if all_norms and all(n is not None for n in all_norms) else None
    labels = np.concatenate(all_labels) \
        if all_labels and all(l is not None for l in all_labels) else None
    if labels is not None and meta.get("colmap_units_per_metre") and len(all_pts) > 1:
        frame_of = np.concatenate([np.full(len(p), i) for i, p in enumerate(all_pts)])
        labels, changed = vote_labels(np.vstack(all_pts), labels, frame_of,
                                      VOTE_VOXEL_M * meta["colmap_units_per_metre"])
        meta["labels_changed_by_vote"] = changed
        print(f"Labels voted across keyframes: {changed:,} points changed")
    cloud = PointCloud(np.vstack(all_pts), np.vstack(all_cols), normals, labels,
                       label_names if labels is not None else None)
    print(f"Fused: {len(cloud):,} points; cleaning up")
    cloud = trim_far_points(cloud, factor=8)
    extent = np.linalg.norm(cloud.points.max(0) - cloud.points.min(0))
    cloud = remove_outliers(cloud, voxel_size=extent / 200, min_neighbors=4)
    if cloud.labels is not None:
        counts = {name: int((cloud.labels == i).sum())
                  for i, name in enumerate(cloud.label_names) if i > 0}
        counts = {name: n for name, n in counts.items() if n}
        meta["labels"] = counts
        meta["points_dropped_unreliable"] = dropped_unreliable
        meta["points_on_mirror_planes"] = filled_on_planes
        meta["mirror_planes"] = {label: {"outlines": len(rs), "filled": sum(1 for r in rs if r["filled"]),
                                         "pixels": sum(r["filled"] for r in rs)}
                                 for label, rs in fills.items()}
        summary = ", ".join(f"{name} {n:,}" for name, n in
                            sorted(counts.items(), key=lambda kv: -kv[1]))
        print(f"Labelled: {summary or 'nothing detected'}"
              f" ({filled_on_planes:,} points on mirrors/windows/screens stood on their planes, "
              f"{dropped_unreliable:,} dropped)")

    out = Path(args.output) if args.output else space / "cloud-dense.ply"
    save_ply(cloud, out)
    meta["points"] = len(cloud)
    # The metadata describes this cloud, so it goes beside it: densify.json
    # for the default output, <output>.json for a custom one.
    meta_path = out.with_suffix(".json") if args.output else space / "densify.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Dense cloud: {len(cloud):,} points -> {out}")


if __name__ == "__main__":
    main()
