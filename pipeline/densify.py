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
from semantics import (SEGMENTER_ID, UNRELIABLE, VOCABULARY, Detector, Segmenter,
                       default_device, pixel_labels)

MOGE_CHECKPOINT = "Ruicheng/moge-2-vitl-normal"
MIN_ANCHORS = 12


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


def fit_scale(pred_depth, sparse_depth):
    """Robust single scale for metric predictions: sparse ~ s * predicted.
    Returns None when too few anchors land on valid predicted depth."""
    valid = np.isfinite(pred_depth) & (pred_depth > 0)
    if valid.sum() < 8:
        return None
    ratio = sparse_depth[valid] / pred_depth[valid]
    s = float(np.median(ratio))
    for _ in range(3):
        keep = np.abs(np.log(ratio / s)) < 0.25
        if keep.sum() < 8:
            break
        s = float(np.median(ratio[keep]))
    return s


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
    they came from, for a pixel grid of one frame."""
    fx, fy, cx, cy = cam["params"][:4]
    W, H = img.size
    us, vs = np.meshgrid(np.arange(0, W, stride), np.arange(0, H, stride))
    us, vs = us.ravel(), vs.ravel()
    z = z_at(us, vs)
    ok = np.isfinite(z) & (z > 0.05) & (z < max_depth)
    us, vs, z = us[ok], vs[ok], z[ok]
    rays = np.column_stack([(us - cx) / fx, (vs - cy) / fy, np.ones(len(us))])
    cam_pts = rays * z[:, None]
    world_pts = (info["R"].T @ (cam_pts - info["t"]).T).T
    rgb = np.asarray(img.convert("RGB"))[vs, us]
    normals = None
    if n_at is not None:
        cam_normals = n_at(us, vs).astype(np.float64)
        normals = (info["R"].T @ cam_normals.T).T.astype(np.float32)
    return world_pts.astype(np.float32), rgb.astype(np.uint8), normals, (us, vs)


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
    parser.add_argument("--depth-model", choices=["moge", "da2"], default="moge",
                        help="monocular depth network (default moge)")
    parser.add_argument("--work-size", type=int, default=1280,
                        help="longest side, in pixels, that MoGe predicts at (default 1280)")
    parser.add_argument("--no-semantics", action="store_true",
                        help="skip object detection: no labels on the cloud, and no "
                             "filtering of mirrors, windows and screens")
    parser.add_argument("--no-outlines", action="store_true",
                        help="label whole detection rectangles instead of cutting "
                             "each one to the object's outline with SAM 2.1")
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

    import torch
    from PIL import Image

    device = default_device()
    if args.depth_model == "moge":
        predictor = MoGeDepth(device, args.work_size)
        print(f"MoGe-2 ({MOGE_CHECKPOINT}) on {device}")
    else:
        from transformers import pipeline as hf_pipeline
        predictor = hf_pipeline(
            "depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf",
            device=device,
        )
        print(f"Depth Anything V2 (small) on {device}")

    use_detector = args.depth_model == "moge" and not args.no_semantics
    label_index, label_names, unreliable_ids = None, None, np.zeros(0, int)

    all_pts, all_cols, all_norms, all_labels = [], [], [], []
    dropped_unreliable = 0
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
            pts, cols, _, _ = back_project(
                info, cam, img,
                lambda us, vs: 1.0 / np.maximum(a * pred_inv[vs, us] + b, 1e-6),
                args.stride, np.percentile(depth_true, 98) * 3,
            )
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
                s = fit_scale(depth[sy, sx], depth_true)
            frames.append({
                "id": image_id, "depth": depth, "k": k, "s": s, "normal": normal,
                "detections": [],
                "cap": np.percentile(depth_true, 98) * 3 if len(px) >= MIN_ANCHORS else None,
            })
            note = f"scale {s:.3f}" if s is not None else "no scale fit"
            print(f"  [{i+1}/{len(keys)}] {info['name']}: {note} ({len(px)} anchors)")

        # One model in memory at a time (8 GB Macs): free MoGe before the
        # detector loads, and the detector before back-projection.
        del predictor
        release_model_memory(torch)
        if use_detector:
            try:
                detector = Detector(device=device)
                label_names = ["unlabelled"] + list(VOCABULARY)
                label_index = {name: i for i, name in enumerate(label_names)}
                unreliable_ids = np.array([label_index[name] for name in UNRELIABLE])
                print(f"Object detection: GroundingDINO-tiny on {detector.device}")
                for f in frames:
                    name = images[f["id"]]["name"]
                    f["detections"] = detector.detect(Image.open(workspace / "images" / name))
                    seen = ", ".join(sorted({d["label"] for d in f["detections"]}))
                    print(f"  {name}: {seen or 'nothing detected'}")
                # Kept so stage 3 can show Claude the frame where each object
                # was actually seen (agent.py object_frames).
                meta["detections"] = {
                    images[f["id"]]["name"]: [
                        {"label": d["label"], "score": d["score"],
                         "box": [round(v) for v in d["box"]]} for d in f["detections"]]
                    for f in frames if f["detections"]}
                del detector
                release_model_memory(torch)
            except Exception as exc:  # transformers or weights missing
                label_index = None
                print(f"Object detection unavailable ({exc}); continuing without labels")
        if label_index is not None:
            # A rectangle around a bed also holds floor and curtain; cut each
            # one down to the object's own outline before labelling points.
            meta["outlines"] = "boxes"
            if not args.no_outlines and any(f["detections"] for f in frames):
                try:
                    segmenter = Segmenter(device=device)
                    print(f"Object outlines: SAM 2.1 hiera-tiny on {segmenter.device}")
                    for f in frames:
                        name = images[f["id"]]["name"]
                        done = segmenter.outline(Image.open(workspace / "images" / name),
                                                 f["detections"])
                        print(f"  {name}: {done}/{len(f['detections'])} outlined")
                    del segmenter
                    release_model_memory(torch)
                    meta["outlines"] = SEGMENTER_ID
                except Exception as exc:  # weights missing, or the model failed
                    for f in frames:
                        for det in f["detections"]:
                            det.pop("mask", None)
                    print(f"Object outlines unavailable ({exc}); "
                          "labelling whole detection rectangles")

        fitted = np.array([f["s"] for f in frames if f["s"] is not None])
        if len(fitted) == 0:
            sys.exit("No keyframe sees enough sparse points to fix the scale; "
                     "check the COLMAP model (reconstruct.py) and retry.")
        s_med = float(np.median(fitted))
        spread = float(np.subtract(*np.percentile(fitted, [75, 25])) / s_med)
        caps = [f["cap"] for f in frames if f["cap"] is not None]
        default_cap = float(np.median(caps)) if caps else np.inf
        print(f"Scale: 1 m = {s_med:.4f} COLMAP units "
              f"(interquartile spread {spread:.0%} over {len(fitted)} frames)")
        if spread > 0.3:
            print("Warning: keyframes disagree on scale by more than 30%, so the "
                  "camera model is probably inconsistent (pieces joined at "
                  "different scales). Rerun reconstruct.py with --mapper incremental.")

        # Pass 2: back-project. A frame without a trustworthy fit (none, or
        # more than 25% off the median) uses the median scale.
        on_median = 0
        for f in frames:
            info = images[f["id"]]
            cam = cameras[info["camera_id"]]
            img = Image.open(workspace / "images" / info["name"])
            s = f["s"]
            if s is None or abs(np.log(s / s_med)) > np.log(1.25):
                s, on_median = s_med, on_median + 1
            depth, k, normal = f["depth"], f["k"], f["normal"]
            rows, cols = depth.shape

            def pixel(us, vs, k=k, rows=rows, cols=cols):
                """Full-frame pixel -> index into the working-size maps."""
                return (np.minimum((vs * k).astype(int), rows - 1),
                        np.minimum((us * k).astype(int), cols - 1))

            pts, rgb, nrm, (us, vs) = back_project(
                info, cam, img,
                lambda us, vs: s * depth[pixel(us, vs)],
                args.stride, f["cap"] if f["cap"] is not None else default_cap,
                n_at=(lambda us, vs: normal[pixel(us, vs)]) if normal is not None else None,
            )
            labels = None
            if label_index is not None:
                labels = pixel_labels(us, vs, f["detections"], label_index)
                # Mirrors, windows and screens: the depth there is a
                # reflection or the view outside, so drop those points.
                keep = ~np.isin(labels, unreliable_ids)
                dropped_unreliable += int((~keep).sum())
                pts, rgb, labels = pts[keep], rgb[keep], labels[keep]
                if nrm is not None:
                    nrm = nrm[keep]
            all_pts.append(pts)
            all_cols.append(rgb)
            all_norms.append(nrm)
            all_labels.append(labels)
        print(f"Fused {len(frames)} keyframes ({on_median} on the median scale)")
        meta.update({"colmap_units_per_metre": s_med, "scale_spread": spread,
                     "scale_inconsistent": spread > 0.3,
                     "frames_on_median_scale": on_median})

    if not all_pts:
        sys.exit("No keyframe could be densified.")
    normals = np.vstack(all_norms) \
        if all_norms and all(n is not None for n in all_norms) else None
    labels = np.concatenate(all_labels) \
        if all_labels and all(l is not None for l in all_labels) else None
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
        summary = ", ".join(f"{name} {n:,}" for name, n in
                            sorted(counts.items(), key=lambda kv: -kv[1]))
        print(f"Labelled: {summary or 'nothing detected'}"
              f" (dropped {dropped_unreliable:,} points on mirrors/windows/screens)")

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
