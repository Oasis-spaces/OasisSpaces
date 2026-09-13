#!/usr/bin/env python3
"""OasisSpaces reconstruction pipeline.

Turns a folder of photos (or a walkthrough video) of a space into a point
cloud ready for the editor:

    images/video -> COLMAP SfM (features, matching, mapping) -> sparse cloud
                 -> optional CUDA dense reconstruction -> cleanup -> cloud.ply

Usage:
    python3 pipeline/reconstruct.py <images_dir | video_file> --name kitchen
    python3 pipeline/reconstruct.py walkthrough.mp4 --name loft --fps 2
    python3 pipeline/reconstruct.py photos/ --name office --dense
    python3 pipeline/reconstruct.py room.mov --name room --mapper incremental

Mapping uses COLMAP's global mapper (GLOMAP, built into COLMAP 4.x) by
default: it solves all cameras at once and fragments hard captures far less
than the incremental mapper, which remains available as --mapper incremental
and as an automatic fallback.

Output lands in spaces/<name>/:
    workspace/   COLMAP database, extracted frames, sparse model
    cloud.ply    cleaned point cloud (open it in the editor)
"""

from __future__ import annotations

import argparse
import json
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from pointcloud import (
    load_ply, remove_outliers, save_ply, trim_far_points, voxel_downsample,
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".heic", ".tif", ".tiff", ".bmp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
# Largest believable camera move between consecutive video frames, as a
# fraction of the scene's size (coherent models stay under ~0.1).
MAX_PATH_JUMP = 0.3
# Random seeds tried for the global mapper before falling back to incremental.
GLOBAL_MAPPER_SEEDS = (0, 1, 2)
# COLMAP 4.x feature types: classic SIFT, or the learned ALIKED detector, which
# finds far more on the low-texture surfaces (white walls) that SIFT has
# nothing to grip. ALIKED is matched by brute force, not LightGlue: on this
# CPU-only build LightGlue took 99 s for one frame's neighbours and then
# crashed inside Apple's ONNX runtime, while brute force matched 42 frames in
# 3 s, and the pan capture solved 42/42 frames in one model with it.
FEATURE_TYPES = {
    "sift": ("SIFT", "SIFT_BRUTEFORCE"),
    "aliked": ("ALIKED_N16ROT", "ALIKED_BRUTEFORCE"),
}


def run(command: list[str], log_file: Path, check: bool = True) -> bool:
    """Run a step, logging its output. With check=False, report failure
    instead of exiting so the caller can fall back."""
    print(f"  $ {' '.join(command[:4])} ...")
    with open(log_file, "a") as log:
        log.write(f"\n=== {' '.join(command)} ===\n")
        log.flush()
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        if not check:
            return False
        sys.exit(
            f"Step failed ({command[0]} {command[1] if len(command) > 1 else ''}), "
            f"see log: {log_file}"
        )
    return True


def registered_images(model: Path) -> int:
    """Number of frames COLMAP solved a camera for (images.bin header)."""
    with open(model / "images.bin", "rb") as f:
        return struct.unpack("<Q", f.read(8))[0]


def solved_models(sparse_dir: Path) -> list[Path]:
    """Model folders (COLMAP 4 also leaves a project.ini beside them)."""
    return sorted(d for d in sparse_dir.iterdir() if (d / "images.bin").exists())


def camera_path(model: Path):
    """Frames in video order: names, camera centres, frame numbers, and the
    scene's size (5-95 percentile extent of the points), or None when the
    model is too small to judge."""
    from densify import read_images_bin, read_points3d_bin

    images = sorted(read_images_bin(model / "images.bin").values(),
                    key=lambda v: v["name"])
    points = np.array(list(read_points3d_bin(model / "points3D.bin").values()))
    if len(images) < 3 or len(points) < 100:  # too few points to size the scene
        return None
    names = [v["name"] for v in images]
    centers = np.array([-v["R"].T @ v["t"] for v in images])
    index = np.array([int("".join(filter(str.isdigit, name)) or 0) for name in names])
    scene = float(np.linalg.norm(np.percentile(points, 95, 0) - np.percentile(points, 5, 0)))
    return names, centers, index, scene


def camera_path_jump(model: Path, skip=()) -> float:
    """Largest camera move between consecutive video frames, as a fraction of
    the scene's size. A coherent walkthrough stays well under 0.1. A global
    solve that glued separate pieces together at different scales makes the
    camera jump by more than the whole scene, while each piece's reprojection
    error still looks healthy. Frames in `skip` (detours) are left out."""
    path = camera_path(model)
    if path is None:
        return 0.0
    names, centers, index, scene = path
    keep = np.array([name not in set(skip) for name in names])
    centers, index = centers[keep], index[keep]
    if len(centers) < 2 or scene <= 0:
        return 0.0
    steps = np.linalg.norm(np.diff(centers, axis=0), axis=1) \
        / np.maximum(np.diff(index), 1)
    return float(steps.max() / scene)


# A few frames the solver placed wrongly show up as a detour: the camera leaps
# away and leaps straight back a few frames later, landing near where it left.
# A quick pan onto something seen for a moment does this (walkthrough frames
# 112-114 matched 86-376 points, against ~3,000 for their neighbours, and sat
# 3.3 units away). Unlike pieces glued at different scales, the path either
# side agrees, so only those frames are dropped from the model.
DETOUR_STEP = 0.1          # a leap: a per-frame move above this share of the scene...
DETOUR_MEDIAN_FACTOR = 8   # ...and this many times the median move
MAX_DETOUR_FRAMES = 6


def detour_frames(model: Path) -> list[str]:
    """Names of frames on a detour (see DETOUR_STEP)."""
    path = camera_path(model)
    if path is None:
        return []
    names, centers, index, scene = path
    steps = np.linalg.norm(np.diff(centers, axis=0), axis=1) / np.maximum(np.diff(index), 1)
    leaps = np.where((steps > DETOUR_STEP * scene)
                     & (steps > DETOUR_MEDIAN_FACTOR * np.median(steps)))[0]
    misplaced = set()
    for a in leaps:
        for b in leaps[leaps > a]:
            if b - a > MAX_DETOUR_FRAMES:
                break
            back = np.linalg.norm(centers[b + 1] - centers[a]) / max(index[b + 1] - index[a], 1)
            if back <= DETOUR_STEP * scene:
                misplaced.update(range(a + 1, b + 1))
                break
    return [names[i] for i in sorted(misplaced)]


def drop_detours(model: Path, workspace: Path, log: Path) -> list[str]:
    """Remove detour frames from a model, and record them in
    workspace/dropped-frames.json."""
    names = detour_frames(model)
    if not names:
        return []
    listing = workspace / f"dropped-frames-{model.name}.txt"
    listing.write_text("\n".join(names) + "\n")
    run(["colmap", "image_deleter", "--input_path", str(model),
         "--output_path", str(model), "--image_names_path", str(listing)], log)
    record_path = workspace / "dropped-frames.json"
    record = json.loads(record_path.read_text()) if record_path.exists() else {}
    record[model.name] = names
    record_path.write_text(json.dumps(record, indent=1) + "\n")
    print(f"  model {model.name}: dropped {len(names)} misplaced frame(s) "
          f"({', '.join(names)})")
    return names


def require_binary(name: str, install_hint: str) -> None:
    if shutil.which(name) is None:
        sys.exit(f"{name} not found. Install it with: {install_hint}")


def sharpness(path: Path) -> float:
    """Higher = sharper. Variance of image gradients on a downscaled copy."""
    from PIL import Image

    img = Image.open(path).convert("L")
    img.thumbnail((480, 480))
    arr = np.asarray(img, dtype=np.float32)
    gy, gx = np.gradient(arr)
    return float((gx * gx + gy * gy).mean())


def extract_frames(video: Path, images_dir: Path, fps: float, log: Path) -> None:
    """Extract at 3x the requested rate, keep the sharpest frame per window.

    Motion blur varies frame to frame (hand shake, walking bounce); dense
    extraction plus sharpness selection means a fast or shaky video still
    contributes its crispest moments at the requested spacing.
    """
    oversample = 3
    print(f"Extracting frames from {video.name} at {fps} fps "
          f"({oversample}x oversampled, keeping sharpest per window)")
    images_dir.mkdir(parents=True, exist_ok=True)
    run(
        ["ffmpeg", "-y", "-i", str(video), "-vf", f"fps={fps * oversample}",
         "-qscale:v", "2", str(images_dir / "candidate_%05d.jpg")],
        log,
    )
    candidates = sorted(images_dir.glob("candidate_*.jpg"))
    kept = 0
    for start in range(0, len(candidates), oversample):
        window = candidates[start:start + oversample]
        best = max(window, key=sharpness)
        kept += 1
        best.rename(images_dir / f"frame_{kept:05d}.jpg")
        for other in window:
            if other.exists():
                other.unlink()
    print(f"Kept {kept} sharpest frames of {len(candidates)} candidates")


def collect_images(source: Path, images_dir: Path) -> int:
    images_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for path in sorted(source.iterdir()):
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            target = images_dir / path.name
            if not target.exists():
                shutil.copy2(path, target)
            count += 1
    return count


def cuda_available() -> bool:
    try:
        output = subprocess.run(
            ["colmap", "-h"], capture_output=True, text=True
        ).stdout
    except OSError:
        return False
    return "with CUDA" in output


def gpu_flags(subcommand: str, new_prefix: str, old_prefix: str) -> list[str]:
    """Force CPU SIFT on headless non-CUDA Linux (e.g. a bare cloud VM).

    COLMAP's GPU SIFT needs CUDA or a display there, while macOS builds
    handle themselves — so only intervene where use_gpu=1 would crash.
    The option prefix also changed between COLMAP 3.x and 4.x, so detect
    which spelling this build understands instead of assuming a version.
    """
    if sys.platform == "darwin" or cuda_available():
        return []
    result = subprocess.run(
        ["colmap", subcommand, "--help"], capture_output=True, text=True
    )
    help_text = result.stdout + result.stderr
    for prefix in (new_prefix, old_prefix):
        if f"--{prefix}.use_gpu" in help_text:
            return [f"--{prefix}.use_gpu", "0"]
    return []


def sparse_reconstruction(
    workspace: Path, images_dir: Path, sequential: bool, mapper: str,
    features: str, log: Path
) -> Path:
    database = workspace / "database.db"
    sparse_dir = workspace / "sparse"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    extractor_type, matcher_type = FEATURE_TYPES[features]

    print(f"COLMAP: extracting features ({extractor_type})")
    run(
        ["colmap", "feature_extractor",
         "--database_path", str(database),
         "--image_path", str(images_dir),
         "--ImageReader.camera_model", "OPENCV",
         "--ImageReader.single_camera", "1",
         "--FeatureExtraction.type", extractor_type,
         *gpu_flags("feature_extractor", "FeatureExtraction", "SiftExtraction")],
        log,
    )

    matcher = "sequential_matcher" if sequential else "exhaustive_matcher"
    print(f"COLMAP: matching features ({matcher}, {matcher_type})")
    run(
        ["colmap", matcher,
         "--database_path", str(database),
         "--FeatureMatching.type", matcher_type,
         *gpu_flags(matcher, "FeatureMatching", "SiftMatching")],
        log,
    )

    mapping_args = ["--database_path", str(database),
                    "--image_path", str(images_dir),
                    "--output_path", str(sparse_dir)]
    if mapper == "global":
        rejected = workspace / "sparse-global-rejected"
        shutil.rmtree(rejected, ignore_errors=True)
        # The global solve varies from run to run on hard captures, and a
        # re-solve only costs the mapping (features and matches are reused),
        # so try a few seeds before falling back to the slower incremental mapper.
        for attempt, seed in enumerate(GLOBAL_MAPPER_SEEDS, 1):
            print("COLMAP: global mapping (all cameras solved together)"
                  + (f", attempt {attempt} with seed {seed}" if attempt > 1 else ""))
            problem = None
            if not run(["colmap", "global_mapper", "--default_random_seed", str(seed),
                        *mapping_args], log, check=False):
                problem = "the global mapper failed"
            elif not solved_models(sparse_dir):
                problem = "the global mapper registered nothing"
            elif sequential:
                # Check the model everything downstream uses: the one with the
                # most points.
                main_model = max(solved_models(sparse_dir),
                                 key=lambda m: (m / "points3D.bin").stat().st_size)
                # A detour of a few misplaced frames is dropped below, so it
                # does not condemn the whole solve.
                jump = camera_path_jump(main_model, skip=detour_frames(main_model))
                if jump > MAX_PATH_JUMP:
                    problem = (f"between two consecutive frames the camera jumps "
                               f"{jump:.1f}x the scene's size, so it joined separate "
                               f"pieces at inconsistent scales")
            if problem is None:
                break
            print(f"Global model rejected: {problem}.")
            # Keep the latest rejected attempt for debugging, out of downstream's way.
            shutil.rmtree(rejected, ignore_errors=True)
            sparse_dir.rename(rejected)
            sparse_dir.mkdir()
        else:
            print("Falling back to the incremental mapper.")
            mapper = "incremental"
    if mapper == "incremental":
        print("COLMAP: incremental mapping — this is the slow part")
        run(["colmap", "mapper", *mapping_args], log)

    models = solved_models(sparse_dir)
    if not models:
        sys.exit(
            "COLMAP could not register the images into a model. "
            "Capture more overlapping photos (60-80% overlap between shots) "
            f"and retry. Log: {log}"
        )

    if sequential:
        (workspace / "dropped-frames.json").unlink(missing_ok=True)
        for model in models:
            drop_detours(model, workspace, log)

    # The mapper may fragment a difficult capture into several models;
    # export each and keep the one with the most points.
    total_frames = sum(1 for p in images_dir.iterdir()
                       if p.suffix.lower() in IMAGE_EXTENSIONS)
    best_ply, best_model, best_points = None, None, -1
    for model in models:
        model_ply = workspace / f"model_{model.name}.ply"
        run(
            ["colmap", "model_converter",
             "--input_path", str(model),
             "--output_path", str(model_ply),
             "--output_type", "PLY"],
            log,
        )
        n = len(load_ply(model_ply))
        print(f"  model {model.name}: {n:,} points, "
              f"{registered_images(model)} of {total_frames} frames")
        if n > best_points:
            best_ply, best_model, best_points = model_ply, model, n
    if len(models) > 1:
        print(
            f"Note: reconstruction fragmented into {len(models)} pieces "
            f"(weak matching); using the largest. More overlap or more "
            f"texture in the capture will help."
        )
    return best_ply, best_model


def dense_reconstruction(
    workspace: Path, images_dir: Path, model_dir: Path, log: Path
) -> Path:
    dense_dir = workspace / "dense"
    dense_dir.mkdir(parents=True, exist_ok=True)
    print("COLMAP: undistorting images")
    run(
        ["colmap", "image_undistorter",
         "--image_path", str(images_dir),
         "--input_path", str(model_dir),
         "--output_path", str(dense_dir)],
        log,
    )
    print("COLMAP: dense stereo (CUDA)")
    run(
        ["colmap", "patch_match_stereo", "--workspace_path", str(dense_dir)],
        log,
    )
    print("COLMAP: fusing depth maps")
    fused = dense_dir / "fused.ply"
    run(
        ["colmap", "stereo_fusion",
         "--workspace_path", str(dense_dir),
         "--output_path", str(fused)],
        log,
    )
    return fused


def cleanup(raw_ply: Path, output: Path, voxel_size: float | None) -> None:
    cloud = load_ply(raw_ply)
    print(f"Cleanup: {len(cloud):,} raw points")
    if len(cloud) == 0:
        sys.exit("Reconstruction produced an empty cloud — nothing to save.")
    if voxel_size is not None and voxel_size <= 0:
        sys.exit("--voxel must be positive.")
    # Always drop far-field triangulation junk: a handful of wildly distant
    # points otherwise wreck camera fitting in every viewer.
    cloud = trim_far_points(cloud)
    if len(cloud) < 20_000 and voxel_size is None:
        # A small sparse cloud has no density to spare — keep every point.
        save_ply(cloud, output)
        print(f"Cloud is sparse; keeping all {len(cloud):,} points -> {output}")
        return
    if voxel_size is None:
        # COLMAP's scene scale is arbitrary; size voxels off the extent.
        extent = cloud.points.max(axis=0) - cloud.points.min(axis=0)
        voxel_size = float(np.linalg.norm(extent)) / 800
    if voxel_size > 0:
        cloud = remove_outliers(cloud, voxel_size=voxel_size * 4, min_neighbors=3)
        cloud = voxel_downsample(cloud, voxel_size=voxel_size)
    save_ply(cloud, output)
    print(f"Cleanup: {len(cloud):,} points kept -> {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="folder of photos, or a video file")
    parser.add_argument("--name", required=True, help="space name (output folder)")
    parser.add_argument("--fps", type=float, default=2.0,
                        help="frames per second to extract from video (default 2)")
    parser.add_argument("--features", choices=list(FEATURE_TYPES), default="sift",
                        help="feature type: COLMAP's SIFT, or learned ALIKED + "
                             "LightGlue (slower on a CPU-only build, better on "
                             "low-texture walls)")
    parser.add_argument("--mapper", choices=["global", "incremental"],
                        default="global",
                        help="COLMAP mapper (default global; incremental is "
                             "the pre-4.0 behaviour)")
    parser.add_argument("--dense", action="store_true",
                        help="attempt CUDA dense reconstruction after SfM")
    parser.add_argument("--voxel", type=float, default=None,
                        help="cleanup voxel size in scene units "
                             "(default: scene extent / 800)")
    args = parser.parse_args()

    require_binary("colmap", "brew install colmap")
    source = Path(args.source).expanduser()
    if not source.exists():
        sys.exit(f"Source not found: {source}")

    root = Path(__file__).resolve().parent.parent
    space_dir = root / "spaces" / args.name
    workspace = space_dir / "workspace"
    images_dir = workspace / "images"
    workspace.mkdir(parents=True, exist_ok=True)
    log = workspace / "colmap.log"

    # A rerun must not mix with stale state: features/models/frames from a
    # previous attempt (or a different --fps / source) would corrupt the
    # solve. Wipe everything derived, including old frames.
    for stale in [workspace / "database.db", workspace / "sparse",
                  workspace / "sparse-global-rejected", workspace / "dense", images_dir,
                  *workspace.glob("model_*.ply"), workspace / "sparse.ply",
                  workspace / "dropped-frames.json", *workspace.glob("dropped-frames-*.txt")]:
        if stale.is_dir():
            shutil.rmtree(stale)
        elif stale.exists():
            stale.unlink()

    is_video = source.is_file() and source.suffix.lower() in VIDEO_EXTENSIONS
    if is_video:
        require_binary("ffmpeg", "brew install ffmpeg")
        extract_frames(source, images_dir, args.fps, log)
    elif source.is_dir():
        count = collect_images(source, images_dir)
        if count < 10:
            sys.exit(
                f"Only {count} images found in {source} — a space needs at "
                "least a few dozen overlapping photos from every angle."
            )
        print(f"Using {count} images from {source}")
    else:
        sys.exit(f"{source} is neither an image folder nor a video file")

    sparse_ply, best_model = sparse_reconstruction(
        workspace, images_dir, sequential=is_video, mapper=args.mapper,
        features=args.features, log=log
    )

    result_ply = sparse_ply
    if args.dense:
        if cuda_available():
            result_ply = dense_reconstruction(
                workspace, images_dir, best_model, log
            )
        else:
            print(
                "Skipping dense reconstruction: this COLMAP build has no CUDA "
                "(normal on macOS). The sparse cloud is still produced; for a "
                "dense cloud run the dense step on a CUDA machine."
            )

    cleanup(result_ply, space_dir / "cloud.ply", args.voxel)
    print(
        f"\nDone. View it: serve the project root (python3 -m http.server 8734)"
        f"\nthen open http://localhost:8734/editor/?load=/spaces/{args.name}/cloud.ply"
    )


if __name__ == "__main__":
    main()
