#!/usr/bin/env python3
"""Train a space's splat with Spirula Studio, as a plug-in trainer for stage 4.

Spirula Studio (https://github.com/harry7557558/spirula-studio, GPL-3.0) is run
as a separate program, like COLMAP and Blender: nothing of it is linked or
copied here. It brings what OpenSplat lacks: normal and optional depth
supervision from its own MoGe-based `geometry` step, per-photo exposure and
lens correction (PPISP, bilateral grid), MCMC-style densification, and a
Vulkan backend that runs on Apple Silicon.

The space's own cameras, frames and dense-cloud seed go in, so only the
trainer differs from the OpenSplat runs and Claude can judge them alike
(pipeline/agent.py choose_training). The result comes back as
<space>/splat-spirula.ply in the camera solve's frame and the pipeline's
62-property layout (Spirula writes standard 3DGS PLY and, with the default
`--scene-center none`, keeps the dataset's frame).

The binary: $SPIRULA, else tools/spirula-studio/Spirula Studio.app (macOS) or
tools/spirula-studio/spirula, else /Applications/Spirula Studio.app. Install
by downloading a release from GitHub and unzipping it there.

Usage:
    python3 pipeline/splat_spirula.py spaces/<name> [--iters 15000] [--divisor 4]
        [--cap 300000] [--depth-weight 0] [--floaters off|mild|strong]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
from pointcloud import space_model_dir  # noqa: E402

# Settings that fit an 8 GB Mac (measured Sep 2026 on an M2: 15,000 steps at
# quarter resolution with 300k splats took 32 minutes and 3.4 GB of swap; at
# half resolution with 600k splats the machine ran out of disk and memory).
DEFAULTS = {"iters": 15000, "divisor": 4, "cap": 300_000, "depth_weight": 0.0, "floaters": "off"}
GEOMETRY_MAX_SIZE = 1064


def find_binary() -> Path | None:
    candidates = [os.environ.get("SPIRULA"),
                  ROOT / "tools" / "spirula-studio" / "Spirula Studio.app" / "Contents" / "MacOS" / "spirula",
                  ROOT / "tools" / "spirula-studio" / "spirula",
                  "/Applications/Spirula Studio.app/Contents/MacOS/spirula",
                  shutil.which("spirula")]
    for c in candidates:
        if c and Path(c).is_file() and os.access(c, os.X_OK):
            return Path(c)
    return None


def dataset_dir(space: Path) -> Path:
    """<space>/splat-spirula/: a COLMAP-layout dataset of the space's own
    frames and cameras, seeded from the dense cloud (pipeline/splat_seed.py
    must have run, which stage 4 does before any training)."""
    model = space_model_dir(space)
    if model is None:
        raise SystemExit(f"no camera model in {space / 'workspace' / 'sparse'}")
    seed = space / "splat-project" / "points3D.bin"
    if not seed.exists():
        raise SystemExit(f"{seed} missing: run pipeline/splat_seed.py {space} first")
    data = space / "splat-spirula"
    sparse = data / "sparse" / "0"
    sparse.mkdir(parents=True, exist_ok=True)
    for name, target in (("images", space / "workspace" / "images"),
                         ("sparse/0/cameras.bin", model / "cameras.bin"),
                         ("sparse/0/images.bin", model / "images.bin")):
        link = data / name
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(target.resolve())
    # The seed is copied, not linked: splat_seed.py rewrites its file.
    if not (sparse / "points3D.bin").exists() or (sparse / "points3D.bin").stat().st_mtime < seed.stat().st_mtime:
        shutil.copyfile(seed, sparse / "points3D.bin")
    return data


def run(command: list, log_path: Path, done_file: Path | None = None) -> int:
    """Run Spirula, appending its (carriage-return heavy) output to the log
    and echoing its step lines.

    Spirula never exits after training: it keeps serving a viewer, and keeps
    its output open. So the output is read as it arrives (a blocking read of
    a fixed size waits forever on a short run), and the process is ended once
    it says training is complete, or once `done_file` (the final checkpoint's
    splat) has been on disk, unchanged, for ten seconds."""
    import selectors

    with open(log_path, "ab") as log:
        log.write((" ".join(str(c) for c in command) + "\n").encode())
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        fd = proc.stdout.fileno()
        selector = selectors.DefaultSelector()
        selector.register(fd, selectors.EVENT_READ)
        buffer, last_shown, finished = b"", time.time(), False
        while proc.poll() is None and not finished:
            if selector.select(timeout=5):
                chunk = os.read(fd, 65536)
                if not chunk:
                    break
                log.write(chunk)
                log.flush()
                buffer += chunk
                while b"\r" in buffer or b"\n" in buffer:
                    cut = min(i for i in (buffer.find(b"\r"), buffer.find(b"\n")) if i >= 0)
                    line, buffer = buffer[:cut].decode("utf-8", "replace"), buffer[cut + 1:]
                    shown = ("complete" in line.lower() or "error" in line.lower() or "written" in line
                             or (line.startswith("step") and time.time() - last_shown > 60))
                    if shown:
                        print("   ", line[:120], flush=True)
                        last_shown = time.time()
                    finished = finished or "Training complete" in line
            if done_file is not None and done_file.exists() and time.time() - done_file.stat().st_mtime > 10:
                finished = True
        if finished and proc.poll() is None:
            time.sleep(3)   # the checkpoint is written before "Training complete"
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
            return 0
        return proc.wait()


def train(space: Path, out: Path, iters: int, divisor: int, cap: int, depth_weight: float,
          floaters: str, log_path: Path | None = None) -> dict:
    binary = find_binary()
    if binary is None:
        raise SystemExit("Spirula Studio not found: set SPIRULA, or install it under tools/spirula-studio/")
    log_path = log_path or space / "splat-spirula.log"
    data = dataset_dir(space)
    # The cameras the solve placed, not the frames in the folder: Spirula trains on those.
    from densify import read_images_bin
    n_frames = len(read_images_bin(space_model_dir(space) / "images.bin"))
    started = time.time()
    # Depth and normal maps from Spirula's own MoGe (normals always; depth only
    # when it is used). A map already on disk is kept.
    geometry = [str(binary), "geometry", str(data), "--max-size", str(GEOMETRY_MAX_SIZE)]
    if depth_weight > 0:
        geometry.append("--depth")
    if run(geometry, log_path) != 0:
        raise SystemExit("spirula geometry failed; see " + str(log_path))
    geometry_seconds = round(time.time() - started)
    run_dir = data / "runs"
    shutil.rmtree(run_dir, ignore_errors=True)
    command = [str(binary), "train", "3dgs", "--data", str(data),
               "--output-dir-prefix", str(run_dir), "--output-dir-name", "run",
               "--num-iterations", str(iters), "--train-resolution-divisor", str(divisor),
               "--cap-max", str(cap), "--depth-supervision-weight", str(depth_weight),
               "--floater-suppression", floaters]
    started_train = time.time()
    code = run(command, log_path, done_file=run_dir / "run" / f"step-{iters:09d}.ckpt" / "splat.ply")
    produced = sorted((run_dir / "run").glob("step-*.ckpt/splat.ply"))
    if code != 0 and not produced:
        raise SystemExit(f"spirula train failed (exit {code}); see {log_path}")
    latest = produced[-1]
    shutil.copyfile(latest, out)
    # Checkpoints are 100 MB+ each; only the result is kept.
    shutil.rmtree(run_dir, ignore_errors=True)
    record = {"trainer": "spirula", "binary": str(binary), "frames": n_frames, "iters": iters,
              "divisor": divisor, "cap": cap, "depth_weight": depth_weight, "floaters": floaters,
              "geometry_seconds": geometry_seconds, "train_seconds": round(time.time() - started_train),
              "checkpoint": latest.parent.name, "result": str(out)}
    out.with_suffix(".json").write_text(json.dumps(record, indent=1) + "\n")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("space")
    parser.add_argument("--out", default="splat-spirula.ply")
    parser.add_argument("--iters", type=int, default=DEFAULTS["iters"])
    parser.add_argument("--divisor", type=int, default=DEFAULTS["divisor"])
    parser.add_argument("--cap", type=int, default=DEFAULTS["cap"])
    parser.add_argument("--depth-weight", type=float, default=DEFAULTS["depth_weight"])
    parser.add_argument("--floaters", choices=["off", "mild", "strong"], default=DEFAULTS["floaters"])
    args = parser.parse_args()
    space = Path(args.space).resolve()
    record = train(space, space / args.out, args.iters, args.divisor, args.cap, args.depth_weight, args.floaters)
    print(f"wrote {record['result']} ({record['train_seconds']}s training, "
          f"{record['geometry_seconds']}s geometry)")


if __name__ == "__main__":
    main()
