#!/usr/bin/env python3
"""A splat predicted in one pass by AnySplat, aligned to the camera solve and
polished: a candidate for stage 4 beside the trained splats.

AnySplat (github.com/InternRobotics/AnySplat; weights lhjiang/anysplat) is a
network trained once on thousands of scenes. Given frames and nothing else,
it predicts every frame's camera and a splat of the scene in one pass, in
seconds on a GPU, at 448 pixels a frame. That is a head start, not a finished
splat: a short OpenSplat run on the full-size frames polishes it.

  1. Views: VIEWS frames spread evenly along the walk, from the frames the
     solve placed (tools/anysplat_infer.py runs AnySplat on them, in its own
     environment: ANYSPLAT_PYTHON, ANYSPLAT_DIR).
  2. Alignment: AnySplat's frame has its own origin, orientation and scale.
     Its camera for each view is compared with the solve's for the same frame:
     the rotation that best turns its cameras into the solve's, then the scale
     and offset that best carry its camera centres onto the solve's, refitted
     without the worst fifth. A misfit above MAX_ALIGN_M (median camera centre)
     or MAX_ALIGN_DEG (median camera direction) means the two disagree about
     the room, and the prediction is not used.
  3. Conversion to the pipeline's 62-property splat: centres and shapes
     carried through the alignment (a shape's covariance turns and scales
     with it; its axes and sizes are read back from the covariance, which is
     what AnySplat renders with), opacity stored before the sigmoid as
     splat files store it (AnySplat keeps it after), colour's base band kept.
     The view-dependent colour bands are dropped: turning them needs their
     own rotation, and the polish learns them again.
  4. Polish: POLISH_STEPS of OpenSplat from the converted splat, on the
     solve's cameras and full frames at 1/DOWNSCALE (the same project the
     quick splat trains on).

Writes <space>/splat-anysplat.ply (the candidate), splat-anysplat-ff.ply (the
prediction before the polish), anysplat/raw.npz, and splat-anysplat.json
(views, alignment misfit, counts, seconds).

Licences: AnySplat's repository and weights are MIT, but its network is
fine-tuned from VGGT-1B (CC BY-NC 4.0) and some of its helper code comes from
Naver's CroCo/DUSt3R (CC BY-NC-SA 4.0): a trial, not something to ship.

Usage:
    python3 pipeline/splat_anysplat.py spaces/<name> [--views 24] [--polish-steps 2000] [--downscale 4]
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

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(ROOT / "tools"))
from pointcloud import space_model_dir  # noqa: E402

DEFAULTS = {"views": 24, "polish_steps": 2000, "downscale": 4, "shape": "square"}
MAX_ALIGN_M = 0.25        # median distance between a view's predicted and solved camera centre
MAX_ALIGN_DEG = 10.0      # median angle between a view's predicted and solved camera direction
KEEP_SHARE = 0.8          # the alignment is refitted on this share of the views it fits best


def find_install() -> tuple[Path, str] | None:
    """AnySplat's clone and the Python of its environment, or None."""
    for candidate in (os.environ.get("ANYSPLAT_DIR"), ROOT / "tools" / "anysplat", "/content/AnySplat"):
        if candidate and (Path(candidate) / "src" / "model" / "model" / "anysplat.py").exists():
            return Path(candidate), os.environ.get("ANYSPLAT_PYTHON") or sys.executable
    return None


def choose_views(names: list[str], count: int) -> list[str]:
    """`count` frames spread evenly along the walk (frame names sort in time)."""
    names = sorted(names)
    if len(names) <= count:
        return names
    picks = np.linspace(0, len(names) - 1, count).round().astype(int)
    return [names[i] for i in sorted(set(picks))]


def solved_cam2world(images: dict) -> dict[str, np.ndarray]:
    """Each placed frame's camera-to-world 4x4, from COLMAP's world-to-camera."""
    out = {}
    for im in images.values():
        m = np.eye(4)
        m[:3, :3] = im["R"].T
        m[:3, 3] = -im["R"].T @ im["t"]
        out[im["name"]] = m
    return out


def rotation_angle(R: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0))))


def similarity(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, dict]:
    """The scale s, rotation R and offset t with dst ≈ s·R·src + t, from two
    sets of cameras (V, 4, 4, camera-to-world) of the same frames. The
    rotation comes from the cameras' own rotations (a walk along a line does
    not pin it down from centres alone), the scale and offset from the centres,
    refitted on the KEEP_SHARE that fit best. Misfits are in dst's units."""
    Ra, Rd = src[:, :3, :3], dst[:, :3, :3]
    U, _, Vt = np.linalg.svd(np.einsum("vij,vkj->ik", Rd, Ra))
    R = U @ np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))]) @ Vt
    ca, cd = src[:, :3, 3] @ R.T, dst[:, :3, 3]

    def fit(keep):
        ma, md = ca[keep].mean(0), cd[keep].mean(0)
        spread = ((ca[keep] - ma) ** 2).sum()
        s = float(((ca[keep] - ma) * (cd[keep] - md)).sum() / spread) if spread > 0 else 1.0
        return s, md - s * ma

    keep = np.ones(len(ca), bool)
    s, offset = fit(keep)
    if len(ca) >= 5:
        misfit = np.linalg.norm(s * ca + offset - cd, axis=1)
        keep = misfit <= np.quantile(misfit, KEEP_SHARE)
        s, offset = fit(keep)
    misfit = np.linalg.norm(s * ca + offset - cd, axis=1)
    turn = [rotation_angle(Rd[v] @ (R @ Ra[v]).T) for v in range(len(Ra))]
    return s, R, offset, {"scale": s, "centre_misfit_median": float(np.median(misfit)),
                          "centre_misfit_max": float(misfit.max()),
                          "direction_misfit_median_deg": float(np.median(turn)),
                          "views_kept": int(keep.sum()), "views": len(ca)}


def quaternions(R: np.ndarray) -> np.ndarray:
    """(N, 3, 3) rotation matrices to (N, 4) unit quaternions, w x y z."""
    tr = np.trace(R, axis1=1, axis2=2)
    q = np.empty((len(R), 4))
    cases = np.stack([tr, R[:, 0, 0], R[:, 1, 1], R[:, 2, 2]], axis=1).argmax(axis=1)
    for case in range(4):
        m = R[cases == case]
        if not len(m):
            continue
        if case == 0:
            s = np.sqrt(1 + np.trace(m, axis1=1, axis2=2)) * 2
            v = [s / 4, (m[:, 2, 1] - m[:, 1, 2]) / s, (m[:, 0, 2] - m[:, 2, 0]) / s, (m[:, 1, 0] - m[:, 0, 1]) / s]
        else:
            i = case - 1
            j, k = (i + 1) % 3, (i + 2) % 3
            s = np.sqrt(1 + m[:, i, i] - m[:, j, j] - m[:, k, k]) * 2
            v = [None] * 4
            v[0] = (m[:, k, j] - m[:, j, k]) / s
            v[1 + i] = s / 4
            v[1 + j] = (m[:, j, i] + m[:, i, j]) / s
            v[1 + k] = (m[:, k, i] + m[:, i, k]) / s
        q[cases == case] = np.stack(v, axis=1)
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def to_splat(raw: dict, s: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """AnySplat's prediction carried into the solve's frame, as the pipeline's
    62-property splat (tools/splat_tools.SPLAT_DTYPE)."""
    from splat_tools import SPLAT_DTYPE

    means = s * raw["means"].astype(np.float64) @ R.T + t
    cov = (s * s) * np.einsum("ij,njk,lk->nil", R, raw["covariances"].astype(np.float64), R)
    cov = (cov + cov.transpose(0, 2, 1)) / 2
    values, axes = np.linalg.eigh(cov)
    axes[np.linalg.det(axes) < 0, :, 2] *= -1          # a rotation, not a reflection
    scales = np.sqrt(np.clip(values, 1e-18, None))
    opacity = np.clip(raw["opacities"].astype(np.float64), 1e-4, 1 - 1e-4)
    out = np.zeros(len(means), dtype=SPLAT_DTYPE)
    for k, axis in enumerate("xyz"):
        out[axis] = means[:, k]
    for k in range(3):
        out[f"f_dc_{k}"] = raw["dc"][:, k]
        out[f"scale_{k}"] = np.log(scales[:, k])
    out["opacity"] = np.log(opacity / (1 - opacity))
    q = quaternions(axes)
    for k in range(4):
        out[f"rot_{k}"] = q[:, k]
    return out


def predict(space: Path, frames: list[str], raw_path: Path, shape: str) -> dict:
    install = find_install()
    if install is None:
        raise SystemExit("AnySplat not found: set ANYSPLAT_DIR (a clone of InternRobotics/AnySplat) "
                         "and ANYSPLAT_PYTHON (its environment's python)")
    directory, python = install
    images = space / "workspace" / "images"
    command = [python, str(ROOT / "tools" / "anysplat_infer.py"), "--anysplat-dir", str(directory),
               "--out", str(raw_path), "--shape", shape, *[str(images / f) for f in frames]]
    print("   $ " + " ".join(command[:6]) + f" ... ({len(frames)} frames)", flush=True)
    done = subprocess.run(command, capture_output=True, text=True)
    (space / "anysplat" / "infer.log").write_text(done.stdout + done.stderr)
    if done.returncode != 0 or not raw_path.exists():
        raise SystemExit("AnySplat failed; see " + str(space / "anysplat" / "infer.log") + "\n"
                         + (done.stderr or done.stdout)[-1500:])
    return json.loads(raw_path.with_suffix(".json").read_text())


def polish(space: Path, start: Path, out: Path, steps: int, downscale: int) -> bool:
    """OpenSplat from the converted splat, on the project the quick splat trains on."""
    binary = os.environ.get("OPENSPLAT") or str(ROOT / "tools" / "opensplat")
    project = space / "splat-project"
    if not Path(binary).exists() or not project.exists():
        print(f"   no polish: {'OpenSplat' if not Path(binary).exists() else 'splat-project'} missing", flush=True)
        return False
    command = [binary, str(project), "-n", str(steps), "-d", str(downscale), "--resume", str(start),
               "-o", str(out)]
    print("   $ " + " ".join(command), flush=True)
    with open(space / "anysplat" / "polish.log", "w") as log:
        code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode
    return code == 0 and out.exists()


def main() -> None:
    from densify import read_images_bin
    from splat_tools import write_splat

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("space", type=Path)
    parser.add_argument("--out", default="splat-anysplat.ply")
    parser.add_argument("--views", type=int, default=DEFAULTS["views"])
    parser.add_argument("--polish-steps", type=int, default=DEFAULTS["polish_steps"])
    parser.add_argument("--downscale", type=int, default=DEFAULTS["downscale"])
    parser.add_argument("--shape", choices=["square", "full"], default=DEFAULTS["shape"])
    parser.add_argument("--reuse-prediction", action="store_true",
                        help="use anysplat/raw.npz from an earlier run instead of predicting again")
    args = parser.parse_args()
    space = args.space.resolve()
    model = space_model_dir(space)
    if model is None:
        raise SystemExit(f"no camera model in {space / 'workspace' / 'sparse'}")
    images = read_images_bin(model / "images.bin")
    solved = solved_cam2world(images)
    (space / "anysplat").mkdir(exist_ok=True)
    raw_path = space / "anysplat" / "raw.npz"
    started = time.time()
    if args.reuse_prediction and raw_path.exists():
        info = json.loads(raw_path.with_suffix(".json").read_text())
    else:
        info = predict(space, choose_views(list(solved), args.views), raw_path, args.shape)
    raw = dict(np.load(raw_path))
    frames = [str(f) for f in raw["frames"]]

    s, R, t, fit = similarity(raw["cam2world"], np.stack([solved[f] for f in frames]))
    units = None
    try:
        units = json.loads((space / "densify.json").read_text()).get("colmap_units_per_metre")
    except (OSError, ValueError):
        pass
    misfit_m = fit["centre_misfit_median"] / units if units else None
    record = {"trainer": "anysplat", "prediction": info, "frames": frames, "alignment": fit,
              "centre_misfit_median_m": None if misfit_m is None else round(misfit_m, 3)}
    print(f"   aligned {len(frames)} views: median camera misfit "
          + (f"{misfit_m:.2f} m" if misfit_m is not None else f"{fit['centre_misfit_median']:.3f} units")
          + f", {fit['direction_misfit_median_deg']:.1f} deg", flush=True)
    bad = ((misfit_m is not None and misfit_m > MAX_ALIGN_M)
           or fit["direction_misfit_median_deg"] > MAX_ALIGN_DEG)
    if bad:
        record["result"] = None
        (space / "splat-anysplat.json").write_text(json.dumps(record, indent=1) + "\n")
        raise SystemExit(f"AnySplat's cameras disagree with the solve (median "
                         f"{misfit_m if misfit_m is not None else fit['centre_misfit_median']:.2f}"
                         f"{' m' if misfit_m is not None else ' units'}, "
                         f"{fit['direction_misfit_median_deg']:.1f} deg): its splat is not used")

    splat = to_splat(raw, s, R, t)
    ff = space / "splat-anysplat-ff.ply"
    write_splat(ff, splat, comment="Generated by opensplat at iteration 0")
    out = space / args.out
    polished = args.polish_steps > 0 and polish(space, ff, out, args.polish_steps, args.downscale)
    if not polished:
        shutil.copyfile(ff, out)
    record.update({"gaussians": len(splat), "polish_steps": args.polish_steps if polished else 0,
                   "downscale": args.downscale, "seconds": round(time.time() - started), "result": str(out)})
    (space / "splat-anysplat.json").write_text(json.dumps(record, indent=1) + "\n")
    print(f"wrote {out} ({len(splat):,} gaussians predicted"
          + (f", polished {args.polish_steps} steps" if polished else ", not polished") + ")")


if __name__ == "__main__":
    main()
