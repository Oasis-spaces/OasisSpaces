#!/usr/bin/env python3
"""Choose the best splat of a video, with Claude as the judge.

The same video can end up with several splats: runs on different machines
(the walkthrough's Mac run had clean curtains, its Colab run blotchy ones),
short and long training, and each with and without the surface fill. Nothing
used to compare them, so a worse result could pass. This renders every
candidate from the camera positions of the same video frames (each run uses
its own camera solve for that frame), puts the renders beside the real frame,
and:

  - measures how closely each render matches the photo (PSNR, and SSIM,
    which punishes blur and blotches more than colour shifts);
  - asks Claude to rank the candidates against the photos, looking for what
    a person would notice: blotches and streaks, holes, haze, smears, wrong
    colours, painted-over or missing things, softness.

Two kinds of views are compared. Trained views are frames the splats were
built from, where every splat can look good by copying the photo. New views
are real frames of the video between two trained frames, which no candidate
has seen: the camera is placed between its neighbours' positions, so they show
how a splat holds up from a new place, which is what a viewer does. The grid is
the frames reconstruct.py picked its training frames from (3x 2 fps), which it
keeps in workspace/video-grid/ with the step each kept frame came from (older
spaces: the video is decoded once and each trained frame is found on the grid
by its picture). A new view is the grid frame halfway between two trained ones.

Claude's pick wins; the measurements are recorded next to it and decide on
their own when Claude is not available (new views first). The chosen splat is copied to
splats/<video>/best.splat (with its .ply and starting camera), every
candidate's viewer file to splats/<video>/candidates/, and the comparison
sheets and choice.json beside them.

Usage:
    python3 tools/splat_choose.py ~/Downloads/IMG_4182.MOV
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(ROOT / "tools"))
from densify import read_cameras_bin, read_images_bin  # noqa: E402
from splat_export import model_dir  # noqa: E402

VIEWS = 4
NEW_VIEWS = 4
TILE_W, TILE_H = 216, 384          # the frames are portrait phone video
LETTERS = "ABCDEFGH"
# reconstruct.py extracts at 3x its 2 fps rate and keeps the sharpest frame of
# every three, so trained frames sit on this grid of the video.
GRID_FPS = 6
THUMB_POOL = 8                     # tiles are pooled 8x8 to match frames by picture
# A new view needs trained frames on both sides within this many grid steps
# (half a second at 2 fps), or the camera between them is a guess.
MAX_NEIGHBOUR_STEPS = 4


def candidates_for(video: Path) -> list[dict]:
    """Every finished splat whose agent report names this video: the trained
    splat, and the filled one when its fill was kept."""
    found = []
    for report in sorted((ROOT / "spaces").glob("*/agent-report.json")):
        try:
            source = json.loads(report.read_text()).get("source")
        except ValueError:
            continue
        if not source or Path(source).name != video.name:
            continue
        space = report.parent
        if model_dir(space) is None:
            continue
        if (space / "splat.ply").exists():
            found.append({"label": f"{space.name} trained", "space": space, "ply": space / "splat.ply"})
        fill = space / "splat-filled.fill.json"
        if (space / "splat-filled.ply").exists() and fill.exists() \
                and json.loads(fill.read_text()).get("kept"):
            found.append({"label": f"{space.name} filled", "space": space,
                          "ply": space / "splat-filled.ply"})
    return found


def camera_poses(space: Path) -> dict:
    """frame name -> (R, t, vertical field of view in degrees), world to camera."""
    model = model_dir(space)
    cameras = read_cameras_bin(model / "cameras.bin")
    out = {}
    for info in read_images_bin(model / "images.bin").values():
        cam = cameras[info["camera_id"]]
        fov_y = float(np.degrees(2 * np.arctan(cam["height"] / 2 / cam["params"][1])))
        out[info["name"]] = (np.asarray(info["R"], float), np.asarray(info["t"], float), fov_y)
    return out


def view_matrix(R: np.ndarray, t: np.ndarray) -> list[float]:
    """splat-viewer's column-major world-to-camera matrix."""
    return [float(v) for v in (R[0][0], R[1][0], R[2][0], 0, R[0][1], R[1][1], R[2][1], 0,
                               R[0][2], R[1][2], R[2][2], 0, t[0], t[1], t[2], 1)]


def camera_views(space: Path) -> dict:
    """frame name -> (viewer view matrix, vertical field of view in degrees)."""
    return {name: (view_matrix(R, t), fov) for name, (R, t, fov) in camera_poses(space).items()}


def between(pose_a, pose_b, f: float):
    """The camera a share f of the way from pose a to pose b: its centre on
    the straight line, its rotation turned evenly (slerp)."""
    from scipy.spatial.transform import Rotation, Slerp

    (Ra, ta, fov), (Rb, tb, _) = pose_a, pose_b
    centre = (1 - f) * (-Ra.T @ ta) + f * (-Rb.T @ tb)
    R = Slerp([0.0, 1.0], Rotation.from_matrix(np.stack([Ra, Rb])))([f]).as_matrix()[0]
    return view_matrix(R, -R @ centre), fov


def video_grid(video: Path, spaces: list[Path]) -> np.ndarray:
    """Every grid frame of the video as a TILE_W x TILE_H RGB tile. Read from
    a space's workspace/video-grid/ (reconstruct.py keeps one); otherwise the
    video is decoded once and the tiles are cached in the first space."""
    for space in spaces:
        tiles = sorted((space / "workspace" / "video-grid").glob("step_*.jpg"))
        if tiles:
            return np.stack([np.asarray(Image.open(t).convert("RGB").resize((TILE_W, TILE_H), Image.BILINEAR))
                             for t in tiles])
    hwaccel = ["-hwaccel", "videotoolbox"] if sys.platform == "darwin" else []
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", *hwaccel, "-i", str(video),
         "-vf", f"fps={GRID_FPS},scale={TILE_W}:{TILE_H}", "-pix_fmt", "rgb24",
         "-f", "rawvideo", "-"], capture_output=True, check=True).stdout
    grid = np.frombuffer(raw, np.uint8).reshape(-1, TILE_H, TILE_W, 3)
    grid_dir = spaces[0] / "workspace" / "video-grid"
    grid_dir.mkdir(parents=True, exist_ok=True)
    for step, tile in enumerate(grid):
        Image.fromarray(tile).save(grid_dir / f"step_{step:05d}.jpg", quality=92)
    return grid


def recorded_steps(space: Path) -> dict | None:
    """frame name -> grid step, as reconstruct.py recorded it, if it did."""
    record = space / "workspace" / "video-grid" / "frames.json"
    if not record.exists():
        return None
    data = json.loads(record.read_text())
    return data["frames"] if data.get("fps") == GRID_FPS else None


def pooled(tiles: np.ndarray) -> np.ndarray:
    """Greyscale thumbnails, THUMB_POOL x THUMB_POOL blocks averaged."""
    grey = tiles.astype(np.float32).mean(-1)
    h, w = TILE_H // THUMB_POOL, TILE_W // THUMB_POOL
    return grey[..., :h * THUMB_POOL, :w * THUMB_POOL].reshape(
        *grey.shape[:-2], h, THUMB_POOL, w, THUMB_POOL).mean((-3, -1))


def grid_steps(space: Path, names, grid_thumbs: np.ndarray) -> dict[str, int]:
    """frame name -> its step on the video grid, found by matching pictures."""
    steps = {}
    for name in names:
        img = Image.open(space / "workspace" / "images" / name)
        img.draft("RGB", (TILE_W * 2, TILE_H * 2))
        tile = np.asarray(img.convert("RGB").resize((TILE_W, TILE_H), Image.BILINEAR))
        steps[name] = int(np.argmin(((grid_thumbs - pooled(tile)) ** 2).mean((1, 2))))
    return steps


def new_view_shots(video: Path, candidates: list[dict], poses: dict, log=print) -> list[dict]:
    """Up to NEW_VIEWS real video frames that no candidate was trained on, each
    with every candidate's camera placed between its two nearest trained frames."""
    grid = video_grid(video, [c["space"] for c in candidates])
    thumbs = pooled(grid)
    steps = {}
    for c in candidates:
        known = recorded_steps(c["space"])
        if known and set(poses[c["label"]]) <= set(known):
            steps[c["label"]] = {name: known[name] for name in poses[c["label"]]}
        else:
            steps[c["label"]] = grid_steps(c["space"], poses[c["label"]], thumbs)
    trained = set().union(*(set(v.values()) for v in steps.values()))
    best_in_gap = {}
    for k in range(len(grid)):
        if k in trained:
            continue
        cams, gap = {}, None
        for c in candidates:
            by_step = steps[c["label"]]
            before = max(((s, n) for n, s in by_step.items() if s < k), default=None)
            after = min(((s, n) for n, s in by_step.items() if s > k), default=None)
            if before is None or after is None or after[0] - before[0] > MAX_NEIGHBOUR_STEPS:
                break
            f = (k - before[0]) / (after[0] - before[0])
            cams[c["label"]] = between(poses[c["label"]][before[1]], poses[c["label"]][after[1]], f)
            gap = gap or (before[0], abs(f - 0.5))
        else:
            # One view per gap between trained frames: the one nearest halfway,
            # where the camera is furthest from anything a splat was fitted to.
            if gap[0] not in best_in_gap or gap[1] < best_in_gap[gap[0]][0]:
                best_in_gap[gap[0]] = (gap[1], k, cams)
    options = [(k, cams) for _, (_, k, cams) in sorted(best_in_gap.items())]
    if not options:
        log("  no new views: the trained frames are too far apart or cover every grid frame")
        return []
    spread = np.linspace(len(options) * 0.1, len(options) * 0.9 - 1, min(NEW_VIEWS, len(options)))
    picks = [options[i] for i in sorted({int(round(v)) for v in spread})]
    return [{"key": f"video {k / GRID_FPS:.2f}s", "photo": grid[k], "new": True, "cams": cams}
            for k, cams in picks]


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Mean structural similarity of two greyscale images in 0-255."""
    from scipy.ndimage import gaussian_filter

    a, b = a.astype(np.float64), b.astype(np.float64)
    mu_a, mu_b = gaussian_filter(a, 1.5), gaussian_filter(b, 1.5)
    var_a = gaussian_filter(a * a, 1.5) - mu_a ** 2
    var_b = gaussian_filter(b * b, 1.5) - mu_b ** 2
    cov = gaussian_filter(a * b, 1.5) - mu_a * mu_b
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    return float(np.mean(((2 * mu_a * mu_b + c1) * (2 * cov + c2))
                         / ((mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2))))


def compare(video: Path, candidates: list[dict], out_dir: Path, log=print):
    """Render, measure and draw sheets; returns (measurements, sheet paths,
    the views compared)."""
    from splat_render import render
    from splat_tools import read_splat

    poses = {c["label"]: camera_poses(c["space"]) for c in candidates}
    common = sorted(set.intersection(*(set(v) for v in poses.values())))
    if not common:
        raise SystemExit("the candidates share no video frames to compare from")
    picks = [common[round(i)] for i in np.linspace(len(common) * 0.1, len(common) * 0.9 - 1, VIEWS)]
    reference_space = candidates[0]["space"]
    shots = [{"key": name, "new": False,
              "photo": np.asarray(Image.open(reference_space / "workspace" / "images" / name)
                                  .convert("RGB").resize((TILE_W, TILE_H), Image.LANCZOS)),
              "cams": {label: (view_matrix(R, t), fov)
                       for label, (R, t, fov) in ((lab, poses[lab][name]) for lab in poses)}}
             for name in picks]
    if video.exists():
        try:
            shots += new_view_shots(video, candidates, poses, log)
        except (OSError, subprocess.CalledProcessError, ValueError) as exc:
            log(f"  no new views ({type(exc).__name__}: {exc})")
    renders = {}
    measured = {c["label"]: {kind: {"psnr": [], "ssim": [], "gaps": []} for kind in ("trained", "new")}
                for c in candidates}
    for c in candidates:
        arr, _ = read_splat(c["ply"])
        for n, shot in enumerate(shots):
            view, fov = shot["cams"][c["label"]]
            image, coverage = render(arr, view, TILE_W, TILE_H, fov, with_coverage=True)
            renders[(c["label"], n)] = image
            real = shot["photo"].astype(np.float64)
            mse = np.mean((image.astype(np.float64) - real) ** 2)
            m = measured[c["label"]]["new" if shot["new"] else "trained"]
            m["psnr"].append(10 * np.log10(255 ** 2 / max(mse, 1e-9)))
            m["ssim"].append(ssim(image.mean(-1), real.mean(-1)))
            m["gaps"].append(float((coverage < 0.5).mean()))
        log(f"  rendered {c['label']}")
    summary = {}
    for label, kinds in measured.items():
        row = {k: round(float(np.mean(v)), 3) for k, v in kinds["trained"].items()}
        if kinds["new"]["psnr"]:
            row.update({f"new_{k}": round(float(np.mean(v)), 3) for k, v in kinds["new"].items()})
        summary[label] = row

    font = ImageFont.load_default(size=16)
    sheets = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for n, shot in enumerate(shots):
        sheet = Image.new("RGB", ((len(candidates) + 1) * (TILE_W + 8), TILE_H + 26), "white")
        draw = ImageDraw.Draw(sheet)
        sheet.paste(Image.fromarray(shot["photo"]), (0, 26))
        draw.text((4, 4), "NEW VIEW (not trained on)" if shot["new"] else "trained view",
                  fill="black", font=font)
        for k, c in enumerate(candidates, 1):
            sheet.paste(Image.fromarray(renders[(c["label"], n)]), (k * (TILE_W + 8), 26))
            draw.text((k * (TILE_W + 8) + 4, 4), LETTERS[k - 1], fill="black", font=font)
        path = out_dir / f"compare-{n + 1}.png"
        sheet.save(path)
        sheets.append(path)
    return summary, sheets, [{"view": shot["key"], "new": shot["new"]} for shot in shots]


PROMPT = """You are the final quality judge for 3D room reconstructions (Gaussian splats) made from one phone video.
Each image shows one moment of the video: the real video frame on the far left, then renders of {count} candidate
reconstructions (labelled {letters}) from that camera position. A good candidate looks like the photo.

There are two kinds of images, named above the photo:
- "trained view": the candidates were built from this frame, so each can look good here by copying the photo.
- "NEW VIEW (not trained on)": a real video frame none of the candidates was built from, rendered from a camera placed
  between the frames around it. This is what a person walking through the room sees, so weigh new views more heavily.
  The camera is interpolated, so a small shift in framing against the photo is expected; judge the reconstruction.

Compare the candidates against the photo and against each other, and judge what a person walking through the room would notice:
- blotches, streaks or smears on surfaces that are plain in the photo (curtains, walls, bedding)
- black holes or see-through gaps
- haze, fog or floating specks in front of things that are sharp in the photo
- wrong or shifted colours
- things painted over, missing, doubled or melted
- overall softness where the photo is crisp
Small differences in brightness between a render and the photo are expected; judge the reconstruction, not the exposure.

Rank all candidates from best to worst over all views. Only call two candidates equal if you truly cannot tell them apart.
Reply with a single JSON object and nothing else:
{{"ranking": [letters best first], "best": letter, "reasons": {{letter: one sentence on what is good or bad about it}}, "confidence": "high" | "medium" | "low"}}"""


# Claude's pick stands unless it measures clearly worse at the new views than
# another candidate that is up for choice: lower on both of these at once.
GUARD_NEW_SSIM = 0.01
GUARD_NEW_PSNR = 0.3


def published_best(video: Path) -> dict | None:
    """The splat splats/<video>/ currently publishes as the video's best, as a
    candidate ({label, space, ply}); None before the first choice."""
    folders = sorted((ROOT / "splats").glob(f"{video.stem}*")) if (ROOT / "splats").exists() else []
    record = folders[0] / "choice.json" if folders else None
    if not record or not record.exists():
        return None
    choice = json.loads(record.read_text())
    best = next((c for c in choice.get("candidates", {}).values() if c.get("label") == choice.get("best")), None)
    if not best or not Path(best["splat"]).exists():
        return None
    ply = Path(best["splat"])
    space = ply.parent
    return {"label": f"published best ({best['label']})", "space": space, "ply": ply}


def guarded(record: dict, choosable: set[str]) -> dict:
    """Keep Claude's pick among the `choosable` labels unless another choosable
    candidate beats it at the new views by GUARD_NEW_SSIM and GUARD_NEW_PSNR
    both. Sets record["chosen"] (a choosable label) and, when the numbers
    overrule, record["overruled"]."""
    rows = {c["label"]: c for c in record["candidates"].values()}
    ranking = [record["candidates"][L]["label"] for L in ((record.get("claude") or {}).get("ranking") or [])
               if L in record["candidates"]]
    by_claude = next((label for label in ranking if label in choosable), None)
    new_view = lambda label: (rows[label].get("new_ssim", rows[label]["ssim"]),
                              rows[label].get("new_psnr", rows[label]["psnr"]))
    by_numbers = max(choosable, key=new_view)
    chosen = by_claude or by_numbers
    if by_claude and by_claude != by_numbers:
        (s_pick, p_pick), (s_num, p_num) = new_view(by_claude), new_view(by_numbers)
        if s_num - s_pick > GUARD_NEW_SSIM and p_num - p_pick > GUARD_NEW_PSNR:
            chosen = by_numbers
            record["overruled"] = (f"Claude picked {by_claude}, but at the new views it measures "
                                   f"SSIM {s_pick:.3f} / {p_pick:.1f} dB against {s_num:.3f} / {p_num:.1f} dB "
                                   f"for {by_numbers}; kept {by_numbers}")
    record["chosen"] = chosen
    return record


def judge(video: Path, candidates: list[dict], out_dir: Path, log=print, advisor=None) -> dict:
    """Compare `candidates` ({label, space, ply}) with sheets in `out_dir`;
    Claude ranks them, the measurements decide without Claude. Returns the
    record: candidates by letter with their measurements, Claude's verdict,
    and the best label."""
    summary, sheets, views = compare(video, candidates, out_dir, log)
    letters = {LETTERS[k]: c["label"] for k, c in enumerate(candidates)}
    by_numbers = max(summary, key=lambda label: (summary[label].get("new_ssim", summary[label]["ssim"]),
                                                 summary[label]["ssim"]))
    verdict = None
    if len(candidates) > 1:
        if advisor is None:
            from advisor import Advisor

            advisor = Advisor()
        if advisor.available:
            verdict = advisor.ask_json(PROMPT.format(count=len(candidates),
                                                     letters=", ".join(letters)), sheets,
                                       max_tokens=2000)
    decided = bool(verdict and verdict.get("best") in letters)
    return {
        "video": str(video), "views_compared": views,
        "candidates": {letter: {"label": label, "splat": str(next(c["ply"] for c in candidates
                                                                  if c["label"] == label)),
                                **summary[label]}
                       for letter, label in letters.items()},
        "claude": verdict, "best_by_numbers": by_numbers,
        "best": letters[verdict["best"]] if decided else by_numbers,
        "decided_by": "claude" if decided else "numbers",
    }


def choose(video: Path, log=print, advisor=None) -> dict | None:
    """Compare every candidate splat of `video`, publish the best to
    splats/<video>/ and return the choice record."""
    candidates = candidates_for(video)
    if not candidates:
        log(f"no finished splats for {video.name}")
        return None
    existing = sorted((ROOT / "splats").glob(f"{video.stem}*")) if (ROOT / "splats").exists() else []
    out_dir = existing[0] if existing and existing[0].is_dir() else ROOT / "splats" / video.stem
    record = judge(video, candidates, out_dir, log, advisor)
    best = next(c for c in candidates if c["label"] == record["best"])

    # Publish: the best splat, every candidate's viewer file, the evidence.
    (out_dir / "candidates").mkdir(parents=True, exist_ok=True)
    for c in candidates:
        stem = c["label"].replace(" ", "-")
        for ext in (".splat", ".view.json"):
            src = c["ply"].with_suffix(ext) if ext == ".splat" else c["ply"].with_name(c["ply"].stem + ext)
            if src.exists():
                shutil.copyfile(src, out_dir / "candidates" / f"{stem}{ext}")
    shutil.copyfile(best["ply"], out_dir / "best.ply")
    if best["ply"].with_suffix(".splat").exists():
        shutil.copyfile(best["ply"].with_suffix(".splat"), out_dir / "best.splat")
    else:
        from splat_export import ply_to_splat

        ply_to_splat(best["ply"], out_dir / "best.splat")
    view = best["ply"].with_name(best["ply"].stem + ".view.json")
    if view.exists():
        shutil.copyfile(view, out_dir / "best.view.json")
    (out_dir / "choice.json").write_text(json.dumps(record, indent=1) + "\n")
    log(f"  best for {video.name}: {record['best']} (decided by {record['decided_by']}) "
        f"-> {out_dir / 'best.splat'}")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video")
    args = parser.parse_args()
    record = choose(Path(args.video).expanduser())
    if record:
        for letter, c in record["candidates"].items():
            new = (f"; new views SSIM {c['new_ssim']:.3f}, PSNR {c['new_psnr']:.1f} dB, gaps {c['new_gaps']:.1%}"
                   if "new_ssim" in c else "")
            print(f"  {letter} {c['label']}: trained views SSIM {c['ssim']:.3f}, PSNR {c['psnr']:.1f} dB, "
                  f"gaps {c['gaps']:.1%}{new}")
        if record["claude"]:
            print("  Claude ranking:", record["claude"].get("ranking"), "-", record["claude"].get("confidence"))
            for letter, why in (record["claude"].get("reasons") or {}).items():
                print(f"    {letter}: {why}")


if __name__ == "__main__":
    main()
