#!/usr/bin/env python3
"""Choose the best splat of a video, with Claude as the judge.

The same video can end up with several splats: runs on different machines
(the walkthrough's Mac run had clean curtains, its Colab run blotchy ones),
and each run's trained splat with and without the surface fill. Nothing used
to compare them, so a worse result could pass. This renders every candidate
from the camera positions of the same video frames (each run uses its own
camera solve for that frame), puts the renders beside the real frame, and:

  - measures how closely each render matches the photo (PSNR, and SSIM,
    which punishes blur and blotches more than colour shifts);
  - asks Claude to rank the candidates against the photos, looking for what
    a person would notice: blotches and streaks, holes, haze, smears, wrong
    colours, painted-over or missing things, softness.

Claude's pick wins; the measurements are recorded next to it and decide on
their own when Claude is not available. The chosen splat is copied to
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
TILE_W, TILE_H = 216, 384          # the frames are portrait phone video
LETTERS = "ABCDEFGH"


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


def camera_views(space: Path) -> dict:
    """frame name -> (viewer view matrix, vertical field of view in degrees)."""
    model = model_dir(space)
    cameras = read_cameras_bin(model / "cameras.bin")
    out = {}
    for info in read_images_bin(model / "images.bin").values():
        R, t = info["R"], info["t"]
        cam = cameras[info["camera_id"]]
        fov_y = float(np.degrees(2 * np.arctan(cam["height"] / 2 / cam["params"][1])))
        view = [R[0][0], R[1][0], R[2][0], 0, R[0][1], R[1][1], R[2][1], 0,
                R[0][2], R[1][2], R[2][2], 0, t[0], t[1], t[2], 1]
        out[info["name"]] = ([float(v) for v in view], fov_y)
    return out


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
    frame names)."""
    from splat_render import render
    from splat_tools import read_splat

    views = {c["label"]: camera_views(c["space"]) for c in candidates}
    common = sorted(set.intersection(*(set(v) for v in views.values())))
    if not common:
        raise SystemExit("the candidates share no video frames to compare from")
    picks = [common[round(i)] for i in np.linspace(len(common) * 0.1, len(common) * 0.9 - 1, VIEWS)]
    reference_space = candidates[0]["space"]
    frames = {name: np.asarray(Image.open(reference_space / "workspace" / "images" / name)
                               .convert("RGB").resize((TILE_W, TILE_H), Image.LANCZOS))
              for name in picks}
    renders = {}
    measured = {c["label"]: {"psnr": [], "ssim": [], "gaps": []} for c in candidates}
    for c in candidates:
        arr, _ = read_splat(c["ply"])
        for name in picks:
            view, fov = views[c["label"]][name]
            image, coverage = render(arr, view, TILE_W, TILE_H, fov, with_coverage=True)
            renders[(c["label"], name)] = image
            real = frames[name].astype(np.float64)
            mse = np.mean((image.astype(np.float64) - real) ** 2)
            measured[c["label"]]["psnr"].append(10 * np.log10(255 ** 2 / max(mse, 1e-9)))
            measured[c["label"]]["ssim"].append(ssim(image.mean(-1), real.mean(-1)))
            measured[c["label"]]["gaps"].append(float((coverage < 0.5).mean()))
        log(f"  rendered {c['label']}")
    summary = {label: {k: round(float(np.mean(v)), 3) for k, v in m.items()}
               for label, m in measured.items()}

    font = ImageFont.load_default(size=16)
    sheets = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for n, name in enumerate(picks, 1):
        sheet = Image.new("RGB", ((len(candidates) + 1) * (TILE_W + 8), TILE_H + 26), "white")
        draw = ImageDraw.Draw(sheet)
        sheet.paste(Image.fromarray(frames[name]), (0, 26))
        draw.text((4, 4), "video frame", fill="black", font=font)
        for k, c in enumerate(candidates, 1):
            sheet.paste(Image.fromarray(renders[(c["label"], name)]), (k * (TILE_W + 8), 26))
            draw.text((k * (TILE_W + 8) + 4, 4), LETTERS[k - 1], fill="black", font=font)
        path = out_dir / f"compare-{n}.png"
        sheet.save(path)
        sheets.append(path)
    return summary, sheets, picks


PROMPT = """You are the final quality judge for 3D room reconstructions (Gaussian splats) made from one phone video.
Each image shows the same moment of the video: the real video frame on the far left, then renders of {count} candidate
reconstructions (labelled {letters}) from exactly that camera position. A good candidate looks like the photo.

Compare the candidates against the photo and against each other, and judge what a person walking through the room would notice:
- blotches, streaks or smears on surfaces that are plain in the photo (curtains, walls, bedding)
- black holes or see-through gaps
- haze or fog in front of things that are sharp in the photo
- wrong or shifted colours
- things painted over, missing, doubled or melted
- overall softness where the photo is crisp
Small differences in framing or brightness between a render and the photo are expected; judge the reconstruction, not the exposure.

Rank all candidates from best to worst over all views. Only call two candidates equal if you truly cannot tell them apart.
Reply with a single JSON object and nothing else:
{{"ranking": [letters best first], "best": letter, "reasons": {{letter: one sentence on what is good or bad about it}}, "confidence": "high" | "medium" | "low"}}"""


def choose(video: Path, log=print, advisor=None) -> dict | None:
    """Compare every candidate splat of `video`, publish the best to
    splats/<video>/ and return the choice record."""
    candidates = candidates_for(video)
    if not candidates:
        log(f"no finished splats for {video.name}")
        return None
    existing = sorted((ROOT / "splats").glob(f"{video.stem}*")) if (ROOT / "splats").exists() else []
    out_dir = existing[0] if existing and existing[0].is_dir() else ROOT / "splats" / video.stem
    summary, sheets, picks = compare(video, candidates, out_dir, log)
    letters = {LETTERS[k]: c["label"] for k, c in enumerate(candidates)}
    by_numbers = max(summary, key=lambda label: (summary[label]["ssim"], summary[label]["psnr"]))
    verdict = None
    if len(candidates) > 1:
        if advisor is None:
            from advisor import Advisor

            advisor = Advisor()
        if advisor.available:
            verdict = advisor.ask_json(PROMPT.format(count=len(candidates),
                                                     letters=", ".join(letters)), sheets,
                                       max_tokens=1500)
    best_label = by_numbers
    if verdict and verdict.get("best") in letters:
        best_label = letters[verdict["best"]]
    best = next(c for c in candidates if c["label"] == best_label)

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
    record = {
        "video": str(video), "frames_compared": picks,
        "candidates": {letter: {"label": label, "splat": str(next(c["ply"] for c in candidates
                                                                  if c["label"] == label)),
                                **summary[label]}
                       for letter, label in letters.items()},
        "claude": verdict, "best_by_numbers": by_numbers, "best": best_label,
        "decided_by": "claude" if verdict and verdict.get("best") in letters else "numbers",
    }
    (out_dir / "choice.json").write_text(json.dumps(record, indent=1) + "\n")
    log(f"  best for {video.name}: {best_label} (decided by {record['decided_by']}) -> {out_dir / 'best.splat'}")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video")
    args = parser.parse_args()
    record = choose(Path(args.video).expanduser())
    if record:
        for letter, c in record["candidates"].items():
            print(f"  {letter} {c['label']}: SSIM {c['ssim']:.3f}, PSNR {c['psnr']:.1f} dB, gaps {c['gaps']:.1%}")
        if record["claude"]:
            print("  Claude ranking:", record["claude"].get("ranking"), "-", record["claude"].get("confidence"))
            for letter, why in (record["claude"].get("reasons") or {}).items():
                print(f"    {letter}: {why}")


if __name__ == "__main__":
    main()
