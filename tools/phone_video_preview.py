#!/usr/bin/env python3
"""Turn a `phonesim --video <clip> --dump <folder>` run into things a person can look at.

For every clip it writes a side-by-side preview video: the clip on the left with a
white box around the upright 3:4 middle the phone analyses, the phone's outlined
frames on the right, and a caption with the clip's numbers. It also writes contact
sheets (one row per clip, a handful of evenly spaced outlined frames) and a single
reel that plays every clip's preview in turn.

    tools/phone_video_preview.py --clips videos/public --dumps <folder-with-one-dump-per-clip> --out videos/public/analysis

The dump for `<clips>/<name>.mp4` is expected at `<dumps>/<name>/` (frame_00001.jpg …,
detections.jsonl) with phonesim's log at `<dumps>/<name>.log`; clips without a dump are
skipped. Needs ffmpeg (with hstack/overlay/drawbox) and Pillow.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FONT = "/System/Library/Fonts/Helvetica.ttc"
PANEL_H = 720  # height of both panels in the side-by-side video
ANALYSED_W = 540  # 3:4 panel at that height
CLIP_W = 1280  # 16:9 panel at that height
CAPTION_H = 96  # caption strip above the panels
SHEET_FRAMES = 6
SHEET_CELL = (240, 320)


def font(size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(FONT, size)
    except OSError:
        return ImageFont.load_default()


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def clip_fps(clip: Path, frames: int) -> float:
    """Analysed frames per second: phonesim samples at --fps from the start of the clip."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(clip)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    duration = float(out) if out else 0.0
    return frames / duration if duration > 0 else 3.0


def summarise(dump: Path, log: Path) -> dict:
    """Numbers for the caption, from detections.jsonl and phonesim's closing lines."""
    frames = 0
    kept = 0
    dropped = 0
    labels: dict[str, int] = {}
    with open(dump / "detections.jsonl") as f:
        for line in f:
            record = json.loads(line)
            frames += 1
            for inst in record.get("instances", []):
                if inst.get("bare"):
                    dropped += 1
                else:
                    kept += 1
                    labels[inst["label"]] = labels.get(inst["label"], 0) + 1
    ms = None
    if log.exists():
        m = re.search(r"(\d+) ms a frame", log.read_text())
        if m:
            ms = int(m.group(1))
    top = sorted(labels.items(), key=lambda kv: -kv[1])[:6]
    return {"frames": frames, "kept": kept, "dropped": dropped, "ms": ms, "top": top, "labels": labels}


def caption_png(name: str, info: dict, path: Path, width: int) -> None:
    per_frame = info["kept"] / info["frames"] if info["frames"] else 0.0
    line1 = f"{name}   {info['frames']} frames at 3 a second   {per_frame:.1f} things outlined a frame   {info['dropped']} dropped as bare wall"
    if info["ms"]:
        line1 += f"   {info['ms']} ms a frame on this Mac"
    line2 = "left: the clip, white box = the upright middle the phone analyses      right: what the phone keeps (things dropped as bare wall are not drawn)"
    seen = ", ".join(f"{label} {n}" for label, n in info["top"])
    line3 = f"most often: {seen}" if seen else "nothing outlined"
    img = Image.new("RGBA", (width, CAPTION_H), (0, 0, 0, 255))
    draw = ImageDraw.Draw(img)
    draw.text((12, 6), line1, font=font(24), fill=(255, 255, 255, 255))
    draw.text((12, 40), line2, font=font(19), fill=(220, 220, 220, 255))
    draw.text((12, 66), line3, font=font(19), fill=(220, 220, 220, 255))
    img.save(path)


def preview_video(clip: Path, dump: Path, caption: Path, out: Path, fps: float) -> None:
    filt = (
        f"[1:v]fps={fps:.4f},scale={CLIP_W}:{PANEL_H},"
        f"drawbox=x='(iw-ih*3/4)/2':y=0:w='ih*3/4':h=ih:color=white@0.9:t=4[orig];"
        f"[0:v]scale={ANALYSED_W}:{PANEL_H}[ann];"
        f"[orig][ann]hstack=inputs=2[row];"
        f"[row]pad=iw:ih+{CAPTION_H}:0:{CAPTION_H}:black[padded];"
        f"[padded][2:v]overlay=0:0[out]"
    )
    run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-framerate", f"{fps:.4f}", "-i", str(dump / "frame_%05d.jpg"),
        "-i", str(clip),
        "-i", str(caption),
        "-filter_complex", filt, "-map", "[out]", "-shortest",
        "-c:v", "libx264", "-crf", "23", "-pix_fmt", "yuv420p", "-r", "30",
        str(out),
    ])


def contact_sheets(rows: list[tuple[str, dict, Path]], out_dir: Path, per_sheet: int = 6) -> list[Path]:
    """One row per clip: its name and numbers, then evenly spaced outlined frames."""
    sheets = []
    cell_w, cell_h = SHEET_CELL
    label_w = 230
    for s in range(0, len(rows), per_sheet):
        chunk = rows[s:s + per_sheet]
        width = label_w + SHEET_FRAMES * cell_w
        height = len(chunk) * cell_h
        sheet = Image.new("RGB", (width, height), (20, 20, 20))
        draw = ImageDraw.Draw(sheet)
        for r, (name, info, dump) in enumerate(chunk):
            y0 = r * cell_h
            frames = sorted(dump.glob("frame_*.jpg"))
            picks = [frames[round(i * (len(frames) - 1) / (SHEET_FRAMES - 1))] for i in range(SHEET_FRAMES)] if frames else []
            per_frame = info["kept"] / info["frames"] if info["frames"] else 0.0
            text = [name, f"{info['frames']} frames", f"{per_frame:.1f} outlined a frame", f"{info['dropped']} dropped as", "bare wall", ""]
            text += [f"{label} {n}" for label, n in info["top"]]
            for i, t in enumerate(text):
                draw.text((10, y0 + 10 + 22 * i), t, font=font(17 if i else 20), fill=(235, 235, 235))
            for c, frame in enumerate(picks):
                im = Image.open(frame).convert("RGB").resize((cell_w, cell_h), Image.LANCZOS)
                sheet.paste(im, (label_w + c * cell_w, y0))
            draw.line([(0, y0 + cell_h - 1), (width, y0 + cell_h - 1)], fill=(60, 60, 60))
        path = out_dir / f"contact-sheet-{s // per_sheet + 1}.jpg"
        sheet.save(path, quality=88)
        sheets.append(path)
    return sheets


def reel(previews: list[Path], out: Path) -> None:
    listing = out.with_suffix(".txt")
    listing.write_text("".join(f"file '{p.resolve()}'\n" for p in previews))
    run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy", str(out)])
    listing.unlink()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", required=True, type=Path, help="folder of .mp4 clips")
    ap.add_argument("--dumps", required=True, type=Path, help="folder holding one phonesim dump per clip, named like the clip")
    ap.add_argument("--out", required=True, type=Path, help="where previews, contact sheets and the reel go")
    ap.add_argument("--fps", type=float, default=None, help="analysed frames a second (default: worked out from the dump and the clip)")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    rows: list[tuple[str, dict, Path]] = []
    previews: list[Path] = []
    for clip in sorted(args.clips.glob("*.mp4")):
        name = clip.stem
        dump = args.dumps / name
        if not (dump / "detections.jsonl").exists():
            print(f"{name}: no dump, skipped", file=sys.stderr)
            continue
        info = summarise(dump, args.dumps / f"{name}.log")
        if info["frames"] == 0:
            print(f"{name}: empty dump, skipped", file=sys.stderr)
            continue
        fps = args.fps or clip_fps(clip, info["frames"])
        caption = args.out / f"{name}-caption.png"
        caption_png(name, info, caption, CLIP_W + ANALYSED_W)
        preview = args.out / f"{name}-preview.mp4"
        preview_video(clip, dump, caption, preview, fps)
        caption.unlink()
        rows.append((name, info, dump))
        previews.append(preview)
        per_frame = info["kept"] / info["frames"]
        print(f"{name}: {info['frames']} frames, {per_frame:.1f} outlined a frame, {info['dropped']} dropped as bare wall -> {preview.name}")

    if not rows:
        print("nothing to show", file=sys.stderr)
        return 1
    sheets = contact_sheets(rows, args.out)
    reel(previews, args.out / "all-clips-reel.mp4")
    print("contact sheets:", ", ".join(p.name for p in sheets))
    print("reel:", (args.out / "all-clips-reel.mp4").name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
