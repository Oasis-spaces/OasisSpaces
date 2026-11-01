#!/usr/bin/env python3
"""Real rooms with measured truth, for tools/score_space.py: ARKitScenes.

ARKitScenes (Apple, 2021) is 5,000 handheld iPad videos of real homes, each with
the ARKit camera path, a LiDAR mesh of the room and hand-drawn boxes around
its furniture (17 kinds: bed, sofa, table, chair, cabinet, toilet, stove...).
That is everything score_space.py needs to say how right a space is, across
every kind of room, not just the ones we happen to film.

    # pick scenes, the same number of each kind of room
    python3 tools/benchmark_arkitscenes.py pick --per-type 3 --out benchmark/scenes.json

    # one scene: its truth folder, and with --video the .mov to run the pipeline on
    python3 tools/benchmark_arkitscenes.py fetch 42444946 benchmark/42444946 [--video]

    # after the pipeline has made spaces/<name> from each video and score_space.py
    # has scored them: the results by kind of room
    python3 tools/benchmark_arkitscenes.py report spaces/aks-* [--out benchmark/report.md]

A fetched scene's folder:
    truth/truth.json   camera path (t from the video's start), boxes, room type
    truth/surface.npy  the LiDAR mesh's vertices (subsampled), metres, z up
    raw/               Apple's files as downloaded
    <id>.mov           the video (with --video, ~0.5-1 GB)

Licence: ARKitScenes is under Apple's own licence: free for research and for
evaluating our pipeline; the data must not be trained on for, or shipped in,
the product. These files stay out of git (benchmark/ is ignored).
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import random
import shutil
import sys
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import score_space  # noqa: E402

BASE = "https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1"
SURFACE_POINTS = 600_000
TYPES = ("bathroom", "kitchen", "bedroom", "living room", "dining or office", "other")


def url(fold: str, video_id: str, name: str) -> str:
    return f"{BASE}/raw/{fold}/{video_id}/{name}"


def download(src: str, dst: Path, log=print) -> Path:
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    log(f"  downloading {src.rsplit('/', 1)[-1]}")
    with urllib.request.urlopen(src, timeout=60) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, length=1 << 20)
    tmp.rename(dst)
    return dst


def metadata(cache: Path) -> list[dict]:
    path = download(f"{BASE}/raw/metadata.csv", cache / "metadata.csv")
    return list(csv.DictReader(io.StringIO(path.read_text())))


def fold_of(video_id: str, cache: Path) -> str:
    for row in metadata(cache):
        if row["video_id"] == str(video_id):
            return row["fold"]
    raise SystemExit(f"{video_id} is not in ARKitScenes' metadata")


# ------------------------------------------------------------------ reading Apple's files
def read_boxes(annotation: Path) -> list[dict]:
    """The labelled boxes: rows of normalizedAxes are the box's axes (Apple's
    compute_box_3d puts corners at axesᵀ · local + centroid)."""
    boxes = []
    for item in json.loads(annotation.read_text()).get("data", []):
        obb = item["segments"]["obbAligned"]
        boxes.append({"label": item["label"], "centre": [float(v) for v in obb["centroid"]],
                      "size": [float(v) for v in obb["axesLengths"]],
                      "axes": np.asarray(obb["normalizedAxes"], float).reshape(3, 3).tolist()})
    return boxes


def read_cameras(traj: Path) -> list[dict]:
    """Camera centres from lowres_wide.traj. Each line is a timestamp, an
    axis-angle and a translation of the world-to-camera transform (Apple's
    loader inverts it to get the pose), so the centre is -Rᵀt."""
    from scipy.spatial.transform import Rotation

    rows = np.loadtxt(traj, ndmin=2)
    R = Rotation.from_rotvec(rows[:, 1:4]).as_matrix()
    centres = -np.einsum("nji,nj->ni", R, rows[:, 4:7])
    t0 = rows[0, 0]
    return [{"t": round(float(ts - t0), 4), "centre": c.round(4).tolist()} for ts, c in zip(rows[:, 0], centres)]


def read_ply_vertices(path: Path) -> np.ndarray:
    """The vertices of a binary little-endian PLY (float x, y, z first)."""
    with open(path, "rb") as f:
        header = []
        while True:
            line = f.readline()
            if not line:
                raise SystemExit(f"{path}: no end_header")
            header.append(line.decode("ascii", "replace").strip())
            if header[-1] == "end_header":
                break
        if not any(h.startswith("format binary_little_endian") for h in header):
            raise SystemExit(f"{path}: only binary little-endian PLY is read")
        count, props, in_vertex = 0, [], False
        sizes = {"float": "<f4", "float32": "<f4", "double": "<f8", "uchar": "u1", "uint8": "u1",
                 "char": "i1", "short": "<i2", "ushort": "<u2", "int": "<i4", "uint": "<u4"}
        for h in header:
            parts = h.split()
            if parts[:2] == ["element", "vertex"]:
                count, in_vertex = int(parts[2]), True
            elif parts[:1] == ["element"]:
                in_vertex = False
            elif in_vertex and parts[:1] == ["property"]:
                props.append((parts[2], sizes[parts[1]]))
        data = np.frombuffer(f.read(count * np.dtype(props).itemsize), dtype=props, count=count)
    return np.column_stack([data["x"], data["y"], data["z"]]).astype(np.float32)


# ------------------------------------------------------------------ commands
def fetch(video_id: str, out: Path, video: bool, cache: Path, log=print) -> Path:
    fold = fold_of(video_id, cache)
    raw = out / "raw"
    annotation = download(url(fold, video_id, f"{video_id}_3dod_annotation.json"), raw / f"{video_id}_3dod_annotation.json", log)
    mesh = download(url(fold, video_id, f"{video_id}_3dod_mesh.ply"), raw / f"{video_id}_3dod_mesh.ply", log)
    traj = download(url(fold, video_id, "lowres_wide.traj"), raw / "lowres_wide.traj", log)
    if video:
        download(url(fold, video_id, f"{video_id}.mov"), out / f"{video_id}.mov", log)

    boxes = read_boxes(annotation)
    vertices = read_ply_vertices(mesh)
    if len(vertices) > SURFACE_POINTS:
        vertices = vertices[np.random.default_rng(0).choice(len(vertices), SURFACE_POINTS, replace=False)]
    truth = out / "truth"
    truth.mkdir(parents=True, exist_ok=True)
    np.save(truth / "surface.npy", vertices)
    kind = score_space.room_type(b["label"] for b in boxes)
    (truth / "truth.json").write_text(json.dumps({
        "source": f"ARKitScenes {fold} {video_id}", "room_type": kind,
        "cameras": read_cameras(traj), "boxes": boxes}) + "\n")
    floor, ceiling = score_space.level(vertices[:, 2].astype(float), "floor"), score_space.level(vertices[:, 2].astype(float), "ceiling")
    log(f"{video_id}: {kind}, {len(boxes)} boxes ({', '.join(sorted({b['label'] for b in boxes}))}), "
        f"{len(vertices):,} surface points"
        + (f", floor to ceiling {ceiling - floor:.2f} m" if None not in (floor, ceiling) else ""))
    return truth


def pick(per_type: int, fold: str, cache: Path, seed: int, budget: int, log=print) -> dict:
    """Scenes in equal numbers per kind of room, decided from their labels
    (which needs each candidate's small annotation file)."""
    rows = [r for r in metadata(cache) if r["fold"] == fold and r.get("is_in_threedod") == "True"
            and r.get("sky_direction") == "Up"]
    random.Random(seed).shuffle(rows)
    chosen = {k: [] for k in TYPES}
    tried = 0
    for r in rows:
        if all(len(v) >= per_type for v in chosen.values()) or tried >= budget:
            break
        vid = r["video_id"]
        tried += 1
        try:
            annotation = download(url(fold, vid, f"{vid}_3dod_annotation.json"),
                                  cache / "annotations" / f"{vid}.json", log=lambda *_: None)
        except OSError as e:
            log(f"  {vid}: {e}")
            continue
        labels = [b["label"] for b in read_boxes(annotation)]
        kind = score_space.room_type(labels)
        if len(chosen[kind]) < per_type:
            chosen[kind].append({"video_id": vid, "fold": fold, "labels": sorted(set(labels))})
            log(f"  {kind:17s} {vid}  {', '.join(sorted(set(labels)))}")
    short = {k: per_type - len(v) for k, v in chosen.items() if len(v) < per_type}
    if short:
        log(f"after {tried} annotations, still short of: {short}")
    return {"fold": fold, "seed": seed, "per_type": per_type, "scenes": chosen}


def report(spaces: list[Path]) -> str:
    """Median results by kind of room, from each space's score.json."""
    by_type: dict[str, list[dict]] = {}
    for space in spaces:
        path = space / "score.json"
        if path.exists():
            r = json.loads(path.read_text())
            by_type.setdefault(r["room_type"], []).append(r)
    if not by_type:
        return "No scored spaces (run score_space.py on each first)."

    def med(values, fmt):
        values = [v for v in values if v is not None]
        return fmt.format(float(np.median(values))) if values else "–"

    lines = ["| room | n | scale error | walls off | floor off | footprint IoU | coverage | furniture found | box IoU |",
             "|---|---|---|---|---|---|---|---|---|"]
    every = [r for rs in by_type.values() for r in rs]
    for kind, rs in sorted(by_type.items()) + [("all", every)]:
        f = [r.get("furniture", {}) for r in rs]
        lines.append(" | ".join([
            f"| {kind}", str(len(rs)),
            med([abs(r["scale_error"]) for r in rs], "{:.0%}"),
            med([r.get("wall_error_m") for r in rs], "{:.2f} m"),
            med([abs(r["floor_error_m"]) if r.get("floor_error_m") is not None else None for r in rs], "{:.2f} m"),
            med([r.get("footprint", {}).get("iou") for r in rs], "{:.2f}"),
            med([r.get("coverage") for r in rs], "{:.0%}"),
            med([x.get("recall") for x in f], "{:.0%}"),
            med([x.get("median_iou") for x in f], "{:.2f}")]) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache", type=Path, default=ROOT / "benchmark" / "cache")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("pick")
    p.add_argument("--per-type", type=int, default=3)
    p.add_argument("--fold", default="Validation")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--budget", type=int, default=300, help="most annotations to look at")
    p.add_argument("--out", type=Path, default=ROOT / "benchmark" / "scenes.json")
    f = sub.add_parser("fetch")
    f.add_argument("video_id")
    f.add_argument("out", type=Path)
    f.add_argument("--video", action="store_true", help="also download the .mov (large)")
    r = sub.add_parser("report")
    r.add_argument("spaces", type=Path, nargs="+")
    r.add_argument("--out", type=Path)
    args = parser.parse_args()

    if args.command == "pick":
        result = pick(args.per_type, args.fold, args.cache, args.seed, args.budget)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=1) + "\n")
        print(f"wrote {args.out}")
    elif args.command == "fetch":
        fetch(args.video_id, args.out, args.video, args.cache)
    else:
        text = report(args.spaces)
        if args.out:
            args.out.write_text(text + "\n")
        print(text)


if __name__ == "__main__":
    main()
