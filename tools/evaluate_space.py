#!/usr/bin/env python3
"""Score reconstructed spaces so pipeline changes can be compared.

For each space: how many frames COLMAP registered and into how many models,
sparse and dense point counts, how consistent the per-frame MoGe scale was,
floor size and wall height in metres (when densify.py recorded a metric
scale), wall flatness, and the splat's size. Compare the metres against a
tape measure; wall height is the observed extent, so it reads low when the
capture missed the floor or ceiling seam.

Usage:
    python3 tools/evaluate_space.py spaces/first-test spaces/first-test-v2
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
from pointcloud import load_ply

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def ply_vertex_count(path: Path) -> int | None:
    """Vertex count from a PLY header, without reading the body."""
    if not path.exists():
        return None
    with open(path, "rb") as f:
        for _ in range(64):
            tokens = f.readline().decode("ascii", "replace").split()
            if tokens[:2] == ["element", "vertex"]:
                return int(tokens[2])
            if tokens[:1] == ["end_header"]:
                break
    return None


def bin_count(path: Path) -> int:
    with open(path, "rb") as f:
        return struct.unpack("<Q", f.read(8))[0]


def colmap_stats(space: Path) -> dict:
    sparse = space / "workspace" / "sparse"
    models = [d for d in sparse.iterdir() if (d / "images.bin").exists()] \
        if sparse.exists() else []
    solved = sorted((bin_count(d / "images.bin"), bin_count(d / "points3D.bin"))
                    for d in models)
    images = space / "workspace" / "images"
    total = sum(1 for p in images.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS) \
        if images.exists() else None
    frames, points = solved[-1] if solved else (0, 0)
    return {"models": len(models), "frames": frames, "total": total,
            "sparse": points}


def structure_stats(space: Path, units_per_metre: float | None) -> dict:
    """Floor size, wall height and wall flatness from shapes.json."""
    shapes_path = space / "shapes.json"
    cloud_path = next((p for p in (space / "cloud-dense.ply", space / "cloud.ply")
                       if p.exists()), None)
    if not shapes_path.exists() or cloud_path is None:
        return {}
    shapes = json.loads(shapes_path.read_text())
    world = np.array(shapes["world"])
    P = load_ply(cloud_path).points.astype(np.float64)
    if len(P) > 400_000:
        P = P[np.random.default_rng(0).choice(len(P), 400_000, replace=False)]
    P = P @ world.T
    extent = float(np.linalg.norm(np.percentile(P, 98, 0) - np.percentile(P, 2, 0)))
    band = extent * 0.024  # twice shapes.py's RANSAC threshold

    to_m = (lambda v: v / units_per_metre) if units_per_metre else None
    stats = {}
    floors = [p for p in shapes["planes"] if p["kind"] == "floor_or_ceiling"]
    # Measured walls that get built: not ones dropped in review, and not the
    # inferred walls that close unfilmed sides (no points to measure them by).
    walls = [p for p in shapes["planes"] if p["kind"] == "wall"
             and p.get("build", True) and p.get("source") != "inferred"]
    stats["walls"] = len(walls)
    if floors and to_m:
        floor = max(floors, key=lambda p: p["points"])
        stats["floor_m"] = (to_m(2 * floor["half_a"]), to_m(2 * floor["half_b"]))
    if walls and to_m:
        stats["wall_height_m"] = float(np.median([to_m(2 * w["half_b"]) for w in walls]))

    rms = []
    for wall in walls:
        n, c = np.array(wall["normal"]), np.array(wall["center"])
        a, b = np.array(wall["axis_a"]), np.array(wall["axis_b"])
        rel = P - c
        dist = rel @ n
        near = ((np.abs(rel @ a) <= wall["half_a"]) & (np.abs(rel @ b) <= wall["half_b"])
                & (np.abs(dist) < band))
        if near.sum() > 100:
            rms.append(float(np.sqrt(np.mean(dist[near] ** 2))))
    if rms:
        wall_rms = float(np.median(rms))
        stats["wall_rms_pct"] = 100 * wall_rms / extent
        if to_m:
            stats["wall_rms_cm"] = 100 * to_m(wall_rms)
    return stats


def evaluate(space: Path) -> dict:
    row = {"space": space.name, **colmap_stats(space)}
    row["dense"] = ply_vertex_count(space / "cloud-dense.ply")
    meta_path = space / "densify.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    row["depth"] = meta.get("depth_model", "da2" if row["dense"] else None)
    row["scale_spread"] = meta.get("scale_spread")
    row.update(structure_stats(space, meta.get("colmap_units_per_metre")))
    row["splat"] = ply_vertex_count(space / "splat.ply")
    return row


def fmt(value, spec="", none="—"):
    if value is None:
        return none
    if isinstance(value, tuple):
        return " x ".join(format(v, spec) for v in value)
    return format(value, spec)


def main() -> None:
    spaces = [Path(p) for p in sys.argv[1:]]
    if not spaces:
        sys.exit(__doc__)
    columns = [
        ("space", lambda r: r["space"]),
        ("models", lambda r: fmt(r["models"])),
        ("frames", lambda r: f"{r['frames']}/{fmt(r['total'])}"),
        ("sparse pts", lambda r: fmt(r["sparse"], ",")),
        ("dense pts", lambda r: fmt(r["dense"], ",")),
        ("depth", lambda r: fmt(r["depth"])),
        ("scale spread", lambda r: fmt(r["scale_spread"], ".0%")),
        ("floor m", lambda r: fmt(r.get("floor_m"), ".2f")),
        ("wall h m", lambda r: fmt(r.get("wall_height_m"), ".2f")),
        ("walls", lambda r: fmt(r.get("walls"))),
        ("wall rms", lambda r: ("—" if "wall_rms_pct" not in r else
                                f"{r['wall_rms_pct']:.2f}%"
                                + (f" ({r['wall_rms_cm']:.1f} cm)" if "wall_rms_cm" in r else ""))),
        ("splat", lambda r: fmt(r["splat"], ",")),
    ]
    rows = [[get(r) for _, get in columns] for r in map(evaluate, spaces)]
    widths = [max(len(name), *(len(r[i]) for r in rows)) for i, (name, _) in enumerate(columns)]
    print("  ".join(name.ljust(w) for (name, _), w in zip(columns, widths)))
    for r in rows:
        print("  ".join(cell.ljust(w) for cell, w in zip(r, widths)))
    print("\nwall rms: RMS distance of points near each wall to its plane, median over "
          "walls, as % of the room's diagonal (and in cm when there is a metric scale).")


if __name__ == "__main__":
    main()
