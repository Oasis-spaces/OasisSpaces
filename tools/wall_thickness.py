#!/usr/bin/env python3
"""How sharp a cloud's walls and floor are: the spread of its points around the
room's planes, in centimetres. Lower is better: a wall the keyframes agree on is
thin; one they disagree on is smeared.

For each built wall and the floor in shapes.json, the points within 15 cm of
the plane (inside the middle 80% of its extent) give the median and the 90th
percentile distance to it. Several clouds of the same space can be compared,
each against the same planes:

    python3 tools/wall_thickness.py spaces/<name> cloud-dense.ply cloud-moge3.ply cloud-da3.ply
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
from pointcloud import load_ply  # noqa: E402

BAND_M = 0.15


def thickness(space: Path, cloud_name: str) -> list[dict]:
    shapes = json.loads((space / "shapes.json").read_text())
    metre = json.loads((space / "densify.json").read_text())["colmap_units_per_metre"]
    world = np.array(shapes["world"], float)
    pts = load_ply(space / cloud_name).points.astype(np.float64) @ world.T
    rows = []
    for i, p in enumerate(shapes["planes"]):
        if p.get("label") not in ("wall", "floor") or p.get("build") is False or p.get("source") == "inferred":
            continue
        n, c = np.array(p["normal"], float), np.array(p["center"], float)
        a = np.array(p["axis_a"], float)
        b = np.array(p["axis_b"], float) if "axis_b" in p else np.cross(n, a)
        rel = pts - c
        d = rel @ n
        inside = (np.abs(rel @ a) < 0.8 * p["half_a"]) & (np.abs(rel @ b) < 0.8 * p.get("half_b", 1e9))
        near = inside & (np.abs(d) < BAND_M * metre)
        if near.sum() < 100:
            continue
        rows.append({"plane": f"{p['label']} {i}", "points": int(near.sum()),
                     "median_cm": float(np.median(np.abs(d[near])) / metre * 100),
                     "p90_cm": float(np.percentile(np.abs(d[near]), 90) / metre * 100)})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("space", type=Path)
    parser.add_argument("clouds", nargs="+")
    args = parser.parse_args()
    results = {c: thickness(args.space, c) for c in args.clouds if (args.space / c).exists()}
    planes = [r["plane"] for r in next(iter(results.values()))]
    print(f"{'surface':12s}" + "".join(f"{c:>26s}" for c in results))
    for k, plane in enumerate(planes):
        print(f"{plane:12s}" + "".join(f"{rows[k]['median_cm']:>11.2f} cm (90%: {rows[k]['p90_cm']:5.2f})"
                                       if k < len(rows) else f"{'':>26s}" for rows in results.values()))
    print(f"{'mean':12s}" + "".join(f"{np.mean([r['median_cm'] for r in rows]):>11.2f} cm (90%: "
                                    f"{np.mean([r['p90_cm'] for r in rows]):5.2f})" for rows in results.values()))


if __name__ == "__main__":
    main()
