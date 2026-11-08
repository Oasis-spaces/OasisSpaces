#!/usr/bin/env python3
"""Give a space its real scale from one measured length.

A reconstruction from video alone gets its metres from monocular depth, which
the keyframes can disagree on by 15% or more (the bedroom's did). One length a
person measured (a door's width, a wall) fixes it: the space viewer shows the
length as reconstructed, the person enters the real one, and this writes the
correction into densify.json, which every later stage reads its metres from:

    colmap_units_per_metre        corrected: the video's value / (real / reconstructed)
    colmap_units_per_metre_video  the video's own value, kept
    scale_measured                {real_m, reconstructed_m, factor}

Then the viewer's export is written again in real metres.

    python3 tools/set_real_scale.py spaces/<name> --real 0.90 --reconstructed 0.83
    python3 tools/set_real_scale.py spaces/<name> --layout bedroom-layout.json   # the viewer's export
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))


def set_scale(space: Path, real_m: float, reconstructed_m: float) -> dict:
    meta_path = Path(space) / "densify.json"
    meta = json.loads(meta_path.read_text())
    video = meta.get("colmap_units_per_metre_video", meta["colmap_units_per_metre"])
    if not (real_m > 0 and reconstructed_m > 0):
        raise SystemExit("both lengths must be positive")
    # reconstructed_m is in the metres of the video's own scale (the viewer's export
    # before any correction), so the factor applies to that value, not to an earlier fix.
    factor = real_m / reconstructed_m
    meta["colmap_units_per_metre_video"] = video
    meta["colmap_units_per_metre"] = video / factor
    meta["scale_measured"] = {"real_m": real_m, "reconstructed_m": reconstructed_m, "factor": round(factor, 5)}
    meta_path.write_text(json.dumps(meta, indent=1) + "\n")
    return meta["scale_measured"]


def from_layout(path: Path) -> tuple[float, float]:
    layout = json.loads(Path(path).read_text())
    scale = layout.get("scale")
    if not scale:
        raise SystemExit(f"{path} sets no real length (use 'Set real length' in the viewer first)")
    measure = layout["measures"][scale["measure"]]
    reconstructed = sum((a - b) ** 2 for a, b in zip(measure["a"], measure["b"])) ** 0.5
    return scale["real_m"], reconstructed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("space", type=Path)
    parser.add_argument("--real", type=float, help="the real length, metres")
    parser.add_argument("--reconstructed", type=float, help="the same length as reconstructed, metres")
    parser.add_argument("--layout", type=Path, help="the space viewer's exported layout, instead of the two lengths")
    args = parser.parse_args()
    real, reconstructed = from_layout(args.layout) if args.layout else (args.real, args.reconstructed)
    if real is None or reconstructed is None:
        parser.error("give --real and --reconstructed, or --layout")
    done = set_scale(args.space, real, reconstructed)
    change = (done["factor"] - 1) * 100
    print(f"{args.space.name}: real scale from a {real:.3f} m length (reconstructed {reconstructed:.3f} m): "
          f"{abs(change):.1f}% {'larger' if change >= 0 else 'smaller'} than the video said")
    import space_viewer_export
    if (args.space / "cloud-dense.ply").exists() and (args.space / "shapes.json").exists():
        out = space_viewer_export.export(args.space)
        print(f"viewer export rewritten in real metres: {out}")


if __name__ == "__main__":
    main()
