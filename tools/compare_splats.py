#!/usr/bin/env python3
"""Judge any splat files of one space against each other with the pipeline's
own judge (tools/splat_choose.judge): renders beside the real frames at
trained views and at views no splat trained on, SSIM/PSNR/gaps, and three
Claude votes. Used to test outside trainers (Spirula Studio) against ours.

Usage:
    python3 tools/compare_splats.py spaces/walkthrough-full videos/IMG_4182.MOV runs/spirula-compare \
        "ours quick=splat-quick.ply" "spirula depth+normal=/path/to/splat.ply"
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "pipeline"), str(ROOT / "tools")]
from advisor import Advisor  # noqa: E402
from splat_choose import judge  # noqa: E402


def main() -> None:
    space, video, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    candidates = []
    for spec in sys.argv[4:]:
        label, ply = spec.split("=", 1)
        ply = Path(ply)
        if not ply.is_absolute():
            ply = space / ply
        candidates.append({"label": label, "space": space, "ply": ply})
    record = judge(video, candidates, out, log=print, advisor=Advisor())
    out.mkdir(parents=True, exist_ok=True)
    (out / "choice.json").write_text(json.dumps(record, indent=1) + "\n")
    for letter, c in record["candidates"].items():
        print(f"  {letter} {c['label']:28} trained SSIM {c['ssim']:.3f} PSNR {c['psnr']:.2f} gaps {c['gaps']:.2%}"
              f" | new SSIM {c.get('new_ssim', float('nan')):.3f} PSNR {c.get('new_psnr', float('nan')):.2f}"
              f" gaps {c.get('new_gaps', float('nan')):.2%}")
    claude = record.get("claude") or {}
    print("best:", record["best"], "| decided by", record["decided_by"], "|", claude.get("agreement"), claude.get("confidence"))
    for letter, why in (claude.get("reasons") or {}).items():
        print(f"  {letter}: {why}")


if __name__ == "__main__":
    main()
