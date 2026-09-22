#!/usr/bin/env python3
"""Turn each object job into a clean textured mesh with TRELLIS, on a GPU.

This runs on the machine with the GPU (Colab), not on the Mac. It reads the
jobs pipeline/object_models.py wrote (a frame, that object's cut-out and the
camera that took it) and writes one .glb per object beside them.

TRELLIS (microsoft/TRELLIS, MIT, weights MIT and ungated) makes a 3D asset
from a single masked image: a sparse voxel structure, then a sparse latent
decoded as Gaussians and as a mesh, which are baked together into a textured
GLB. It is the architecture SAM 3D Objects also builds on, whose own weights
are gated.

On a T4 (16 GB, no bfloat16) it runs in float16 with the xformers attention
backend; the mesh and its texture are decoded after the Gaussians are freed.

Usage (on the GPU machine):
    python object_mesh.py /content/jobs [--only B13] [--steps 12]
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("ATTN_BACKEND", "xformers")   # flash-attn needs Ampere or newer
os.environ.setdefault("SPCONV_ALGO", "native")      # no benchmarking: each object runs once

MODEL = "microsoft/TRELLIS-image-large"
TEXTURE_SIZE = 1024
SIMPLIFY = 0.95              # share of triangles dropped before baking the texture


def cut_out(image_path: Path, mask_path: Path):
    """The object on its own, as TRELLIS wants it: RGBA, cropped to the mask
    with a small margin, and square, so the object fills the frame the way its
    training images did."""
    import numpy as np
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    mask = np.asarray(Image.open(mask_path).convert("L")) > 127
    rows, cols = np.flatnonzero(mask.any(axis=1)), np.flatnonzero(mask.any(axis=0))
    top, bottom, left, right = rows[0], rows[-1], cols[0], cols[-1]
    side = max(bottom - top, right - left)
    margin = round(side * 0.08)
    middle_y, middle_x = (top + bottom) // 2, (left + right) // 2
    half = side // 2 + margin
    box = (middle_x - half, middle_y - half, middle_x + half, middle_y + half)
    rgba = np.dstack([np.asarray(image), (mask * 255).astype(np.uint8)])
    return Image.fromarray(rgba).crop(box)


def run(jobs: Path, only: list[str] | None, steps: int, log=print) -> list[dict]:
    import torch
    from trellis.pipelines import TrellisImageTo3DPipeline
    from trellis.utils import postprocessing_utils

    folders = sorted(p for p in jobs.iterdir() if p.is_dir() and (p / "job.json").exists())
    folders = [p for p in folders if not only or p.name in only]
    if not folders:
        raise SystemExit(f"no jobs in {jobs}")
    log(f"loading {MODEL}")
    pipeline = TrellisImageTo3DPipeline.from_pretrained(MODEL)
    pipeline.cuda()
    done = []
    for folder in folders:
        job = json.loads((folder / "job.json").read_text())
        started = time.time()
        views = sorted(folder.glob("view-*.jpg"))
        cuts = [cut_out(v, v.with_suffix(".png")) for v in views] or [cut_out(folder / "image.jpg",
                                                                             folder / "mask.png")]
        for n, cut in enumerate(cuts):
            cut.save(folder / f"cutout-{n}.png")
        log(f"{folder.name} {job['label']}: {len(cuts)} view(s), largest {max(c.size[0] for c in cuts)}px")
        sampler = {"steps": steps, "cfg_strength": 7.5}
        latent = {"steps": steps, "cfg_strength": 3.0}
        # Several views of one object beat one: a single view leaves every other
        # side to be invented. "stochastic" is TRELLIS's own default for them.
        outputs = (pipeline.run_multi_image(cuts, seed=1, formats=["gaussian", "mesh"],
                                            sparse_structure_sampler_params=sampler,
                                            slat_sampler_params=latent, mode="stochastic")
                   if len(cuts) > 1 else
                   pipeline.run(cuts[0], seed=1, formats=["gaussian", "mesh"],
                                sparse_structure_sampler_params=sampler, slat_sampler_params=latent))
        glb = postprocessing_utils.to_glb(outputs["gaussian"][0], outputs["mesh"][0],
                                          simplify=SIMPLIFY, texture_size=TEXTURE_SIZE, verbose=False)
        glb.export(folder / "model.glb")
        record = {"id": folder.name, "label": job["label"], "model": "trellis",
                  "views": len(cuts), "seconds": round(time.time() - started),
                  "vertices": int(len(glb.vertices)), "faces": int(len(glb.faces)),
                  "size": [round(float(v), 4) for v in glb.bounding_box.extents],
                  "steps": steps, "texture": TEXTURE_SIZE}
        (folder / "model.json").write_text(json.dumps(record, indent=1) + "\n")
        log(f"  {record['faces']:,} faces in {record['seconds']}s -> {folder.name}/model.glb")
        done.append(record)
        del outputs
        torch.cuda.empty_cache()
    (jobs / "models.json").write_text(json.dumps(done, indent=1) + "\n")
    return done


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("jobs", type=Path)
    parser.add_argument("--only", nargs="*", help="only these piece ids")
    parser.add_argument("--steps", type=int, default=12, help="sampler steps (12 is the demo's fast setting)")
    args = parser.parse_args()
    done = run(args.jobs.resolve(), args.only, args.steps)
    print(f"{len(done)} model(s) written")


if __name__ == "__main__":
    main()
