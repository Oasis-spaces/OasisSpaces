#!/usr/bin/env python3
"""AnySplat's one-pass prediction for a set of frames, run inside AnySplat's
own environment.

pipeline/splat_anysplat.py runs this with AnySplat's Python (ANYSPLAT_PYTHON):
AnySplat pins its own PyTorch, xformers and gsplat, which cannot share the
pipeline's environment. It writes the raw prediction, in AnySplat's own frame
and units, to an .npz; the pipeline aligns it to the camera solve and converts
it to a splat.

    $ANYSPLAT_PYTHON tools/anysplat_infer.py --anysplat-dir DIR --out raw.npz frame_00001.jpg ...

What the .npz holds:
  means (N, 3), covariances (N, 3, 3), opacities (N,) in 0..1 (after the
  sigmoid, as AnySplat renders them), dc (N, 3) the colour's zeroth
  spherical-harmonics band; cam2world (V, 4, 4) and intrinsics (V, 3, 3,
  normalised by the input's width and height) for each input frame, in the
  order given; frames, the names; crops (V, 4), each input's crop box in the
  frame's own pixels.

Frames go in as AnySplat was trained: the short side scaled to 448 and a
centred square cut out ("square"). A centred crop keeps the camera's centre
and direction, so the predicted cameras compare directly with the solve's.
"--shape full" keeps the whole frame (448 on the short side, the long side
rounded to the model's 14-pixel patches), which a portrait video otherwise
loses a third of; the model was trained on squares, so it is a trial.

Two changes to how AnySplat runs, neither to what it predicts:
  - AnySplat builds its network from VGGT-1B's weights (5 GB, CC BY-NC 4.0),
    then loads its own checkpoint over every one of them. Here it is built
    from an empty VGGT instead, and loading checks that the checkpoint
    filled every VGGT weight.
  - The VGGT part runs in bfloat16. A GPU older than Ampere (Colab's T4) has
    no fast attention for bfloat16, and PyTorch falls back to attention that
    holds every pair of the ~25,000 image patches of 24 views at once (about
    20 GB). There, attention runs in float16 instead, as VGGT's own demo
    does on those GPUs.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

SIZE = 448
PATCH = 14


def load(path: Path, shape: str):
    """The frame as AnySplat takes it (3, H, W in 0..1) and its crop box."""
    import torch
    from PIL import Image

    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = SIZE / min(w, h)
    rw, rh = max(SIZE, round(w * scale)), max(SIZE, round(h * scale))
    img = img.resize((rw, rh), Image.BICUBIC)
    if shape == "square":
        cw = ch = SIZE
    else:
        cw, ch = rw // PATCH * PATCH, rh // PATCH * PATCH
    left, top = (rw - cw) // 2, (rh - ch) // 2
    img = img.crop((left, top, left + cw, top + ch))
    box = [left / scale, top / scale, (left + cw) / scale, (top + ch) / scale]
    array = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
    return array, box


VGGT_PARTS = ("encoder.aggregator.", "encoder.camera_head.", "encoder.depth_head.", "encoder.point_head.")


def load_model(checkpoint: str):
    """AnySplat with its checkpoint, built without VGGT-1B's weights."""
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    from src.model.encoder.vggt.models.vggt import VGGT
    from src.model.model.anysplat import AnySplat

    VGGT.from_pretrained = classmethod(lambda cls, *args, **kwargs: cls())
    model = AnySplat.from_pretrained(checkpoint)
    local = Path(checkpoint) / "model.safetensors"
    path = local if local.exists() else hf_hub_download(checkpoint, "model.safetensors")
    with safe_open(str(path), "pt") as f:
        stored = set(f.keys())
    empty = [k for k in model.state_dict() if k.startswith(VGGT_PARTS) and k not in stored]
    if empty:
        sys.exit(f"the checkpoint does not hold {len(empty)} VGGT weights (first: {empty[0]}); "
                 "this script assumes it holds them all")
    return model


def half_attention_without_bf16(torch) -> bool:
    """On a GPU older than Ampere, attention given bfloat16 runs in float16."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] >= 8:
        return False
    import torch.nn.functional as F
    attention = F.scaled_dot_product_attention

    def half(q, k, v, *args, **kwargs):
        if q.dtype != torch.bfloat16:
            return attention(q, k, v, *args, **kwargs)
        return attention(q.half(), k.half(), v.half(), *args, **kwargs).to(torch.bfloat16)

    F.scaled_dot_product_attention = half
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("frames", nargs="+", type=Path)
    parser.add_argument("--anysplat-dir", type=Path, required=True, help="a clone of InternRobotics/AnySplat")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shape", choices=["square", "full"], default="square")
    parser.add_argument("--checkpoint", default="lhjiang/anysplat")
    args = parser.parse_args()

    sys.path.insert(0, str(args.anysplat_dir.resolve()))
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    half_attention = half_attention_without_bf16(torch)
    started = time.time()
    model = load_model(args.checkpoint).to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    loaded = time.time()
    views = [load(f, args.shape) for f in args.frames]
    images = torch.stack([v[0] for v in views]).unsqueeze(0).to(device)
    # AnySplat's autocast is CUDA's: without it on a CPU, the bfloat16 trunk
    # hands bfloat16 to float heads, which then fail. CPU is for checks only.
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=device.type == "cpu"):
        gaussians, pose = model.inference(images)
    if device.type == "cuda":
        torch.cuda.synchronize()
    predicted = time.time()

    cam2world = pose["extrinsic"][0].float().cpu().numpy()
    if cam2world.shape[-2] == 3:            # 3x4: complete it
        bottom = np.tile(np.array([[[0, 0, 0, 1.0]]]), (len(cam2world), 1, 1))
        cam2world = np.concatenate([cam2world, bottom], axis=1)
    np.savez_compressed(
        args.out,
        means=gaussians.means[0].float().cpu().numpy(),
        covariances=gaussians.covariances[0].float().cpu().numpy(),
        opacities=gaussians.opacities[0].float().cpu().numpy().reshape(-1),
        dc=gaussians.harmonics[0][..., 0].float().cpu().numpy(),
        cam2world=cam2world,
        intrinsics=pose["intrinsic"][0].float().cpu().numpy(),
        frames=np.array([f.name for f in args.frames]),
        crops=np.array([v[1] for v in views], dtype=np.float32),
    )
    info = {"device": str(device), "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "float16_attention": half_attention, "views": len(views), "shape": args.shape, "input": list(images.shape[-2:]),
            "gaussians": int(gaussians.means.shape[1]), "load_seconds": round(loaded - started, 1),
            "predict_seconds": round(predicted - loaded, 1),
            "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if device.type == "cuda" else None}
    args.out.with_suffix(".json").write_text(json.dumps(info, indent=1) + "\n")
    print(json.dumps(info))


if __name__ == "__main__":
    main()
