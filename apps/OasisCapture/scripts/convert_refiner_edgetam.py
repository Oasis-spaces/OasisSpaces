#!/usr/bin/env python3
"""Build the mask refiner from EdgeTAM: MaskEncoder.mlpackage and MaskDecoder.mlpackage.

EdgeTAM (Meta Reality Labs, CVPR 2025, https://github.com/facebookresearch/EdgeTAM)
is SAM 2 rebuilt for phones: a RepViT image encoder, 13.9 M parameters in all,
Apache-2.0 for code and weights, and its checkpoint is committed in the GitHub repo
(no Hugging Face). Its family runs about eight times faster than MobileSAM on an A15
(EdgeSAM, the same encoder family: 38.7 FPS on an iPhone 14 against MobileSAM's
4.9), and the same weights also track objects through video (its memory path, not
exported here). On our clips, carried through 20 and 48 frames on a CPU, it held
the TV, sofa, rug, table, plant, dresser and bed as they left the view and came back.

This replaces convert_refiner.py (MobileSAM) without changing the files' names:
  MaskEncoder: a 1024x1024 RGB image (0-255) -> embedding (1, 256, 64, 64) plus the
               two higher-resolution maps SAM 2's decoder uses, feat_s0 (1, 32, 256, 256)
               and feat_s1 (1, 64, 128, 128). Once a frame.
  MaskDecoder: embedding, feat_s0, feat_s1, a box (x0, y0, x1, y1) and a point
               (x, y, label) in the 1024 square's pixels -> a 256x256 mask (logits; > 0 is
               inside) and a score. Once per thing. The point's label is 1 for a point on
               the thing (from the detector's own mask), or -1 for none.
The app tells this refiner from MobileSAM by the decoder's inputs (feat_s0 is only here).

Three things Meta's coreml/export_to_coreml.py does not do, all checked against
SAM2ImagePredictor with --check:
  * ImageNet mean and std are applied inside the encoder (the official export only
    divides by 255);
  * a box is two corner points labelled 2 and 3, as SAM2ImagePredictor encodes it (the
    official prompt encoder export ignores its "boxes" input), and the points are
    embedded without in-place boolean indexing, so the graph converts;
  * the decoder asks for one mask (multimask_output=False), which keeps SAM 2's
    stability fallback; the official trace always returns the three-mask output.
A point on the thing matters: on a framed TV on a panelled wall, the box alone gave
the wall panel; box plus a point on the screen gave the screen.

The image may be stretched to the square, as EdgeTAM was trained, or letterboxed
(long side 1024, the rest black), as the app feeds MobileSAM today; --check measures
both against the predictor.

Needs EdgeTAM installed without its CUDA extension, and coremltools:
    SAM2_BUILD_CUDA=0 pip install --no-build-isolation -e <EdgeTAM clone>
    ~/.venvs/oasis-coreml/bin/python apps/OasisCapture/scripts/convert_refiner_edgetam.py [--check frame.jpg ...]
--check compares the PyTorch wrapper; add --coreml (on a Mac, after building) to run
the fp16 packages themselves on the Neural Engine and CPU and compare their masks too.
The checkpoint is fetched from the EdgeTAM repo into ~/.cache/oasis-models/edgetam.pt.
"""

import argparse
import sys
import urllib.request
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
CACHE = Path.home() / ".cache" / "oasis-models"
CHECKPOINT = CACHE / "edgetam.pt"
CHECKPOINT_URL = "https://raw.githubusercontent.com/facebookresearch/EdgeTAM/main/checkpoints/edgetam.pt"
CONFIG = "configs/edgetam.yaml"
ENCODER_OUT = ROOT / "Resources" / "MaskEncoder.mlpackage"
DECODER_OUT = ROOT / "Resources" / "MaskDecoder.mlpackage"
SIZE = 1024
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def load():
    """EdgeTAM with the weights of its checkpoint. The backbone's ImageNet weights are
    not downloaded first (timm would fetch them from Hugging Face); the checkpoint holds
    every weight anyway."""
    if not CHECKPOINT.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        print(f"fetching {CHECKPOINT_URL}")
        urllib.request.urlretrieve(CHECKPOINT_URL, CHECKPOINT)
    import timm.models
    from sam2.build_sam import build_sam2
    from sam2.modeling.backbones import timm as sam2_timm

    create = timm.models.create_model
    sam2_timm.create_model = lambda *a, **k: create(*a, **dict(k, pretrained=False))
    try:
        return build_sam2(CONFIG, str(CHECKPOINT), device="cpu").eval()
    finally:
        sam2_timm.create_model = create


class Encoder(torch.nn.Module):
    """1024x1024 RGB (0-255) -> embedding and SAM 2's two high-resolution maps."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1) * 255)
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1) * 255)

    def forward(self, image):
        model = self.model
        backbone_out = model.forward_image((image - self.mean) / self.std)
        _, feats, _, sizes = model._prepare_backbone_features(backbone_out)
        # As SAM2ImagePredictor.set_image: the lowest-resolution map gets no_mem_embed.
        feats[-1] = feats[-1] + model.no_mem_embed
        maps = [f.permute(1, 2, 0).reshape(1, -1, h, w) for f, (h, w) in zip(feats, sizes)]
        return maps[-1], maps[0], maps[1]


class Decoder(torch.nn.Module):
    """Embedding + maps + a box and a point -> the single best 256x256 mask and its score."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        prompt = model.sam_prompt_encoder
        self.register_buffer("not_a_point", prompt.not_a_point_embed.weight.detach().clone())
        self.register_buffer("label_embeds", torch.cat([e.weight.detach() for e in prompt.point_embeddings[:4]]))

    def forward(self, embedding, feat_s0, feat_s1, box, point):
        prompt = self.model.sam_prompt_encoder
        coords = torch.cat([box.reshape(1, 2, 2), point[:, :2].reshape(1, 1, 2)], dim=1)
        labels = torch.cat([torch.tensor([[2.0, 3.0]]), point[:, 2:3]], dim=1)          # (1, 3)
        # PromptEncoder._embed_points, without in-place boolean indexing.
        pe = prompt.pe_layer.forward_with_coords(coords + 0.5, (SIZE, SIZE))
        pad = (labels == -1).float().unsqueeze(-1)
        sparse = pe * (1 - pad) + pad * self.not_a_point
        for k in range(4):
            sparse = sparse + (labels == k).float().unsqueeze(-1) * self.label_embeds[k]
        dense = prompt.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(1, -1, *embedding.shape[-2:])
        masks, scores, _, _ = self.model.sam_mask_decoder(
            image_embeddings=embedding, image_pe=prompt.get_dense_pe(),
            sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
            multimask_output=False, repeat_image=False, high_res_features=[feat_s0, feat_s1])
        return masks[:, 0], scores[:, 0]


def square(image, mode: str):
    """The 1024 square fed to the encoder, and the scale from image pixels (x, y) to it."""
    from PIL import Image

    if mode == "stretch":
        return image.resize((SIZE, SIZE), Image.BILINEAR), (SIZE / image.width, SIZE / image.height)
    s = SIZE / max(image.size)
    padded = Image.new("RGB", (SIZE, SIZE))
    padded.paste(image.resize((round(image.width * s), round(image.height * s)), Image.BILINEAR), (0, 0))
    return padded, (s, s)


def mask_in_image(logits, image, mode: str):
    """A 256x256 mask of the square back to the image's pixels."""
    from PIL import Image

    m = (logits > 0).astype(np.uint8) * 255
    if mode == "letterbox":
        s = SIZE / max(image.size)
        m = m[:round(image.height * s / 4), :round(image.width * s / 4)]
    return np.array(Image.fromarray(m).resize(image.size, Image.BILINEAR)) > 127


def check(paths: list[str], boxes: list[list[float]], points: list[list[float]], coreml: bool = False) -> None:
    """The wrapper (what is converted) against SAM2ImagePredictor on each image and box,
    stretched and letterboxed; IoU of the final masks in image pixels. With `coreml`, the
    built fp16 packages too, on the stretched square the app feeds them, loaded for the
    Neural Engine and CPU as the app will (needs macOS)."""
    from PIL import Image
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    if coreml:
        if sys.platform != "darwin":
            sys.exit("--coreml runs the Core ML packages, which needs macOS")
        import coremltools as ct

        units = ct.ComputeUnit.CPU_AND_NE
        ml_encoder = ct.models.MLModel(str(ENCODER_OUT), compute_units=units)
        ml_decoder = ct.models.MLModel(str(DECODER_OUT), compute_units=units)
    model = load()
    encoder, decoder, predictor = Encoder(model).eval(), Decoder(model).eval(), SAM2ImagePredictor(model)
    iou = lambda a, b: float((a & b).sum()) / max(1, float((a | b).sum()))
    for path, box, point in zip(paths, boxes, points):
        image = Image.open(path).convert("RGB")
        with torch.inference_mode():
            predictor.set_image(image)
            have_point = point is not None
            kw = dict(point_coords=np.array([point], np.float32), point_labels=np.array([1])) if have_point else {}
            ref, ref_score, _ = predictor.predict(box=np.array(box, np.float32), multimask_output=False, **kw)
            ref = ref[0] > 0
            line = [f"{Path(path).name}: reference {ref.mean():.3f} of the image, score {float(ref_score[0]):.2f}"]
            for mode in ("stretch", "letterbox"):
                sq, (sx, sy) = square(image, mode)
                x = torch.from_numpy(np.asarray(sq, np.float32)).permute(2, 0, 1)[None]
                emb, s0, s1 = encoder(x)
                b = torch.tensor([[box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy]])
                p = torch.tensor([[point[0] * sx, point[1] * sy, 1.0]] if have_point else [[0.0, 0.0, -1.0]])
                logits, score = decoder(emb, s0, s1, b, p)
                ours = mask_in_image(logits[0].numpy(), image, mode)
                line.append(f"{mode} IoU {iou(ours, ref):.3f} (score {float(score[0]):.2f})")
                if coreml and mode == "stretch":
                    feats = ml_encoder.predict({"image": sq})
                    out = ml_decoder.predict({"embedding": feats["embedding"], "feat_s0": feats["feat_s0"],
                                              "feat_s1": feats["feat_s1"], "box": b.numpy(), "point": p.numpy()})
                    cm = mask_in_image(np.asarray(out["mask"]).reshape(-1, 256, 256)[0], image, mode)
                    line.append(f"Core ML IoU {iou(cm, ref):.3f}, {iou(cm, ours):.3f} against PyTorch "
                                f"(score {float(np.ravel(out['score'])[0]):.2f})")
        print("  " + "; ".join(line))


def convert() -> None:
    import coremltools as ct

    model = load()
    encoder = Encoder(model).eval()
    traced = torch.jit.trace(encoder, torch.rand(1, 3, SIZE, SIZE) * 255)
    mlmodel = ct.convert(
        traced,
        inputs=[ct.ImageType(name="image", shape=(1, 3, SIZE, SIZE), color_layout=ct.colorlayout.RGB)],
        outputs=[ct.TensorType(name="embedding"), ct.TensorType(name="feat_s0"), ct.TensorType(name="feat_s1")],
        minimum_deployment_target=ct.target.iOS17, compute_precision=ct.precision.FLOAT16,
        convert_to="mlprogram")
    mlmodel.short_description = ("EdgeTAM image encoder (RepViT, Apache-2.0): 1024x1024 image -> "
                                 "256x64x64 embedding + feat_s0 32x256x256 + feat_s1 64x128x128")
    mlmodel.save(str(ENCODER_OUT))
    print(f"wrote {ENCODER_OUT}")

    decoder = Decoder(model).eval()
    example = (torch.rand(1, 256, 64, 64), torch.rand(1, 32, 256, 256), torch.rand(1, 64, 128, 128),
               torch.tensor([[200.0, 300.0, 700.0, 900.0]]), torch.tensor([[450.0, 600.0, 1.0]]))
    traced = torch.jit.trace(decoder, example)
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="embedding", shape=(1, 256, 64, 64)),
                ct.TensorType(name="feat_s0", shape=(1, 32, 256, 256)),
                ct.TensorType(name="feat_s1", shape=(1, 64, 128, 128)),
                ct.TensorType(name="box", shape=(1, 4)), ct.TensorType(name="point", shape=(1, 3))],
        outputs=[ct.TensorType(name="mask"), ct.TensorType(name="score")],
        minimum_deployment_target=ct.target.iOS17, compute_precision=ct.precision.FLOAT16,
        convert_to="mlprogram")
    mlmodel.short_description = ("EdgeTAM prompt encoder + mask decoder: embedding, maps, box and point "
                                 "-> 256x256 mask logits and a score")
    mlmodel.save(str(DECODER_OUT))
    print(f"wrote {DECODER_OUT}")
    for out in (ENCODER_OUT, DECODER_OUT):
        size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
        print(f"  {out.name}: {size / 1e6:.1f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", nargs="+", metavar="IMAGE", help="compare with SAM2ImagePredictor instead")
    parser.add_argument("--box", nargs=4, type=float, action="append", metavar=("X0", "Y0", "X1", "Y1"),
                        help="one per --check image, in its pixels")
    parser.add_argument("--point", nargs=2, type=float, action="append", metavar=("X", "Y"),
                        help="optional, one per --check image: a point on the thing")
    parser.add_argument("--coreml", action="store_true",
                        help="with --check: also run the built Core ML packages (macOS)")
    args = parser.parse_args()
    if args.check:
        if not args.box or len(args.box) != len(args.check):
            sys.exit("give one --box per --check image")
        points = args.point if args.point and len(args.point) == len(args.check) else [None] * len(args.check)
        check(args.check, args.box, points, coreml=args.coreml)
    else:
        convert()
