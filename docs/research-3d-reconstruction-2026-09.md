# Open-source 3D reconstruction tools, surveyed 2026-09-18

Goal of the survey: OasisSpaces must turn an ordinary phone video of a cluttered
room into an *editable* 3D model (walls, floor, doors, furniture as separate
objects) plus a photoreal splat, on an 8 GB Mac and a free Colab T4. Sources:
GitHub (READMEs, licences, activity), r/GaussianSplatting, r/photogrammetry,
r/computervision threads, X. Star counts and dates are as of 2026-09-18.

## Where we stand

Our pipeline: COLMAP cameras → MoGe-2 depth + GroundingDINO/SAM outlines →
plane fitting + Claude review → parametric Blender room → OpenSplat splat +
fill, Claude judging every step. Measured plateau: splats memorise the
training frames (SSIM 0.82 at trained views vs 0.73 a step away), long
training and floater pruning did not help, and the room model's furniture is
generic boxes from a library, not the real objects.

## The tools that matter for us

### 1. Feed-forward geometry: cameras + metric depth in one pass (replaces COLMAP + MoGe on weak captures)

| Tool | What it gives | Fit | Licence |
|---|---|---|---|
| [MapAnything](https://github.com/facebookresearch/map-anything) (Meta, 3DV 2026, 3.7k★, active) | metric depth, poses, intrinsics, confidence for 2–1000+ views in one forward pass; exports COLMAP; memory-efficient mode | Best candidate to replace stage 1+2's cameras and per-frame depth. Directly gives the per-frame depth maps depth-supervised training needs, in one consistent metric frame. Takes optional known intrinsics/poses too. | code Apache-2.0; weights `facebook/map-anything-apache` (Apache) or CC-BY-NC |
| [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3) (ByteDance, 6.4k★, active) | any-view depth + poses; a metric model; a **3D Gaussian head** (feed-forward splat); `da3 video` CLI; streaming mode under 12 GB | Strong alternative front end; the Gaussian head is worth a test as a splat initialiser. | code Apache-2.0; small models Apache, Giant/Nested CC-BY-NC |
| [VGGT](https://github.com/facebookresearch/vggt) (Meta, CVPR 2025 best paper, 14k★) | poses, depth, point maps; `demo_colmap.py` → gsplat | Predecessor of the two above; commercial checkpoint exists. MapAnything wraps it. | commercial-use licence (non-military); one checkpoint |
| [LingBot-Map](https://github.com/robbyant/lingbot-map) (ECCV 2026 oral) | streaming reconstruction of very long videos (25k-frame indoor walkthrough demo) | Interesting for long walkthroughs; heavy GPU; users report artefacts. Watch, don't adopt. | Apache-2.0 |
| [SLAM3R](https://github.com/PKU-VCL-3DV/SLAM3R) (CVPR 2025) | real-time dense point cloud from video | What SpatialLM's own example uses to get a cloud from video. | non-commercial |
| [AnySplat](https://github.com/InternRobotics/AnySplat) (SIGGRAPH Asia 2025, MIT) | feed-forward Gaussians + poses from unposed views | Pose-free splat; small views (448 px); untested on rooms of our size. Secondary. | MIT |

Community note (X, Apr 2026): a comparison table of COLMAP / DUSt3R / MASt3R /
CUT3R / Fast3R / VGGT / π³ / MapAnything / DA3 exists (@gabriberton); MapAnything
1.1 (Jan 2026) claims parity or better than DA3 1.1 and Pi3X.

### 2. Editable layout: walls, doors, windows, furniture boxes with names (replaces most of stage 3)

| Tool | What it gives | Fit | Licence |
|---|---|---|---|
| [SpatialLM 1.1](https://github.com/manycore-research/SpatialLM) (NeurIPS 2025, 4.7k★) | from a **metric, z-up point cloud**: walls, doors, windows and oriented boxes for **59 furniture categories**; categories can be restricted per room; 0.5B (Qwen) and 1B (Llama) models | Our dense cloud is already metric and gravity-aligned (shapes.py's up vector), so it can be fed directly. Claude's object list can pick the categories. Replaces plane fitting + box clustering; keep Claude's review on top. Caveat: trained mostly on Chinese apartments (a second sofa was missed in their example). | code Apache-2.0; 1.1 weights CC-BY-NC-4.0 (Sonata encoder) |
| [SceneScript](https://github.com/facebookresearch/scenescript) (Meta) | layout + boxes as a token language from egocentric video | Same idea, trained on Aria synthetic data. | CC-BY-NC |
| Ov3R (CVPR 2026), SegVGGT, OCH3R | semantic / instance 3D from video | Papers; no usable code found yet. | — |

### 3. Real furniture as separate textured objects (replaces the parametric furniture library)

| Tool | What it gives | Fit | Licence |
|---|---|---|---|
| [SAM 3D Objects](https://github.com/facebookresearch/sam-3d-objects) (Meta, 7.4k★, weights + encoder released 2025-11/2026-06) | from one image + a mask: textured 3D object (mesh and Gaussian splat), pose and layout; multi-object scene mode; built for clutter and occlusion | Exactly the "editable extraction": per-object meshes placed in the scene. Our stage 2 already has each object's best frame and SAM outline. Runs locally now (the earlier paid API is no longer the only route). | SAM License: use, modify, redistribute allowed, with attribution and terms; no non-commercial clause seen |
| [PartCrafter](https://github.com/wgsxm/PartCrafter) (NeurIPS 2025, MIT, 2.5k★) | one image → part-level meshes; a scene variant (3D-Front) | Trained on renders; real photos need style transfer. Secondary. | MIT |
| [MILo](https://github.com/Anttwo/MILo) (SIGGRAPH Asia 2025) | mesh extracted during splat training; Blender add-on to edit/animate the splat through the mesh; `--imp_metric indoor` | Splat → editable mesh path if we want a whole-room mesh. | Gaussian-Splatting licence (non-commercial research) |
| [2DGS](https://github.com/hbb1/2d-gaussian-splatting), [PGSR](https://github.com/zju3dv/PGSR) | surface-accurate splats → meshes | Older; MILo supersedes. | mixed |

### 4. Splat trainers that already do what we were about to build

| Tool | What it gives | Fit | Licence |
|---|---|---|---|
| [Spirula Studio](https://github.com/harry7557558/spirula-studio) (441★, very active, macOS validated Aug 2026) | one binary: frame extraction, its own SfM, AI masking, MCMC/IGS+/MRNF training, **depth/normal regularisation**, exposure/white-balance (bilateral grid + PPISP), meshing; Vulkan so it runs on Apple Silicon and on CUDA; CLI; 10M Gaussians in 8 GB | The strongest "just use it" option: a Reddit user reports it fixed textureless walls ("Spirula depth+normal → training") that Brush did not. Runs on this Mac without CUDA. Test it on both videos before writing our own depth-supervised trainer. | GPL-3.0 (fine to run as a tool; not to embed) |
| [LichtFeld Studio](https://github.com/MrNeRF/LichtFeld-Studio) (3.7k★) | MCMC, bilateral grid, 3DGUT, PPISP; the community's quality reference | NVIDIA sm75+ (T4 qualifies), CUDA 12.8, Linux from source, Windows binaries. No Mac. | GPL-3.0 |
| [gsplat](https://github.com/nerfstudio-project/gsplat) 1.6 (5.7k★) | the library; depth loss in its trainer; MCMC strategy | What our own trainer would be built on (splat_train.py, started). | Apache-2.0 |
| [DN-Splatter](https://github.com/maturk/dn-splatter) | depth + normal priors on nerfstudio; several depth losses; meshing | The research reference for our idea; last update Nov 2024; heavy install. | Apache-2.0 |
| [Brush](https://github.com/ArthurBrussee/brush) (5.1k★, Apache) | Rust/WebGPU trainer: Mac, Linux, Windows, browser; masks; no depth supervision | A cross-platform OpenSplat replacement for the Mac, not a quality jump. | Apache-2.0 |
| [OOOSplat](https://github.com/ooolabdev/ooosplat) (1k★) | one-click desktop app: COLMAP + Brush + editor | Same components as ours; useful to compare output, not to adopt. | Apache-2.0 |
| [Adaptive Frame Extractor](https://github.com/morishuz/adaptive-frame-extractor) (MIT) | motion-aware frame selection instead of fixed fps | Cheap improvement to stage 1's frame picking. | MIT |

### What the community says about our exact problem (plain walls, phone video)

- r/GaussianSplatting "How do you preserve solid, stable walls in indoor 3DGS?" (Aug 2026): a real-estate pipeline much like ours (4K video, COLMAP, dense seed, per-image exposure, normal constraints) still gets unstable neutral walls. Answers: depth priors (Depth Anything 3 instead of PatchMatch), LiDAR where available, capture paths that look along walls, hybrid 2DGS. Resolution and frame count (1080p vs 4K, 300 vs 1200 frames, 30k vs 120k steps) made **no meaningful difference** for them, which matches our long-training result.
- "Best Open Source Resources / hidden gems" (Sep 2026): Spirula Studio is the current recommendation, "checks a lot of boxes", including from people who were about to buy Postshot. Meta's Hyperscape is the quality bar (phone-grade cameras + accurate poses from the headset), which supports the capture-app direction.
- r/computervision on LingBot-Map: feed-forward models look great in demos, but "the thing that always bites you downstream is scale", and real-time claims assume data-centre GPUs.

## What this changes for OasisSpaces

1. **Front end:** try MapAnything (Apache weights) on both videos: poses + metric depth for every frame in one pass, exported to COLMAP. If it holds up on the pan video (where COLMAP gave 13% scale spread), it replaces stage 1 and the depth half of stage 2, and hands depth-supervised training its depth maps for free.
2. **Layout:** feed our metric, z-up dense cloud to SpatialLM 1.1 with Claude's object names as the category list; compare its walls/doors/boxes with stage 3's, and keep Claude's review as the judge. Non-commercial weights: fine for evaluation, a decision to make before shipping.
3. **Objects:** run SAM 3D Objects on each object's best frame + SAM outline (both already produced by stage 2) to get a textured mesh per piece of furniture, placed by SpatialLM's box. This is the editable-3D deliverable.
4. **Splat:** test Spirula Studio (Mac, no CUDA) on both videos before finishing our own gsplat trainer; if its depth/normal-regularised training fixes the walls, use it as the trainer behind stage 4 and keep our Claude judging around it. Otherwise finish `pipeline/splat_train.py` (gsplat + depth loss) on Colab, with MapAnything/DA3 depth maps.
5. **Stage 1:** adopt motion-aware frame selection.

Constraints that stay: no re-filming as a fix; Claude judges, models produce.
