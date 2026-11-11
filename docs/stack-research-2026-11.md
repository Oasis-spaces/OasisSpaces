# Stack research, November 2026

What we run today, where it is weak (measured on the bedroom video), and what
open-source or free tools would improve it. Three areas get their own sections:
depth analysis, pattern recognition, gap filling. Follows the October audit
(`audit-prompts-and-stack-2026-10.md`); where that audit already decided
something, this says whether the decision still holds.

Licences were read from each project's own page on 11 Nov 2026. "Unverified"
means the page did not state it; check before shipping.

## 1. What we run today

| Stage | Tool | Licence | Where it runs | Weakness seen on the bedroom |
| --- | --- | --- | --- | --- |
| Camera solve | COLMAP 4.2 global mapper, SIFT | BSD | CPU | Works. Slow on CPU; loses frames at sharp turns in other spaces |
| Solve gap fill | MapAnything | Apache model available (`--apache`) | GPU | Not exercised on CPU runs |
| Depth | MoGe-2 ViT-L, one frame at a time, scaled per keyframe to COLMAP points | MIT code | CPU/GPU | Scale spread 15% between keyframes; one frame at 14.2 against a median of 3.28 still fused; frames from the start and end of the walk disagree by about 20% |
| False depth | `mirrors.py` plane fill for mirrors, windows, screens | ours | CPU | Works when the surface is detected |
| Object marks | Claude on gridded keyframes | API | API | Good. What it does not mark is never found |
| Outlines | SAM 2.1 hiera-tiny, box prompt, `pick_outline` | Apache | CPU | Per-keyframe only on CPU; no tracking between keyframes; small model |
| Shapes | RANSAC planes, labelled-point boxes, room outline | ours | CPU | Wardrobe depth uncertain; half the floor never seen |
| Judgement | Claude with brief, room record, measured rechecks | API | API | Room score went from −0.65 to +0.31 |
| Splat | OpenSplat | AGPLv3 | GPU | Not run on the bedroom. AGPL is a problem for a hosted product |
| Fill after removal | LaMa | Apache | CPU/GPU | October audit: 3 of 3 fills rejected by review |
| Furniture models | TRELLIS | MIT | GPU 16 GB+ | Not run on the bedroom |
| Viewer | three.js point cloud + room model, Poly Haven CC0 | MIT / CC0 | Browser | Points only; flat floor patch where a piece is removed |

The room score is the yardstick for every change below: a tool is adopted when
it raises the score on the bedroom and on the ARKitScenes benchmark
(`tools/benchmark_arkitscenes.py`), not because its paper says so.

## 2. Depth analysis

The biggest measured weakness. Each keyframe gets its own depth and its own
scale, and nothing makes neighbouring keyframes agree.

### Changes that need no new model (do first)

1. **Reject outlier keyframes.** A keyframe whose scale is more than about 25%
   from the median of its neighbours is not fused. Frame 45 (scale 14.2) would
   go.
2. **Weight scale by anchors.** Fit each keyframe's scale with RANSAC over its
   COLMAP points, weight by the number of inliers, then smooth the scale along
   the walk. A keyframe with few anchors borrows from its neighbours.
3. **Cross-keyframe check.** Project each keyframe's depth into the next
   keyframe; drop points that disagree by more than a few percent. This is the
   classic multi-view consistency filter and removes most floating points.
4. **Scale and shift, not scale alone.** Fit `a·d + b` per keyframe where the
   anchors spread over enough depth range; monocular depth is often off by a
   shift as well as a scale.
5. **Real size from one tape measure.** Already built (`set_real_scale.py`,
   viewer "Set real length"). Ask for it in the capture flow.

### Models worth a trial

| Tool | What it adds | Licence | Compute | Verdict |
| --- | --- | --- | --- | --- |
| **MoGe-3** (ViT-L 370M, ViT-G 1.25B) | Much finer point-map detail than MoGe-2, metric scale and normals | Code MIT; weights have no licence tag on Hugging Face (unverified) | ViT-L fits a T4 | Drop-in trial: same API family as what we run. First candidate |
| **Depth Anything 3** BASE / SMALL | Takes many frames at once and can take our COLMAP poses as input, so its depth agrees between frames by construction | Apache-2.0 (BASE, SMALL, METRIC-LARGE, MONO-LARGE). Giant, Large and Nested are CC BY-NC: do not use | BASE 0.12B: T4, likely CPU | Second candidate. Attacks the 15% spread directly |
| **Prior Depth Anything** | Depth completion: takes sparse measured points (our COLMAP points) plus the image, returns dense depth that passes through them | Repo Apache-2.0; backbone and weights unverified | GPU | Replaces "scale a guess" with "fill between measurements". Trial after the two above |
| OMNI-DC, Marigold-DC | Same idea as above | Check each | GPU; Marigold is slow | Fallbacks |
| VGGT-Ω, π³, D4RT | Poses and depth in one forward pass | VGGT family weights have been non-commercial; verify | 16 GB+ | Watch, not adopt. COLMAP plus MapAnything already covers this |
| MapAnything (already in) | Can also give dense metric depth from poses | Apache variant | GPU | Test its depth output against MoGe on the same frames; costs nothing to try |
| Depth Anything V4 | Appeared on arXiv in August, withdrawn within a day after its main claim failed | n/a | n/a | Ignore |

The October decision "stay on MoGe-2" is superseded: MoGe-3 is out and DA3 has
Apache models that use known poses.

### Using depth better downstream

- **Normals.** MoGe-2/3 `-normal` models give surface normals. Use them to
  tell a wardrobe front from the wall behind it (same direction, different
  depth) and to sharpen RANSAC walls.
- **Depth in the splat.** Train with a depth loss (see section 5), so the
  splat surface sits where the measured depth is.
- **Depth confidence in the record.** Report per-keyframe scale and agreement
  to the agent, so it trusts box sizes from consistent frames over others.

## 3. Pattern recognition

Claude decides what things are; SAM cuts their outlines; points inherit labels.
The weak links are outline quality on CPU and the lack of 3D consistency.

| Tool | What it adds | Licence | Compute | Verdict |
| --- | --- | --- | --- | --- |
| **SAM 2.1 base-plus or large** | Better outlines than tiny from the same box prompt | Apache | base-plus is usable on CPU for ~30 keyframes | Do now. One-line change |
| **SAM 3 / 3.1** | Finds every instance of a named thing and tracks it through the video; 3.1 tracks many objects at once | SAM License: royalty-free, redistribution under the same terms, no military use; commercial use not excluded. Gated on Hugging Face (access pending) | GPU 16 GB; or Meta's hosted API at about $0.20 per 1,000 video frames | Trial once access lands. Claude names the objects, SAM 3 finds and tracks all of them, Claude checks. Meta itself recommends pairing it with a language model for complex phrases |
| **SAM 2 video tracking** (already coded, CUDA only) | Outlines on every frame, not only keyframes | Apache | T4 | Turn on for GPU runs. More frames per object means better boxes |
| **Lifting masks to 3D** (Segment-then-Splat, SuperGSeg, Seed2GS) | Each splat belongs to one object, so removing a piece removes exactly its splats | Research code; check each | GPU | After the splat stage moves to gsplat. Our current rule (point inside box) is the cheap version |
| **SpatialLM 1.1** | Reads a point cloud, writes walls, doors, windows and furniture boxes as text | Qwen variant Apache, but its point encoder weights are CC BY-NC | GPU | Not for the product. Useful as a second opinion in benchmarks. Its own scores are weak on wardrobes (29–40 F1) and cabinets (11–15), strong on beds (97) |
| PolyLayout and similar | Multi-room floor plans as polygons | Unverified | GPU | Watch. `room_outline.py` already handles any shape |
| **Claude for openings** | Doors and windows marked on frames like furniture, then placed on the nearest wall | API | API | Do now. Uses what we have; SpatialLM's door and window scores (47–51) are no better than what Claude sees |
| YOLOE-26, RF-DETR | Fast on-phone detection for capture guidance | Check (YOLO family is AGPL) | Phone | Only for live capture hints, as the October audit said |

Rules that matter more than the model:

- **Vote across frames.** A point's label is the majority over all keyframes
  that see it, weighted by outline confidence. One wrong outline stops
  mattering.
- **Parts are not pieces.** Headboard, drawer, door: recorded as parts of a
  piece, never scored as a separate piece.
- **Repeating patterns.** Tiles, wardrobe door panels and floorboards give a
  free scale and alignment check: Claude counts the repeats, the measured
  spacing must be constant along the surface. A cheap test for depth drift.

## 4. Gap filling

Three different gaps, three different tools.

### 4a. Geometry nobody filmed (half the bedroom floor)

- **Now, no model:** floor and walls are planes, so extend them inside the
  outline and mark the result as inferred (already the rule in the brief).
  Colour the fill from the seen part of the same surface; for patterned
  floors, tile a rectified patch.
- **Depth completion** (section 2) fills holes inside frames that were
  filmed.
- **Capture guidance** fixes the cause: tell the person which part of the
  floor and which walls have not been seen before they stop filming.

### 4b. What is behind a removed piece (images)

A ladder; go up only when the step below fails review.

| Step | Tool | Licence | Compute | Notes |
| --- | --- | --- | --- | --- |
| 1 | LaMa (in) | Apache | CPU | Fine for plain walls and floors; failed 3 of 3 on real furniture removal |
| 2 | **RORem** | Apache repo, SDXL-inpainting base (OpenRAIL++) | 12 GB, T4 with the fast variant | Trained specifically to remove objects rather than invent new ones. First upgrade |
| 3 | **OSOR** (SDXL checkpoint) | Code MIT, SDXL checkpoint OpenRAIL++-M. Its FLUX checkpoint is non-commercial | 12–16 GB | Also removes the shadow and reflection of the piece |
| 4 | **Qwen-Image-Edit** | Apache-2.0 | 24 GB+, or quantised on 16 GB | Strongest open editor; heavy. OmniGen 2 (Apache) is the lighter alternative |
| — | FLUX.1 Fill / Kontext dev | Non-commercial | — | Do not use, despite the October note; the Apache options have caught up |

Every fill stays labelled as generated, and each is checked by Claude against
the surrounding frames before it is used.

### 4c. Holes in the 3D scene

| Tool | What it does | Licence | Verdict |
| --- | --- | --- | --- |
| **Depth-guided splat fill** (SplatFill, Inpaint360GS) | Fill the image, lift the fill with depth, train the splat on it, keep views consistent | Unverified, research code | Trial after 4b works in 2D. Our own version is simple: RORem fill, MoGe depth on the filled image, add to the splat |
| Difix3D+ | One-step diffusion that cleans up splat artefacts | NVIDIA licence, likely non-commercial | Benchmark only |
| GSFixer and similar | Same goal | Check | Watch |

### 4d. Whole furniture models (the unseen back of a wardrobe)

| Tool | Licence | VRAM | Verdict |
| --- | --- | --- | --- |
| **TRELLIS.2** (4B, textured GLB) | MIT; its renderer dependencies carry their own licences | 24 GB | Replaces TRELLIS when a 24 GB GPU is available |
| **TripoSG** | MIT | 8 GB | Fits a free T4. Default for free compute |
| Pixal3D, Direct3D-S2, Hi3DGen | MIT | 12–24 GB | Alternatives; try on the pieces TripoSG gets wrong |
| Step1X-3D | Apache | 16 GB+ | Alternative |
| PartCrafter | MIT | 16 GB+ | Parts (doors, drawers) as separate meshes; useful for editing later |
| SAM 3D Objects | SAM License | GPU | Places the model in the scene from one image with its mask; trial with SAM 3 |
| Hunyuan3D 2.x | Excludes EU, UK, South Korea | — | Do not use |
| SPAR3D / SF3D | Free under $1M revenue | 8 GB | Acceptable now, a liability later |

The model is always scaled and placed to the measured box, never the other way
round.

## 5. The splat stage

- **Replace OpenSplat with gsplat** (Apache, v1.5). It has depth loss, MCMC
  densification and exposure correction, and removes the AGPL dependency.
  **Brush** (Apache, WebGPU) trains on a Mac without CUDA and is the option
  for local runs.
- Add depth and normal priors in the DN-Splatter style, using the MoGe output
  we already compute. Indoor walls stop being fog.
- For a mesh rather than a splat (collision, measuring): 2DGS / PGSR / MILo.
  Trial later; the plane-and-box room model already covers editing.

## 6. Free compute

| Service | Free allowance | GPU | Fit |
| --- | --- | --- | --- |
| **Kaggle** | 30 GPU hours a week, guaranteed, 9-hour sessions | T4 ×2 or P100, 16 GB | Best free home for the whole pipeline. Has an API, so `colab_pipeline.py` can be ported |
| Colab (free / AI Pro) | Variable; T4 on this account | T4 | What we use; sessions die, as the last run showed |
| Lightning AI | Monthly free GPU hours | T4 / L4 | Persistent disk, SSH. Good second home |
| Hugging Face ZeroGPU | Minutes a day | 48 GB class | Single heavy calls only (TRELLIS.2, Qwen edit), not the pipeline |
| Modal | About $30 a month in credits, card required | Any | Per-second billing; good for the heavy single stages |
| CPU (this container, a Mac) | — | — | Stages 1–3 already run on CPU in about an hour |

Everything through stage 3, including the proposed depth changes, DA3-BASE,
SAM 2.1 base-plus and TripoSG, fits a T4.

## 7. Order of work

Each step is measured on the bedroom and on ARKitScenes before the next.

1. **Depth without new models**: outlier keyframes, anchor-weighted and
   smoothed scale, cross-keyframe check, scale-and-shift. Report the spread
   before and after (now 15%).
2. **SAM 2.1 base-plus** in place of tiny; **label voting** across frames.
3. **MoGe-3 against MoGe-2**, then **DA3-BASE with COLMAP poses**, on the same
   keyframes. Keep whichever lowers the spread and raises the room score.
4. **Doors and windows from Claude marks**; parts not scored as pieces.
5. **Inferred floor and wall fill** in the viewer with real colour, marked as
   inferred.
6. **Port the Colab driver to Kaggle** so GPU runs stop depending on one
   service.
7. **gsplat with depth loss** replaces OpenSplat.
8. **RORem** as the fill after LaMa; OSOR next if shadows remain.
9. **TripoSG** on free GPUs, **TRELLIS.2** where 24 GB is available.
10. **SAM 3.1** when access is granted: named-object tracking through the
    whole video.

Not adopting: FLUX Fill/Kontext, Hunyuan3D, Difix3D+, SpatialLM, DA3 Large and
Giant (all licence), VGGT-class solvers (no gain over what runs), Depth
Anything V4 (withdrawn).

## Sources

- MoGe: github.com/microsoft/MoGe
- Depth Anything 3: github.com/ByteDance-Seed/Depth-Anything-3
- Prior Depth Anything: github.com/SpatialVision/Prior-Depth-Anything
- SAM 3: github.com/facebookresearch/sam3 (LICENSE)
- SpatialLM: github.com/manycore-research/SpatialLM
- RORem, OSOR, Qwen-Image-Edit: project pages and the Hugging Face forum
  thread on object removal (Sep 2026)
- TRELLIS.2: github.com/microsoft/TRELLIS.2; TripoSG: github.com/VAST-AI-Research/TripoSG
- gsplat: github.com/nerfstudio-project/gsplat; Brush: github.com/ArthurBrussee/brush
- Kaggle, Lightning AI, Modal, Hugging Face: each service's pricing page
