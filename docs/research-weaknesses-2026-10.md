# Improving on the weaknesses of our best result (October 2026)

Research done on 6 October 2026 with Firecrawl (web search, page reads, and
its paper index), aimed at the specific faults our best rooms still have.
Our best results at the time: `walkthrough-ma` (joined solve, six measured
pieces, scene) and the pan with the phone's second opinion (walls, bed, and
a wardrobe placed from the phone's own detection).

The post-capture pipeline is this session's side; in-capture (Oasis Capture)
is the phone app's. Findings are split the same way.

## The weaknesses, as measured here

| # | Weakness | Where we saw it |
|---|---|---|
| 1 | Furniture with a mirror or dark front leaves no depth points | The pan's wardrobe: 0 body points; mirror points dropped |
| 2 | A piece's depth and extent are unobserved when only its front is seen, or it is under clutter | Wardrobe depth assumed; walkthrough bed 0.9 m wide by its masks against 1.57 |
| 3 | Per-keyframe detection flickers and mislabels | Door labelled wardrobe; wardrobe labelled hanging cloth; 12 keyframes of 42 |
| 4 | The model judge is not repeatable and measures badly | Same rows: plausible twice, structural once; three wrong wardrobe placements |
| 5 | Metric scale from single-image depth | Keyframes agree within 13%; only 2 on the median scale |
| 6 | What the video never shows | The pan sees 11% of the floor |

## What the literature says, weakness by weakness

### 4. The judge: this is a known, measured failure, with a known fix

- **LEGO-Anything** (AWS, Sept 2026) measured it directly: asked which of two
  renders is the better reconstruction, six frontier models agree with the
  true geometric metric **45.8%** of the time (chance is 50%); self-judging is
  no better than judging another model's work. Their conclusion: "Refinement
  should therefore be grounded in deterministic evidence rather than
  self-assessment." Their plugin replaces free-form self-correction with
  measured residuals (projected extent, relative depth against SAM 3 regions),
  and adds **version control**: every edit is scored and accepted, repaired or
  rolled back. It improved all six models, by up to 62.7%.
- **IDEAL-Bench** (Dartmouth, July 2026): across 15 models, object recognition
  is 70-90% while position within 0.3 m is **under 12%** for every model
  (Claude Sonnet 4.6: 4.9%). "Current VLMs are trained to describe scenes
  rather than to measure them."
- **LiteReality-Agent** (Cambridge, Oct 2026) is the closest system to ours:
  an agent edits a `Room.py` in an observe-edit-verify loop. Its evidence
  tools are the ones we arrived at independently: `render_and_compare` pairs a
  render with the capture photo **from the same calibrated camera**, at room,
  wall and object scope; `measurement` returns metres; `stitch_wall` projects
  frames onto a wall's plane with a metric grid so a fixture can be read off
  in coordinates. Layout repair is **deterministic first** (wall alignment,
  corner fitting, separation), bounded (0.6 m horizontal, 15% size for beds
  and tables, 35% others), and "measured objects cannot be deleted to remove
  violations". Quality gates are geometric (mesh intersection, support), not
  visual. Code and capture app are public.

**For us:** our overrule rule and pose-matched rows are the right direction
and should go further. Use Claude to *name and explain*, never to *place or
grade geometry*. Concretely: (a) score every structural claim against a
measured residual before acting on it, not only "misplaced"; (b) version
control for stage 3: score the room before and after each review pass and
roll back a pass that scores lower, instead of the current "fewer structural
problems" comparison; (c) object-scope render pairs with the box drawn in,
as LiteReality does, instead of whole-frame rows only.

### 3. Detection that flickers: track concepts through every frame

- **SAM 3 / 3.1** (Meta, Nov 2025 on): text-prompted detection, segmentation
  **and tracking of all instances through a video**. One call replaces
  GroundingDINO per keyframe plus SAM 2.1 per box, and gives the same object
  one identity in every frame. Cost scales with the number of tracked objects.
  In `ultralytics` (`SAM3VideoSemanticPredictor`) and Meta's repo; CUDA.
- **Boxer** (Meta, ECCV 2026): lifts open-world 2D boxes to 3D oriented boxes
  from posed images and optional sparse depth, then fuses them across views.
- **Cubify Anything** (Apple, CVPR 2025): 3D boxes from a single RGB or RGB-D
  frame; weights for both; runs on CUDA, MPS or CPU; research licence.
  **Rooms from Motion** (NeurIPS 2025) optimises global boxes against every
  observation when poses are known. **BoxFusion** fuses per-frame boxes by
  maximising the 2D overlap of projected boxes across views with a particle
  search: the same idea as our `placement.py`, as a published method.
- **Lucida** (ByteDance, Aug 2026): detects on keyframes, then *consolidates
  each instance over the full sequence* (propagating with video tracking where
  keyframes disagree), keeps an "evidence bundle" per object (selected views,
  partial points, a representative box), and re-examines the inventory against
  the whole capture. mAP 0.592 against Boxer's 0.351 on their real benchmark.
  Their stated limit is ours too: "objects that remain missing after scene
  parsing cannot be recovered by the subsequent stages".
- **SpatialLM 1.1** (NeurIPS 2025) is *not* the answer for furniture: its own
  zero-shot numbers on video are wardrobe 29-40 F1, cabinet 11-15.

**For us:** replace per-keyframe detection with SAM 3 concept tracking over
all frames on Colab. That gives masks in every frame with stable identity,
which is exactly what `placement.py` was starved of (12 keyframes, and labels
that changed between them). Then lift with Boxer or Cubify Anything as a
third opinion beside our measured boxes and the phone's.

### 1. Mirrors: detect the mirror, then use its plane

- **Mirror3D** (SFU): segment the mirror, estimate its plane from the depth of
  a thin border strip around it, and write that plane's depth into the mirror
  region. RMSE in mirror regions 1.245 -> 0.749 on NYUv2. Code and 7,011
  annotated mirrors released.
- **RRG-SLAM** (Sept 2026) identifies reflective planes by three cues together:
  planar geometry, a semantic class likely to reflect, and **high colour
  variance over time** at the same surface (a reflection changes with the
  viewpoint; paint does not). Reflection-dominated pixels are masked out of
  tracking.

**For us:** we drop points on mirrors, which deleted the wardrobe's front. Do
what Mirror3D does instead: keep the mirror's mask, take the depth of its
frame (the border strip), and fill the mask with that plane. The pan's
wardrobe would then have had a front. The temporal-variance cue is cheap to
add to densify as a check on which "mirror" detections are real.

### 2. Unseen depth and clutter: complete the object, not the box

- **FIRE3D** (UIUC/Cornell, Sept 2026): from a casual RGB video (poses and
  depth estimated by Pi3), one feed-forward pass predicts every object's pose,
  box, **amodally complete** mesh and texture, in under a minute; trained on
  80k scenes. Beats Boxer on a dataset Boxer was trained on.
- **Lucida**'s generate step: pick complementary views of one object, have an
  image model synthesise an occluder-free picture, lift it to 3D, then align.
- **Amodal3R**, **InstaScene**, **Axolotl3D**: occlusion-aware completion
  conditioned on partial points, masks and cameras.

**For us:** our clean furniture models are parametric boxes sized by
measurement. A completed mesh per object would give a real depth for the
wardrobe and a bed without its clutter. FIRE3D is the one to try first (one
pass, video in); it needs a CUDA GPU, and no release is confirmed in what I
read, so check before planning on it.

### 5. Scale

- A CVPR 2026 workshop benchmark puts **MoGe-2 at 19% scale error on real
  video** (59% on synthetic), the best monocular method; stereo methods reach
  2-10%. Our 13% keyframe spread is in line with that: this is the method's
  limit, not our bug.
- **Depth Anything 3** (ICLR 2026 oral): multi-view consistent geometry from
  any number of views, 23.6% better geometry than VGGT.
- **KitchenTwin**: a VLM names an object of known size to anchor metric scale,
  with gravity and Manhattan constraints.

**For us:** stop expecting single-image depth to fix scale. Either take it
from the phone (ARKit poses are metric; `--mapper priors` already accepts
them) or anchor on known sizes: a door is ~2.0 m, a bed is one of a few
standard lengths. That is a cheap cross-check on every room.

### 6. In-capture (for the phone app's side)

- **IntelliCap** (Aug 2025): during scanning, stripes mark surfaces not yet
  covered (from the phone's own scene mesh), and spheres mark objects that
  need more angles, chosen by segmentation plus a language model's ranking of
  which categories are hard (reflective, transparent). Heavy vision runs on a
  server from keyframes sent every 5 s; the phone only tracks, draws and
  captures. A 12-person study showed better reconstructions than free capture.
- **LiteReality Scan** (free, iOS, LiDAR): exports posed RGB frames, LiDAR
  depth, ARKit poses and the RoomPlan layout in one bundle. That bundle is the
  shape of data a post-capture agent wants. RoomPlan needs LiDAR.
- **COVER** (April 2026): a coverage-based view criterion derived from Fisher
  information that reduces to "look at what past cameras covered least".

**For the app:** what the pan shows the phone should save, in order of value
to post-capture: (1) ARKit poses and intrinsics per frame (metric scale and
no split solves); (2) its per-frame detections **with outlines and track
ids** (today the simulator's dump has labels and 3D bounds but no outlines);
(3) on Pro phones, LiDAR depth and RoomPlan's walls; (4) a coverage signal
while filming: which walls and how much floor have been seen. And one
guidance rule the pan would have benefited from: when a large piece is only
ever seen at the edge of the frame, ask for one square-on view of it.

## What I would do, in order

*Status, 10 October 2026:* items 1-3 are built (`pipeline/tracking.py`,
`pipeline/mirrors.py`, `pipeline/room_score.py`; see the README). SAM 3's
weights turned out to be handed out on request only, so the tracker carries
GroundingDINO's detections with SAM 2.1's video model instead; the detector
is the one part to swap when access comes.

1. **SAM 3 concept tracking over all frames** in densify (Colab). Fixes
   flicker and mislabels at the source and feeds placement with every frame.
2. **Mirror planes instead of dropped points** (Mirror3D's border-strip
   plane). Small change in densify; restores mirrored fronts.
3. **Version control for stage 3**: a deterministic room score (mask
   agreement of every built box, collisions, support) recorded per pass, and
   roll back any pass that lowers it. Retires most of the judge's influence.
4. **Scale anchors from known sizes** as a check on every room.
5. **A third opinion on boxes** (Boxer or Cubify Anything) beside ours and the
   phone's, fused by overlap.
6. Trial **FIRE3D** or Lucida-style completion for furniture meshes, once a
   release is confirmed.

And still the cheapest: a tape measure on the two rooms. Every benchmark above
has ground truth; we have none.

## Sources

- LEGO-Anything: https://arxiv.org/abs/2609.36380
- IDEAL-Bench: https://arxiv.org/abs/2607.03614
- LiteReality-Agent: https://arxiv.org/abs/2610.01863 · https://litereality.github.io/agent/ · https://apps.apple.com/gb/app/litereality/id6774158260
- Lucida: https://arxiv.org/abs/2608.30821
- FIRE3D: https://arxiv.org/abs/2609.08848
- SAM 3: https://github.com/facebookresearch/sam3 · https://docs.ultralytics.com/models/sam-3
- Boxer: https://facebookresearch.github.io/boxer/ · https://arxiv.org/abs/2604.05212
- Cubify Anything: https://github.com/apple/ml-cubifyanything · Rooms from Motion: https://openreview.net/forum?id=0NexR6LDiG
- BoxFusion: https://arxiv.org/abs/2506.15610
- SpatialLM: https://github.com/manycore-research/SpatialLM
- Mirror3D: https://arxiv.org/abs/2106.06629 · RRG-SLAM: https://arxiv.org/abs/2609.34527
- MoGe-2: https://arxiv.org/abs/2507.02546 · Depth Anything 3: https://openreview.net/forum?id=yirunib8l8
- Stereo benchmark (scale error): https://openaccess.thecvf.com/content/CVPR2026W/3DMV/papers/Tan_Benchmarking_Stereo_Geometry_Estimation_in_the_Wild_CVPRW_2026_paper.pdf
- IntelliCap: https://arxiv.org/abs/2508.13043 · COVER: https://arxiv.org/abs/2604.05259
- KitchenTwin: https://arxiv.org/abs/2603.24684
- VIGA: https://arxiv.org/abs/2601.11109 · SEIG: https://arxiv.org/abs/2606.02580
