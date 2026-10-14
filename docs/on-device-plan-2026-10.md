# Reconstruction on the phone: the plan (October 2026)

The owner's constraint: reconstruction must not need a GPU server, Colab or CUDA. Everything that
turns a capture into a room has to run on the phone, built from open-source parts that a commercial
app can ship. The owner offered to obtain whatever licences that takes; the asks are at the end.

The floor device is the **iPhone 13**: A15, 4 GB of memory, no LiDAR.

What stays off the phone:

- **Claude.** It stays in the loop over the network, for naming things and for the reviews it does
  today. That is judgement served by Anthropic's API; no GPU server, Colab or CUDA machine of ours is
  involved. How the phone reaches it is in section 10.
- **One-time work:** converting models (a Mac, `build.sh`) and any retraining a licence fix needs.

Where the phone's own GPU is used:

- **While recording:** the live analysis should move to the Neural Engine and CPU, so recording stays
  cool and smooth (section 1).
- **After capture:** the stages may use the GPU. Every open on-phone splat trainer found is GPU-only.
  LaMa needs fp32, which the Neural Engine cannot run; the CPU can, more slowly.

This plan rests on three things:

- a search of the field in seven layers, each checked by a second pass that tried to refute it;
- hands-on tests of the parts that could be run here;
- the earlier audit (`docs/audit-prompts-and-stack-2026-10.md`).

The draft was then checked claim by claim against the sources and the code, and corrected. The
app's existing recording, live models and review screen run on the iPhone 13 today. None of the new
parts below has run on a phone: there was no Swift toolchain, Mac or phone here. The measurements
the first device run has to take have their own section.

## In short

The stages follow the README's full chain: cameras, dense cloud and names, room shape, room model and
render check, splat, fill, mixed scene.

| Stage | Today | On the phone | State |
|---|---|---|---|
| Camera poses | COLMAP global, then incremental fallbacks; MapAnything on CUDA | ARKit's poses (already in `frames.jsonl`), read back through keyframe anchors, then the vendored camera-only bundle adjustment behind a held-out gate | Recording side written (not built); refinement vendored; adapter to write |
| Live analysis while recording | SegFormer-B0, MoGe-2-S, YOLOE-11m-seg and MobileSAM, on the phone, with Core ML free to use the GPU | The same models on `.cpuAndNeuralEngine` once MLComputePlan shows where each op runs; EdgeTAM in place of MobileSAM | EdgeTAM converter built; its wrapper checked in PyTorch |
| Dense cloud | MoGe-2-L on keyframes (Mac or Colab), scaled to COLMAP; mirror, window and screen planes | MoGe-2-S on keyframes after recording, scaled per frame to ARKit's points (`DepthScale`), mirror planes ported, TSDF-fused over the same poses as everything else | `DepthScale` runs live per frame; the pass and the fusion are to write |
| Objects, outlines, names | Claude's room list steers GroundingDINO; SAM 2.1 per keyframe; SAM 2 video tracking on CUDA | YOLOE boxes, EdgeTAM outlines, identity from `ObjectTracker`'s ARKit/depth voxels; Claude renames and drops tracks from crops | Pieces exist; the post-capture pass is to write |
| Room shape | `shapes.py`, `placement.py`, `room_score.py` on the Mac | Planes from MoGe depth plus ARKit's plane classes, a floor-plan fit, and the silhouette search and room score ported to Swift | To port |
| Room model and render check | `blender_room.py`, rendered from the video's cameras by `room_views.py` in Blender | The room built in SceneKit (the app already uses it for plane outlines and wireframe boxes in the live view) and rendered offscreen for Claude's render check | To write |
| Splat | OpenSplat (AGPL) on a Colab T4 or the Mac's Metal build | The vendored Swift + Metal trainer (Apache-2.0), seeded from the fused cloud | Vendored; adapter to write; untimed on an A15 |
| Surface fill | Big-LaMa on the Mac's CPU or the Colab VM (fill step, object removal, mixed scene) | The same Big-LaMa as Core ML (`SurfaceFill.mlpackage`: 512, fp32 on the GPU, int8 weights) | Converter built; its weights checked in PyTorch; Core ML `--check` needs a Mac. `surface_fill.py` now fills at 512 too |
| Mixed scene | `mixed_scene.py` (Mac or Colab); the scene viewer served to the phone by the paired Mac | Shell textures, pieces and cuboid models built on the phone; the scene viewer bundled in the app | To port, after the splat |
| Viewing and editing | MetalSplatter on the Mac and on the phone (renders sent by the Mac); editing in the Mac app | MetalSplatter for the trainer's PLY; layout editing in the bundled scene viewer; splat editing by sharing the Mac app's editing model, with a touch layer | Splat viewing in the app; splat editing to decide |

Measured here this round:

- **Fill:** Big-LaMa beat MI-GAN-512, the mobile inpainting GAN, in 23 of 24 fills on our floors and
  walls. Run at 512, LaMa fills large holes better than at full size. Footprints gained 3.6 dB and
  unseen blobs 1.0 dB; thin strips lost 0.2-0.3 dB. At full size, big holes came back as a grey
  waffle. Big-LaMa converts to Core ML. With int8 weights the package is 58 MB instead of 205 MB, and
  its own weights, run in PyTorch, moved the fill by −0.05 dB.
- **Outlines and tracking:** our EdgeTAM export wrapper, run in PyTorch before conversion, matches
  EdgeTAM's own predictor (box logits to 3e-5). The fp16 Core ML packages (encoder 10.0 MB, decoder
  12.4 MB) have not been run yet; that needs a Mac. Run with its own video memory on a CPU, EdgeTAM
  carried eight kinds of object through 20 and 48 frames of our clips (details in section 4).
- **Why ARKit's poses:** the two clips COLMAP struggles with solve with the new fallbacks on 4 of 4
  and 3 of 4 fresh runs, but on a knife edge (see the audit). ARKit tracks every frame at capture
  time, with metric scale.

## The stack, stage by stage

### 1. Recording (`Recorder.swift`, `CaptureController.swift`, `SceneRunner.swift`)

**Already written** (commit 90598e8 and its follow-up; not yet built or run; `tools/check_phone_capture.py`
checks every field on a recording, and `docs/device-run-1.md` is the first run, step by step):

- An `ARAnchor` at the camera every 15 written frames, twice a second.
  - ARKit can move anchors when it corrects its map (relocalising, closing a loop), so where they end
    up should be corrected keyframe poses. Nobody has measured whether per-keyframe anchors actually
    capture those corrections, on any phone; the first recordings will show it for the A15.
  - `capture.json`'s `keyframeAnchors` gives each anchor's pose as placed and at the end, and how many
    times it moved. An update that moves it under half a millimetre and turns it under about 0.03° is
    not counted.
  - The vendored code reads back keyframe anchors the same way.
- The motion sensors at 100 Hz: the samples since the last frame ride on each `frames.jsonl` line as
  `imu` (time, bias-corrected gyro, acceleration with gravity). The jump check below does not need
  them. They are there for an inertial bundle adjustment later; the vendored one uses ARKit's
  frame-to-frame motion, not raw IMU.
- `jump: true` on a frame whose pose is more than 15 cm from the last (relocalising). Tracking state
  was already written per frame.
- **Plane classes.** `capture.json`'s `planeClasses` lists every plane ARKit reported while recording,
  with its alignment, raw classification (or "none" and why), extent, and whether ARKit removed it.
  The room map's own `roomMap.planes` writes an unclassified vertical plane as "wall", so the wall hit
  rate (measurement 2) is read from `planeClasses`.

**Still to do:**

- **Compute units.** The live models load with `.all`, falling back to `.cpuAndGPU`
  (`SceneRunner.swift:154`). `.all` lets Core ML put any op on the GPU.
  - Run MLComputePlan on the iPhone 13 first, to see which ops leave the Neural Engine.
  - Then load the models with `.cpuAndNeuralEngine`. Ops the Neural Engine rejects then go to the CPU
    instead of the GPU, which keeps the GPU for ARKit and the camera preview while recording.
- **A live "the view is not shifting" rule** (audit decision 7): the distance moved over the last few
  seconds against the depth in view, both from ARKit. It needs tuning on real app recordings.

### 2. Camera poses (replaces COLMAP and MapAnything)

ARKit's visual-inertial tracking is the pose source. Nothing open-source beats it on a phone:
RTAB-Map's author builds his iOS app on it.

It is not exact on a phone without LiDAR, though. The published figures are for an iPhone 11:

- 0.37 m mean trajectory error on a 6-13 m lab path;
- about 25 cm of drift in a placed hologram after walking about 7 m away and back (43 cm expected
  across many environments).

Rotation probably stays near 1° over about a minute. That is inferred from LaMAR's LiDAR-device
captures, where it averages 3.5° after five minutes, and is unmeasured on an A15. So refinement is
likely needed.

Refinement can also do harm. In a 2026 preprint (arXiv 2608.21008), COLMAP-style
triangulate-then-adjust made ARKit poses slightly worse in 15 of 15 ScanNet++ rooms: 0.55° / 19 mm
became 0.74° / 21 mm, against a COLMAP-derived reference. Those were laser-registered poses from LiDAR
iPhones. Whether the same holds for raw iPhone 13 poses is untested.

**First choice: the vendored camera-only refinement.** This is `CameraOnlyPoseRefinement` and
`CameraOnlyTracker` in `Capture/CameraOnlyTracker.swift`, solved by `BundleAdjuster.swift`. It uses
only Apple frameworks: no Ceres, no COLMAP.

- **Tracking:** Shi-Tomasi corners are tracked keyframe to keyframe with Lucas-Kanade, starting where
  ARKit's motion predicts. A step is kept only if:
  - tracking back returns within 0.7 px;
  - the patches correlate at 0.8 or better;
  - the match lies within 4 px of the epipolar line.

  A failed step ends the track.
- **The solve:** the tracks are triangulated from ARKit's poses, then the poses are bundle-adjusted.
  That is the triangulate-then-adjust family that failed in the study above. What differs is a tight
  ARKit motion prior (0.3 mm and 0.01° per step) and a gate.
- **The gate:** the result is applied only if the held-out tracks' median reprojection improves by at
  least 3%. That gate shares the solver's matcher, and its author found it can pass a solve that
  makes photos line up worse. The camera-only path has no photo check, so the held-out splat PSNR
  (measurement 3) is the real test.
- **What it fixes:** by its author's account, it tightens frame-to-frame and multi-view consistency
  but does not correct slow drift. Its measured gains are millimetres to a centimetre, on two
  iPhone 17 Pro LiDAR scans replayed without depth, timed on a Mac.
- **What it does not include:** the vendored `OfflinePoseRefinement.swift` and
  `LoopClosureRefiner.swift` are its LiDAR path and do not run without depth. Without LiDAR there is
  no loop closure in the vendored code. Drift of tens of centimetres has to come out through ARKit's
  own corrections, read back through the keyframe anchors. Nothing shows that the fallback below
  removes slow drift either. If ARKit's corrections are not enough, evaluate a pose-graph and
  loop-closure layer over ARKit's odometry, such as RTAB-Map (BSD-3). Its iOS app so far requires
  LiDAR.

**The fallback** is for when the gate keeps rejecting, or the result is worse than COLMAP's. Rebuild the triangulation-free formulation of arXiv 2608.21008 with Ceres (BSD-3):

- per-observation ray depth;
- symmetric two-view residuals;
- an orientation prior;
- a fallback to ARKit's pose.

Its code is not public; see the licence asks.

**Intrinsics:** the vendored solve keeps ARKit's pinhole intrinsics fixed and models no lens
distortion. Calibrate fx, fy, cx, cy, k1 and k2 once per phone model, offline (COLMAP on the Mac, or
`AVCameraCalibrationData`), if the fallback needs them.

**Measure before trusting any of it:**

- Record 10-20 rooms with the app, and run each through today's COLMAP chain on the Mac as the
  reference.
- Compare the trajectory error (Sim(3)-aligned), tape-measured distances, and the splat's PSNR on
  held-out views.
- Ship ARKit-only poses if the PSNR gap to COLMAP is under about 0.5 dB.

What this removes: COLMAP and MapAnything from the capture path. COLMAP's bundled SiftGPU is licensed
for non-profit use only, so this also settles that question for the product.

### 3. Dense cloud (replaces `densify.py`'s MoGe-2-L pass)

**Depth per keyframe.** After recording, once the live models are released, run MoGe-2-S on the
keyframes. It is already in the app as `RoomMetricDepth` (MIT).

- Take the field of view from ARKit's intrinsics.
- Fit each keyframe's scale and shift to ARKit's feature points with the existing `DepthScale`.
- A frame without a sound fit falls back to MoGe's own metric scale. That covers frames with fewer
  than 12 usable points, and fits that `DepthScale.correction` rejects. On bare walls that scale
  measured anywhere from a fifth of the truth to twice it. So give those frames little or no weight
  in the fusion, unless their scale can be recovered (next point).
- **Bare walls** are where per-frame scale fails. In the live app, borrowing a neighbouring frame's
  correction directly measured worse than none. Recovering scale through the fused geometry, from
  frames that have points, is a different method and untested. Measure it on the reference rooms
  (measurement 4) before relying on it, and use the wall/floor masks with ARKit's planes.

**Mirrors, windows and screens.** Port `mirrors.py` into this pass. Each such outline takes the depth
of the plane through the strip of frame or wall around it, as it does today. Fused as they are, their
depth is the reflection's or the view's, beyond the surface. The TSDF would grow a phantom room behind
the glass and carve away the mirror, the window's wall or the wardrobe front.

**Fusion.** Fuse with a TSDF at 2-3 cm voxels over the same poses as everything else: ARKit's,
refined where the gate accepted the refinement.

- The CPU references are the vendored `SurfaceTSDF` and swift-tsdf. `SurfaceTSDF` is Apache-2.0,
  uses sparse 8³ blocks, defaults to 2 cm and has its own tests. swift-tsdf is MIT but small and
  unvetted, and has only run on LiDAR depth (30-90 s a scan on an iPhone 12 Pro).
- A Metal kernel is the goal.
- The vendored `RefusionEngine` re-fuses stored keyframe depth over anchor-corrected poses and wraps
  `SurfaceTSDF`. It is the alternative to compare against, once MoGe depth is written in its format
  and its LiDAR noise weighting is replaced.

**Labels carry through the fusion.** Each voxel keeps a vote of the object ids from EdgeTAM's
keyframe masks. The splat's cut into pieces (section 8) uses whether each point lies on a detected
object, as the dense cloud records today, to drop paint left hanging off the walls. Without it, the
pieces are cut by their boxes alone.

**Memory.** A larger post-capture MoGe input needs its own fixed-shape `.mlpackage`. At about 2,500
tokens, that is 812x602 at 4:3. At the default memory limit a 4 GB phone risks killing the pass, and
the increased-memory-limit entitlement may do nothing on the iPhone 13 (section 7). So size the input
to fit the default limit, and measure peak memory in Instruments with and without the entitlement
(measurement 4).

**Fix the yardstick first** (measurement 4). The project's depth error figures were measured against
a reconstruction scaled by MoGe-2-L's own metric scale. So they show MoGe-S agreeing with MoGe-L, not
true metres. Re-measure two rooms against tape-measured furniture and ARKit's metric poses, with and
without `DepthScale`.

### 4. Objects, outlines and names (replaces GroundingDINO, SAM 2.1 and `tracking.py`)

**Boxes:** the YOLOE model the app already runs. `RoomObjects` has 71 classes baked from
`object-classes.json`: 70 room things plus person. See the licence asks: YOLOE needs an Ultralytics
licence, and its class embeddings come from Apple's MobileCLIP. YOLOE-26 is the upgrade once that is
settled (audit decision 4).

**Outlines: EdgeTAM** (`scripts/convert_refiner_edgetam.py`, `REFINER=edgetam`), in place of
MobileSAM. Its encoder family (RepViT) ran about eight times faster than MobileSAM on an A15: EdgeSAM
38.7 FPS against 4.9 on an iPhone 14.

Our export fixes three things that Meta's Core ML export leaves out:

- ImageNet normalisation in the graph;
- the box as corner points labelled 2 and 3 (Meta's prompt-encoder export ignores its boxes input);
- the single-mask output (Meta's traced decoder always returns the three-mask output).

(Commit ec6bef8's message also listed the high-resolution maps; Meta's export already outputs them.)
The app also gives EdgeTAM a point on the thing, from the detector's own mask. On a framed TV, the
box alone outlined the wall panel; box plus point outlined the screen.

**Identity through the video comes from geometry, not SAM 2's memory.** `ObjectTracker` already gives
each object one identity from ARKit poses and depth voxels. After recording:

- every 3rd to 6th frame gets one EdgeTAM encoder pass;
- each tracked object is projected to a box, with a depth test for whether it is in view;
- one decoder pass per visible object outlines it.

This drops SAM 2's forward-and-back propagation and its duplicate merging. A walkthrough study (3AM,
arXiv 2601.08831) found SAM 2's appearance memory loses objects on ScanNet++ room scans: tracking
recall 0.02, against 0.56 for its own geometry-aware method. How well our geometry holds identity on
our recordings is still to be measured. `ObjectTracker`'s own notes record one failure: turning on
the spot made one monitor into three along the line of sight.

**Names.** Today Claude writes the room's object list from ten frames, and that list is what
GroundingDINO searches for. It includes room-specific names and "elsewhere" exclusions. On the phone
YOLOE's 71 classes are fixed:

- keyframe crops of tracked objects go to Claude, which renames them and assigns roles;
- Claude can also drop tracks, and the "elsewhere" exclusions are applied to tracks afterwards;
- Claude cannot add things YOLOE missed. That is the cost of a fixed vocabulary on the phone.

**Measured here, on this cloud machine's 4-core CPU.** EdgeTAM's video predictor took one box per
object and propagated it forwards and back with its appearance memory, which the plan does not use.

- In the 20-frame clip it held the TV, sofa, rug and TV stand.
- In the 48-frame clip it held the dining table (which dropped out for a frame), sofa, plant, dresser
  and bed. Its TV track, prompted by a box alone, outlined the frame and wall panel around the TV
  instead of the screen.
- It took about 2.6 s a frame for four objects and 4.2 s for six, including memory attention and the
  backward pass.

This shows the weights segment our furniture. It does not show that the planned per-frame,
box-prompted pass works, or how fast it runs on a phone.

### 5. Room shape (ports `shapes.py`, `placement.py` and `room_score.py`)

**Planes.** Fit planes in each keyframe's MoGe-2-S depth with a grid-cell plane fitter (the CAPE
method, MIT). Tag them wall, floor or ceiling from the segmenter and from ARKit's plane classes.
`.wall`, `.floor`, `.door` and `.window` are available on A12 and later without LiDAR, but how often
and how accurately they fire on an iPhone 13 is unpublished (measurement 2).

**Floor plan.** Fuse the planes in world space over the same poses as the dense cloud, rasterise them
to a 3-5 cm floor-plan grid, and fit lines. Snap the lines to the room's dominant direction (`RoomMap`
has it), and close the outline using the camera path as seeds.

**Boxes.** Keep `ObjectTracker`'s mask-plus-depth boxes, and port `placement.py`'s silhouette-IoU
search to Swift, fed by EdgeTAM keyframe masks. Merge duplicates, and never let a box cross a wall.

**The room score.** Port `room_score.py`, so a measured check still rolls back any edit that makes the
room worse.

**Not adopted:**

- RoomFormer did worse than a non-learned baseline on data it was not trained on, and its good numbers
  need ScanNet-derived training data, which is non-commercial.
- SpatialLM uses a non-commercial encoder and needs CUDA.
- BoxerNet is non-commercial (Meta). It falls from 0.412 to 0.126 mAP on CA-1M without depth, and is
  untested with monocular depth such as MoGe-2.

On LiDAR iPhones, RoomPlan inside our own ARSession is an optional upgrade.

### 6. Room model and render check (replaces `blender_room.py` and `room_views.py`)

Blender does not run on iOS. Today Blender builds the room from walls and boxes and renders it from
the video's own cameras. Claude grades those renders against the real frames, and a structural
problem sends the room back to its structure review.

The app already uses SceneKit, for plane outlines and wireframe boxes in the live view; it builds no
room model yet. On the phone:

- build the room's walls and floor, and furniture from simple parts (a bed is base, mattress and
  pillow; a wardrobe has doors), as SceneKit geometry;
- keep the render check's colour conventions (sand floor, grey walls, blue-grey furniture, beige for
  walls nobody filmed), or change its prompt to match;
- render it offscreen (`SCNRenderer`) from an angle, from above, and from keyframe poses beside their
  frames, as today, and send those to Claude.

### 7. Splat (replaces OpenSplat)

**The trainer:** `third_party/arkit-3dgs-scanner`, the last Apache-2.0 commit (d5d8e31) of Kuo
Feng-Yuan's Swift + Metal trainer. `OASIS-PROVENANCE.md` there records why this commit and what we
owe. The trainer has:

- MRNF densification, a Mip filter and a PPISP colour model;
- memory plans that include 4 GB phones (a 900 MB ceiling under 5 GB);
- checkpoint and resume;
- SOG export, plus a PLY writer (`GaussianExport.writePLY`) that the app itself never calls.

It uses only Apple frameworks.

**The adapter** (to write):

- keyframes as JPEG;
- our column-major transform transposed to its row-major one;
- the fused cloud written as the scan's seed cloud (`points.ply`).

The trainer seeds one round Gaussian per point and, by default, jitters seeds by up to 3 cm. Seeding
flat discs on the surfaces would change the trainer itself (`GaussianModel.initialize` takes no
orientation). Files we change must be marked as changed (Apache-2.0).

**Why the seed matters.** Under short phone budgets a dense prior matters more than optimiser tricks.
PocketGS measured 21.15 dB without its multi-view-stereo prior and 23.67 dB with it (500 iterations,
iPhone 15). How much our MoGe-fused cloud gains at 4,000 iterations is still to be measured.

**Settings on 4 GB phones:** Quick (4,000 iterations, 960 px, SH degree 2), capped at 200-300k
Gaussians. Request the increased-memory-limit entitlement, but plan the budget as if it does nothing
on the iPhone 13: the trainer's author notes it takes effect only on Pro and larger-memory iPhones.
Run in the foreground with checkpoints; background GPU access on an A15 is unverified.

**Time, untested.** No A15 measurement exists for any open trainer.

- The author's iPhone 17 Pro, on the build before his speed-up (the one closest to our snapshot),
  took 87-95 ms a step at a 600k cap, throttled.
- Scaled by the A15's 2.9 times lower Metal score, Quick's 4,000 steps take about 17-18 minutes at
  that cap. At 200-300k Gaussians a step costs less, by an unmeasured amount.
- Today's Colab T4 takes 4-13 minutes.
- A short preview run (500-1,000 iterations on the dense prior) can show something within minutes.

**Viewing.** MetalSplatter reads PLY, SPZ and `.splat`, not the trainer's SOG.

- `writePLY` writes in a Y-down COLMAP frame that the viewer must rotate 180° about X.
- Its colours are before the PPISP correction, which lives in a separate `ppisp.json`.
- SPZ means converting that PLY with MetalSplatter's SplatIO.

**Not adopted:**

- msplat (Apache-2.0): its host code shares OpenSplat's AGPL names and signatures without credit; it
  stays a speed reference only.
- Feed-forward splats: PocketSplat needs 2.6-2.9 GB, and Apple's SHARP is research-only.

**Brush** (Apache-2.0, Rust/wgpu) is the second track and the only route to one trainer on iOS and
Android. On iOS it runs only through john-rocky's ios-main patches. Those peaked at 3.1 GB on an
iPhone 17 Pro with no splat cap (1.2 GB on a masked capture), so a 4 GB phone needs `max_splats` at
about 150-250k.

**Off the phone:** the long splat (Colab or Mac) and Spirula Studio (GPL-3.0, Mac) stay research
tools. This also
retires OpenSplat (AGPL-3.0) from the product, and with it audit decision 1.

### 8. Surface fill and the mixed scene

The fill keeps `surface_fill`'s design: a rectified photo of each floor and wall, LaMa for what no
frame saw, and flat blobs in the splat. On the phone it runs `SurfaceFill.mlpackage`, built by
`apps/OasisCapture/scripts/convert_fill_lama.py`.

**LaMa against MI-GAN.** Measured here on patches of floor and wall from 8 of our clips: two wood
floors, a patterned rug, rug and tile, plain and shiplap walls, and a kitchen wall. Each patch got
three holes of known content: a furniture footprint, a region no frame saw, and thin strips. Scores
are PSNR inside the hole; times are on this machine's CPU.

| | Footprint | Unseen blob | Strips | CPU time |
|---|---|---|---|---|
| Big-LaMa (Apache-2.0, 51 M parameters) | 28.71 dB | 32.46 dB | 33.86 dB | about 1.5 s |
| MI-GAN-512 (MIT, 6 M parameters) | 24.24 dB | 27.49 dB | 31.37 dB | about 1.0 s |

LaMa was higher in 23 of 24 fills. The eight unseen-blob fills were also looked at. MI-GAN drew a
streak across a wood floor, scribbles on a rug, and an object on a kitchen wall that was never there.
LaMa's blob fills were clean. So there is no MI-GAN tier. OpenCV's FSR (Apache-2.0, no weights)
remains an option for tiny holes.

**Fill at 512, not at the photo's size.** Big-LaMa's Fourier units become fixed-size DFT matrix
products in Core ML, so the model's shape cannot vary (enumerated shapes fail to convert). Testing
whether that costs anything showed the opposite. The test used 1024-pixel crops of the same clips;
one came from a 720p clip and was upscaled to 1024.

| | Footprint | Unseen blob | Strips | CPU time |
|---|---|---|---|---|
| LaMa at 1024 (what `surface_fill` did) | 19.69 dB | 23.51 dB | 27.48 dB | 6.8 s |
| LaMa at 512, the fill scaled back up | 23.25 dB | 24.52 dB | 27.17 dB | 1.5 s |

At 1024 a footprint-sized hole came back as a grey, waffle-textured patch. At 512:

- one wood floor's planks and the shiplap lines carried on through it;
- on the other wood floor the fill stretched the nearby rug across the hole;
- 512 was better in 7 of 8 footprint holes;
- per patch it was worse on the kitchen-wall footprint and one plain-wall blob.

`surface_fill.inpaint` now works at 512 on the Mac too. It box-filters the photo down, counts a pixel
as unknown if any of it was, and puts the fill back only where the photo was unknown. Rerun through
that code:

- footprints: 19.69 to 23.26 dB;
- blobs: 23.51 to 24.49 dB;
- strips: 27.47 to 27.27 dB.

A surface photo is 5 mm a pixel, and the fill reaches the splat as blobs 12 mm apart. So for surfaces
up to about 6 m across, filling at 512 (under 12 mm a pixel) should lose nothing the blobs could
show. That is reasoning, not tested on rectified photos. Larger floors are filled coarser than the
blob grid, and a mixed scene's textures are softer where filled, which beats the waffle.

**The Core ML model** is 512 by 512, with fp32 compute on the GPU (`CPU_AND_GPU`). The Neural Engine
runs fp16 only, and LaMa breaks in fp16: mlx-lama-swift found fp16 collapses the Fourier
convolutions, and CoreMLaMa's author has had no working fp16 Neural Engine reports.

The weights are int8 for the plain convolutions, except the first and the last:

- The package is 58 MB instead of 205 MB.
- Its own weights, read back from the package and run in PyTorch, moved the fill by −0.05 dB on
  average and −0.61 dB at worst over the 24 fills.
- Quantising the first and last convolutions as well cost 11 dB in a PyTorch simulation.
- Quantising the three transposed convolutions doubled the loss.
- The DFT matrices must stay fp32: their sine matrix has an all-zero row, which per-channel int8
  scaling turns into NaN.

The phone pads a shrunk photo to the 512 square; `surface_fill.inpaint` pads only to a multiple of 8.

`--check` on a Mac runs the package against PyTorch; the Core ML runtime needs macOS. The model is not
in `build.sh` or `project.yml` yet: the app gets a fill stage when it gets the trainer. One report
puts a different Core ML LaMa at about 2 s for 800x800 on an iPhone 13 Pro, with compute unit and
precision not stated. If the iPhone 13 is close, a dozen fills (floor, walls, flat furniture faces)
take under half a minute. That is an estimate until measurement 6.

**The mixed scene** (`mixed_scene.py`, the last step today) moves after the splat. It has five parts:

- **Shell textures:** `surface_fill`'s frame projection, with SurfaceFill for what no frame saw.
- **Pieces:** the splat cut by the measured boxes and by the fused cloud's object labels (section 3).
- **Cuboid models:** their sides photographed at 2.5 mm a texel from full-size frames. On a 4 GB phone
  that means decoding one 4K frame at a time from the video.
- **Review renders for Claude:** from the trainer's own Metal renderer, or MetalSplatter offscreen.
- **The scene viewer** (three.js and Spark): served today to the phone by the paired Mac. Bundle it in
  the app's web view instead.

**Off the phone:** clean object meshes from images (TRELLIS, SAM 3D Objects) need a datacentre GPU.
On the phone, pieces get the cuboid models only.

### 9. Viewing and editing

The splat shows in MetalSplatter, already in the app. Today the room renders and the scene viewer
reach the phone from the paired Mac, and splat editing (remove, add, recolour, resize) is in the Mac
app.

- **Layout editing comes with section 8.** The scene viewer is itself an editor: moving, turning,
  hiding and swapping pieces come to the phone with the bundled viewer.
- **Splat editing.** The Mac app's editing model (`apps/SplatViewer/Sources/Editing`: selection,
  furniture, surface patches) is plain Swift (Foundation, simd, SplatIO) and can be shared with the
  phone. Its input layer (`Viewer/SceneEditing.swift`) is AppKit mouse-and-keyboard code, and the
  phone needs a touch version.
- **Filling what a removed object hid.** The Mac app's Swift editing covers the gap with thin
  Gaussians (`Editing/SurfacePatch.swift`). `splat_edit.py remove` (Python, on the Mac) fills it with
  Big-LaMa through `surface_fill`, like the fill step. A phone port would keep the patches or call
  SurfaceFill.

Whether splat editing moves to the phone, or stays a Mac feature for now, is the owner's call.

### 10. Claude and the agent

**Claude is called from the Mac today**, by `pipeline/advisor.py` through the `claude` CLI login or
an API key. Colab jobs relay their questions back to the Mac. Its calls:

- how to retry a weak solve, and whether a joined MapAnything walk is sound (both go away with ARKit
  poses);
- choosing frames;
- naming the room's objects;
- checking detection boxes;
- the structure review;
- where a dropped piece stands, when no keyframe saw it;
- the render check;
- each splat's opening view;
- keeping or rejecting each fill;
- picking the best splat;
- the scene review;
- the capture advice.

**The Mac stays the Claude gateway** (the owner's decision, 14 October 2026). The phone cannot ship
an API key, and no proxy will hold one.

- The phone does the computing, and sends each review's images and question to the Mac's station, as
  it sends recordings today: over the local network, or through the Render relay, which the Mac's
  cloud worker answers.
- The Mac's `pipeline/advisor.py` asks Claude with its own login and sends the answer back.
- With no Mac reachable, the reviews wait. Or the agent goes on with its numeric rules, as
  `advisor.py` already does when Claude is offline.

Nothing on the phone asks Claude yet. This lands with the first phone stage that needs a review.

**What the phone must produce for those reviews:**

- contact sheets of frames;
- keyframes with every detection box drawn (the label review);
- the plan of the room's candidate walls and boxes, and the per-object frame sheet (the structure
  review and where a piece stands);
- room renders (section 6);
- six candidate opening views rendered from the splat;
- splat renders at trained and held-out views;
- before-and-after renders of each fill;
- the capture map's metrics (`capture-map.json`) for the capture advice. The shot list is computed
  for the report, not shown to Claude.

**The checks that overrule Claude** port with them:

- the room score;
- the mask check that downgrades a render-check "misplaced" claim when the piece's own masks agree
  with its box;
- the guard on boxes the structure review adds;
- the splat guard (SSIM 0.01 and PSNR 0.3 dB);
- the fill's pixel-change check.
 The agent's own logic in `pipeline/agent.py` (which
reviews run, retries, rollbacks and the report) ports to Swift.

### 11. Android (later)

The same stages apply:

- ARCore for poses, and ARCore's Depth API where the phone has it;
- ExecuTorch 1.5, with XNNPACK everywhere and the QNN backend on Snapdragon;
- LiteRT as the per-model fallback;
- Brush for the splat.

Every Android figure found was for flagships, so the mid-range target has to be measured. Not
started.

### Left off the phone

- COLMAP's optional dense multi-view stereo (`reconstruct.py --dense`, CUDA), which the agent never
  runs. (MoGe-2-S with TSDF fusion replaces `densify.py`'s MoGe-2-L pass, not this.)
- MapAnything.
- The long splat (Colab or Mac) and Spirula Studio (Mac).
- Image-to-3D object meshes: TRELLIS (`tools/object_mesh.py`) is a Colab GPU job, and SAM 3D Objects
  is not set up (its weights are gated).

These stay research tools off the phone.

## Measurements the first iPhone 13 run has to take

1. **Recording and the live chain:**
   - The recording changes build, and `imu`, `jump` and `keyframeAnchors` come out as described.
   - Each live model's latency and memory, with the others running.
   - Where each op runs (MLComputePlan), before and after `.cpuAndNeuralEngine`.
   - EdgeTAM against MobileSAM.
   - Whether `BGTaskScheduler.supportedResources` includes the GPU on an A15.
2. **ARKit:**
   - Does the A15 move keyframe anchors?
   - Trajectory error and tape distances against COLMAP on 10-20 rooms.
   - The hit rate of ARKit's plane classes (wall, floor, door, window) on those rooms.
3. **Camera-only refinement:** time, heat and battery; how often its gate accepts; and held-out splat
   PSNR against COLMAP poses.
4. **Depth:**
   - The yardstick: MoGe-2-S with and without `DepthScale`, against tape-measured furniture and
     ARKit's metric poses in two rooms.
   - The post-capture pass: time per keyframe, and peak memory with and without the entitlement.
   - Bare walls: depth error on frames with no sound `DepthScale` fit, with scale recovered through
     the fused geometry against MoGe's own scale.
   - TSDF fusion: time and memory at 2-3 cm.
5. **Outlines and room shape:**
   - The post-capture EdgeTAM pass: time per frame and per object, and peak memory.
   - Identity held across the recording.
   - The room-shape port against `shapes.py` and the room score on the reference rooms.
6. **Trainer and fill:**
   - Quick at a 200-300k cap: minutes and peak memory, with and without the entitlement, and whether
     it finishes without thermal pauses. The vendored snapshot has no training benchmark;
     `--benchmark-training` came later, under the non-commercial licence. So we write our own timing
     harness.
   - SurfaceFill: `--check` on a Mac, then time and peak memory per 512 fill on the A15's GPU.
7. **The exposure lock** from the audit: `exposure` in `frames.jsonl` should be constant within a
   recording.

## Licences: what to ask for

### Blocking a commercial release now (models the app already ships)

1. **SegFormer-B0** (`RoomSegmentation`, the wall/floor/ceiling labels) is blocked twice over.
   NVIDIA's licence allows research or evaluation only, and its ADE20K training data is
   non-commercial too. Swapping in another ADE20K-trained model does not fix it.
   - *Ask:* NVIDIA Research licensing (nvidia.com/en-us/research/inquiries) for commercial terms, and
     MIT CSAIL (contact on ade20k.csail.mit.edu/terms) about ADE20K-trained models.
   - *Or:* retrain a small segmenter on COCO-Stuff and Hypersim, with our own labelled captures. That
     is a one-off GPU training job, not part of reconstruction.
     - COCO-Stuff's annotations are CC BY 4.0; counsel should check the Flickr image terms.
     - Hypersim is CC BY-SA 3.0; counsel should advise whether share-alike reaches the weights.
2. **YOLOE-11m-seg** (`RoomObjects`): the code and weights are AGPL-3.0.
   - *Ask:* Ultralytics Enterprise (ultralytics.com/license). Get it in writing that the licence
     covers:
     - YOLOE-11 (trained by THU-MIG) and YOLOE-26;
     - our own fine-tunes and retrained models (a SigLIP 2 text stage, a YOLO26-sem surface
       segmenter);
     - iOS and Android redistribution of the exported models.
3. **MobileCLIP**, the source of YOLOE's baked class embeddings. Apple's ML Research Model terms allow
   research only and exclude product development, so computing the embeddings once at build time is
   not a way round it.
   - *Ask:* Apple Legal, or the ml-mobileclip maintainers (the repository lists no contact), for a
     commercial licence or a written statement covering embeddings computed with the model. Expect a
     no.
   - Also ask Ultralytics whether it warrants or indemnifies the MobileCLIP-derived embeddings. It
     cannot sublicense Apple's weights.
   - *If refused*, one of two routes:
     - retrain YOLOE's text stage with a SigLIP 2 text tower (Apache-2.0);
     - or drop the open vocabulary: fine-tune a closed-set detector on our room classes, with Claude
       doing the naming. D-FINE has Apache-2.0 code and needs no paid licence, but its Objects365
       checkpoints wait on item 5.
4. **MoGe-2's training data.** The weights are MIT, but the training sets include non-commercial ones
   (ScanNet++, Waymo Open, Argoverse 2).
   - *Ask:* counsel to review. Optionally ask the microsoft/MoGe maintainers to confirm commercial use
     of the released weights.
5. **Detector training data.** YOLOE-11m-seg (shipped now), YOLOE-26, a SigLIP 2-retrained YOLOE and
   D-FINE's Objects365 checkpoints are trained on Objects365, GQA (Visual Genome images) and
   Flickr30k Entities.
   - *Ask:* counsel to review those terms before a commercial release, and ask Ultralytics whether it
     warrants or indemnifies them (item 2).
   - Add the CC BY attributions for Objects365 to the app's notices.

### For the new on-phone stages

6. **The splat trainer: nothing to buy.** The vendored snapshot is Apache-2.0.
   - Keep `LICENSE` and `NOTICE`, mark any file we change as changed, and credit Kuo Feng-Yuan in the
     app's acknowledgements.
   - *Optional ask:* a commercial licence for his later, PolyForm Noncommercial work (1.39x faster
     training steps, a training benchmark), through github.com/KuoFengYuan.
7. **EdgeTAM and LaMa: nothing to buy.** Both are Apache-2.0.
   - Their licence texts and any NOTICE files go in the acknowledgements, and files we change (our
     EdgeTAM export wrapper) say so.
   - Credit Microsoft MoGe-2 (MIT) and DINOv2 (Apache-2.0) there too.
8. **Places2 training data.** Big-LaMa (like MI-GAN and AOT-GAN) is licensed Apache-2.0, but it was
   trained on Places2, whose terms could not be read here. Whether those terms restrict models trained
   on it is open.
   - *Ask:* counsel to review places2.csail.mit.edu's terms. If they are unclear, ask the MIT CSAIL
     Places team for a written statement on commercial use of models trained on it.
9. **Apple entitlements**, in the paid developer team (Signing & Capabilities):
   - Increased Memory Limit (`com.apple.developer.kernel.increased-memory-limit`), possibly ineffective
     on non-Pro iPhones (section 7).
   - Background GPU Access, if training should continue in the background. Whether the A15 grants it
     is measurement 1.
10. **arXiv 2608.21008's refinement**, only if the vendored refinement fails its measurements. Ask the
    author, Nikolaos Kyriazis (address in the paper), for the code under BSD or Apache, or for
    written permission to reimplement it.
11. **A weight mirror** (access, not a licence). This cloud environment cannot reach Hugging Face or
    Apple's model CDN. Someone with access should put these in a GitHub release of our own, with
    their licence files and SHA-256 hashes:
    - the MoGe-2 and SegFormer weights, and anything else Hugging Face-only that we adopt;
    - a copy of `big-lama.pt`, which today sits on a third party's GitHub release.
12. **A Claude proxy** (section 10), if the owner wants the phone-only path. This needs no licence,
    but it is a server-side component holding the API key.

### Settled by moving to the phone

- **OpenSplat** (AGPL-3.0) leaves the product with the vendored trainer.
- **COLMAP's bundled SiftGPU** (non-profit only) leaves the capture path with ARKit poses. If COLMAP
  stays in the Mac tools for commercial work, build it with GPU SIFT off.

### For Android, later

- A Qualcomm AI Hub account (free) for Snapdragon NPU exports.
- A read of the QAIRT and AI Hub model licences before shipping their runtime libraries.
- ARCore's terms and its privacy disclosure.

### Avoid

These have no commercial path or are non-commercial:

- SHARP (Apple)
- EdgeSAM
- UniDepth
- Depth Anything 3 Large and Giant
- MVSAnywhere and SimpleRecon (Niantic)
- BoxerNet (Meta)
- the GPL VIO and SLAM systems (ORB-SLAM3, VINS, OpenVINS)

Datasets with non-commercial terms (ScanNet, Structured3D, CA-1M, ADE20K) need counsel's view before
any model trained on them ships.

## Order of work

1. **On a Mac with the iPhone 13**, as `docs/device-run-1.md` sets out command by command:
   - build the app, which compiles the recording changes for the first time;
   - record three test captures and check them with `tools/check_phone_capture.py`;
   - take measurements 1 and 7;
   - build with `REFINER=edgetam` and run its `--check --coreml` (the fp16 packages against EdgeTAM's
     own predictor);
   - run SurfaceFill's `--check`, and time it on the phone in Xcode's Performance report;
   - then switch the compute units.
2. **Reference rooms:** record 10-20 rooms with the app, and run the Mac pipeline on them as the
   reference. Take measurement 2 and measurement 4's yardstick.
3. **Poses and trainer:** write the camera-only refinement's adapter (our `frames.jsonl` and keyframe
   anchors in, refined poses out), the trainer's adapter and a timing harness. Then take measurements
   3 and 6 on those rooms.
4. **The post-capture pass:** MoGe-2-S on keyframes, mirror planes, TSDF fusion with labels, and
   EdgeTAM outlines on tracked objects. Then take the rest of measurement 4, and measurement 5's
   EdgeTAM pass and identity.
5. **Port the rest:**
   - room shape and score, then measurement 5's room-shape comparison;
   - the SceneKit room model and render check;
   - the fill with `SurfaceFill.mlpackage`;
   - the review images and the agent's logic;
   - then the mixed scene.
6. **In parallel, the licence asks.** Start with these, which decide whether the live models can ship
   at all:
   - Ultralytics;
   - Apple (MobileCLIP);
   - NVIDIA and MIT CSAIL (SegFormer and its ADE20K data);
   - counsel on MoGe-2's training data.

   The owner has decided the Claude gateway: the Mac (section 10). Whether splat editing moves to
   the phone (section 9) is still open.
