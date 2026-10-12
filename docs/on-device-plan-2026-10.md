# Reconstruction on the phone: the plan (October 2026)

The owner's constraint: reconstruction must not need a GPU server, Colab or CUDA. Everything that
turns a capture into a room has to run on the phone, built from open-source parts that a commercial
app can ship. The owner offered to obtain whatever licences that takes; the asks are at the end.

The floor device is the **iPhone 13**: A15, 4 GB of memory, no LiDAR. Claude stays in the loop over
the network (naming things, judging captures). That is judgement, not compute, and no GPU anywhere
is involved. One-time work also stays off the phone: converting models (a Mac, `build.sh`) and
any retraining a licence fix needs.

Where the phone's own GPU is used: the live analysis while recording should move to the Neural
Engine and CPU, so recording stays cool and smooth (step 1 below). The stages after capture may use
the GPU, because the splat trainer and LaMa have no other place to run. Every open on-phone splat
trainer found is GPU-only.

This plan rests on a per-layer search of the field (seven layers, each checked by a second pass
that tried to refute it), on hands-on tests of the parts that could be run here, and on the earlier
audit (`docs/audit-prompts-and-stack-2026-10.md`). Nothing below has run on an iPhone yet: no
Swift toolchain or phone was available. The measurements the first device run has to take are
listed in their own section.

## In short

| Stage | Today | On the phone | State |
|---|---|---|---|
| Camera poses (stage 1) | COLMAP global, then incremental fallbacks; MapAnything on CUDA | ARKit's poses, already in `frames.jsonl`, refined by the vendored camera-only bundle adjustment, kept only when held-out tracks improve | Code vendored (`third_party/arkit-3dgs-scanner`); adapter to write |
| Live analysis while recording | SegFormer-B0, MoGe-2-S, YOLOE-11m-seg and MobileSAM, on the phone, with Core ML free to use the GPU | The same models, moved to `.cpuAndNeuralEngine` once MLComputePlan shows where each op runs. EdgeTAM in place of MobileSAM | EdgeTAM converter built and checked (`REFINER=edgetam`) |
| Dense cloud (stage 2) | MoGe-2-L on keyframes (Mac or Colab), scaled to COLMAP | MoGe-2-S on keyframes after recording, scaled per frame to ARKit's points (`DepthScale`), fused over ARKit poses | `DepthFusion` runs live already; the post-capture pass is to write |
| Objects and outlines (stage 2) | GroundingDINO and SAM 2.1 per keyframe, SAM 2 video tracking on CUDA | YOLOE boxes, EdgeTAM outlines, identity from `ObjectTracker`'s ARKit/depth voxels; Claude names objects from crops | Pieces exist; post-capture pass to write |
| Room shape (stage 3) | `shapes.py`, `placement.py` on the Mac | Planes from MoGe depth plus ARKit's wall/floor/door/window classes, a floor-plan fit, and `placement.py`'s silhouette search ported to Swift | To port |
| Splat (stage 5) | OpenSplat (AGPL) on a Colab T4 | The vendored Swift + Metal trainer (Apache-2.0), seeded from the fused cloud | Vendored; adapter to write; untimed on an A15 |
| Surface fill | Big-LaMa on the Mac's CPU | The same Big-LaMa as Core ML (`SurfaceFill.mlpackage`, 512, fp32 on the GPU, int8 weights) | Converter built and checked; `surface_fill.py` now fills at 512 too |
| Viewing | MetalSplatter (Mac app, and the phone's review screen) | MetalSplatter, from the trainer's PLY export | In the app |

Measured here this round:

- **Fill:** Big-LaMa beat MI-GAN-512, the mobile inpainting GAN, in 23 of 24 fills on our floors and
  walls. Run at 512, LaMa fills large holes better than at full size: footprints +3.6 dB, unseen
  blobs +1.0 dB, thin strips −0.2 dB. At full size, big holes came back as a grey waffle. Big-LaMa
  converts to Core ML, and int8 weights cut it from 196 MB to 58 MB for −0.05 dB.
- **Outlines and tracking:** EdgeTAM (Apache-2.0, 13.9 M parameters) carried seven kinds of furniture
  through 20 and 48 frames on our clips. Our Core ML wrapper matches its own predictor (box logits to
  3e-5). Encoder 10.0 MB, decoder 12.4 MB.
- **Why ARKit's poses:** the two clips COLMAP struggles with solve with the new fallbacks on 4 of 4
  and 3 of 4 fresh runs, but on a knife edge (see the audit). ARKit tracks every frame at capture
  time, with metric scale.

## The stack, stage by stage

### 1. Recording (`Recorder.swift`, `SceneRunner.swift`)

Small changes that make everything after them better:

- **An `ARAnchor` at the camera every keyframe**, with the anchors' final transforms saved when
  recording stops. ARKit moves its anchors when it corrects drift (relocalisation, loop closure), so
  the final anchor poses are corrected keyframe poses for free. No one has measured whether an A15
  without LiDAR updates them, so also log every anchor-update event. The first recordings will show
  whether this happens.
- **Raw motion** from CoreMotion at 100-200 Hz into its own file, for bundle adjustment priors and
  jump detection.
- **Tracking flags kept per frame** (already written as `tracking`), plus a jump flag when ARKit's
  pose leaps between frames (relocalisation).
- **Compute units.** The live models (detector, refiner encoder and decoder, segmenter, depth) load
  with `.all`, which lets Core ML put any op on the GPU, falling back to `.cpuAndGPU`
  (`SceneRunner.swift:154`). Run MLComputePlan on the iPhone 13 first, to see which ops leave the
  Neural Engine, then load them with `.cpuAndNeuralEngine`; ops the Neural Engine rejects then go to
  the CPU instead of the GPU. That keeps the GPU for ARKit and the camera preview while recording.
- **A live "the view is not shifting" rule** (audit decision 7): distance moved over the last few
  seconds against the depth in view, both from ARKit. Needs tuning on real app recordings.

### 2. Camera poses (replaces COLMAP and MapAnything)

ARKit's visual-inertial tracking is the pose source; nothing open-source beats it on a phone
(RTAB-Map's author builds his iOS app on it). It is not exact on a phone without LiDAR. The only
published numbers are for an iPhone 11: 0.37 m mean trajectory error on a 6-13 m lab path, and
25-43 cm of drift after walking away and back. Rotation stays near 1°. So refinement is needed, and
it must be gated: in a 2026 study (arXiv 2608.21008), COLMAP-style triangulate-then-adjust made
ARKit's room poses worse in 15 of 15 rooms.

- **First choice: the vendored refinement** (`Capture/CameraOnlyTracker.swift`,
  `BundleAdjuster.swift`, `OfflinePoseRefinement.swift`, `LoopClosureRefiner.swift`). It tracks
  Shi-Tomasi corners keyframe to keyframe with Lucas-Kanade, starting where ARKit's motion predicts.
  A track is kept only if it tracks back to its start, its patches correlate and it lies on the
  epipolar line. It then bundle-adjusts with ARKit's frame-to-frame motion as a prior, and applies
  the result only if held-out tracks improve. Loop closure is rigid and never changes scale. Only
  Apple frameworks; no Ceres, no COLMAP.
- **If that gate keeps rejecting, or the result is worse than COLMAP:** rebuild the
  triangulation-free formulation of arXiv 2608.21008 (per-observation ray depth, symmetric two-view
  residuals, an orientation prior, falling back to ARKit) with Ceres (BSD-3). Its code is not
  public; see the licence asks.
- **Intrinsics:** self-calibrate fx, fy, cx, cy, k1 and k2 once per phone model.
- **Measure before trusting it:** 10-20 rooms recorded with the app, each also run through today's
  COLMAP chain on the Mac as the reference. Compare trajectory error (Sim(3)-aligned), tape-measured
  distances, and the splat's PSNR on held-out views. Ship ARKit-only poses if the PSNR gap to COLMAP
  is under about 0.5 dB.

What this removes: COLMAP and MapAnything from the capture path. COLMAP's bundled SiftGPU is
licensed for non-profit use only, so this also settles that question for the product.

### 3. Dense cloud (replaces `densify.py`'s MoGe-2-L pass)

- After recording, once the live models are released, run **MoGe-2-S** (already in the app as
  `RoomMetricDepth`, MIT) on the keyframes. Take the field of view from ARKit's intrinsics. Fit each
  keyframe's scale and shift to ARKit's feature points with the existing `DepthScale`. MoGe's own
  metric scale is used only for frames with no points.
- **Fuse with a TSDF** (2-3 cm voxels, hashed) over the refined poses, in a Metal compute kernel, with
  swift-tsdf (MIT) as the tested CPU reference. The vendored code's `PointCloudFusion` and
  `RefusionEngine` are Apache-2.0 alternatives to compare against.
- **Bare walls** are where per-frame depth scale fails: carry scale from neighbouring frames that
  have points, through the fusion, and use the wall/floor masks with ARKit's planes.
- A larger post-capture MoGe input (about 2,500 tokens, 812x602 at 4:3) needs its own fixed-shape
  `.mlpackage` and the increased-memory-limit entitlement. Measure peak memory in Instruments first.
- **Fix the yardstick first.** The project's depth error figures were measured against a
  reconstruction scaled by MoGe-2-L's own metric scale, so they show MoGe-S agreeing with MoGe-L,
  not true metres. Re-measure two rooms against tape-measured furniture and ARKit's metric poses.

### 4. Objects and outlines (replaces GroundingDINO, SAM 2.1 and `tracking.py`)

- **Boxes:** the YOLOE model the app already runs (`RoomObjects`, 71 room classes). See the licence
  asks: YOLOE needs an Ultralytics licence, and its class embeddings come from Apple's MobileCLIP.
  YOLOE-26 is the upgrade once that is settled (audit decision 4).
- **Outlines: EdgeTAM** (`scripts/convert_refiner_edgetam.py`, `REFINER=edgetam`). It replaces
  MobileSAM; its encoder family ran about eight times faster than MobileSAM's on an A15 (EdgeSAM
  38.7 FPS against 4.9 on an iPhone 14). Our export fixes what Meta's Core ML export leaves out:
  normalisation in the graph, the two high-resolution maps, the box as corner points labelled 2 and
  3, and the single-mask output. The app gives it a point on the thing from the detector's own mask:
  on a framed TV the box alone outlined the wall panel, box plus point the screen.
- **Identity through the video:** not SAM 2's memory. `ObjectTracker` already gives each object one
  identity from ARKit poses and depth voxels. After recording, every 3rd to 6th frame gets one
  EdgeTAM encoder pass, each tracked object is projected to a box (with a depth test for whether it is
  in view), and one decoder pass per visible object outlines it. This drops SAM 2's
  forward-and-back propagation and its duplicate merging. A published walkthrough study found
  appearance memory loses objects on room walkthroughs; our geometry does not.
- **Names:** keyframe crops of tracked objects go to Claude for naming and roles, as the room's object
  list does today. No text encoder runs on the phone for this.
- Measured here on a CPU: EdgeTAM's video tracking carried the TV, sofa, rug, table, plant, dresser
  and bed through 20 and 48 frames, about 2.6 s a frame for four objects and 4.2 s for six. That
  was this cloud machine's 4-core CPU, not the phone's Neural Engine; it shows the model holds our
  objects, not its phone speed.

### 5. Room shape (ports `shapes.py` and `placement.py`)

- **Planes per keyframe** from MoGe-2-S depth with a grid-cell plane fitter (the CAPE method, MIT),
  tagged wall, floor or ceiling by the segmenter and by ARKit's plane classes (`.wall`, `.floor`,
  `.door`, `.window` work on A12 and later without LiDAR).
- Fuse them in world space over ARKit's poses. Rasterise to a 3-5 cm floor-plan grid, fit lines, snap
  them to the room's dominant direction (`RoomMap` has it), and close the outline using the camera
  path as seeds.
- **Boxes:** keep `ObjectTracker`'s mask-plus-depth boxes and port `placement.py`'s silhouette-IoU
  search to Swift, fed by EdgeTAM keyframe masks. Merge duplicates, and never let a box cross a wall.
- Not adopted: RoomFormer did worse than a non-learned baseline on data it was not trained on, and its
  good numbers need ScanNet-derived training data, which is non-commercial. SpatialLM uses a
  non-commercial encoder and needs CUDA. BoxerNet is non-commercial (Meta) and untested without
  sensor depth.
- On LiDAR iPhones, RoomPlan inside our own ARSession is an optional upgrade.

### 6. Splat (replaces OpenSplat on Colab)

- **The trainer:** `third_party/arkit-3dgs-scanner`, the last Apache-2.0 commit (d5d8e31) of Kuo
  Feng-Yuan's Swift + Metal trainer. It uses MRNF densification, a Mip filter and a PPISP colour
  model, and has memory plans that include 4 GB phones, checkpoint and resume, and SOG/PLY export.
  It uses only Apple frameworks. `OASIS-PROVENANCE.md` there records why this commit and what we
  owe.
- **The adapter** (to write): keyframes as JPEG, our column-major transform transposed to its
  row-major one, and the Gaussians **seeded from the fused cloud** as flat discs on the surfaces. The
  iPhone 13 has no LiDAR, and under phone budgets the dense prior is what matters: PocketGS measured
  21.15 dB without it and 23.67 dB with it.
- **Settings on 4 GB phones:** Quick (4,000 iterations, 960 px, SH degree 2), capped at 200-300k
  Gaussians, with the increased-memory-limit entitlement. Run it in the foreground with checkpoints;
  background GPU access on an A15 is unverified.
- **Time, untested:** no A15 measurement exists for any open trainer. Scaling the author's later
  build on an iPhone 17 Pro (70-76 ms a step at a 600k cap) by the A15's 2.9 times lower Metal score
  gives roughly 7-15 minutes for Quick, with thermal slowdowns. Today's Colab T4 takes 4-13 minutes.
  A short preview run (500-1,000 iterations on the dense prior) can show something within minutes.
- **Viewing:** the trainer writes SOG by default; MetalSplatter reads PLY, SPZ and `.splat`, so export
  PLY (`GaussianExport.writePLY`) or SPZ.
- Not adopted: msplat (Apache-2.0, but its host code shares OpenSplat's AGPL names and signatures
  without credit; speed reference only), and feed-forward splats (PocketSplat needs 2.6-2.9 GB;
  Apple's SHARP is research-only). Brush (Apache-2.0, Rust/wgpu) is the second track: the only route
  to one trainer on iOS and Android, but it peaked at 3.1 GB on an iPhone 17 Pro.

This also retires OpenSplat (AGPL-3.0) from the product, and with it audit decision 1.

### 7. Surface fill (`surface_fill.py` on the phone)

The fill keeps `surface_fill`'s design: a rectified photo of each floor and wall, LaMa for what no
frame saw, and flat blobs in the splat. What runs it on the phone is
`apps/OasisCapture/scripts/convert_fill_lama.py`, which builds `SurfaceFill.mlpackage`.

Measured here, on patches of floor and wall from 8 of our clips (wood, a patterned rug, rug and tile,
plain and shiplap walls, a kitchen wall), each with three holes of known content: a furniture
footprint, a region no frame saw, and thin strips. Scores are PSNR inside the hole.

| | Footprint | Unseen blob | Strips | CPU time |
|---|---|---|---|---|
| Big-LaMa (Apache-2.0, 51 M parameters) | 28.71 dB | 32.46 dB | 33.86 dB | 1.5-1.9 s |
| MI-GAN-512 (MIT, 6 M parameters) | 24.24 dB | 27.49 dB | 31.37 dB | 1.0 s |

LaMa was higher in 23 of 24 fills. Looked at, MI-GAN drew a streak across a wood floor, scribbles on
a rug and an object on a kitchen wall that was never there; LaMa's fills were clean. So there is no
MI-GAN tier. OpenCV's FSR (Apache-2.0, no weights) remains an option for tiny holes.

**Fill at 512, not at the photo's size.** Big-LaMa's Fourier units become fixed-size DFT matrix
products in Core ML, so the model's shape cannot vary (enumerated shapes fail to convert). Testing
whether that costs anything showed the opposite. On 1024-pixel crops of the same clips:

| | Footprint | Unseen blob | Strips | CPU time |
|---|---|---|---|---|
| LaMa at 1024 (what `surface_fill` did) | 19.69 dB | 23.51 dB | 27.48 dB | 6.8 s |
| LaMa at 512, the fill scaled back up | 23.25 dB | 24.52 dB | 27.17 dB | 1.5 s |

At 1024 a footprint-sized hole came back as a grey, waffle-textured patch; at 512 the planks and the
shiplap lines carried on through it. `surface_fill.inpaint` now works at 512 on the Mac too (box
filter down, a pixel unknown if any of it was, the fill put back only where the photo was unknown).
Rerun through that code: footprints 19.69 to 23.26 dB, blobs 23.51 to 24.49 dB, strips 27.47 to
27.27 dB. A surface photo is 5 mm a pixel, and the fill reaches the splat as blobs 12 mm apart, so
the finer pixels showed nothing there. A mixed scene's textures are softer where filled, which beats
the waffle.

**The Core ML model.** 512 by 512, fp32 compute on the GPU (`CPU_AND_GPU`): the Neural Engine runs
fp16 only, and LaMa breaks in fp16 (mlx-lama-swift found it collapses the Fourier convolutions).
Weights are int8 for the plain convolutions, except the first and the last:

- The package is 58 MB instead of 196 MB.
- Its own weights, read back from the package and run in PyTorch, moved the fill by −0.05 dB on
  average and −0.61 dB at worst over the 24 fills.
- Quantising the first and last convolutions as well cost 11 dB in simulation.
- Quantising the three transposed convolutions doubled the loss.
- The DFT matrices must stay fp32: their sine matrix has an all-zero row, which per-channel int8
  scaling turns into NaN.

`--check` on a Mac compares the package with PyTorch (the Core ML runtime needs macOS). It is not in
`build.sh` or `project.yml` yet: the app gets a fill stage when it gets the trainer. A one-off report
puts Core ML LaMa at about 2 s for 800x800 on an iPhone 13 Pro; for a floor and four walls that
would be well under a minute.

### 8. Android (later)

Same stages: ARCore for poses (ARCore's Depth API where the phone has it), ExecuTorch 1.5 with
XNNPACK everywhere and the QNN backend on Snapdragon, LiteRT as the per-model fallback, and Brush for
the splat. Every Android figure found was for flagships, so the mid-range target has to be
measured. Not started.

## Measurements the first iPhone 13 run has to take

1. **Live chain:** each model's latency and memory with the other three running, and where each op
   runs (MLComputePlan) before and after `.cpuAndNeuralEngine`. Then EdgeTAM against MobileSAM.
2. **ARKit poses:** does the A15 update keyframe anchors? Trajectory error and tape distances against
   COLMAP on 10-20 rooms.
3. **Camera-only refinement:** time, heat and battery, how often its gate accepts, and held-out splat
   PSNR against COLMAP poses.
4. **Post-capture MoGe-2-S pass:** time per keyframe and peak memory.
5. **Trainer:** minutes and peak memory for Quick at a 200-300k cap, with and without the
   increased-memory-limit entitlement, and whether it finishes without thermal pauses. The vendored
   snapshot has no benchmark mode; that came later, under the non-commercial licence. So we write our
   own timing harness.
6. **SurfaceFill:** `--check` on a Mac, then time and peak memory per 512 fill on the A15's GPU.
7. **The exposure lock** from the audit: `exposure` in `frames.jsonl` should be constant within a
   recording.

## Licences: what to ask for

### Blocking a commercial release now (models the app already ships)

1. **SegFormer-B0** (`RoomSegmentation`, wall/floor/ceiling labels). NVIDIA's licence allows research
   or evaluation only, and its ADE20K training data is non-commercial too. Swapping in another
   ADE20K-trained model does not fix it.
   - *Ask:* NVIDIA Research licensing (nvidia.com/en-us/research/inquiries) for commercial terms, and
     MIT CSAIL (contact on ade20k.csail.mit.edu/terms) for ADE20K-trained models.
   - *Or:* retrain a small segmenter on COCO-Stuff (CC BY 4.0 annotations) and Hypersim
     (CC BY-SA 3.0), with our own labelled captures. That is a one-off GPU training job, not part of
     reconstruction.
2. **YOLOE-11m-seg** (`RoomObjects`). Code and weights are AGPL-3.0.
   - *Ask:* Ultralytics Enterprise (ultralytics.com/license). Get in writing that YOLOE-11 (trained
     by THU-MIG) and YOLOE-26 are covered models, and that iOS and Android redistribution of the
     exported models is covered.
3. **MobileCLIP**, the source of YOLOE's baked class embeddings. Apple's ML Research Model terms allow
   research only and exclude product development, so computing the embeddings once at build time is
   not a way round it.
   - *Ask:* Apple, through the apple/ml-mobileclip repository; the terms give no commercial path, so
     expect a no. Also ask Ultralytics whether their licence covers this.
   - *If refused:* retrain YOLOE's text stage with a SigLIP 2 text tower (Apache-2.0). Or drop the
     open vocabulary: fine-tune a closed-set detector on our 71 room classes (D-FINE, Apache-2.0, needs
     no paid licence), with Claude doing the naming.
4. **Places2 dataset terms** cover the Big-LaMa weights (and MI-GAN's, and AOT-GAN's).
   - *Ask:* counsel to review places2.csail.mit.edu's terms; if unclear, the MIT CSAIL Places team for
     a written statement on commercial use of models trained on it.
5. **MoGe-2's training data.** The weights are MIT, but the training sets include non-commercial ones
   (ScanNet++, Waymo Open, Argoverse 2).
   - *Ask:* counsel review; optionally ask the microsoft/MoGe maintainers to confirm commercial use of
     the released weights.

### For the new on-phone stages

6. **The splat trainer: nothing to buy.** The vendored snapshot is Apache-2.0. Keep `LICENSE` and
   `NOTICE`, and credit Kuo Feng-Yuan in the app's acknowledgements.
   - *Optional ask:* a commercial licence for his later, PolyForm Noncommercial work (1.39x faster
     training steps, a benchmark mode), through github.com/KuoFengYuan.
7. **EdgeTAM and LaMa: nothing to buy.** Apache-2.0; their licence texts go in the acknowledgements.
8. **Apple entitlements**, in the paid developer team (Signing & Capabilities):
   - Increased Memory Limit (`com.apple.developer.kernel.increased-memory-limit`), needed for training
     and the larger depth pass on 4 GB phones.
   - Background GPU Access, if training should continue in the background; whether the A15 grants it
     must be checked on the device.
9. **arXiv 2608.21008's refinement** (only if the vendored refinement fails its measurements): ask the
   author, Nikolaos Kyriazis (address in the paper), for the code under BSD or Apache, or written
   permission to reimplement it.
10. **A weight mirror** (access, not a licence). The build machine cannot reach Hugging Face or Apple's
    model CDN. Someone with access should put the MoGe-2 weights (and anything else Hugging Face-only
    that we adopt) in a GitHub release of our own, with their licence files and SHA-256 hashes.

### Settled by moving to the phone

- **OpenSplat** (AGPL-3.0) leaves the product with the vendored trainer.
- **COLMAP's bundled SiftGPU** (non-profit only) leaves the capture path with ARKit poses. If COLMAP
  stays in the Mac tools for commercial work, build it with GPU SIFT off.

### For Android, later

- A Qualcomm AI Hub account (free) for Snapdragon NPU exports, and a read of the QAIRT and AI Hub
  model licences before shipping their runtime libraries.
- ARCore's terms and its privacy disclosure.

### Avoid

These have no commercial path or are non-commercial:

- SHARP (Apple)
- EdgeSAM
- UniDepth
- Depth Anything 3 Large and Giant
- MVSAnywhere and SimpleRecon (Niantic; ask only if wanted)
- BoxerNet (Meta; ask only if it ever passes a test with our depth)
- the GPL VIO and SLAM systems (ORB-SLAM3, VINS, OpenVINS)

Datasets with non-commercial terms (ScanNet, Structured3D, CA-1M, ADE20K) need counsel's view
before any model trained on them ships.

## Order of work

1. On a Mac with the iPhone 13: build the app with `REFINER=edgetam`, take measurement 1, switch the
   compute units, and add the recording changes (anchors, motion, jump flag).
2. Record 10-20 rooms with the app; run the Mac pipeline on them as the reference.
3. Write the trainer adapter and timing harness; measurements 3 and 5 on those rooms.
4. The post-capture pass: MoGe-2-S on keyframes, TSDF fusion, EdgeTAM outlines on tracked objects.
5. Port the room shape and the fill; wire `SurfaceFill.mlpackage` in.
6. In parallel: the licence asks, starting with Ultralytics, Apple (MobileCLIP) and NVIDIA
   (SegFormer). Those three decide whether the live models can ship at all.
