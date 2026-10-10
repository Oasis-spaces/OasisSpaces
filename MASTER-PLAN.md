# Oasis Spaces — master plan, project map and status

*Written 10 October 2026. The README explains how each part works; this file
says where the whole project stands, what it is for, what is missing, and what
is needed to run it.*

## 1. The goal

**A phone video of a room becomes a 3D room that can be edited:** every piece
of furniture can be deleted, moved, turned and resized; walls, floors and
furniture can be given new colours and textures; dimensions can be changed;
and any 3D object can be added to the scan. The capture is guided on the
phone while recording; the reconstruction runs on a Mac (with a Colab GPU for
the heavy stages); Claude acts as an agent that names what it sees, reviews
the measured room and judges the results — but never places or measures
geometry itself, because that is what it is worst at (see
`docs/research-weaknesses-2026-10.md`).

The original note that started it is `OasisSpaces.md`: "images from every
angle, stitched, a cloud of the whole thing, such that we can edit any cloud".

## 2. Where we stand (10 Oct 2026)

| Part | State | Where |
|---|---|---|
| Phone capture app (iOS 18, no network while recording) | Working on the test iPhone 13: two-tier tips, live rules with haptics, four on-device models (YOLOE-11m detector, MobileSAM refiner, SegFormer-B0 surfaces, MoGe-2-small metric depth), neon outlines that stay on their things, a tracked room map, sends recordings to the Mac or through the relay | `apps/OasisCapture` |
| Phone↔Mac link | Working: Bonjour discovery on Wi-Fi, HTTP job API on the Mac, the same API on a free Render relay (Supabase storage) for off-network use, pairing by code or account | `apps/OasisLink`, `relay/` |
| Mac app (Splat Viewer) | Working: library, tabs, fly-through, phone captures run through the pipeline, room editing (select, drag, resize, turn, delete, repaint walls, add library furniture, undo), mixed-scene viewer window | `apps/SplatViewer` |
| Stage 1 — cameras | Working: COLMAP global mapper, MapAnything gap-fill for frames COLMAP cannot place (CUDA), pose priors from the phone accepted | `pipeline/reconstruct.py`, `pipeline/mapanything_solve.py` |
| Stage 2 — dense cloud with names | Working: MoGe-2 metric depth, GroundingDINO names, SAM 2.1 outlines; **new:** objects tracked through every frame (Colab), mirrors/windows/screens stood on their planes | `pipeline/densify.py`, `pipeline/tracking.py`, `pipeline/mirrors.py` |
| Stage 3 — the room (walls, floor, boxes) | Working: RANSAC walls, boxes per named object, Claude's structure review, placement by masks, the phone's second opinion, Blender room, pose-matched render check; **new:** a measured room score that rolls back any edit that makes the room worse, a doorway term, slab repairs | `pipeline/shapes.py`, `tools/classify_shapes.py`, `pipeline/placement.py`, `pipeline/room_score.py`, `pipeline/agent.py` |
| Stage 4 — splat and scene | Working: OpenSplat quick/long training (Colab or Mac), Claude's choice at held-out views, floor/wall fill with LaMa, the mixed scene (textured shell + movable scanned pieces + cuboid models), browser scene viewer | `pipeline/splat_seed.py`, `tools/splat_*.py`, `tools/surface_fill.py`, `tools/mixed_scene.py`, `scene-viewer/` |
| Running on Colab from the Mac | Working: one command uploads code, video and space, installs, runs each stage, relays Claude's checks, fetches results | `tools/colab_pipeline.py`, `tools/claude_relay.py`, `notebooks/OasisSpaces_Colab.ipynb` |
| Website | Separate repo (Oasis-spaces/website); this repo keeps the landing page and 3D previews | `index.html`, `*.js`, `styles.css` |

Two rooms are the test set: the **pan** (IMG_4138, 42 frames, a bedroom
panned from one spot) and the **walkthrough** (IMG_4182, 186 frames, a walk
through a flat). Both are reconstructed end to end; their spaces live in
`spaces/` (gitignored: personal footage).

## 3. What has been achieved

Measured, on the two rooms:

- **Cameras:** the walkthrough places 183 of 186 frames in one model (COLMAP
  alone: 167), reprojection 0.78 px; the pan 42 of 42.
- **A room from a pan:** four walls, a bed within 25 cm of its measured box
  from masks alone, a wardrobe placed by the phone's own detector; Claude's
  render check now passes it at the first try with only minor notes.
- **Measured over opinion:** the room score (`pipeline/room_score.py`) keeps
  a measured piece a judge wants dropped (the pan's chest of drawers at 0.73
  agreement, three times), and drops a door built as a "wardrobe" and a
  7 cm "chest of drawers" that was the bed's base front — the two mistakes
  that had survived every review before.
- **Every frame labelled:** on Colab the pan's 38 objects are carried
  through all 42 frames in 8 minutes; placement evidence went from ≤12
  keyframes to up to 40 frames per object.
- **No more holes at windows:** 273,030 of the 304,464 points that used to
  be dropped on mirrors, windows and screens now stand on their planes.
- **Splats:** 125k–187k Gaussians per room, trained in 4–13 minutes on a
  T4; the best of each video is published in `splats/`. Long training (30k
  steps) measured no better than quick (10k) at held-out views.
- **The mixed scene:** a textured mesh shell with each piece of furniture as
  a movable scanned piece or a clean model, reviewed by Claude per surface
  and per piece, viewable and editable in the browser, the Mac app and the
  phone app.
- **The phone:** instant guidance with no model calls over the network; the
  room map's boxes are the furniture's size, checked against the videos with
  `phonesim` on the Mac; the phone's perception serves the pipeline as a
  second opinion (it placed the pan's wardrobe where the cloud had nothing).
- **Judging the judge:** research (Oct 2026) found model judges agree with
  true geometry about as often as a coin; the pipeline now uses Claude to
  *name and explain* and measurements to *decide*.

## 4. What was built on 10 October

1. **Tracking through the video** (`pipeline/tracking.py`): GroundingDINO
   names things on ~20 spaced frames; SAM 2.1's video model carries each
   through every frame and back to the start; one identity, one name by vote;
   only the room's names are tracked (furniture, storage, fixtures,
   mirrors/windows, hanging things, floor coverings); float16 on CUDA. SAM 3
   would do both in one model but its weights are gated.
2. **Mirror planes** (`pipeline/mirrors.py`): the plane of the strip round a
   mirror, window or screen fills its outline, Mirror3D-style.
3. **The room score with rollback** (`pipeline/room_score.py`, `agent.py`):
   agreement of every built piece with its outlines, less collisions, the
   walk inside a piece, and a piece standing in a doorway (by fixture
   outlines, or by a track the detector part-named a door, in full when the
   piece is thinner than 0.2 m). Each review edit and each second-opinion
   addition is measured and undone if it lowers the score; the recheck keeps
   the room that measures higher; thin slabs inside pieces are repaired
   before and after the review.
4. Fixes found by running: the Colab driver's log parse, the `claude` CLI's
   two-document output at a session limit, one weak "door" vote not being a
   verdict, a door's track not being wardrobe evidence.

## 5. Issues still open

- **Systematic mislabels.** The detector calls the pan's door a wardrobe two
  times in three, the bed's base a chest of drawers, the wardrobe's body a
  hanging cloth. Tracking consolidates these as well as it fixes flicker;
  geometry rules catch some. A stronger detector (SAM 3, gated) is the real
  fix.
- **The score's blind spots.** Its agreement term uses the same outlines the
  labels came from, so it cannot judge a name; walls are not measured at all
  ("the room reads deeper than the video" is still only the judge's word).
- **Sizes and depth.** A wardrobe seen only from the front gets a typical
  depth; the phone's wardrobe is narrower than the real almirah; masks
  cannot see through clutter (the walkthrough's bed comes out 0.9 m wide by
  its masks against 1.57 m measured).
- **Scale and truth.** Monocular depth gives ±13% scale (the method's limit);
  neither room has been measured with a tape, so every number above is
  relative.
- **Repeatability.** Claude's reviews differ run to run (one run dropped wall
  W2, three did not); the measured score limits the damage but does not
  remove the variance.
- **Compute.** Tracking needs CUDA; a free Colab session gives ~40 usable
  minutes after installs and uploads, and the walkthrough's tracked densify
  has not yet completed inside one (two attempts). The Mac has 8 GB: one
  model at a time, one agent stage at a time.
- **Limits of the `claude` CLI** stop a run mid-way (now reported correctly).
- **Splat quality on plain walls** plateaued; the mixed scene is the answer
  chosen, and its pieces seen from unfilmed angles are still hazy scans
  until swapped for models.
- **Android** is not started (the rules are portable by design).

## 6. What is still needed for the final goal

Against "editable: deletable, colours, textures, dimensions, any object added":

| Capability | Today | Needed |
|---|---|---|
| Delete a piece | Yes — Splat Viewer (⌫), `splat_edit.py remove`, scene viewer (hide); the floor and wall behind are filled | Fill quality behind large pieces on sparse walls |
| Move / turn | Yes — Splat Viewer handles, scene viewer drag/turn, saved layouts | — |
| Resize (dimensions) | Splat Viewer: corner, height handles, exact cm in the inspector (Gaussians scaled); models: cuboid parts scale | Resizing a *scanned* piece without stretching its texture: regenerate a model at the new size (TRELLIS job exists: `tools/object_mesh.py`, GPU) |
| Colours | Walls repainted from swatches or any colour (only the wall's own paint changes); library pieces recoloured | Per-piece recolour of scanned pieces and models in the scene viewer and the phone |
| Textures | Shell textures come from the frames (LaMa continues them); not user-editable | A texture picker for walls/floor (tile, wood, paint) applied to the shell's materials; UV-mapped shell |
| Add any 3D object | Library furniture only (sofa, bed, table, chair, desk, wardrobe, box, as blobs or cuboid models) | Import any GLB/USDZ into the scene viewer and the apps, placed on the floor, scaled in metres, saved with the layout; a catalogue of generated models |
| Clean furniture models | Cuboid models in the scan's colours; TRELLIS meshes on a GPU | Run the model job in the Colab flow by default; amodal completion (FIRE3D or Lucida-style) once released |
| Correct rooms every time | Measured score, rollback, repairs; passes on the pan | A wall score (do the cameras' walls meet the floor where the frames say?); ground truth from a tape measure of both rooms; SAM 3 when access is granted; scale anchors from known sizes (doors ~2.0 m) |
| Tracking in every run | Colab only | Fit the walkthrough inside a session (fewer tracked names or a paid runtime), or a Mac path that batches objects |
| Phone on both platforms | iPhone (iOS 18) | Android with ARCore feeding the same `capture-rules.json` and the same models exported for TFLite/NNAPI |
| Accounts and sharing | Supabase project, relay jobs table, a test user | Sign-in in the apps everywhere, scenes stored per account, links to share a room |

Order I would do them in: (1) the wall score and ground truth, because every
other number rests on them; (2) GLB import and per-piece recolour in the scene
viewer, because that completes "editable" for a user; (3) model generation in
the Colab flow; (4) SAM 3 when access comes; (5) Android.

## 7. Project map

Everything tracked in the repository, by folder. (`spaces/`, `videos/`,
`splats/*.ply`, `runs/`, built apps, Core ML models and the OpenSplat binary
are gitignored: large, generated, or personal footage.)

### Root
- `README.md` — how every part works, with the reasons behind each rule.
- `MASTER-PLAN.md` — this file.
- `OasisSpaces.md` — the original idea, one paragraph.
- `index.html`, `main.js`, `hero.js`, `room.js`, `scene.js`, `editor.js`, `styles.css` — the public landing page with its 3D previews (the marketing site itself is the Oasis-spaces/website repo).
- `logo.png` — the logo.
- `.claude/launch.json` — dev servers for the desktop app's preview pane (editor on 8734, website).

### `pipeline/` — the reconstruction, stage by stage
- `agent.py` — runs the stages, adapts, asks Claude (frame choice, object naming, label review, structure review, render check, splat choice, scene review, capture advice), records every decision in `agent-report.json`; the measured room score, rollback and repairs live here with `room_score.py`.
- `advisor.py` — Claude access: the `claude` CLI (default), the API, or the relay folder on a VM; tolerant of the CLI's two-document output.
- `reconstruct.py` — frames and camera poses (ffmpeg + COLMAP global mapper, incremental fallback, `--mapper priors` for phone poses).
- `mapanything_solve.py` — MapAnything places the frames COLMAP could not (CUDA), guided by COLMAP's lens and cameras.
- `densify.py` — MoGe-2 metric depth fused over the poses; names and outlines per keyframe; tracking and mirror planes; writes `cloud-dense.ply` and `densify.json`.
- `depth_maps.py` — MoGe-2 depth for every registered frame in the solve's units (for the phone's tools).
- `semantics.py` — the object vocabulary and roles, GroundingDINO detector, SAM 2.1 outlines, pixel labels.
- `tracking.py` — objects carried through the video with SAM 2.1's video model; votes, merges, doorish tracks.
- `mirrors.py` — the plane of a mirror's surroundings written into its outline.
- `shapes.py` — planes and boxes from the labelled dense cloud (`shapes.json`).
- `placement.py` — a piece placed by its masks: candidate boxes' silhouettes through the real cameras against the outlines.
- `room_score.py` — the measured room score (agreement, collision, walk, doorway).
- `object_models.py` — per-object jobs (a frame and a cut-out) for image-to-3D models.
- `pointcloud.py` — PLY I/O, downsampling, outlier removal.
- `splat_seed.py`, `splat_export.py`, `splat_spirula.py` — OpenSplat project seeded from the dense cloud; compact `.splat` and starting view; Spirula Studio as a plug-in trainer.

### `tools/` — stage helpers and the editing tools
- `classify_shapes.py`, `blender_room.py`, `blender_views.py`, `furniture_library.py`, `shapes_to_obj.py` — label and sanity-check boxes, finish the room, build it in Blender with parametric furniture, render it from the video's own cameras, export OBJ.
- `room_views.py`, `plan_image.py`, `object_frames.py`, `capture_map.py` — the pose-matched render rows, the plan with the walk, the frames that show each object, the capture map.
- `phone_objects.py`, `phone_sim_export.py`, `phone_capture_export.py`, `phone_video_preview.py` — the phone's perception run over a space on the Mac (`phonesim`) as a second opinion; exports and previews.
- `splat_edit.py`, `splat_tools.py`, `splat_prune.py`, `splat_render.py`, `splat_choose.py`, `compare_splats.py`, `surface_fill.py` — remove/add furniture in a splat, composite PLYs, prune floaters, CPU render, Claude's choice of the best splat, floor/wall/face fill with LaMa.
- `mixed_scene.py`, `mesh_render.py`, `object_mesh.py` — the textured shell and movable pieces; GLB drawing; TRELLIS meshes on a GPU.
- `evaluate_space.py`, `show_prompts.py` — scoring spaces; printing every question the pipeline asks Claude.
- `colab_pipeline.py`, `claude_relay.py`, `paced_proxy.py` — the Colab driver, the Claude relay, the paced upload proxy for this network.
- `tools/tests/` — twelve test suites (`python3 tools/tests/test_<name>.py`): advisor parse, capture map, mirrors, mixed scene, phone objects, placement, pose priors, room score, room views, stage-1 gaps, structure review, tracking.
- Gitignored but needed locally: `tools/opensplat` + `tools/default.metallib` (OpenSplat built for Metal, AGPL, patched), `tools/colab-cache/` (the OpenSplat binary built on Colab), `tools/spirula-studio/`.

### `apps/OasisCapture/` — the iPhone capture app
- `Sources/` — `OasisCaptureApp.swift` (tabs), `CaptureScreen.swift` + `CaptureController.swift` (camera, ARKit, rules, haptics), `FrameAnalyzer.swift` (the four Core ML models), `OutlineOverlay.swift`, `RoomMapView.swift`, `SceneRunner.swift`, `ScanCloud.swift`, `Recorder.swift` (video.mov, frames.jsonl poses, capture.json), `ReviewView.swift`, `TipsView.swift`, `AppLog.swift`, `Link/` (scans list, Mac view, the room scene and splat screens).
- `Packages/CaptureRules/` — the portable rules and perception, no ARKit: `RuleEngine.swift`, `RuleConfig.swift`, `FrameSample.swift`, `Objects.swift`, `Sightings.swift`, `ObjectTracker.swift`, `Outlines.swift`, `DepthFusion.swift`, `MetricDepth.swift`, `RoomMap.swift`; specs in `Resources/capture-rules.json`, `object-classes.json`, `detection-classes.json`; `phonesim` (the phone's perception over a processed space, on the Mac); `Tests/` (`swift test`).
- `Resources/` — `Info.plist` and the Core ML models (gitignored, ~150 MB): `RoomObjects` (YOLOE-11m), `MaskEncoder`/`MaskDecoder` (MobileSAM), `RoomSegmentation` (SegFormer-B0), `RoomDepth`, `RoomMetricDepth` (MoGe-2 small).
- `scripts/` — `convert_objects.py`, `convert_refiner.py`, `convert_segmentation.py`, `convert_depth_moge.py` (PyTorch → Core ML).
- `project.yml`, `build.sh` — XcodeGen project; build, install and launch on the paired iPhone.

### `apps/OasisLink/` — the phone↔Mac link (Swift package)
- `Models.swift`, `Station.swift` + `HTTPServer.swift` (the Mac's job server), `StationClient.swift` (the phone's client, paced retry), `Discovery.swift` (Bonjour), `CloudLink.swift` (the relay), `Account.swift`, `AccountStore.swift`, `AccountView.swift` (Supabase accounts), `Resources/cloud.json`, `supabase.json`; `Tests/`.

### `apps/SplatViewer/` — the Mac app
- `Sources/App/` (app, self-test, edit test), `UI/` (home, tabs, viewer, editor overlay), `Viewer/` (MetalSplatter scene, fly camera, occupancy grid for walking through things, thumbnails, scene editing), `Editing/` (editor model, room model, furniture, Gaussian maths, surface patches), `Model/` (library, splat files), `Station/` (the station service, the pipeline runner, phone captures, room scenes, the cloud worker for the relay).
- `project.yml`, `build.sh` — XcodeGen; builds `Splat Viewer.app`, `--install` copies it to `~/Applications`.

### `relay/` — the off-network relay (FastAPI, on Render)
- `main.py`, `requirements.txt`, `tests/test_relay.py`, `README.md`. Service `oasis-relay` on Render, Supabase project `oasis-spaces` (jobs table, `oasis` bucket).

### Viewers and the browser editor
- `scene-viewer/` — the mixed-scene viewer (three.js 0.180 + Spark, vendored in `vendor/`): select, drag, turn, hide, swap scan/model, save a layout; served by the Mac app and the phone app too.
- `splat-viewer/` — a WebGL splat viewer (MIT, with `convert.py`).
- `editor/index.html` — a browser point-cloud editor.

### Colab, scripts, docs
- `notebooks/OasisSpaces_Colab.ipynb` — the GPU notebook: install cells (COLMAP, Python deps, Blender, OpenSplat, MapAnything), clone, Claude relay, upload, the four stages, download.
- `scripts/process_video.sh` — the whole chain without the agent; `scripts/make_sample.py` — a synthetic demo room.
- `docs/research-3d-reconstruction-2026-09.md` — the tool survey that shaped pipeline v2; `docs/research-weaknesses-2026-10.md` — the October research on our measured weaknesses and the plan that followed.
- `splats/<video>/best.splat` — the published best splat of each test video.

## 8. Prerequisites to run everything

**Mac (Apple silicon, macOS 15; 8 GB RAM works, 16 GB is comfortable; ~50 GB free disk):**
```bash
brew install colmap ffmpeg pytorch blender xcodegen
pip3 install --break-system-packages 'numpy>=2.3' 'transformers>=5' pillow anthropic
pip3 install --break-system-packages --no-deps \
  "git+https://github.com/EasternJournalist/utils3d.git@3fab839f0be9931dac7c8488eb0e1600c236e183" \
  "git+https://github.com/microsoft/MoGe.git@925b8ed835a7a9cdb7578ba15c658a0afc969030"
```
- Homebrew's `pytorch`, not pip's (`tools/opensplat` is linked against it); `numpy >= 2.3` (older numpy silently miscomputes on this Python 3.14).
- OpenSplat built locally for Metal into `tools/opensplat` with `tools/default.metallib` (the README's OpenSplat note; AGPL, kept out of git).
- Blender on the path or `BLENDER=/path/to/blender` (the room build and the render check).
- LaMa's `big-lama.pt` in `~/.cache/oasisspaces/` (downloaded on first use; the notebook fetches it on Colab).
- Model weights download on first run from Hugging Face: MoGe-2 (`Ruicheng/moge-2-vitl-normal`), GroundingDINO-tiny, SAM 2.1 hiera-tiny. SAM 3 needs an access request on Hugging Face and a login (`huggingface-cli login`); it is not wired in yet.
- **Claude:** the `claude` CLI signed in (Claude Code) — the agent's default backend, model `claude-opus-5`; or `ANTHROPIC_API_KEY` for the API backend. Without either the pipeline runs with its deterministic rules only.
- Run one agent stage at a time on an 8 GB Mac: `python3 pipeline/agent.py videos/<video>.MOV --name <space> --stage reconstruct|densify|shapes|splat`.

**Colab (for stage 1's MapAnything gap-fill, tracking, and splat training):**
- `uv tool install google-colab-cli --with 'jupyter-kernel-client<1.0'`, then `colab sessions` once to sign in.
- `python3 tools/colab_pipeline.py videos/<video>.MOV --name <space> [--stages ...]` does the rest; the Claude relay runs on the Mac with the local `claude` login. Free sessions last about an hour; uploads go in 20 MB parts.

**iPhone app:** Xcode 16 with the iOS 18 SDK, XcodeGen, a paired iPhone on iOS 18 (tested on an iPhone 13, no LiDAR), a signing team (a free personal team works; installs expire after 7 days). The Core ML models are built once with the scripts in `apps/OasisCapture/scripts` in a Python 3.11 venv with `ultralytics 8.4`, `coremltools 9`, `mobile_sam` and the MoGe checkpoint, or copied from a machine that has them. Then `apps/OasisCapture/build.sh`.

**Mac app:** Xcode 16, XcodeGen; `apps/SplatViewer/build.sh --install`. MetalSplatter is fetched by Swift Package Manager. The app needs the repository beside it to run the pipeline on phone captures and to serve the scene viewer.

**Relay and accounts (optional, for phones off the Mac's Wi-Fi):** a Supabase project (URL and publishable key) and a Render web service running `relay/` with `SUPABASE_URL`, `SUPABASE_KEY`; redeploy with Render's `trigger_deploy` after pushing (no webhook).

**Pushing from this Mac's network:** uploads faster than ~170 KB/s are corrupted, so run `python3 tools/paced_proxy.py` and push with `git -c http.proxy=http://127.0.0.1:8899 push origin main`. Never commit clouds, splats or videos.

**Tests:** `for t in tools/tests/test_*.py; do python3 $t; done` on the Mac; `swift test` in `apps/OasisCapture/Packages/CaptureRules` and `apps/OasisLink`; `pytest relay/tests`.
