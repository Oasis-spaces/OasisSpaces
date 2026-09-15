# OasisSpaces

Turn photos of any space into an editable 3D point cloud.

**Run it for free:**
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Oasis-spaces/OasisSpaces/blob/main/notebooks/OasisSpaces_Colab.ipynb)
— the whole pipeline (cameras, metric dense cloud, Blender room, Gaussian splat) on
Google Colab's free GPU, one stage at a time. The public
website is hosted free at **https://oasisspaces-0ddh.onrender.com**, and the editor at
**https://oasisspaces-0ddh.onrender.com/editor/**.

The idea (from [OasisSpaces.md](OasisSpaces.md)): capture a space from every
angle, stitch the images together, and build a cloud of the whole space that
can then be edited. This repo implements that as a two-part system:

```
photos / walkthrough video
        │
        ▼
  pipeline/reconstruct.py        editor/index.html
  ┌──────────────────────┐       ┌─────────────────────┐
  │ 1. frame extraction  │       │ view · orbit · zoom │
  │ 2. feature matching  │  PLY  │ box-select points   │
  │ 3. camera solving    │ ────► │ delete · crop       │
  │ 4. triangulation     │       │ undo · export PLY   │
  │ 5. cleanup           │       └─────────────────────┘
  └──────────────────────┘
```

The "stitching" is COLMAP's Structure-from-Motion: it finds the same visual
features across overlapping photos, solves for where every camera was, and
triangulates each matched feature into a 3D point — producing one coherent
cloud of the space.

## Setup

```bash
brew install colmap ffmpeg     # reconstruction + video frame extraction
brew install pytorch           # depth network + tools/opensplat (use brew's, not pip's)
pip3 install --break-system-packages 'numpy>=2.3'   # see note below

# MoGe-2 metric depth for pipeline/densify.py. Pinned: newer MoGe drops
# macOS. --no-deps keeps Homebrew's torch (pip's torch crashes tools/opensplat).
pip3 install --break-system-packages --no-deps \
  "git+https://github.com/EasternJournalist/utils3d.git@3fab839f0be9931dac7c8488eb0e1600c236e183" \
  "git+https://github.com/microsoft/MoGe.git@925b8ed835a7a9cdb7578ba15c658a0afc969030"
```

(Homebrew's Python refuses plain `pip3 install` — PEP 668. Either pass
`--break-system-packages` as above, or use a venv:
`python3 -m venv .venv && .venv/bin/pip install 'numpy>=2.3'` and run the
pipeline with `.venv/bin/python`.)

## Capturing a space

Reconstruction quality is decided at capture time:

- **Overlap is everything.** Each photo should share 60–80% of its view with
  the previous one. Move in small steps, don't pivot in place.
- Circle the room along the walls, then cross it diagonally. Capture corners
  from both sides. A few dozen to a few hundred photos is typical.
- Or just record a slow walkthrough **video** — the pipeline extracts frames
  for you. Budget 3–4 minutes of slow walking per room, in three loops: eye
  level, tilted up at the wall–ceiling seam, tilted down at the wall–floor
  seam. Film along each wall and square-on to it.
- Lock exposure and focus (long-press **AE/AF Lock** in the Camera app) and
  turn every light on; close curtains on bright windows.
- Avoid blur, mirrors, and glass. Bare white walls have nothing to match on:
  a few pieces of painter's tape or sticky notes, spread out, help a lot.
- Tape-measure one wall or door per room; image-only reconstructions have no
  reliable scale.

## Building the cloud

```bash
# from a folder of photos
python3 pipeline/reconstruct.py ~/Pictures/living-room --name living-room

# from a walkthrough video (extracts 2 frames/sec by default)
python3 pipeline/reconstruct.py walkthrough.mp4 --name living-room --fps 2
```

Output: `spaces/living-room/cloud.ply` (plus the COLMAP workspace next to it
for reruns/debugging). Progress is logged to `spaces/<name>/workspace/colmap.log`.

Mapping uses COLMAP's **global mapper** (GLOMAP, built into COLMAP 4.x): it
solves all cameras together, which is faster and fragments hard captures far
less than the classic incremental mapper. It can also go wrong the opposite
way: on a long walkthrough it once joined four separate pieces at scales
400x apart into one "complete" model. So for video, each global model is
checked, and if the camera jumps more than 30% of the scene's size between
consecutive frames the model is rejected (kept in
`workspace/sparse-global-rejected/`) and the incremental mapper runs
instead. `--mapper incremental` forces the old behaviour, which is also the
fallback if the global mapper fails. The log lists each model with how many
frames it registered. `densify.py` adds a second check: it warns when
keyframes disagree on metric scale by more than 30%.

`--dense` runs COLMAP's dense multi-view stereo after SfM for a far denser
cloud, but that step needs CUDA — on a Mac it is skipped with a note. The
sparse cloud is usually plenty to see and edit the space; for dense results,
copy the workspace to a CUDA machine and rerun with `--dense`.

## The agent

`python3 pipeline/agent.py <video> --name kitchen` runs every stage and adapts
between them, so a capture can be processed unattended:

- it reads what each stage produced (frames registered, whether the solve
  fragmented, how well keyframes agree on metric scale, what was detected,
  how flat the walls are) and decides to accept, retry differently, or stop;
- a weak solve is retried with learned features or the incremental mapper,
  and whichever attempt registered more frames is kept;
- keyframes disagreeing on scale by more than 30% trigger a rebuild of the
  cameras, because that means pieces were joined at different scales;
- every decision, its evidence, the final measurements and advice for the
  next shoot land in `spaces/<name>/agent-report.json`.

Claude makes the judgement calls that measurements cannot, through
`pipeline/advisor.py`: it looks at frames from the capture and chooses how to
retry a weak solve, checks the detector's labels against what is actually in
frame, judges the finished room against a real photo, and writes the capture
advice. The advisor uses the `claude` CLI (your existing login, no API key;
run `claude login` if the session has expired), or the Anthropic API when
`ANTHROPIC_API_KEY` is set, and otherwise reports `offline` and leaves the
agent to its numeric rules. A measurement always overrules an opinion: if
Claude says "accept" while under 35% of frames were placed, the agent retries
anyway and records the override. `--no-claude` skips it entirely.

## The full chain

`scripts/process_video.sh <video> <name> [fps]` runs the same stages without
the agent's judgement:

1. `pipeline/reconstruct.py` — frames and camera poses (above).
2. `pipeline/densify.py` — MoGe-2 predicts metric depth for keyframes; each
   frame is scaled to COLMAP's units and back-projected into
   `cloud-dense.ply`. Frames that see too few sparse points still fuse, using
   the median scale. `densify.json` records the model used and how many
   COLMAP units make one metre. `--depth-model da2` uses the older Depth
   Anything V2 path.

   It also runs `pipeline/semantics.py`: an open-vocabulary detector
   (GroundingDINO-tiny, local, no API) names the objects in each keyframe.
   A detection is a rectangle, and a rectangle round a bed also holds floor
   and curtain, so SAM 2.1 (hiera-tiny, local) cuts each one to the object's
   outline first (`--no-outlines` keeps rectangles). Every dense point knows
   the frame and pixel it came from, so those outlines label the points
   directly. Points on mirrors, windows and screens are dropped instead,
   because monocular depth there is a reflection or the view outside.
   `--no-semantics` turns this off.
3. `pipeline/shapes.py` and `tools/classify_shapes.py` — planes and boxes.
   Walls come from RANSAC; the floor and ceiling are the lowest upward-facing
   and highest downward-facing height levels inside the walls, so a strip of
   floor is enough and a lower corridor floor seen through a door is ignored.
   Detected furniture is excluded from plane fitting and becomes one box per
   object; the storage labels (wardrobe, cabinet, shelf, bookcase, chest of
   drawers) are grouped first, because a cupboard filmed side-on comes out as
   slivers of each. Whatever is left is grouped geometrically as before. The
   classifier keeps a detected object's identity but still applies its
   sanity checks, and marks a box `build: false` when it is scan debris
   (implausibly large, floating, or too sparse) so the Blender room skips it.
4. `tools/blender_room.py` — a parametric Blender room.
5. `pipeline/splat_seed.py` then `tools/opensplat` — a Gaussian splat. The
   seed is the dense cloud (voxel-downsampled to 250k points), not COLMAP's
   sparse points, so low-texture walls start filled in. `pipeline/splat_export.py`
   then writes `splat.splat` beside `splat.ply`: the viewer's compact format,
   about 8x smaller. It also writes `splat.view.json`, the starting camera: of
   the views at, just behind or just ahead of each capture position, level and
   turned a little either way, the one whose screen shows the most splat with
   the least blur (no wall pressed against the lens, no cupboard filmed
   edge-on). The viewer opens there, with a fixed 55° vertical field of view.
   Open it with `splat-viewer/index.html?url=../spaces/<name>/splat.splat`.

## Editing the splat

`tools/splat_edit.py` edits a trained splat using what stage 3 measured, with
no stage re-run:

```bash
python3 tools/splat_edit.py objects spaces/<name>          # the objects, sizes, positions in metres
python3 tools/splat_edit.py remove spaces/<name> B1        # take the bed out
python3 tools/splat_edit.py add spaces/<name> sofa --at B1 --against-wall
python3 tools/splat_edit.py look spaces/<name> B1          # a viewer link looking at it
```

`remove` deletes the blobs in the object's box (plus a margin for a blanket
over the edge, anything above it that is unlike the wall behind, and parts
growing up out of it away from the walls, like a headboard taller than the
box), but never the floor, another piece of furniture or the wall paint. The
floor and wall it hid were never filmed, so they get the same blended fill as
`fill-room`, limited to around the object: texture and colour come from
beyond its footprint (not its shadow), and what is left of it on the surface
is replaced. `add` builds a sofa, armchair,
bed, table, chair, desk, wardrobe or box from simple parts as blobs, at a
typical size or `--size W D H` in metres, standing on the floor at `--at x y`
(metres from the room's centre) or where an object stood, optionally backed
against the nearest wall. `fill-floor`, `fill-walls`, `fill-objects` and `fill-room` (all three) fill
floor, wall and flat furniture faces the splat has no blobs for (a phone at eye height never sees the floor near you, or the
wall behind a curtain) with the room's own flooring and paint
(`tools/surface_fill.py`). Every frame is projected onto each surface stage 3
measured, keeping only views nothing blocks. An AI inpainting model (LaMa,
local) continues it into the gaps, taking broad colour from the splat's own
surface and fine detail from the photo. The fill is flat blobs that fade into
the splat's surface. A spot counts as a gap only if the splat has nothing at
all near the plane, so a recessed door or window is never painted over.
Anything reconstructed behind a wall marks an opening (a doorway, a window)
that is never filled. A furniture face is only filled when it faces into the
room and the splat shows it as a flat surface (within 6 cm, over 30% of it),
like a wardrobe's doors; a blanket over a bed's side is left alone. It needs `~/.cache/oasisspaces/big-lama.pt` (IOPaint's
big-lama checkpoint, 206 MB). Edits chain in `splat-edited.ply` / `.splat`
(`--fresh` starts again). Added furniture is evenly shaded, a placement
preview rather than a filmed object; a patch is a plausible guess, not what
was really behind the object.

`tools/opensplat` loads its Metal shaders from `tools/default.metallib`;
keep the two files together.

## Editing the cloud

Serve the project root and open the editor:

```bash
python3 -m http.server 8734
```

Then visit <http://localhost:8734/editor/>. Open any `.ply` (file picker or
drag-and-drop), or click **Sample room** for the demo space.

- **Navigate** (`V`): orbit / pan / zoom
- **Select** (`S`): drag a box to select points — `⇧` adds, `⌥` removes
- `X` delete selection · `C` crop to selection · `I` invert · `Esc` clear
- `⌘Z` undo · **Export PLY** downloads the edited cloud

The synthetic demo space is not checked in — generate it once with
`python3 scripts/make_sample.py` (the hosted deployment generates it at
build time).

## Layout

```
pipeline/agent.py         runs every stage, adapts, and asks Claude for judgement
pipeline/advisor.py       Claude access for the agent (CLI, API, or offline)
pipeline/reconstruct.py   capture -> cloud.ply (wraps COLMAP + ffmpeg)
pipeline/densify.py       MoGe-2 depth fused over COLMAP poses -> cloud-dense.ply
pipeline/semantics.py     open-vocabulary object detection for keyframes (local)
pipeline/shapes.py        planes and boxes from the dense cloud
pipeline/splat_seed.py    OpenSplat project seeded from the dense cloud
pipeline/splat_export.py  compact .splat for the viewer, and its starting camera
tools/splat_edit.py       remove objects from a splat and add furniture to it
tools/object_frames.py    the frames that show each object best, for Claude's review
pipeline/pointcloud.py    PLY I/O, voxel downsample, outlier removal (numpy)
tools/                    shape classifier, Blender room, OpenSplat binary + metallib
scripts/process_video.sh  the full chain, video -> splat
index.html, *.js, styles.css   the public website: landing page with 3D previews
editor/index.html         browser point-cloud editor (Three.js)
scripts/make_sample.py    synthetic demo room
spaces/<name>/            one folder per captured space
```

## Note: numpy on Python 3.14

numpy older than 2.3 silently corrupts array arithmetic on Python 3.14
(`b = a + scalar` can mutate `a` in place for large arrays inside functions —
a temporary-elision bug). This machine was hit by it and numpy was upgraded
to 2.5.2. If you recreate the environment, make sure `numpy >= 2.3`.

## Where this can go next

- **Harder captures**: when even the global mapper splits a capture, solve
  poses on Colab with MP-SfM or MapAnything (Apache weights), or log ARKit
  poses at capture time.
- **Better walls in splats**: train on Colab with depth and normal priors
  (gsplat / DN-Splatter / PlanarGS) and exposure compensation (PPISP).
- **Learned structure**: PlanarSplatting for planes, SpatialLM for doors,
  windows and furniture boxes.
- **Meshing**: Poisson reconstruction over the dense cloud for solid
  surfaces (Open3D once it supports this Python, or CloudCompare today).
- **Editor**: lasso selection, plane snapping, measurements, per-space
  gallery.
