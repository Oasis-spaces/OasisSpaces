# OasisSpaces

Turn photos of any space into an editable 3D point cloud.

**Run it for free:**
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sanskargupta1808/OasisSpaces/blob/main/notebooks/OasisSpaces_Colab.ipynb)
— reconstruction on Google Colab's free GPU (unlocks the dense step), and the
public website is hosted free at **https://oasisspaces.onrender.com**, and the
editor at **https://oasisspaces.onrender.com/editor/**.

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
pip3 install --break-system-packages 'numpy>=2.3'   # see note below
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
  for you.
- Avoid blur, bare white walls, mirrors, and glass (nothing to match on).
  Even, diffuse lighting works best.

## Building the cloud

```bash
# from a folder of photos
python3 pipeline/reconstruct.py ~/Pictures/living-room --name living-room

# from a walkthrough video (extracts 2 frames/sec by default)
python3 pipeline/reconstruct.py walkthrough.mp4 --name living-room --fps 2
```

Output: `spaces/living-room/cloud.ply` (plus the COLMAP workspace next to it
for reruns/debugging). Mapping takes minutes to hours depending on image
count — progress is logged to `spaces/<name>/workspace/colmap.log`.

`--dense` runs COLMAP's dense multi-view stereo after SfM for a far denser
cloud, but that step needs CUDA — on a Mac it is skipped with a note. The
sparse cloud is usually plenty to see and edit the space; for dense results,
copy the workspace to a CUDA machine and rerun with `--dense`.

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
pipeline/reconstruct.py   capture -> cloud.ply (wraps COLMAP + ffmpeg)
pipeline/pointcloud.py    PLY I/O, voxel downsample, outlier removal (numpy)
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

- **Dense clouds**: run the `--dense` step on a CUDA box, or swap in
  OpenMVS for CPU densification.
- **Gaussian splatting**: the COLMAP workspace (`workspace/sparse/0`) is
  exactly the input gsplat/OpenSplat need for photorealistic view synthesis.
- **Meshing**: Poisson reconstruction over the dense cloud for solid
  surfaces (Open3D once it supports this Python, or CloudCompare today).
- **Editor**: lasso selection, plane snapping, measurements, per-space
  gallery.
