# First device run: the iPhone 13 and the Mac

Step 1 of the order of work in `docs/on-device-plan-2026-10.md`, command by command. It builds the
capture app with this branch's new recording code, checks recordings off the phone, times the live
models on the phone, and runs the Core ML checks that need a Mac.

Who runs it: you, or a Claude Code session running on the Mac. Start one with
`claude remote-control` in the clone, or use the Claude Desktop app. A cloud session cannot reach the
Mac or the phone.

The Mac stays the Claude gateway. Nothing on the phone asks Claude yet. When phone stages need
reviews, they will send their questions to the Mac's station, as recordings go today.

## Before you start

- **The Mac:**
  - Xcode and XcodeGen (`brew install xcodegen`);
  - the clone of this repository.
  - the `~/.venvs/oasis-coreml` environment that `apps/OasisCapture/build.sh` converts the models with.
    It is a Python 3.12 virtual environment with:
    - torch 2.5.1;
    - transformers 4.46;
    - coremltools 8.3 (9 also works);
    - ultralytics 8.4 (`convert_objects.py`);
    - MoGe installed with `--no-deps` (`convert_depth_moge.py`);
    - MobileSAM (`convert_refiner.py`);
    - Pillow and numpy.

    Each converter's docstring names what it needs. If the environment already built the app's
    models before, it is ready.
- **The iPhone 13:**
  - paired with the Mac, unlocked, with Developer Mode on;
  - `xcrun devicectl list devices` lists it as `available (paired)`. Its Identifier column is the
    `<id>` used below.

## 1. Get the branch

```bash
cd <the clone>
git fetch origin
git checkout claude/eager-goldberg-9ixhyg
git pull
```

## 2. Build today's app with the new recording code

```bash
apps/OasisCapture/build.sh
```

This builds with MobileSAM, as before. It is the first time anyone has compiled the new code; none of
it was compiled when it was written. If the build fails, the errors print and the full log is
`apps/OasisCapture/build/xcodebuild.log`. The new code is in three files:

- `Sources/Recorder.swift`: keyframe anchors, motion samples, the jump flag and plane classes;
- `Sources/CaptureController.swift`: adding the anchors to the session, and passing on updates and
  removals;
- `Sources/SceneRunner.swift`: the EdgeTAM path. It compiles now; which refiner runs is decided at
  run time by the packages bundled, and step 6 bundles EdgeTAM.

Fix them, then commit and push the fixes to this branch.

## 3. Record three test captures

1. **A loop:** walk slowly round a room for 30-45 seconds and end where you started. This is the case
   where ARKit may correct itself, which the keyframe anchors should show.
2. **A turn:** 20 seconds, with one quick turn away and back.
3. **The wall test:** 20 seconds facing plain walls. This shows whether ARKit classifies planes on
   them.

Let each recording finish; tap Stop and wait for the review. A recording the app was closed in has no
`capture.json`.

**Copy them off the phone.** This sends nothing through the pipeline:

```bash
xcrun devicectl device info files --device <id> --domain-type appDataContainer \
    --domain-identifier com.oasisspaces.capture --subdirectory Documents/Captures
xcrun devicectl device copy from --device <id> --domain-type appDataContainer \
    --domain-identifier com.oasisspaces.capture --source Documents/Captures/<Capture-...> \
    --destination ~/oasis-device-run/<Capture-...>
```

If `copy from` will not take a folder, copy `frames.jsonl`, `capture.json` and `video.mov` one at a
time.

**Or send them to the Mac from the app's Scans tab.** This needs Splat Viewer built
(`apps/SplatViewer/build.sh`) and open. Its Phone captures section should say it is ready on this
network, with its Pipeline folder set to the clone, and the phone paired in the app's Mac tab. Each
send also queues a full pipeline run on the Mac.

- Sent over the local network, a recording lands in
  `~/Library/Application Support/Splat Viewer/Station/jobs/<id>/input/`.
- Sent through the cloud, it lands in `~/Library/Application Support/Splat Viewer/Station/Cloud/<id>/`.

## 4. Check each recording

```bash
python3 tools/check_phone_capture.py ~/oasis-device-run/<Capture-...>
```

Each line starts with ok, warn or fail, then says what it found:

- whether the recording finished;
- frames and tracking;
- whether the exposure held (measurement 7);
- the 100 Hz motion samples;
- jumps;
- the keyframe anchors: any missing, how many ARKit moved, and how far (measurement 2);
- ARKit's plane classes, and how much of the vertical area it classified.

`--json` prints everything.

## 5. Time the live models (measurement 1)

1. **The app's own timings.** The log keeps only the app's last launch. Copy it right after
   recording, before the app is opened again; if it was reopened, open Scan and record 30 seconds
   first.

   ```bash
   xcrun devicectl device copy from --device <id> --domain-type appDataContainer \
       --domain-identifier com.oasisspaces.capture --source Documents/oasis-capture.log \
       --destination oasis-capture.log
   grep -E "loaded in|analysis #|models ready" oasis-capture.log | tail -40
   ```

   The lines give each model's load time and the unit it was loaded for, and the milliseconds per
   analysed frame.
2. **Where each op runs, on the phone.** In Xcode, open each package in
   `apps/OasisCapture/Resources/`:
   - `RoomObjects`
   - `MaskEncoder` and `MaskDecoder`
   - `RoomSegmentation`
   - `RoomMetricDepth`

   On the Performance tab, add a report on the connected iPhone. Make one with compute units "All"
   (what the app uses today) and one with "CPU and Neural Engine". Note each report's median
   prediction time and the share of ops on the Neural Engine, GPU and CPU. The report covers time
   and placement only, not memory.

## 6. EdgeTAM in place of MobileSAM

Once:

```bash
git clone https://github.com/facebookresearch/EdgeTAM ~/EdgeTAM
SAM2_BUILD_CUDA=0 ~/.venvs/oasis-coreml/bin/pip install --no-build-isolation -e ~/EdgeTAM
~/.venvs/oasis-coreml/bin/pip install timm
```

Then:

```bash
rm -rf apps/OasisCapture/Resources/MaskEncoder.mlpackage apps/OasisCapture/Resources/MaskDecoder.mlpackage
REFINER=edgetam apps/OasisCapture/build.sh
~/.venvs/oasis-coreml/bin/python apps/OasisCapture/scripts/convert_refiner_edgetam.py \
    --check <frame.jpg> --box X0 Y0 X1 Y1 --point X Y --coreml
```

**The test image** must be a JPEG or PNG; Pillow cannot read HEIC. A frame from one of the
recordings works. ffmpeg turns it upright by itself:

```bash
ffmpeg -i ~/oasis-device-run/<Capture-...>/video.mov -vf "select=eq(n\,100)" -frames:v 1 frame.jpg
```

- The box and the point are in that image's pixels, around a TV or a sofa and on it. The check prints
  the image's size.
- Each image gets one line. Compare its "Core ML IoU" with the "stretch IoU" just before it: the
  Core ML packages run on the Neural Engine and CPU, and should agree closely with the PyTorch
  wrapper.
- If the conversion fails with coremltools 8.3, try coremltools 9. It converted with 9.0 in the
  cloud.

Then repeat step 5 for `MaskEncoder` and `MaskDecoder`, and record once with the app to see the
outlines.

## 7. The surface fill on the phone

```bash
~/.venvs/oasis-coreml/bin/python apps/OasisCapture/scripts/convert_fill_lama.py
~/.venvs/oasis-coreml/bin/python apps/OasisCapture/scripts/convert_fill_lama.py --check <two or three room photos>
```

- The first command fetches `big-lama.pt` (206 MB, GitHub) if it is missing.
- `--check` compares the Core ML package with PyTorch. The two should agree to well over 30 dB.
- **Time on the phone:** open `apps/OasisCapture/Resources/SurfaceFill.mlpackage` in Xcode and run
  its Performance report on the iPhone with compute units "CPU and GPU". That gives the A15's time
  per 512 fill.
- **Peak memory:** the report does not measure memory. That half of measurement 6 waits for the
  fill stage in the app, profiled with Instruments (Allocations).

## 8. Only then: the compute units

Choose per model, from step 5's two reports: a model whose "CPU and Neural Engine" report is as fast
as its "All" report moves to `.cpuAndNeuralEngine`.

1. In `SceneRunner.swift`, the call `Self.model(name, units: [.all, .cpuAndGPU])` sits in the loop
   over all five models. Make it choose by name, for example:

   ```swift
   neuralEngine.contains(name) ? [.cpuAndNeuralEngine, .cpuOnly] : [.all, .cpuAndGPU]
   ```

2. Fix the unit label in `model(_:units:)`.
3. Rebuild, and compare the analysis milliseconds in the log.

## What to send back

Paste these into the conversation, or commit them under `docs/` on this branch:

- `check_phone_capture.py` output for each recording;
- the log lines from step 5;
- per model, both Performance reports' median times and their Neural Engine, GPU and CPU splits;
- the `--check` output of steps 6 and 7, and SurfaceFill's Performance time;
- any build errors and how they were fixed.
