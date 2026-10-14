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
  - the `~/.venvs/oasis-coreml` environment from the README (Python 3.12, torch, transformers,
    coremltools);
  - the clone of this repository.
- **The iPhone 13:**
  - paired with the Mac, unlocked, with Developer Mode on;
  - `xcrun devicectl list devices` should list it as `available (paired)`.

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
- `Sources/SceneRunner.swift`: the EdgeTAM path, built in step 6.

Fix them, then commit and push the fixes to this branch.

## 3. Record three test captures

1. **A loop:** walk slowly round a room for 30-45 seconds and end where you started. This is the case
   where ARKit may correct itself, which the keyframe anchors should show.
2. **A turn:** 20 seconds, with one quick turn away and back.
3. **The wall test:** 20 seconds facing plain walls. This shows whether ARKit classifies planes on
   them.

Send each one to the Mac from the app's Scans tab. Each lands in
`~/Library/Application Support/Splat Viewer/Station/jobs/<id>/input/` (`video.mov`, `frames.jsonl`,
`capture.json`).

## 4. Check each recording

```bash
python3 tools/check_phone_capture.py ~/"Library/Application Support/Splat Viewer/Station/jobs/<id>/input"
```

Each line ends in ok, warn or fail:

- frames and tracking;
- whether the exposure held (measurement 7);
- the 100 Hz motion samples;
- jumps;
- the keyframe anchors: how many ARKit moved, and how far (measurement 2);
- ARKit's plane classes, and how much of the vertical area it classified.

`--json` prints everything.

## 5. Time the live models (measurement 1)

1. **The app's own timings.** Copy the app's log:

   ```bash
   xcrun devicectl device copy from --device <id> --domain-type appDataContainer \
       --domain-identifier com.oasisspaces.capture --source Documents/oasis-capture.log \
       --destination oasis-capture.log
   grep -E "loaded in|analysis #|models ready" oasis-capture.log | tail -40
   ```

   The lines give each model's load time and compute unit, and the milliseconds per analysed frame.
2. **Where each op runs, on the phone.** In Xcode, open each package in
   `apps/OasisCapture/Resources/`:
   - `RoomObjects`
   - `MaskEncoder` and `MaskDecoder`
   - `RoomSegmentation`
   - `RoomMetricDepth`

   On the Performance tab, add a report on the connected iPhone. Note the median prediction time and
   the share of ops on the Neural Engine, GPU and CPU.

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
    --check <a room photo with a TV or a sofa> --box X0 Y0 X1 Y1 --point X Y --coreml
```

`--coreml` runs the fp16 packages on the Neural Engine and CPU and compares their masks with
EdgeTAM's own predictor. The IoU should be close to the PyTorch line above it.

- If the conversion fails on the environment's coremltools 8.3, try coremltools 9. It converted
  with 9.0 in the cloud.

Then repeat step 5 for `MaskEncoder` and `MaskDecoder`, and record once with the app to see the
outlines.

## 7. The surface fill on the phone

```bash
~/.venvs/oasis-coreml/bin/python apps/OasisCapture/scripts/convert_fill_lama.py
~/.venvs/oasis-coreml/bin/python apps/OasisCapture/scripts/convert_fill_lama.py --check <two or three room photos>
```

- The first command fetches `big-lama.pt` (206 MB, GitHub) if it is missing.
- `--check` compares the Core ML package with PyTorch. The two should agree to well over 30 dB.
- Then open `apps/OasisCapture/Resources/SurfaceFill.mlpackage` in Xcode and run its Performance
  report on the iPhone. That is the A15's time and memory per 512 fill (measurement 6), without the
  app.

## 8. Only then: the compute units

If step 5 shows a model's ops mostly on the Neural Engine, load it with `.cpuAndNeuralEngine`
instead of `.all`:

- in `SceneRunner.swift` (`Self.model(name, units: [.all, .cpuAndGPU])`), make it
  `[.cpuAndNeuralEngine, .cpuOnly]` and fix the unit label in `model(_:units:)`;
- rebuild, and compare the analysis milliseconds in the log.

## What to send back

Paste these into the conversation, or commit them under `docs/` on this branch:

- `check_phone_capture.py` output for each recording;
- the log lines from step 5;
- per model, the Performance report's median time and its Neural Engine, GPU and CPU split;
- the `--check` output of steps 6 and 7, and SurfaceFill's Performance numbers;
- any build errors and how they were fixed.
