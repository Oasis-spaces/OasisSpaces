# Provenance of this copy (OasisSpaces)

This folder is a verbatim copy of **ARKit 3DGS Scanner** by Kuo Feng-Yuan
(KuoFengYuan), <https://github.com/KuoFengYuan/arkit-3dgs-scanner>, at commit
`d5d8e313890e9a08221e92fa9623bd5ccad00ab0` (30 September 2026, "enhance: copyright
headers, a bilingual copyright notice, and an independent rotmat2qvec (#55)"),
copied on 12 October 2026 with `git archive`.

## Why this commit, and why a copy

- At this commit `LICENSE` is the Apache License 2.0, and every Swift and Metal
  source file carries `SPDX-License-Identifier: Apache-2.0`.
- Later commits are under the PolyForm Noncommercial License 1.0.0, which forbids
  commercial use. The upstream `docs/LICENSING.md` (current head) states:
  "Material published through commit `d5d8e313890e9a08221e92fa9623bd5ccad00ab0`
  on 2026-09-30 was offered under Apache 2.0 ... It does not revoke Apache 2.0
  permissions already granted for previously published material, including
  commercial use."
- The upstream README links an `apache-2.0-final` tag, but the remote has no
  tags (`git ls-remote --tags` returned nothing on 12 October 2026). The commit
  survives only in `main`'s history, so a force-push upstream would lose it.
  This copy keeps it.

**Do not update this folder from upstream.** Anything after the commit above is
non-commercial. Later improvements (the author reports 1.39x faster training
steps and a `--benchmark-training` mode) need a written commercial licence from
the author.

## What was left out, and what was changed

- Left out: `docs/media/` (19 MB of demo images and recordings, so some
  pictures in the docs do not resolve), `AGENTS*.md` and `CONTRIBUTING*.md`
  (upstream's instructions for its own contributors and agents).
- Changed: nothing. This file is the only addition.

## Obligations (Apache-2.0, section 4)

Keep `LICENSE` and `NOTICE` with any copy or derivative, and credit
"Kuo Feng-Yuan (KuoFengYuan)" as the original author, including in the capture
app's acknowledgements screen once any of this code ships in it. Files we
modify must say that they were changed.

## What OasisSpaces wants from it

The owner's constraint (October 2026) is that reconstruction must not need a
server or Colab GPU, so it moves onto the phone. This trainer is the
best-documented open on-phone path for that:

- `arkit-3dgs-scanner/Training/`: a Swift + Metal Gaussian-splat trainer (MRNF
  densification, Mip filter, PPISP colour model), with memory plans that include
  4 GB phones, checkpoint and resume, and SOG/PLY export.
- `arkit-3dgs-scanner/Capture/`: camera-only pose refinement from ARKit poses
  (`CameraOnlyTracker`, `OfflinePoseRefinement`, `BundleAdjuster`,
  `LoopClosureRefiner`), surface TSDF and point-cloud fusion. These are the
  on-phone replacement for COLMAP and MapAnything.
- Only Apple frameworks: no libtorch, nothing AGPL.

Its input is a scan folder of JPEG frames, each with pinhole intrinsics and
ARKit's camera-to-world transform, row-major (`TrainingFrame` in
`Training/TrainingDataset.swift`). OasisCapture records the video plus
`frames.jsonl` with the same transform column-major and the same intrinsics, so
an adapter has to:

1. pick keyframes and write them as JPEG;
2. transpose the transform;
3. seed the Gaussians from our MoGe-2 + DepthFusion cloud (no LiDAR on the
   iPhone 13). The PocketGS paper measured 21.15 dB without a dense prior and
   23.67 dB with it.

`docs/audit-prompts-and-stack-2026-10.md` and `docs/on-device-plan-2026-10.md`
record the evidence and the measurements still to take.

**Not measured yet.** This snapshot has never been timed on an iPhone. The
author's published phone timings (iPhone 17 Pro, 553-769 s for 10k iterations)
come from later, non-commercial builds. The first job on a device is our own
timing and memory harness on the iPhone 13 (A15, 4 GB).
