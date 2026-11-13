#!/usr/bin/env python3
"""Run the pipeline's GPU work on Kaggle's free GPUs, from here.

Kaggle gives 30 GPU hours a week (two T4s, 9-hour sessions), but a Kaggle
notebook runs as a batch job: it cannot ask Claude anything while it runs, and
Claude cannot reach into it. So the work is split by what each step needs:

  - here, with Claude: the camera solve, naming the room's objects, marking
    them on the frames (densify.py --marks-only, kept in marks.json), the
    structure review and every other judgement;
  - on Kaggle, without Claude: the GPU steps (depth and SAM 3 tracking, splat
    training, fills, models), each one job that starts from the space as it
    stands here and sends its new and changed files back.

A job: this repository's working tree and the space's inputs go up as two
private Kaggle datasets (new versions each run), a notebook script is pushed
that installs what the job needs, runs it, and packs every file of the space it
created or changed into result.tar.gz; this script waits for it, downloads that
and unpacks it into spaces/<name>.

Needs a Kaggle token (KAGGLE_API_TOKEN, ~/.kaggle/access_token, or a cloud
session's network secret: tools/kaggle_cli.py), the account's phone verified
(no GPU or internet otherwise), and for SAM 3 a Hugging Face token saved in the
account's notebook secrets as HF_TOKEN.

Usage:
    python3 tools/kaggle_pipeline.py spaces/bedroom --job densify --user <kaggle name>
    python3 tools/kaggle_pipeline.py spaces/bedroom --job sam31-test --user <kaggle name>
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KAGGLE = [sys.executable, str(ROOT / "tools" / "kaggle_cli.py")]
POLL_SECONDS = 30
KERNEL = "oasis-probe"            # the account's notebook with HF_TOKEN attached (Add-ons > Secrets)
MAX_HOURS = 9                     # Kaggle ends a GPU session after 9 hours

# What every job installs: the notebook's Python cell (MoGe-2 pinned) without Colab's magics.
INSTALL_PYTHON = """
pip -q install 'numpy>=2.3' 'transformers>=5' plyfile kernels   # kernels: SAM 3's mask clean-up
pip -q install --no-deps \\
  "git+https://github.com/EasternJournalist/utils3d.git@3fab839f0be9931dac7c8488eb0e1600c236e183" \\
  "git+https://github.com/microsoft/MoGe.git@925b8ed835a7a9cdb7578ba15c658a0afc969030"
"""
# MoGe-3 needs MoGe's current code (it keeps v2 too) and FlexGEMM; Depth Anything 3
# is installed without its long dependency list (xformers would replace PyTorch).
INSTALL_DEPTH_NEW = """
pip -q install click scipy trimesh einops omegaconf safetensors imageio e3nn addict pycolmap evo \
  "utils3d_moge @ git+https://github.com/EasternJournalist/utils3d-moge.git@62f09d58509485564e24d5d9f6aac9ee9ebc0c37" \
  "pipeline @ git+https://github.com/EasternJournalist/pipeline.git@1c511390d90226c00c101f34b84df26a0f8789b4" \
  "flex-gemm @ git+https://github.com/JeffreyXiang/FlexGEMM.git@b2fadb29d41846c7981ade6801ffc689fae119cf"
pip -q install --no-deps --force-reinstall "git+https://github.com/microsoft/MoGe.git"
pip -q install --no-deps --ignore-requires-python "git+https://github.com/ByteDance-Seed/Depth-Anything-3.git"   # it declares python <= 3.13, which pip reads as 3.13.0
"""
# OpenSplat (AGPL-3.0), built against the machine's PyTorch for its GPU, pinned to the
# commit the pipeline was tested with (the notebook's install-opensplat cell).
INSTALL_OPENSPLAT = """
apt-get -qq -o DPkg::Lock::Timeout=600 update > /dev/null
apt-get -qq -o DPkg::Lock::Timeout=600 install -y libopencv-dev > /dev/null
# A binary built by an earlier job comes back with the space (tools-cache/).
CACHED=$(ls /kaggle/temp/repo/spaces/*/tools-cache/opensplat 2>/dev/null | head -1)
if [ -n "$CACHED" ] && "$CACHED" --help > /dev/null 2>&1; then
  mkdir -p /kaggle/temp/OpenSplat/build && cp "$CACHED" /kaggle/temp/OpenSplat/build/opensplat
fi
if [ ! -x /kaggle/temp/OpenSplat/build/opensplat ]; then
  cd /kaggle/temp && rm -rf OpenSplat
  git clone -q https://github.com/WebODM/OpenSplat.git
  cd OpenSplat && git checkout -q 2947938b7a2152b560489a963e47e961825dd32a
  export PATH=/usr/local/cuda/bin:$PATH
  TORCH_PREFIX=$(python -c "import torch; print(torch.utils.cmake_prefix_path)")
  ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
  # The driver's libcuda.so.1 is there, but not the plain libcuda.so the linker asks for.
  mkdir -p /kaggle/temp/cudalink
  CUDA_LIB=$(ldconfig -p | grep 'libcuda.so.1 ' | awk '{print $NF}' | head -1)
  [ -n "$CUDA_LIB" ] || CUDA_LIB=$(find /usr/local/cuda* -name 'libcuda.so' -path '*stubs*' 2>/dev/null | head -1)
  ln -sf "$CUDA_LIB" /kaggle/temp/cudalink/libcuda.so
  export LIBRARY_PATH=/kaggle/temp/cudalink:$LIBRARY_PATH
  mkdir -p build && cd build
  cmake -DCMAKE_PREFIX_PATH="$TORCH_PREFIX" -DGPU_RUNTIME=CUDA \\
        -DCMAKE_EXE_LINKER_FLAGS="-L/kaggle/temp/cudalink" \\
        -DCMAKE_CUDA_ARCHITECTURES="$ARCH" -DCMAKE_BUILD_TYPE=Release .. > cmake.log 2>&1 || { tail -40 cmake.log; exit 1; }
  echo "Building OpenSplat (about 10 minutes)..."
  make -j"$(nproc)" opensplat > make.log 2>&1 || { tail -60 make.log; exit 1; }
fi
ls -la /kaggle/temp/OpenSplat/build/opensplat
"""
# gsplat (Apache-2.0), compiled for the machine's GPU and PyTorch; a tiny render
# right after proves the build before hours of training depend on it.
INSTALL_GSPLAT = """
export PATH=/usr/local/cuda/bin:$PATH
export TORCH_CUDA_ARCH_LIST=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
export MAX_JOBS=$(nproc)
pip -q install ninja jaxtyping rich
echo "Building gsplat (about 10-20 minutes)..."
pip install --no-build-isolation "git+https://github.com/nerfstudio-project/gsplat.git@v1.5.3" > /kaggle/temp/gsplat-build.log 2>&1 \
  || { tail -80 /kaggle/temp/gsplat-build.log; exit 1; }
python3 -c "
import torch; from gsplat import rasterization
n = 1000; d = 'cuda'
r, a, _ = rasterization(torch.randn(n, 3, device=d) + torch.tensor([0, 0, 5.], device=d),
    torch.nn.functional.normalize(torch.randn(n, 4, device=d), dim=-1), torch.rand(n, 3, device=d) * 0.05,
    torch.rand(n, device=d), torch.rand(n, 3, device=d), torch.eye(4, device=d)[None],
    torch.tensor([[[300., 0, 160], [0, 300, 120], [0, 0, 1]]], device=d), 320, 240)
print('gsplat renders:', tuple(r.shape), float(a.mean()))"
"""
INSTALL_SAM31 = """
pip -q install einops 'git+https://github.com/facebookresearch/sam3.git'
pip -q install 'numpy>=2.3'     # sam3 pulls numpy 1.26, which the rest no longer builds against
"""

# Each job: the space's files it needs (globs under spaces/<name>), what it installs,
# and what it runs (in the repository's root, with SPACE set to spaces/<name>).
JOBS = {
    "densify": {
        "inputs": ["objects.json", "workspace/images/*.jpg", "workspace/sparse/0/*",
                   "workspace/claude-marks/marks.json", "workspace/dropped-frames.json"],
        "install": [INSTALL_PYTHON, INSTALL_SAM31],      # SAM 3.1 first, SAM 3 if it will not load
        "run": ['python3 pipeline/densify.py "$SPACE" --detector claude --track on --tracker auto'],
        # SAM 3 tracks the whole walk in chunks (pipeline/tracking.py); both T4s are not needed.
    },
    # The two newer depth networks on the same keyframes, beside the MoGe-2 cloud,
    # for tools/wall_thickness.py to compare. No labels: depth only.
    "depth-compare": {
        "inputs": ["objects.json", "densify.json", "workspace/images/*.jpg", "workspace/sparse/0/*"],
        "install": [INSTALL_PYTHON, INSTALL_DEPTH_NEW],
        "run": ['python3 pipeline/densify.py "$SPACE" --depth-model moge3 --no-semantics --output "$SPACE/cloud-moge3.ply"; '
                'python3 pipeline/densify.py "$SPACE" --depth-model da3 --no-semantics --output "$SPACE/cloud-da3.ply"; '
                'ls "$SPACE"/cloud-*.ply'],
    },
    "depth-da3": {
        "inputs": ["objects.json", "densify.json", "workspace/images/*.jpg", "workspace/sparse/0/*"],
        "install": [INSTALL_PYTHON, INSTALL_DEPTH_NEW],
        "run": ['python3 pipeline/densify.py "$SPACE" --depth-model da3 --no-semantics --output "$SPACE/cloud-da3.ply"'],
    },
    # Stage 4's training: the quick and the long splat, without Claude (the
    # choices between trainings, fills and scenes are made here afterwards).
    "splat-train": {
        "inputs": ["objects.json", "densify.json", "shapes.json", "cloud-dense.ply", "agent-report.json",
                   "frames-chosen.json", "workspace/images/*.jpg", "workspace/sparse/0/*", "tools-cache/opensplat"],
        "install": [INSTALL_PYTHON, INSTALL_OPENSPLAT],
        "run": ['mkdir -p "$SPACE/tools-cache" && cp -n /kaggle/temp/OpenSplat/build/opensplat "$SPACE/tools-cache/" ; '
                'OPENSPLAT=/kaggle/temp/OpenSplat/build/opensplat python3 pipeline/agent.py "$SPACE/workspace/images" '
                '--name "$(basename "$SPACE")" --stage splat --splat-steps train-quick train-long --long-splat on '
                '--no-claude --accept-warnings'],
    },
    # The gsplat training (exposure per frame, MoGe depth, full resolution), then
    # every splat scored on the frames it held out (pipeline/splat_gsplat.py).
    "splat-gsplat": {
        "inputs": ["cloud-dense.ply", "workspace/images/*.jpg", "workspace/sparse/0/*",
                   "splat-quick.ply", "splat-long.ply"],
        "install": [INSTALL_PYTHON, INSTALL_GSPLAT],
        "run": ['python3 pipeline/splat_seed.py "$SPACE" && '
                'python3 pipeline/splat_gsplat.py train "$SPACE" --minutes 390 && '
                'python3 pipeline/splat_gsplat.py eval "$SPACE" splat-quick.ply splat-long.ply splat-gsplat.ply'
                ' $(ls "$SPACE"/splat-gsplat_*.ply 2>/dev/null | xargs -rn1 basename)'],
    },
    # Continue the gsplat training to 30000 steps on the first GPU while OpenSplat's quick
    # and long runs are trained on the second WITHOUT the held-out frames: then every
    # splat is scored on frames none of them saw, and the novel-view strips show what a
    # web copy loses (flat colour only; cropped to the room and reduced to 300k splats).
    "splat-resume": {
        "inputs": ["cloud-dense.ply", "workspace/images/*.jpg", "workspace/sparse/0/*", "scene/scene.json",
                   "splat-quick.ply", "splat-long.ply", "splat-gsplat.ply", "splat-gsplat.json",
                   "tools-cache/opensplat"],
        "install": [INSTALL_PYTHON, INSTALL_GSPLAT, INSTALL_OPENSPLAT],
        "run": ["""set -x
python3 pipeline/splat_seed.py "$SPACE"
python3 pipeline/splat_seed.py "$SPACE" --hold-out "$SPACE/splat-gsplat.json"
OPENSPLAT=/kaggle/temp/OpenSplat/build/opensplat
( CUDA_VISIBLE_DEVICES=1 $OPENSPLAT "$SPACE/splat-project-heldout" -n 10000 -d 4 -o "$SPACE/splat-quick-heldout.ply" \
  && CUDA_VISIBLE_DEVICES=1 $OPENSPLAT "$SPACE/splat-project-heldout" -n 30000 -d 2 -o "$SPACE/splat-long-heldout.ply" ) \
  > "$SPACE/baseline-heldout.log" 2>&1 &
CUDA_VISIBLE_DEVICES=0 python3 pipeline/splat_gsplat.py train "$SPACE" --steps 30000 --resume splat-gsplat.ply \
  --start-step $(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['steps'])" "$SPACE/splat-gsplat.json") \
  --output splat-gsplat-30k.ply --minutes 330
rm -f "$SPACE"/splat-gsplat-30k_*.ply
wait
tail -3 "$SPACE/baseline-heldout.log"
python3 pipeline/splat_gsplat.py eval "$SPACE" splat-quick.ply splat-long.ply splat-quick-heldout.ply \
  splat-long-heldout.ply splat-gsplat.ply splat-gsplat-30k.ply splat-gsplat-30k.ply:dc splat-gsplat-30k.ply:dc300k
python3 pipeline/splat_gsplat.py novel "$SPACE" splat-gsplat-30k.ply splat-gsplat-30k.ply:dc splat-gsplat-30k.ply:dc300k
"""],
    },
    "sam31-test": {
        "inputs": ["objects.json", "workspace/images/*.jpg"],
        "install": [INSTALL_PYTHON, INSTALL_SAM31],
        "run": ['python3 tools/sam31_probe.py "$SPACE"'],
    },
}

KERNEL_SCRIPT = r'''
import glob, os, subprocess, sys, tarfile, time
from pathlib import Path
JOB = {job!r}
NAME = {name!r}
INSTALL = {install!r}
RUN = {run!r}
work = Path("/kaggle/working"); repo = Path("/kaggle/temp/repo"); repo.mkdir(parents=True, exist_ok=True)
def unpack(stem, dataset, into):
    """A dataset's archive, wherever Kaggle mounted it: as uploaded, or already unpacked."""
    subprocess.run("find /kaggle/input -maxdepth 4 | head -30", shell=True)
    archives = glob.glob(f"/kaggle/input/**/{{stem}}.tar.gz", recursive=True)
    if archives:
        with tarfile.open(archives[0]) as t: t.extractall(into, filter="data")
        return
    mounted = [d for d in glob.glob(f"/kaggle/input/**/{{dataset}}", recursive=True) if os.path.isdir(d)]
    if not mounted: raise SystemExit(f"{{dataset}} not found in /kaggle/input")
    source = Path(mounted[0])
    inner = [p for p in source.iterdir()]
    if len(inner) == 1 and inner[0].is_dir() and inner[0].name.startswith(stem):
        source = inner[0]
    subprocess.run(f"cp -r '{{source}}'/. '{{into}}'/", shell=True, check=True)
unpack("code", "oasis-code", repo)
space = repo / "spaces" / NAME; space.mkdir(parents=True, exist_ok=True)
unpack("space", "oasis-space-" + NAME, space)
try:
    from kaggle_secrets import UserSecretsClient
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
    print("Hugging Face token: from the notebook's secrets")
except Exception as exc:
    print(f"No Hugging Face token ({{exc}}); gated models will not download")
for step in INSTALL:
    subprocess.run(["bash", "-c", "set -e\n" + step], check=True)
started = time.time() - 2
env = dict(os.environ, SPACE=str(space.relative_to(repo)), PYTHONUNBUFFERED="1")
code = 0
for command in RUN:
    print(f"$ {{command}}", flush=True)
    code = subprocess.run(["bash", "-c", command], cwd=repo, env=env).returncode
    if code:
        print(f"exit {{code}}", flush=True)
        break
changed = [p for p in space.rglob("*") if p.is_file() and p.stat().st_mtime >= started]
with tarfile.open(work / "result.tar.gz", "w:gz") as t:
    for p in changed:
        t.add(p, arcname=str(p.relative_to(space)))
(work / "result.json").write_text(__import__("json").dumps(
    {{"job": JOB, "exit": code, "files": [str(p.relative_to(space)) for p in changed]}}))
print(f"{{len(changed)}} files changed; exit {{code}}")
'''


def kaggle(*args: str, check: bool = True) -> str:
    result = subprocess.run([*KAGGLE, *args], capture_output=True, text=True)
    out = (result.stdout + result.stderr).strip()
    if check and result.returncode:
        raise SystemExit(f"kaggle {' '.join(args)} failed:\n{out}")
    return out


def code_files() -> list[Path]:
    """The repository's files as they stand here, committed or not (git ls-files)."""
    listed = subprocess.run(["git", "ls-files", "-co", "--exclude-standard"], cwd=ROOT,
                            capture_output=True, text=True, check=True).stdout.split("\n")
    skip = ("spaces/", "videos/", "splats/", "third_party/")
    return [ROOT / f for f in listed if f and not f.startswith(skip) and (ROOT / f).is_file()]


def pack(files: list[Path], base: Path, out: Path) -> None:
    with tarfile.open(out, "w:gz") as t:
        for f in files:
            t.add(f, arcname=str(f.relative_to(base)))


def publish_dataset(slug: str, title: str, archive: Path, stem: str) -> None:
    """Create or version a private dataset holding one archive; wait until Kaggle has it ready."""
    with tempfile.TemporaryDirectory() as folder:
        folder = Path(folder)
        shutil.copy2(archive, folder / f"{stem}.tar.gz")
        (folder / "dataset-metadata.json").write_text(json.dumps(
            {"title": title, "id": slug, "licenses": [{"name": "other"}]}))
        exists = "ready" in kaggle("datasets", "status", slug, check=False).lower()
        if exists:
            print(kaggle("datasets", "version", "-p", str(folder), "-m", time.strftime("%Y-%m-%d %H:%M"),
                         "--dir-mode", "skip").splitlines()[-1])
        else:
            print(kaggle("datasets", "create", "-p", str(folder)).splitlines()[-1])
    for _ in range(120):
        time.sleep(10)
        if "ready" in kaggle("datasets", "status", slug, check=False).lower():
            return
    raise SystemExit(f"dataset {slug} did not become ready")


def run_job(space: Path, job: str, user: str, keep: Path) -> int:
    spec = JOBS[job]
    name = space.name
    inputs = sorted({p for pattern in spec["inputs"] for p in space.glob(pattern) if p.is_file()})
    missing = [pattern for pattern in spec["inputs"] if not any(space.glob(pattern))
               and not pattern.endswith("dropped-frames.json") and not pattern.startswith("tools-cache/")]
    if missing:
        raise SystemExit(f"{space} lacks what the {job} job needs: {', '.join(missing)}")
    with tempfile.TemporaryDirectory(dir=keep) as tmp:
        tmp = Path(tmp)
        code = code_files()
        pack(code, ROOT, tmp / "code.tar.gz")
        pack(inputs, space, tmp / "space.tar.gz")
        print(f"Uploading the code ({len(code)} files) and {name}'s inputs for {job} "
              f"({len(inputs)} files, {(tmp / 'space.tar.gz').stat().st_size / 1e6:.0f} MB)")
        publish_dataset(f"{user}/oasis-code", "oasis code", tmp / "code.tar.gz", "code")
        publish_dataset(f"{user}/oasis-space-{name}", f"oasis space {name}", tmp / "space.tar.gz", "space")
        kernel = tmp / "kernel"
        kernel.mkdir()
        (kernel / "job.py").write_text(KERNEL_SCRIPT.format(job=job, name=name, install=spec["install"],
                                                            run=spec["run"]))
        # Every job runs as the one notebook whose secrets hold HF_TOKEN: Kaggle
        # attaches a secret to a notebook by hand, and the API cannot.
        slug = f"{user}/{KERNEL}"
        (kernel / "kernel-metadata.json").write_text(json.dumps({
            "id": slug, "title": KERNEL.replace("-", " "), "code_file": "job.py", "language": "python",
            "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_internet": True,
            "machine_shape": "NvidiaTeslaT4",
            "dataset_sources": [f"{user}/oasis-code", f"{user}/oasis-space-{name}"]}))
        print(kaggle("kernels", "push", "-p", str(kernel)).splitlines()[-1])
    deadline = time.time() + MAX_HOURS * 3600
    status = ""
    while time.time() < deadline:
        time.sleep(POLL_SECONDS)
        status = kaggle("kernels", "status", slug, check=False).splitlines()[-1]
        if any(word in status.lower() for word in ("complete", "error", "cancel")):
            break
    print(status)
    out = keep / f"kaggle-{job}"
    out.mkdir(parents=True, exist_ok=True)
    kaggle("kernels", "output", slug, "-p", str(out), "--force", check=False)
    result = out / "result.tar.gz"
    if result.exists():
        with tarfile.open(result) as t:
            t.extractall(space, filter="data")
        summary = json.loads((out / "result.json").read_text())
        print(f"{len(summary['files'])} files back in {space}; the job exited {summary['exit']}")
        return summary["exit"]
    for log in sorted(out.glob("*.log")):
        print(f"No result came back; the end of the notebook's log ({log}):")
        print(log.read_text()[-4000:])
    return 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("space", type=Path)
    parser.add_argument("--job", choices=sorted(JOBS), required=True)
    parser.add_argument("--user", default=os.environ.get("KAGGLE_USERNAME"), required=not os.environ.get("KAGGLE_USERNAME"),
                        help="the Kaggle account's name (kaggle.com/<name>)")
    parser.add_argument("--keep", type=Path, default=Path(tempfile.gettempdir()) / "oasis-kaggle",
                        help="where the job's downloads and logs are kept")
    args = parser.parse_args()
    os.environ["KAGGLE_USERNAME"] = args.user
    args.keep.mkdir(parents=True, exist_ok=True)
    sys.exit(run_job(args.space.resolve(), args.job, args.user, args.keep))


if __name__ == "__main__":
    main()
