#!/usr/bin/env python3
"""Run the whole pipeline on a Colab GPU from this Mac, stage by stage, with
Claude's checks answered here.

Everything the notebook does by hand, and what the first terminal-driven run
taught (Sep 2026), done the same way every time:

  - a Colab session from Google's Colab CLI (`colab`, signed in once);
  - this repository's current commit uploaded (the VM does not clone GitHub,
    so it runs exactly the code here, pushed or not);
  - the notebook's install cells (COLMAP, MoGe-2 and LaMa, Blender, OpenSplat)
    run in parallel in the background on the VM;
  - the video uploaded in 20 MB parts (one large upload fails) and checked by md5;
  - tools/claude_relay.py started here, so every Claude check of the agent on
    the VM (object naming and label review, structure review and render check,
    start views, fill review, splat choices) is answered with this Mac's
    `claude` login;
  - pipeline/agent.py run one stage at a time; after every stage the space's
    new and changed files come back to spaces/<name> here, also in checked
    parts, because a free Colab session can die within the hour and takes
    the VM's disk with it;
  - at the end, tools/splat_choose.py here compares the new splats with every
    earlier splat of the same video and publishes the best to splats/.

Stage 4 runs as its steps, each its own job, fetched before the next:
train-quick, train-long (saved every 10,000 steps and resumed from the newest
save), choose-training, fill; choose-best then runs here. A session installs
only what its steps need, and the OpenSplat binary built on Colab is kept in
tools/colab-cache/ and reused by later sessions with the same PyTorch and GPU.

If the session dies, run the same command again with --stages from the step
that did not finish: a new session gets the code and the space as it stands
here, and carries on.

Usage:
    python3 tools/colab_pipeline.py videos/IMG_4182.MOV --name walkthrough-colab3
    python3 tools/colab_pipeline.py videos/IMG_4182.MOV --name walkthrough-colab3 --stages train-long choose-training fill
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Stages 1-3, then stage 4 as its steps (pipeline/agent.py SPLAT_STEPS), each
# its own job on the VM with its results fetched before the next starts. The
# last step, choosing the best splat of the video, runs here after the fetch,
# where every earlier splat of the video is.
STEPS = ["reconstruct", "densify", "shapes", "train-quick", "train-long", "choose-training", "fill"]
SPLAT_STEPS = STEPS[3:]
# The notebook install cells each step needs. OpenSplat links OpenCV, which the
# COLMAP cell installs, so training needs that cell too.
NEEDS = {
    "reconstruct": ["install-colmap", "install-python"],
    "densify": ["install-python"],
    "shapes": ["install-python", "install-blender"],
    "train-quick": ["install-colmap", "install-python", "install-opensplat"],
    "train-long": ["install-colmap", "install-python", "install-opensplat"],
    "choose-training": ["install-python"],
    "fill": ["install-python"],
}
CACHE = ROOT / "tools" / "colab-cache"   # the OpenSplat binary built on Colab, reused
OPENSPLAT_BIN = "/content/OpenSplat/build/opensplat"
REMOTE_ROOT = "/content/OasisSpaces"
REMOTE_VIDEOS = "/content/videos"
REMOTE_WORK = "/content/oasis-run"
RELAY = "/content/claude-relay"
PART_BYTES = 20 * 1024 * 1024
INSTALL_CELLS = ["install-colmap", "install-python", "install-blender", "install-opensplat"]
# What the notebook's helper cell sets before running the agent.
REMOTE_ENV = {
    "BLENDER": "/opt/blender/blender",
    "OPENSPLAT": "/content/OpenSplat/build/opensplat",
    "OASIS_RENDER_ENGINE": "CYCLES",
    "OASIS_CLAUDE_RELAY": RELAY,
}
# Kept on the VM only: COLMAP's database is rebuilt by reconstruct.py, and the
# splat project is links plus a seed file rebuilt by splat_seed.py.
NOT_FETCHED = ("workspace/database.db", "splat-project/")
FETCH_DURING_MINUTES = 10


EXEC_ATTEMPTS = 4


class SessionEnded(RuntimeError):
    """The Colab session is gone (and the VM's disk with it)."""


def log(text: str) -> None:
    print(time.strftime("%H:%M:%S"), text, flush=True)


class Colab:
    """One Colab CLI session."""

    def __init__(self, session: str):
        self.session = session

    def cli(self, *args: str, timeout: float = 900) -> subprocess.CompletedProcess:
        return subprocess.run(["colab", *args], capture_output=True, text=True, timeout=timeout)

    def alive(self) -> bool:
        listed = self.cli("sessions", timeout=120)
        return listed.returncode == 0 and self.session in listed.stdout

    def create(self, gpu: str) -> None:
        log(f"creating Colab session {self.session} ({gpu})")
        made = self.cli("new", "-s", self.session, "--gpu", gpu, timeout=900)
        if made.returncode != 0 or not self.alive():
            sys.exit(f"could not create the session:\n{made.stdout[-2000:]}\n{made.stderr[-2000:]}")

    def python(self, code: str, timeout: float = 300) -> str:
        """Run Python in the session's kernel; returns what it printed. A call
        lost to the network (a dropped connection, a busy VM) is tried again;
        a session that has ended stops the run with that said."""
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(code)
            path = f.name
        try:
            for attempt in range(EXEC_ATTEMPTS):
                try:
                    result = self.cli("exec", "-s", self.session, "-f", path, "--timeout", str(timeout),
                                      timeout=timeout + 120)
                except subprocess.TimeoutExpired:
                    result = None
                if result is not None and result.returncode == 0:
                    return "\n".join(line for line in result.stdout.splitlines()
                                     if not line.startswith("[colab]"))
                detail = (result.stderr[-1200:] + result.stdout[-600:]) if result else "no reply"
                if "not found" in detail and "Session" in detail or not self.alive():
                    raise SessionEnded(f"the Colab session {self.session} has ended")
                log(f"  colab exec did not answer (attempt {attempt + 1}/{EXEC_ATTEMPTS}); retrying")
                time.sleep(30 * (attempt + 1))
            raise RuntimeError(f"colab exec failed: {detail}")
        finally:
            Path(path).unlink()

    def shell(self, command: str, timeout: float = 300) -> str:
        """Run a shell command on the VM (always from /content: exec resets the
        working directory for every call)."""
        code = (f"import subprocess\n"
                f"r = subprocess.run({command!r}, shell=True, cwd='/content', capture_output=True, "
                f"text=True, executable='/bin/bash')\n"
                f"print(r.stdout[-30000:]); print(r.stderr[-8000:])\n"
                f"print('EXIT', r.returncode)\n")
        out = self.python(code, timeout)
        lines = out.rstrip().splitlines()
        if not lines or not lines[-1].startswith("EXIT "):
            raise RuntimeError(f"no exit status from: {command}\n{out[-1500:]}")
        if lines[-1] != "EXIT 0":
            raise RuntimeError(f"failed on the VM: {command}\n{out[-3000:]}")
        return "\n".join(lines[:-1])

    def running(self, name: str) -> bool:
        """A job started by background() whose process is still alive."""
        # By its pid file, or by its command line for jobs started without one;
        # the [/] keeps pgrep from matching the shell running this very check.
        pattern = f"[/]{REMOTE_WORK.lstrip('/')}/{name}.log"
        out = self.shell(f"if ! test -f {REMOTE_WORK}/{name}.exit && "
                         f"{{ kill -0 $(cat {REMOTE_WORK}/{name}.pid 2>/dev/null) 2>/dev/null || "
                         f"pgrep -f {shlex.quote(pattern)} > /dev/null; }}; then echo alive; fi",
                         timeout=120)
        return out.strip() == "alive"

    def background(self, command: str, name: str) -> bool:
        """Start a shell command on the VM that outlives the exec call; its
        output goes to REMOTE_WORK/<name>.log and its exit code to <name>.exit.
        A job of that name still running is left alone (a re-run of this
        script must not start a second install into the same folder).
        Returns whether it started."""
        if self.running(name):
            log(f"  {name} is already running on the VM")
            return False
        wrapped = f"({command}) > {REMOTE_WORK}/{name}.log 2>&1; echo $? > {REMOTE_WORK}/{name}.exit"
        # Started from the kernel fully detached (own session, no pipes back),
        # so the exec call returns at once. A shell "a && b && nohup c &" would
        # background the whole chain with the exec's output pipes still open,
        # and the kernel would wait for the job to end before answering anything.
        self.python(f"import os, subprocess\n"
                    f"os.makedirs('{REMOTE_WORK}', exist_ok=True)\n"
                    f"exit_file = '{REMOTE_WORK}/{name}.exit'\n"
                    f"os.path.exists(exit_file) and os.remove(exit_file)\n"
                    f"p = subprocess.Popen(['bash', '-c', {wrapped!r}], stdin=subprocess.DEVNULL, "
                    f"stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True, "
                    f"cwd='/content')\n"
                    f"open('{REMOTE_WORK}/{name}.pid', 'w').write(str(p.pid))\n"
                    f"print('started', p.pid)\n", timeout=120)
        return True

    def finished(self, name: str) -> int | None:
        out = self.shell(f"cat {REMOTE_WORK}/{name}.exit 2>/dev/null || true", timeout=120).strip()
        return int(out) if out.strip().lstrip("-").isdigit() else None

    def log_since(self, name: str, offset: int) -> tuple[str, int]:
        code = (f"import os\np = '{REMOTE_WORK}/{name}.log'\n"
                f"data = open(p, 'rb').read() if os.path.exists(p) else b''\n"
                f"print(len(data)); print(data[{offset}:].decode('utf-8', 'replace')[-20000:])\n")
        out = self.python(code, timeout=120)
        first, _, rest = out.partition("\n")
        return rest, int(first.strip()) if first.strip().isdigit() else offset

    def md5(self, remote: str) -> str | None:
        out = self.shell(f"md5sum {shlex.quote(remote)} 2>/dev/null | cut -d' ' -f1 || true", timeout=600)
        return out.strip() or None

    def put(self, local: Path, remote: str) -> None:
        """Upload in PART_BYTES parts, join on the VM and check the md5; a file
        already there with the same md5 is not sent again."""
        digest = file_md5(local)
        if self.md5(remote) == digest:
            log(f"  {local.name} already on the VM")
            return
        size = local.stat().st_size
        parts = (size + PART_BYTES - 1) // PART_BYTES
        staging = f"{REMOTE_WORK}/parts/{local.name}"
        self.shell(f"rm -rf {shlex.quote(staging)} && mkdir -p {shlex.quote(staging)} "
                   f"$(dirname {shlex.quote(remote)})", timeout=120)
        with tempfile.TemporaryDirectory() as tmp, open(local, "rb") as src:
            for n in range(parts):
                part = Path(tmp) / f"part-{n:04d}"
                part.write_bytes(src.read(PART_BYTES))
                for attempt in range(4):
                    sent = self.cli("upload", "-s", self.session, str(part), f"{staging}/part-{n:04d}")
                    if sent.returncode == 0:
                        break
                    log(f"  part {n + 1}/{parts} failed ({sent.stderr.strip()[-100:]}), retrying")
                    time.sleep(5 * (attempt + 1))
                else:
                    raise RuntimeError(f"could not upload {local.name}")
                part.unlink()
                if parts > 3 and (n + 1) % 5 == 0:
                    log(f"  {local.name}: {n + 1}/{parts} parts")
        self.shell(f"cat {shlex.quote(staging)}/part-* > {shlex.quote(remote)} && rm -rf {shlex.quote(staging)}",
                   timeout=900)
        if self.md5(remote) != digest:
            raise RuntimeError(f"{local.name} arrived damaged (md5 differs)")
        log(f"  uploaded {local.name} ({size / 1e6:.0f} MB, {parts} parts, md5 checked)")

    def get(self, remote: str, local: Path) -> None:
        """Download in parts (split on the VM), join here and check the md5."""
        digest = self.md5(remote)
        if digest is None:
            raise RuntimeError(f"{remote} is not on the VM")
        staging = f"{REMOTE_WORK}/out-parts"
        names = self.shell(f"rm -rf {staging} && mkdir -p {staging} && "
                           f"split -b {PART_BYTES} -d -a 4 {shlex.quote(remote)} {staging}/part- && "
                           f"ls {staging}", timeout=900).split()
        with open(local, "wb") as out:
            for n, name in enumerate(sorted(names)):
                with tempfile.TemporaryDirectory() as tmp:
                    target = Path(tmp) / name
                    for attempt in range(4):
                        got = self.cli("download", "-s", self.session, f"{staging}/{name}", str(target))
                        if got.returncode == 0 and target.exists():
                            break
                        log(f"  download of part {n + 1}/{len(names)} failed, retrying")
                        time.sleep(5 * (attempt + 1))
                    else:
                        raise RuntimeError(f"could not download {remote}")
                    out.write(target.read_bytes())
        if file_md5(local) != digest:
            raise RuntimeError(f"{remote} arrived damaged (md5 differs)")


def file_md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def install_scripts() -> dict[str, str]:
    """The notebook's install cells, as bash scripts."""
    notebook = json.loads((ROOT / "notebooks" / "OasisSpaces_Colab.ipynb").read_text())
    scripts = {}
    for cell in notebook["cells"]:
        if cell.get("id") in INSTALL_CELLS:
            source = "".join(cell["source"])
            if not source.startswith("%%bash"):
                raise SystemExit(f"notebook cell {cell['id']} is not a bash cell")
            scripts[cell["id"]] = source.split("\n", 1)[1]
    missing = set(INSTALL_CELLS) - set(scripts)
    if missing:
        raise SystemExit(f"notebook install cells not found: {sorted(missing)}")
    return scripts


def upload_code(vm: Colab) -> str:
    """This repository's HEAD commit, unpacked over REMOTE_ROOT (spaces there are kept)."""
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                           capture_output=True, text=True).stdout
    changed = [line for line in dirty.splitlines() if not line.endswith(".claude/launch.json")]
    if changed:
        log("  note: uncommitted changes are not uploaded: " + ", ".join(l[3:] for l in changed))
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True).stdout.strip()
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / f"oasis-{commit}.tar"
        subprocess.run(["git", "archive", "--format=tar", "-o", str(archive), "HEAD"], cwd=ROOT, check=True)
        vm.put(archive, f"{REMOTE_WORK}/code.tar")
    vm.shell(f"mkdir -p {REMOTE_ROOT} && tar -xf {REMOTE_WORK}/code.tar -C {REMOTE_ROOT} && "
             f"echo {commit} > {REMOTE_ROOT}/COMMIT", timeout=300)
    return commit


def opensplat_key(vm: Colab) -> str:
    """What a built OpenSplat binary depends on: Colab's PyTorch and the GPU."""
    out = vm.shell("python -c 'import torch; print(torch.__version__)' && "
                   "nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1", timeout=300)
    torch_version, cap = (out.split() + ["?", "?"])[:2]
    return f"opensplat-torch{torch_version.replace('+', '_')}-sm{cap.replace('.', '')}"


def start_installs(vm: Colab, needed: list[str]) -> None:
    scripts = install_scripts()
    for name in [n for n in INSTALL_CELLS if n in needed]:
        if vm.finished(name) == 0:
            continue  # installed by an earlier run in this session
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / f"{name}.sh"
            path.write_text(scripts[name])
            vm.put(path, f"{REMOTE_WORK}/{name}.sh")
        command = f"bash {REMOTE_WORK}/{name}.sh"
        if name == "install-opensplat":
            command = opensplat_job(vm, command)
        vm.background(command, name)
    log("  installs running on the VM: " + ", ".join(n for n in needed if vm.finished(n) != 0))


def opensplat_job(vm: Colab, build: str) -> str:
    """OpenSplat links OpenCV from the COLMAP cell, so its job waits for that
    cell, as the notebook's cell order does. A binary built in an earlier
    session for this PyTorch and GPU (tools/colab-cache/) is tried first, and
    the ~10 minute build runs only if it does not start."""
    cached = f"{REMOTE_WORK}/opensplat-cached"
    local = CACHE / opensplat_key(vm)
    if local.exists():
        vm.put(local, cached)
        log(f"  OpenSplat binary from the cache: {local.name}")
    wait = (f"while [ ! -f {REMOTE_WORK}/install-colmap.exit ]; do sleep 5; done; "
            f"[ \"$(cat {REMOTE_WORK}/install-colmap.exit)\" = 0 ] || exit 1; ")
    try_cached = (f"if [ -f {cached} ]; then mkdir -p $(dirname {OPENSPLAT_BIN}) && "
                  f"cp {cached} {OPENSPLAT_BIN} && chmod +x {OPENSPLAT_BIN} && "
                  f"{remote_env()} && {OPENSPLAT_BIN} --help > /dev/null && "
                  f"{{ echo 'OpenSplat from the cache'; exit 0; }}; "
                  f"echo 'the cached OpenSplat does not run here; building'; rm -f {OPENSPLAT_BIN}; fi; ")
    return wait + try_cached + build


def keep_opensplat(vm: Colab) -> None:
    key = opensplat_key(vm)
    if (CACHE / key).exists():
        return
    CACHE.mkdir(parents=True, exist_ok=True)
    vm.get(OPENSPLAT_BIN, CACHE / key)
    (CACHE / key).chmod(0o755)
    log(f"  kept the built OpenSplat as tools/colab-cache/{key}")


def wait_installs(vm: Colab, needed: list[str]) -> None:
    pending = [n for n in INSTALL_CELLS if n in needed]
    while pending:
        for name in list(pending):
            code = vm.finished(name)
            if code is None:
                continue
            tail, _ = vm.log_since(name, 0)
            if code != 0:
                raise SystemExit(f"install {name} failed (exit {code}):\n{tail[-3000:]}")
            log(f"  {name} done: {tail.strip().splitlines()[-1] if tail.strip() else ''}")
            pending.remove(name)
        if pending:
            time.sleep(30)


def upload_space(vm: Colab, name: str) -> None:
    """A space as it stands here, for a new session carrying on from a later stage."""
    space = ROOT / "spaces" / name
    if not space.exists():
        return
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / f"{name}.tar"
        saves = sorted((int(p.stem.rsplit("_", 1)[1]), p.name) for p in space.glob("splat-long_*.ply")
                       if p.stem.rsplit("_", 1)[1].isdigit())
        superseded = {f"{name}/{n}" for _, n in saves[:-1]}
        with tarfile.open(archive, "w") as tar:
            tar.add(space, arcname=name,
                    filter=lambda info: None if (any(skip in info.name for skip in NOT_FETCHED)
                                                 or info.name in superseded) else info)
        vm.put(archive, f"{REMOTE_WORK}/space-{name}.tar")
    vm.shell(f"mkdir -p {REMOTE_ROOT}/spaces && tar -xf {REMOTE_WORK}/space-{name}.tar -C {REMOTE_ROOT}/spaces",
             timeout=600)
    log(f"  uploaded spaces/{name}")


def fetch_space(vm: Colab, name: str) -> int:
    """Bring the space's new and changed files here (all of them, the dense
    cloud included), so nothing is lost if the session dies."""
    manifest_remote = f"{REMOTE_WORK}/fetched-{name}.json"
    code = f"""
import json, os, tarfile
root = '{REMOTE_ROOT}/spaces/{name}'
manifest_path = '{manifest_remote}'
sent = json.load(open(manifest_path)) if os.path.exists(manifest_path) else {{}}
import re
saves = sorted((int(m.group(1)), f) for f in os.listdir(root)
               for m in [re.fullmatch(r'splat-long_(\\d+)\\.ply', f)] if m)
superseded = {{f for _, f in saves[:-1]}}   # only the newest long-training save crosses
now, changed = {{}}, []
for folder, dirs, files in os.walk(root):
    for f in files:
        full = os.path.join(folder, f)
        rel = os.path.relpath(full, root)
        if any(skip in rel for skip in {NOT_FETCHED!r}) or os.path.islink(full):
            continue
        if rel in superseded:
            continue
        st = os.stat(full)
        now[rel] = [st.st_size, int(st.st_mtime)]
        if sent.get(rel) != now[rel]:
            changed.append(rel)
with tarfile.open('{REMOTE_WORK}/fetch-{name}.tar', 'w') as tar:
    for rel in changed:
        tar.add(os.path.join(root, rel), arcname=rel)
json.dump(now, open(manifest_path + '.next', 'w'))
print(json.dumps(changed))
"""
    # Only the count comes back through exec; the names travel inside the tar.
    code = code.replace("print(json.dumps(changed))", "print('CHANGED', len(changed))")
    counted = [l for l in vm.python(code, timeout=900).splitlines() if l.startswith("CHANGED ")]
    if not counted:
        raise RuntimeError(f"could not list the new files of spaces/{name} on the VM")
    count = int(counted[-1].split()[1])
    if not count:
        return 0
    local_space = ROOT / "spaces" / name
    local_space.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "fetch.tar"
        vm.get(f"{REMOTE_WORK}/fetch-{name}.tar", archive)
        with tarfile.open(archive) as tar:
            tar.extractall(local_space, filter="data")
    vm.shell(f"mv {manifest_remote}.next {manifest_remote}", timeout=120)
    return count


def remote_env() -> str:
    exports = [f"export {k}={shlex.quote(v)}" for k, v in REMOTE_ENV.items()]
    # OpenSplat links against Colab's PyTorch libraries.
    exports.append('export LD_LIBRARY_PATH="$(python -c \'import os, torch; '
                   'print(os.path.join(os.path.dirname(torch.__file__), "lib"))\'):$LD_LIBRARY_PATH"')
    return " && ".join(exports)


def gate_of(vm: Colab, name: str, stage: str, since: int) -> dict:
    """The stage's gate from the VM's agent-report.json (only the gate crosses:
    the whole report can be larger than exec output carries). A report older
    than `since` is an earlier attempt's, not this run's."""
    code = (f"import json, os\np = '{REMOTE_ROOT}/spaces/{name}/agent-report.json'\n"
            f"fresh = os.path.exists(p) and os.path.getmtime(p) >= {since}\n"
            f"gate = json.load(open(p)).get('gates', {{}}).get('{stage}') if fresh else None\n"
            f"print('GATE', json.dumps(gate))\n")
    lines = [l for l in vm.python(code, timeout=120).splitlines() if l.startswith("GATE ")]
    return (json.loads(lines[-1][5:]) if lines else None) or {"status": "stop", "why": "no gate recorded"}


def run_stage(vm: Colab, video_remote: str, name: str, step: str, extra: list[str]) -> dict:
    """pipeline/agent.py --stage on the VM, its output shown here as it runs.
    During stage 4 the space is fetched every FETCH_DURING_MINUTES, so a
    trained splat is already here if the session dies during the next one."""
    job = f"{name}-{step}"
    stage = "splat" if step in SPLAT_STEPS else step
    command = (f"{remote_env()} && mkdir -p {RELAY} && cd {REMOTE_ROOT} && "
               f"python pipeline/agent.py {shlex.quote(video_remote)} --name {shlex.quote(name)} "
               f"--stage {stage} " + (f"--splat-steps {step} " if stage == "splat" else "")
               + " ".join(shlex.quote(a) for a in extra))
    since = int(vm.shell("date +%s", timeout=120).split()[0])
    vm.background(command, job)
    offset, last_fetch = 0, time.time()
    while True:
        if step.startswith("train") and time.time() - last_fetch > FETCH_DURING_MINUTES * 60:
            try:
                log(f"  fetched {fetch_space(vm, name)} file(s) so far")
            except RuntimeError as exc:
                log(f"  fetch during the stage failed ({exc}); trying again later")
            last_fetch = time.time()
        text, offset = vm.log_since(job, offset)
        for line in text.splitlines():
            if line.strip() and "Loading weights" not in line and "it/s]" not in line:
                print("    vm |", line[:300], flush=True)
        code = vm.finished(job)
        if code is not None:
            text, offset = vm.log_since(job, offset)
            for line in text.splitlines():
                if line.strip():
                    print("    vm |", line[:300], flush=True)
            break
        time.sleep(20)
    gate = gate_of(vm, name, stage, since)
    if code != 0 and gate.get("why") == "no gate recorded":
        gate["why"] = f"the agent exited with {code}; see its log above"
    return gate


def main() -> None:
    # A stopped driver still runs its finally blocks (the relay is stopped with it).
    signal.signal(signal.SIGTERM, lambda *_: sys.exit("stopped"))
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video")
    parser.add_argument("--name", required=True, help="space name (spaces/<name> here and on the VM)")
    parser.add_argument("--session", default="oasis", help="Colab CLI session name")
    parser.add_argument("--gpu", default="T4")
    parser.add_argument("--stages", nargs="+", choices=STEPS, default=STEPS,
                        help="stages 1-3 and stage 4's steps to run, in order")
    parser.add_argument("--agent-args", default="", help="extra pipeline/agent.py arguments")
    parser.add_argument("--keep-session", action="store_true", help="do not stop the session at the end")
    args = parser.parse_args()

    video = Path(args.video).resolve()
    if not video.is_file():
        sys.exit(f"video not found: {video}")
    vm = Colab(args.session)
    fresh = not vm.alive()
    if fresh:
        vm.create(args.gpu)
    log("code: " + upload_code(vm))
    needed = sorted({cell for step in args.stages for cell in NEEDS[step]}, key=INSTALL_CELLS.index)
    installed = all(vm.finished(c) == 0 for c in needed)
    if not installed:
        start_installs(vm, needed)   # skips anything installed or still running
    video_remote = f"{REMOTE_VIDEOS}/{video.name}"
    log(f"video: {video.name}")
    vm.put(video, video_remote)
    if args.stages[0] != STEPS[0]:
        remote_has = vm.shell(f"test -f {REMOTE_ROOT}/spaces/{args.name}/agent-report.json && echo yes || true",
                              timeout=120).strip() == "yes"
        if not remote_has:
            upload_space(vm, args.name)
        else:
            # Carrying on in the same session: bring what the VM has here first.
            log(f"  fetched {fetch_space(vm, args.name)} file(s) already on the VM")
    if not installed:
        log("waiting for the installs: " + ", ".join(needed))
        wait_installs(vm, needed)
    if "install-opensplat" in needed:
        try:
            keep_opensplat(vm)
        except RuntimeError as exc:
            log(f"  could not keep the OpenSplat binary ({exc})")

    relay_log = ROOT / "spaces" / f"{args.name}-relay.log"
    relay_log.parent.mkdir(parents=True, exist_ok=True)
    relay = subprocess.Popen([sys.executable, str(ROOT / "tools" / "claude_relay.py"),
                              "--session", args.session, "--remote", RELAY],
                             stdout=open(relay_log, "a"), stderr=subprocess.STDOUT)
    log(f"Claude relay running here (log: {relay_log.relative_to(ROOT)})")
    extra = shlex.split(args.agent_args)
    done_steps: list[str] = []
    try:
        for step in args.stages:
            log(f"{step} on the VM")
            gate = run_stage(vm, video_remote, args.name, step, extra)
            log(f"{step}: {gate.get('status', '?').upper()} - {gate.get('why', '')}")
            log(f"  fetched {fetch_space(vm, args.name)} new or changed file(s) into spaces/{args.name}")
            if gate.get("status") == "stop":
                rest = " ".join(args.stages[args.stages.index(step):])
                sys.exit(f"{step} stopped; fix it, then run again with --stages {rest}")
            done_steps.append(step)
    except SessionEnded as exc:
        rest = " ".join(s for s in args.stages if s not in done_steps)
        sys.exit(f"{exc}. Everything fetched so far is in spaces/{args.name}; carry on with:\n"
                 f"  python3 tools/colab_pipeline.py {args.video} --name {args.name} --stages {rest}")
    finally:
        relay.terminate()

    if any(step in args.stages for step in ("choose-training", "fill")):
        log("choose-best: comparing every splat of this video here (tools/splat_choose.py)")
        subprocess.run([sys.executable, str(ROOT / "tools" / "splat_choose.py"), str(video)], cwd=ROOT)
    if not args.keep_session:
        vm.cli("stop", "-s", args.session, timeout=300)
        log(f"stopped session {args.session}")


if __name__ == "__main__":
    main()
