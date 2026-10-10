"""Run sweep jobs on Kaggle CPU kernels (4 cores each; the account allows 5 concurrent CPU sessions).

    python scripts/kaggle_job.py bundle                          # git archive of HEAD -> private dataset <user>/ae-bundle
    python scripts/kaggle_job.py push <job> <jobs.txt> <grid>    # one kernel; runs every line of jobs.txt, 4 at a time
    python scripts/kaggle_job.py status <job>
    python scripts/kaggle_job.py pull <job>                      # outputs -> models/<grid>/..., never overwriting

jobs.txt: one run per line, "<name> <seed> <sweep args...>"; a line "COMMON <args...>" sets arguments shared by all.
Remotely: `uv sync --frozen` installs the exact lockfile environment (Python 3.14, jax, optax, ...); then the gate
tests must pass (non-zero exit -> kernel status ERROR, nothing runs); then the jobs; settings.json, evals.csv,
steps.csv, logs and every checkpoint are copied to the output (post-hoc metrics at any training step).
Pitfalls inherited from gfn_lab's runner: Kaggle auto-extracts archives (both layouts are handled); push by hand
when a slot frees (shell retry loops did not register success); credentials stay in ~/.kaggle.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / ".scratch" / "kaggle"                      # .scratch/ is git-ignored
USER = json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text())["username"]
DATASET = f"{USER}/ae-bundle"
EXCLUDE = ["models/trafl_b15_k32", "models/trafl_uniform_x", "results/hypotheses"]   # large, not needed remotely
GATE = ("tests/test_arms.py tests/test_paths.py tests/test_estimators.py tests/test_entppo.py "
        "tests/test_flow_balance.py tests/test_decoders.py tests/test_analysis.py tests/test_trafl.py")

JOB = r'''
import glob, os, shutil, subprocess, sys, tarfile, time
t0 = time.time()
root = "/kaggle/working/repo"
tars = [f for f in glob.glob("/kaggle/input/**/ae_bundle.tar.gz", recursive=True) if os.path.isfile(f)]
if tars:
    os.makedirs(root, exist_ok=True)
    with tarfile.open(tars[0]) as t:
        t.extractall(root)
else:                                                   # Kaggle auto-extracted the archive: copy the tree
    marker = glob.glob("/kaggle/input/**/pyproject.toml", recursive=True)
    assert marker, "bundle not found under /kaggle/input"
    shutil.copytree(os.path.dirname(marker[0]), root)
def run(cmd, check=True):
    print("==>", cmd, flush=True)
    r = subprocess.run(cmd, shell=True, cwd=root)
    print("<== rc", r.returncode, f"{time.time() - t0:.0f}s", flush=True)
    if check and r.returncode != 0:
        sys.exit(1)                                     # -> kernel status ERROR, not a misleading COMPLETE
    return r.returncode
run("pip install -q uv && uv sync --frozen -q")
run("uv run --no-sync python -m pytest -q -p no:cacheprovider " + GATE)
env = 'XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" JAX_PLATFORMS=cpu'
os.makedirs(os.path.join(root, "models", GRID, "logs"), exist_ok=True)
lines = [l.split() for l in JOBS.strip().splitlines() if l.strip()]
common = " ".join(next((l[1:] for l in lines if l[0] == "COMMON"), []))
with open(os.path.join(root, "jobs.cmd"), "w") as f:
    for name, seed, *args in (l for l in lines if l[0] != "COMMON"):
        # copy each run's folder and log to the output as soon as it ends, so a session timeout loses only unfinished runs
        f.write(f"{env} uv run --no-sync python -m src.rl.sweep {common} --seed {seed} --output models/{GRID}/{name} "
                f"{' '.join(args)} > models/{GRID}/logs/{name}_s{seed}.log 2>&1 && echo done {name} s{seed} || echo FAILED {name} s{seed}; "
                f"mkdir -p /kaggle/working/out/{GRID}/{name} /kaggle/working/out/{GRID}/logs; "
                f"cp -r models/{GRID}/{name}/. /kaggle/working/out/{GRID}/{name}/; "
                f"cp models/{GRID}/logs/{name}_s{seed}.log /kaggle/working/out/{GRID}/logs/\n")
run("cat jobs.cmd | xargs -P 4 -I CMD bash -c CMD", check=False)
out = "/kaggle/working/out"
for d, _, files in os.walk(os.path.join(root, "models", GRID)):
    keep = [f for f in files if f.endswith((".json", ".csv", ".log", ".png", ".npz"))]   # every checkpoint (small)
    for f in keep:
        target = os.path.join(out, os.path.relpath(d, os.path.join(root, "models")))
        os.makedirs(target, exist_ok=True)
        shutil.copy(os.path.join(d, f), target)
shutil.rmtree(root)
print("done", f"{time.time() - t0:.0f}s", flush=True)
'''


def bundle() -> None:
    d = WORK / "bundle"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    spec = ["."] + [f":(exclude){p}" for p in EXCLUDE]
    subprocess.run(["git", "archive", "--format=tar.gz", "-o", str(d / "ae_bundle.tar.gz"), "HEAD", "--", *spec],
                   cwd=ROOT, check=True)
    (d / "dataset-metadata.json").write_text(json.dumps({"title": "ae-bundle", "id": DATASET,
                                                         "licenses": [{"name": "CC0-1.0"}]}))
    exists = subprocess.run(["kaggle", "datasets", "status", DATASET], capture_output=True).returncode == 0
    cmd = ["kaggle", "datasets", "version", "-p", str(d), "-m", "update"] if exists else \
          ["kaggle", "datasets", "create", "-p", str(d)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout
    print(f"bundle of {commit.strip()} ({(d / 'ae_bundle.tar.gz').stat().st_size / 1e6:.1f} MB):",
          r.stdout[-400:], r.stderr[-400:])


def push(job: str, jobs_file: Path, grid: str) -> None:
    d = WORK / "jobs" / job
    d.mkdir(parents=True, exist_ok=True)
    (d / "job.py").write_text(f"GRID = {grid!r}\nGATE = {GATE!r}\nJOBS = {jobs_file.read_text()!r}\n" + JOB)
    (d / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{USER}/ae-{job}", "title": f"ae-{job}", "code_file": "job.py", "language": "python",
        "kernel_type": "script", "is_private": "true", "enable_gpu": "false", "enable_internet": "true",
        "dataset_sources": [DATASET], "competition_sources": [], "kernel_sources": []}, indent=1))
    r = subprocess.run(["kaggle", "kernels", "push", "-p", str(d)], capture_output=True, text=True)
    print(r.stdout[-500:], r.stderr[-500:])


def status(job: str) -> None:
    r = subprocess.run(["kaggle", "kernels", "status", f"{USER}/ae-{job}"], capture_output=True, text=True)
    print(r.stdout.strip(), r.stderr.strip()[-300:])


def pull(job: str) -> None:
    d = WORK / "out" / job
    d.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(["kaggle", "kernels", "output", f"{USER}/ae-{job}", "-p", str(d)], capture_output=True, text=True)
    print(r.stdout[-300:], r.stderr[-300:])
    new = skipped = 0
    for src in (d / "out").rglob("*") if (d / "out").exists() else []:
        if src.is_file():
            target = ROOT / "models" / src.relative_to(d / "out")
            if target.exists():
                skipped += 1
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, target)
            new += 1
    print(f"{job}: {new} new files into models/, {skipped} already present (kernel log in {d})")


if __name__ == "__main__":
    command = sys.argv[1]
    if command == "bundle":
        bundle()
    elif command == "push":
        push(sys.argv[2], Path(sys.argv[3]), sys.argv[4])
    elif command == "status":
        status(sys.argv[2])
    elif command == "pull":
        pull(sys.argv[2])
    else:
        raise SystemExit(__doc__)
