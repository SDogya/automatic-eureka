#!/usr/bin/env bash
# Reuse the pretrained launcher bootstrap; launch ONE shared TraFL policy.
set -euo pipefail

PACKAGE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT="$HOME/trafl_bd3lm"
CONFIG="$PACKAGE/config.yaml"
GPUS=0,1,2,3
MODE=train
RESUME=""
FOREGROUND=0
while (( $# )); do
  case "$1" in
    --output) ROOT=$2; shift 2 ;;
    --config) CONFIG=$2; shift 2 ;;
    --gpus) GPUS=$2; shift 2 ;;
    --mode) MODE=$2; shift 2 ;;
    --resume) RESUME=$2; shift 2 ;;
    --foreground) FOREGROUND=1; shift ;;
    --help|-h)
      echo 'Usage: bash run.sh [--gpus 0,1,2,3] [--output DIR] [--config YAML]'
      echo '                   [--mode train|smoke] [--resume CHECKPOINT] [--foreground]'
      echo 'Default: install/download/train detached; inspect the printed launcher log.'
      exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
ROOT=$(realpath -m "$ROOT")
CONFIG=$(realpath "$CONFIG")
if [[ -n "$RESUME" ]]; then RESUME=$(realpath "$RESUME"); fi
LAUNCH=${TRAFL_LAUNCH_DIR:-"$ROOT/launches/$(date -u +%Y%m%dT%H%M%S)-$$"}
mkdir -p "$LAUNCH" "$ROOT/models" "$ROOT/datasets" "$ROOT/runs"
if [[ "$CONFIG" != "$LAUNCH/requested.yaml" ]]; then
  cp "$CONFIG" "$LAUNCH/requested.yaml"
fi
if (( ! FOREGROUND )); then
  ARGS=(--foreground --gpus "$GPUS" --output "$ROOT" --config "$LAUNCH/requested.yaml" --mode "$MODE")
  if [[ -n "$RESUME" ]]; then ARGS+=(--resume "$RESUME"); fi
  nohup env TRAFL_LAUNCH_DIR="$LAUNCH" bash "$PACKAGE/run.sh" "${ARGS[@]}" \
    </dev/null >"$LAUNCH/launcher.log" 2>&1 &
  printf '%s\n' "$!" > "$LAUNCH/launcher.pid"
  printf 'PID: %s\nLog: %s\nEnvironment: %s\nResults: %s\n' \
    "$!" "$LAUNCH/launcher.log" "$ROOT/.venv" "$ROOT/runs"
  exit 0
fi
# Avoid reinstalling shared packages while another training uses this root.
exec 9>"$ROOT/environment.lock"
flock -n 9 || { echo "Output root already active: $ROOT" >&2; exit 1; }
if command -v uv >/dev/null 2>&1; then
  UV=$(command -v uv)
else
  mkdir -p "$ROOT/tools"
  curl --fail --location --silent --show-error https://astral.sh/uv/0.12.20/install.sh -o "$ROOT/tools/install-uv.sh"
  UV_INSTALL_DIR="$ROOT/tools" UV_NO_MODIFY_PATH=1 sh "$ROOT/tools/install-uv.sh"
  UV="$ROOT/tools/uv"
fi
"$UV" python install 3.12
if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
  "$UV" venv --python 3.12 "$ROOT/.venv"
fi
"$UV" pip sync --python "$ROOT/.venv/bin/python" --require-hashes \
  --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu126 \
  --index-strategy unsafe-best-match "$PACKAGE/requirements.lock"
mkdir -p "$ROOT/vendor"
if [[ ! -d "$ROOT/vendor/dllm/.git" ]]; then
  git clone https://github.com/ZHZisZZ/dllm.git "$ROOT/vendor/dllm"
fi
git -C "$ROOT/vendor/dllm" checkout --detach ca176752fbceec49c6b4777a2c18ae88e4eb10ed
"$UV" pip install --python "$ROOT/.venv/bin/python" --no-deps --no-build-isolation --editable "$ROOT/vendor/dllm"
"$UV" pip freeze --python "$ROOT/.venv/bin/python" > "$LAUNCH/packages.txt"

export HF_HOME="$ROOT/models"
export HF_DATASETS_CACHE="$ROOT/datasets"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPUS"
cd "$PACKAGE"
# Download once BEFORE torchrun. Workers then use these shared caches and a
# resolved dataset revision. Current training reads HF GSM8K, not the Kaggle copy.
"$ROOT/.venv/bin/python" - "$LAUNCH/requested.yaml" "$LAUNCH/config.yaml" "$ROOT" "$RESUME" <<'PY'
from pathlib import Path
import sys
import torch
import yaml
from datasets import load_dataset
from huggingface_hub import HfApi, snapshot_download

requested, effective, root, resume = sys.argv[1:]
cfg = yaml.safe_load(Path(requested).read_text())
resume = resume or cfg.get('resume_checkpoint')
if resume:
    saved = torch.load(Path(resume)/'training_state.pt', map_location='cpu', weights_only=False)['config']
    assert saved.get('training_protocol') == cfg['training_protocol'], 'Resume sampler protocol differs'
    cfg = saved | {key: cfg[key] for key in ('gpu', 'local_files_only', 'output_dir')}
    cfg['resume_checkpoint'] = str(Path(resume).resolve())
model = snapshot_download(cfg['model_id'], revision=cfg['model_revision'])
cfg['dataset_revision'] = cfg['dataset_revision'] or HfApi().dataset_info(cfg['dataset_id']).sha
load_dataset(cfg['dataset_id'], cfg['dataset_config'], revision=cfg['dataset_revision'])
cfg['local_files_only'] = True
cfg['output_dir'] = str(Path(root)/'runs')
Path(effective).write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
print(f'Model: {model}\nDataset: {cfg["dataset_id"]}@{cfg["dataset_revision"]}', flush=True)
PY

IFS=, read -r -a GPU_IDS <<< "$GPUS"
ARGS=(--config "$LAUNCH/config.yaml" --mode "$MODE")
if [[ -n "$RESUME" ]]; then ARGS+=(--resume "$RESUME"); fi
exec "$ROOT/.venv/bin/python" -u -m torch.distributed.run --standalone \
  --nproc-per-node="${#GPU_IDS[@]}" run.py "${ARGS[@]}"
