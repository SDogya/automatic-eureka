#!/usr/bin/env bash
# Install in the output directory, launch detached, and archive the result.
set -euo pipefail

if [[ ${1:-} == --internal ]]; then
  shift
  ROOT=$1
  RUN=$2
  RESUME=$3
  shift 3
  PACKAGE="$RUN/source/pretrain_a100"
  archive_failure() {
    rc=$?
    if (( rc != 0 )) && [[ ! -f "$RUN/archive.json" ]]; then
      printf '{"state":"bootstrap_failed","complete":false,"exit_code":%d}\n' "$rc" > "$RUN/status.json"
      target="$ROOT/archives/$(basename "$RUN")-incomplete.tar.gz"
      tar -czf "$target.tmp" -C "$(dirname "$RUN")" "$(basename "$RUN")"
      mv "$target.tmp" "$target"
      (cd "$ROOT/archives" && sha256sum "$(basename "$target")" > "$(basename "$target").sha256")
    fi
  }
  trap archive_failure EXIT
  if command -v uv >/dev/null 2>&1; then
    UV=$(command -v uv)
  else
    mkdir -p "$ROOT/tools"
    curl --fail --location --silent --show-error https://astral.sh/uv/0.12.20/install.sh -o "$ROOT/tools/install-uv.sh"
    UV_INSTALL_DIR="$ROOT/tools" UV_NO_MODIFY_PATH=1 sh "$ROOT/tools/install-uv.sh"
    UV="$ROOT/tools/uv"
  fi
  "$UV" --version > "$RUN/logs/uv-version.txt"
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
  "$UV" pip install --python "$ROOT/.venv/bin/python" --no-deps --no-build-isolation \
    --editable "$ROOT/vendor/dllm"
  "$UV" pip freeze --python "$ROOT/.venv/bin/python" > "$RUN/logs/packages.txt"
  if [[ $RESUME == 1 ]]; then
    "$ROOT/.venv/bin/python" "$PACKAGE/run.py" --mode prepare --root "$ROOT" --run "$RUN" --resume
  else
    "$ROOT/.venv/bin/python" "$PACKAGE/run.py" --mode prepare --root "$ROOT" --run "$RUN" "$@"
  fi
  "$ROOT/.venv/bin/python" -u "$PACKAGE/run.py" --mode run --root "$ROOT" --run "$RUN"
  exit 0
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
OUTPUT="$HOME/bd3lm_benchmark"
RESUME_DIR=""
FOREGROUND=0
PREPARE_ARGS=()
while (( $# )); do
  case "$1" in
    --attempts|--gpus|--seed-start|--config)
      PREPARE_ARGS+=("$1" "$2"); shift 2 ;;
    --smoke) PREPARE_ARGS+=("$1"); shift ;;
    --output) OUTPUT=$2; shift 2 ;;
    --resume) RESUME_DIR=$2; shift 2 ;;
    --foreground) FOREGROUND=1; shift ;;
    --help|-h)
      cat <<'HELP'
Usage: bash pretrain_a100/run.sh [--attempts N] [--gpus 0,1,2,3]
       [--seed-start 1] [--output DIR] [--config YAML] [--smoke] [--foreground]
       bash pretrain_a100/run.sh --resume RUN_DIR [--foreground]

Default: four A100 workers, native single-answer sampling, 1319 tasks, 16 attempts.
Launch is detached by default and prints the PID, log and progress paths.
--smoke runs two tasks with two attempts, unless --attempts is also supplied.
Resume uses its saved source/config and rejects generation overrides.
HELP
      exit 0 ;;
    *) printf 'Unknown option: %s\n' "$1" >&2; exit 2 ;;
  esac
done

if [[ -n "$RESUME_DIR" ]]; then
  (( ${#PREPARE_ARGS[@]} == 0 )) || { echo 'Resume uses saved settings; omit generation overrides.' >&2; exit 2; }
  RUN=$(realpath "$RESUME_DIR")
  ROOT=$(dirname "$(dirname "$RUN")")
  [[ -f "$RUN/manifest.json" ]] || { echo 'Resume requires a prepared run with manifest.json.' >&2; exit 2; }
  RESUME=1
else
  ROOT=$(realpath -m "$OUTPUT")
  mkdir -p "$ROOT/runs" "$ROOT/archives"
  RUN="$ROOT/runs/$(date -u +%Y%m%dT%H%M%S)-$$"
  mkdir -p "$RUN/source/pretrain_a100" "$RUN/logs"
  cp -a "$SCRIPT_DIR/." "$RUN/source/pretrain_a100/"
  if [[ -d "$SCRIPT_DIR/../.git" ]]; then
    git -C "$SCRIPT_DIR/.." rev-parse HEAD > "$RUN/logs/git-commit.txt"
    git -C "$SCRIPT_DIR/.." status --short -- pretrain_a100 > "$RUN/logs/git-status.txt"
  fi
  RESUME=0
fi
mkdir -p "$RUN/logs" "$ROOT/archives"
# The environment and native checkout are shared by runs in this output root.
exec 8>"$ROOT/environment.lock"
flock -n 8 || { echo "Output directory already active: $ROOT" >&2; exit 1; }
# Hold through installation, preparation and execution, including after nohup.
exec 9>"$RUN/launch.lock"
flock -n 9 || { echo "Run already active: $RUN" >&2; exit 1; }
printf '{"state":"starting","complete":false}\n' > "$RUN/status.json"
if (( FOREGROUND )); then
  bash "$RUN/source/pretrain_a100/run.sh" --internal "$ROOT" "$RUN" "$RESUME" "${PREPARE_ARGS[@]}"
else
  nohup bash "$RUN/source/pretrain_a100/run.sh" --internal "$ROOT" "$RUN" "$RESUME" \
    "${PREPARE_ARGS[@]}" </dev/null >"$RUN/logs/launcher.log" 2>&1 &
  LAUNCH_PID=$!
  printf '%s\n' "$LAUNCH_PID" > "$RUN/launcher.pid"
  printf 'PID: %s\nRun: %s\nLog: %s\nProgress: %s\n' \
    "$LAUNCH_PID" "$RUN" "$RUN/logs/launcher.log" "$RUN/progress.json"
fi
