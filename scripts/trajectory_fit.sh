#!/bin/bash
# Exact chain-rule checks for every run in models/trafl at the given checkpoint, in parallel.
# Usage: scripts/trajectory_fit.sh step_20000.npz [jobs]
cd "$(dirname "$0")/.."
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" JAX_PLATFORMS=cpu
CHECKPOINT="$1" ls -d models/trafl/seed_*/*_[0-9] | xargs -P "${2:-8}" -n 1 sh -c \
  'uv run python -m src.metrics.trajectory_fit "$0" --config configs/potts_tfbind8.json --checkpoint '"$1"' \
    > "$0/trajectory_fit.log" 2>&1 && echo "$(date +%H:%M:%S) done $0" || echo "$(date +%H:%M:%S) FAILED $0"'
