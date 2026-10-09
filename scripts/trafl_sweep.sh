#!/bin/bash
# TraFL on configs/potts_tfbind8.json: for each seed and width 2^k, one job trains from the
# pretrained Potts model (finetune_k) and from random weights (random_init_k), until the loss
# plateaus. Jobs run in parallel, single-threaded. Usage: scripts/trafl_sweep.sh [jobs]
cd "$(dirname "$0")/.."
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
for seed in 1 2 3; do for k in 7 5 3; do echo "$seed $k"; done; done |
  xargs -P "${1:-6}" -n 2 sh -c 'uv run python -m src.rl.sweep --config configs/potts_tfbind8.json \
    --seed $0 --exponent $1 --steps 60000 --save-every 250 --schedule constant --learning-rate 1e-3 \
    --stop-on-plateau --output models/trafl > /dev/null 2>&1 \
    && echo "$(date +%H:%M:%S) finished seed=$0 k=$1" || echo "$(date +%H:%M:%S) FAILED seed=$0 k=$1"'
echo "$(date +%H:%M:%S) sweep done"
