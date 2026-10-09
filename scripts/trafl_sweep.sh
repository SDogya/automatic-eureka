#!/bin/bash
# Retrain the selected pair (k = 5): pretrained start seed 1 and random start seed 2, with the
# fixed settings (beta = 1.5, K = 32 masks, constant lr 1e-3, stop on a loss plateau).
cd "$(dirname "$0")/.."
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" JAX_PLATFORMS=cpu
out=models/trafl_b15_k32
uv run python -m src.rl.sweep --config configs/potts_tfbind8.json --exponent 5 --seed 1 --only finetune --output $out &
uv run python -m src.rl.sweep --config configs/potts_tfbind8.json --exponent 5 --seed 2 --only random_init --output $out &
wait
