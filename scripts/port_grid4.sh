#!/bin/bash
# Port grid 4 (results/port_grid4/SPEC.md): six methods from the transformer d32 reference (broader base), 2 seeds,
# 6 single-thread runs in parallel on the local machine; grid 3's settings otherwise.
cd "$(dirname "$0")/.."
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" JAX_PLATFORMS=cpu
EVAL="1 2 4 8 16 32 64 128 256 512 1024 2048 4096"
export COMMON="--config configs/potts_tfbind8.json --start models/pretrain_b0.75/transformer_d32_l2_h4_f128/best.npz --exponent 5 --steps 10000 --save-every 500 --no-plateau-stop --learning-rate 1e-4 --eval-steps $EVAL"
job_list() {
  for seed in 1 2; do
    echo "justgrpo         $seed --arm justgrpo"
    echo "trafl_comp4_b0.5 $seed --arm trafl --mask-scheme comp --mask-samples 4 --beta 0.5"
    echo "tb_b0.5          $seed --arm tb --beta 0.5"
    echo "espo             $seed --arm espo --mask-scheme comp --mask-samples 4 --kappa 0.05"
    echo "rspo             $seed --arm rspo --mask-samples 4 --rspo-lambda 0.01"
    echo "grpo             $seed --arm grpo"
  done
}
mkdir -p models/port_grid4/logs
job_list | xargs -P 6 -L 1 bash -c 'name=$0; seed=$1; shift 1; uv run --no-sync python -u -m src.rl.sweep $COMMON --seed $seed --output models/port_grid4/$name "$@" > models/port_grid4/logs/${name}_s${seed}.log 2>&1 && echo "$(date +%H:%M) done $name s$seed" || echo "$(date +%H:%M) FAILED $name s$seed"'
