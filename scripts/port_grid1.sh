#!/bin/bash
# Port grid 1 (results/port_grid1/SPEC.md): every arm from the k = 5 MLP reference, 2 seeds, 6 runs in parallel.
cd "$(dirname "$0")/.."
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" JAX_PLATFORMS=cpu
EVAL="1 2 4 8 16 32 64 128 256 512 1024 2048"
export COMMON="--config configs/potts_tfbind8.json --exponent 5 --only finetune --steps 5000 --save-every 250 --no-plateau-stop --learning-rate 1e-4 --eval-steps $EVAL"
job_list() {
  for seed in 1 2; do
    echo "trafl_iid32     $seed --arm trafl --mask-samples 32"
    echo "trafl_comp4     $seed --arm trafl --mask-scheme comp --mask-samples 4"
    echo "trafl_comp4_var $seed --arm trafl --mask-scheme comp --mask-samples 4 --var-lambda 1"
    echo "trafl_exactlik  $seed --arm trafl --estimator exact_lik"
    echo "tb              $seed --arm tb"
    echo "entppo          $seed --arm entppo --ppo-epochs 4"
    echo "espo            $seed --arm espo --mask-scheme comp --mask-samples 4 --kappa 0.05"
    echo "espo_ppo        $seed --arm espo_ppo --mask-scheme comp --mask-samples 4 --kappa 0.05 --ppo-epochs 8"
    echo "grpo            $seed --arm grpo"
    echo "justgrpo        $seed --arm justgrpo"
    echo "rspo            $seed --arm rspo --mask-samples 4 --rspo-lambda 0.01"
  done
}
mkdir -p models/port_grid1/logs
job_list | xargs -P 6 -L 1 bash -c 'name=$0; seed=$1; shift 2; uv run --no-sync python -m src.rl.sweep $COMMON --seed $seed --output models/port_grid1/$name "$@" > models/port_grid1/logs/${name}_s${seed}.log 2>&1 && echo "done $name s$seed" || echo "FAILED $name s$seed"'
