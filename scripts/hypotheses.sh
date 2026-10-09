#!/bin/bash
# H1 (paths), H2 (surrogate), H3 (fixed point) for every run in models/trafl, in parallel.
# Results: results/hypotheses/h{1,2,3}/<seed>_<run>/. Usage: scripts/hypotheses.sh [jobs]
cd "$(dirname "$0")/.."
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" JAX_PLATFORMS=cpu
mkdir -p results/hypotheses/logs
for run in models/trafl/seed_*/*_[0-9]; do
  for h in h1_paths h2_surrogate h3_fixed_point; do echo "$h $run"; done
done | xargs -P "${1:-8}" -n 2 sh -c 'log="results/hypotheses/logs/$0_$(echo $1 | tr / _).log"; \
  uv run python -W ignore -m src.hypotheses.$0 "$1" > "$log" 2>&1 \
  && echo "$(date +%H:%M:%S) done $0 $1" || echo "$(date +%H:%M:%S) FAILED $0 $1"'
echo "$(date +%H:%M:%S) all hypotheses done"
