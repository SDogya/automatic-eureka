# Grid 3, long decoder-rollout runs: does decoder-sampled TraFL settle elsewhere, or only move slower?

Written 2026-10-11 05:33, before any run.

## Setup

- **Runs:** TraFL comp 4, β 0.5. Decoder rollouts against uniform rollouts, 2 seeds each, 30,000 updates.
- **Everything else is grid 3:** broader base, mlp_5 reference, learning rate 1e-4, same eval steps; checkpoints every
  1,000 updates.
- **Jobs:** `scripts/port_grid3_long_jobs.txt`, one Kaggle CPU kernel (4 runs in parallel, ≈ 1.5–2 h).

## Why

At 10k updates the decoder-rollout runs had moved half as far as the uniform ones:
- terminal KL from the reference 0.55 vs 1.12;
- KL(target ‖ p) 0.68 vs 0.32.

The decoder-rollout runs were still creeping: +0.04 over the last 6k updates.

## What theory says

TraFL's per-string fixed point does not depend on the sampler when every string can be sampled. Here every string has
positive decoder probability (`results/report/decoder_true_law.json`), so in the limit both arms should reach the same
target. The decoder only puts mass on a few strings, though, so convergence elsewhere is slow.

## Predictions, stated now

- **P1, speed and not a different fixed point.** Decoder rollouts keep moving toward the target, but slowly. At
  30k updates:
  - terminal KL from the reference: 0.6–0.8 (uniform ≈ 1.1);
  - KL(target ‖ p) still above the uniform run's at 10k.

  If they reach within 10 % of the uniform run's terminal KL, P1's "slow" part is wrong. If they stop entirely (change
  < 0.01 over the last 10k), they settle somewhere else, and that needs an explanation beyond coverage.
- **P2, drift.** The late path / terminal ratio stays higher with decoder rollouts than with uniform ones at the same
  horizon.
- **P3, deployed law.** Under the decoder at the end, the score is within ±0.02 of the uniform run's. Top-1 % diversity
  under the decoder is not higher than the uniform run's.

## Readout

`python -m src.analysis.summary` and `report` on `models/port_grid3_long`, plus the four trajectories at 1k / 4k /
10k / 20k / 30k.
