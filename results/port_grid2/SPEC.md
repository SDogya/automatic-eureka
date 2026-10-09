# Port grid 2: β frontier, the penalty where drift appears, references; flow-balance arms

Written 2026-10-09 23:15, before any run; grid 1's reading (`../port_grid1/READING.md`) motivates each block. Runs on 2 Kaggle CPU
kernels (`scripts/kaggle_job.py`; jobs in `scripts/port_grid2_jobs_{a,b}.txt`). Environment (b) (Potts → Potts +
TFBind8), prompts `uniform`, 256 × 5 per update, Adam lr 1e-4, **10,000 updates**. Exact evaluation at 1, 2, 4, …,
4096 and every 500. 2 seeds.

## Blocks

| Block | Question | Runs |
|---|---|---|
| A, kernel a | Does the per-string excess appear wherever the terminal law becomes stationary? | MLP k = 5 reference; TraFL 32 IID, TraFL comp 4, TB (RTB), Ent-PPO (μ 4), DB, SubTB (λ 0.9); β ∈ {0.5, 1.5, 4.5}: 36 |
| B, kernel b | Does the variance penalty remove the drift where it appears (terminal KL ≥ 3 in grid 1)? | TraFL comp 4 + λ ∈ {1, 4} at β = 4.5 (TraFL comp 4 at 4.5 is in A): 4 |
| C, kernel b | Reference size / family | β = 1.5; references MLP k = 3 (688 parameters), k = 7 (25,888), transformer d32 (26,020); TraFL comp 4, TB, ESPO (κ 0.05), GRPO: 24 |

## Readings, stated before running

- **A.** Where an arm's terminal KL stops moving, its path / terminal ratio over the late windows grows for the
  per-string arm (TraFL) and stays flat for TB, Ent-PPO, DB and SubTB.
  - Prediction: TraFL's ratio grows in the last two windows at β 0.5 and 1.5, by more than its seed range; the
    balance arms stay within ±20 % of their first in-band window.
  - DB and SubTB share RTB's target: path KL at matched terminal KL within ±20 % of TB.
  - Refuted for TraFL if its late-window ratio is flat (within ±20 %) at a β where its terminal KL is stationary.
- **B.** At terminal KL 3, TraFL comp 4 + λ = 1 / 4 has lower path KL than TraFL comp 4, approaching TB's. Refuted
  if ≥ at both λ.
- **C.** Exploratory: no direction stated. gfn_lab S2 (a fixed backbone over a larger space shrank the excess)
  suggests the excess depends on capacity relative to the state space. Reported as found.
- **Scope.** Same as grid 1: one environment, an ≈ 100-string reference law, 2 seeds.
