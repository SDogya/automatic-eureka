# Port grid 3: reading against SPEC.md

Written 2026-10-11 03:50, after all 54 runs (2 seeds each, no failures; kernel grid3-1 finished; its backups were not needed).
Tables: `summary.md` (matched distance 1), `ratio_windows.md`, `matched_*.md`. Every checkpoint: `runs.csv`.
Seed ranges are tight (≤ ±3 %).

**R1, per-string drift where the terminal law settles: supported.** Late-window path / terminal ratio:

| arm | late ratio |
|---|---|
| unpenalised TraFL estimators (32 IID, comp 4, pairwise, exact likelihood) | 0.17–0.19 |
| ESPO, decoder rollouts | 0.82 |
| TB | 0.09–0.10 |
| DB | 0.10 |
| SubTB | 0.09 |
| Ent-PPO | 0.07 |

TB and DB rise 20–45 % from their first in-band window and then flatten. The stated "within ±20 %" is mildly
violated, but the separation holds.

**R2, ordering among TraFL estimators: refuted in its ordering, met in its refutation clause.**
- The unpenalised estimators are indistinguishable (late ratio 0.17–0.19; path KL at matched terminal KL 1, β 0.5:
  0.15–0.19).
- Comp 4 + λ = 1 is the lowest (0.11), so the explicit order-variance penalty is what anchors.
- The implicit mask-variance penalty does not: its size, from K = 4 to 32, paired or not, unbiased or not, makes no
  difference here. This is consistent with the penalty measurement in the report (mask variance ≠ order variance).

**R3, diversity at matched KL(p ‖ p_ref) = 1** (distinct top-1 % strings in 16 draws):

| arm | random order | decoder |
|---|---|---|
| TraFL variants | 7.67–7.87 | 3.1–3.75 |
| TB, DB, SubTB | 7.62–7.70 | 3.1–3.2 |
| Ent-PPO | 7.56 | 3.3 |
| ESPO | 7.48 | **4.05** (highest) |
| ESPO-PPO | 7.39 | 3.65 |
| GRPO | 7.18 | 3.2 |
| RSPO | 6.94 | 3.1 |
| JustGRPO | 6.93 | 3.0 |

- Under random order, the prediction holds for TraFL and the balance arms against the reward maximisers; RSPO groups
  with the maximisers.
- Under the decoder, ESPO leads.
- Score at matched KL: 0.61–0.65 for every uniform-rollout arm.

**R4, the training sampler, decoder vs uniform rollouts** (per-string arms; no direction was stated):

| arm | late ratio, decoder vs uniform | other |
|---|---|---|
| TraFL comp 4 | 0.278 vs 0.190 (+46 %) | — |
| TraFL exact likelihood | 0.246 vs 0.175 (+40 %) | — |
| ESPO | — | path KL at matched distance 0.62 vs 0.10 (6×); score 0.571 vs 0.629; diversity 4.6 / 2.9 vs 7.5 / 4.1 |
| RSPO | — | score 0.620 vs 0.605; diversity 6.2 / 1.6 vs 6.9 / 3.1 |

Sampling training completions with the low-confidence decoder, as TraFL's paper does, makes the per-string objectives
off-policy for the model's own random-order law. Here it increases path drift and reduces diversity: training on the
decoder's own outputs reinforces its mode-seeking.

**Scope.** Broader base (Potts β 0.75, MLP k = 5 reference), one reward, 2 seeds, β ∈ {0.25, 0.5}. Comparisons are at
matched distance. Same-β comparisons across centring (TraFL, TB) and raw-reward (DB, SubTB, Ent-PPO) arms are not
same-target comparisons (PORT_NOTES, "Units and targets").
