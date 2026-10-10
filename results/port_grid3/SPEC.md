# Port grid 3: every family on the broader base, organised by what is scored, how, and from which sampler

Written 2026-10-10 04:54, before any run. **Base (b′):** pretraining Potts β = 0.75 (`configs/potts_b0.75.json`), reference
`models/pretrain_b0.75/mlp_5`. Foundation numbers are in `results/foundation/diagnostics_potts_b0.75.json`:
- the law's 99 % support is 11,205 strings;
- the reference is not a memorised sample (memorisation index 0.93, KL(law ‖ ref) 0.10 < KL(sample ‖ ref) 0.42);
- TraFL's target spans ≈ 1,230 strings at β 0.5.

**Unchanged from grid 2:** reward log R = 3.5 (z_Potts + z_TFBind8), prompts `uniform`, 256 × 5 per update, lr 1e-4,
10,000 updates, exact evaluation at 1, 2, …, 4096 and every 500, every checkpoint kept, 2 seeds.

## The design (the principal's 2 × 2, plus the sampler)

| What is scored | Exact | Estimated |
|---|---|---|
| **trajectory τ** (target P_ref e^{βr}, path pinned to c_ref) | TB = RTB; Ent-PPO (soft-RL); DB / SubTB (local credit) | — |
| **string y** (path not pinned) | TraFL with exact likelihood | TraFL square: 32 IID / 4 comp masks (variance penalty Var/K); pairwise U-statistic, 4 IID (unbiased, no penalty); comp 4 + λ = 1 explicit penalty; ESPO (k2 + ratio); RSPO (λ 0.01) |

- Third axis, the training sampler: uniform random order (default) vs the low-confidence-remasking decoder at T 1,
  as TraFL's paper samples. Only for the per-string losses, which never score the sampled order.
- Reward maximisers for reference: GRPO (τ, exact), JustGRPO (AR), ESPO-PPO (μ 8).
- β ∈ {0.25, 0.5} for the balance and TraFL arms. β ≥ 1.5 collapses TraFL's target to ≤ 20 strings on this base.

Runs:
- 20 per-string;
- 16 per-trajectory;
- 12 sampler axis (TraFL comp 4, TraFL exact likelihood, ESPO, RSPO at β 0.5, uniform vs decoder, where not already
  in the other blocks);
- 6 reward maximisers;

54 in all, on 3 Kaggle CPU kernels.

## Readings, stated before running

- **R1 (gfn_lab S2 / grids 1–2, on a non-trivial base).** Over the late windows where an arm's terminal KL is
  stationary, path / terminal grows for the per-string arms and stays flat or falls for TB, DB, SubTB and Ent-PPO.
  Refuted if a per-string arm stays within ±20 % at a β where its terminal KL is stationary.
- **R2 (the variance identity).** Among the TraFL estimators, at matched terminal KL ≥ 1 and in the late-window
  ratio, path drift orders as
  - pairwise ≈ exact likelihood (no penalty) ≥ 32 IID (Var/32) ≥ comp 4 (Var/4) > comp 4 + λ = 1.

  Grid 2 already showed comp 4 > 32 IID at KL 3 (β 1.5), so this ordering is at risk. It is refuted if comp 4 + λ
  is not the lowest.
- **R3 (TraFL's diversity claim, exact).** At matched KL(p_θ ‖ p_ref), top-1 % mass and top_distinct16 (distinct
  correct strings in 16 draws, the analogue of TraFL's LLM judge on correct solutions):
  - under the random-order law: balance arms and TraFL ≥ ESPO / RSPO ≥ GRPO / JustGRPO;
  - under the decoder: no prediction (grid 1 showed collapse on the narrow base).
- **R4 (sampler).** Exploratory, no direction. How decoder rollouts change a per-string arm's reward, random-order
  entropy, decoder entropy and top_distinct16 at matched KL.
- **Scope.** One reward, one reference family and size, 2 seeds. Decoder-level path distances are computed post hoc
  from the kept checkpoints (`src/analysis/endpoints.py`).
