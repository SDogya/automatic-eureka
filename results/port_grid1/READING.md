# Port grid 1: reading against SPEC.md

Written 2026-10-09 22:15, after all 22 runs (11 configurations × seeds 1, 2; no failures). Tables:
`matched_path.md`, `matched_score.md`, `matched_entropy.md`, `ratio_windows.md`; every checkpoint is in
`runs.csv`. Matched values interpolate log-log at each run's *first* crossing of the level (`src/analysis/compare.py`).
Seed ranges are tight, typically ±1–3 %.

## P1, per-string vs per-trajectory: refuted by its stated rule

- **Up to terminal KL 1 every TraFL variant matches TB.** Path KL at KL(p_ref ‖ p_θ) = 0.3:

  | arm | path KL |
  |---|---|
  | Ent-PPO | 0.0204 |
  | TraFL exact likelihood | 0.0200 |
  | TraFL 32 IID | 0.0208 |
  | TraFL comp 4 + λ var | 0.0210 |
  | TraFL comp 4 | 0.0213 |
  | TB | 0.0217 |

  Exact-likelihood TraFL sits *below* TB beyond the seed range at 0.3 and 1, which is the stated refutation
  condition.
- **The per-string excess appears only late**, at terminal KL 3:

  | arm | path KL at KL 3 |
  |---|---|
  | Ent-PPO | 0.22 |
  | TB | 0.27 |
  | TraFL exact likelihood | 0.46 |
  | TraFL 32 IID | 0.64 |
  | TraFL comp 4 | 1.20 |

  The predicted order among the TraFL estimators is reversed (exact < 32 IID < comp 4). The variance-penalty run
  never reached KL 3 within 5,000 updates, so the fix is untested where the drift appears.
- **ESPO settles near a stationary terminal law (KL ≈ 1)**, and its path / terminal ratio then grows over
  training. By update window 30–166 → 166–910 → 910–5001:
  - single update: 0.08 → 0.28 → 1.6;
  - PPO: 0.26 → 1.7 → 1.9.

  This is gfn_lab S2's per-string pattern, cleanly.
- **JustGRPO has about twice the path KL** of the other arms at every level ≤ 1. Its training sampler is
  left-to-right, so the random-order path law is off-policy for it.
- **Reading:** here, path drift is a late, accumulating effect. It shows where the terminal law stops moving (ESPO)
  or at large distance (TraFL at KL 3). At small distances no objective moves the path law more than another, except
  JustGRPO.

## P2, TraFL's claim re-scoped: partly met

- **Reward efficiency is identical across arms.** At matched KL(p_θ ‖ p_ref) every arm has the same expected
  TFBind8 score within 1–2 %: 0.43 / 0.49 / 0.62 at 0.1 / 0.3 / 1. The reward maximisers (GRPO, JustGRPO, RSPO)
  reach 0.91–0.95 against 0.73–0.80 by going much further (KL 7–9 against 2–3), as predicted.
- **Diversity at matched KL** (random-order entropy, `matched_entropy.md`), at KL 1:

  | arm | entropy |
  |---|---|
  | ESPO | 1.62–1.69 |
  | TB, TraFL 32 IID, TraFL exact likelihood | 1.49–1.51 |
  | JustGRPO | 1.46 |
  | GRPO, RSPO, TraFL comp 4 | ≈ 1.40 |
  | Ent-PPO | 1.27 |

  At KL 3 GRPO and RSPO fall to 0.58–0.67 while JustGRPO keeps 1.6. The predicted "TB / TraFL ≥ ESPO" is wrong:
  ESPO is the most diverse at a matched distance.
- **Under the low-confidence-remasking decoder (T 0.6) every arm is near-deterministic** (entropy ≤ 0.06). It
  decodes essentially one string: score 0.713 for TraFL, TB, ESPO and Ent-PPO; 0.947 for GRPO, JustGRPO and RSPO. No
  diversity ranking exists under the realistic decoder here.

## Scope and what this does not show

- **One reference:** the k = 5 MLP on Potts, an ≈ 100-string law (FRAME R11 fails).
- **One environment, 2 seeds, one knob setting per arm.** The frontier comes from the training trajectory, not a
  β / κ / λ sweep.
- **5,000 updates at lr 1e-4.** TraFL's terminal law is still moving at the end, so the stationary regime, where
  S2's drift lives, is reached here only by ESPO.

Next, to make these readings decisive:
- a β grid for TraFL / TB / Ent-PPO, so a stationary terminal law is reached at several distances;
- the variance penalty at a setting that reaches KL ≥ 3;
- other references (MLP k = 3, 7; transformer sizes);
- the realistic environment (plan step 6).
