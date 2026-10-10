# Port notes: TraFL baselines and a transformer on this repo's length-8 DNA task

Local branch `port-baselines` (from `origin/anton-dev`). Nothing is pushed. Written 2026-10-09.

## What was added (all exact-gated; `uv run python -m pytest`)

| Piece | Where | Gate |
|---|---|---|
| ESPO, single update and PPO-clipped reuse (ESPO's μ) | `src/rl/losses/baselines.py` | first PPO epoch = single update = explicit formula (`tests/test_arms.py`) |
| GRPO on the exact sampled-trajectory log-probability | same | per-trajectory gradient = string-level gradient (Fisher identity, exact sums) |
| JustGRPO: the denoiser as a left-to-right policy, token-level clip, KL 0 | same + `src/rl/samplers.py` | AR law sums to 1; equals the trajectory law along the ascending order × u!; sampler frequencies match it |
| TraFL with the paper's complementary mask pairs (`--mask-scheme comp`) | `src/rl/losses/masks.py`, `trafl.py` | pairs partition the hidden set; uniform model gives −log 4 |
| TraFL + across-order variance penalty (`--var-lambda`) | `baselines.order_variance` | unbiased for Var over orders (Monte Carlo vs exact) |
| `tb` arm = the existing exact trajectory-balance loss (RTB-type) | `src/rl/train.py` | existing tests |
| TraFL estimators from the bitseq thread (`--estimator`): split product, pairwise U-statistic, exact mask mean, exact completion likelihood | `src/rl/losses/estimators.py` | exact log p = sum over orders (value and gradient); exact mean = IID limit; U-statistics unbiased, square biased |
| Entropic PPO (`--arm entppo`): torchgfn's `EntPPOGFlowNet` with RTB's soft reward (log P_ref per step, beta r at exit) | `src/rl/losses/entppo.py` | expected gradient = trajectory-KL gradient (exact); value targets = soft reward-to-go |
| RSPO (`--arm rspo`, arXiv 2605.10218 Eq. 3.3; lambda 0.01) | `baselines.rspo_loss` | gradient = lambda * grad of 0.5 (delta_hat - A / lambda)^2 |
| Relative detailed balance / sub-trajectory balance (`--arm db`, `--arm subtb`; the log Z head is the state flow V(s), V(y) = beta r) | `src/rl/losses/flow_balance.py` | one-step residuals telescope to RTB's residual; SubTB -> DB (lambda -> 0) and -> RTB (lambda -> inf) |
| Kaggle CPU runner: lockfile env remotely, gate tests first, 4 single-thread runs per kernel, outputs copied as each run ends | `scripts/kaggle_job.py` | smoke kernel: env in 21 s, 34 gate tests, transformer job ran |
| Decoder rollouts (`--rollout decoder`): training completions from low-confidence remasking, as TraFL's paper samples (per-string arms only) | `src/rl/samplers.complete_proposal` | terminal frequencies and first-step (position, letter) law = exact decoder law, incl. tempered-scoring mutation |
| Exact diversity in TraFL's sense: distinct strings in 16 draws, overall and among top-1 % reward strings, random-order and decoder | `train.diversity` | uniform and point-mass identities |
| Decoder-level path distance (Hellinger excess) and post-hoc endpoint evaluation | `metrics/decoders.proposal_hellinger`, `analysis/endpoints.py` | zero for identical models; trajectory BC = Monte Carlo |
| Foundation diagnostics (concentration, memorisation index, TraFL target spread, order imbalance, decoders) and a broader base (pretraining Potts β 0.75) | `analysis/foundation.py`, `configs/potts_b0.75.json`, `models/pretrain_b0.75/` | — (exact computations) |
| Exact low-confidence-remasking decoder (LLaDA / Fast-dLLM, T = 0.6) in every evaluation (`llada_*` columns) | `src/metrics/decoders.py` | step law = one-step simulation (catches tempered scoring); terminal law = simulation |
| Exact distance from the reference: terminal KL both ways, trajectory KL, path KL = KL_traj − KL_term; AR-decoder score and KL | `src/metrics/paths.py`, `train.py` | self-distance 0; state recursion = Monte Carlo; AR law aligned with string order |
| Transformer + architecture registry + size ablation (568 to 399k parameters) | `src/transformer.py`, `src/architecture.py`, `models/pretrain/transformer_*` | shapes; parameter counts; bit-exact checkpoints; normalised exact law; forward = torch reference to 4e-6 |

`--arm trafl` with default flags reproduces the pre-port code **bit-for-bit** (6-step run, same seed: every
logged loss and exact metric identical). Gates were mutation-checked (each fails when its piece is broken).

## Things to know before reading any comparison

0. **Two arms are adaptations, not literal ports.**
   - **Ent-PPO** is defined without a reference: target R/Z with a fixed backward policy. Here the per-step soft
     reward is log P_ref instead of log P_B. That is exactly the standard form with P_B := the reference's own
     backward policy and terminal reward p_ref(y) R^beta, because sum_t log P_ref = sum_t log P_B^ref + log p_ref(y).
     The optimum and the expected gradient at lambda = 1 are the same; per-step credit assignment (GAE at
     lambda < 1) differs.
   - **RSPO** is a squared-residual regression in disguise (gradient identity above): TraFL-like, with batch centering
     in place of log Z and beta ~ 1 / lambda.
   - Path targets at the optimum:
     - c_ref(tau | y): RTB / `tb`, `db`, `subtb`, `entppo`;
     - uniform order: a GFlowNet with uniform P_B (not added; can be);
     - not pinned: the per-string objectives (TraFL's surrogate, exact likelihood, RSPO, ESPO's k2).

1. **The TraFL runs here use 32 IID masks** (`trafl_b15_k32`, sweep default `--mask-samples 32`); the paper uses 4
   as two complementary pairs, and the README says 4. In gfn_lab's analysis the squared loss over K masks adds
   Var_masks / K, which anchors the path law. With K = 32 that anchor is ~8× weaker, so these runs should drift
   more than the paper's estimator. `--mask-scheme comp --mask-samples 4` gives the paper's form.
2. **Arms have different knobs and fixed points.** TraFL and TB target p_ref · exp(β r). ESPO's k2 coefficient κ has
   no exact match to β. GRPO and JustGRPO have no anchor and go towards maximal reward. A fair comparison is along
   frontiers: reward vs KL(p ‖ p_ref), and path KL at matched terminal KL, not endpoints at one setting.
   `kl` / `reverse_kl` in `evals.csv` are distances to TraFL's target, not the other arms' objectives.
3. **JustGRPO is trained with its own left-to-right sampler.** Its random-order metrics are an off-policy view;
   the `ar_*` columns are its own decoder. The paper decodes in parallel at inference.
4. **ESPO details not fixed by the paper:** ε = 0.2 (TRL default); μ = 8 updates per batch in gfn_lab's earlier
   port; masks shared by θ, θ_old and the reference.
5. **GRPO** here is the exact sampled-path form. d1's diffu-GRPO one-step mean-field estimator is not ported.
6. **Prompts:** default `--context-source uniform` reveals random letters, so many prompts sit off the reference's
   support. That is the repo's existing choice and it is kept for every arm.
7. **Size ablation:** 4⁸ = 65,536 strings and 5⁸ = 390,625 contexts. The MLPs (k = 1..7) have ≤ 25k parameters;
   the transformer sizes go beyond the context count, to see where a model can store the table.

## Units and targets across arms (checked 2026-10-10)

- **Group centering shrinks the tilt.** `trafl`, `tb` and `tb_dec` centre rewards within the group of G = 5. Their
  stationary tilt is β (1 − 1/G) = 0.8 β (FRAME.md, exact-score idealisation). `db`, `subtb` and `entppo` use raw
  β r: V(y) = β r(y) must depend on y alone, so they cannot centre.
  - **The same nominal β is therefore not the same target across those two groups.** Comparisons at matched
    distance (all READINGs) are unaffected.
  - To compare by target, use β_db = 0.8 β_tb.
- **TraFL's estimators** (`--estimator` other than square) score per token, so β / u always applies. A fixed bug made
  it follow `--normalization`.
- **`tb_dec` is exploratory.** The decoder law has true zeros, and the policy's zero set differs from the reference's,
  so log P_θ^dec / P_ref^dec is unbounded (floored at 1e-30). The rising loss in its smoke run is consistent with this.
  It is the concrete reason per-trajectory objectives under argmax-type decoders are ill-conditioned.

## Pipeline facts from an 8-arm, 300-step pilot (setup facts, not results)

- **Grid design:** at their learning rate (1e-3, 256 prompts × 5) every arm passes terminal KL ≈ 3 by update 50.
  Matched comparisons at KL 0.1–1 need `--eval-steps 1 2 4 8 16 32 64` and/or a lower learning rate.
- **Decoder:** the low-confidence-remasking decoder at T = 0.6 collapses to about one string after post-training
  (entropy 0.001–0.015 nats, from 0.72 at the reference), while the random-order sampler keeps 0.5–1.3 nats. Read
  diversity per decoder.
- **Analysis:** `python -m src.analysis.report --root <runs> --out <dir>` writes the matched / window tables and
  four figures. Colour follows the arm in a fixed order (validated palette); line style marks the configuration.

## Next

The answers to the first round of questions and the ordered plan with its gates are in `INTEGRATION_PLAN.md`. The
one open decision is the environment for the realistic reference (plan §1).
