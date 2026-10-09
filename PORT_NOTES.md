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
     - c_ref(tau | y): RTB / `tb`, `entppo`;
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

## Next

The answers to the first round of questions and the ordered plan with its gates are in `INTEGRATION_PLAN.md`. The
one open decision is the environment for the realistic reference (plan §1).
