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
| Exact distance from the reference: terminal KL both ways, trajectory KL, path KL = KL_traj − KL_term; AR-decoder score and KL | `src/metrics/paths.py`, `train.py` | self-distance 0; state recursion = Monte Carlo; AR law aligned with string order |
| Transformer + architecture registry + size ablation | `src/transformer.py`, `src/architecture.py` | in progress (separate agent) |

`--arm trafl` with default flags reproduces the pre-port code **bit-for-bit** (6-step run, same seed: every
logged loss and exact metric identical). Gates were mutation-checked (each fails when its piece is broken).

## Things to know before reading any comparison

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

## Open questions for you

- **Second environment (fork).** `main`'s mode-distance reward (R = e^{1−d}, 4 modes) is not in `anton-dev`. It can be
  added as `reward: "modes"` in a few lines, but on `main` the MLP is pretrained on π itself, so post-training
  toward p_ref · R would aim at roughly π². Which reference do you want there: Potts-pretrained (as here), a
  high-temperature π, or something else? Or skip it?
- Push `port-baselines` to the remote, or open a PR into `anton-dev`? Not done without your word.
- Pre-stated plan for the first comparison, Potts + TFBind8, from the MLP references k ∈ {3, 5, 7} plus 2–3
  transformer sizes, 2–3 seeds:
  - arms: TraFL (32 IID as here; 4 comp as the paper), TraFL + λ = 1 variance penalty, TB, ESPO (κ grid),
    ESPO-PPO (μ = 8), GRPO, JustGRPO;
  - reading: reward vs KL(p ‖ p_ref) frontiers, path KL at matched terminal KL, local-maxima coverage, and AR vs
    random decoder;
  - prediction from gfn_lab: the per-string arms (TraFL, ESPO) drift in path law more than TB and GRPO, the
    penalty removes most of that, and 32 IID masks drift more than 4 comp.
