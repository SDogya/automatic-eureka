# Integration plan: TraFL, its baselines and gfn_lab's analysis on this repo

Written 2026-10-09, after the first port (`PORT_NOTES.md`). One plan, ordered so that every step ends with the whole
test suite green, and no step changes an existing number without an equivalence check. Facts are tagged
**[verified]** (checked in code, data or paper text here) or **[to verify]**.

## 1. Answers that shape the plan

- **Which losses TraFL compares with [verified, both paper versions].** The direct baselines are ESPO and JustGRPO.
  D1 (diffu-GRPO) and DMPO enter only through published numbers; SPG appears in related work. Both direct
  baselines are ported, and each was checked against its paper's text:
  - **ESPO:** ratio exp((ELBO_θ − ELBO_old) / L), k2 = ½ (ELBO_θ − ELBO_ref)², masks shared.
  - **JustGRPO:** AR policy with future positions masked, token-level clip, KL 0, one update per batch.
- **RSPO** (Relative Score Policy Optimization, arXiv 2605.10218, ICML 2026) is not a TraFL baseline. It regresses
  the log-ratio log π/π_ref onto a reward-implied target, which makes it a close relative of TraFL, so it is worth
  an arm. Its exact objective is [to verify] from the paper before implementing.
- **Mask counts [verified].** All four committed TraFL runs on `anton-dev` use K = 32 IID masks with the "paper"
  normalisation (their `settings.json`). `main` has no TraFL code.
- **bitseq-thread estimators** (`research/bitseq/REPORT.md` §3), mapped here:
  - exact mask mean: all 2^u − 1 masks, count-uniform then subset-uniform. This is the K → ∞ limit of the IID
    surrogate, i.e. the exact ELBO / u;
  - four-mask square: TraFL with K = 4 IID;
  - split product U₂₊₂ = mean(δ₁, δ₂) · mean(δ₃, δ₄), unbiased for δ̄²;
  - the six-pair U-statistic (1/6) Σ_{j<k} δ_j δ_k: also unbiased for δ̄², with lower variance than U₂₊₂;
  - exact completion ("completion balance"): exact log p(y|x) by the subset recursion;
  - sampled-path TB: the existing `tb` arm, RTB-type with the reference.
- **Entropic PPO.** torchgfn 2.4.1 ships `EntPPOGFlowNet` (soft-RL PPO with a backward policy and a soft value).
  With a reference, the soft reward is Σ_t [log P_ref − log P_θ](step) + β log R, and the optimum is RTB's
  target. Port it in JAX:
  - gate on the soft-RL identity: at ratio 1 its expected gradient equals the gradient of the trajectory-level
    reverse KL to P_ref R^β / Z;
  - [to verify] whether a direct numeric cross-check against torchgfn's loss on identical trajectories with a table
    policy is cheap.
- **Decoders.** Real dLLM evaluation is confidence-ordered: TraFL uses Fast-dLLM low-confidence remasking (T 0.6,
  top-p 0.91), and JustGRPO and LLaDA use similar decoders.
  - Every arm gets exact evaluation under a confidence decoder (most confident position, token at T ∈ {1, 0.6}),
    besides the random-order and left-to-right ones.
  - Training samplers stay each method's own.
  - Cost: one recursion over the 390,625 states with one position per state.
- **Potts.** log R = β Σ_{i<j} J_ij(x_i, x_j) with random Gaussian couplings: a synthetic, rugged energy landscape
  (the q = 4 generalisation of Ising), not biology. [verified] At β = 2, 93 strings carry 99 % of its mass, and the
  10,000 MH pretraining samples contain only 160 distinct strings. The reference is effectively a ~100-string law,
  which any model can store (FRAME R11 fails). Post-training target Potts + TFBind8: 1,433 strings carry 99 %.
- **Realistic environments here.** TFBind8 is real: wet-lab binding for all 65,536 8-mers. Options for the
  pretrain + post-train pair:
  - **(a)** reference pretrained on lower-scoring TFBind8 sequences (the design-bench offline protocol),
    post-trained toward exp(β · score): a real reward and a data-derived reference;
  - **(b)** the current Potts → Potts + TFBind8, kept for continuity;
  - **(c)** `main`'s mode reward with a reference that is not π itself.

  Recommendation: (a) as the primary environment, (b) kept, (c) only if planted modes are wanted for coverage
  metrics. **Needs the principal's choice.**
- **Compute [verified].** 21 ms/step (TraFL K = 32) and 32 ms/step (ESPO-PPO μ = 8), single-thread, MLP k = 5,
  256 × 5 rollouts. That is 5–10 min per 10k-step run with exact evaluations, about 10 runs in parallel locally.
  - No Kaggle infrastructure is needed for MLP grids. Transformer step times will be measured before deciding.
  - JAX on Apple GPU (jax-metal) is not used: it is experimental.
- **Generalised analysis.** One adapter turns each run (`settings.json` + `evals.csv`) into rows of a tidy table.
  gfn_lab's analyses are ported as functions on that table: frontiers, matched-at-terminal-KL tables,
  ratio-by-window and the mechanism figure. The bitseq figures belong to the Codex thread: the ideas are ported,
  the files are not touched.

## 2. Interfaces, each with one source of truth

1. **Model:** `architecture.Architecture` (init / forward / save / load). The RL loop still loads MLPs only; step 1
   moves it onto the registry.
2. **Loss contract:** `loss(policy, forward, contexts (C,L), completions (C,G,L), [orders], rewards (C,G), key,
   **knobs) -> scalar`. The reference is passed as `Reference(forward, params)`; log Z appears only in balance
   losses.
3. **Samplers:** `complete_uniform` / `complete_ar(forward, params, tokens, key) -> (tokens, orders)`.
4. **Exact decoders for evaluation:** uniform, AR, confidence(T) → log p(y) over the 65,536 strings, in one string
   order (gated).
5. **Run manifest:** `settings.json` records arm, knobs, the architecture spec, the reference path and the
   environment config. `evals.csv` has fixed column names, the `EvalMetric` fields, which the adapter reads.

## 3. Order of work (each step: build, gate, full suite green, commit; ✓ = done 2026-10-09, suite 81 green)

| Step | What | Gate |
|---|---|---|
| 1 ✓ | RL loop on the registry: any checkpoint as start/reference; random init by spec | MLP runs bit-identical to now; transformer run normalised and finite |
| 2 ✓ | Confidence decoder in exact evaluation | recursion = Monte Carlo of the sampler |
| 3 ✓ | bitseq estimator arms: exact mask mean, U₂₊₂, six-pair U, exact completion | exact mean = expectation of the IID surrogate (exhaustive); U-statistics unbiased for δ̄²; exact log p = `metrics.trajectories.log_p_y` |
| 4 ✓ | Entropic PPO | soft-RL gradient identity at ratio 1 |
| 5 ✓ | RSPO, after reading its objective | its stated identity |
| 6 | Environment choice, then a realistic reference | reference report: KL to its data law, concentration, held-out-by-string NLL |
| 7 ✓ | Analysis adapter + plots | adapter reproduces the `evals.csv` numbers |
| 8 (grids 1, 2A ✓; 2B/C, 3 running) | Pre-stated grid, runs, readings | predictions written before running; environment (b): `results/port_grid1/`, `results/port_grid2/`; grid 3 on the broader base (`results/port_grid3/`). Summary: `results/report/REPORT.md` |

## 4. Known issues found in the existing code (transformer-port review), and what is done about them

1. Validation is not held out by string: 99.5 % of validation rows occur in training. "Best epoch" is then a
   training-loss criterion. This is fixed in the new environment's pretraining (step 6), not retro-fitted.
2. Old `training.json` reports contain `config.modes`, which the current `Config` rejects (`extra="forbid"`):
   readers parse the raw JSON.
3. `run_ablation` overwrites its summary: a guard will be added.
4. MLP references of width ≤ 32 were still improving at epoch 100 (best = last). This is reported as part of the
   reference-quality factor.
5. Evaluation closes a memmap through a private attribute: left as is (it works, numpy 2).

## 5. Limits

- Exact path metrics cover the empty prompt; per-prompt versions need a per-prompt recursion (feasible, separate).
- RSPO and Entropic PPO are as faithful as their papers and code allow.
- Nothing here says anything about 8B-scale dLLMs.
