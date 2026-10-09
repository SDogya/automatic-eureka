# Port grid 1: every arm from one reference, environment (b) (Potts → Potts + TFBind8)

Written 2026-10-09 21:29, before any run. Code: branch `port-baselines` at the commit that adds this file. Runs:
`scripts/port_grid1.sh` (checkpoints in the git-ignored `models/port_grid1/`). Report:
`python -m src.analysis.report --root models/port_grid1 --out results/port_grid1`.

## Setup (fixed for every arm)
- Reference and start: `models/pretrain/mlp_5/best.npz` (MLP, width 32, 3,424 parameters).
- Reward: log R = 3.5 (z_Potts + z_TFBind8) (`configs/potts_tfbind8.json`); prompts `--context-source uniform`.
- 256 prompts × 5 completions per update; Adam.
- Learning rate **1e-4**, not the repo's 1e-3: at 1e-3 every arm passed terminal KL ≈ 3 by update 50 in the pilot,
  leaving no points at matched levels 0.1–1.
- 5,000 updates, no plateau stop. Exact evaluation at updates 1, 2, 4, …, 2048 and every 250.
- 2 seeds (1, 2).

## Arms and knobs (one setting each; the frontier comes from the training trajectory)

| Group | Arms |
|---|---|
| Q1: per-string vs per-trajectory | TraFL as in the repo (32 IID masks); TraFL with the paper's 4 comp masks; TraFL 4 comp + λ = 1 variance penalty; TraFL with exact likelihood; TB (RTB-type); Ent-PPO (μ = 4) — all at β = 1.5 |
| Q2: the baselines TraFL is compared with, plus relatives | ESPO single update (κ = 0.05, 4 comp masks); ESPO-PPO (μ = 8); GRPO (sampled path); JustGRPO; RSPO (λ = 0.01, 4 masks) |

## Readings, stated before running
- **P1 (gfn_lab S2 mechanism, bitseq endpoints).** Path KL at matched terminal KL(p_ref ‖ p_θ):
  - TB ≈ Ent-PPO < TraFL comp 4 ≤ TraFL 32 IID < TraFL exact likelihood;
  - the λ = 1 penalty brings comp 4 to about TB's level;
  - the path / terminal ratio grows over training for the per-string arms and stays flat for TB and Ent-PPO.
  - P1 is refuted if TraFL exact likelihood or 32 IID sits at or below TB beyond the seed range at ≥ 2 matched levels.
- **P2 (TraFL's claim, re-scoped).**
  - The reward maximisers (GRPO, JustGRPO, RSPO at λ = 0.01) reach higher scores at higher terminal KL.
  - At matched KL(p_θ ‖ p_ref), TraFL and TB score comparably to them.
  - Under the random-order sampler, terminal entropy at matched KL ranks TB / TraFL ≥ ESPO > GRPO / JustGRPO.
  - Under the low-confidence-remasking decoder (T 0.6) all arms are near-deterministic, so no diversity ranking
    follows there. (The pilot showed this collapse; it is stated here as an expectation, not a test.)
- **Scope.** One reference (≈ 100-string Potts law, FRAME R11 fails), one environment, 2 seeds. Nothing here
  generalises beyond this setting.
