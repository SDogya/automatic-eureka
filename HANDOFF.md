# Handoff: TraFL baselines port (branch `port-baselines`), state on 2026-10-11

## State in one paragraph

- Every planned arm is ported and gated: TraFL (5 estimators × IID / complementary masks × order-variance penalty ×
  uniform / decoder rollouts), TB (= RTB), DB, SubTB, Ent-PPO, ESPO, ESPO-PPO, GRPO, JustGRPO, RSPO; `tb_dec` is
  exploratory. There are also the MLP and transformer backbones, with the transformer size ablation.
- 97 tests collected, all green at the last full run; the suite is the gate on the maths.
- Grids 1, 2A and 3 are finished and read. 2B/C were lost to Kaggle's 12-hour limit.
- **Running (2026-10-11):** kernel `ae-grid3-long`, TraFL comp 4 β 0.5, decoder vs uniform rollouts, 30k updates,
  2 seeds each (`results/port_grid3_long/SPEC.md`, pre-stated). Pull with `python3 scripts/kaggle_job.py pull
  grid3-long`. Kernels `ae-grid3-1`, `-1b` and `-1c` are COMPLETE and hold no slot.
- The report is `results/report/REPORT.md` (no longer a draft).

## Read in this order

1. `results/report/REPORT.md`: what is compared, whether the base is sound, all results, correctness, limits.
2. `PORT_NOTES.md`: what was added (with gates); caveats to read before any comparison; units and targets across
   arms (0.8 β centering).
3. `results/port_grid{1,3}/SPEC.md` + `READING.md`: pre-stated predictions and their verdicts. Grid 2: `SPEC.md`
   and the block A endpoints.
4. `INTEGRATION_PLAN.md`: steps and gates. Only step 6 (the realistic environment) is open.
5. `FORMULATIONS_V2.md` (RU, canonical, for other project members) with `FORMULATIONS_V2_AUDIT_DETAILS.md` and
   `TRAFL_OBJECTIVE_DISAMBIGUATION.md`.

## Findings that carry forward (numbers in REPORT §2–4)

- **Base.** On Potts β 2, the reference is effectively a 100-string law, and learning cannot be told from
  memorisation. Use the broader base: Potts β 0.75 (`configs/potts_b0.75.json`, `data/potts_b0.75/`,
  `models/pretrain_b0.75/`, diagnostics in `results/foundation/`).
- **Path structure.** Under random order, paths given y are near-uniform at the reference (order KL 0.04–0.10).
  Path structure in practice comes from the low-confidence decoder. Report every path and diversity number per
  sampler.
- **Grid 3 (broader base, 54 runs).**
  - Per-string TraFL objectives drift in path law at a settled terminal law: late ratio 0.17–0.19, against
    0.07–0.10 for the balance arms.
  - The explicit order-variance penalty cuts this to 0.11. Mask count, pairing and estimator do not matter.
  - Score at matched KL is equal across arms.
  - Diversity at matched KL: TraFL and the balance arms are highest under random order; ESPO is highest under the
    decoder.
  - Decoder rollouts (the paper's sampling) raise drift by 40–500 % and cut diversity.
- **Mask-penalty puzzle.** Grid 2A, env b: comp 4 drifted more than 32 IID, although its implicit penalty is 3–4×
  larger (`mask_penalty.json`). On the broader base the two are equal. The two candidate explanations are untested
  (REPORT §3).
- **Transformer size ablation** (Potts β 2, `models/pretrain/transformer_ablation.json`):
  - KL to the law falls from 0.97 (568 parameters) to 0.07 at 26k, then plateaus (0.057 at 399k).
  - On that base, sample = law, so the "learns pointwise" question needs the broader base. There, the MLP k = 5
    reference learns, not stores: memorisation index 0.93.
  - No post-training on a transformer reference has run: grid 2C was lost.

## Commands (repo root; `uv run --no-sync`, Python 3.14 / JAX)

```bash
uv run --no-sync python -m pytest tests/ -q                       # gates; run before trusting any new number
uv run --no-sync python -m src.rl.sweep --help                    # one run: --arm, --estimator, --mask-scheme, --var-lambda, --rollout, --eval-steps, --start ...
# Kaggle (CPU kernels, 4 single-thread runs each); bundle = git archive of HEAD, so commit first
python scripts/kaggle_job.py bundle
python scripts/kaggle_job.py push <job> scripts/port_grid3_jobs_0.txt port_grid3
python scripts/kaggle_job.py status <job>
python scripts/kaggle_job.py pull <job>                            # paged, resumable; analysis files only ('all' adds checkpoints)
python3 scripts/kaggle_pull_pages.py arsen4ikvar/ae-<job> .scratch/kaggle/out/<job> '<regex>'   # targeted re-pull
# analysis
uv run --no-sync python -m src.analysis.report  --root models/port_grid3 --out results/port_grid3 --levels 0.3 1 2
uv run --no-sync python -m src.analysis.summary --root models/port_grid3 --out results/port_grid3 --level 1.0
uv run --no-sync python -m src.analysis.endpoints ...              # post-hoc decoder metrics at chosen checkpoints
uv run --no-sync python scripts/mask_penalty.py                    # implicit mask penalty per estimator
```

Run checkpoints live in `models/port_*/` (git-ignored); tables and figures go to `results/`.

## Kaggle gotchas (learned the hard way)

- **12-hour session limit.** A kernel that hits it keeps no outputs. The runner now copies each run to
  `/kaggle/working/out` as it ends, so plan each kernel well under 12 hours. Transformer runs take ≈ 4–5 hours on
  Kaggle CPU, so give them a GPU kernel or a lighter eval schedule.
- **Pulls.** The CLI returns only the first 20-file page and aborts on transient SSL errors. Always pull with
  `kaggle_pull_pages.py` (pages of 200, retries, resume). Filtered paging can still miss files; check run counts
  after a pull and re-pull missing runs with a targeted regex.
- **Slots.** The account has 5 concurrent CPU sessions, shared with the gfn_lab session. Message it before taking
  more than 2.
- **Pushing.** A push beyond 5 running sessions is rejected ("Maximum batch CPU session count of 5 reached"), not
  queued: Kaggle's QUEUED state is only for accepted sessions waiting for a machine. Unattended shell retry loops
  failed on 2026-10-09 for an undiagnosed reason, plausibly the Mac sleeping or going offline. Push by hand, or run
  the loop under `caffeinate -i` and confirm with `kaggle kernels status`.
- **Credentials.** They stay in `~/.kaggle`; never print them. The W&B key is never put in YAML or on a command
  line.

- **Decoder on the true law** (exact; REPORT §2, `scripts/decoder_true_law.py`):
  - the decoder concentrates even the true Potts law (99 % on 19 strings);
  - the reference (KL 0.10 from the law) decides where it lands: low-reward strings (score 0.30 vs 0.38);
  - order inconsistency is not the cause.

## Open decisions for the principal

1. **The realistic environment (plan step 6).** The broader Potts base is the working proposal. A real-data
   reference (held-out-by-string pretraining) is not built.
2. **The transformer reference arm.** Re-run grid 2C on the broader base, on a GPU kernel or with sparser evals?
3. **`tb_dec`: recommend dropping it.**
   - **It is not a numerical problem.** The decoder law has exact zeros, and θ and the reference have different zero
     sets, so the log ratio is unbounded.
   - **Its target may be unreachable.** The decoder's step law is a fixed function of the denoiser's marginals, not
     a free policy.
   - **ε-smoothing would not fix it.** Mixing ε into the *scored* law changes the target (it then depends on ε) and
     cures only the zeros.
   - **Where ε does belong.** If we want balance arms trained on decoder samples, use off-policy `tb`:
     - it still scores the random-order law, so its fixed point is unchanged;
     - rollouts come from the decoder mixed with ε uniform-order steps, for coverage;
     - the current guard rejects `--rollout decoder` for `tb` too strictly.

## Next steps, by value

1. **Test the two mask-penalty explanations.** Truncated estimand vs order variance: compute both on the grid 2A
   checkpoints, which needs checkpoint pulls (`pull <job> all`) for the runs concerned. This is cheap.
2. **Grid 4 on the broader base.** Reference size (MLP k 3 / 5 / 7, transformer d32), plus penalty λ at a larger β.
   Write SPEC.md first, as for grids 1–3.
3. **β-matched rather than distance-matched tables.** Use β_db = 0.8 β_tb.

## Housekeeping

- Untracked `models/pretrain/transformer_*/{last.npz,*.png,best_distribution.parquet}` and
  `models/port_grid1_driver.log` are by-products of the pretraining runs. They are not needed: `best.npz` and
  `training.json` are tracked.
- Local constraints from the principal:
  - no big local grids (smoke and verification are fine);
  - this machine is often on battery;
  - do not touch other threads' folders in gflownet-dllm.
