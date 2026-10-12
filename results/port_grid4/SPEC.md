# Grid 4: the same methods on a transformer backbone (broader base)

Written 2026-10-12 12:53, before any run. Run locally (M2 Pro, AC power): `scripts/port_grid4.sh`, 6 single-thread
runs in parallel. A smoke test gave 0.12 s per update and 7 s per exact evaluation, so each run is ≈ 25–35 minutes.

## What runs

| | |
|---|---|
| reference = start | transformer d32 (2 layers, 4 heads; 26,020 parameters), pretrained on Potts β 0.75 |
| reference quality | KL(Potts ‖ ref) 0.144 against the MLP's 0.099; memorisation index 0.91 (learns, not stores); order imbalance 0.087 |
| arms (2 seeds each) | TraFL comp 4 β 0.5, TB β 0.5, ESPO κ 0.05, RSPO λ 0.01, GRPO, JustGRPO |
| everything else | grid 3's settings (10k updates, learning rate 1e-4, same eval steps) |
| MLP twins | the grid-3 runs (MLP k = 5 reference, 3,424 parameters) |

**What differs between the twins.** The pair differs in the reference *and* the policy family, so "backbone" here
means the pair. The references are close (KL 0.10 vs 0.14 from the law).

## Predictions, stated now (MLP values from REPORT §4b)

- **T1, ordering on wasted KL.** At matched attained tilt b\* = 0.2, wasted KL orders as: GRPO, JustGRPO and RSPO, then
  ESPO, then TraFL ≈ TB. The MLP values were 0.33–0.35 > 0.23 > 0.15–0.16.
- **T2, per-string drift.** TraFL's late path / terminal ratio exceeds TB's. The MLP values were 0.19 vs 0.10.
- **T3, the effective exponent (open).**
  - The comparison point is TraFL's stationary b\* / stated on the MLP: 0.80.
  - Near 1 would mean the shortfall depends on the backbone, which fits sampler weighting under parameter sharing.
  - Near 0.8 would mean it does not.
  - TB, still rising on the MLP, gives the other comparison.
- **T4, JustGRPO's lenses.**
  - At 10k, its left-to-right and decoder laws coincide (TV < 0.1), and its random-order law is broader.
  - If attention shares more across orders, the gap between those views could shrink.

## Readout

The same tools as for grid 3, on `models/port_grid4`:
- `summary`, `report`, `tilt` and `attained`;
- the three-lens table for JustGRPO, GRPO, TraFL and TB.

Results go in `READING.md` and REPORT §4c.
