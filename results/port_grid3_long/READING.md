# Long decoder-rollout runs: reading against SPEC.md

The kernel `ae-grid3-long` ran the 4 runs in 50 minutes, with the gate passing (45 tests). Trajectories are in
`models/port_grid3_long/*/seed_*/*/evals.csv`. The final-checkpoint tilt analysis is in `tilt.md` and `tilt.png`, and
order heatmaps in `orders.png`. Values are seed means; the seeds agree to ±2 % unless a range is given.

| update | terminal KL from ref (uniform / decoder rollouts) | KL(target ‖ p) | path / terminal | decoder score | top-1 % distinct in 16 decoder draws |
|---|---|---|---|---|---|
| 10k | 1.12 / 0.55 | 0.32 / 0.67 | 0.21 / 0.30 | 0.734 / 0.720 | 4.0 / 2.9 |
| 20k | 1.16 / 0.63 | 0.30 / 0.64 | 0.21 / 0.31 | 0.771 / 0.721 | 4.2 / 3.0 |
| 30k | 1.14 / 0.59 | 0.29 / 0.65 | 0.21 / 0.32 | 0.759 / 0.714 | 4.6 / 2.4 |

**P1 (slow, not a different fixed point): refuted.**
- Decoder-rollout TraFL stops at terminal KL ≈ 0.6: 0.60 / 0.63 / 0.60 / 0.59 at 15 / 20 / 25 / 30k. Over the last 10k
  it changes by −0.035, a fluctuation with no upward trend.
- Its fitted tilt is b* = 0.22, against 0.33 with uniform rollouts (stated 0.4).
- So in practice it settles elsewhere, even though every string has positive decoder probability.
- Candidate explanation, untested: a squared residual is only pushed to zero where strings are sampled. The model
  shares parameters across strings, so the fitted law is weighted by the sampler, and the decoder's weight on most
  strings is tiny. Uniform rollouts show a milder version of the same shortfall (b* / stated 0.81; `tilt.md`).

**P2 (higher drift with decoder rollouts): supported.**
- Path / terminal is 0.32 vs 0.21 at 30k.
- The uniform run's ratio has plateaued at 0.21 since 15k; the decoder run's is still rising slowly.

**P3 (deployed law): split.**
- The score part is refuted: decoder score 0.714 vs 0.759, outside the stated ±0.02.
- The diversity part is supported: 2.4 vs 4.6.

**Not predicted: the uniform run's decoded law keeps moving after its string law has stopped.**
- From 10k to 30k:
  - its random-order terminal KL stays at 1.12–1.16;
  - its decoder law moves further from the reference's (KL 6.1 → 11.5);
  - its decoder score rises from 0.734 to 0.759.
- The per-string objective leaves the conditionals free, and the decoder reads conditionals, not the string law.
  Here the drift went in a benign direction, but nothing in the objective controls it.
- The decoded law is also seed-fragile: TV to the decoded target is 0.30 vs 0.73 for the two seeds.
