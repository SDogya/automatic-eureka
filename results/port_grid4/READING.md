# Grid 4 (transformer backbone): reading against SPEC.md

All 12 runs finished locally on the M2 Pro: 6 parallel single-thread runs, 25–60 minutes each, 12:55–16:34 on
2026-10-12, no failures. Tables: `summary.md` (matched distance 1), `tilt.md` (final checkpoints), `matched_tilt.md`
(matched attained tilt). Figures: `tilt.png`, `attained.png`, `orders.png`, `path_by_string.png`. MLP twins are grid 3.
Seed ranges are tight unless given.

| method | b* / stated (MLP → transformer) | wasted KL @ b* 0.2 | late path / terminal | score @ KL 1 | top-1 % distinct @ KL 1, random / decoder (transformer) |
|---|---|---|---|---|---|
| TraFL comp 4, β 0.5 | 0.80 → **0.82** | 0.15 → 0.12 | 0.19 → **0.26** | 0.637 → 0.629 | 7.65 / **4.34** |
| TB, β 0.5 | 0.91 → **0.95** | 0.16 → 0.14 | 0.10 → 0.09 | 0.639 → 0.618 | 7.29 / 2.86 |
| ESPO, κ 0.05 | — | 0.23 → 0.24 | — | 0.629 → 0.607 | 6.97 / 3.11 |
| GRPO | — | 0.33 → 0.34 | — | 0.617 → 0.611 | 6.78 / 3.22 |
| JustGRPO (random-order view) | — | 0.33 → 0.40 | — | 0.606 → 0.593 | 6.41 / 3.41 |
| RSPO, λ 0.01 | — | 0.35 → 0.41 | — | 0.605 → 0.595 | 6.29 / 3.52 |

**T1 (wasted-KL ordering): supported.** At b\* = 0.2 the order is RSPO 0.41 ≈ JustGRPO 0.40 > GRPO 0.34 > ESPO 0.24 >
TB 0.14 ≈ TraFL 0.12, the same order as on the MLP.

**T2 (TraFL drifts more than TB): supported, more strongly.**
- Late path / terminal is 0.26 against 0.09, 3.0× (MLP 1.9×).
- Path KL at matched terminal KL 1 is 0.24 against 0.13.

**T3 (does the effective exponent depend on the backbone?): no.**
- TraFL reaches b\* / stated = 0.82 on the transformer against 0.80 on the MLP. The mass-weighted slope is 0.31.
- TB reaches 0.95 (0.91 on the MLP, where it was still rising).
- So TraFL's shortfall survives a change of backbone, and TB nearly closes its gap. The shortfall is specific to
  TraFL's objective, or how it is trained, not to the MLP.

**T4 (do JustGRPO's views converge with attention?): refuted. The gap grows.**
- Its left-to-right and decoder laws are the same single string: TV ≤ 0.001, entropy 0.00–0.01.
- Its random-order law is broader than on the MLP: entropy 3.8–4.4 against 2.1, TV 0.79–0.82 from its left-to-right
  law against 0.61.
- The transformer carries left-to-right training over to other orders *less*.

**Not predicted.**
- **Which arm keeps the most top-string diversity under the decoder depends on the backbone and on how arms are
  matched.** There is no consistent leader, and decoder metrics are seed-fragile:

  | | matched KL 1 | matched tilt b\* 0.2 |
  |---|---|---|
  | transformer | TraFL 4.34 (others 2.86–3.52) | RSPO 3.81 (TraFL 3.11) |
  | MLP (grid 3) | ESPO 4.05 (TraFL 3.1–3.7) | TraFL 3.75–4.03 |

  *Corrected 2026-10-18: this bullet first said TraFL leads on both backbones.*
- **ESPO is backbone-sensitive:** path KL at matched terminal KL 1 is 0.28 on the transformer against 0.10 on the MLP.
- **The unanchored arms collapse completely:** GRPO's laws all sit on one string. RSPO's final path KL is 144 in the
  random-order view.
- **TraFL is slow to tilt:** it reaches b\* = 0.2 after 293 updates, against 22–58 for the others (on the MLP: 429 vs
  23–188).
- **Context for "decoder TV to decoded ref" ≈ 1:** the target itself, decoded, is also 0.994 from the decoded
  reference. Disjoint decoded laws are expected even for a perfect fit.
