Every metric at the first crossing of attained tilt b* = level (seed mean; '-' = never reached). b* from expected_log_reward; π_b = p_ref e^{b r} / Z_b.


**b\* = 0.1**

| configuration | stated b | update reached | KL(p ‖ π_b*) | KL(p ‖ p_ref) | path KL | top-1 % distinct in 16, random | same, decoder | score | score, decoder |
|---|---|---|---|---|---|---|---|---|---|
| trafl β0.5 square comp ×4 | 0.4 | 59 | 0.0707 | 0.158 | 0.0403 | 3.24 | 0.804 | 0.453 | 0.346 |
| tb β0.5 | 0.4 | 12 | 0.0813 | 0.17 | 0.0236 | 3.6 | 1.7 | 0.47 | 0.46 |
| espo κ0.05 comp ×4 | - | 21 | 0.116 | 0.204 | 0.0406 | 3.37 | 1.26 | 0.466 | 0.441 |
| grpo | - | 12 | 0.106 | 0.194 | 0.0283 | 3.62 | 1.66 | 0.476 | 0.473 |
| justgrpo μ1 | - | 16 | 0.124 | 0.212 | 0.0342 | 3.45 | 1.49 | 0.47 | 0.406 |
| rspo λ0.01 iid ×4 | - | 16 | 0.129 | 0.216 | 0.0359 | 3.44 | 1.51 | 0.471 | 0.397 |

**b\* = 0.2**

| configuration | stated b | update reached | KL(p ‖ π_b*) | KL(p ‖ p_ref) | path KL | top-1 % distinct in 16, random | same, decoder | score | score, decoder |
|---|---|---|---|---|---|---|---|---|---|
| trafl β0.5 square comp ×4 | 0.4 | 293 | 0.121 | 0.521 | 0.101 | 5.68 | 3.11 | 0.542 | 0.688 |
| tb β0.5 | 0.4 | 41 | 0.142 | 0.544 | 0.0736 | 5.7 | 2.77 | 0.544 | 0.719 |
| espo κ0.05 comp ×4 | - | 58 | 0.237 | 0.641 | 0.157 | 5.55 | 2.92 | 0.546 | 0.709 |
| grpo | - | 22 | 0.338 | 0.736 | 0.107 | 5.95 | 3.08 | 0.576 | 0.707 |
| justgrpo μ1 | - | 30 | 0.397 | 0.798 | 0.131 | 5.9 | 3.31 | 0.569 | 0.71 |
| rspo λ0.01 iid ×4 | - | 32 | 0.407 | 0.807 | 0.138 | 5.84 | 3.81 | 0.566 | 0.743 |

**b\* = 0.3**

| configuration | stated b | update reached | KL(p ‖ π_b*) | KL(p ‖ p_ref) | path KL | top-1 % distinct in 16, random | same, decoder | score | score, decoder |
|---|---|---|---|---|---|---|---|---|---|
| trafl β0.5 square comp ×4 | 0.4 | 1759 | 0.171 | 1.12 | 0.226 | 8.06 | 4.35 | 0.647 | 0.735 |
| tb β0.5 | 0.4 | 182 | 0.131 | 1.08 | 0.124 | 7.51 | 2.84 | 0.631 | 0.724 |
| espo κ0.05 comp ×4 | - | 148 | 0.259 | 1.2 | 0.353 | 7.54 | 3.05 | 0.638 | 0.786 |
| grpo | - | 32 | 0.643 | 1.59 | 0.234 | 8.27 | 3.45 | 0.668 | 0.828 |
| justgrpo μ1 | - | 49 | 0.772 | 1.7 | 0.341 | 7.53 | 2.54 | 0.668 | 0.846 |
| rspo λ0.01 iid ×4 | - | 48 | 0.735 | 1.64 | 0.306 | 7.45 | 2.77 | 0.671 | 0.838 |
