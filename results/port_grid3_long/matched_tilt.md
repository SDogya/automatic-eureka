Every metric at the first crossing of attained tilt b* = level (seed mean; '-' = never reached). b* from expected_log_reward; π_b = p_ref e^{b r} / Z_b.


**b\* = 0.1**

| configuration | stated b | update reached | KL(p ‖ π_b*) | KL(p ‖ p_ref) | path KL | top-1 % distinct in 16, random | same, decoder | score | score, decoder |
|---|---|---|---|---|---|---|---|---|---|
| trafl β0.5 square comp ×4 | 0.4 | 123 | 0.0776 | 0.169 | 0.0137 | 3.82 | 2.67 | 0.474 | 0.652 |
| trafl β0.5 square comp ×4 decoder | 0.4 | 169 | 0.09 | 0.182 | 0.0146 | 3.9 | 2.44 | 0.482 | 0.677 |

**b\* = 0.2**

| configuration | stated b | update reached | KL(p ‖ π_b*) | KL(p ‖ p_ref) | path KL | top-1 % distinct in 16, random | same, decoder | score | score, decoder |
|---|---|---|---|---|---|---|---|---|---|
| trafl β0.5 square comp ×4 | 0.4 | 436 | 0.151 | 0.57 | 0.0498 | 6.24 | 3.74 | 0.568 | 0.731 |
| trafl β0.5 square comp ×4 decoder | 0.4 | 3779 | 0.139 | 0.556 | 0.128 | 6.13 | 2.7 | 0.567 | 0.716 |

**b\* = 0.3**

| configuration | stated b | update reached | KL(p ‖ π_b*) | KL(p ‖ p_ref) | path KL | top-1 % distinct in 16, random | same, decoder | score | score, decoder |
|---|---|---|---|---|---|---|---|---|---|
| trafl β0.5 square comp ×4 | 0.4 | 1886 | 0.198 | 1.18 | 0.159 | 8.24 | 3.51 | 0.665 | 0.729 |
| trafl β0.5 square comp ×4 decoder | 0.4 | - | - | - | - | - | - | - | - |
