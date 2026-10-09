Median path KL / terminal KL by update window (terminal KL in (0.05, 1.5)); value (checkpoints)

| configuration | 1-5 | 5-30 | 30-166 | 166-910 | 910-5001 |
|---|---|---|---|---|---|
| entppo[beta=1.5, gae_lambda=0.7, ppo_epochs=4, eps_clip=0.2] mlp_w32 from mlp_5 | - | 0.069 (4) | - | - | - |
| espo[kappa=0.05, mask_scheme=comp, mask_samples=4] mlp_w32 from mlp_5 | - | - | 0.083 (6) | 0.278 (10) | 1.637 (38) |
| espo_ppo[kappa=0.05, mask_scheme=comp, mask_samples=4, ppo_epochs=8, eps_clip=0.2] mlp_w32 from mlp_5 | 0.099 (2) | 0.083 (4) | 0.263 (6) | 1.729 (10) | 1.939 (38) |
| grpo[] mlp_w32 from mlp_5 | - | - | 0.081 (4) | - | - |
| justgrpo[ppo_epochs=1, eps_clip=0.2] mlp_w32 from mlp_5 | - | - | 0.150 (6) | - | - |
| rspo[rspo_lambda=0.01, advantage_std=True, mask_scheme=iid, mask_samples=4] mlp_w32 from mlp_5 | - | - | 0.079 (4) | - | - |
| tb[beta=1.5] mlp_w32 from mlp_5 | - | - | 0.073 (4) | - | - |
| trafl[beta=1.5, estimator=square, mask_scheme=comp, mask_samples=4, var_lambda=0.0, normalization=paper] mlp_w32 from mlp_5 | - | - | 0.071 (5) | 0.092 (3) | - |
| trafl[beta=1.5, estimator=square, mask_scheme=comp, mask_samples=4, var_lambda=1.0, normalization=paper] mlp_w32 from mlp_5 | - | - | 0.070 (4) | 0.087 (4) | - |
| trafl[beta=1.5, estimator=exact_lik, mask_scheme=iid, mask_samples=32, var_lambda=0.0, normalization=paper] mlp_w32 from mlp_5 | - | - | 0.068 (6) | - | - |
| trafl[beta=1.5, estimator=square, mask_scheme=iid, mask_samples=32, var_lambda=0.0, normalization=paper] mlp_w32 from mlp_5 | - | - | 0.072 (6) | - | - |
