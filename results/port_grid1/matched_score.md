Expected TFBind8 score at matched KL(p_theta || p_ref); mean [min-max] (seeds)

| configuration | 0.1 | 0.3 | 1 | 3 |
|---|---|---|---|---|
| entppo[beta=1.5, gae_lambda=0.7, ppo_epochs=4, eps_clip=0.2] mlp_w32 from mlp_5 | 0.429 [0.429-0.429] (2) | 0.503 [0.503-0.503] (2) | 0.618 [0.618-0.618] (2) | 0.793 [0.793-0.794] (2) |
| espo[kappa=0.05, mask_scheme=comp, mask_samples=4] mlp_w32 from mlp_5 | 0.428 [0.427-0.429] (2) | 0.491 [0.491-0.491] (2) | 0.624 [0.624-0.625] (2) | - |
| espo_ppo[kappa=0.05, mask_scheme=comp, mask_samples=4, ppo_epochs=8, eps_clip=0.2] mlp_w32 from mlp_5 | 0.427 [0.424-0.429] (2) | 0.490 [0.489-0.491] (2) | 0.623 [0.623-0.623] (2) | - |
| grpo[] mlp_w32 from mlp_5 | 0.428 [0.428-0.429] (2) | 0.493 [0.492-0.493] (2) | 0.620 [0.619-0.620] (2) | 0.786 [0.786-0.786] (2) |
| justgrpo[ppo_epochs=1, eps_clip=0.2] mlp_w32 from mlp_5 | 0.424 [0.424-0.424] (2) | 0.490 [0.490-0.490] (2) | 0.617 [0.617-0.618] (2) | 0.777 [0.776-0.778] (2) |
| rspo[rspo_lambda=0.01, advantage_std=True, mask_scheme=iid, mask_samples=4] mlp_w32 from mlp_5 | 0.427 [0.427-0.427] (2) | 0.493 [0.493-0.494] (2) | 0.620 [0.620-0.621] (2) | 0.788 [0.786-0.789] (2) |
| tb[beta=1.5] mlp_w32 from mlp_5 | 0.429 [0.429-0.429] (2) | 0.492 [0.492-0.493] (2) | 0.621 [0.621-0.621] (2) | - |
| trafl[beta=1.5, estimator=square, mask_scheme=comp, mask_samples=4, var_lambda=0.0, normalization=paper] mlp_w32 from mlp_5 | 0.432 [0.431-0.433] (2) | 0.496 [0.496-0.496] (2) | 0.623 [0.623-0.623] (2) | - |
| trafl[beta=1.5, estimator=square, mask_scheme=comp, mask_samples=4, var_lambda=1.0, normalization=paper] mlp_w32 from mlp_5 | 0.432 [0.431-0.433] (2) | 0.496 [0.496-0.496] (2) | 0.623 [0.623-0.623] (2) | - |
| trafl[beta=1.5, estimator=exact_lik, mask_scheme=iid, mask_samples=32, var_lambda=0.0, normalization=paper] mlp_w32 from mlp_5 | 0.428 [0.427-0.429] (2) | 0.490 [0.489-0.490] (2) | 0.620 [0.620-0.621] (2) | - |
| trafl[beta=1.5, estimator=square, mask_scheme=iid, mask_samples=32, var_lambda=0.0, normalization=paper] mlp_w32 from mlp_5 | 0.428 [0.427-0.428] (2) | 0.490 [0.489-0.490] (2) | 0.621 [0.620-0.621] (2) | - |
