Terminal entropy (random-order sampler) at matched KL(p_theta || p_ref); mean [min-max] (seeds)

| configuration | 0.1 | 0.3 | 1 | 3 |
|---|---|---|---|---|
| entppo[beta=1.5, gae_lambda=0.7, ppo_epochs=4, eps_clip=0.2] mlp_w32 from mlp_5 | 2.673 [2.649-2.697] (2) | 2.260 [2.240-2.279] (2) | 1.267 [1.261-1.273] (2) | 1.152 [1.138-1.166] (2) |
| espo[kappa=0.05, mask_scheme=comp, mask_samples=4] mlp_w32 from mlp_5 | 2.652 [2.641-2.662] (2) | 2.372 [2.365-2.379] (2) | 1.615 [1.611-1.619] (2) | - |
| espo_ppo[kappa=0.05, mask_scheme=comp, mask_samples=4, ppo_epochs=8, eps_clip=0.2] mlp_w32 from mlp_5 | 2.640 [2.637-2.643] (2) | 2.384 [2.380-2.389] (2) | 1.690 [1.560-1.821] (2) | - |
| grpo[] mlp_w32 from mlp_5 | 2.699 [2.695-2.703] (2) | 2.393 [2.387-2.398] (2) | 1.421 [1.418-1.423] (2) | 0.668 [0.634-0.702] (2) |
| justgrpo[ppo_epochs=1, eps_clip=0.2] mlp_w32 from mlp_5 | 2.684 [2.678-2.690] (2) | 2.367 [2.357-2.378] (2) | 1.462 [1.455-1.470] (2) | 1.605 [1.600-1.611] (2) |
| rspo[rspo_lambda=0.01, advantage_std=True, mask_scheme=iid, mask_samples=4] mlp_w32 from mlp_5 | 2.689 [2.687-2.691] (2) | 2.372 [2.366-2.378] (2) | 1.405 [1.397-1.414] (2) | 0.580 [0.530-0.630] (2) |
| tb[beta=1.5] mlp_w32 from mlp_5 | 2.714 [2.709-2.719] (2) | 2.419 [2.419-2.419] (2) | 1.486 [1.483-1.488] (2) | - |
| trafl[beta=1.5, estimator=square, mask_scheme=comp, mask_samples=4, var_lambda=0.0, normalization=paper] mlp_w32 from mlp_5 | 2.633 [2.619-2.647] (2) | 2.306 [2.294-2.318] (2) | 1.400 [1.393-1.407] (2) | - |
| trafl[beta=1.5, estimator=square, mask_scheme=comp, mask_samples=4, var_lambda=1.0, normalization=paper] mlp_w32 from mlp_5 | 2.634 [2.619-2.648] (2) | 2.306 [2.295-2.318] (2) | 1.410 [1.404-1.416] (2) | - |
| trafl[beta=1.5, estimator=exact_lik, mask_scheme=iid, mask_samples=32, var_lambda=0.0, normalization=paper] mlp_w32 from mlp_5 | 2.685 [2.671-2.699] (2) | 2.433 [2.417-2.449] (2) | 1.506 [1.494-1.517] (2) | - |
| trafl[beta=1.5, estimator=square, mask_scheme=iid, mask_samples=32, var_lambda=0.0, normalization=paper] mlp_w32 from mlp_5 | 2.685 [2.671-2.698] (2) | 2.429 [2.413-2.445] (2) | 1.497 [1.491-1.504] (2) | - |
