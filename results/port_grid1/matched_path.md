Path KL at matched terminal KL(p_ref || p_theta); mean [min-max] (seeds)

| configuration | 0.1 | 0.3 | 1 | 3 |
|---|---|---|---|---|
| entppo[beta=1.5, gae_lambda=0.7, ppo_epochs=4, eps_clip=0.2] mlp_w32 from mlp_5 | 0.0071 [0.0070-0.0073] (2) | 0.0204 [0.0199-0.0209] (2) | 0.0660 [0.0650-0.0671] (2) | 0.2221 [0.2213-0.2229] (2) |
| espo[kappa=0.05, mask_scheme=comp, mask_samples=4] mlp_w32 from mlp_5 | 0.0082 [0.0080-0.0083] (2) | 0.0235 [0.0233-0.0238] (2) | 0.1403 [0.1324-0.1483] (2) | - |
| espo_ppo[kappa=0.05, mask_scheme=comp, mask_samples=4, ppo_epochs=8, eps_clip=0.2] mlp_w32 from mlp_5 | 0.0096 [0.0095-0.0098] (2) | 0.0244 [0.0244-0.0244] (2) | 0.1299 [0.1292-0.1306] (2) | - |
| grpo[] mlp_w32 from mlp_5 | 0.0082 [0.0081-0.0083] (2) | 0.0242 [0.0241-0.0244] (2) | 0.0864 [0.0860-0.0868] (2) | 0.3435 [0.3405-0.3464] (2) |
| justgrpo[ppo_epochs=1, eps_clip=0.2] mlp_w32 from mlp_5 | 0.0146 [0.0144-0.0148] (2) | 0.0463 [0.0458-0.0468] (2) | 0.1510 [0.1490-0.1529] (2) | 0.5706 [0.5604-0.5809] (2) |
| rspo[rspo_lambda=0.01, advantage_std=True, mask_scheme=iid, mask_samples=4] mlp_w32 from mlp_5 | 0.0080 [0.0079-0.0081] (2) | 0.0238 [0.0237-0.0238] (2) | 0.0895 [0.0887-0.0904] (2) | 0.3960 [0.3723-0.4197] (2) |
| tb[beta=1.5] mlp_w32 from mlp_5 | 0.0073 [0.0071-0.0076] (2) | 0.0217 [0.0212-0.0222] (2) | 0.0748 [0.0739-0.0757] (2) | 0.2682 [0.2673-0.2691] (2) |
| trafl[beta=1.5, estimator=square, mask_scheme=comp, mask_samples=4, var_lambda=0.0, normalization=paper] mlp_w32 from mlp_5 | 0.0076 [0.0074-0.0078] (2) | 0.0213 [0.0211-0.0215] (2) | 0.0786 [0.0785-0.0787] (2) | 1.2016 [0.9825-1.4208] (2) |
| trafl[beta=1.5, estimator=square, mask_scheme=comp, mask_samples=4, var_lambda=1.0, normalization=paper] mlp_w32 from mlp_5 | 0.0075 [0.0073-0.0077] (2) | 0.0210 [0.0209-0.0212] (2) | 0.0777 [0.0776-0.0778] (2) | - |
| trafl[beta=1.5, estimator=exact_lik, mask_scheme=iid, mask_samples=32, var_lambda=0.0, normalization=paper] mlp_w32 from mlp_5 | 0.0069 [0.0067-0.0072] (2) | 0.0200 [0.0196-0.0204] (2) | 0.0699 [0.0686-0.0711] (2) | 0.4607 [0.4524-0.4690] (2) |
| trafl[beta=1.5, estimator=square, mask_scheme=iid, mask_samples=32, var_lambda=0.0, normalization=paper] mlp_w32 from mlp_5 | 0.0072 [0.0069-0.0075] (2) | 0.0208 [0.0204-0.0213] (2) | 0.0741 [0.0731-0.0751] (2) | 0.6427 [0.6349-0.6505] (2) |
