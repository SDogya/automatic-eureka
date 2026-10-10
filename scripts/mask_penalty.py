# python scripts/mask_penalty.py   (from the repo root)
"""The variance penalty each TraFL mask estimator carries implicitly.

A squared residual of a K-mask average has expectation (mean)^2 + Var_masks(average): the second term is the implicit
anchor of the path law (gfn_lab S2 identity). For completions y drawn from a policy, this script estimates
Var_masks(average log-ratio) per completion by redrawing the masks R times, for 32 independent masks (this repo's TraFL
runs), 4 independent masks, and 4 masks as 2 complementary pairs (the paper's form), and reports the mean over
completions. Policies: grid 2 block A final checkpoints (TraFL 32 IID and TraFL comp 4 at beta 0.5) and the reference.
Writes results/report/mask_penalty.json.
"""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from src.architecture import load as load_checkpoint
from src.config import LENGTH, MASK
from src.rl.losses.estimators import per_mask_log_ratio
from src.rl.losses.trafl import Reference
from src.rl.samplers import complete_uniform

REFERENCE = Path("models/pretrain/mlp_5/best.npz")
POLICIES = {"reference (theta = ref)": REFERENCE,
            "TraFL 32 IID, beta 0.5, update 10k": Path("models/port_grid2/trafl_iid32_b0.5/seed_1/finetune_5/step_10000.npz"),
            "TraFL comp 4, beta 0.5, update 10k": Path("models/port_grid2/trafl_comp4_b0.5/seed_1/finetune_5/step_10000.npz")}
SCHEMES = (("32 independent", 32, "iid"), ("4 independent", 4, "iid"), ("4 = 2 complementary pairs", 4, "comp"))
REDRAWS, N = 400, 256

ref_arch, ref_params = load_checkpoint(REFERENCE)
reference = Reference(ref_arch.forward, ref_params)
out = {}
for name, path in POLICIES.items():
    arch, params = load_checkpoint(path)
    ys, _ = complete_uniform(arch.forward, params, jnp.full((N, LENGTH), MASK, dtype=jnp.int32), jax.random.key(0))
    contexts = jnp.full((N, LENGTH), MASK, dtype=jnp.int32)           # empty prompt: u = 8 hidden positions
    completions = ys[:, None, :]
    row = {}
    for label, k, scheme in SCHEMES:
        fn = jax.jit(lambda key: per_mask_log_ratio(arch.forward, params, reference, contexts, completions, key, k,
                                                    scheme).mean(axis=0)[:, 0])
        draws = np.stack([np.asarray(fn(key)) for key in jax.random.split(jax.random.key(1), REDRAWS)])  # (R, N)
        row[label] = {"penalty_mean": float(draws.var(axis=0, ddof=1).mean()),
                      "score_mean": float(draws.mean())}
    single = jax.jit(lambda key: per_mask_log_ratio(arch.forward, params, reference, contexts, completions, key, 1,
                                                    "iid")[0, :, 0])
    one = np.stack([np.asarray(single(key)) for key in jax.random.split(jax.random.key(2), REDRAWS)])
    row["single-mask variance (sigma^2)"] = float(one.var(axis=0, ddof=1).mean())
    out[name] = row
    print(name, json.dumps(row))
Path("results/report").mkdir(parents=True, exist_ok=True)
Path("results/report/mask_penalty.json").write_text(json.dumps(out, indent=1) + "\n")
