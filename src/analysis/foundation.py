"""Is the setting a sound base for comparing post-training methods? Exact diagnostics of a reward / reference pair.

python -m src.analysis.foundation --config configs/potts_tfbind8.json --references models/pretrain/mlp_5/best.npz ...

For the reward law pi and each reference p_ref (exact random-order terminal law):
  concentration   entropy and the number of strings that carry 99 % of the mass (65,536 strings in all)
  memorisation    the pretraining sample (<data_dir>/data8.parquet): distinct strings, their mass under the pretraining
                  law and under p_ref, KL(empirical || p_ref) next to KL(law || p_ref), and the index
                  p_ref(sample) / law(sample): 1 = learned the law, >> 1 = stored the sample. When the law lives on
                  fewer strings than the sample covers (Potts beta 2) the two are indistinguishable
  TraFL target    p* proportional to p_ref e^{beta r} for each beta: entropy, 99 % support, and the mass p* puts where
                  p_ref < 1e-6 (outside what the reference can generate in practice)
  order imbalance KL(c_ref(tau | y) || uniform order), averaged over y ~ p_ref (Monte Carlo over trajectories): 0 iff
                  every reveal order of every string is equally likely under the reference
  decoders        entropy and expected TFBind8 score under the random-order, left-to-right and low-confidence-
                  remasking (T 0.6) decoders
"""

import argparse
import json
from math import lgamma
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pyarrow.parquet as pq

from ..architecture import load as load_checkpoint
from ..config import LENGTH, MASK, load_config
from ..data import build_target, tfbind8_scores
from ..evaluation import build_contexts, joint_log_probs
from ..metrics.decoders import proposal_terminal_log_probs
from ..metrics.paths import log_conditionals
from ..rl.losses.trajectory_balance import log_p_trajectory
from ..rl.samplers import complete_uniform
from ..rl.train import ar_log_probs

POWERS = 4 ** np.arange(LENGTH - 1, -1, -1)


def support99(p: np.ndarray) -> int:
    return int(np.searchsorted(np.cumsum(np.sort(p)[::-1]), 0.99) + 1)


def entropy(log_p: np.ndarray) -> float:
    p = np.exp(log_p)
    ok = p > 0
    return float(-p[ok] @ log_p[ok])


def pretraining_law(pretrain_config: Path) -> np.ndarray:
    """The pretraining law (the pretraining config's own reward), log-probabilities over all strings."""
    return build_target(load_config(pretrain_config)).log_pi


def sample_ids(data_path: Path) -> np.ndarray:
    table = pq.read_table(data_path)
    column = next(c for c in table.column_names if c in ("sequence", "string", "x"))
    seqs = table.column(column).to_pylist()
    return np.array([sum("ACGT".index(c) * int(p) for c, p in zip(s, POWERS)) for s in seqs])


def order_imbalance(forward, params, log_p: np.ndarray, n: int = 20000, seed: int = 0) -> float:
    """E_{y ~ p_ref} KL(c_ref(. | y) || uniform) = E_tau[log P(tau) - log p(y)] + log L!  (Monte Carlo)."""
    empty = jnp.full((n, LENGTH), MASK, dtype=jnp.int32)
    ys, orders = complete_uniform(forward, params, empty, jax.random.key(seed))
    lp_tau = np.asarray(log_p_trajectory(forward, params, empty, ys, orders), dtype=np.float64)
    lp_y = log_p[np.asarray(ys) @ POWERS]
    return float(np.mean(lp_tau - lp_y) + lgamma(LENGTH + 1))


def diagnose(config_path: Path, pretrain_config: Path, references: list[Path], betas: tuple[float, ...]) -> dict:
    config = load_config(config_path)
    reward = build_target(config)
    scores = tfbind8_scores()
    contexts = build_contexts()
    out = {"reward_law": {"entropy": entropy(reward.log_pi), "support99": support99(reward.pi)}}
    log_potts = pretraining_law(pretrain_config)
    out["pretraining_law"] = {"entropy": entropy(log_potts), "support99": support99(np.exp(log_potts))}
    ids = sample_ids(load_config(pretrain_config).data_dir / f"data{LENGTH}.parquet")
    distinct, counts = np.unique(ids, return_counts=True)
    empirical = np.full(len(reward.pi), -np.inf)
    empirical[distinct] = np.log(counts / counts.sum())
    out["pretraining_sample"] = {"rows": int(len(ids)), "distinct": int(len(distinct)),
                                 "potts_mass_on_sample": float(np.exp(log_potts[distinct]).sum())}
    out["references"] = {}
    for path in references:
        arch, params = load_checkpoint(path)
        cond = log_conditionals(arch.forward, params, contexts)
        log_p = joint_log_probs(contexts, cond)
        p = np.exp(log_p)
        row = {"architecture": arch.spec.model_dump(), "parameters": arch.parameter_count(params),
               "entropy": entropy(log_p), "support99": support99(p),
               "mass_on_sample": float(p[distinct].sum()),
               "memorisation_index": float(p[distinct].sum() / np.exp(log_potts[distinct]).sum()),  # 1 = learned the law
               "kl_empirical_to_ref": float(np.exp(empirical[distinct]) @ (empirical[distinct] - log_p[distinct])),
               "kl_potts_to_ref": float(np.exp(log_potts) @ (log_potts - log_p)),
               "order_imbalance": order_imbalance(arch.forward, params, log_p)}
        decoders = {"random_order": log_p, "left_to_right": ar_log_probs(arch.forward, params, reward.tokens),
                    "low_confidence_T0.6": proposal_terminal_log_probs(contexts, cond, 0.6)}
        row["decoders"] = {name: {"entropy": entropy(lp), "expected_score": float(np.exp(lp) @ scores)}
                           for name, lp in decoders.items()}
        row["trafl_target"] = {}
        for beta in betas:
            log_w = log_p + beta * reward.log_reward
            log_star = log_w - np.logaddexp.reduce(log_w)
            star = np.exp(log_star)
            row["trafl_target"][str(beta)] = {
                "entropy": entropy(log_star), "support99": support99(star),
                "mass_where_ref_below_1e-6": float(star[p < 1e-6].sum()),
                "expected_score": float(star @ scores), "kl_to_ref": float(star @ (log_star - log_p))}
        out["references"][str(path.parent.name)] = row
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--pretrain-config", type=Path, default=Path("configs/potts.json"))
    parser.add_argument("--references", type=Path, nargs="+", required=True)
    parser.add_argument("--betas", type=float, nargs="+", default=[0.5, 1.5, 4.5])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = diagnose(args.config, args.pretrain_config, args.references, tuple(args.betas))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=1)[:4000])


if __name__ == "__main__":
    main()
