"""How biased is TraFL's surrogate? Compare loss gradients on one fixed rollout batch.

Reference: the exact sequence-level trajectory-balance loss with log p(y | x) computed exactly;
its gradient uses the Fisher identity grad log p(y|x) = E_{tau ~ p(tau|x,y)} grad log p(tau|x),
evaluated by enumerating all reveal orders. log Z is set to its per-context optimum for every
estimator (residuals centered within the group), so only the policy gradients are compared.
"""

import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

from ..config import LENGTH, MASK, Config
from ..data import build_target
from ..model import complete, forward, load_parameters
from ..rl.losses.trafl import surrogate_log_prob
from ..rl.losses.trajectory_balance import log_p_trajectory
from .trajectories import Model, posterior_over_orders


def centered_square(delta: jax.Array) -> jax.Array:
    return jnp.mean((delta - delta.mean(axis=1, keepdims=True)) ** 2)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def gradient_bias(config: Config, checkpoint: Path, reference_checkpoint: Path, contexts_count: int = 32,
                  group: int = 4, draws: int = 200, mask_samples: int = 4, seed: int = 0) -> dict:
    """All residuals include the frozen reference p_ref (TraFL's step-0 model)."""
    began = time.time()
    policy = load_parameters(checkpoint)
    ref_params = load_parameters(reference_checkpoint)
    target = build_target(config)
    log_reward = jnp.asarray(target.log_reward, dtype=jnp.float32)
    powers = jnp.asarray(4 ** np.arange(LENGTH - 1, -1, -1, dtype=np.int32))
    key = jax.random.key(seed)
    base_key, mask_key, fill_key, draw_key = jax.random.split(key, 4)

    # One fixed rollout batch, built exactly as in TraFL training.
    base, _ = complete(policy, jnp.full((contexts_count, LENGTH), MASK, jnp.int32), base_key)
    hide = jax.random.bernoulli(mask_key, 0.5, base.shape)
    hide = jnp.where(hide.any(axis=1, keepdims=True), hide, True)
    contexts = jnp.where(hide, MASK, base)
    completions, orders = complete(policy, jnp.repeat(contexts, group, axis=0), fill_key)
    completions = completions.reshape(contexts_count, group, LENGTH)
    orders = orders.reshape(contexts_count, group, LENGTH)
    rewards = log_reward[completions @ powers]
    hidden = hide.sum(axis=1)

    # Exact posteriors over reveal orders for every completion (enumeration).
    model, ref_model = Model(forward, policy), Model(forward, ref_params)
    exact_ref = np.zeros((contexts_count, group))
    rows_context, rows_completion, rows_order, rows_weight, rows_owner = [], [], [], [], []
    exact_log_p, gaps = np.zeros((contexts_count, group)), np.zeros((contexts_count, group))
    contexts_np, completions_np = np.asarray(contexts), np.asarray(completions)
    for c in range(contexts_count):
        for g in range(group):
            post = posterior_over_orders(model, contexts_np[c], completions_np[c, g])
            exact_log_p[c, g], gaps[c, g] = post.log_p_y, post.elbo_gap
            exact_ref[c, g] = posterior_over_orders(ref_model, contexts_np[c], completions_np[c, g]).log_p_y
            padded = np.full((len(post.orders), LENGTH), -1, dtype=np.int32)
            padded[:, :post.orders.shape[1]] = post.orders
            rows_order.append(padded)
            rows_weight.append(np.exp(post.log_posterior))
            rows_owner.append(np.full(len(padded), c * group + g))
            rows_context.append(np.tile(contexts_np[c], (len(padded), 1)))
            rows_completion.append(np.tile(completions_np[c, g], (len(padded), 1)))
    rows = [jnp.asarray(np.concatenate(r)) for r in (rows_context, rows_completion, rows_order)]
    weights = jnp.asarray(np.concatenate(rows_weight), dtype=jnp.float32)
    owner = jnp.asarray(np.concatenate(rows_owner))
    exact = jnp.asarray(exact_log_p, dtype=jnp.float32)
    ref_exact = jnp.asarray(exact_ref, dtype=jnp.float32)
    repeated = jnp.broadcast_to(contexts[:, None, :], completions.shape)
    ref_trajectory = log_p_trajectory(forward, ref_params, repeated, completions, orders)
    print(f"[{checkpoint.parent.name}/{checkpoint.name}] rollouts and {len(weights)} exact "
          f"(completion, order) pairs ready ({time.time() - began:.1f}s)", flush=True)

    def exact_sequence(p):
        fisher = jax.ops.segment_sum(weights * log_p_trajectory(forward, p, *rows), owner,
                                     num_segments=contexts_count * group)
        log_p = exact.reshape(-1) + fisher - jax.lax.stop_gradient(fisher)
        return centered_square(log_p.reshape(contexts_count, group) - ref_exact - rewards)

    def exact_sequence_paper_scale(p):
        fisher = jax.ops.segment_sum(weights * log_p_trajectory(forward, p, *rows), owner,
                                     num_segments=contexts_count * group)
        log_p = (exact.reshape(-1) + fisher - jax.lax.stop_gradient(fisher)).reshape(contexts_count, group)
        return centered_square((log_p - ref_exact - rewards) / hidden[:, None])

    def trajectory_sampled(p):
        return centered_square(log_p_trajectory(forward, p, repeated, completions, orders)
                               - ref_trajectory - rewards)

    def surrogate(normalization):
        scale = 1.0 / hidden[:, None] if normalization == "paper" else 1.0

        def loss(p, k):
            s = surrogate_log_prob(forward, p, contexts, completions, k, mask_samples, normalization)
            s_ref = surrogate_log_prob(forward, ref_params, contexts, completions, k, mask_samples, normalization)
            return centered_square(s - s_ref - scale * rewards)

        return loss

    flat = lambda tree: np.asarray(ravel_pytree(tree)[0], dtype=np.float64)
    reference = flat(jax.grad(exact_sequence)(policy))
    results = {"checkpoint": str(checkpoint), "reference": str(reference_checkpoint),
               "contexts": contexts_count, "group": group,
               "draws": draws, "mask_samples": mask_samples,
               "mean_hidden": float(hidden.mean()),
               "elbo_gap_mean": float(gaps.mean()), "elbo_gap_max": float(gaps.max()),
               "estimators": {}}

    def record(name, expected, singles=None):
        entry = {"cosine_to_exact": cosine(expected, reference),
                 "norm_ratio": float(np.linalg.norm(expected) / np.linalg.norm(reference))}
        if singles is not None:
            entry["single_draw_cosine_mean"] = float(np.mean([cosine(s, reference) for s in singles]))
            entry["single_draw_cosine_to_own_mean"] = float(np.mean([cosine(s, expected) for s in singles]))
        results["estimators"][name] = entry
        print(f"    {name:32s} cos(E grad, exact)={entry['cosine_to_exact']:+.4f}  "
              f"|E grad|/|exact|={entry['norm_ratio']:.4f}"
              + (f"  single-draw cos={entry['single_draw_cosine_mean']:+.4f}" if singles is not None else ""),
              flush=True)

    record("exact sequence, paper 1/u scale", flat(jax.grad(exact_sequence_paper_scale)(policy)))
    record("exact trajectory (sampled tau)", flat(jax.grad(trajectory_sampled)(policy)))
    for normalization in ("paper", "elbo"):
        grad_fn = jax.jit(jax.grad(surrogate(normalization)))
        singles = [flat(grad_fn(policy, k)) for k in jax.random.split(draw_key, draws)]
        record(f"TraFL surrogate ({normalization}), K={mask_samples}", np.mean(singles, axis=0), singles)
    results["seconds"] = time.time() - began
    return results


def run_bias(config: Config, checkpoints: list[Path], output: Path) -> list[dict]:
    """Each checkpoint is compared with step_000.npz of its own run as the reference."""
    output.mkdir(parents=True, exist_ok=True)
    reports = []
    for checkpoint in checkpoints:
        print(f"== {checkpoint} (reference {checkpoint.parent / 'step_000.npz'})", flush=True)
        reports.append(gradient_bias(config, checkpoint, checkpoint.parent / "step_000.npz"))
        (output / "gradient_bias.json").write_text(json.dumps(reports, indent=2) + "\n")
    return reports
