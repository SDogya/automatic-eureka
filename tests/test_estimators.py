"""Gates for the bitseq-thread estimators (src/rl/losses/estimators.py)."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from src.config import MASK
from src.metrics.trajectories import Model, log_p_y
from src.model import forward, init_parameters
from src.rl.losses.estimators import exact_log_p, exact_mask_mean_log_ratio, per_mask_log_ratio, residual_loss
from src.rl.losses.trafl import Reference

POLICY = init_parameters(jax.random.key(41), 16)
REFERENCE = Reference(forward, init_parameters(jax.random.key(42), 16))
CONTEXTS = jnp.asarray([[0, MASK, 2, MASK, 1, MASK, 3, 0], [MASK] * 8, [MASK, 1, MASK, MASK, MASK, 2, MASK, 3]])
COMPLETIONS = jnp.asarray([[[0, 1, 2, 3, 1, 0, 3, 0], [0, 2, 2, 2, 1, 1, 3, 0]],
                           [[3, 1, 0, 2, 2, 0, 1, 3], [0, 0, 1, 1, 2, 2, 3, 3]],
                           [[2, 1, 0, 0, 3, 2, 1, 3], [1, 1, 1, 2, 0, 2, 0, 3]]])


def test_exact_log_p_matches_the_order_sum() -> None:
    """The differentiable subset recursion equals metrics.trajectories.log_p_y (sum over all u! orders), u = 3, 8, 5."""
    got = np.asarray(exact_log_p(forward, POLICY, CONTEXTS, COMPLETIONS))
    model = Model(forward, POLICY)
    want = np.array([[log_p_y(model, np.asarray(c), np.asarray(y)) for y in ys] for c, ys in zip(CONTEXTS, COMPLETIONS)])
    np.testing.assert_allclose(got, want, atol=2e-5)
    grads = jax.grad(lambda p: exact_log_p(forward, p, CONTEXTS, COMPLETIONS).sum())(POLICY)
    assert all(np.isfinite(np.asarray(g)).all() for g in jax.tree.leaves(grads))
    # gradient = gradient of logsumexp over all 3! orders of the exact trajectory law (first prompt, u = 3)
    from itertools import permutations
    from src.rl.losses.trajectory_balance import log_p_trajectory
    orders = jnp.asarray([list(o) + [-1] * 5 for o in permutations([1, 3, 5])])
    c, ys = CONTEXTS[0], COMPLETIONS[0]

    def by_orders(p):
        return sum(jax.nn.logsumexp(log_p_trajectory(forward, p, jnp.broadcast_to(c, (6, 8)), jnp.broadcast_to(y, (6, 8)),
                                                     orders)) for y in ys)

    want_g = jax.grad(by_orders)(POLICY)
    got_g = jax.grad(lambda p: exact_log_p(forward, p, CONTEXTS[:1], COMPLETIONS[:1]).sum())(POLICY)
    for a, b in zip(jax.tree.leaves(got_g), jax.tree.leaves(want_g)):
        np.testing.assert_allclose(a, b, atol=2e-5)


def test_exact_mask_mean_is_the_limit_of_the_iid_surrogate() -> None:
    exact = np.asarray(exact_mask_mean_log_ratio(forward, POLICY, REFERENCE, CONTEXTS, COMPLETIONS))
    draws = np.asarray(per_mask_log_ratio(forward, POLICY, REFERENCE, CONTEXTS, COMPLETIONS, jax.random.key(1), 40000))
    se = draws.std(axis=0) / np.sqrt(len(draws))
    assert (np.abs(draws.mean(axis=0) - exact) < 4 * se + 1e-5).all()
    same = exact_mask_mean_log_ratio(forward, REFERENCE.params, REFERENCE, CONTEXTS, COMPLETIONS)
    np.testing.assert_allclose(same, 0.0, atol=1e-6)


@pytest.mark.parametrize("estimator", ["split", "pairwise"])
def test_u_statistics_are_unbiased_and_square_is_not(estimator: str) -> None:
    """E[split] = E[pairwise] = delta_bar^2 per string; E[square] = delta_bar^2 + Var / 4 (exact delta_bar from the
    exact mask mean)."""
    shift = jnp.asarray([[0.3, -0.2], [0.1, 0.4], [-0.5, 0.0]])
    target = float(jnp.mean((exact_mask_mean_log_ratio(forward, POLICY, REFERENCE, CONTEXTS, COMPLETIONS) - shift) ** 2))

    def value(name: str, key: jax.Array) -> jax.Array:
        return residual_loss(name, forward, POLICY, REFERENCE, CONTEXTS, COMPLETIONS, shift, key, 4)[0]

    keys = jax.random.split(jax.random.key(2), 30000)
    unbiased = np.asarray(jax.vmap(lambda k: value(estimator, k))(keys))
    square = np.asarray(jax.vmap(lambda k: value("square", k))(keys))
    se = unbiased.std() / np.sqrt(len(keys))
    assert abs(unbiased.mean() - target) < 4 * se
    assert square.mean() - target > 8 * square.std() / np.sqrt(len(keys))
