"""Gates for relative detailed balance / sub-trajectory balance (src/rl/losses/flow_balance.py)."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from src.config import MASK
from src.model import forward, init_parameters
from src.rl.losses.flow_balance import db_loss, step_residuals, subtb_loss
from src.rl.losses.trafl import Reference
from src.rl.losses.trajectory_balance import trajectory_residuals
from src.rl.train import init_log_z, log_z_forward

POLICY = init_parameters(jax.random.key(61), 16)
REFERENCE = Reference(forward, init_parameters(jax.random.key(62), 16))
VALUE = init_log_z(jax.random.key(63), 16)
CONTEXTS = jnp.asarray([[0, MASK, 2, MASK, 1, MASK, 3, 0], [MASK] * 8])
COMPLETIONS = jnp.asarray([[[0, 1, 2, 3, 1, 0, 3, 0], [0, 2, 2, 2, 1, 1, 3, 0]],
                           [[3, 1, 0, 2, 2, 0, 1, 3], [0, 0, 1, 1, 2, 2, 3, 3]]])
ORDERS = jnp.asarray([[[5, 1, 3, -1, -1, -1, -1, -1], [3, 5, 1, -1, -1, -1, -1, -1]],
                      [[7, 0, 3, 1, 6, 2, 5, 4], [0, 1, 2, 3, 4, 5, 6, 7]]])
REWARDS = jnp.asarray([[0.4, -1.2], [2.0, 0.3]])
BETA = 1.5
ARGS = (POLICY, VALUE, forward, log_z_forward, REFERENCE, CONTEXTS, COMPLETIONS, ORDERS, REWARDS, BETA)


def test_db_residuals_telescope_to_the_rtb_residual() -> None:
    """Sum_t [V(s_t) + d_t - V(s_{t+1})] = V(x) + log P_theta(tau) - log P_ref(tau) - beta r: RTB's residual with
    log Z(x) = V(x), computed independently by trajectory_balance.trajectory_residuals."""
    d, values, steps = step_residuals(*ARGS)
    one_step = jnp.where(steps, values[..., :-1] + d - values[..., 1:], 0.0).sum(-1)
    rtb = trajectory_residuals((POLICY, VALUE), forward, log_z_forward, CONTEXTS, COMPLETIONS, ORDERS,
                               jnp.zeros_like(REWARDS), 1.0, reference=REFERENCE) - BETA * REWARDS
    np.testing.assert_allclose(one_step, rtb, atol=2e-5)


def test_subtb_limits() -> None:
    """lam -> 0: only one-step spans count, so SubTB = the per-trajectory mean of DB's squared residuals. lam -> large:
    the full span dominates, so SubTB = the mean squared RTB residual (u = 3 and u = 8 separately)."""
    d, values, steps = step_residuals(*ARGS)
    one = jnp.where(steps, (values[..., :-1] + d - values[..., 1:]) ** 2, 0.0)
    per_trajectory_db = jnp.mean(one.sum(-1) / steps.sum(-1))               # each trajectory weighted equally
    np.testing.assert_allclose(subtb_loss(*ARGS, lam=1e-6), per_trajectory_db, rtol=1e-4)
    full = (values[..., 0] + d.sum(-1) - jnp.take_along_axis(values, steps.sum(-1)[..., None], axis=-1)[..., 0])
    for row, u in ((0, 3), (1, 8)):
        args = (POLICY, VALUE, forward, log_z_forward, REFERENCE, CONTEXTS[row:row + 1], COMPLETIONS[row:row + 1],
                ORDERS[row:row + 1], REWARDS[row:row + 1], BETA)
        np.testing.assert_allclose(subtb_loss(*args, lam=1e6), jnp.mean(full[row] ** 2), rtol=1e-3)


def test_gradients_reach_policy_and_value() -> None:
    g_pol, g_val = jax.grad(lambda p, v: subtb_loss(p, v, *ARGS[2:]), argnums=(0, 1))(POLICY, VALUE)
    for g in (g_pol, g_val):
        assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(g))
        assert max(float(jnp.abs(x).max()) for x in jax.tree.leaves(g)) > 1e-4


def test_subtb_ignores_trajectories_without_steps() -> None:
    contexts = jnp.concatenate([CONTEXTS, jnp.asarray([[0, 1, 2, 3, 0, 1, 2, 3]])])
    completions = jnp.concatenate([COMPLETIONS, jnp.asarray([[[0, 1, 2, 3, 0, 1, 2, 3]] * 2])])
    orders = jnp.concatenate([ORDERS, jnp.full((1, 2, 8), -1)])
    rewards = jnp.concatenate([REWARDS, jnp.zeros((1, 2))])
    with_empty = subtb_loss(POLICY, VALUE, forward, log_z_forward, REFERENCE, contexts, completions, orders, rewards, BETA)
    np.testing.assert_allclose(with_empty, subtb_loss(*ARGS), rtol=1e-5)
