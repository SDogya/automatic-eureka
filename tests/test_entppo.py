"""Gates for Entropic PPO (src/rl/losses/entppo.py), exact over all trajectories of a small prompt."""

from itertools import permutations, product

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from src.config import MASK
from src.model import forward, init_parameters
from src.rl.losses.entppo import entppo_loss, trajectory_states
from src.rl.losses.trafl import Reference
from src.rl.losses.trajectory_balance import log_p_trajectory
from src.rl.train import init_log_z, log_z_forward

POLICY = init_parameters(jax.random.key(51), 16)
REFERENCE = Reference(forward, init_parameters(jax.random.key(52), 16))
VALUE = init_log_z(jax.random.key(53), 16)
PROMPT = np.array([0, MASK, 2, MASK, 1, MASK, 3, 0])
HIDDEN = [1, 3, 5]
BETA = 0.8


def all_trajectories() -> tuple[jax.Array, jax.Array, jax.Array]:
    ys, os_ = [], []
    for letters in product(range(4), repeat=3):
        y = PROMPT.copy()
        y[HIDDEN] = letters
        for order in permutations(HIDDEN):
            ys.append(y)
            os_.append(list(order) + [-1] * 5)
    ys, os_ = jnp.asarray(np.array(ys)), jnp.asarray(np.array(os_, dtype=np.int32))
    rewards = jnp.asarray(np.random.default_rng(5).normal(size=len(ys)), dtype=jnp.float32)
    return ys, os_, rewards


def test_expected_policy_gradient_is_trajectory_kl_gradient() -> None:
    """At theta = theta_old and lambda = 1, E_{tau ~ P_theta}[grad policy loss] = grad KL(P_theta || P_ref e^{beta r} / Z),
    whatever the value function (the baseline cancels exactly in the sum)."""
    ys, os_, rewards = all_trajectories()
    contexts = jnp.broadcast_to(jnp.asarray(PROMPT), ys.shape)
    weights = jnp.exp(log_p_trajectory(forward, POLICY, contexts, ys, os_))
    assert float(weights.sum()) == pytest.approx(1.0, abs=1e-5)

    def one(y, o, r):
        def loss(p):
            return entppo_loss(p, VALUE, forward, log_z_forward, REFERENCE, jnp.asarray(PROMPT)[None], y[None, None],
                               o[None, None], r[None, None], BETA, old=(POLICY, VALUE), gae_lambda=1.0,
                               value_weight=0.0)[0]
        return jax.grad(loss)(POLICY)

    per = jax.vmap(one)(ys, os_, rewards)
    expected = jax.tree.map(lambda g: jnp.tensordot(weights, g, axes=1), per)

    def kl(p):
        lp = log_p_trajectory(forward, p, contexts, ys, os_)
        lr = log_p_trajectory(forward, REFERENCE.params, contexts, ys, os_) + BETA * rewards
        return jnp.sum(jnp.exp(lp) * (lp - lr))

    want = jax.grad(kl)(POLICY)
    for a, b in zip(jax.tree.leaves(expected), jax.tree.leaves(want)):
        np.testing.assert_allclose(a, b, atol=3e-5)
    assert max(float(jnp.abs(g).max()) for g in jax.tree.leaves(want)) > 1e-3


def test_value_targets_are_soft_reward_to_go_at_lambda_one() -> None:
    """With lambda = 1 the value loss pulls V(s_t) toward the Monte Carlo soft return-to-go sum_{k >= t} g_k."""
    ys, os_, rewards = all_trajectories()
    y, o, r = ys[100], os_[100], rewards[100]
    contexts = jnp.asarray(PROMPT)[None]
    states = trajectory_states(contexts, y[None, None], o[None, None])[0, 0]          # (L+1, L)
    log_q = jax.nn.log_softmax(forward(POLICY, states), -1)
    log_r = jax.nn.log_softmax(forward(REFERENCE.params, states), -1)
    g = [float(log_r[t, o[t], y[o[t]]] - log_q[t, o[t], y[o[t]]]) for t in range(3)] + [BETA * float(r)]
    rtg = np.cumsum(g[::-1])[::-1]
    v_target = jnp.asarray(rtg, dtype=jnp.float32)
    # a value function equal to the targets has zero value loss
    zero_value = lambda _, s: jnp.zeros(len(s))  # noqa: E731
    _, stats = entppo_loss(POLICY, None, forward, zero_value, REFERENCE, contexts, y[None, None], o[None, None],
                           r[None, None], BETA, old=(POLICY, None), gae_lambda=1.0)
    want = float(jnp.mean(v_target**2))                                               # V = 0 so loss = mean target^2
    assert float(stats["value_loss"]) == pytest.approx(want, rel=1e-5)
