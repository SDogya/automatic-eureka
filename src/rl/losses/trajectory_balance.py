"""Exact trajectory-balance loss for the random-order masked generator.

Unlike TraFL's sequence-level surrogate, the residual uses the exact log-probability of
the sampled denoising trajectory tau = (x = z_u, ..., z_0 = y):
    log p(tau | x) = sum_t [log q(a_t | z_t, i_t) - log u_t],  u_t = hidden count of z_t,
    delta = log p_theta(tau | x) - log p_ref(tau | x) - beta * centered_reward + log Z(x).
No assumption on p(tau | x, y) is needed.
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp

from ...config import MASK
from .trafl import Forward, LogZ, Reference


def log_p_trajectory(forward: Forward, params: object, contexts: jax.Array,
                     completions: jax.Array, orders: jax.Array) -> jax.Array:
    """Exact log p(tau | x) for trajectories given as (context, completion, reveal order).

    Shapes (..., L); orders hold the position revealed at each step and -1 after the last.
    """
    length = completions.shape[-1]
    valid = orders >= 0
    safe = jnp.where(valid, orders, 0)
    revealed_now = jax.nn.one_hot(safe, length, dtype=jnp.int32) * valid[..., None]  # (..., T, L)
    revealed_before = jnp.cumsum(revealed_now, axis=-2) - revealed_now
    states = jnp.where(revealed_before > 0, completions[..., None, :], contexts[..., None, :])
    log_q = jax.nn.log_softmax(forward(params, states), axis=-1)  # (..., T, L, 4)
    at_position = jnp.take_along_axis(log_q, safe[..., None, None], axis=-2)[..., 0, :]  # (..., T, 4)
    letters = jnp.take_along_axis(completions, safe, axis=-1)
    chosen = jnp.take_along_axis(at_position, letters[..., None], axis=-1)[..., 0]
    hidden = jnp.sum(contexts == MASK, axis=-1, keepdims=True)
    remaining = jnp.maximum(hidden - jnp.arange(length), 1)
    return jnp.sum(jnp.where(valid, chosen - jnp.log(remaining), 0.0), axis=-1)


def trajectory_residuals(params: tuple[object, object], forward: Forward, log_z: LogZ,
                         contexts: jax.Array, completions: jax.Array, orders: jax.Array,
                         rewards: jax.Array, beta: float | jax.Array,
                         reference: Reference | None = None) -> jax.Array:
    """Exact trajectory-balance residuals, shape (C, G).

    contexts (C, L); completions and orders (C, G, L); rewards (C, G). reference=None is the
    uniform reference p_ref(tau | x) = 1 / (u! 4^u), constant per context, absorbed by log Z.
    """
    policy, z_params = params
    repeated = jnp.broadcast_to(contexts[:, None, :], completions.shape)
    log_p = log_p_trajectory(forward, policy, repeated, completions, orders)
    log_ref = 0.0 if reference is None else log_p_trajectory(
        reference.forward, reference.params, repeated, completions, orders)
    centered = rewards - rewards.mean(axis=1, keepdims=True)
    beta = jnp.asarray(beta)
    beta = beta[:, None] if beta.ndim == 1 else beta
    return log_p - log_ref - beta * centered + log_z(z_params, contexts)[:, None]


def trajectory_balance_loss(*args: object, **kwargs: object) -> jax.Array:
    """Mean squared residual; same arguments as trajectory_residuals."""
    return jnp.mean(trajectory_residuals(*args, **kwargs) ** 2)
