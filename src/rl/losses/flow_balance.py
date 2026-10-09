"""Relative detailed balance and sub-trajectory balance: local-credit versions of RTB (the `tb` arm).

RTB's target P_theta(tau) proportional to P_ref(tau) e^{beta r(y)} has a state flow, the soft value
V(s) = log sum_{completions} P_ref(rest | s) e^{beta r(y)}, with V(y) = beta r(y) at a complete string and V(x) = log Z(x)
at the prompt. At the target every step satisfies
    V(s_t) + log P_theta(a_t | s_t) - log P_ref(a_t | s_t) - V(s_{t+1}) = 0
(the uniform position choice 1/u is common to both policies and cancels). V is the learned value network (the log Z
head evaluated on every state), except V(y) := beta r(y), fixed.
  db      mean over all steps of the squared one-step residual (relative detailed balance)
  subtb   per trajectory, the lambda^{j-i}-weighted mean over its sub-trajectories s_i -> s_j of the squared summed
          residual (relative SubTB, Madan et al. 2023; weights normalised within each trajectory as torchgfn's
          "geometric_within"); then the mean over trajectories. Its full-span term is RTB's residual, log Z = V(x)
"""

import jax
import jax.numpy as jnp

from ...config import LENGTH
from .entppo import ValueForward, _step_terms, trajectory_states
from .trafl import Forward, Reference


def step_residuals(policy: object, value_params: object, forward: Forward, value_forward: ValueForward,
                   reference: Reference, contexts: jax.Array, completions: jax.Array, orders: jax.Array,
                   rewards: jax.Array, beta: float) -> tuple[jax.Array, jax.Array, jax.Array]:
    """(per-step log-ratio d_t = log P_theta - log P_ref (C,G,L), values V(s_0..s_L) with V(s_u) = beta r (C,G,L+1),
    step mask (C,G,L)). Steps t >= u are padding."""
    states = trajectory_states(contexts, completions, orders)
    steps = orders >= 0
    u = steps.sum(axis=-1)
    log_new, _ = _step_terms(forward, policy, states, orders, completions)
    log_ref, _ = jax.lax.stop_gradient(_step_terms(reference.forward, reference.params, states, orders, completions))
    d = jnp.where(steps, log_new - log_ref, 0.0)
    values = value_forward(value_params, states.reshape(-1, LENGTH)).reshape(states.shape[:-1])
    terminal = jnp.arange(LENGTH + 1)[None, None, :] == u[..., None]
    values = jnp.where(terminal, beta * rewards[..., None], values)
    return d, values, steps


def db_loss(*args: object, **kwargs: object) -> jax.Array:
    """Mean squared one-step residual V(s_t) + d_t - V(s_{t+1}) over the steps of every trajectory."""
    d, values, steps = step_residuals(*args, **kwargs)
    res = values[..., :-1] + d - values[..., 1:]
    return jnp.sum(jnp.where(steps, res**2, 0.0)) / jnp.maximum(steps.sum(), 1)


def subtb_loss(*args: object, lam: float = 0.9, **kwargs: object) -> jax.Array:
    """Weighted mean over sub-trajectories i < j <= u of (V(s_i) + sum_{t=i}^{j-1} d_t - V(s_j))^2, weight lam^{j-i}."""
    d, values, steps = step_residuals(*args, **kwargs)
    cum = jnp.concatenate([jnp.zeros(d.shape[:-1] + (1,)), jnp.cumsum(d, axis=-1)], axis=-1)   # (C,G,L+1)
    res = values[..., :, None] + (cum[..., None, :] - cum[..., :, None]) - values[..., None, :]   # [i, j]
    i, j = jnp.arange(LENGTH + 1)[:, None], jnp.arange(LENGTH + 1)[None, :]
    u = steps.sum(axis=-1)[..., None, None]
    valid = (i < j) & (j <= u)
    # lam^(j-i) normalised within each trajectory (torchgfn's "geometric_within"), in log space: no overflow for any lam
    log_w = jnp.where(valid, (j - i).astype(jnp.float32) * jnp.log(lam), -jnp.inf)
    weight = jnp.exp(log_w - jnp.max(log_w, axis=(-2, -1), keepdims=True))
    per_trajectory = jnp.sum(weight * res**2, axis=(-2, -1)) / jnp.maximum(jnp.sum(weight, axis=(-2, -1)), 1e-30)
    return jnp.mean(per_trajectory)
