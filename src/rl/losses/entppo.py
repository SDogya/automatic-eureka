"""Entropic PPO (Zykova-Myzina et al., arXiv 2606.15793; torchgfn 2.4.1 `EntPPOGFlowNet`) for the random-order
masked generator, with RTB's soft reward so that the target is TraFL's p_ref(tau | x) * exp(beta r(y)).

torchgfn's soft MDP rewards a non-exit step with log P_B and the exit with log R. This port puts the frozen reference's
forward step log P_ref(a_t | s_t) where torchgfn has log P_B, and beta r(y) at the exit. That is the soft-RL dictionary
for RTB: the soft return sum_t [log P_ref - log P_theta](a_t | s_t) + beta r(y) is the negated trajectory KL to
P_ref * e^{beta r} / Z, up to log Z. The rest follows torchgfn:
  g_t = r_t - log pi_old(a_t | s_t)                       one-step soft return (rollout policy, frozen)
  A_t = GAE(lambda) of delta_t = g_t + V_old(s_{t+1}) - V_old(s_t), V(sink) = 0, V(y) learned (-> beta r(y) + ...)
  policy loss = -mean_traj sum_t [ PPOClip(rho_t, A_t) - KL(pi_theta(. | s_t) || pi_old(. | s_t)) ]
  value loss  = mean over valid steps of (V_theta(s_t) - (A_t + V_old(s_t)))^2
Steps: t = 0..u-1 reveal one hidden position each (uniform position, letter ~ q); t = u is the exit from the full
string (probability 1: ratio 1, KL 0). The uniform position term cancels in every ratio, return and KL.
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp

from ...config import LENGTH, MASK
from .trafl import Forward, Reference

ValueForward = Callable[[object, jax.Array], jax.Array]  # (params, states (N, L)) -> (N,)


def trajectory_states(contexts: jax.Array, completions: jax.Array, orders: jax.Array) -> jax.Array:
    """States s_0..s_L along each trajectory, shape (C, G, L + 1, L); s_t = the state before step t."""
    valid = orders >= 0
    safe = jnp.where(valid, orders, 0)
    revealed = jnp.cumsum(jax.nn.one_hot(safe, LENGTH, dtype=jnp.int32) * valid[..., None], axis=-2)   # after step t
    before = jnp.concatenate([jnp.zeros_like(revealed[..., :1, :]), revealed], axis=-2)              # (C,G,L+1,L)
    repeated = jnp.broadcast_to(contexts[:, None, None, :], before.shape)
    return jnp.where(before > 0, completions[..., None, :], repeated)


def _step_terms(forward: Forward, params: object, states: jax.Array, orders: jax.Array,
                completions: jax.Array) -> tuple[jax.Array, jax.Array]:
    """(log q(a_t | s_t) for t < L, full log q at every state): (C,G,L), (C,G,L+1,L,4)."""
    log_q = jax.nn.log_softmax(forward(params, states), axis=-1)
    safe = jnp.where(orders >= 0, orders, 0)
    at_pos = jnp.take_along_axis(log_q[..., :LENGTH, :, :], safe[..., None, None], axis=-2)[..., 0, :]   # (C,G,L,4)
    letters = jnp.take_along_axis(completions, safe, axis=-1)
    return jnp.take_along_axis(at_pos, letters[..., None], axis=-1)[..., 0], log_q


def entppo_loss(policy: object, value_params: object, forward: Forward, value_forward: ValueForward,
                reference: Reference, contexts: jax.Array, completions: jax.Array, orders: jax.Array,
                rewards: jax.Array, beta: float, *, old: tuple[object, object], gae_lambda: float = 0.7,
                eps_clip: float = 0.2, value_weight: float = 1.0) -> tuple[jax.Array, dict]:
    """Ent-PPO policy + value loss. old = (policy, value params) at rollout time (frozen)."""
    states = trajectory_states(contexts, completions, orders)                       # (C,G,L+1,L)
    steps = orders >= 0                                                             # (C,G,L) reveal steps
    u = steps.sum(axis=-1)                                                          # (C,G)
    t = jnp.arange(LENGTH + 1)
    exit_step = t[None, None, :] == u[..., None]                                    # (C,G,L+1)
    live = t[None, None, :] <= u[..., None]                                         # states s_0..s_u

    log_new, q_new = _step_terms(forward, policy, states, orders, completions)
    log_old, q_old = jax.lax.stop_gradient(_step_terms(forward, old[0], states, orders, completions))
    log_ref, _ = jax.lax.stop_gradient(_step_terms(reference.forward, reference.params, states, orders, completions))

    pad = jnp.zeros(log_old.shape[:-1] + (1,))
    reveal_return = jnp.where(steps, log_ref - log_old, 0.0)                       # g_t, t < u
    g = jnp.concatenate([reveal_return, pad], axis=-1) + jnp.where(exit_step, beta * rewards[..., None], 0.0)

    flat = states.reshape(-1, LENGTH)
    v_old = jax.lax.stop_gradient(value_forward(old[1], flat)).reshape(live.shape)
    v_old = jnp.where(live, v_old, 0.0)
    v_next = jnp.concatenate([v_old[..., 1:], jnp.zeros_like(v_old[..., :1])], axis=-1)
    delta = jnp.where(live, g + v_next - v_old, 0.0)

    def back(carry: jax.Array, d: jax.Array) -> tuple[jax.Array, jax.Array]:
        carry = d + gae_lambda * carry
        return carry, carry

    _, adv_rev = jax.lax.scan(back, jnp.zeros(delta.shape[:-1]), jnp.moveaxis(delta, -1, 0)[::-1])
    advantages = jax.lax.stop_gradient(jnp.moveaxis(adv_rev[::-1], 0, -1))       # (C,G,L+1)
    targets = jax.lax.stop_gradient(advantages + v_old)

    ratio = jnp.exp(jnp.clip(log_new - log_old, -80.0, 80.0))                      # (C,G,L)
    a_steps = advantages[..., :LENGTH]
    clipped = jnp.minimum(ratio * a_steps, jnp.clip(ratio, 1 - eps_clip, 1 + eps_clip) * a_steps)
    hidden = states[..., :LENGTH, :] == MASK                                        # (C,G,L,L)
    p_new = jnp.exp(q_new[..., :LENGTH, :, :])
    kl_pos = jnp.sum(p_new * (q_new[..., :LENGTH, :, :] - q_old[..., :LENGTH, :, :]), axis=-1)  # (C,G,L,L)
    kl = jnp.sum(jnp.where(hidden, kl_pos, 0.0), axis=-1) / jnp.maximum(hidden.sum(axis=-1), 1)
    surrogate = jnp.where(steps, clipped - kl, 0.0).sum(axis=-1) + jnp.where(exit_step, advantages, 0.0).sum(axis=-1)
    policy_loss = -jnp.mean(surrogate)

    v_new = value_forward(value_params, flat).reshape(live.shape)
    value_loss = jnp.sum(jnp.where(live, (v_new - targets) ** 2, 0.0)) / jnp.maximum(live.sum(), 1)
    return policy_loss + value_weight * value_loss, {"policy_loss": policy_loss, "value_loss": value_loss,
                                                     "advantage_mean": jnp.mean(advantages[..., 0])}
