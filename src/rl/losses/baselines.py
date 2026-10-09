"""Post-training baselines TraFL is compared with, ported from gfn_lab (gfnlab/objectives.py, torch) to JAX.

All losses are model-agnostic (forward(params, tokens (..., L)) -> logits (..., L, 4)). Shapes: contexts (C, L);
completions and orders (C, G, L); rewards (C, G) = ln R. Completions are fixed data: no gradient flows through
which completions were sampled. A = group-standardised advantage (r - mean_G) / (std_G + eps), detached.

  espo       ESPO-style single on-policy update (Ou et al., arXiv 2512.03759): -A * rho + kappa * k2 with
             rho = exp(s_theta - s_old), s = per-token masked score averaged over masks (old = theta: rho = 1, its
             gradient is A * grad s_theta), k2 = 0.5 * (u * (s_theta - s_ref))^2 on the same masks.
  espo_ppo   the same with PPO clipping against a frozen old policy, reused for several updates per batch.
  grpo       GRPO-style policy gradient on the exact log-probability of the sampled uniform-order trajectory.
  justgrpo   JustGRPO (Ni et al., arXiv 2601.15165): the denoiser read as a left-to-right policy over the hidden
             positions, pi_AR(y_k | y_<k, x) = softmax of the logits at position k with every later hidden position
             masked; token-level PPO clip, 1/|o| per sequence, KL coefficient 0 (their setting).
TraFL's explicit across-order variance penalty (gfn_lab's fix) is `order_variance`.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from ...config import MASK
from .masks import Scheme, draw_masks, masked_scores
from .trafl import Forward, Reference
from .trajectory_balance import log_p_trajectory


def group_advantage(rewards: jax.Array, eps: float = 1e-4) -> jax.Array:
    """(r - mean) / (std + eps) within each context's group (unbiased std, as torch's default)."""
    centered = rewards - rewards.mean(axis=1, keepdims=True)
    return jax.lax.stop_gradient(centered / (jnp.std(rewards, axis=1, keepdims=True, ddof=1) + eps))


def _hidden(contexts: jax.Array, completions: jax.Array) -> jax.Array:
    return jnp.broadcast_to(contexts[:, None, :] == MASK, completions.shape)


def _mean_valid(per: jax.Array, count: jax.Array) -> jax.Array:
    valid = count > 0
    return jnp.sum(jnp.where(valid, per, 0.0)) / jnp.maximum(valid.sum(), 1)


def espo_loss(policy: object, forward: Forward, contexts: jax.Array, completions: jax.Array, rewards: jax.Array,
              key: jax.Array, *, kappa: float, reference: Reference, old: object | None = None, samples: int = 4,
              scheme: Scheme = "comp", eps_clip: float = 0.2) -> jax.Array:
    """ESPO (old=None: single on-policy update, rho = 1; old given: PPO-clipped reuse). Masks shared by theta,
    theta_old and the reference."""
    hidden = _hidden(contexts, completions)
    count = hidden.sum(axis=-1)
    masks = draw_masks(key, hidden, samples, scheme)
    s = masked_scores(forward, policy, completions, masks).mean(axis=0)
    s_ref = jax.lax.stop_gradient(masked_scores(reference.forward, reference.params, completions, masks).mean(axis=0))
    s_old = jax.lax.stop_gradient(s if old is None else masked_scores(forward, old, completions, masks).mean(axis=0))
    advantage = group_advantage(rewards)
    rho = jnp.exp(s - s_old)
    surrogate = jnp.minimum(rho * advantage, jnp.clip(rho, 1 - eps_clip, 1 + eps_clip) * advantage)
    k2 = 0.5 * (count * (s - s_ref)) ** 2
    return _mean_valid(-surrogate + kappa * k2, count)


def grpo_loss(policy: object, forward: Forward, contexts: jax.Array, completions: jax.Array, orders: jax.Array,
              rewards: jax.Array) -> jax.Array:
    """-A * log p_theta(tau | x) of the sampled trajectory (one fresh update: PPO ratio 1, no KL, no reference)."""
    repeated = jnp.broadcast_to(contexts[:, None, :], completions.shape)
    log_p = log_p_trajectory(forward, policy, repeated, completions, orders)
    return _mean_valid(-group_advantage(rewards) * log_p, _hidden(contexts, completions).sum(axis=-1))


def ar_orders(contexts: jax.Array) -> jax.Array:
    """Hidden positions in ascending order, then -1: the left-to-right reveal order, shape like contexts."""
    length = contexts.shape[-1]
    hidden = contexts == MASK
    order = jnp.argsort(jnp.where(hidden, jnp.arange(length), length + jnp.arange(length)), axis=-1)
    return jnp.where(jnp.arange(length) < hidden.sum(axis=-1, keepdims=True), order, -1).astype(jnp.int32)


class TokenLogProbs(NamedTuple):
    values: jax.Array  # (..., L): log pi_AR of the letter revealed at each step, 0 after the last step
    valid: jax.Array   # (..., L): step < u


def ar_token_log_probs(forward: Forward, params: object, contexts: jax.Array, completions: jax.Array) -> TokenLogProbs:
    """Per-step log pi_AR(y_k | y_<k, x) along the left-to-right order (contexts broadcast to completions)."""
    contexts = jnp.broadcast_to(contexts[..., None, :] if contexts.ndim < completions.ndim else contexts,
                                completions.shape)
    orders = ar_orders(contexts)
    length = completions.shape[-1]
    valid = orders >= 0
    safe = jnp.where(valid, orders, 0)
    revealed_now = jax.nn.one_hot(safe, length, dtype=jnp.int32) * valid[..., None]
    revealed_before = jnp.cumsum(revealed_now, axis=-2) - revealed_now
    states = jnp.where(revealed_before > 0, completions[..., None, :], contexts[..., None, :])
    log_q = jax.nn.log_softmax(forward(params, states), axis=-1)
    at_position = jnp.take_along_axis(log_q, safe[..., None, None], axis=-2)[..., 0, :]
    chosen = jnp.take_along_axis(at_position, jnp.take_along_axis(completions, safe, axis=-1)[..., None], axis=-1)[..., 0]
    return TokenLogProbs(jnp.where(valid, chosen, 0.0), valid)


def justgrpo_loss(policy: object, forward: Forward, contexts: jax.Array, completions: jax.Array, rewards: jax.Array,
                  *, old: object | None = None, eps_clip: float = 0.2) -> jax.Array:
    """JustGRPO objective (negated): mean over sequences of (1/u) sum_k min(rho_k A, clip(rho_k) A)."""
    current = ar_token_log_probs(forward, policy, contexts, completions)
    previous = current.values if old is None else ar_token_log_probs(forward, old, contexts, completions).values
    rho = jnp.exp(current.values - jax.lax.stop_gradient(previous))
    advantage = group_advantage(rewards)[..., None]
    tokens = jnp.minimum(rho * advantage, jnp.clip(rho, 1 - eps_clip, 1 + eps_clip) * advantage)
    count = current.valid.sum(axis=-1)
    per = jnp.sum(jnp.where(current.valid, tokens, 0.0), axis=-1) / jnp.maximum(count, 1)
    return _mean_valid(-per, count)


def complete_ar(forward: Forward, params: object, tokens: jax.Array, key: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Fill every MASK of a batch (B, L) left to right, sampling each letter from pi_AR at temperature 1.
    Returns the completed tokens and the reveal order (as model.complete)."""
    orders = ar_orders(tokens)

    def fill(step: int, carry: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array]:
        state, state_key = carry
        state_key, symbol_key = jax.random.split(state_key)
        position = orders[:, step]
        rows = jnp.arange(state.shape[0])
        safe = jnp.maximum(position, 0)
        symbol = jax.random.categorical(symbol_key, forward(params, state)[rows, safe]).astype(state.dtype)
        state = state.at[rows, safe].set(jnp.where(position >= 0, symbol, state[rows, safe]))
        return state, state_key

    state, _ = jax.lax.fori_loop(0, tokens.shape[-1], fill, (tokens, key))
    return state, orders


def random_orders(key: jax.Array, hidden: jax.Array, draws: int) -> jax.Array:
    """`draws` independent uniform reveal orders of the hidden positions, shape (draws, *hidden.shape), -1 padded."""
    length = hidden.shape[-1]

    def one(k: jax.Array) -> jax.Array:
        order = jnp.argsort(jnp.where(hidden, jax.random.uniform(k, hidden.shape), 2.0), axis=-1)
        return jnp.where(jnp.arange(length) < hidden.sum(axis=-1, keepdims=True), order, -1).astype(jnp.int32)

    return jax.vmap(one)(jax.random.split(key, draws))


def order_variance(policy: object, forward: Forward, reference: Reference, contexts: jax.Array,
                   completions: jax.Array, key: jax.Array) -> jax.Array:
    """0.5 (X_s1 - X_s2)^2 for two independent uniform orders, X_s = (log p_theta - log p_ref)(tau_s) / u: an
    unbiased estimate of Var_s(X_s), the across-order variance a per-string score leaves unpenalised. Shape (C, G)."""
    hidden = _hidden(contexts, completions)
    count = hidden.sum(axis=-1)
    orders = random_orders(key, hidden, 2)
    repeated = jnp.broadcast_to(contexts[:, None, :], completions.shape)
    x = jnp.stack([(log_p_trajectory(forward, policy, repeated, completions, o)
                    - jax.lax.stop_gradient(log_p_trajectory(reference.forward, reference.params, repeated,
                                                             completions, o))) / jnp.maximum(count, 1)
                   for o in orders])
    return 0.5 * (x[0] - x[1]) ** 2
