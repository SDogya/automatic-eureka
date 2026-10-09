"""Mask draws and masked per-token scores shared by the ELBO-type losses (TraFL, ESPO).

hidden (..., L) marks the positions U(x) a completion fills, u = |U(x)|. A mask is a non-empty subset of U(x);
its score is the per-token average log q(y_i | y with the mask hidden) over the masked positions, the
"paper" normalisation of TraFL's surrogate (Eq. 4 of arXiv 2605.13935).

Schemes:
  iid   each draw: l ~ U{1..u}, then a uniform l-subset of U(x) (this repo's TraFL default);
  comp  draws come in antithetic pairs: l ~ U{1..u-1}, a uniform l-subset S and its complement U(x) \\ S
        (gfn_lab's "comp" scheme, standing in for the paper's complementary mask pairs); u = 1 gives S = U(x) twice.
"""

from typing import Literal

import jax
import jax.numpy as jnp

from ...config import MASK

Scheme = Literal["iid", "comp"]


def _subset(key: jax.Array, hidden: jax.Array, low: int, high: jax.Array) -> jax.Array:
    """A uniform subset of the hidden positions of size l ~ U{low..high} (high broadcast over rows)."""
    size_key, order_key = jax.random.split(key)
    l = jax.random.randint(size_key, high.shape, low, high + 1)
    scores = jnp.where(hidden, jax.random.uniform(order_key, hidden.shape), jnp.inf)
    rank = jnp.argsort(jnp.argsort(scores, axis=-1), axis=-1)
    return hidden & (rank < l[..., None])


def draw_masks(key: jax.Array, hidden: jax.Array, samples: int, scheme: Scheme) -> jax.Array:
    """Boolean masks, shape (samples, *hidden.shape); every mask is non-empty where u >= 1."""
    count = hidden.sum(axis=-1)
    if scheme == "iid":
        return jax.vmap(lambda k: _subset(k, hidden, 1, jnp.maximum(count, 1)))(jax.random.split(key, samples))
    if samples % 2:
        raise ValueError("comp masks come in pairs: samples must be even")
    first = jax.vmap(lambda k: _subset(k, hidden, 1, jnp.maximum(count - 1, 1)))(jax.random.split(key, samples // 2))
    second = jnp.where((count >= 2)[..., None], hidden & ~first, hidden)
    return jnp.concatenate([first, second])


def masked_scores(forward: object, params: object, completions: jax.Array, masks: jax.Array) -> jax.Array:
    """Per-token average log q of the masked letters for every mask, shape masks.shape[:-1]."""
    log_q = jax.nn.log_softmax(forward(params, jnp.where(masks, MASK, completions)), axis=-1)
    chosen = jnp.take_along_axis(log_q, jnp.broadcast_to(completions, masks.shape)[..., None], axis=-1)[..., 0]
    count = masks.sum(axis=-1)
    return jnp.where(count > 0, jnp.sum(jnp.where(masks, chosen, 0.0), axis=-1) / jnp.maximum(count, 1), 0.0)
