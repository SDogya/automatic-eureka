"""TraFL-type residual estimators from the bitseq thread (research/bitseq/REPORT.md §3), as options of the TraFL arm.

Each mask k gives a per-token log-ratio a_k = s_theta,k - s_ref,k (masks.masked_scores, the same mask for actor and
reference) and a residual delta_k = a_k - (beta / u) * centred reward + log Z(x). Estimators of the per-string loss:
  square       (mean_k delta_k)^2 over K masks = delta_bar^2 + Var_k / K in expectation (this repo's TraFL)
  split        U_{2+2} = mean(delta_1, delta_2) * mean(delta_3, delta_4): unbiased for delta_bar^2 (bitseq)
  pairwise     mean over the K(K-1)/2 pairs of delta_j * delta_k: unbiased, lower variance than split
  exact_masks  delta of the exact mask mean (all 2^u - 1 masks, count-uniform then subset-uniform = the K -> inf
               limit of the IID "paper" surrogate): no estimator at all
  exact_lik    delta with (log p_theta(y|x) - log p_ref(y|x)) / u, exact by the subset recursion (bitseq "exact
               completion")
"""

from math import comb
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np

from ...config import LENGTH, MASK
from .masks import Scheme, draw_masks, masked_scores
from .trafl import Forward, Reference

Estimator = Literal["square", "split", "pairwise", "exact_masks", "exact_lik"]

BITS = np.array([[(s >> i) & 1 for i in range(LENGTH)] for s in range(2**LENGTH)], dtype=bool)   # (2^L, L)
SIZE = BITS.sum(axis=1)
NEG = -1e30  # finite "minus infinity": logsumexp / where over unreachable subsets keep finite (non-NaN) gradients


def per_mask_log_ratio(forward: Forward, policy: object, reference: Reference, contexts: jax.Array,
                       completions: jax.Array, key: jax.Array, samples: int, scheme: Scheme = "iid") -> jax.Array:
    """a_k, shape (K, C, G); the reference term carries no gradient."""
    hidden = jnp.broadcast_to(contexts[:, None, :] == MASK, completions.shape)
    masks = draw_masks(key, hidden, samples, scheme)
    ref = jax.lax.stop_gradient(masked_scores(reference.forward, reference.params, completions, masks))
    return masked_scores(forward, policy, completions, masks) - ref


def exact_mask_mean_log_ratio(forward: Forward, policy: object, reference: Reference, contexts: jax.Array,
                              completions: jax.Array) -> jax.Array:
    """sum over non-empty S within U(x) of w(S) * a_S, w(S) = 1 / (u * C(u, |S|)), shape (C, G)."""
    hidden = jnp.broadcast_to(contexts[:, None, :] == MASK, completions.shape)
    count = hidden.sum(axis=-1)
    bank = jnp.asarray(BITS[1:])                                                    # (2^L - 1, L) non-empty subsets
    inside = ~jnp.any(bank[:, None, None, :] & ~hidden[None], axis=-1)              # (S, C, G): S within U(x)
    masks = bank[:, None, None, :] & inside[..., None]
    binom = jnp.asarray([[comb(u, k) for k in range(LENGTH + 1)] for u in range(LENGTH + 1)], dtype=jnp.float32)
    weight = jnp.where(inside, 1.0 / (jnp.maximum(count, 1) * binom[count, jnp.asarray(SIZE[1:])[:, None, None]]), 0.0)
    ref = jax.lax.stop_gradient(masked_scores(reference.forward, reference.params, completions, masks))
    return jnp.sum(weight * (masked_scores(forward, policy, completions, masks) - ref), axis=0)


def exact_log_p(forward: Forward, params: object, contexts: jax.Array, completions: jax.Array) -> jax.Array:
    """log p(y | x) of the random-order generator, differentiable, by the recursion over revealed subsets R of U(x):
    f(R) = logsumexp_{i in R} [f(R \\ i) + log q(y_i | x, y_{R \\ i}) - log(u - |R \\ i|)], log p = f(U). (C, G)."""
    hidden = jnp.broadcast_to(contexts[:, None, :] == MASK, completions.shape)
    count = hidden.sum(axis=-1)
    bits = jnp.asarray(BITS)
    states = jnp.where(bits[:, None, None, :] & hidden[None], completions[None],
                       jnp.broadcast_to(contexts[None, :, None, :], (2**LENGTH, *completions.shape)))
    log_q = jax.nn.log_softmax(forward(params, states), axis=-1)                   # (2^L, C, G, L, 4)
    chosen = jnp.take_along_axis(log_q, jnp.broadcast_to(completions[None], states.shape)[..., None], axis=-1)[..., 0]
    inside = ~jnp.any(bits[:, None, None, :] & ~hidden[None], axis=-1)              # (2^L, C, G)
    parent = np.array([[s & ~(1 << i) for i in range(LENGTH)] for s in range(2**LENGTH)])
    f = jnp.where(jnp.asarray(SIZE == 0)[:, None, None], 0.0, NEG) * jnp.ones(inside.shape)
    for k in range(1, LENGTH + 1):
        terms = (f[parent] + jnp.moveaxis(chosen[parent, :, :, np.arange(LENGTH)], 0, 0)
                 - jnp.log(jnp.maximum(count - (k - 1), 1))[None, None])             # (2^L, L, C, G)
        terms = jnp.where(jnp.asarray(BITS)[:, :, None, None], terms, NEG)
        new = jax.nn.logsumexp(terms, axis=1)
        f = jnp.where(jnp.asarray(SIZE == k)[:, None, None] & inside, new, f)
    full = jnp.sum(hidden * (2 ** jnp.arange(LENGTH)), axis=-1)                     # index of U(x)
    return jnp.take_along_axis(f, full[None], axis=0)[0]


def residual_loss(estimator: Estimator, forward: Forward, policy: object, reference: Reference,
                  contexts: jax.Array, completions: jax.Array, shift: jax.Array, key: jax.Array,
                  samples: int, scheme: Scheme = "iid") -> tuple[jax.Array, jax.Array]:
    """(loss, delta_bar) with delta = a - shift; shift = (beta / u) * centred reward - log Z(x), shape (C, G)."""
    if estimator == "exact_masks":
        delta = exact_mask_mean_log_ratio(forward, policy, reference, contexts, completions) - shift
        return jnp.mean(delta**2), delta
    if estimator == "exact_lik":
        u = jnp.maximum((contexts == MASK).sum(axis=-1), 1)[:, None]
        ref = jax.lax.stop_gradient(exact_log_p(reference.forward, reference.params, contexts, completions))
        delta = (exact_log_p(forward, policy, contexts, completions) - ref) / u - shift
        return jnp.mean(delta**2), delta
    if estimator in ("split", "pairwise") and scheme != "iid":
        raise ValueError("split / pairwise are unbiased only for independent masks; comp pairs are dependent")
    deltas = per_mask_log_ratio(forward, policy, reference, contexts, completions, key, samples, scheme) - shift[None]
    if estimator == "square":
        return jnp.mean(deltas.mean(axis=0) ** 2), deltas.mean(axis=0)
    if estimator == "split":
        if samples != 4:
            raise ValueError("split product uses exactly 4 masks")
        return jnp.mean(deltas[:2].mean(axis=0) * deltas[2:].mean(axis=0)), deltas.mean(axis=0)
    if estimator == "pairwise":
        total, squares = deltas.sum(axis=0), (deltas**2).sum(axis=0)
        return jnp.mean((total**2 - squares) / (samples * (samples - 1))), deltas.mean(axis=0)
    raise ValueError(estimator)
