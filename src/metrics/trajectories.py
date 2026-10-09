"""Exact trajectory probabilities of the random-order masked generator.

A prompt x is a partial string with hidden positions U(x), u = |U(x)|. A trajectory
tau = (z_u, ..., z_0) starts at z_u = x and reveals one uniformly chosen hidden
position per step; it ends at the completion y = z_0. Given y, a trajectory is fixed
by its reveal order sigma (a permutation of U(x)), so u! trajectories reach each y:

    p(tau | x)    = prod_t (1 / t) q(a_t | z_t, i_t) = (1 / u!) prod_t q(a_t | z_t, i_t)
    p(y | x)      = sum_sigma p(tau_sigma(y) | x)
    p(tau | x, y) = p(tau | x) / p(y | x)  for tau ending at y.
"""

from collections.abc import Callable
from itertools import permutations
from math import lgamma
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from ..config import ALPHABET, MASK, FloatArray, IntArray
from ..data import enumerate_tokens
from ..evaluation import build_contexts, joint_log_probs

Forward = Callable[[object, jax.Array], jax.Array]  # (params, tokens (..., L)) -> logits (..., L, 4)


class Model(NamedTuple):
    forward: Forward
    params: object

    def log_conditionals(self, tokens: IntArray, batch_size: int = 65536) -> FloatArray:
        """log q(a | z, i) for every row z and position i, shape (N, L, 4), float64."""
        parts = [np.asarray(jax.nn.log_softmax(self.forward(self.params, jnp.asarray(tokens[s:s + batch_size])),
                                               axis=-1), dtype=np.float64)
                 for s in range(0, len(tokens), batch_size)]
        return np.concatenate(parts) if parts else np.zeros((0, tokens.shape[1], len(ALPHABET)))


class Posterior(NamedTuple):
    orders: IntArray  # (u!, u): reveal orders as positions of the string
    log_p_tau: FloatArray  # log p(tau_sigma | x)
    log_p_y: float  # log p(y | x)
    log_posterior: FloatArray  # log p(tau_sigma | x, y)
    entropy: float  # H(tau | x, y), at most log u!
    elbo: float  # E_{sigma uniform} log p(tau_sigma | x) + log u!
    elbo_gap: float  # log p(y | x) - elbo = KL(uniform orders || posterior) >= 0


def hidden_positions(prompt: IntArray) -> IntArray:
    return np.flatnonzero(np.asarray(prompt) == MASK).astype(np.int32)


def check_completion(prompt: IntArray, y: IntArray) -> None:
    revealed = np.asarray(prompt) != MASK
    if np.any(np.asarray(y) == MASK) or np.any(np.asarray(y)[revealed] != np.asarray(prompt)[revealed]):
        raise ValueError("y must be a full string that agrees with the prompt")


def log_p_tau(model: Model, prompt: IntArray, y: IntArray, orders: IntArray) -> FloatArray:
    """log p(tau | x) for trajectories to y given as reveal orders, shape (S,) for orders (S, u)."""
    prompt, y, orders = np.asarray(prompt), np.asarray(y), np.asarray(orders, dtype=np.int32)
    check_completion(prompt, y)
    count, length = orders.shape[1], len(prompt)
    if sorted(orders[0].tolist()) != hidden_positions(prompt).tolist():
        raise ValueError("Orders must permute the hidden positions of the prompt")
    if count == 0:
        return np.zeros(len(orders))
    # State before step t: the prompt plus y on the first t positions of the order.
    revealed = np.zeros((len(orders), count, length), dtype=bool)
    for t in range(1, count):
        revealed[np.arange(len(orders)), t:, orders[:, t - 1]] = True
    states = np.where(revealed, y, prompt).astype(np.int32)
    log_q = model.log_conditionals(states.reshape(-1, length)).reshape(len(orders), count, length, -1)
    rows, steps = np.arange(len(orders))[:, None], np.arange(count)[None, :]
    chosen = log_q[rows, steps, orders, y[orders]]
    return chosen.sum(axis=1) - lgamma(count + 1)


def posterior_over_orders(model: Model, prompt: IntArray, y: IntArray) -> Posterior:
    """All u! trajectories from x to y, with p(y | x) and p(tau | x, y). Exact, u <= 8."""
    orders = np.array(list(permutations(hidden_positions(prompt).tolist())), dtype=np.int32)
    orders = orders.reshape(len(orders), -1)
    log_tau = log_p_tau(model, prompt, y, orders)
    log_y = float(np.logaddexp.reduce(log_tau))
    log_post = log_tau - log_y
    count = orders.shape[1]
    elbo = float(log_tau.mean() + lgamma(count + 1))
    return Posterior(
        orders=orders, log_p_tau=log_tau, log_p_y=log_y, log_posterior=log_post,
        entropy=float(-(np.exp(log_post) @ log_post)), elbo=elbo, elbo_gap=log_y - elbo,
    )


def log_p_y(model: Model, prompt: IntArray, y: IntArray) -> float:
    """log p(y | x) by summing all u! reveal orders."""
    return posterior_over_orders(model, prompt, y).log_p_y


def log_p_all_y(model: Model, prompt: IntArray) -> tuple[IntArray, FloatArray]:
    """All completions y of the prompt and log p(y | x), by dynamic programming over
    the 5^u partial strings between x and full strings (same recursion as the exact KL)."""
    prompt = np.asarray(prompt, dtype=np.int32)
    hidden = hidden_positions(prompt)
    contexts = build_contexts(len(hidden), len(ALPHABET))
    states = np.tile(prompt, (len(contexts.tokens), 1))
    states[:, hidden] = contexts.tokens
    log_cond = model.log_conditionals(states)[:, hidden, :]
    log_p = joint_log_probs(contexts, log_cond)
    completions = np.tile(prompt, (len(log_p), 1))
    completions[:, hidden] = enumerate_tokens(len(ALPHABET), len(hidden))
    return completions, log_p
