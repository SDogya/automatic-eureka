"""Exact path-law distance from the reference at the empty prompt, for any denoiser.

Under the random-order generator a trajectory visits partial strings s; rho(s) is the probability that the
reference visits s. The trajectory KL decomposes over states (chain rule):
    KL_traj = KL(P_ref || P_theta) = sum_s rho_ref(s) * (1/u(s)) sum_{i hidden in s} KL(q_ref(.|s, i) || q_theta(.|s, i)),
the position choice 1/u being common to both. The path part is KL_traj - KL_term = E_{y~p_ref} KL(c_ref(tau|y) ||
c_theta(tau|y)), the conditional path law's distance given the completion (the quantity gfn_lab's S2 calls kl_path).
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from ..config import MASK, FloatArray
from ..evaluation import Contexts


class PathKL(NamedTuple):
    kl_traj: float
    kl_term: float
    kl_path: float


def log_conditionals(forward: object, params: object, contexts: Contexts, batch_size: int = 16384) -> FloatArray:
    """log q(a | s, i) for every partial string s (all 5^L), float64, shape (N, L, 4); full strings left at 0."""
    predict = jax.jit(forward)
    out = np.zeros((len(contexts.tokens), contexts.tokens.shape[1], MASK), dtype=np.float64)
    ids = np.flatnonzero(contexts.known < contexts.tokens.shape[1])
    for start in range(0, len(ids), batch_size):
        batch = ids[start:start + batch_size]
        logits = np.asarray(predict(params, jnp.asarray(contexts.tokens[batch])), dtype=np.float64)
        out[batch] = logits - np.logaddexp.reduce(logits, axis=-1, keepdims=True)
    return out


def visit_log_mass(contexts: Contexts, log_cond: FloatArray) -> FloatArray:
    """log rho(s) for every partial string: the probability the generator passes through s (same recursion as
    evaluation.joint_log_probs, kept for all states)."""
    length = contexts.tokens.shape[1]
    log_mass = np.full(len(contexts.tokens), -np.inf, dtype=np.float64)
    log_mass[-1] = 0.0
    for count in range(1, length + 1):
        ids = np.flatnonzero(contexts.known == count)
        tokens = contexts.tokens[ids]
        contributions = np.full((len(ids), length), -np.inf, dtype=np.float64)
        for position in range(length):
            selected = np.flatnonzero(tokens[:, position] != MASK)
            symbols = tokens[selected, position]
            parents = ids[selected] + (MASK - symbols) * contexts.powers[position]
            contributions[selected, position] = (log_mass[parents] + log_cond[parents, position, symbols]
                                                 - np.log(length - count + 1))
        log_mass[ids] = np.logaddexp.reduce(contributions, axis=1)
    return log_mass


def path_kl(contexts: Contexts, log_cond_ref: FloatArray, log_cond_theta: FloatArray) -> PathKL:
    length = contexts.tokens.shape[1]
    mass_ref = visit_log_mass(contexts, log_cond_ref)
    mass_theta = visit_log_mass(contexts, log_cond_theta)
    partial = np.flatnonzero(contexts.known < length)
    hidden = contexts.tokens[partial] == MASK
    step_kl = np.sum(np.exp(log_cond_ref[partial]) * (log_cond_ref[partial] - log_cond_theta[partial]), axis=-1)
    per_state = np.sum(np.where(hidden, step_kl, 0.0), axis=-1) / hidden.sum(axis=-1)
    kl_traj = float(np.exp(mass_ref[partial]) @ per_state)
    p_ref, log_t = mass_ref[contexts.full_ids], mass_theta[contexts.full_ids]
    kl_term = float(np.exp(p_ref) @ (p_ref - log_t))
    return PathKL(kl_traj, kl_term, kl_traj - kl_term)
