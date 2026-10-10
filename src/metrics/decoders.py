"""Exact terminal law of the low-confidence-remasking decoder (LLaDA / Fast-dLLM, one token per step), for any
denoiser, from the empty prompt.

At a partial string s every hidden position j proposes a letter v_j ~ softmax(logits_j / T); each proposal is scored
by its untempered probability q_j(v_j) = softmax(logits_j)(v_j); the highest score is committed (ties: lowest index).
The induced step law (gfn_lab's "proposal" rule, gated there) is
    P(i, v | s) = q^T_i(v) * prod_{j hidden, j != i} P_j(proposal at j does not beat (i, v)),
"does not beat" being q_j(v_j) <= q_i(v) for j > i and q_j(v_j) < q_i(v) for j < i. As T -> 0 it becomes greedy
confidence decoding. The terminal law sums this over every path (forward recursion over the 5^L partial strings).
"""

import numpy as np

from ..config import MASK, FloatArray
from ..evaluation import Contexts


def proposal_step_law(log_cond: FloatArray, hidden: np.ndarray, temperature: float) -> FloatArray:
    """P(i, v | s) for a batch of states: log_cond (B, L, 4) untempered log q, hidden (B, L) -> (B, L, 4)."""
    conf = np.exp(log_cond)
    tempered = log_cond / temperature
    q = np.exp(tempered - np.logaddexp.reduce(tempered, axis=-1, keepdims=True))
    length = log_cond.shape[1]
    later = np.arange(length)[None, :] > np.arange(length)[:, None]               # [i, j]: j after i
    ci = conf[:, :, :, None, None]                                                  # (B, i, v, 1, 1)
    cj = conf[:, None, None, :, :]                                                  # (B, 1, 1, j, v_j)
    not_beaten = np.where(later[None, :, None, :, None], cj <= ci, cj < ci)         # (B, i, v, j, v_j)
    p_j = (q[:, None, None, :, :] * not_beaten).sum(-1)                             # (B, i, v, j)
    others = hidden[:, None, None, :] & (np.arange(length)[None, None, None, :] != np.arange(length)[None, :, None, None])
    p_j = np.where(others, p_j, 1.0)
    law = q * np.prod(p_j, axis=-1)
    return np.where(hidden[..., None], law, 0.0)


def proposal_terminal_log_probs(contexts: Contexts, log_cond: FloatArray, temperature: float,
                                chunk: int = 16384) -> FloatArray:
    """log p(y) of every full string under the decoder, in the full_ids order of evaluation.joint_log_probs."""
    length = contexts.tokens.shape[1]
    mass = np.zeros(len(contexts.tokens), dtype=np.float64)
    mass[-1] = 1.0                                                                  # the all-mask state
    for count in range(0, length):
        ids = np.flatnonzero(contexts.known == count)
        for start in range(0, len(ids), chunk):
            parents = ids[start:start + chunk]
            parents = parents[mass[parents] > 0]
            if len(parents) == 0:
                continue
            tokens = contexts.tokens[parents]
            law = proposal_step_law(log_cond[parents], tokens == MASK, temperature)   # (B, L, 4)
            for position in range(length):
                for letter in range(MASK):
                    weight = mass[parents] * law[:, position, letter]
                    live = weight > 0
                    children = parents[live] - (MASK - letter) * contexts.powers[position]
                    np.add.at(mass, children, weight[live])
    full = mass[contexts.full_ids]
    with np.errstate(divide="ignore"):
        return np.log(full)


def proposal_hellinger(contexts: Contexts, log_cond_p: FloatArray, log_cond_q: FloatArray, temperature: float,
                       chunk: int = 16384) -> tuple[float, float, float]:
    """(h2_traj, h2_term, h2_traj - h2_term) between two denoisers under the decoder, from the empty prompt.

    Squared Hellinger distance h2 = 1 - BC with BC the Bhattacharyya coefficient. The trajectory BC is the forward
    recursion m(child) += m(s) * sqrt(P(a | s) Q(a | s)) over the 5^L partial strings; the terminal BC is
    sum_y sqrt(p(y) q(y)). By data processing h2_traj >= h2_term; their difference is the path part, bounded in
    [0, 1] and finite under near-deterministic decoders where path KL is not (gfn_lab S2's measure)."""
    length = contexts.tokens.shape[1]
    m = np.zeros(len(contexts.tokens), dtype=np.float64)
    mass_p = np.zeros_like(m)
    mass_q = np.zeros_like(m)
    m[-1] = mass_p[-1] = mass_q[-1] = 1.0
    for count in range(0, length):
        ids = np.flatnonzero(contexts.known == count)
        for start in range(0, len(ids), chunk):
            parents = ids[start:start + chunk]
            parents = parents[(mass_p[parents] > 0) | (mass_q[parents] > 0)]
            if len(parents) == 0:
                continue
            hidden = contexts.tokens[parents] == MASK
            law_p = proposal_step_law(log_cond_p[parents], hidden, temperature)
            law_q = proposal_step_law(log_cond_q[parents], hidden, temperature)
            bc = np.sqrt(law_p * law_q)
            for position in range(length):
                for letter in range(MASK):
                    children = parents - (MASK - letter) * contexts.powers[position]
                    np.add.at(m, children, m[parents] * bc[:, position, letter])
                    np.add.at(mass_p, children, mass_p[parents] * law_p[:, position, letter])
                    np.add.at(mass_q, children, mass_q[parents] * law_q[:, position, letter])
    full = contexts.full_ids
    h2_traj = 1.0 - float(m[full].sum())
    h2_term = 1.0 - float(np.sqrt(mass_p[full] * mass_q[full]).sum())
    return h2_traj, h2_term, h2_traj - h2_term
