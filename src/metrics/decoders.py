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
