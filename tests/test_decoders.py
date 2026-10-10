"""Gates for the exact low-confidence-remasking decoder law (src/metrics/decoders.py)."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")

from src.config import LENGTH, MASK
from src.evaluation import build_contexts
from src.metrics.decoders import proposal_step_law, proposal_terminal_log_probs
from src.metrics.paths import log_conditionals
from src.model import forward, init_parameters

PARAMS = init_parameters(jax.random.key(31), 16)


@pytest.fixture(scope="module")
def table():
    contexts = build_contexts()
    return contexts, log_conditionals(forward, PARAMS, contexts)


def test_step_law_is_normalised(table) -> None:
    contexts, log_cond = table
    ids = np.random.default_rng(0).choice(np.flatnonzero(contexts.known < LENGTH), 500, replace=False)
    law = proposal_step_law(log_cond[ids], contexts.tokens[ids] == MASK, 0.6)
    np.testing.assert_allclose(law.sum(axis=(1, 2)), 1.0, atol=1e-10)
    assert (law[contexts.tokens[ids] != MASK] == 0).all()


@pytest.mark.parametrize("temperature", [1.0, 0.6])
def test_terminal_law_matches_the_sampling_procedure(table, temperature: float) -> None:
    """Simulate the decoder itself (propose at every hidden position, score untempered, commit the best, lowest index
    on ties) and compare string frequencies with the exact law on its 15 most likely strings."""
    contexts, log_cond = table
    log_p = proposal_terminal_log_probs(contexts, log_cond, temperature)
    assert np.exp(log_p).sum() == pytest.approx(1.0, abs=1e-10)
    rng, n = np.random.default_rng(1), 30000
    state = np.full((n, LENGTH), MASK)
    rows = np.arange(n)
    for _ in range(LENGTH):
        ids = state @ contexts.powers
        lq = log_cond[ids]                                                          # (n, L, 4)
        tempered = lq / temperature
        q = np.exp(tempered - np.logaddexp.reduce(tempered, axis=-1, keepdims=True))
        proposal = (rng.random((n, LENGTH, 1)) > np.cumsum(q, axis=-1)).sum(-1).clip(max=3)
        score = np.take_along_axis(np.exp(lq), proposal[..., None], axis=-1)[..., 0]
        score = np.where(state == MASK, score, -1.0)
        best = score.argmax(axis=-1)                                                # first maximum = lowest index
        state[rows, best] = proposal[rows, best]
    full_ids = state @ (4 ** np.arange(LENGTH - 1, -1, -1))
    freq = np.bincount(full_ids, minlength=4**LENGTH) / n
    exact = np.exp(log_p)
    top = np.argsort(-exact)[:15]
    se = np.sqrt(exact[top] * (1 - exact[top]) / n)
    assert (np.abs(freq[top] - exact[top]) < 4 * se + 1e-4).all(), (freq[top], exact[top])


@pytest.mark.parametrize("temperature", [0.3, 1.0])
def test_step_law_matches_one_step_simulation(table, temperature: float) -> None:
    """At fixed states, simulate one decoder step 200k times; every (position, letter) cell matches the formula.
    Sensitive to scoring the proposals with tempered instead of untempered probabilities (gfn_lab gate G2b)."""
    contexts, log_cond = table
    rng = np.random.default_rng(2)
    ids = rng.choice(np.flatnonzero((contexts.known >= 1) & (contexts.known <= 5)), 6, replace=False)
    ids = np.append(ids, len(contexts.tokens) - 1)                                   # plus the all-mask state
    law = proposal_step_law(log_cond[ids], contexts.tokens[ids] == MASK, temperature)
    n = 200000
    for k, state_id in enumerate(ids):
        lq = log_cond[state_id]
        tempered = lq / temperature
        q = np.exp(tempered - np.logaddexp.reduce(tempered, axis=-1, keepdims=True))
        proposal = (rng.random((n, LENGTH, 1)) > np.cumsum(q, axis=-1)[None]).sum(-1).clip(max=3)
        score = np.where(contexts.tokens[state_id] == MASK, np.exp(lq)[np.arange(LENGTH), proposal], -1.0)
        best = score.argmax(axis=-1)
        freq = np.zeros((LENGTH, 4))
        np.add.at(freq, (best, proposal[np.arange(n), best]), 1.0 / n)
        se = np.sqrt(law[k] * (1 - law[k]) / n)
        assert (np.abs(freq - law[k]) < 5 * se + 2e-4).all(), (state_id, np.abs(freq - law[k]).max())


def test_decoder_hellinger(table) -> None:
    """Zero for identical models; h2_traj >= h2_term; the trajectory BC equals E_{tau ~ P}[sqrt(Q(tau) / P(tau))] by
    simulating the decoder under P and scoring each visited step with both exact step laws."""
    from src.metrics.decoders import proposal_hellinger
    contexts, log_cond = table
    same = proposal_hellinger(contexts, log_cond, log_cond, 0.6)
    assert abs(same[0]) < 1e-9 and abs(same[1]) < 1e-9
    other = log_conditionals(forward, init_parameters(jax.random.key(32), 16), contexts)
    h2_traj, h2_term, excess = proposal_hellinger(contexts, log_cond, other, 1.0)
    assert h2_traj >= h2_term > 0 and excess >= 0
    rng, n = np.random.default_rng(3), 20000
    state, ratio = np.full((n, LENGTH), MASK), np.ones(n)
    for _ in range(LENGTH):
        ids = state @ contexts.powers
        law_p = proposal_step_law(log_cond[ids], state == MASK, 1.0)
        law_q = proposal_step_law(other[ids], state == MASK, 1.0)
        flat = law_p.reshape(n, -1)
        pick = (rng.random((n, 1)) > np.cumsum(flat, axis=1)).sum(1).clip(max=flat.shape[1] - 1)
        pos, letter = pick // 4, pick % 4
        ratio *= np.sqrt(law_q.reshape(n, -1)[np.arange(n), pick] / flat[np.arange(n), pick])
        state[np.arange(n), pos] = letter
    assert abs(ratio.mean() - (1 - h2_traj)) < 4 * ratio.std() / np.sqrt(n)
