"""Gates for the exact path metric (src/metrics/paths.py) and the model-agnostic sampler (src/rl/samplers.py)."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from src.config import LENGTH, MASK
from src.evaluation import build_contexts, joint_log_probs
from src.metrics.paths import log_conditionals, path_kl, visit_log_mass
from src.model import complete, forward, init_parameters
from src.rl.losses.trajectory_balance import log_p_trajectory
from src.rl.samplers import complete_uniform

REF = init_parameters(jax.random.key(21), 16)
POL = init_parameters(jax.random.key(22), 16)


@pytest.fixture(scope="module")
def tables():
    contexts = build_contexts()
    return contexts, log_conditionals(forward, REF, contexts), log_conditionals(forward, POL, contexts)


def test_uniform_sampler_reproduces_model_complete() -> None:
    tokens = jnp.full((64, LENGTH), MASK, dtype=jnp.int32).at[:, 2].set(1)
    a, oa = complete(POL, tokens, jax.random.key(3))
    b, ob = complete_uniform(forward, POL, tokens, jax.random.key(3))
    assert (np.asarray(a) == np.asarray(b)).all() and (np.asarray(oa) == np.asarray(ob)).all()


def test_visit_mass_matches_joint_and_self_distance_is_zero(tables) -> None:
    contexts, ref, pol = tables
    np.testing.assert_allclose(visit_log_mass(contexts, ref)[contexts.full_ids], joint_log_probs(contexts, ref),
                               atol=1e-12)
    same = path_kl(contexts, ref, ref)
    assert abs(same.kl_traj) < 1e-12 and abs(same.kl_term) < 1e-12
    d = path_kl(contexts, ref, pol)
    assert d.kl_traj >= d.kl_term > 0 and d.kl_path >= 0


def test_trajectory_kl_matches_monte_carlo(tables) -> None:
    """KL_traj by the state recursion equals E_{tau ~ P_ref}[log P_ref(tau) - log P_theta(tau)] by sampling."""
    contexts, ref, pol = tables
    exact = path_kl(contexts, ref, pol).kl_traj
    n = 20000
    empty = jnp.full((n, LENGTH), MASK, dtype=jnp.int32)
    ys, orders = complete_uniform(forward, REF, empty, jax.random.key(7))
    diff = np.asarray(log_p_trajectory(forward, REF, empty, ys, orders) - log_p_trajectory(forward, POL, empty, ys, orders),
                      dtype=np.float64)
    assert abs(diff.mean() - exact) < 4 * diff.std() / np.sqrt(n)


def test_ar_exact_law_is_normalised_and_aligned_with_string_order() -> None:
    """For a context-free denoiser the left-to-right law equals the random-order law string by string, so the
    exact AR evaluator and the joint recursion index strings identically."""
    from src.data import enumerate_tokens
    from src.rl.train import ar_log_probs
    table = jax.random.normal(jax.random.key(9), (LENGTH, 4))

    def fixed(_: object, tokens: jax.Array) -> jax.Array:
        return jnp.broadcast_to(table, (*tokens.shape, 4))

    contexts = build_contexts()
    full = enumerate_tokens(4, LENGTH)
    log_ar = ar_log_probs(fixed, None, full)
    np.testing.assert_allclose(np.exp(log_ar).sum(), 1.0, atol=1e-5)
    np.testing.assert_allclose(log_ar, joint_log_probs(contexts, log_conditionals(fixed, None, contexts)), atol=1e-4)
    mlp = ar_log_probs(forward, POL, full)
    np.testing.assert_allclose(np.exp(mlp).sum(), 1.0, atol=1e-4)


def test_decoder_sampler_matches_the_exact_decoder_law(tables) -> None:
    """complete_proposal (JAX) draws strings with the frequencies of metrics.decoders.proposal_terminal_log_probs."""
    from src.metrics.decoders import proposal_terminal_log_probs
    from src.rl.samplers import complete_proposal
    contexts, _, pol = tables
    for temperature in (1.0, 0.6):
        exact = np.exp(proposal_terminal_log_probs(contexts, pol, temperature))
        n = 40000
        ys, orders = complete_proposal(forward, POL, jnp.full((n, LENGTH), MASK, dtype=jnp.int32), jax.random.key(8),
                                       temperature)
        assert (np.asarray(orders) >= 0).all() and (np.sort(np.asarray(orders), axis=1) == np.arange(LENGTH)).all()
        freq = np.bincount(np.asarray(ys) @ (4 ** np.arange(LENGTH - 1, -1, -1)), minlength=4**LENGTH) / n
        top = np.argsort(-exact)[:15]
        se = np.sqrt(exact[top] * (1 - exact[top]) / n)
        assert (np.abs(freq[top] - exact[top]) < 4 * se + 1e-4).all(), (temperature, freq[top], exact[top])


def test_decoder_sampler_first_step_matches_step_law(tables) -> None:
    """The (position, letter) of the first committed token equals proposal_step_law at the empty state, at T = 0.3
    where scoring proposals with tempered instead of untempered probabilities changes the law (gfn_lab gate G2b)."""
    from src.metrics.decoders import proposal_step_law
    from src.rl.samplers import complete_proposal
    contexts, _, pol = tables
    empty = len(contexts.tokens) - 1
    law = proposal_step_law(pol[empty][None], np.ones((1, LENGTH), dtype=bool), 0.3)[0]
    n = 200000
    ys, orders = complete_proposal(forward, POL, jnp.full((n, LENGTH), MASK, dtype=jnp.int32), jax.random.key(9), 0.3)
    first = np.asarray(orders)[:, 0]
    letter = np.asarray(ys)[np.arange(n), first]
    freq = np.zeros((LENGTH, 4))
    np.add.at(freq, (first, letter), 1.0 / n)
    se = np.sqrt(law * (1 - law) / n)
    assert (np.abs(freq - law) < 5 * se + 2e-4).all(), np.abs(freq - law).max()


def test_diversity_metric() -> None:
    """E[# distinct in k draws]: 1 for a point mass, k * (1 - (1 - 1/n)^k) / ... = n (1 - (1 - 1/n)^k) for uniform."""
    from src.rl.train import diversity
    n, k = 100, 16
    uniform = np.full(n, 1 / n)
    total, mass, top = diversity(uniform, np.arange(10), k)
    assert total == pytest.approx(n * (1 - (1 - 1 / n) ** k)) and mass == pytest.approx(0.1)
    assert top == pytest.approx(10 * (1 - (1 - 1 / n) ** k))
    point = np.zeros(n); point[3] = 1.0
    assert diversity(point, np.array([3]), k) == pytest.approx((1.0, 1.0, 1.0))
