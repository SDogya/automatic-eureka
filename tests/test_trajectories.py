from math import lgamma, log

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from src.config import MASK, encode
from src.evaluation import build_contexts, joint_log_probs
from src.metrics.trajectories import Model, log_p_all_y, log_p_tau, posterior_over_orders
from src.model import forward, init_parameters

MODEL = Model(forward, init_parameters(jax.random.key(11), 16))
PROMPT = np.array([0, MASK, 2, MASK, MASK, 1, MASK, 3], dtype=np.int32)
Y = encode(["ACGTACCT"])[0]


def test_empty_prompt_matches_exact_joint_distribution() -> None:
    empty = np.full(8, MASK, dtype=np.int32)
    completions, log_p = log_p_all_y(MODEL, empty)
    contexts = build_contexts()
    reference = joint_log_probs(contexts, MODEL.log_conditionals(contexts.tokens))
    np.testing.assert_allclose(log_p, reference, rtol=0, atol=1e-10)
    np.testing.assert_allclose(np.exp(log_p).sum(), 1, atol=1e-10)
    index = int(np.flatnonzero((completions == Y).all(axis=1))[0])
    assert posterior_over_orders(MODEL, empty, Y).log_p_y == pytest.approx(log_p[index], abs=1e-6)


def test_prompt_posterior_is_consistent() -> None:
    completions, log_p = log_p_all_y(MODEL, PROMPT)
    assert len(completions) == 4**4 and np.all(completions[:, PROMPT != MASK] == PROMPT[PROMPT != MASK])
    np.testing.assert_allclose(np.exp(log_p).sum(), 1, atol=1e-10)
    posterior = posterior_over_orders(MODEL, PROMPT, Y)
    index = int(np.flatnonzero((completions == Y).all(axis=1))[0])
    assert posterior.log_p_y == pytest.approx(log_p[index], abs=1e-6)
    assert len(posterior.orders) == 24
    np.testing.assert_allclose(np.exp(posterior.log_posterior).sum(), 1, atol=1e-12)
    assert 0 <= posterior.entropy <= log(24) + 1e-12
    assert posterior.elbo_gap >= -1e-12


def test_context_free_model_has_uniform_orders() -> None:
    table = jax.random.normal(jax.random.key(3), (8, 4))
    model = Model(lambda _, tokens: jnp.broadcast_to(table, (*tokens.shape, 4)), None)
    posterior = posterior_over_orders(model, PROMPT, Y)
    assert posterior.entropy == pytest.approx(lgamma(5))
    assert posterior.elbo_gap == pytest.approx(0, abs=1e-9)
    log_q = np.asarray(jax.nn.log_softmax(table))
    expected = sum(log_q[i, Y[i]] for i in np.flatnonzero(PROMPT == MASK))
    assert posterior.log_p_y == pytest.approx(expected, abs=1e-6)


def test_rejects_inconsistent_completion() -> None:
    with pytest.raises(ValueError):
        log_p_tau(MODEL, PROMPT, encode(["CACGTACT"])[0], np.array([[1, 3, 4, 6]]))


def test_jax_trajectory_log_prob_matches_numpy() -> None:
    from src.rl.losses.trajectory_balance import log_p_trajectory

    posterior = posterior_over_orders(MODEL, PROMPT, Y)
    orders = np.full((len(posterior.orders), 8), -1, dtype=np.int32)
    orders[:, :4] = posterior.orders
    exact = log_p_trajectory(forward, MODEL.params, jnp.asarray(np.tile(PROMPT, (len(orders), 1))),
                             jnp.asarray(np.tile(Y, (len(orders), 1))), jnp.asarray(orders))
    np.testing.assert_allclose(exact, posterior.log_p_tau, atol=1e-5)


def test_sampled_orders_are_valid_trajectories() -> None:
    from src.model import complete

    tokens, orders = complete(MODEL.params, jnp.asarray(np.tile(PROMPT, (64, 1))), jax.random.key(9))
    tokens, orders = np.asarray(tokens), np.asarray(orders)
    hidden = sorted(np.flatnonzero(PROMPT == MASK).tolist())
    assert all(sorted(row[:4].tolist()) == hidden and np.all(row[4:] == -1) for row in orders)
    assert np.all(tokens[:, PROMPT != MASK] == PROMPT[PROMPT != MASK]) and np.all(tokens != MASK)


def test_chain_rule_decomposition_matches_direct_trajectory_kl() -> None:
    from src.metrics.trajectory_fit import direct_kl_tau, prompt_fit

    reference, trained = MODEL, Model(forward, init_parameters(jax.random.key(12), 16))
    prompt = np.array([0, MASK, 2, MASK, 1, 1, MASK, 3], dtype=np.int32)
    log_reward = np.random.default_rng(4).normal(size=4**8) * 2
    fit = prompt_fit(reference, trained, prompt, log_reward, mass=1.0)
    assert fit["target_mass_covered"] == pytest.approx(1.0)
    assert fit["kl_tau"] == pytest.approx(direct_kl_tau(reference, trained, prompt, log_reward), rel=1e-5)
    same = prompt_fit(reference, reference, prompt, log_reward, mass=1.0)
    assert same["kl_paths_given_y"] == pytest.approx(0, abs=1e-9)
