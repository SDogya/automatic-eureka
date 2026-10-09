from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from src.config import MASK, Config, encode
from src.data import token_ids
from src.hypotheses.common import Run, log_target_y, order_row, posterior_pair
from src.hypotheses.h3_fixed_point import weighted_fit
from src.metrics.trajectories import Model
from src.model import forward, init_parameters
from src.rl.losses.trafl import surrogate_log_prob

PROMPT = np.array([0, MASK, 2, MASK, MASK, 1, MASK, 3], dtype=np.int32)
Y = encode(["ACGTACCT"])[0]
RUN = Run(Path("."), Config(), Model(forward, init_parameters(jax.random.key(1), 16)),
          np.random.default_rng(0).normal(size=4**8) * 2)
MODEL = Model(forward, init_parameters(jax.random.key(2), 16))


def test_h1_residual_splits_into_finals_and_dropped_term() -> None:
    ys, log_ref, log_star = log_target_y(RUN, PROMPT)
    np.testing.assert_allclose(np.exp(log_star).sum(), 1, atol=1e-10)
    post, post_ref = posterior_pair(RUN, MODEL, PROMPT, Y)
    index = int(np.flatnonzero((ys == Y).all(axis=1))[0])
    log_z = np.logaddexp.reduce(log_ref + RUN.log_reward[token_ids(ys)])
    for row, order in enumerate(post.orders):
        assert order_row(PROMPT, np.r_[order, [-1] * 4]) == row
        log_star_tau = post_ref.log_p_tau[row] + RUN.log_reward[token_ids(Y[None])[0]] - log_z
        delta_tau = post.log_p_tau[row] - log_star_tau
        delta_y = post.log_p_y - log_star[index]
        log_rho = post.log_posterior[row] - post_ref.log_posterior[row]
        assert delta_tau == pytest.approx(delta_y + log_rho, abs=1e-6)


def test_h2_surrogate_expectation_is_the_scaled_elbo() -> None:
    post, post_ref = posterior_pair(RUN, MODEL, PROMPT, Y)
    args = (jnp.asarray(PROMPT[None]), jnp.asarray(Y[None, None]), jax.random.key(3), 20000, "paper")
    draw = surrogate_log_prob(forward, MODEL.params, *args) - surrogate_log_prob(forward, RUN.reference.params, *args)
    assert float(draw[0, 0]) == pytest.approx((post.elbo - post_ref.elbo) / 4, abs=0.01)


def test_weighted_fit_recovers_line() -> None:
    x = np.linspace(-1, 2, 30)
    slope, intercept, r2 = weighted_fit(x, 2 * x + 1, np.random.default_rng(1).random(30))
    assert (slope, intercept, r2) == pytest.approx((2, 1, 1))


def test_exact_surrogate_expectation_equals_elbo_over_u() -> None:
    from src.hypotheses.h2_figure import surrogate_expectation
    from src.metrics.trajectories import log_p_all_y, posterior_over_orders

    ys, _ = log_p_all_y(MODEL, PROMPT)
    expected = surrogate_expectation(MODEL, PROMPT)
    for index in (0, 37, 255):
        assert expected[index] == pytest.approx(posterior_over_orders(MODEL, PROMPT, ys[index]).elbo / 4, abs=1e-5)
