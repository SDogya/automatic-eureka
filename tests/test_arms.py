"""Gates for the ported baselines (src/rl/losses/baselines.py, masks.py): identities that, if broken, make a
comparison wrong rather than merely crash. Exact enumeration on small prompts throughout."""

from itertools import permutations, product
from math import lgamma

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from src.config import MASK
from src.metrics.trajectories import Model, log_p_y
from src.model import forward, init_parameters
from src.rl.losses.baselines import (ar_orders, ar_token_log_probs, complete_ar, espo_loss, group_advantage,
                                     grpo_loss, justgrpo_loss, order_variance)
from src.rl.losses.masks import draw_masks, masked_scores
from src.rl.losses.trafl import Reference, surrogate_log_prob
from src.rl.losses.trajectory_balance import log_p_trajectory

jax.config.update("jax_enable_x64", False)
POLICY = init_parameters(jax.random.key(11), 16)
REFERENCE = Reference(forward, init_parameters(jax.random.key(12), 16))
PROMPT = jnp.asarray([0, MASK, 2, MASK, 1, MASK, 3, 0])                   # u = 3 hidden positions: 1, 3, 5
HIDDEN = np.flatnonzero(np.asarray(PROMPT) == MASK)


def all_completions(prompt: jax.Array) -> jax.Array:
    hidden = np.flatnonzero(np.asarray(prompt) == MASK)
    rows = []
    for letters in product(range(4), repeat=len(hidden)):
        row = np.asarray(prompt).copy()
        row[hidden] = letters
        rows.append(row)
    return jnp.asarray(np.array(rows, dtype=np.int32))


def orders_for(hidden: np.ndarray) -> np.ndarray:
    pad = 8 - len(hidden)
    return np.array([list(p) + [-1] * pad for p in permutations(hidden)], dtype=np.int32)


def test_comp_masks_are_antithetic_pairs() -> None:
    contexts = jnp.asarray([[MASK] * 8, list(PROMPT), [0, 1, 2, MASK, 0, 1, 2, 3], [0, 1, 2, 3, 0, 1, 2, 3]])
    hidden = contexts == MASK
    masks = np.asarray(draw_masks(jax.random.key(0), hidden, 8, "comp"))
    h = np.asarray(hidden)
    for k in range(4):
        a, b = masks[k], masks[k + 4]
        for row in range(4):
            u = h[row].sum()
            if u >= 2:
                assert not (a[row] & b[row]).any() and ((a[row] | b[row]) == h[row]).all()
                assert a[row].any() and b[row].any()
            elif u == 1:
                assert (a[row] == h[row]).all() and (b[row] == h[row]).all()
            else:
                assert not a[row].any() and not b[row].any()


def test_comp_surrogate_for_uniform_model() -> None:
    def uniform(_: object, tokens: jax.Array) -> jax.Array:
        return jnp.zeros((*tokens.shape, 4))

    completions = all_completions(PROMPT)[None, :4]
    value = surrogate_log_prob(uniform, None, PROMPT[None], completions, jax.random.key(1), 8, "paper", "comp")
    np.testing.assert_allclose(value, -np.log(4), rtol=1e-6)


def test_espo_first_ppo_epoch_equals_single_update_and_formula() -> None:
    """With theta_old = theta the clipped ratio is inactive: gradients of espo (old=None), espo_ppo (old=theta)
    and the explicit -A s_theta + kappa/2 (u (s_theta - s_ref))^2 on the same masks coincide (gfn_lab gate T12)."""
    completions = all_completions(PROMPT)[jnp.asarray([3, 17, 40, 63, 5, 22])].reshape(2, 3, 8)
    contexts = jnp.stack([PROMPT, PROMPT])
    rewards = jnp.asarray([[0.3, -1.0, 2.0], [1.5, 0.2, -0.7]])
    key, kappa = jax.random.key(2), 0.3

    def explicit(p: object) -> jax.Array:
        masks = draw_masks(key, jnp.broadcast_to(contexts[:, None] == MASK, completions.shape), 4, "comp")
        s = masked_scores(forward, p, completions, masks).mean(0)
        s_ref = masked_scores(forward, REFERENCE.params, completions, masks).mean(0)
        return jnp.mean(-group_advantage(rewards) * s + kappa * 0.5 * (3 * (s - s_ref)) ** 2)

    single = jax.grad(lambda p: espo_loss(p, forward, contexts, completions, rewards, key, kappa=kappa,
                                          reference=REFERENCE))(POLICY)
    ppo = jax.grad(lambda p: espo_loss(p, forward, contexts, completions, rewards, key, kappa=kappa,
                                       reference=REFERENCE, old=POLICY))(POLICY)
    direct = jax.grad(explicit)(POLICY)
    for a, b, c in zip(jax.tree.leaves(single), jax.tree.leaves(ppo), jax.tree.leaves(direct)):
        np.testing.assert_allclose(a, b, atol=1e-6)
        np.testing.assert_allclose(a, c, atol=1e-6)
    assert max(float(jnp.abs(g).max()) for g in jax.tree.leaves(single)) > 1e-4


def test_trajectory_law_is_normalised_and_matches_exact_evaluator() -> None:
    """The per-trajectory log-probability that grpo uses sums to one over all (y, order) pairs and its order
    marginal equals the exact log p(y | x) of the metrics module."""
    ys = all_completions(PROMPT)
    orders = jnp.asarray(orders_for(HIDDEN))
    lp = jnp.stack([log_p_trajectory(forward, POLICY, jnp.broadcast_to(PROMPT, ys.shape), ys,
                                     jnp.broadcast_to(o, ys.shape)) for o in orders], axis=1)   # (Y, u!)
    log_marginal = jax.nn.logsumexp(lp, axis=1)
    assert float(jax.nn.logsumexp(log_marginal)) == pytest.approx(0.0, abs=1e-5)
    model = Model(forward, POLICY)
    exact = np.array([log_p_y(model, np.asarray(PROMPT), np.asarray(y)) for y in ys[:8]])
    np.testing.assert_allclose(log_marginal[:8], exact, atol=1e-5)


def test_grpo_expected_gradient_is_fisher_identity() -> None:
    """E_tau[A(y) grad log p(tau|x)] = sum_y p(y|x) A(y) grad log p(y|x): grpo's per-trajectory gradient is unbiased
    for the string-level policy gradient (exact sums over all 64 completions x 6 orders)."""
    ys = all_completions(PROMPT)
    orders = jnp.asarray(orders_for(HIDDEN))
    adv = jnp.asarray(np.random.default_rng(3).normal(size=len(ys)), dtype=jnp.float32)

    def log_tau(p: object) -> jax.Array:
        return jnp.stack([log_p_trajectory(forward, p, jnp.broadcast_to(PROMPT, ys.shape), ys,
                                           jnp.broadcast_to(o, ys.shape)) for o in orders], axis=1)

    weights = jnp.exp(log_tau(POLICY))
    by_path = jax.grad(lambda p: jnp.sum(jax.lax.stop_gradient(weights) * adv[:, None] * log_tau(p)))(POLICY)
    p_y = weights.sum(1)
    by_string = jax.grad(lambda p: jnp.sum(p_y * adv * jax.nn.logsumexp(log_tau(p), axis=1)))(POLICY)
    for a, b in zip(jax.tree.leaves(by_path), jax.tree.leaves(by_string)):
        np.testing.assert_allclose(a, b, atol=2e-5)
    # and grpo_loss itself is -mean(A * log p(tau)) on the sampled pairs
    sample = jnp.asarray([[3, 40]]), jnp.asarray([[0, 4]])
    cs, os_ = ys[sample[0]], orders[sample[1]]
    rewards = jnp.asarray([[1.0, -2.0]])
    value = grpo_loss(POLICY, forward, PROMPT[None], cs, os_, rewards)
    expect = -jnp.mean(group_advantage(rewards) * log_p_trajectory(forward, POLICY, jnp.broadcast_to(PROMPT, cs.shape), cs, os_))
    assert float(value) == pytest.approx(float(expect), rel=1e-6)


def test_justgrpo_ar_policy_is_a_normalised_left_to_right_law() -> None:
    ys = all_completions(PROMPT)
    tok = ar_token_log_probs(forward, POLICY, jnp.broadcast_to(PROMPT, ys.shape), ys)
    log_ar = tok.values.sum(-1)
    assert float(jax.nn.logsumexp(log_ar)) == pytest.approx(0.0, abs=1e-5)
    assert np.asarray(ar_orders(PROMPT[None]))[0].tolist() == [1, 3, 5, -1, -1, -1, -1, -1]
    # the AR law is the uniform-order trajectory law along the ascending order, times u!
    asc = jnp.broadcast_to(ar_orders(PROMPT[None])[0], ys.shape)
    lp = log_p_trajectory(forward, POLICY, jnp.broadcast_to(PROMPT, ys.shape), ys, asc)
    np.testing.assert_allclose(log_ar, lp + lgamma(len(HIDDEN) + 1), atol=1e-5)


def test_justgrpo_sampler_matches_its_law() -> None:
    prompt = jnp.asarray([0, 1, MASK, 3, MASK, 1, 2, 3])
    ys = all_completions(prompt)
    exact = np.exp(np.asarray(ar_token_log_probs(forward, POLICY, jnp.broadcast_to(prompt, ys.shape), ys).values.sum(-1)))
    out, orders = complete_ar(forward, POLICY, jnp.broadcast_to(prompt, (40000, 8)), jax.random.key(4))
    assert (np.asarray(orders)[:, :2] == [2, 4]).all() and (np.asarray(orders)[:, 2:] == -1).all()
    ids = np.asarray(out)[:, 2] * 4 + np.asarray(out)[:, 4]
    freq = np.bincount(ids, minlength=16) / len(ids)
    assert np.abs(freq - exact).max() < 4 * np.sqrt(exact.max() / len(ids)) + 1e-3


def test_justgrpo_single_update_gradient() -> None:
    """policy update steps = 1 (the paper's setting): rho = 1, gradient = -mean_i (A_i / u) grad log pi_AR(y_i | x)."""
    ys = all_completions(PROMPT)[jnp.asarray([[1, 9, 30, 50]])]
    rewards = jnp.asarray([[0.0, 1.0, 3.0, -1.0]])
    got = jax.grad(lambda p: justgrpo_loss(p, forward, PROMPT[None], ys, rewards))(POLICY)
    adv = group_advantage(rewards)
    want = jax.grad(lambda p: -jnp.mean(adv * ar_token_log_probs(forward, p, PROMPT[None], ys).values.sum(-1) / 3))(POLICY)
    for a, b in zip(jax.tree.leaves(got), jax.tree.leaves(want)):
        np.testing.assert_allclose(a, b, atol=1e-6)


def test_order_variance_is_unbiased() -> None:
    """E over independent uniform order pairs of 0.5 (X_s1 - X_s2)^2 equals Var_s(X_s) (gfn_lab gate T15)."""
    y = all_completions(PROMPT)[jnp.asarray([[13]])]
    orders = jnp.asarray(orders_for(HIDDEN))
    rep = jnp.broadcast_to(PROMPT, (len(orders), 8))
    yy = jnp.broadcast_to(y[0, 0], (len(orders), 8))
    x = (log_p_trajectory(forward, POLICY, rep, yy, orders) - log_p_trajectory(forward, REFERENCE.params, rep, yy, orders)) / 3
    target = float(jnp.var(x))
    draws = jax.vmap(lambda k: order_variance(POLICY, forward, REFERENCE, PROMPT[None], y, k)[0, 0])(
        jax.random.split(jax.random.key(5), 20000))
    se = float(jnp.std(draws)) / np.sqrt(len(draws))
    assert abs(float(draws.mean()) - target) < 4 * se + 1e-6
    assert target > 1e-4
