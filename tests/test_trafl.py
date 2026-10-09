import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from src.config import MASK, encode
from src.model import forward, init_parameters
from src.rl.losses.trafl import Reference, surrogate_log_prob, trafl_loss

CONTEXTS = jnp.asarray([[0, MASK, 2, MASK, MASK, 1, MASK, 3], [MASK] * 8, [0, 1, 2, 3, 0, 1, 2, 3]])
COMPLETIONS = jnp.stack([jnp.asarray(encode(["AACGTACT", "ATCATAGT"])),
                         jnp.asarray(encode(["GGGGCCCC", "TTTTAAAA"])),
                         jnp.asarray(encode(["ACGTACGT", "ACGTACGT"]))])
HIDDEN = np.asarray([4, 8, 0])


def uniform(_: object, tokens: jax.Array) -> jax.Array:
    return jnp.zeros((*tokens.shape, 4))


@pytest.mark.parametrize(("normalization", "expected"), [("paper", -np.log(4) * (HIDDEN > 0)),
                                                         ("elbo", -np.log(4) * HIDDEN)])
def test_surrogate_for_uniform_model(normalization: str, expected: np.ndarray) -> None:
    value = surrogate_log_prob(uniform, None, CONTEXTS, COMPLETIONS, jax.random.key(0), 8, normalization)
    np.testing.assert_allclose(value, np.repeat(expected[:, None], 2, axis=1), rtol=1e-6)


def test_elbo_is_unbiased_for_context_free_model() -> None:
    table = jax.random.normal(jax.random.key(1), (8, 4))

    def fixed(_: object, tokens: jax.Array) -> jax.Array:
        return jnp.broadcast_to(table, (*tokens.shape, 4))

    log_q = np.asarray(jax.nn.log_softmax(table))
    exact = np.array([[sum(log_q[i, y[i]] for i in range(8) if c[i] == MASK) for y in ys]
                      for c, ys in zip(np.asarray(CONTEXTS), np.asarray(COMPLETIONS))])
    estimate = surrogate_log_prob(fixed, None, CONTEXTS, COMPLETIONS, jax.random.key(2), 20000, "elbo")
    np.testing.assert_allclose(estimate, exact, atol=0.05)


def test_loss_gradients_with_mlp_and_reference() -> None:
    params = (init_parameters(jax.random.key(3), 16), jnp.zeros(()))
    rewards = jnp.asarray([[1.0, 3.0], [0.5, -0.5], [2.0, 2.0]])

    def log_z(scalar: jax.Array, contexts: jax.Array) -> jax.Array:
        return jnp.full(len(contexts), scalar)

    reference = Reference(forward, init_parameters(jax.random.key(4), 16))
    key = jax.random.key(5)
    loss, (policy_grad, z_grad) = jax.value_and_grad(trafl_loss)(
        params, forward, log_z, CONTEXTS, COMPLETIONS, rewards, key, beta=1.0, reference=reference)
    assert np.isfinite(float(loss))
    assert all(np.isfinite(np.asarray(g)).all() for g in jax.tree.leaves(policy_grad))
    log_p = surrogate_log_prob(forward, params[0], CONTEXTS, COMPLETIONS, key)
    log_ref = surrogate_log_prob(forward, reference.params, CONTEXTS, COMPLETIONS, key)
    delta = log_p - log_ref - (rewards - rewards.mean(axis=1, keepdims=True))
    assert float(loss) == pytest.approx(float(jnp.mean(delta**2)), rel=1e-5)
    assert float(z_grad) == pytest.approx(2 * float(jnp.mean(delta)), rel=1e-4, abs=1e-6)
