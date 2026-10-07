from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from src.config import SampleRequest, encode
from src.model import (
    forward, init_parameters, load_parameters, masked_loss, parameter_count,
    sample, save_parameters,
)


def test_model_shapes_and_empty_mask_gradient() -> None:
    params = init_parameters(jax.random.key(1))
    tokens = jnp.asarray(encode(["ACGTACGT", "AAAAAAAA"]))
    assert parameter_count(params) == 12352
    logits = forward(params, tokens)
    assert logits.shape == (2, 8, 4)
    np.testing.assert_allclose(np.asarray(jax.nn.softmax(logits)).sum(axis=-1), 1, atol=1e-6)
    mask = jnp.zeros_like(tokens, dtype=bool)
    loss, gradients = jax.value_and_grad(masked_loss)(params, tokens, mask)
    assert float(loss) == 0
    assert all(np.all(np.asarray(g) == 0) for g in jax.tree.leaves(gradients))
    zeros = jax.tree.map(jnp.zeros_like, params)
    mask = jnp.asarray([[True, False] * 4, [False] * 8])
    assert float(masked_loss(zeros, tokens, mask)) == pytest.approx(2 * np.log(4))


def test_sampling_preserves_context_and_checkpoint(tmp_path: Path) -> None:
    params = init_parameters(jax.random.key(2))
    path = tmp_path / "model.npz"
    save_parameters(path, params)
    restored = load_parameters(path)
    for a, b in zip(jax.tree.leaves(params), jax.tree.leaves(restored)):
        np.testing.assert_array_equal(a, b)
    request = SampleRequest(context="A??T?C??", count=64)
    tokens = sample(restored, request)
    assert tokens.shape == (64, 8)
    for position, symbol in ((0, 0), (3, 3), (5, 1)):
        assert np.all(tokens[:, position] == symbol)
    np.testing.assert_array_equal(tokens, sample(restored, request))
    full = sample(restored, SampleRequest(context="ACGTACGT", count=3))
    np.testing.assert_array_equal(full, np.tile(encode(["ACGTACGT"]), (3, 1)))


def test_uniform_sampler_frequencies() -> None:
    params = jax.tree.map(jnp.zeros_like, init_parameters(jax.random.key(5)))
    tokens = sample(params, SampleRequest(count=4096, seed=73))
    for position in range(8):
        frequencies = np.bincount(tokens[:, position], minlength=4) / len(tokens)
        assert np.max(np.abs(frequencies - 0.25)) < 0.04
