from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from src.config import MASK, encode
from src.model import (
    expected_parameter_count, forward, init_parameters, load_parameters, masked_loss,
    parameter_count, complete, save_parameters,
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
    context = jnp.tile(jnp.array([0, MASK, MASK, 3, MASK, 1, MASK, MASK], dtype=jnp.int32), (64, 1))
    tokens, orders = complete(restored, context, jax.random.key(4))
    tokens = np.asarray(tokens)
    assert tokens.shape == (64, 8) and np.all(tokens != MASK)
    for position, symbol in ((0, 0), (3, 3), (5, 1)):
        assert np.all(tokens[:, position] == symbol)
    np.testing.assert_array_equal(tokens, np.asarray(complete(restored, context, jax.random.key(4))[0]))
    assert np.all(np.asarray(orders)[:, 5:] == -1)
    full = jnp.asarray(np.tile(encode(["ACGTACGT"]), (3, 1)))
    np.testing.assert_array_equal(complete(restored, full, jax.random.key(5))[0], full)


def test_uniform_sampler_frequencies() -> None:
    params = jax.tree.map(jnp.zeros_like, init_parameters(jax.random.key(5)))
    tokens = np.asarray(complete(params, jnp.full((4096, 8), MASK, dtype=jnp.int32), jax.random.key(73))[0])
    for position in range(8):
        frequencies = np.bincount(tokens[:, position], minlength=4) / len(tokens)
        assert np.max(np.abs(frequencies - 0.25)) < 0.04


@pytest.mark.parametrize("exponent", range(1, 8))
def test_ablation_widths_and_checkpoint_roundtrip(exponent: int, tmp_path: Path) -> None:
    width = 2**exponent
    params = init_parameters(jax.random.key(3), width)
    assert [layer.weight.shape for layer in params] == [(40, width), (width, width), (width, 32)]
    assert parameter_count(params) == expected_parameter_count(width) == 41 * width + (width + 1) * (width + 32)
    save_parameters(tmp_path / "model.npz", params)
    restored = load_parameters(tmp_path / "model.npz")
    assert [layer.weight.shape for layer in restored] == [layer.weight.shape for layer in params]
