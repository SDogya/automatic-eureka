"""A width-parameterized JAX MLP and random-order masked generation."""

from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .config import ALPHABET, LENGTH, MASK


class Layer(NamedTuple):
    weight: jax.Array
    bias: jax.Array


Params = tuple[Layer, Layer, Layer]
DEFAULT_WIDTH = 80


def shapes(width: int) -> tuple[tuple[int, int], ...]:
    """One-hot input -> width -> width -> logits for every position and letter."""
    return (LENGTH * (MASK + 1), width), (width, width), (width, LENGTH * len(ALPHABET))


def expected_parameter_count(width: int) -> int:
    return sum((rows + 1) * columns for rows, columns in shapes(width))


def init_parameters(key: jax.Array, width: int = DEFAULT_WIDTH) -> Params:
    keys = jax.random.split(key, 3)

    def layer(shape: tuple[int, int], layer_key: jax.Array) -> Layer:
        limit = np.sqrt(6 / sum(shape))
        weight = jax.random.uniform(layer_key, shape, dtype=jnp.float32, minval=-limit, maxval=limit)
        return Layer(weight, jnp.zeros(shape[1], dtype=jnp.float32))

    first, second, third = shapes(width)
    return layer(first, keys[0]), layer(second, keys[1]), layer(third, keys[2])


def forward(params: Params, tokens: jax.Array) -> jax.Array:
    hidden = jax.nn.one_hot(tokens, MASK + 1, dtype=jnp.float32).reshape((*tokens.shape[:-1], LENGTH * (MASK + 1)))
    for layer in params[:-1]:
        hidden = jax.nn.gelu(hidden @ layer.weight + layer.bias, approximate=False)
    logits = hidden @ params[-1].weight + params[-1].bias
    return logits.reshape((*tokens.shape[:-1], LENGTH, len(ALPHABET)))


def masked_nll(logits: jax.Array, tokens: jax.Array, mask: jax.Array) -> jax.Array:
    """Mean over the batch of the summed negative log-likelihood at the masked positions."""
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    nll = -jnp.take_along_axis(log_probs, tokens[..., None], axis=-1)[..., 0]
    return jnp.mean(jnp.sum(jnp.where(mask, nll, 0), axis=-1))


def masked_loss(params: Params, tokens: jax.Array, mask: jax.Array) -> jax.Array:
    return masked_nll(forward(params, jnp.where(mask, MASK, tokens)), tokens, mask)


def parameter_count(params: Params) -> int:
    return sum(int(array.size) for layer in params for array in layer)


def save_parameters(path: Path, params: Params) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{
        f"{name}{i}": np.asarray(array)
        for i, layer in enumerate(params) for name, array in zip(("w", "b"), layer)
    })


def load_parameters(path: Path) -> Params:
    with np.load(path, allow_pickle=False) as saved:
        if set(saved.files) != {f"{name}{i}" for i in range(3) for name in ("w", "b")}:
            raise ValueError("Invalid checkpoint fields")
        layers: list[Layer] = []
        for i, shape in enumerate(shapes(int(saved["w0"].shape[1]))):
            weight, bias = saved[f"w{i}"], saved[f"b{i}"]
            if weight.shape != shape or bias.shape != (shape[1],):
                raise ValueError("Invalid checkpoint shapes")
            if any(a.dtype != np.float32 or not np.isfinite(a).all() for a in (weight, bias)):
                raise ValueError("Checkpoint must contain finite float32 parameters")
            layers.append(Layer(jnp.asarray(weight), jnp.asarray(bias)))
    return layers[0], layers[1], layers[2]


def complete(params: Params, tokens: jax.Array, key: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Fill every MASK of a batch (B, LENGTH), one uniformly random hidden position per step.

    Returns the completed tokens and the reveal order (B, LENGTH): the position revealed at
    each step, -1 once nothing is hidden. Steps 0..u-1 are the denoising trajectory.
    """

    def fill(step: int, carry: tuple[jax.Array, jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array, jax.Array]:
        state, order, state_key = carry
        state_key, position_key, symbol_key = jax.random.split(state_key, 3)
        hidden = state == MASK
        position = jax.random.categorical(position_key, jnp.where(hidden, 0.0, -1e9))
        rows = jnp.arange(state.shape[0])
        symbol = jax.random.categorical(symbol_key, forward(params, state)[rows, position]).astype(jnp.int32)
        update = hidden[rows, position]
        state = state.at[rows, position].set(jnp.where(update, symbol, state[rows, position]))
        return state, order.at[:, step].set(jnp.where(update, position, -1)), state_key

    order = jnp.full(tokens.shape, -1, dtype=jnp.int32)
    state, order, _ = jax.lax.fori_loop(0, LENGTH, fill, (tokens, order, key))
    return state, order
