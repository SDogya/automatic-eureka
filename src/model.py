"""A 13,800-parameter JAX MLP and random-order masked generation."""

from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .config import ALPHABET, LENGTH, MASK, IntArray, SampleRequest, validate_tokens


class Layer(NamedTuple):
    weight: jax.Array
    bias: jax.Array


Params = tuple[Layer, Layer, Layer]
SHAPES = ((LENGTH * (MASK + 1), 80), (80, 80), (80, LENGTH * len(ALPHABET)))


def init_parameters(key: jax.Array) -> Params:
    keys = jax.random.split(key, 3)

    def layer(shape: tuple[int, int], layer_key: jax.Array) -> Layer:
        limit = np.sqrt(6 / sum(shape))
        weight = jax.random.uniform(layer_key, shape, dtype=jnp.float32, minval=-limit, maxval=limit)
        return Layer(weight, jnp.zeros(shape[1], dtype=jnp.float32))

    return (layer(SHAPES[0], keys[0]), layer(SHAPES[1], keys[1]), layer(SHAPES[2], keys[2]))


def forward(params: Params, tokens: jax.Array) -> jax.Array:
    hidden = jax.nn.one_hot(tokens, MASK + 1, dtype=jnp.float32).reshape((*tokens.shape[:-1], LENGTH * (MASK + 1)))
    for layer in params[:-1]:
        hidden = jax.nn.gelu(hidden @ layer.weight + layer.bias, approximate=False)
    logits = hidden @ params[-1].weight + params[-1].bias
    return logits.reshape((*tokens.shape[:-1], LENGTH, len(ALPHABET)))


def masked_loss(params: Params, tokens: jax.Array, mask: jax.Array) -> jax.Array:
    logits = forward(params, jnp.where(mask, MASK, tokens))
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    nll = -jnp.take_along_axis(log_probs, tokens[..., None], axis=-1)[..., 0]
    return jnp.mean(jnp.sum(jnp.where(mask, nll, 0), axis=-1))


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
        for i, shape in enumerate(SHAPES):
            weight, bias = saved[f"w{i}"], saved[f"b{i}"]
            if weight.shape != shape or bias.shape != (shape[1],):
                raise ValueError("Invalid checkpoint shapes")
            if any(a.dtype != np.float32 or not np.isfinite(a).all() for a in (weight, bias)):
                raise ValueError("Checkpoint must contain finite float32 parameters")
            layers.append(Layer(jnp.asarray(weight), jnp.asarray(bias)))
    return layers[0], layers[1], layers[2]


def sample(params: Params, request: SampleRequest) -> IntArray:
    context = [MASK if c == "?" else ALPHABET.index(c) for c in request.context]
    initial = jnp.tile(jnp.array(context, dtype=jnp.int32), (request.count, 1))
    steps = request.context.count("?")
    key = jax.random.fold_in(jax.random.key(request.seed), 4)

    @jax.jit
    def complete(tokens: jax.Array, initial_key: jax.Array) -> jax.Array:
        def fill(_: int, carry: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array]:
            state, state_key = carry
            state_key, position_key, symbol_key = jax.random.split(state_key, 3)
            position = jax.random.categorical(position_key, jnp.where(state == MASK, 0.0, -jnp.inf))
            rows = jnp.arange(request.count)
            logits = forward(params, state)[rows, position]
            symbol = jax.random.categorical(symbol_key, logits).astype(jnp.int32)
            return state.at[rows, position].set(symbol), state_key

        return jax.lax.fori_loop(0, steps, fill, (tokens, initial_key))[0]

    tokens = np.asarray(complete(initial, key), dtype=np.int32)
    validate_tokens(tokens)
    return tokens
