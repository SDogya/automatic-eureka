"""A bidirectional masked-diffusion transformer denoiser in pure JAX.

Same interface as the MLP in model.py: ``forward(params, tokens) -> logits`` with tokens
(..., LENGTH) in {0..3 letters, MASK = 4} and logits (..., LENGTH, 4) for every position.

With token embedding E in R^{(MASK+1) x d}, learned position embedding P in R^{LENGTH x d}
and N pre-norm encoder layers,

    x_0       = E[tokens] + P
    x_{n+1/2} = x_n + MHA(LN(x_n))        full attention over all LENGTH positions, no causal mask
    x_{n+1}   = x_{n+1/2} + FF(LN(x_{n+1/2})),   FF(z) = gelu(z W_1 + b_1) W_2 + b_2   (exact GELU)
    logits    = LN(x_N) W_head + b_head

MHA splits d into `heads` heads of size d/heads: a_h = softmax(q_h k_h^T / sqrt(d/heads)) and the
concatenated heads a_h v_h are mixed by one d x d projection. No dropout. LayerNorm has eps 1e-5
(the torch default) and a biased variance.

The attention weights are stored with explicit head axes, so ``forward(params, tokens)`` needs
no hyperparameters besides the arrays: d, layers, heads and ff are all read from their shapes.
"""

from collections.abc import Mapping
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .config import ALPHABET, LENGTH, MASK

LAYER_NORM_EPS = 1e-5
EMBEDDING_STD = 0.02


class Block(NamedTuple):
    """One pre-norm encoder layer."""

    ln1_scale: jax.Array  # (d,)
    ln1_bias: jax.Array  # (d,)
    qkv_weight: jax.Array  # (d, 3, heads, d / heads): axis 1 is query / key / value
    qkv_bias: jax.Array  # (3, heads, d / heads)
    proj_weight: jax.Array  # (heads, d / heads, d)
    proj_bias: jax.Array  # (d,)
    ln2_scale: jax.Array  # (d,)
    ln2_bias: jax.Array  # (d,)
    ff_in_weight: jax.Array  # (d, ff)
    ff_in_bias: jax.Array  # (ff,)
    ff_out_weight: jax.Array  # (ff, d)
    ff_out_bias: jax.Array  # (d,)


class Params(NamedTuple):
    token_embedding: jax.Array  # (MASK + 1, d)
    position_embedding: jax.Array  # (LENGTH, d)
    blocks: tuple[Block, ...]
    final_scale: jax.Array  # (d,)
    final_bias: jax.Array  # (d,)
    head_weight: jax.Array  # (d, 4)
    head_bias: jax.Array  # (4,)


def check_dimensions(d: int, layers: int, heads: int, ff: int) -> None:
    if min(d, layers, heads, ff) < 1:
        raise ValueError("d, layers, heads and ff must be positive")
    if d % heads:
        raise ValueError(f"d = {d} must be divisible by heads = {heads}")


def parameter_shapes(d: int, layers: int, heads: int, ff: int) -> dict[str, tuple[int, ...]]:
    """Name -> shape of every array, in checkpoint order. The one source for init and loading."""
    check_dimensions(d, layers, heads, ff)
    block = {
        "ln1_scale": (d,), "ln1_bias": (d,),
        "qkv_weight": (d, 3, heads, d // heads), "qkv_bias": (3, heads, d // heads),
        "proj_weight": (heads, d // heads, d), "proj_bias": (d,),
        "ln2_scale": (d,), "ln2_bias": (d,),
        "ff_in_weight": (d, ff), "ff_in_bias": (ff,),
        "ff_out_weight": (ff, d), "ff_out_bias": (d,),
    }
    if tuple(block) != Block._fields:
        raise ArithmeticError("Block shapes out of sync with Block fields")
    shapes: dict[str, tuple[int, ...]] = {
        "token_embedding": (MASK + 1, d), "position_embedding": (LENGTH, d),
    }
    for layer in range(layers):
        shapes.update({f"blocks.{layer}.{name}": shape for name, shape in block.items()})
    shapes.update({
        "final_scale": (d,), "final_bias": (d,),
        "head_weight": (d, len(ALPHABET)), "head_bias": (len(ALPHABET),),
    })
    return shapes


def expected_parameter_count(d: int, layers: int, heads: int, ff: int) -> int:
    """Closed form, written from the architecture and not from parameter_shapes.

    Embeddings (MASK + 1 + LENGTH) d; per layer attention 4d^2 + 4d (qkv 3d^2 + 3d, projection
    d^2 + d), two LayerNorms 4d, feed-forward 2 d ff + ff + d; final LayerNorm 2d; head 4d + 4.
    The number of heads does not change the count.
    """
    check_dimensions(d, layers, heads, ff)
    per_layer = 4 * d * d + 4 * d + 4 * d + 2 * d * ff + ff + d
    return (MASK + 1 + LENGTH) * d + layers * per_layer + 2 * d + len(ALPHABET) * (d + 1)


def to_arrays(params: Params) -> dict[str, jax.Array]:
    """Flatten to {name: array} with the names of parameter_shapes."""
    arrays: dict[str, jax.Array] = {
        "token_embedding": params.token_embedding, "position_embedding": params.position_embedding,
    }
    for layer, block in enumerate(params.blocks):
        arrays.update({f"blocks.{layer}.{name}": array for name, array in block._asdict().items()})
    arrays.update(
        final_scale=params.final_scale, final_bias=params.final_bias,
        head_weight=params.head_weight, head_bias=params.head_bias,
    )
    return arrays


def from_arrays(arrays: Mapping[str, jax.Array], layers: int) -> Params:
    """Inverse of to_arrays."""
    return Params(
        arrays["token_embedding"], arrays["position_embedding"],
        tuple(Block(**{name: arrays[f"blocks.{layer}.{name}"] for name in Block._fields})
              for layer in range(layers)),
        arrays["final_scale"], arrays["final_bias"], arrays["head_weight"], arrays["head_bias"],
    )


def dimensions(params: Params) -> tuple[int, int, int, int]:
    """(d, layers, heads, ff) read from the array shapes."""
    first = params.blocks[0]
    return (params.token_embedding.shape[1], len(params.blocks),
            first.qkv_weight.shape[2], first.ff_in_weight.shape[1])


def init_parameters(key: jax.Array, d: int, layers: int, heads: int, ff: int) -> Params:
    """Glorot-uniform weights, zero biases, unit LayerNorm scales, N(0, 0.02^2) embeddings."""
    shapes = parameter_shapes(d, layers, heads, ff)
    keys = jax.random.split(key, len(shapes))
    arrays: dict[str, jax.Array] = {}
    for array_key, (name, shape) in zip(keys, shapes.items()):
        kind = name.rsplit(".", 1)[-1]
        if kind.endswith("_embedding"):
            array = EMBEDDING_STD * jax.random.normal(array_key, shape, dtype=jnp.float32)
        elif kind.endswith("_scale"):
            array = jnp.ones(shape, dtype=jnp.float32)
        elif kind.endswith("_bias"):
            array = jnp.zeros(shape, dtype=jnp.float32)
        else:
            # Glorot limit sqrt(6 / (fan_in + fan_out)), taking the head axes with the matrix
            # side they belong to: qkv is d -> 3d and the projection is d -> d.
            fan_in = int(np.prod(shape[:2])) if kind == "proj_weight" else shape[0]
            fan_out = shape[2] if kind == "proj_weight" else int(np.prod(shape[1:]))
            limit = np.sqrt(6 / (fan_in + fan_out))
            array = jax.random.uniform(array_key, shape, dtype=jnp.float32, minval=-limit, maxval=limit)
        arrays[name] = array
    return from_arrays(arrays, layers)


def layer_norm(x: jax.Array, scale: jax.Array, bias: jax.Array) -> jax.Array:
    centered = x - jnp.mean(x, axis=-1, keepdims=True)
    variance = jnp.mean(centered * centered, axis=-1, keepdims=True)
    return centered * jax.lax.rsqrt(variance + LAYER_NORM_EPS) * scale + bias


def attention(block: Block, x: jax.Array) -> jax.Array:
    """Full bidirectional multi-head self-attention on (..., LENGTH, d)."""
    qkv = jnp.einsum("...ld,dthk->...lthk", x, block.qkv_weight) + block.qkv_bias
    query, key, value = (qkv[..., i, :, :] for i in range(3))
    head_size = block.qkv_weight.shape[-1]
    scores = jnp.einsum("...qhk,...phk->...hqp", query, key) * float(head_size) ** -0.5
    weights = jax.nn.softmax(scores, axis=-1)
    context = jnp.einsum("...hqp,...phk->...qhk", weights, value)
    return jnp.einsum("...qhk,hkd->...qd", context, block.proj_weight) + block.proj_bias


def feed_forward(block: Block, x: jax.Array) -> jax.Array:
    hidden = jax.nn.gelu(x @ block.ff_in_weight + block.ff_in_bias, approximate=False)
    return hidden @ block.ff_out_weight + block.ff_out_bias


def forward(params: Params, tokens: jax.Array) -> jax.Array:
    """Logits (..., LENGTH, 4) for tokens (..., LENGTH); a token outside 0..MASK gives NaN."""
    tokens = jnp.asarray(tokens)
    if tokens.shape[-1] != LENGTH:
        raise ValueError(f"Tokens must have trailing dimension {LENGTH}, got {tokens.shape}")
    # mode="fill" turns an out-of-range token into NaN instead of a silently clamped embedding.
    x = jnp.take(params.token_embedding, tokens, axis=0, mode="fill", fill_value=jnp.nan)
    x = x + params.position_embedding
    for block in params.blocks:
        x = x + attention(block, layer_norm(x, block.ln1_scale, block.ln1_bias))
        x = x + feed_forward(block, layer_norm(x, block.ln2_scale, block.ln2_bias))
    return layer_norm(x, params.final_scale, params.final_bias) @ params.head_weight + params.head_bias
