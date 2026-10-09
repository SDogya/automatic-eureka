"""Model-agnostic samplers: the random-order generator (model.complete, with forward passed in) and JustGRPO's
left-to-right one."""

import jax
import jax.numpy as jnp

from ..config import MASK
from .losses.baselines import complete_ar  # noqa: F401  (re-exported: JustGRPO's sampler)
from .losses.trafl import Forward


def complete_uniform(forward: Forward, params: object, tokens: jax.Array, key: jax.Array) -> tuple[jax.Array, jax.Array]:
    """model.complete for any forward: one uniformly random hidden position per step, letter ~ q at T = 1.
    Same key consumption as model.complete, so with the MLP forward it reproduces it exactly."""

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
    state, order, _ = jax.lax.fori_loop(0, tokens.shape[-1], fill, (tokens, order, key))
    return state, order
