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


def complete_proposal(forward: Forward, params: object, tokens: jax.Array, key: jax.Array,
                      temperature: float = 1.0) -> tuple[jax.Array, jax.Array]:
    """Low-confidence remasking (LLaDA / Fast-dLLM), one token per step: every hidden position proposes a letter at
    temperature T; the proposal with the highest untempered probability is committed (first maximum = lowest index on
    ties). Its exact step law is metrics.decoders.proposal_step_law (gate: tests/test_paths.py). TraFL's paper samples
    its training rollouts with this kind of decoder; per-string losses do not use the sampled order."""

    def fill(step: int, carry: tuple[jax.Array, jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array, jax.Array]:
        state, order, state_key = carry
        state_key, proposal_key = jax.random.split(state_key)
        logits = forward(params, state)
        hidden = state == MASK
        proposal = jax.random.categorical(proposal_key, logits / temperature).astype(jnp.int32)
        confidence = jnp.take_along_axis(jax.nn.softmax(logits, axis=-1), proposal[..., None], axis=-1)[..., 0]
        position = jnp.argmax(jnp.where(hidden, confidence, -1.0), axis=-1)
        rows = jnp.arange(state.shape[0])
        update = hidden[rows, position]
        state = state.at[rows, position].set(jnp.where(update, proposal[rows, position], state[rows, position]))
        return state, order.at[:, step].set(jnp.where(update, position, -1)), state_key

    order = jnp.full(tokens.shape, -1, dtype=jnp.int32)
    state, order, _ = jax.lax.fori_loop(0, tokens.shape[-1], fill, (tokens, order, key))
    return state, order
