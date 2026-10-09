"""TraFL trajectory-balance loss (Ahmadi et al., arXiv 2605.13935), model-agnostic.

The prompt is a partially revealed string c; a completion y fills its masked positions U(c).
Residual: delta = log p_theta(y|c) - log p_ref(y|c) - beta * centered_reward + log Z(c),
loss = mean(delta^2). Completions are fixed data: gradients flow only through the
log-probability surrogate and log Z, never through which completions were sampled.
"""

from collections.abc import Callable
from typing import Literal, NamedTuple

import jax
import jax.numpy as jnp

from ...config import MASK

Forward = Callable[[object, jax.Array], jax.Array]  # (params, tokens (..., L)) -> logits (..., L, 4)
LogZ = Callable[[object, jax.Array], jax.Array]  # (params, contexts (C, L)) -> log Z (C,)
Normalization = Literal["paper", "elbo"]


class Reference(NamedTuple):
    forward: Forward
    params: object


def surrogate_log_prob(forward: Forward, params: object, contexts: jax.Array,
                       completions: jax.Array, key: jax.Array, samples: int = 4,
                       normalization: Normalization = "paper") -> jax.Array:
    """Monte Carlo masked-reconstruction estimate of log p(y | c), shape (C, G).

    For each sample: l ~ U{1..u}, mask l random positions of U(c), score them.
    "paper" weights the sum by 1/l (per-position average, Eq. 4 of the paper);
    "elbo" weights it by u/l, the random-order ELBO, a lower bound on log p(y | c).
    """
    hidden = jnp.broadcast_to(contexts[:, None, :] == MASK, completions.shape)
    count = hidden.sum(axis=-1)

    def one(sample_key: jax.Array) -> jax.Array:
        length_key, order_key = jax.random.split(sample_key)
        l = jax.random.randint(length_key, count.shape, 1, jnp.maximum(count, 1) + 1)
        scores = jnp.where(hidden, jax.random.uniform(order_key, hidden.shape), jnp.inf)
        rank = jnp.argsort(jnp.argsort(scores, axis=-1), axis=-1)
        masked = hidden & (rank < l[..., None])
        log_probs = jax.nn.log_softmax(forward(params, jnp.where(masked, MASK, completions)), axis=-1)
        chosen = jnp.take_along_axis(log_probs, completions[..., None], axis=-1)[..., 0]
        weight = (count if normalization == "elbo" else 1) / l
        return jnp.where(count > 0, weight * jnp.sum(jnp.where(masked, chosen, 0), axis=-1), 0)

    return jnp.mean(jax.vmap(one)(jax.random.split(key, samples)), axis=0)


def trafl_residuals(params: tuple[object, object], forward: Forward, log_z: LogZ,
                    contexts: jax.Array, completions: jax.Array, rewards: jax.Array,
                    key: jax.Array, beta: float | jax.Array, reference: Reference | None = None,
                    samples: int = 4, normalization: Normalization = "paper") -> jax.Array:
    """Trajectory-balance residuals delta, shape (C, G) (Algorithm 1 of the paper).

    params = (policy_params, log_z_params); contexts (C, L); completions (C, G, L)
    agree with contexts on revealed positions; rewards (C, G); beta is a scalar or a
    per-context array (C,). reference=None means a uniform p_ref, whose log-probability
    is constant and absorbed by log Z.
    """
    policy, z_params = params
    log_p = surrogate_log_prob(forward, policy, contexts, completions, key, samples, normalization)
    log_ref = 0.0 if reference is None else surrogate_log_prob(
        reference.forward, reference.params, contexts, completions, key, samples, normalization)
    centered = rewards - rewards.mean(axis=1, keepdims=True)
    beta = jnp.asarray(beta)
    beta = beta[:, None] if beta.ndim == 1 else beta
    return log_p - log_ref - beta * centered + log_z(z_params, contexts)[:, None]


def trafl_loss(*args: object, **kwargs: object) -> jax.Array:
    """Mean squared residual; same arguments as trafl_residuals."""
    return jnp.mean(trafl_residuals(*args, **kwargs) ** 2)
