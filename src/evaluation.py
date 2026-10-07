"""Exact random-order joint probabilities and expected masked validation NLL."""

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from .config import LENGTH, MASK, FloatArray, IntArray
from .data import Target, enumerate_tokens

if TYPE_CHECKING:
    from .model import Params


class Contexts(NamedTuple):
    tokens: IntArray
    known: IntArray
    powers: IntArray
    full_ids: IntArray


def build_contexts(length: int = LENGTH, alphabet_size: int = 4) -> Contexts:
    base = alphabet_size + 1
    tokens = enumerate_tokens(base, length)
    known = np.count_nonzero(tokens != alphabet_size, axis=1).astype(np.int32)
    powers = base ** np.arange(length - 1, -1, -1, dtype=np.int32)
    full_ids = enumerate_tokens(alphabet_size, length) @ powers
    return Contexts(tokens, known, powers, full_ids)


def joint_log_probs(contexts: Contexts, log_conditionals: FloatArray) -> FloatArray:
    """Probability of reaching a full state, summing every reveal-order path."""
    length = contexts.tokens.shape[1]
    mask = log_conditionals.shape[-1]
    log_mass = np.full(len(contexts.tokens), -np.inf, dtype=np.float64)
    log_mass[-1] = 0.0  # The all-mask state is the last base-(q+1) code.
    for count in range(1, length + 1):
        ids = np.flatnonzero(contexts.known == count)
        for start in range(0, len(ids), 65536):
            children_ids = ids[start:start + 65536]
            tokens = contexts.tokens[children_ids]
            contributions = np.full((len(children_ids), length), -np.inf, dtype=np.float64)
            for position in range(length):
                selected = np.flatnonzero(tokens[:, position] != mask)
                children = children_ids[selected]
                symbols = tokens[selected, position]
                parents = children + (mask - symbols) * contexts.powers[position]
                contributions[selected, position] = (
                    log_mass[parents] + log_conditionals[parents, position, symbols]
                    - np.log(length - count + 1)
                )
            log_mass[children_ids] = np.logaddexp.reduce(contributions, axis=1)
    return log_mass[contexts.full_ids]


def expected_masked_loss(
    contexts: Contexts, log_conditionals: FloatArray, tokens: IntArray,
    probability: float = 0.5,
) -> float:
    """Average over examples and all 2^n independent Bernoulli mask patterns."""
    length = tokens.shape[1]
    mask = log_conditionals.shape[-1]
    patterns = ((np.arange(2**length)[:, None] >> np.arange(length)) & 1).astype(bool)
    counts = patterns.sum(axis=1)
    weights = probability**counts * (1 - probability)**(length - counts)
    context_ids = np.where(patterns[None, :, :], mask, tokens[:, None, :]) @ contexts.powers
    losses = np.zeros(context_ids.shape, dtype=np.float64)
    for position in range(length):
        losses -= np.where(
            patterns[None, :, position],
            log_conditionals[context_ids, position, tokens[:, None, position]], 0,
        )
    return float(np.mean(losses @ weights))


def bayes_masked_nll(contexts: Contexts, probabilities: FloatArray,
                     mask_probability: float) -> float:
    """Population Bayes risk: sum of true conditional entropies over masks."""
    length = contexts.tokens.shape[1]
    alphabet_size = int(contexts.tokens.max())
    marginal = np.zeros(len(contexts.tokens), dtype=np.float64)
    marginal[contexts.full_ids] = probabilities
    symbols = np.arange(alphabet_size)
    for known in range(length - 1, -1, -1):
        ids = np.flatnonzero(contexts.known == known)
        position = np.argmax(contexts.tokens[ids] == alphabet_size, axis=1)
        children = ids[:, None] - (alphabet_size - symbols)[None, :] * contexts.powers[position, None]
        marginal[ids] = marginal[children].sum(axis=1)
    risk = 0.0
    for position in range(length):
        ids = np.flatnonzero(contexts.tokens[:, position] == alphabet_size)
        children = ids[:, None] - (alphabet_size - symbols)[None, :] * contexts.powers[position]
        conditional = marginal[children] / marginal[ids, None]
        entropy = -(conditional * np.log(conditional)).sum(axis=1)
        weights = mask_probability**(length - contexts.known[ids]) * (1 - mask_probability)**contexts.known[ids]
        risk += float((weights * marginal[ids]) @ entropy)
    return risk


class Evaluation(NamedTuple):
    validation_loss: float
    kl: float
    probability_sum: float
    log_probs: FloatArray


def evaluate(
    params: "Params", contexts: Contexts, target: Target, validation: IntArray,
    mask_probability: float, batch_size: int, scratch_dir: Path = Path(".scratch"),
) -> Evaluation:
    import jax
    import jax.numpy as jnp

    from .model import forward

    predict = jax.jit(forward)
    scratch_dir.mkdir(parents=True, exist_ok=True)
    # Disk-backed float64 table: 3.125 GB for length 10, cleaned after evaluation.
    with TemporaryDirectory(prefix="conditionals-", dir=scratch_dir) as temporary:
        log_conditionals = np.memmap(Path(temporary) / "log_conditionals.bin", mode="w+",
                                    shape=(len(contexts.tokens), LENGTH, MASK), dtype=np.float64)
        try:
            ids = np.flatnonzero(contexts.known < LENGTH)
            for start in range(0, len(ids), batch_size):
                batch_ids = ids[start:start + batch_size]
                logits = np.array(predict(params, jnp.asarray(contexts.tokens[batch_ids])), dtype=np.float64, copy=True)
                if not np.isfinite(logits).all():
                    raise FloatingPointError("Model produced non-finite logits")
                logits -= logits.max(axis=-1, keepdims=True)
                log_conditionals[batch_ids] = logits - np.log(np.exp(logits).sum(axis=-1, keepdims=True))
            log_p = joint_log_probs(contexts, log_conditionals)
            probability_sum = float(np.exp(log_p).sum())
            if not np.isclose(probability_sum, 1, rtol=0, atol=1e-10):
                raise ArithmeticError(f"Joint distribution is not normalized: {probability_sum}")
            kl = float(target.pi @ (target.log_pi - log_p))
            if kl < -1e-10 or not np.isfinite(kl):
                raise ArithmeticError(f"Invalid KL divergence: {kl}")
            return Evaluation(
                expected_masked_loss(contexts, log_conditionals, validation, mask_probability),
                kl, probability_sum, log_p,
            )
        finally:
            log_conditionals._mmap.close()
