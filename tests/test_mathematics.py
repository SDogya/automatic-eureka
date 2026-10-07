"""Independent checks of MH and the exact random-order probability calculation."""

from itertools import permutations, product
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from src.config import LENGTH, Config, Metric, SampleRequest, TokenBatch, load_config
from src.data import build_kernel, build_target
from src.evaluation import bayes_masked_nll, build_contexts, expected_masked_loss, joint_log_probs


def test_validated_configuration() -> None:
    assert load_config(Path("config.json")) == Config()
    for fields in ({"seed": -1}, {"temperature": 0.0}, {"unknown": 1},
                   {"modes": ("A" * LENGTH,) * 4}):
        with pytest.raises(ValidationError):
            Config.model_validate(fields)
    with pytest.raises(ValidationError):
        SampleRequest(context="ACGT???X")
    with pytest.raises(ValidationError):
        Metric(epoch=0, training_loss=float("nan"))
    for tokens in (np.zeros((2, LENGTH), dtype=np.float64),
                   np.zeros((2, LENGTH - 1), dtype=np.int32),
                   np.full((2, LENGTH), 4, dtype=np.int32)):
        with pytest.raises(ValidationError):
            TokenBatch(tokens=tokens)


def test_local_mh_kernel_and_target() -> None:
    target = build_target(Config())
    kernel = build_kernel(target)
    assert len(target.pi) == 4**LENGTH
    np.testing.assert_allclose(target.pi.sum(), 1, rtol=0, atol=1e-14)
    assert np.all(kernel.weights > 0)
    assert np.all(kernel.stay >= -1e-15)
    np.testing.assert_allclose(kernel.forward(target.pi), target.pi, rtol=1e-12, atol=1e-15)
    for column in range(kernel.neighbors.shape[1]):
        neighbors = kernel.neighbors[:, column]
        assert np.all(np.count_nonzero(target.tokens != target.tokens[neighbors], axis=-1) == 1)
        reverse = np.minimum(1, target.pi / target.pi[neighbors]) / (3 * LENGTH)
        np.testing.assert_allclose(target.pi * kernel.weights[:, column],
                                   target.pi[neighbors] * reverse, rtol=1e-12, atol=1e-15)
    # Independent reference for the blockwise application of P to functions.
    features = np.column_stack((target.log_reward, target.basins))
    expected = np.empty_like(features)
    for feature in range(features.shape[1]):
        expected[:, feature] = (kernel.stay * features[:, feature]
            + np.sum(kernel.weights * features[kernel.neighbors, feature], axis=1))
    np.testing.assert_allclose(kernel.backward(features), expected, rtol=1e-12, atol=1e-15)


def test_dp_matches_enumerating_all_reveal_orders() -> None:
    n, q = 3, 2
    contexts = build_contexts(n, q)
    generator = np.random.default_rng(71)
    probabilities = generator.dirichlet(np.ones(q), size=(len(contexts.tokens), n))
    computed = np.exp(joint_log_probs(contexts, np.log(probabilities)))
    brute: list[float] = []
    for sequence in product(range(q), repeat=n):
        paths: list[float] = []
        for order in permutations(range(n)):
            state, value = [q] * n, 1.0
            for position in order:
                context_id = int(np.array(state) @ contexts.powers)
                value *= probabilities[context_id, position, sequence[position]]
                state[position] = sequence[position]
            paths.append(value)
        brute.append(float(np.mean(paths)))
    np.testing.assert_allclose(computed, brute, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(computed.sum(), 1, rtol=0, atol=1e-14)


def test_oracle_conditionals_recover_joint_target() -> None:
    n, q = 3, 2
    contexts = build_contexts(n, q)
    full = contexts.tokens[contexts.full_ids]
    pi = np.random.default_rng(19).dirichlet(np.ones(len(full)))
    log_probs = np.zeros((len(contexts.tokens), n, q), dtype=np.float64)
    for context_id, state in enumerate(contexts.tokens):
        compatible = np.all((state == q) | (full == state), axis=1)
        for position in np.flatnonzero(state == q):
            for symbol in range(q):
                log_probs[context_id, position, symbol] = np.log(
                    pi[compatible & (full[:, position] == symbol)].sum() / pi[compatible].sum()
                )
    log_p = joint_log_probs(contexts, log_probs)
    np.testing.assert_allclose(np.exp(log_p), pi, rtol=1e-12, atol=1e-15)
    assert abs(float(pi @ (np.log(pi) - log_p))) < 1e-14
    baseline = sum(float(weight) * expected_masked_loss(contexts, log_probs, full[i:i+1], 0.7)
                   for i, weight in enumerate(pi))
    assert bayes_masked_nll(contexts, pi, 0.7) == pytest.approx(baseline, abs=1e-14)
    uniform = np.full(len(full), 1 / len(full))
    assert bayes_masked_nll(contexts, uniform, 0.7) == pytest.approx(n * 0.7 * np.log(q))


def test_mask_expectation_matches_explicit_sum() -> None:
    n, q = 3, 2
    contexts = build_contexts(n, q)
    probabilities = np.random.default_rng(3).dirichlet(np.ones(q), size=(len(contexts.tokens), n))
    sequences = contexts.tokens[contexts.full_ids[:3]]
    p = 0.7
    total = 0.0
    for sequence in sequences:
        for mask in product((False, True), repeat=n):
            state = [q if hidden else int(c) for c, hidden in zip(sequence, mask)]
            context_id = int(np.array(state) @ contexts.powers)
            count = sum(mask)
            loss = sum(-np.log(probabilities[context_id, i, c])
                       for i, (c, hidden) in enumerate(zip(sequence, mask)) if hidden)
            total += p**count * (1-p)**(n-count) * loss / len(sequences)
    assert expected_masked_loss(contexts, np.log(probabilities), sequences, p) == pytest.approx(total)
    uniform = np.full_like(probabilities, -np.log(q))
    assert expected_masked_loss(contexts, uniform, sequences, p) == pytest.approx(n * p * np.log(q))
    assert expected_masked_loss(contexts, uniform, sequences, 1.0) == pytest.approx(n * np.log(q))


def test_blockwise_dp_for_uniform_conditionals() -> None:
    # The largest layer exceeds 65,536 rows, exercising multiple DP blocks.
    contexts = build_contexts(12, 2)
    conditionals = np.full((len(contexts.tokens), 12, 2), -np.log(2), dtype=np.float64)
    log_p = joint_log_probs(contexts, conditionals)
    np.testing.assert_allclose(log_p, -12 * np.log(2), rtol=0, atol=1e-13)
    assert np.exp(log_p).sum() == pytest.approx(1, abs=1e-13)
