"""Exact target probabilities, local MH transitions, and Parquet I/O."""

from hashlib import sha256
from pathlib import Path
from typing import NamedTuple

import numpy as np

from .config import (
    LENGTH, Comparison, Config, Diagnostics, FloatArray, IntArray,
    Metadata, decode, encode, rng, software_versions, validate_tokens,
)


class Target(NamedTuple):
    tokens: IntArray
    distance: IntArray
    log_reward: FloatArray
    log_pi: FloatArray
    pi: FloatArray
    basins: FloatArray
    log_normalizer: float


def enumerate_tokens(base: int, length: int = LENGTH) -> IntArray:
    powers = base ** np.arange(length - 1, -1, -1, dtype=np.int32)
    return (np.arange(base**length, dtype=np.int32)[:, None] // powers) % base


def build_target(config: Config) -> Target:
    tokens = enumerate_tokens(4)
    distances = np.count_nonzero(
        tokens[:, None, :] != encode(config.modes)[None, :, :], axis=-1
    ).astype(np.int32)
    distance = distances.min(axis=1)
    log_reward = 1.0 - distance.astype(np.float64)
    log_weight = log_reward / config.temperature
    maximum = float(log_weight.max())
    log_normalizer = maximum + float(np.log(np.exp(log_weight - maximum).sum()))
    log_pi = log_weight - log_normalizer
    pi = np.exp(log_pi)
    if np.any(pi == 0):
        raise ValueError("Temperature is too small for a full-support float64 target")
    basins = (distances == distance[:, None]).astype(np.float64)
    basins /= basins.sum(axis=1, keepdims=True)
    return Target(tokens, distance, log_reward, log_pi, pi, basins, log_normalizer)


class Kernel(NamedTuple):
    neighbors: IntArray
    weights: FloatArray
    stay: FloatArray

    def forward(self, distribution: FloatArray) -> FloatArray:
        """Row distribution multiplied by P; scatter incoming probability mass."""
        return distribution * self.stay + np.bincount(
            self.neighbors.ravel(),
            weights=(distribution[:, None] * self.weights).ravel(),
            minlength=len(distribution),
        )

    def backward(self, features: FloatArray) -> FloatArray:
        """P applied to functions of the state."""
        return self.stay[:, None] * features + np.einsum(
            "ij,ijk->ik", self.weights, features[self.neighbors]
        )


def build_kernel(target: Target) -> Kernel:
    ids = np.arange(len(target.pi), dtype=np.int32)
    powers = 4 ** np.arange(LENGTH - 1, -1, -1, dtype=np.int32)
    neighbors = np.stack([
        ids + (((target.tokens[:, i] + delta) % 4) - target.tokens[:, i]) * powers[i]
        for i in range(LENGTH) for delta in (1, 2, 3)
    ], axis=1)
    weights = np.exp(np.minimum(target.log_pi[neighbors] - target.log_pi[:, None], 0)) / 24
    return Kernel(neighbors, weights, 1 - weights.sum(axis=1))


def diagnose(config: Config, target: Target) -> Diagnostics:
    kernel = build_kernel(target)
    reverse = np.exp(np.minimum(target.log_pi[:, None] - target.log_pi[kernel.neighbors], 0)) / 24
    balance_error = float(np.max(np.abs(
        target.pi[:, None] * kernel.weights - target.pi[kernel.neighbors] * reverse
    )))
    stationarity_error = float(np.max(np.abs(kernel.forward(target.pi) - target.pi)))
    if max(balance_error, stationarity_error) > 1e-12:
        raise ArithmeticError("MH kernel fails detailed balance or stationarity")

    distribution = np.full_like(target.pi, 1 / len(target.pi))
    burn_steps = 0
    tv = float(np.abs(distribution - target.pi).sum() / 2)
    while tv > config.burn_tv:
        burn_steps += 1
        if burn_steps > config.diagnostic_limit:
            raise RuntimeError("Burn-in criterion not reached within diagnostic_limit")
        distribution = kernel.forward(distribution)
        tv = float(np.abs(distribution - target.pi).sum() / 2)

    features = np.column_stack((target.log_reward, target.basins))
    features -= target.pi @ features
    variance = target.pi @ (features * features)
    evolved = features.copy()
    bounds = np.ones(5, dtype=np.float64)
    save_every = 0
    while float(bounds.max()) > config.correlation_bound:
        save_every += 1
        if save_every > config.diagnostic_limit:
            raise RuntimeError("Correlation criterion not reached within diagnostic_limit")
        evolved = kernel.backward(evolved)
        bounds = np.sqrt((target.pi @ (evolved * evolved)) / variance)
    return Diagnostics(
        burn_steps=burn_steps, save_every=save_every, burn_tv=tv,
        correlation_bounds=tuple(float(v) for v in bounds),
        stationarity_error=stationarity_error, detailed_balance_error=balance_error,
        expected_acceptance=float(target.pi @ kernel.weights.sum(axis=1)),
    )


def token_ids(tokens: IntArray) -> IntArray:
    return tokens @ (4 ** np.arange(LENGTH - 1, -1, -1, dtype=np.int32))


def compare(ids: IntArray, target: Target) -> Comparison:
    frequencies = np.bincount(ids, minlength=len(target.pi)) / len(ids)
    histogram = np.bincount(target.distance[ids], minlength=LENGTH + 1) / len(ids)
    exact_histogram = np.bincount(target.distance, weights=target.pi, minlength=LENGTH + 1)
    return Comparison(
        reward_tv=float(np.abs(histogram - exact_histogram).sum() / 2),
        full_tv=float(np.abs(frequencies - target.pi).sum() / 2),
        basin_means=tuple(float(v) for v in target.basins[ids].mean(axis=0)),
        mean_log_reward=float(target.log_reward[ids].mean()),
    )


def fingerprint(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def generate(config: Config, overwrite: bool = False) -> Metadata:
    # These dependencies are imported only when the installed environment is needed.
    import jax
    import jax.numpy as jnp
    import pyarrow as pa
    import pyarrow.parquet as pq

    from .plots import plot_reward

    path = config.data_dir / "data8.parquet"
    metadata_path = config.data_dir / "metadata.json"
    if not overwrite and (path.exists() or metadata_path.exists()):
        raise FileExistsError("Dataset exists; use --overwrite to regenerate it")
    target = build_target(config)
    diagnostics = diagnose(config, target)
    print(f"MH: burn={diagnostics.burn_steps}, interval={diagnostics.save_every}", flush=True)
    modes = jnp.asarray(encode(config.modes))
    key = jax.random.fold_in(jax.random.key(config.seed), 0)
    key, initial_key = jax.random.split(key)
    initial = jax.random.randint(initial_key, (LENGTH,), 0, 4)

    def step(carry: tuple[jax.Array, jax.Array, jax.Array], _: None
             ) -> tuple[tuple[jax.Array, jax.Array, jax.Array], None]:
        state, state_key, accepted = carry
        state_key, position_key, delta_key, uniform_key = jax.random.split(state_key, 4)
        position = jax.random.randint(position_key, (), 0, LENGTH)
        delta = jax.random.randint(delta_key, (), 1, 4)
        proposal = state.at[position].set((state[position] + delta) % 4)
        old_distance = jnp.min(jnp.sum(state != modes, axis=1))
        new_distance = jnp.min(jnp.sum(proposal != modes, axis=1))
        log_acceptance = jnp.minimum((old_distance - new_distance) / config.temperature, 0)
        accept = jnp.log(jax.random.uniform(uniform_key)) < log_acceptance
        return (jnp.where(accept, proposal, state), state_key, accepted + accept), None

    def collect(carry: tuple[jax.Array, jax.Array, jax.Array], _: None
                ) -> tuple[tuple[jax.Array, jax.Array, jax.Array], jax.Array]:
        carry, _ = jax.lax.scan(step, carry, None, length=diagnostics.save_every)
        return carry, carry[0]

    @jax.jit
    def simulate(state: jax.Array, state_key: jax.Array) -> tuple[jax.Array, jax.Array]:
        carry = (state, state_key, jnp.array(0, dtype=jnp.int32))
        carry, _ = jax.lax.scan(step, carry, None, length=diagnostics.burn_steps)
        carry = (carry[0], carry[1], jnp.array(0, dtype=jnp.int32))
        carry, samples = jax.lax.scan(collect, carry, None, length=config.samples)
        return samples, carry[2]

    samples, accepted = simulate(initial, key)
    tokens = np.asarray(samples, dtype=np.int32)
    ids = token_ids(tokens)
    exact_ids = rng(config.seed, 1).choice(len(target.pi), size=config.samples, p=target.pi).astype(np.int32)
    steps = diagnostics.burn_steps + diagnostics.save_every * np.arange(1, config.samples + 1)
    schema = pa.schema([
        ("sequence", pa.string()), ("distance", pa.int32()),
        ("reward", pa.float64()), ("log_reward", pa.float64()), ("step", pa.int64()),
    ])
    table = pa.Table.from_pydict({
        "sequence": decode(tokens), "distance": target.distance[ids],
        "reward": np.exp(target.log_reward[ids]), "log_reward": target.log_reward[ids],
        "step": steps,
    }, schema=schema)
    config.data_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
    metadata = Metadata(
        config=config, software=software_versions(), diagnostics=diagnostics,
        acceptance_rate=float(accepted) / (config.samples * diagnostics.save_every),
        target_log_normalizer=target.log_normalizer,
        target_basin_means=tuple(float(v) for v in target.pi @ target.basins),
        mcmc=compare(ids, target), exact_sample=compare(exact_ids, target),
        dataset_sha256=fingerprint(path),
    )
    metadata.save(metadata_path)
    plot_reward(config, target, ids, exact_ids)
    return metadata


def load_dataset(config: Config) -> tuple[IntArray, Metadata]:
    import pyarrow.parquet as pq

    path = config.data_dir / "data8.parquet"
    metadata = Metadata.model_validate_json((config.data_dir / "metadata.json").read_text())
    fields = ("modes", "temperature", "seed", "samples", "burn_tv", "correlation_bound", "diagnostic_limit")
    if any(getattr(config, field) != getattr(metadata.config, field) for field in fields):
        raise ValueError("Dataset metadata does not match the generation configuration")
    if fingerprint(path) != metadata.dataset_sha256:
        raise ValueError("Dataset checksum does not match metadata")
    strings = pq.read_table(path, columns=["sequence"]).column("sequence").to_pylist()
    tokens = encode(strings)
    validate_tokens(tokens)
    if len(tokens) != config.samples:
        raise ValueError("Unexpected dataset length")
    return tokens, metadata


def save_model_distribution(config: Config, target: Target,
                            log_probs: FloatArray, epoch: int) -> None:
    """Export the exact best-checkpoint distribution in target enumeration order."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    if log_probs.shape != target.pi.shape or not np.isfinite(log_probs).all():
        raise ValueError("Invalid model log-probability vector")
    probabilities = np.exp(log_probs)
    if not np.isclose(probabilities.sum(), 1, rtol=0, atol=1e-10):
        raise ValueError("Model distribution is not normalized")
    table = pa.Table.from_pydict({
        "sequence": decode(target.tokens),
        "target_probability": target.pi,
        "model_probability": probabilities,
        "model_log_probability": log_probs,
        "log_reward": target.log_reward,
    }).replace_schema_metadata({b"checkpoint": b"best.npz", b"epoch": str(epoch).encode()})
    config.data_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, config.data_dir / "best_distribution.parquet")
