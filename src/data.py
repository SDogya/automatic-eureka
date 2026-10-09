"""Exact target probabilities, local MH transitions, and Parquet I/O."""

from hashlib import sha256
from pathlib import Path
from typing import NamedTuple

import numpy as np

from .config import (
    LENGTH, TFBIND8_PATH, Comparison, Config, Diagnostics, FloatArray, IntArray,
    Metadata, decode, encode, rng, software_versions, validate_tokens,
)


class Target(NamedTuple):
    tokens: IntArray
    modes: tuple[str, str, str, str]
    distance: IntArray
    bin_edges: FloatArray
    reward_bin: IntArray
    log_reward: FloatArray
    log_pi: FloatArray
    pi: FloatArray
    basins: FloatArray
    log_normalizer: float


def enumerate_tokens(base: int, length: int = LENGTH) -> IntArray:
    powers = base ** np.arange(length - 1, -1, -1, dtype=np.int32)
    ids = np.arange(base**length, dtype=np.int32)
    tokens = np.empty((len(ids), length), dtype=np.int32)
    for position, power in enumerate(powers):
        tokens[:, position] = (ids // power) % base
    return tokens


def couplings(config: Config) -> FloatArray:
    """Potts couplings J[i, j, a, b] ~ N(0, 1) for positions i < j, zero otherwise."""
    upper = np.triu(np.ones((LENGTH, LENGTH), dtype=bool), k=1)
    normal = rng(config.seed, 5).standard_normal((LENGTH, LENGTH, 4, 4))
    return np.where(upper[:, :, None, None], normal, 0.0)


def potts_log_reward(tokens: IntArray, coupling: FloatArray, beta: float) -> FloatArray:
    """beta * sum_{i<j} J_ij(x_i, x_j), the quadratic form of the one-hot encoding."""
    energy = np.zeros(len(tokens), dtype=np.float64)
    for i in range(LENGTH):
        for j in range(i + 1, LENGTH):
            energy += coupling[i, j][tokens[:, i], tokens[:, j]]
    return beta * energy


def tfbind8_scores() -> FloatArray:
    """TFBind8 binding scores in [0, 1], aligned with the lexicographic enumeration."""
    import pyarrow.parquet as pq

    table = pq.read_table(TFBIND8_PATH)
    if table.column("sequence").to_pylist() != decode(enumerate_tokens(4)):
        raise ValueError("TFBind8 table is not in lexicographic order")
    return table.column("reward").to_numpy().astype(np.float64)


def local_maxima(log_reward: FloatArray, count: int) -> tuple[str, ...]:
    """The highest strings that beat every single-letter substitution."""
    neighbors = build_kernel_neighbors(enumerate_tokens(4))
    is_maximum = np.all(log_reward[:, None] > log_reward[neighbors], axis=1)
    ids = np.flatnonzero(is_maximum)
    return tuple(decode(enumerate_tokens(4)[ids[np.argsort(-log_reward[ids])][:count]]))


def build_target(config: Config) -> Target:
    tokens = enumerate_tokens(4)
    if config.reward == "potts":
        log_reward = potts_log_reward(tokens, couplings(config), config.beta)
    else:
        # Both terms standardized over all strings (mean 0, variance 1), then scaled by beta.
        potts = potts_log_reward(tokens, couplings(config), 1.0)
        score = tfbind8_scores()
        log_reward = config.beta * ((potts - potts.mean()) / potts.std()
                                    + (score - score.mean()) / score.std())
    # Reference modes for diagnostics and plots: the four highest local maxima.
    modes = local_maxima(log_reward, 4)
    if len(modes) != 4:
        raise ValueError("The target needs at least four local maxima")
    distances = np.count_nonzero(
        tokens[:, None, :] != encode(modes)[None, :, :], axis=-1
    ).astype(np.int32)
    distance = distances.min(axis=1)
    bin_edges = np.linspace(log_reward.min(), log_reward.max(), 25)
    reward_bin = np.clip(np.digitize(log_reward, bin_edges) - 1, 0, len(bin_edges) - 2).astype(np.int32)
    log_weight = log_reward / config.temperature
    maximum = float(log_weight.max())
    log_normalizer = maximum + float(np.log(np.exp(log_weight - maximum).sum()))
    log_pi = log_weight - log_normalizer
    pi = np.exp(log_pi)
    if np.any(pi == 0):
        raise ValueError("Temperature is too small for a full-support float64 target")
    basins = (distances == distance[:, None]).astype(np.float64)
    basins /= basins.sum(axis=1, keepdims=True)
    return Target(tokens, modes, distance, bin_edges, reward_bin,
                  log_reward, log_pi, pi, basins, log_normalizer)


class Kernel(NamedTuple):
    neighbors: IntArray
    weights: FloatArray
    stay: FloatArray

    def forward(self, distribution: FloatArray) -> FloatArray:
        """Row distribution multiplied by P; scatter incoming probability mass."""
        result = distribution * self.stay
        for column in range(self.neighbors.shape[1]):
            result += np.bincount(self.neighbors[:, column],
                                  weights=distribution * self.weights[:, column],
                                  minlength=len(distribution))
        return result

    def backward(self, features: FloatArray) -> FloatArray:
        """P applied to functions of the state."""
        result = self.stay[:, None] * features
        for column in range(self.neighbors.shape[1]):
            result += self.weights[:, column, None] * features[self.neighbors[:, column]]
        return result


def build_kernel_neighbors(tokens: IntArray) -> IntArray:
    """Ids of the 3 * LENGTH strings differing in exactly one position."""
    ids = np.arange(len(tokens), dtype=np.int32)
    powers = 4 ** np.arange(LENGTH - 1, -1, -1, dtype=np.int32)
    return np.stack([
        ids + (((tokens[:, i] + delta) % 4) - tokens[:, i]) * powers[i]
        for i in range(LENGTH) for delta in (1, 2, 3)
    ], axis=1)


def build_kernel(target: Target) -> Kernel:
    neighbors = build_kernel_neighbors(target.tokens)
    weights = np.empty(neighbors.shape, dtype=np.float64)
    for column in range(neighbors.shape[1]):
        weights[:, column] = np.exp(np.minimum(
            target.log_pi[neighbors[:, column]] - target.log_pi, 0)) / (3 * LENGTH)
    return Kernel(neighbors, weights, 1 - weights.sum(axis=1))


def diagnose(config: Config, target: Target) -> Diagnostics:
    kernel = build_kernel(target)
    balance_error = 0.0
    for column in range(kernel.neighbors.shape[1]):
        neighbors = kernel.neighbors[:, column]
        reverse = np.exp(np.minimum(target.log_pi - target.log_pi[neighbors], 0)) / (3 * LENGTH)
        balance_error = max(balance_error, float(np.max(np.abs(
            target.pi * kernel.weights[:, column] - target.pi[neighbors] * reverse))))
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
    bins = len(target.bin_edges) - 1
    histogram = np.bincount(target.reward_bin[ids], minlength=bins) / len(ids)
    exact_histogram = np.bincount(target.reward_bin, weights=target.pi, minlength=bins)
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

    path = config.data_dir / f"data{LENGTH}.parquet"
    metadata_path = config.data_dir / "metadata.json"
    if not overwrite and (path.exists() or metadata_path.exists()):
        raise FileExistsError("Dataset exists; use --overwrite to regenerate it")
    target = build_target(config)
    diagnostics = diagnose(config, target)
    print(f"MH: burn={diagnostics.burn_steps}, interval={diagnostics.save_every}", flush=True)
    log_reward = jnp.asarray(target.log_reward, dtype=jnp.float32)
    powers = jnp.asarray(4 ** np.arange(LENGTH - 1, -1, -1, dtype=np.int32))
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
        gain = log_reward[proposal @ powers] - log_reward[state @ powers]
        log_acceptance = jnp.minimum(gain / config.temperature, 0)
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
    if config.reward == "potts":
        np.save(config.data_dir / "couplings.npy", couplings(config))
    metadata = Metadata(
        reference_modes=target.modes,
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

    path = config.data_dir / f"data{LENGTH}.parquet"
    metadata = Metadata.model_validate_json((config.data_dir / "metadata.json").read_text())
    fields = ("reward", "beta", "temperature", "seed", "samples", "burn_tv", "correlation_bound", "diagnostic_limit")
    if any(getattr(config, field) != getattr(metadata.config, field) for field in fields):
        raise ValueError("Dataset metadata does not match the generation configuration")
    if fingerprint(path) != metadata.dataset_sha256:
        raise ValueError("Dataset checksum does not match metadata")
    if config.reward == "potts" and not np.array_equal(
            np.load(config.data_dir / "couplings.npy"), couplings(config)):
        raise ValueError("Saved Potts couplings do not match the configuration seed")
    strings = pq.read_table(path, columns=["sequence"]).column("sequence").to_pylist()
    tokens = encode(strings)
    validate_tokens(tokens)
    if len(tokens) != config.samples:
        raise ValueError("Unexpected dataset length")
    return tokens, metadata


def save_model_distribution(directory: Path, target: Target,
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
    directory.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, directory / "best_distribution.parquet")
