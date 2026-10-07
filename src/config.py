"""Validated experiment settings and serialized records."""

from importlib.metadata import version
from pathlib import Path
from platform import python_version
from typing import Literal, Self

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator

ALPHABET = "ACGT"
LENGTH = 10
MASK = 4
IntArray = NDArray[np.int32]
FloatArray = NDArray[np.float64]


def software_versions() -> dict[str, str]:
    packages = ("jax", "jaxlib", "optax", "numpy", "pydantic", "pyarrow", "matplotlib")
    return {"python": python_version(), **{name: version(name) for name in packages}}


class Record(BaseModel):
    model_config = ConfigDict(
        extra="forbid", strict=True, frozen=True, allow_inf_nan=False
    )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2) + "\n")


class Config(Record):
    modes: tuple[str, str, str, str] = ("ATGCCTAGAC", "GATCGCATGT", "TGCTGCCACA", "AGGATATATG")
    temperature: float = Field(default=1.0, gt=0, allow_inf_nan=False)
    seed: int = Field(default=42, ge=0, le=2**32 - 1)
    samples: int = Field(default=10_000, ge=10)
    burn_tv: float = Field(default=1e-4, gt=0, lt=1)
    correlation_bound: float = Field(default=0.01, gt=0, lt=1)
    diagnostic_limit: int = Field(default=10_000, ge=1)
    mask_probability: float = Field(default=0.5, gt=0, le=1)
    epochs: int = Field(default=100, ge=1)
    batch_size: int = Field(default=256, ge=1)
    learning_rate: float = Field(default=1e-3, gt=0, allow_inf_nan=False)
    eval_every: int = Field(default=10, ge=1)
    validation_fraction: float = Field(default=0.1, gt=0, lt=1)
    eval_batch_size: int = Field(default=8192, ge=1)
    data_dir: Path = Path("data/length10")
    plots_dir: Path = Path("plots/length10")

    @model_validator(mode="after")
    def check_modes_and_split(self) -> Self:
        for mode in self.modes:
            if len(mode) != LENGTH or set(mode) - set(ALPHABET):
                raise ValueError("Each mode must contain exactly ten A/C/G/T symbols")
        if any(
            sum(a != b for a, b in zip(x, y)) < 6
            for i, x in enumerate(self.modes)
            for y in self.modes[i + 1 :]
        ):
            raise ValueError("Pairwise mode distances must be at least six")
        if not 1 <= round(self.samples * self.validation_fraction) < self.samples:
            raise ValueError("Both training and validation subsets must be nonempty")
        return self


class Diagnostics(Record):
    burn_steps: int
    save_every: int
    burn_tv: float
    correlation_bounds: tuple[float, float, float, float, float]
    stationarity_error: float
    detailed_balance_error: float
    expected_acceptance: float


class Comparison(Record):
    reward_tv: float
    full_tv: float
    basin_means: tuple[float, float, float, float]
    mean_log_reward: float


class Metadata(Record):
    config: Config
    software: dict[str, str]
    diagnostics: Diagnostics
    acceptance_rate: float
    target_log_normalizer: float
    target_basin_means: tuple[float, float, float, float]
    mcmc: Comparison
    exact_sample: Comparison
    dataset_sha256: str


class Metric(Record):
    epoch: int
    training_loss: float
    validation_loss: float | None = None
    kl: float | None = None
    probability_sum: float | None = None


class TrainingReport(Record):
    config: Config
    software: dict[str, str]
    parameter_count: Literal[13800] = 13800
    dataset_sha256: str
    best_epoch: int
    metrics: list[Metric]


class SampleRequest(Record):
    context: str = "??????????"
    count: int = Field(default=10, ge=1)
    seed: int = Field(default=42, ge=0, le=2**32 - 1)

    @model_validator(mode="after")
    def check_context(self) -> Self:
        if len(self.context) != LENGTH or set(self.context) - set(ALPHABET + "?"):
            raise ValueError("Context must contain ten A/C/G/T/? symbols")
        return self


class TokenBatch(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", strict=True)
    tokens: IntArray

    @model_validator(mode="after")
    def check_tokens(self) -> Self:
        tokens = self.tokens
        if tokens.dtype != np.int32 or tokens.ndim != 2 or tokens.shape[1] != LENGTH:
            raise ValueError("Tokens must be an int32 array of shape (N, 10)")
        if np.any((tokens < 0) | (tokens >= len(ALPHABET))):
            raise ValueError("Tokens must be in [0, 3]")
        return self


def load_config(path: Path) -> Config:
    return Config.model_validate_json(path.read_text())


def rng(seed: int, stream: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([seed, stream]))


def encode(strings: tuple[str, ...] | list[str]) -> IntArray:
    return np.array([[ALPHABET.index(c) for c in s] for s in strings], dtype=np.int32)


def decode(tokens: IntArray) -> list[str]:
    return ["".join(ALPHABET[int(c)] for c in row) for row in tokens]


def validate_tokens(tokens: IntArray) -> None:
    TokenBatch(tokens=tokens)
