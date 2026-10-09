"""Validated experiment settings and serialized records."""

from importlib.metadata import version
from pathlib import Path
from platform import python_version
from typing import Annotated, Literal, Self

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator

ALPHABET = "ACGT"
LENGTH = 8
TFBIND8_PATH = Path("data/tfbind8/tfbind8.parquet")
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


class MLPSpec(Record):
    """Masked MLP 40 -> width -> width -> 32 (see model.py)."""

    family: Literal["mlp"] = "mlp"
    width: int = Field(default=80, ge=1)  # equals model.DEFAULT_WIDTH, checked in tests

    @property
    def tag(self) -> str:
        return f"w{self.width}"


class TransformerSpec(Record):
    """Bidirectional pre-norm transformer denoiser (see transformer.py)."""

    family: Literal["transformer"] = "transformer"
    d_model: int = Field(ge=1)
    layers: int = Field(ge=1)
    heads: int = Field(ge=1)
    ff: int = Field(ge=1)

    @model_validator(mode="after")
    def check_heads(self) -> Self:
        if self.d_model % self.heads:
            raise ValueError("d_model must be divisible by heads")
        return self

    @property
    def tag(self) -> str:
        return f"d{self.d_model}_l{self.layers}_h{self.heads}_f{self.ff}"


ArchitectureSpec = Annotated[MLPSpec | TransformerSpec, Field(discriminator="family")]

# Transformer size ablation: depth 2, ff = 4 d, d = 2^2..2^7 -> 568 .. 398,980 parameters, set
# against the MLP widths 2^1..2^7 (184 .. 25,888 parameters).
TRANSFORMER_SIZES = tuple(
    TransformerSpec(d_model=d, layers=2, heads=heads, ff=4 * d)
    for d, heads in ((4, 2), (8, 2), (16, 4), (32, 4), (64, 4), (128, 8))
)


class Config(Record):
    reward: Literal["potts", "potts_tfbind8"] = "potts"
    beta: float = Field(default=1.0, gt=0, allow_inf_nan=False)
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
    data_dir: Path = Path("data")
    plots_dir: Path = Path("plots")
    models_dir: Path = Path("models")
    hidden_exponents: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7)
    architecture: ArchitectureSpec = MLPSpec()
    transformer_sizes: tuple[TransformerSpec, ...] = TRANSFORMER_SIZES

    @model_validator(mode="after")
    def check_split(self) -> Self:
        if not self.hidden_exponents or min(self.hidden_exponents) < 1:
            raise ValueError("Hidden width exponents must be positive")
        if not self.transformer_sizes or len({s.tag for s in self.transformer_sizes}) != len(self.transformer_sizes):
            raise ValueError("Transformer sizes must be nonempty and distinct")
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
    reference_modes: tuple[str, str, str, str] | None = None
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
    hidden_width: int
    parameter_count: int
    dataset_sha256: str
    best_epoch: int
    metrics: list[Metric]
    # Added after the MLP ablation: absent (None) in older reports. hidden_width is the MLP width
    # or the transformer d_model; wall_seconds is the training wall time including evaluations.
    architecture: ArchitectureSpec | None = None
    wall_seconds: float | None = None


class TokenBatch(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", strict=True)
    tokens: IntArray

    @model_validator(mode="after")
    def check_tokens(self) -> Self:
        tokens = self.tokens
        if tokens.dtype != np.int32 or tokens.ndim != 2 or tokens.shape[1] != LENGTH:
            raise ValueError(f"Tokens must be an int32 array of shape (N, {LENGTH})")
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
