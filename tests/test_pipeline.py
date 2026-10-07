from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("optax")
pytest.importorskip("pyarrow")

import pyarrow.parquet as pq

from src.config import Config, SampleRequest
from src.data import generate, load_dataset
from src.model import load_parameters, sample
from src.train import train


def test_short_experiment_and_result_protection(tmp_path: Path) -> None:
    config = Config(samples=32, epochs=1, batch_size=16,
                    data_dir=tmp_path / "data", plots_dir=tmp_path / "plots")
    metadata = generate(config)
    tokens, restored = load_dataset(config)
    assert tokens.shape == (32, 10)
    assert restored == metadata
    with pytest.raises(FileExistsError):
        generate(config)
    report = train(config)
    assert len(report.metrics) == 2
    assert all(m.kl is not None and np.isfinite(m.kl) for m in report.metrics)
    assert all(abs(m.probability_sum - 1) < 1e-10 for m in report.metrics if m.probability_sum is not None)
    assert all((config.plots_dir / name).exists() for name in ("reward.png", "loss.png", "kl.png"))
    params = load_parameters(config.data_dir / "best.npz")
    assert sample(params, SampleRequest(count=4)).shape == (4, 10)
    assert (config.plots_dir / "distribution.png").exists()
    distribution = pq.read_table(config.data_dir / "best_distribution.parquet")
    assert len(distribution) == 4**10
    assert int(distribution.schema.metadata[b"epoch"]) == report.best_epoch
    target = distribution.column("target_probability").to_numpy()
    model = distribution.column("model_probability").to_numpy()
    log_model = distribution.column("model_log_probability").to_numpy()
    assert abs(model.sum() - 1) < 1e-10
    best_metric = next(m for m in report.metrics if m.epoch == report.best_epoch)
    assert float(target @ (np.log(target) - log_model)) == pytest.approx(best_metric.kl, abs=1e-12)
    with pytest.raises(FileExistsError):
        train(config)
    with (config.data_dir / "data10.parquet").open("ab") as file:
        file.write(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        load_dataset(config)
