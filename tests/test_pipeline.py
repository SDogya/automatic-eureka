from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("optax")
pytest.importorskip("pyarrow")

import pyarrow.parquet as pq

from src.config import MASK, Config
from src.ablation import run_ablation
from src.data import generate, load_dataset
from src.model import complete, load_parameters
from src.train import train


def test_short_experiment_and_result_protection(tmp_path: Path) -> None:
    config = Config(samples=32, epochs=1, batch_size=16,
                    data_dir=tmp_path / "data", plots_dir=tmp_path / "plots")
    metadata = generate(config)
    tokens, restored = load_dataset(config)
    assert tokens.shape == (32, 8)
    assert restored == metadata
    with pytest.raises(FileExistsError):
        generate(config)
    model_dir = tmp_path / "models" / "mlp_2"
    report = train(config, 4, model_dir)
    assert (report.hidden_width, report.parameter_count) == (4, 344)
    assert len(report.metrics) == 2
    assert all(m.kl is not None and np.isfinite(m.kl) for m in report.metrics)
    assert all(abs(m.probability_sum - 1) < 1e-10 for m in report.metrics if m.probability_sum is not None)
    assert (config.plots_dir / "reward.png").exists()
    assert all((model_dir / name).exists() for name in ("loss.png", "kl.png", "distribution.png"))
    params = load_parameters(model_dir / "best.npz")
    tokens, _ = complete(params, jax.numpy.full((4, 8), MASK, dtype=jax.numpy.int32), jax.random.key(0))
    assert tokens.shape == (4, 8)
    distribution = pq.read_table(model_dir / "best_distribution.parquet")
    assert len(distribution) == 4**8
    assert int(distribution.schema.metadata[b"epoch"]) == report.best_epoch
    target = distribution.column("target_probability").to_numpy()
    model = distribution.column("model_probability").to_numpy()
    log_model = distribution.column("model_log_probability").to_numpy()
    assert abs(model.sum() - 1) < 1e-10
    best_metric = next(m for m in report.metrics if m.epoch == report.best_epoch)
    assert float(target @ (np.log(target) - log_model)) == pytest.approx(best_metric.kl, abs=1e-12)
    with pytest.raises(FileExistsError):
        train(config, 4, model_dir)
    with (config.data_dir / "data8.parquet").open("ab") as file:
        file.write(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        load_dataset(config)


def test_ablation_writes_one_folder_per_width(tmp_path: Path) -> None:
    config = Config(samples=32, epochs=1, batch_size=16, hidden_exponents=(1, 2),
                    data_dir=tmp_path / "data", plots_dir=tmp_path / "plots",
                    models_dir=tmp_path / "models")
    generate(config)
    ablation = run_ablation(config)
    assert [row.hidden_width for row in ablation.rows] == [2, 4]
    for exponent in (1, 2):
        folder = config.models_dir / f"mlp_{exponent}"
        assert all((folder / name).exists() for name in
                   ("best.npz", "last.npz", "training.json", "loss.png", "kl.png", "distribution.png"))
    assert (config.models_dir / "ablation.png").exists()
    assert (config.models_dir / "ablation.json").exists()


def test_potts_dataset_saves_and_checks_couplings(tmp_path: Path) -> None:
    config = Config(reward="potts", samples=32, data_dir=tmp_path / "data", plots_dir=tmp_path / "data")
    metadata = generate(config)
    assert metadata.reference_modes is not None
    assert (config.data_dir / "couplings.npy").exists()
    tokens, _ = load_dataset(config)
    assert tokens.shape == (32, 8)
    with pytest.raises(ValueError, match="metadata"):
        load_dataset(Config(reward="potts", beta=2.0, samples=32, data_dir=config.data_dir))
