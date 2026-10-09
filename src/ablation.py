"""MLP width ablation: one models/mlp_k/ folder per hidden width 2^k."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from .config import Config, Record, TrainingReport
from .train import train


class AblationRow(Record):
    exponent: int
    hidden_width: int
    parameter_count: int
    best_epoch: int
    best_validation_loss: float
    best_kl: float
    final_kl: float


class AblationReport(Record):
    rows: list[AblationRow]


def summarize(exponent: int, report: TrainingReport) -> AblationRow:
    best = next(m for m in report.metrics if m.epoch == report.best_epoch)
    final = report.metrics[-1]
    if best.validation_loss is None or best.kl is None or final.kl is None:
        raise ValueError("Best and final epochs must be evaluated exactly")
    return AblationRow(
        exponent=exponent, hidden_width=report.hidden_width,
        parameter_count=report.parameter_count, best_epoch=report.best_epoch,
        best_validation_loss=best.validation_loss, best_kl=best.kl, final_kl=final.kl,
    )


def run_ablation(config: Config, overwrite: bool = False) -> AblationReport:
    rows: list[AblationRow] = []
    for exponent in config.hidden_exponents:
        print(f"=== mlp_{exponent}: hidden width {2**exponent} ===", flush=True)
        output_dir = config.models_dir / f"mlp_{exponent}"
        report = train(config, 2**exponent, output_dir, overwrite=overwrite)
        rows.append(summarize(exponent, report))
    ablation = AblationReport(rows=rows)
    ablation.save(config.models_dir / "ablation.json")
    plot_ablation(config, ablation)
    return ablation


def plot_ablation(config: Config, ablation: AblationReport) -> None:
    rows = ablation.rows
    exponents = [row.exponent for row in rows]
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(exponents, [row.best_kl for row in rows], marker="o", markersize=8,
              linewidth=2, color="tab:blue", label="Best validation checkpoint")
    axis.plot(exponents, [row.final_kl for row in rows], marker="s", markersize=8,
              linewidth=2, linestyle="--", color="tab:orange", label=f"Last epoch {config.epochs}")
    axis.set_yscale("log")
    axis.set_xticks(exponents, [f"$2^{{{row.exponent}}}$\n{row.parameter_count:,} p."
                                for row in rows])
    axis.set(xlabel="Hidden width (parameter count)", ylabel="Exact KL(π || pθ), nats",
             title="MLP width ablation: 40 → 2^k → 2^k → 32")
    axis.grid(True, which="major", alpha=0.3)
    axis.legend()
    figure.tight_layout()
    config.models_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(config.models_dir / "ablation.png", dpi=180)
    plt.close(figure)

