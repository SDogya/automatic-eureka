"""Size ablations: models/mlp_k/ per MLP hidden width 2^k, models/transformer_<tag>/ per transformer size."""

import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from .config import Config, Record, TrainingReport, TransformerSpec
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


class TransformerAblationRow(Record):
    tag: str
    d_model: int
    layers: int
    heads: int
    ff: int
    parameter_count: int
    best_epoch: int
    best_validation_loss: float
    best_kl: float
    final_kl: float
    wall_seconds: float


class TransformerAblationReport(Record):
    rows: list[TransformerAblationRow]


def best_and_final(report: TrainingReport) -> tuple[float, float, float]:
    """(validation loss, KL) at the best epoch and KL at the last epoch, all evaluated exactly."""
    best = next(m for m in report.metrics if m.epoch == report.best_epoch)
    final = report.metrics[-1]
    if best.validation_loss is None or best.kl is None or final.kl is None:
        raise ValueError("Best and final epochs must be evaluated exactly")
    return best.validation_loss, best.kl, final.kl


def summarize(exponent: int, report: TrainingReport) -> AblationRow:
    best_loss, best_kl, final_kl = best_and_final(report)
    return AblationRow(
        exponent=exponent, hidden_width=report.hidden_width,
        parameter_count=report.parameter_count, best_epoch=report.best_epoch,
        best_validation_loss=best_loss, best_kl=best_kl, final_kl=final_kl,
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


def summarize_transformer(report: TrainingReport) -> TransformerAblationRow:
    spec = report.architecture
    if not isinstance(spec, TransformerSpec) or report.wall_seconds is None:
        raise ValueError("Report is not from a transformer run")
    best_loss, best_kl, final_kl = best_and_final(report)
    return TransformerAblationRow(
        tag=spec.tag, d_model=spec.d_model, layers=spec.layers, heads=spec.heads, ff=spec.ff,
        parameter_count=report.parameter_count, best_epoch=report.best_epoch,
        best_validation_loss=best_loss, best_kl=best_kl, final_kl=final_kl,
        wall_seconds=report.wall_seconds,
    )


def run_transformer_ablation(config: Config, overwrite: bool = False) -> TransformerAblationReport:
    """Train every config.transformer_sizes entry into models_dir/transformer_<tag>/."""
    summary_path = config.models_dir / "transformer_ablation.json"
    figure_path = config.models_dir / "transformer_ablation.png"
    if not overwrite and (summary_path.exists() or figure_path.exists()):
        raise FileExistsError("Transformer ablation summary exists; use --overwrite to replace it")
    rows: list[TransformerAblationRow] = []
    for spec in config.transformer_sizes:
        print(f"=== transformer_{spec.tag} ===", flush=True)
        report = train(config, output_dir=config.models_dir / f"transformer_{spec.tag}",
                       overwrite=overwrite, architecture=spec)
        rows.append(summarize_transformer(report))
        TransformerAblationReport(rows=rows).save(summary_path)
    ablation = TransformerAblationReport(rows=rows)
    plot_transformer_ablation(config, ablation)
    return ablation


def plot_transformer_ablation(config: Config, ablation: TransformerAblationReport) -> None:
    """KL against parameter count; the MLP runs found in models_dir are drawn for reference."""
    rows = ablation.rows
    figure, axis = plt.subplots(figsize=(8, 4.5))
    # Raw JSON: the committed MLP reports carry a config field (modes) that Config has since dropped.
    mlp = []
    for path in sorted(config.models_dir.glob("mlp_*/training.json")):
        report = json.loads(path.read_text())
        mlp.append((report["parameter_count"],
                    next(m["kl"] for m in report["metrics"] if m["epoch"] == report["best_epoch"])))
    if mlp:
        axis.plot([p for p, _ in mlp], [kl for _, kl in mlp], marker="x", markersize=8, linewidth=1.5,
                  color="gray", label="MLP, best checkpoint")
    axis.plot([row.parameter_count for row in rows], [row.best_kl for row in rows], marker="o",
              markersize=8, linewidth=2, color="tab:blue", label="Transformer, best validation checkpoint")
    axis.plot([row.parameter_count for row in rows], [row.final_kl for row in rows], marker="s",
              markersize=8, linewidth=2, linestyle="--", color="tab:orange",
              label=f"Transformer, last epoch {config.epochs}")
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set(xlabel="Parameter count", ylabel="Exact KL(π || pθ), nats",
             title="Transformer size ablation (depth 2, ff = 4 d_model)")
    for row in rows:
        axis.annotate(f"d={row.d_model}", (row.parameter_count, row.best_kl),
                      textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
    axis.grid(True, which="major", alpha=0.3)
    axis.legend()
    figure.tight_layout()
    config.models_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(config.models_dir / "transformer_ablation.png", dpi=180)
    plt.close(figure)


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

