"""Saved scientific plots; no interactive display is required."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .config import Config, FloatArray, IntArray, TrainingReport
from .data import Target, build_target
from .evaluation import bayes_masked_nll, build_contexts


def plot_reward(
    config: Config, target: Target, mcmc_ids: IntArray, exact_ids: IntArray
) -> None:
    config.plots_dir.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7, 4))
    theoretical = np.bincount(target.distance, weights=target.pi, minlength=9)[::-1]
    axis.bar(
        np.arange(-7, 2),
        theoretical,
        alpha=0.25,
        label="Exact target π",
        color="tab:blue",
    )
    for ids, label, color in (
        (mcmc_ids, "Metropolis–Hastings", "tab:green"),
        (exact_ids, "Independent exact sample", "tab:orange"),
    ):
        axis.hist(
            target.log_reward[ids],
            bins=np.arange(-7.5, 2, 1),
            weights=np.full(len(ids), 1 / len(ids)),
            histtype="step",
            linewidth=1.7,
            label=label,
            color=color,
        )
    axis.set(xlabel="ln R(x)", ylabel="Probability", xticks=np.arange(-7, 2))
    axis.legend()
    figure.tight_layout()
    figure.savefig(config.plots_dir / "reward.png", dpi=180)
    plt.close(figure)


def plot_training(config: Config, report: TrainingReport) -> None:
    config.plots_dir.mkdir(parents=True, exist_ok=True)
    metrics = report.metrics
    evaluated = [m for m in metrics if m.validation_loss is not None]
    target = build_target(report.config)
    baseline = bayes_masked_nll(build_contexts(), target.pi, report.config.mask_probability)
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot(
        [m.epoch for m in metrics],
        [m.training_loss - baseline for m in metrics],
        label="Training: empirical, sampled masks",
    )
    axis.plot(
        [m.epoch for m in evaluated],
        [m.validation_loss - baseline for m in evaluated],
        marker="o",
        label="Validation: empirical, all masks",
    )
    axis.axvline(
        report.best_epoch,
        color="gray",
        linestyle="--",
        label="Best validation checkpoint",
    )
    axis.axhline(0, color="tab:green", linestyle="--", label="Population Bayes baseline")
    axis.set(xlabel="Epoch", ylabel="Masked NLL − L*π (nats/sequence)",
             title=f"Population minimum L*π = {baseline:.5f}")
    axis.legend(fontsize=9)
    figure.text(0.5, 0.015, "Empirical losses can fall below the population baseline.",
                ha="center", fontsize=9)
    figure.tight_layout(rect=(0, 0.05, 1, 1))
    figure.savefig(config.plots_dir / "loss.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 4.5))
    joint_metrics = [m for m in metrics if m.kl is not None]
    entropy = float(-target.pi @ target.log_pi)
    axis.plot([m.epoch for m in joint_metrics], [m.kl for m in joint_metrics],
              marker="o", label="Exact KL (left); cross-entropy (right)")
    axis.axvline(report.best_epoch, color="gray", linestyle="--",
                 label="Best validation checkpoint")
    axis.set(
        xlabel="Epoch",
        ylabel="KL(π || pθ), nats",
        title="Exact random-order joint distribution",
    )
    maximum = max((m.kl for m in joint_metrics), default=0.0)
    axis.set_ylim(bottom=-0.04 * max(maximum, 0.01))
    entropy_axis = axis.twinx()
    lower, upper = axis.get_ylim()
    entropy_axis.set_ylim(lower + entropy, upper + entropy)
    entropy_axis.set_ylabel("H(π, pθ) = H(π) + KL, nats")
    entropy_axis.axhline(entropy, color="tab:green", linestyle="--", linewidth=1.5,
                        label=f"H(π) = {entropy:.5f} (right axis)")
    handles, labels = axis.get_legend_handles_labels()
    entropy_handles, entropy_labels = entropy_axis.get_legend_handles_labels()
    axis.legend(handles + entropy_handles, labels + entropy_labels, fontsize=9)
    figure.tight_layout()
    figure.savefig(config.plots_dir / "kl.png", dpi=180)
    plt.close(figure)


def plot_distribution(config: Config, target: Target,
                      log_probs: FloatArray, epoch: int) -> None:
    """Compare the exact target and learned distribution, without sampled noise."""
    model = np.exp(log_probs)
    target_histogram = np.bincount(target.distance, weights=target.pi, minlength=9)[::-1]
    model_histogram = np.bincount(target.distance, weights=model, minlength=9)[::-1]
    figure, (reward_axis, basin_axis) = plt.subplots(2, 1, figsize=(8, 7))
    reward_axis.bar(np.arange(-7, 2), target_histogram, alpha=0.3, label="Target π")
    reward_axis.stairs(model_histogram, np.arange(-7.5, 2, 1),
                       color="tab:orange", linewidth=2, label=f"Model pθ, epoch {epoch}")
    reward_axis.set(xlabel="ln R(x)", ylabel="Probability", xticks=np.arange(-7, 2),
                    title="Log reward distribution")
    reward_axis.legend()
    positions = np.arange(len(config.modes))
    basin_axis.bar(positions - 0.18, target.pi @ target.basins, width=0.36, label="Target π")
    basin_axis.bar(positions + 0.18, model @ target.basins, width=0.36, label="Model pθ")
    basin_axis.set(xticks=positions, xticklabels=config.modes, ylabel="Probability mass",
                   title="Nearest-mode mass (ties split equally)")
    basin_axis.set_ylim(0, 1.3 * max(float((target.pi @ target.basins).max()),
                                    float((model @ target.basins).max())))
    basin_axis.legend()
    kl = float(target.pi @ (target.log_pi - log_probs))
    tv = float(np.abs(target.pi - model).sum() / 2)
    figure.suptitle(f"Exact distributions: best checkpoint, epoch {epoch} | KL={kl:.5f}, TV={tv:.5f}")
    figure.tight_layout()
    config.plots_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(config.plots_dir / "distribution.png", dpi=180)
    plt.close(figure)
