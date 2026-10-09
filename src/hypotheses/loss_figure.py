"""Training curves of the two selected runs: TraFL loss (trailing 200-step moving average)
and the learning-rate schedule actually used.

python -m src.hypotheses.loss_figure
"""

import csv

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .common import EXPERIMENT, RESULTS, SELECTED, last_step

STARTS = {"finetune": ("Pretrained start", "#2b6cb0"), "random_init": ("Random start", "#dd6b20")}
WINDOW = 200


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """Trailing mean over the last `window` steps; the window grows from 1 at the start."""
    total = np.cumsum(np.r_[0.0, values])
    ends = np.arange(1, len(values) + 1)
    starts = np.maximum(ends - window, 0)
    return (total[ends] - total[starts]) / (ends - starts)


def main() -> None:
    plt.rcParams.update({"font.size": 13, "axes.labelsize": 14})
    figure, (loss_axis, lr_axis) = plt.subplots(2, 1, figsize=(13, 8), sharex=True,
                                                gridspec_kw={"height_ratios": [3, 1]})
    for init, (label, color) in STARTS.items():
        rows = [r for r in csv.DictReader((SELECTED[init] / "steps.csv").open()) if int(r["step"]) <= last_step()]
        steps = np.array([int(r["step"]) for r in rows])
        loss = np.array([float(r["loss"]) for r in rows])
        loss_axis.plot(steps, moving_average(loss, WINDOW), color=color, linewidth=2, label=label)
        lr_axis.plot(steps, [float(r["learning_rate"]) for r in rows], color=color, linewidth=2)
    loss_axis.set(ylabel=f"TraFL loss\n({WINDOW}-step moving average)", yscale="log")
    lr_axis.set(xlabel="TraFL step", ylabel="learning rate", yscale="log")
    for axis in (loss_axis, lr_axis):
        axis.grid(color="#e2e8f0", linewidth=0.8, which="major")
        axis.spines[["top", "right"]].set_visible(False)
    loss_axis.legend(frameon=False)
    figure.tight_layout()
    path = RESULTS / EXPERIMENT / "loss.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    print(path)


if __name__ == "__main__":
    main()
