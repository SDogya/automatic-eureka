"""One summary figure per hypothesis over all runs, built from results/hypotheses/h*/<run>/*.csv.

python -m src.hypotheses.summary h1
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .common import RESULTS

COLORS = {"finetune": "tab:blue", "random_init": "tab:orange"}


def read(hypothesis: str, name: str) -> dict[str, list[dict]]:
    return {path.parent.name: list(csv.DictReader(path.open()))
            for path in sorted((RESULTS / hypothesis).glob(f"*/{name}"))}


def init_of(run: str) -> str:
    return "finetune" if "finetune" in run else "random_init"


def summary_h1() -> Path:
    """Panel A: delta_y vs log rho at each run's last checkpoint, all runs pooled.
    Panel B: distribution of f = log rho / (|delta_y| + |log rho|) at the last checkpoint.
    Panel C: median |f| against the training step, one line per run."""
    runs = read("h1", "samples.csv")
    figure, (scatter, histogram, dynamics) = plt.subplots(1, 3, figsize=(17, 5))
    for init, color in COLORS.items():
        delta_y, log_rho, share = [], [], []
        for run, rows in runs.items():
            if init_of(run) != init:
                continue
            last = max(int(r["step"]) for r in rows)
            final = [r for r in rows if int(r["step"]) == last]
            delta_y += [float(r["delta_y"]) for r in final]
            log_rho += [float(r["log_rho"]) for r in final]
            share += [float(r["share"]) for r in final]
            steps = sorted({int(r["step"]) for r in rows if int(r["step"]) > 0})
            dynamics.plot(steps, [np.median([abs(float(r["share"])) for r in rows if int(r["step"]) == s])
                                  for s in steps], color=color, alpha=0.7, linewidth=1.5)
        scatter.scatter(delta_y, log_rho, s=6, alpha=0.4, color=color, label=f"{init} ({len(share)} samples)")
        histogram.hist(share, bins=np.linspace(-1, 1, 41), alpha=0.6, color=color,
                       label=f"{init}: median |f| = {np.median(np.abs(share)):.2f}")
        dynamics.plot([], [], color=color, label=init)
    scatter.axhline(0, color="black", linewidth=1)
    scatter.set(xlabel="δ_y = log pθ(y|x) − log p*(y|x), nats",
                ylabel="log ρ = log pθ(τ|x,y) − log p_ref(τ|x,y), nats",
                title="H1 ⇔ all points on log ρ = 0 (last checkpoint)")
    scatter.set_xscale("symlog", linthresh=0.01)
    scatter.set_yscale("symlog", linthresh=0.01)
    histogram.set(xlabel="f = log ρ / (|δ_y| + |log ρ|)", ylabel="samples",
                  title="Share of the dropped term in δ_τ = δ_y + log ρ")
    dynamics.set(xlabel="TraFL step", ylabel="median |f| over 128 samples", xscale="log",
                 title="Dynamics, one line per run (k = 3, 5, 7; seeds 1, 2)")
    for axis in (scatter, histogram, dynamics):
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
    figure.suptitle("H1: is the trajectory posterior unchanged, pθ(τ | x, y) = p_ref(τ | x, y)?  "
                    "(x, y, τ) sampled as in training")
    figure.tight_layout()
    path = RESULTS / "h1_summary.png"
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("hypothesis", choices=("h1",))
    args = parser.parse_args()
    print({"h1": summary_h1}[args.hypothesis]())


if __name__ == "__main__":
    main()
