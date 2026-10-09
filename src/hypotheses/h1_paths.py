"""H1: the trajectory posterior is unchanged, p_theta(tau | x, y) = p_ref(tau | x, y).

TraFL drops log rho from the exact trajectory residual:
    delta_tau = log p_theta(tau | x) - log p*(tau | x) = delta_y + log rho(tau, y),
    delta_y   = log p_theta(y | x) - log p*(y | x),
    log rho   = log p_theta(tau | x, y) - log p_ref(tau | x, y).
For (x, y, tau) drawn as in training we record delta_y, log rho and the bounded share
f = log rho / (|delta_y| + |log rho|) in [-1, 1]; H1 holds if f concentrates at 0.

python -m src.hypotheses.h1_paths models/trafl/seed_1/finetune_7 [--steps 0 1000 20000]
"""

import argparse
import csv
from pathlib import Path

import jax
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from ..config import MASK, decode
from ..data import token_ids
from .common import RESULTS, Run, load_run, log_target_y, order_row, pick_steps, posterior_pair, training_batch

FIELDS = ("step", "context", "hidden", "y", "order", "delta_y", "log_rho", "delta_tau", "share")


def h1_rows(run: Run, step: int, contexts: int, group: int, seed: int) -> list[dict]:
    model = run.model(step)
    batch = training_batch(model, jax.random.fold_in(jax.random.key(seed), step), contexts, group)
    rows = []
    for c, prompt in enumerate(batch.contexts):
        ys, _, log_star = log_target_y(run, prompt)
        index = {int(i): n for n, i in enumerate(token_ids(ys))}
        hidden = int((prompt == MASK).sum())
        for g in range(group):
            y, order = batch.completions[c, g], batch.orders[c, g]
            post, post_ref = posterior_pair(run, model, prompt, y)
            row = order_row(prompt, order)
            delta_y = post.log_p_y - log_star[index[int(token_ids(y[None])[0])]]
            log_rho = post.log_posterior[row] - post_ref.log_posterior[row]
            denominator = abs(delta_y) + abs(log_rho)
            rows.append({"step": step, "context": c, "hidden": hidden, "y": decode(y[None])[0],
                         "order": " ".join(map(str, order[:hidden])), "delta_y": delta_y, "log_rho": log_rho,
                         "delta_tau": delta_y + log_rho, "share": log_rho / denominator if denominator else 0.0})
    return rows


def plot_h1(run: Run, rows: list[dict], path: Path) -> None:
    steps = sorted({r["step"] for r in rows})
    shown = [s for s in steps if s > 0][-3:] or steps[-1:]
    figure, axes = plt.subplots(len(shown), 3, figsize=(15, 4 * len(shown)), squeeze=False)
    for line, step in zip(axes, shown):
        part = [r for r in rows if r["step"] == step]
        delta_y = np.array([r["delta_y"] for r in part])
        log_rho = np.array([r["log_rho"] for r in part])
        share = np.array([r["share"] for r in part])
        line[0].scatter(delta_y, log_rho, s=10, alpha=0.6)
        line[0].axhline(0, color="gray", linewidth=1)
        line[0].axvline(0, color="gray", linewidth=1)
        line[0].set(xlabel="δ_y = log pθ(y|x) − log p*(y|x)", ylabel="log ρ (dropped by TraFL)",
                    title=f"step {step}: residual vs dropped term")
        bins = np.linspace(min(delta_y.min(), log_rho.min()), max(delta_y.max(), log_rho.max()), 40)
        line[1].hist(delta_y, bins=bins, alpha=0.6, label="δ_y")
        line[1].hist(log_rho, bins=bins, alpha=0.6, label="log ρ")
        line[1].set(xlabel="nats", ylabel="samples", title="Distributions")
        line[1].legend()
        line[2].hist(share, bins=np.linspace(-1, 1, 41), color="tab:purple")
        line[2].set(xlabel="f = log ρ / (|δ_y| + |log ρ|)", ylabel="samples",
                    title=f"Share of the dropped term, median |f| = {np.median(np.abs(share)):.2f}")
        for axis in line:
            axis.grid(alpha=0.3)
    figure.suptitle(f"H1 on {run.label}: (x, y, τ) from the model, {len(part)} samples per step")
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--steps", type=int, nargs="*")
    parser.add_argument("--contexts", type=int, default=32)
    parser.add_argument("--group", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    for path in args.runs:
        run = load_run(path)
        output = RESULTS / "h1" / run.label
        output.mkdir(parents=True, exist_ok=True)
        rows: list[dict] = []
        for step in pick_steps(run, args.steps):
            part = h1_rows(run, step, args.contexts, args.group, args.seed)
            rows += part
            share = np.abs([r["share"] for r in part])
            print(f"[H1 {run.label}] step {step}: median |f| {np.median(share):.3f}, "
                  f"share of samples with |f| > 0.5: {np.mean(share > 0.5):.2f}", flush=True)
        with (output / "samples.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        plot_h1(run, rows, output / "h1.png")


if __name__ == "__main__":
    main()
