"""H3: at the trajectory-balance fixed point p_theta(y | x) ~ p_ref(y | x) exp(r(y)).

For each prompt x of a fixed set and each checkpoint, a weighted least-squares fit over all
completions y with weights p*(y | x):
    log p_theta(y | x) - log p_ref(y | x) = a * r(y) + b + noise.
H3 holds if a = 1 and the weighted R^2 = 1. The prompt set is fixed per run: for every
u = 1..7, 16 prompts made by drawing y from p*(y | empty) and hiding u random positions,
plus the empty prompt (u = 8).

python -m src.hypotheses.h3_fixed_point models/trafl/seed_1/finetune_7 [--steps 0 1000 20000]
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from ..config import LENGTH, MASK
from ..data import enumerate_tokens, token_ids
from ..metrics.trajectories import log_p_all_y
from .common import RESULTS, Run, load_run, log_target_y, pick_steps

FIELDS = ("step", "prompt", "hidden", "slope", "intercept", "r2")


def prompt_set(run: Run, per_hidden: int = 16, seed: int = 0) -> list[np.ndarray]:
    empty = np.full(LENGTH, MASK, dtype=np.int32)
    _, _, log_star = log_target_y(run, empty)
    generator = np.random.default_rng(seed)
    tokens = enumerate_tokens(4)
    prompts = []
    for hidden in range(1, LENGTH):
        for _ in range(per_hidden):
            prompt = tokens[generator.choice(len(tokens), p=np.exp(log_star))].copy()
            prompt[generator.choice(LENGTH, hidden, replace=False)] = MASK
            prompts.append(prompt)
    return prompts + [empty]


def weighted_fit(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[float, float, float]:
    """Weighted least squares y = a x + b; returns a, b and the weighted R^2."""
    w = w / w.sum()
    x_mean, y_mean = w @ x, w @ y
    variance = w @ (x - x_mean) ** 2
    slope = float(w @ ((x - x_mean) * (y - y_mean)) / variance) if variance > 0 else float("nan")
    intercept = float(y_mean - slope * x_mean)
    residual = y - slope * x - intercept
    total = w @ (y - y_mean) ** 2
    return slope, intercept, float(1 - w @ residual**2 / total) if total > 0 else float("nan")


def h3_rows(run: Run, prompts: list[np.ndarray], steps: list[int], label: str) -> list[dict]:
    targets = [log_target_y(run, prompt) for prompt in prompts]
    rows = []
    for step in steps:
        model = run.model(step)
        for prompt, (ys, log_ref, log_star) in zip(prompts, targets):
            _, log_p = log_p_all_y(model, prompt)
            slope, intercept, r2 = weighted_fit(run.log_reward[token_ids(ys)], log_p - log_ref, np.exp(log_star))
            rows.append({"step": step, "prompt": "".join("?" if c == MASK else "ACGT"[c] for c in prompt),
                         "hidden": int((prompt == MASK).sum()), "slope": slope, "intercept": intercept, "r2": r2})
        part = [r for r in rows if r["step"] == step]
        print(f"[H3 {label}] step {step}: median slope {np.nanmedian([r['slope'] for r in part]):.3f}, "
              f"median R² {np.nanmedian([r['r2'] for r in part]):.3f}", flush=True)
    return rows


def plot_h3(run: Run, rows: list[dict], path: Path) -> None:
    figure, (slope_axis, r2_axis) = plt.subplots(1, 2, figsize=(14, 5))
    colors = plt.cm.viridis(np.linspace(0, 1, LENGTH))
    for hidden in range(1, LENGTH + 1):
        part = [r for r in rows if r["hidden"] == hidden]
        steps = sorted({r["step"] for r in part})
        for axis, key in ((slope_axis, "slope"), (r2_axis, "r2")):
            values = [[r[key] for r in part if r["step"] == s] for s in steps]
            axis.plot(steps, [np.nanmedian(v) for v in values], marker="o", markersize=5, linewidth=2,
                      color=colors[hidden - 1], label=f"u = {hidden}")
            if len(values[0]) > 1:
                axis.fill_between(steps, [np.nanpercentile(v, 25) for v in values],
                                  [np.nanpercentile(v, 75) for v in values], color=colors[hidden - 1], alpha=0.15)
    slope_axis.axhline(1, color="black", linestyle="--", linewidth=1)
    r2_axis.axhline(1, color="black", linestyle="--", linewidth=1)
    slope_axis.set(xlabel="TraFL step", ylabel="slope a", title="log pθ − log p_ref vs r(y): slope (1 = target)")
    r2_axis.set(xlabel="TraFL step", ylabel="weighted R²", title="Weighted R² (1 = exact proportionality)")
    for axis in (slope_axis, r2_axis):
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8, ncol=2)
    figure.suptitle(f"H3 on {run.label}: median over prompts, band = interquartile range")
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--steps", type=int, nargs="*")
    parser.add_argument("--per-hidden", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    for path in args.runs:
        run = load_run(path)
        output = RESULTS / "h3" / run.label
        output.mkdir(parents=True, exist_ok=True)
        rows = h3_rows(run, prompt_set(run, args.per_hidden, args.seed), pick_steps(run, args.steps), run.label)
        with (output / "prompts.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        plot_h3(run, rows, output / "h3.png")


if __name__ == "__main__":
    main()
