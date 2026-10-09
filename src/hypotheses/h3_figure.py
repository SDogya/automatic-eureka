"""H3 figure: the paper's law log p_theta(y|x) - log p_ref(y|x) = beta r(y) - log Z(x).

Grid as in the H1 figure: rows 0, 1/3, 2/3 and all of training; columns start x prompt.
Each point is one final y at (r(y), log p_theta(y|x) - log p_ref(y|x)); shown are the strings
carrying 99% of the target mass p*(y|x) or 99% of the model mass at that step.
Dashed: the law (slope beta, intercept -log Z(x), exact). Solid: p*-weighted least-squares fit.

python -m src.hypotheses.h3_figure
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from ..data import token_ids
from ..metrics.trajectories import log_p_all_y
from .common import EXPERIMENT, RESULTS, SELECTED, load_run
from .h1_figure import FRACTIONS, PROMPTS, STARTS, encode_prompt


def weighted_line(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    w = w / w.sum()
    xm, ym = w @ x, w @ y
    variance = w @ (x - xm) ** 2
    slope = float(w @ ((x - xm) * (y - ym)) / variance) if variance > 0 else 0.0
    return slope, float(ym - slope * xm)


def top_mass(p: np.ndarray, mass: float = 0.99) -> np.ndarray:
    order = np.argsort(-p)
    return order[:int(np.searchsorted(np.cumsum(p[order]), mass)) + 1]


def main() -> None:
    plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12})
    runs = {init: load_run(path) for init, path in SELECTED.items()}
    columns = [(init, text) for init in STARTS for text in PROMPTS]
    figure, axes = plt.subplots(len(FRACTIONS), len(columns), figsize=(20, 20))
    for c, (init, text) in enumerate(columns):
        run, (label, color) = runs[init], STARTS[init]
        prompt = encode_prompt(text)
        ys, log_ref = log_p_all_y(run.reference, prompt)
        reward = run.log_reward[token_ids(ys)] / run.beta  # r(y)
        log_star = log_ref + run.beta * reward
        log_z = float(np.logaddexp.reduce(log_star))
        star = np.exp(log_star - log_z)
        available = np.array(run.steps())
        panels = []
        for fraction in FRACTIONS:
            step = int(available[np.argmin(np.abs(available - fraction * available[-1]))])
            _, log_p = log_p_all_y(run.model(step), prompt)
            shown = np.union1d(top_mass(star), top_mass(np.exp(log_p)))
            panels.append((step, shown, log_p - log_ref))
        x_all = np.concatenate([reward[s] for _, s, _ in panels])
        y_all = np.concatenate([a[s] for _, s, a in panels] + [run.beta * x_all - log_z])
        x_lim = (x_all.min() - 0.05 * np.ptp(x_all), x_all.max() + 0.05 * np.ptp(x_all))
        y_lim = (y_all.min() - 0.05 * np.ptp(y_all), y_all.max() + 0.05 * np.ptp(y_all))
        span = np.array(x_lim)
        for r, (step, shown, change) in enumerate(panels):
            axis = axes[r, c]
            slope, intercept = weighted_line(reward[shown], change[shown], star[shown])
            axis.scatter(reward[shown], change[shown], s=10, alpha=0.6, color=color, linewidths=0)
            axis.plot(span, run.beta * span - log_z, color="#2d3748", linestyle="--", linewidth=1.3,
                      label=f"law  {run.beta:g}·r − {log_z:.1f}")
            axis.plot(span, slope * span + intercept, color=color, linewidth=1.8,
                      label=f"fit  {slope:.2f}·r {intercept:+.1f}")
            axis.set_xlim(*x_lim)
            axis.set_ylim(*y_lim)
            axis.xaxis.set_major_locator(MaxNLocator(5))
            axis.yaxis.set_major_locator(MaxNLocator(5))
            axis.grid(color="#e2e8f0", linewidth=0.8)
            axis.spines[["top", "right"]].set_visible(False)
            axis.legend(frameon=False, fontsize=10, loc="lower right", title=f"step {step}",
                        title_fontsize=11, alignment="left")
            if r == 0:
                axis.set_title(f"{label}\nprompt {text}", color=color)
            if r == len(FRACTIONS) - 1:
                axis.set_xlabel("r(y)")
            if c == 0:
                axis.set_ylabel("log pθ(y | x) − log p_ref(y | x)")
    figure.tight_layout()
    path = RESULTS / EXPERIMENT / "h3.png"
    figure.savefig(path, dpi=130)
    plt.close(figure)
    print(path)


if __name__ == "__main__":
    main()
