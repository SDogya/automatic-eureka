"""H2 figure: TraFL's surrogate difference vs the exact log-probability difference, for every final y.

For a prompt x with u hidden positions and each completion y (one point):
    a(y) = log p_theta(y | x) - log p_ref(y | x)                      (exact, all reveal orders)
    b(y) = E[S_theta(y | x) - S_ref(y | x)]                          (paper's 1/l surrogate, exact
           expectation over l ~ U{1..u} and M ~ U{|M| = l}: a finite sum over all masks M)
If the surrogate tracks the exact quantity, b(y) = a(y) / u + c(x): the dashed line has slope
1/u and passes through the p*(y|x)-weighted centroid. Grid as in the H1 figure.

python -m src.hypotheses.h2_figure
"""

from itertools import combinations
from math import comb

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from ..config import MASK, decode
from ..data import enumerate_tokens
from ..metrics.trajectories import Model, log_p_all_y
from .common import EXPERIMENT, RESULTS, SELECTED, load_run, log_target_y
from .h1_figure import FRACTIONS, PROMPTS, STARTS, encode_prompt


def surrogate_expectation(model: Model, prompt: np.ndarray) -> np.ndarray:
    """E[S(y | x)] for every completion y (lexicographic over hidden positions, as log_p_all_y):
    sum over nonempty M of 1 / (C(u, |M|) |M|) * sum_{i in M} log q(y_i | m_M(y), i), divided by u."""
    hidden = np.flatnonzero(prompt == MASK)
    u = len(hidden)
    letters = enumerate_tokens(4, u)  # (4^u, u): y on the hidden positions
    total = np.zeros(len(letters))
    for size in range(1, u + 1):
        weight = 1.0 / (comb(u, size) * size)
        for masked in combinations(range(u), size):
            kept = [j for j in range(u) if j not in masked]
            states = np.tile(prompt, (4 ** len(kept), 1))
            states[:, hidden[kept]] = enumerate_tokens(4, len(kept))
            log_q = model.log_conditionals(states)  # (4^|kept|, L, 4)
            state = letters[:, kept] @ (4 ** np.arange(len(kept) - 1, -1, -1)) if kept else np.zeros(len(letters), int)
            for j in masked:
                total += weight * log_q[state, hidden[j], letters[:, j]]
    return total / u


def main() -> None:
    plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12})
    runs = {init: load_run(path) for init, path in SELECTED.items()}
    columns = [(init, text) for init in STARTS for text in PROMPTS]
    figure, axes = plt.subplots(len(FRACTIONS), len(columns), figsize=(20, 20))
    for c, (init, text) in enumerate(columns):
        run, (label, color) = runs[init], STARTS[init]
        prompt = encode_prompt(text)
        u = int((prompt == MASK).sum())
        _, log_ref, log_star = log_target_y(run, prompt)
        weights = np.exp(log_star)
        sur_ref = surrogate_expectation(run.reference, prompt)
        available = np.array(run.steps())
        panels = []
        for fraction in FRACTIONS:
            step = int(available[np.argmin(np.abs(available - fraction * available[-1]))])
            model = run.model(step)
            _, log_p = log_p_all_y(model, prompt)
            panels.append((step, log_p - log_ref, surrogate_expectation(model, prompt) - sur_ref))
        x_low = min(a.min() for _, a, _ in panels)
        x_high = max(a.max() for _, a, _ in panels)
        y_low = min(b.min() for _, _, b in panels)
        y_high = max(b.max() for _, _, b in panels)
        x_pad, y_pad = 0.04 * (x_high - x_low or 1), 0.04 * (y_high - y_low or 1)
        for r, (step, a, b) in enumerate(panels):
            axis = axes[r, c]
            if np.ptp(a) == 0 and np.ptp(b) == 0:  # step 0: the model is the reference, every y at (0, 0)
                axis.scatter([a[0]], [b[0]], s=60, color=color)
            else:
                axis.scatter(a, b, s=2 if len(a) > 5000 else 6, alpha=0.25, color=color, linewidths=0, rasterized=True)
            x_line = np.array([x_low - x_pad, x_high + x_pad])
            centre_a, centre_b = weights @ a, weights @ b
            axis.plot(x_line, centre_b + (x_line - centre_a) / u, color="#4a5568", linestyle="--", linewidth=1)
            axis.set_xlim(x_low - x_pad, x_high + x_pad)
            axis.set_ylim(y_low - y_pad, y_high + y_pad)
            axis.xaxis.set_major_locator(MaxNLocator(5))
            axis.yaxis.set_major_locator(MaxNLocator(5))
            axis.grid(color="#e2e8f0", linewidth=0.8)
            axis.spines[["top", "right"]].set_visible(False)
            axis.text(0.04, 0.96, f"step {step}", transform=axis.transAxes, va="top", fontsize=12, color="#2d3748")
            if r == 0:
                axis.set_title(f"{label}\nx = {text}   (u = {u}, slope 1/{u})", color=color)
            if r == len(FRACTIONS) - 1:
                axis.set_xlabel("log pθ(y | x) − log p_ref(y | x)")
            if c == 0:
                axis.set_ylabel("E[Ŝθ(y | x) − Ŝ_ref(y | x)]")
        print(f"[H2 figure] {init} {text} done", flush=True)
    figure.tight_layout()
    path = RESULTS / EXPERIMENT / "h2.png"
    figure.savefig(path, dpi=130)
    plt.close(figure)
    print(path)


if __name__ == "__main__":
    main()
