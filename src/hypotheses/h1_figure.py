"""H1 figure: trajectory posterior during TraFL vs at the start, for the two selected runs.

For each prompt x one fixed final y = argmax_y r(y) over completions of x (the same for both
starts and all steps). Each point is one reveal order sigma of y: x-axis log p_ref(tau | x, y),
y-axis log p_theta(tau | x, y) at the given step. H1 holds when every point lies on the diagonal.
Rows: 0, 1/3, 2/3 and all of each run's training; columns: start x prompt; axis limits
shared within a column.

python -m src.hypotheses.h1_figure
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

from ..config import ALPHABET, MASK, decode
from ..data import token_ids
from ..metrics.trajectories import log_p_all_y
from .common import EXPERIMENT, RESULTS, SELECTED, load_run, posterior_pair

PROMPTS = ("????????", "A??G????")
STARTS = {"finetune": ("Pretrained start", "#2b6cb0"), "random_init": ("Random start", "#dd6b20")}
FRACTIONS = (0, 1 / 3, 2 / 3, 1)  # of each run's training, nearest saved checkpoint


def encode_prompt(text: str) -> np.ndarray:
    return np.array([MASK if c == "?" else ALPHABET.index(c) for c in text], dtype=np.int32)


def best_final(run, prompt: np.ndarray) -> np.ndarray:
    ys, _ = log_p_all_y(run.reference, prompt)
    return ys[int(np.argmax(run.log_reward[token_ids(ys)]))]


def main() -> None:
    plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12})
    runs = {init: load_run(path) for init, path in SELECTED.items()}
    finals = {text: best_final(runs["finetune"], encode_prompt(text)) for text in PROMPTS}
    columns = [(init, text) for init in STARTS for text in PROMPTS]
    figure, axes = plt.subplots(len(FRACTIONS), len(columns), figsize=(20, 20))
    for c, (init, text) in enumerate(columns):
        run, (label, color) = runs[init], STARTS[init]
        prompt, y = encode_prompt(text), finals[text]
        points = []
        available = np.array(run.steps())
        for fraction in FRACTIONS:
            step = int(available[np.argmin(np.abs(available - fraction * available[-1]))])
            post, post_ref = posterior_pair(run, run.model(step), prompt, y)
            points.append((step, post_ref.log_posterior, post.log_posterior))
        low = min(min(a.min(), b.min()) for _, a, b in points)
        high = max(max(a.max(), b.max()) for _, a, b in points)
        pad = 0.04 * (high - low)
        for r, (step, ref, new) in enumerate(points):
            axis = axes[r, c]
            many = len(ref) > 1000
            axis.scatter(ref, new, s=3 if many else 14, alpha=0.2 if many else 0.7, color=color,
                         linewidths=0, rasterized=True)
            axis.plot([low - pad, high + pad], [low - pad, high + pad], color="#4a5568", linestyle="--", linewidth=1)
            axis.set_xlim(low - pad, high + pad)
            axis.set_ylim(low - pad, high + pad)
            axis.set_aspect("equal")
            axis.xaxis.set_major_locator(MaxNLocator(5))
            axis.yaxis.set_major_locator(MaxNLocator(5))
            axis.grid(color="#e2e8f0", linewidth=0.8)
            axis.spines[["top", "right"]].set_visible(False)
            axis.text(0.04, 0.96, f"step {step}", transform=axis.transAxes, va="top", fontsize=12, color="#2d3748")
            if r == 0:
                axis.set_title(f"{label}\nx = {text}   y = {decode(y[None])[0]}", color=color)
            if r == len(FRACTIONS) - 1:
                axis.set_xlabel("log p_ref(τ | x, y)")
            if c == 0:
                axis.set_ylabel("log pθ(τ | x, y)")
    figure.tight_layout()
    path = RESULTS / EXPERIMENT / "h1.png"
    figure.savefig(path, dpi=130)
    plt.close(figure)
    print(path)


if __name__ == "__main__":
    main()
