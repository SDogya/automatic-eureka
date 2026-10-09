"""H2: the masked-reconstruction surrogate stands in for the exact log-ratio.

TraFL's residual uses D_sur = S_theta(y|x) - S_ref(y|x) (1/l-weighted masks, same masks for
both models) in place of D_exact = (log p_theta(y|x) - log p_ref(y|x)) / u. The surrogate's
expectation over masks is exact:
    E S(y|x) = (1/u) E_{sigma uniform} log(u! p(tau_sigma | x)) = (log p(y|x) - gap(y)) / u,
    gap(y) = KL(uniform orders || p(tau | x, y)),
so E D_sur - D_exact = (gap_ref - gap_theta) / u. Only differences between completions of
the same prompt matter (log Z absorbs a constant), so values are also centered per prompt.
A single K-mask draw adds noise on top; it is recorded separately.

python -m src.hypotheses.h2_surrogate models/trafl/seed_1/finetune_7 [--steps 0 1000 20000]
"""

import argparse
import csv
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from ..config import MASK, decode
from ..model import forward
from ..rl.losses.trafl import surrogate_log_prob
from .common import RESULTS, Run, load_run, pick_steps, posterior_pair, training_batch

FIELDS = ("step", "context", "hidden", "y", "d_exact", "d_sur_expected", "d_sur_draw",
          "d_exact_centered", "d_sur_expected_centered", "bias_centered", "noise", "gap", "gap_ref")


def h2_rows(run: Run, step: int, contexts: int, group: int, seed: int, masks: int) -> list[dict]:
    model = run.model(step)
    key = jax.random.fold_in(jax.random.key(seed), step)
    batch = training_batch(model, key, contexts, group)
    draw_key = jax.random.fold_in(key, 1)
    args = (jnp.asarray(batch.contexts), jnp.asarray(batch.completions), draw_key, masks, "paper")
    draw = np.asarray(surrogate_log_prob(forward, model.params, *args)
                      - surrogate_log_prob(forward, run.reference.params, *args), dtype=np.float64)
    rows = []
    for c, prompt in enumerate(batch.contexts):
        hidden = int((prompt == MASK).sum())
        group_rows = []
        for g in range(group):
            post, post_ref = posterior_pair(run, model, prompt, batch.completions[c, g])
            group_rows.append({
                "step": step, "context": c, "hidden": hidden, "y": decode(batch.completions[c, g][None])[0],
                "d_exact": (post.log_p_y - post_ref.log_p_y) / hidden,
                "d_sur_expected": (post.elbo - post_ref.elbo) / hidden,  # = E over masks of S_theta - S_ref
                "d_sur_draw": float(draw[c, g]), "gap": post.elbo_gap, "gap_ref": post_ref.elbo_gap})
        exact_mean = np.mean([r["d_exact"] for r in group_rows])
        expected_mean = np.mean([r["d_sur_expected"] for r in group_rows])
        for r in group_rows:
            r["d_exact_centered"] = r["d_exact"] - exact_mean
            r["d_sur_expected_centered"] = r["d_sur_expected"] - expected_mean
            r["bias_centered"] = r["d_sur_expected_centered"] - r["d_exact_centered"]
            r["noise"] = r["d_sur_draw"] - r["d_sur_expected"]
        rows += group_rows
    return rows


def fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Slope through the origin and R^2 of y against x (both already centered per prompt)."""
    slope = float(x @ y / (x @ x)) if x @ x > 0 else float("nan")
    residual = y - slope * x
    return slope, float(1 - residual @ residual / (y @ y)) if y @ y > 0 else float("nan")


def plot_h2(run: Run, rows: list[dict], path: Path) -> None:
    steps = sorted({r["step"] for r in rows})
    shown = [s for s in steps if s > 0][-3:] or steps[-1:]
    figure, axes = plt.subplots(len(shown), 3, figsize=(15, 4 * len(shown)), squeeze=False)
    for line, step in zip(axes, shown):
        part = [r for r in rows if r["step"] == step]
        exact = np.array([r["d_exact_centered"] for r in part])
        expected = np.array([r["d_sur_expected_centered"] for r in part])
        slope, r2 = fit(exact, expected)
        low, high = min(exact.min(), expected.min()), max(exact.max(), expected.max())
        line[0].scatter(exact, expected, s=10, alpha=0.6)
        line[0].plot([low, high], [low, high], color="black", linewidth=1, label="surrogate = exact")
        line[0].set(xlabel="exact (log pθ − log p_ref)/u, centered per x", ylabel="E over masks of S_θ − S_ref, centered",
                    title=f"step {step}: slope {slope:.2f}, R² {r2:.2f}")
        line[0].legend(fontsize=8)
        line[1].hist([r["bias_centered"] for r in part], bins=40, color="tab:red")
        line[1].set(xlabel="E D_sur − D_exact, centered per x (bias)", ylabel="samples", title="Bias of the surrogate")
        line[2].hist([r["noise"] for r in part], bins=40, color="tab:gray")
        line[2].set(xlabel="one K-mask draw − its expectation", ylabel="samples", title="Mask noise (not bias)")
        for axis in line:
            axis.grid(alpha=0.3)
    figure.suptitle(f"H2 on {run.label}: (x, y) from the model")
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--steps", type=int, nargs="*")
    parser.add_argument("--contexts", type=int, default=32)
    parser.add_argument("--group", type=int, default=4)
    parser.add_argument("--masks", type=int, default=4, help="K masks in one surrogate draw, as in training")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    for path in args.runs:
        run = load_run(path)
        output = RESULTS / "h2" / run.label
        output.mkdir(parents=True, exist_ok=True)
        rows: list[dict] = []
        for step in pick_steps(run, args.steps):
            part = h2_rows(run, step, args.contexts, args.group, args.seed, args.masks)
            rows += part
            slope, r2 = fit(np.array([r["d_exact_centered"] for r in part]),
                            np.array([r["d_sur_expected_centered"] for r in part]))
            print(f"[H2 {run.label}] step {step}: slope {slope:.3f}, R² {r2:.3f}, "
                  f"|bias| median {np.median(np.abs([r['bias_centered'] for r in part])):.4f}, "
                  f"noise std {np.std([r['noise'] for r in part]):.4f}", flush=True)
        with (output / "samples.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        plot_h2(run, rows, output / "h2.png")


if __name__ == "__main__":
    main()
