"""Tables and figures for a folder of post-training runs.

python -m src.analysis.report --root models/<sweep> --out results/<name> [--levels 0.03 0.1 0.3 1]

Writes markdown tables (the table view of every figure) and four single-axis figures, one line per run, colour and
marker by arm in a fixed order (validated categorical palette; the colour follows the arm, never its rank):
  matched_path.md     path KL at matched terminal KL(p_ref || p_theta)   (gfn_lab S2 reading)
  matched_score.md    expected TFBind8 score at matched KL(p_theta || p_ref)
  ratio_windows.md    median path / terminal KL by update window
  frontier.png        score vs KL(p_theta || p_ref)
  path.png            path KL vs terminal KL
  ratio.png           path / terminal ratio over training
  decoder.png         score vs KL to the reference under the low-confidence-remasking decoder
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .compare import format_matched, matched, ratio_windows  # noqa: E402
from .runs import ARM_KNOBS, Run, find_runs  # noqa: E402

ARM_ORDER = tuple(ARM_KNOBS)   # fixed slot per arm
# 8 validated categorical slots; arms 9-10 (db, subtb) reuse slots 2 and 7 (the balance family: tb, entppo) with their
# own markers, so identity never rests on colour alone (legend + marker + table view)
PALETTE = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948", "#eb6834", "#4a3aa7")
MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*", "h", "p")
LINES = ("-", "--", ":", "-.")   # configurations within one arm (colour = arm, line style = configuration)
SHORT = {"beta": "β", "kappa": "κ", "estimator": "", "mask_scheme": "", "mask_samples": "×", "var_lambda": "λvar",
         "ppo_epochs": "μ", "gae_lambda": "gae", "rspo_lambda": "λ", "subtb_lambda": "λsub", "rollout": "",
         "rollout_temperature": "T"}
DEFAULTS = {"normalization": "paper", "eps_clip": 0.2, "advantage_std": True, "var_lambda": 0.0, "rollout": "uniform",
            "rollout_temperature": 1.0}


def short_label(run: Run, vary_arch: bool, vary_start: bool) -> str:
    """Arm plus its knobs, compactly; architecture / start only when they vary across the report."""
    knobs = [f"{SHORT.get(k, k + '=')}{v}" for k, v in run.group[1]
             if k in SHORT and v is not None and DEFAULTS.get(k) != v]          # None: run predates the knob = default
    parts = [run.arm, " ".join(knobs)]
    if vary_arch:
        parts.append(run.architecture)
    if vary_start:
        parts.append(f"from {run.start or 'random'}")
    return " ".join(p for p in parts if p)


def styles(runs: list[Run]) -> dict[tuple, dict]:
    """One style per configuration: colour and marker by arm (fixed slot), line style by configuration in the arm."""
    out, count = {}, {}
    for run in sorted(runs, key=lambda r: r.label):
        if run.group in out:
            continue
        slot = ARM_ORDER.index(run.arm)
        k = count.get(run.arm, 0)
        count[run.arm] = k + 1
        out[run.group] = {"color": PALETTE[slot], "marker": MARKERS[slot], "linestyle": LINES[k % len(LINES)]}
    return out


def windows_for(runs: list[Run], count: int = 5) -> tuple[tuple[int, int], ...]:
    """Log-spaced update windows spanning the evaluated steps (first non-zero step to the last)."""
    steps = np.concatenate([r.evals["step"] for r in runs])
    positive = steps[steps > 0]
    if len(positive) == 0:
        return ((0, 0),)
    edges = np.unique(np.round(np.geomspace(positive.min(), positive.max() + 1, count + 1)).astype(int))
    return tuple((int(a), int(b)) for a, b in zip(edges[:-1], edges[1:]))


def _curves(runs: list[Run], x: str, y: str, path: Path, xlabel: str, ylabel: str, logy: bool = False) -> None:
    figure, axis = plt.subplots(figsize=(9, 6))
    style, labels, seen = styles(runs), _labels(runs), set()
    for run in runs:
        xs, ys = run.evals.get(x), run.evals.get(y)
        if xs is None or ys is None or not np.isfinite(xs).any():
            continue
        keep = np.isfinite(xs) & np.isfinite(ys) & (xs > 0)
        label = labels[run.group] if run.group not in seen else None
        seen.add(run.group)
        axis.plot(xs[keep], ys[keep], linewidth=1.5, markersize=4, alpha=0.85, label=label, **style[run.group])
    axis.set(xscale="log", yscale="log" if logy else "linear", xlabel=xlabel, ylabel=ylabel)
    axis.grid(alpha=0.25, linewidth=0.5)
    axis.legend(fontsize=7, title="configuration (one line per seed)", title_fontsize=7)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def _labels(runs: list[Run]) -> dict[tuple, str]:
    vary_arch = len({r.architecture for r in runs}) > 1
    vary_start = len({r.start for r in runs}) > 1
    return {r.group: short_label(r, vary_arch, vary_start) for r in runs}


def _ratio_figure(table: dict[str, list[tuple[float, int]]], runs: list[Run], windows, path: Path) -> None:
    group_of = {r.label: r.group for r in runs}
    style, labels = styles(runs), _labels(runs)
    centres = [np.sqrt(max(lo, 1) * hi) for lo, hi in windows]
    figure, axis = plt.subplots(figsize=(9, 6))
    for label, cells in table.items():
        medians = [m for m, _ in cells]
        group = group_of[label]
        axis.plot(centres, medians, linewidth=1.5, markersize=5, label=labels[group], **style[group])
    axis.set(xscale="log", yscale="log", xlabel="training update (window centre)",
             ylabel="median path KL / terminal KL")
    axis.grid(alpha=0.25, linewidth=0.5)
    axis.legend(fontsize=6)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def report(root: Path, out: Path, levels: tuple[float, ...], band: tuple[float, float]) -> list[Run]:
    runs = find_runs(root)
    if not runs:
        raise FileNotFoundError(f"no runs (settings.json + evals.csv) under {root}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "matched_path.md").write_text(
        "Path KL at matched terminal KL(p_ref || p_theta); mean [min-max] (seeds)\n\n"
        + format_matched(matched(runs, levels)) + "\n")
    (out / "matched_score.md").write_text(
        "Expected TFBind8 score at matched KL(p_theta || p_ref); mean [min-max] (seeds)\n\n"
        + format_matched(matched(runs, levels, x="kl_to_ref", y="expected_score"), digits=3) + "\n")
    edges = windows_for(runs)
    windows = ratio_windows(runs, edges, band)
    lines = [f"Median path KL / terminal KL by update window (terminal KL in {band}); value (checkpoints)\n",
             "| configuration | " + " | ".join(f"{lo}-{hi}" for lo, hi in edges) + " |",
             "|---|" + "---|" * len(edges)]
    lines += [f"| {label} | " + " | ".join(f"{m:.3f} ({n})" if n else "-" for m, n in cells) + " |"
              for label, cells in windows.items()]
    (out / "ratio_windows.md").write_text("\n".join(lines) + "\n")
    _curves(runs, "kl_to_ref", "expected_score", out / "frontier.png", "KL(p_theta || p_ref)", "expected TFBind8 score")
    _curves(runs, "kl_from_ref", "kl_path_ref", out / "path.png", "terminal KL(p_ref || p_theta)",
            "path KL  E_{y~p_ref} KL(c_ref || c_theta)", logy=True)
    _curves(runs, "llada_kl_to_ref", "llada_expected_score", out / "decoder.png",
            "KL to the reference, low-confidence-remasking decoder (T 0.6)", "expected score, same decoder")
    _ratio_figure(windows, runs, edges, out / "ratio.png")
    columns = sorted({k for r in runs for k in r.evals})
    with (out / "runs.csv").open("w") as f:                    # every checkpoint of every run, for re-analysis
        f.write(",".join(["configuration", "seed", "path"] + columns) + "\n")
        for r in runs:
            for i in range(len(r.evals["step"])):
                values = [repr(float(r.evals[c][i])) if c in r.evals else "" for c in columns]
                f.write(",".join([f'"{r.label}"', str(r.seed), str(r.path)] + values) + "\n")
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--levels", type=float, nargs="+", default=[0.03, 0.1, 0.3, 1.0])
    parser.add_argument("--band", type=float, nargs=2, default=[0.05, 1.5])
    args = parser.parse_args()
    runs = report(args.root, args.out, tuple(args.levels), tuple(args.band))
    print(f"{len(runs)} runs -> {args.out}")


if __name__ == "__main__":
    main()
