"""One figure and one table comparing every configuration on the questions the grids ask.

python -m src.analysis.summary --root models/port_grid3 --out results/port_grid3 [--level 1.0]

Per configuration (mean over seeds, with the seed range), each read at a *matched* distance so arms are compared at
the same movement from the reference, not at one setting's endpoint:
  path        path KL E_{y~p_ref} KL(c_ref || c_theta) at terminal KL(p_ref || p_theta) = level (first crossing)
  late ratio  median path / terminal KL over the last update window where terminal KL is in [0.05, 1.5]
  score       expected TFBind8 score at KL(p_theta || p_ref) = level
  top16       E[distinct top-1 % strings in 16 draws] at KL(p_theta || p_ref) = level, random-order sampler
  top16 dec.  the same under the low-confidence-remasking decoder (T 0.6)
Small multiples, one metric per panel, one shared y axis of configurations; colour follows the arm (fixed order).
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .compare import matched_value  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter  # noqa: E402

from .report import ARM_ORDER, MARKERS, PALETTE, short_label, windows_for  # noqa: E402
from .runs import by_group, find_runs  # noqa: E402

METRICS = (("path", "kl_from_ref", "kl_path_ref", "path KL at matched terminal KL"),
           ("score", "kl_to_ref", "expected_score", "score at matched KL(p‖ref)"),
           ("top16", "kl_to_ref", "top_distinct16", "distinct top-1 % in 16 draws (random order)"),
           ("top16_dec", "kl_to_ref", "llada_top_distinct16", "same, low-confidence decoder T 0.6"))


def late_ratio(run, lo: int, band: tuple[float, float] = (0.05, 1.5)) -> float:
    e = run.evals
    keep = (e["step"] >= lo) & (e.get("kl_from_ref", np.array([])) >= band[0]) & (e["kl_from_ref"] <= band[1])
    return float(np.median(e["kl_path_ref"][keep] / e["kl_from_ref"][keep])) if keep.any() else float("nan")


def summarise(root: Path, level: float) -> list[dict]:
    runs = find_runs(root)
    late_lo = windows_for(runs)[-1][0]
    vary_arch = len({r.architecture for r in runs}) > 1
    vary_start = len({r.start for r in runs}) > 1
    rows = []
    for group in by_group(runs).values():
        row = {"label": short_label(group[0], vary_arch, vary_start), "arm": group[0].arm, "seeds": len(group)}
        for key, x, y, _ in METRICS:
            values = [matched_value(r.evals.get(x, np.array([])), r.evals.get(y, np.array([])), level) for r in group]
            values = [v for v in values if np.isfinite(v)]
            row[key] = (float(np.mean(values)), min(values), max(values)) if values else (np.nan, np.nan, np.nan)
        lates = [v for v in (late_ratio(r, late_lo) for r in group) if np.isfinite(v)]
        row["late_ratio"] = (float(np.mean(lates)), min(lates), max(lates)) if lates else (np.nan, np.nan, np.nan)
        rows.append(row)
    rows.sort(key=lambda r: (ARM_ORDER.index(r["arm"]), r["label"]))
    return rows


def write(rows: list[dict], out: Path, level: float) -> None:
    out.mkdir(parents=True, exist_ok=True)
    panels = [("late_ratio", "late path / terminal ratio")] + [(k, t) for k, _, _, t in METRICS]
    lines = [f"Matched level {level:g}; mean [min-max] over seeds\n",
             "| configuration | seeds | " + " | ".join(t for _, t in panels) + " |", "|---|---|" + "---|" * len(panels)]
    for r in rows:
        cells = [f"{r[k][0]:.3g} [{r[k][1]:.3g}-{r[k][2]:.3g}]" if np.isfinite(r[k][0]) else "-" for k, _ in panels]
        lines.append(f"| {r['label']} | {r['seeds']} | " + " | ".join(cells) + " |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    fig, axes = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 0.32 * len(rows) + 1.6), sharey=True)
    ys = np.arange(len(rows))[::-1]
    for ax, (key, title) in zip(axes, panels):
        for y, r in zip(ys, rows):
            mean, low, high = r[key]
            if not np.isfinite(mean):
                continue
            slot = ARM_ORDER.index(r["arm"])
            ax.plot([low, high], [y, y], color=PALETTE[slot], linewidth=2, alpha=0.6)
            ax.plot(mean, y, MARKERS[slot], color=PALETTE[slot], markersize=6)
        ax.set_title(title, fontsize=8)
        ax.grid(alpha=0.25, linewidth=0.5, axis="x")
        if key in ("late_ratio", "path"):
            ax.set_xscale("log")
            ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2g}"))
            ax.xaxis.set_minor_formatter(NullFormatter())
        else:                                   # never magnify differences below +-5 % of the median into gaps
            finite = [r[key][0] for r in rows if np.isfinite(r[key][0])]
            if finite:
                lo, hi = ax.get_xlim()
                mid = float(np.median(finite))
                ax.set_xlim(min(lo, mid * 0.95), max(hi, mid * 1.05))
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels([r["label"] for r in rows], fontsize=7)
    fig.suptitle(f"Every configuration at matched distance {level:g} (dot = seed mean, bar = seed range)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "summary.png", dpi=140)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--level", type=float, default=1.0)
    args = parser.parse_args()
    write(summarise(args.root, args.level), args.out, args.level)
    print((args.out / "summary.md").read_text()[:3000])


if __name__ == "__main__":
    main()
