# uv run --no-sync python scripts/headline_figure.py   (from the repo root, after the grid analyses)
"""The line's headline figure: three questions, every method, both backbones (results/report/headline.png).

Reads the analysis outputs already written for grid 3 (MLP) and grid 4 (transformer); computes nothing new.
  panel 1  late path / terminal ratio (summary.md: path drift per unit of terminal movement, late window)
  panel 2  fitted tilt b* / stated tilt (tilt.json: final checkpoints; stated = 0.8 beta centred, beta raw)
  panel 3  wasted KL = KL(p || pi_b*) at matched attained tilt b* = 0.2 (matched_tilt.md)
beta 0.5 configurations and the arms without beta; seed means. A missing point means the quantity is undefined for that
arm (no stated tilt; terminal law never settles in the late band) or the configuration was not run on that backbone.
"""

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

GRIDS = {"MLP (grid 3)": Path("results/port_grid3"), "transformer (grid 4)": Path("results/port_grid4")}
ROWS = (("trafl β0.5 square iid ×32", "TraFL, 32 IID masks", "trafl"),
        ("trafl β0.5 square comp ×4", "TraFL, 4 paired masks", "trafl"),
        ("trafl β0.5 pairwise iid ×4", "TraFL, pairwise U-stat.", "trafl"),
        ("trafl β0.5 exact_lik iid ×32", "TraFL, exact likelihood", "trafl"),
        ("trafl β0.5 square comp ×4 λvar1.0", "TraFL + order-variance penalty", "trafl"),
        ("trafl β0.5 square comp ×4 decoder", "TraFL, decoder rollouts", "trafl"),
        ("tb β0.5", "TB (RTB)", "tb"),
        ("db β0.5", "DB", "db"),
        ("subtb β0.5 λsub0.9", "SubTB", "subtb"),
        ("entppo β0.5 gae0.7 μ4", "Ent-PPO", "entppo"),
        ("espo κ0.05 comp ×4", "ESPO", "espo"),
        ("espo_ppo κ0.05 comp ×4 μ8", "ESPO-PPO", "espo_ppo"),
        ("rspo λ0.01 iid ×4", "RSPO", "rspo"),
        ("grpo", "GRPO", "grpo"),
        ("justgrpo μ1", "JustGRPO (random-order view)", "justgrpo"))
INK, MUTED, GRID = "#1f1f1f", "#6b6b6b", "#e3e3e3"
BACKBONE = {"MLP (grid 3)": ("o", "#2a78d6"), "transformer (grid 4)": ("D", "#eb6834")}   # validated slots 1-2


def number(cell: str) -> float:
    m = re.match(r"\s*(-?[\d.]+(?:e-?\d+)?)", cell)
    return float(m.group(1)) if m else float("nan")


def table(path: Path, block: str | None = None) -> dict[str, list[str]]:
    """Markdown table rows keyed by the first cell; with block, only the table under that bold heading."""
    lines = path.read_text().splitlines()
    if block is not None:
        start = next(i for i, line in enumerate(lines) if line.strip() == block)
        lines = lines[start + 1:]
        stop = next((i for i, line in enumerate(lines) if line.startswith("**b")), len(lines))
        lines = lines[:stop]
    rows = {}
    for line in lines:
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) > 2 and not set(cells[0]) <= set("-") and cells[0] != "configuration":
            rows[cells[0]] = cells
    return rows


def collect() -> dict[str, dict[str, tuple[float, float, float]]]:
    out = {}
    for name, root in GRIDS.items():
        summary = table(root / "summary.md")
        tilt = {c["label"]: c for c in json.loads((root / "tilt.json").read_text())["configurations"]}
        matched = table(root / "matched_tilt.md", r"**b\* = 0.2**")
        values = {}
        for label, _, _ in ROWS:
            late = number(summary[label][2]) if label in summary else np.nan
            ratio = tilt[label]["ratio"] if label in tilt and tilt[label].get("ratio") is not None else np.nan
            wasted = number(matched[label][3]) if label in matched else np.nan
            values[label] = (late, ratio, wasted)
        out[name] = values
    return out


def main() -> None:
    data = collect()
    panels = (("path drift per unit of string-law movement\n(late path KL / terminal KL)", 0, None),
              ("achieved tilt / stated tilt\n(1 = reaches its own target)", 1, 1.0),
              ("wasted KL at matched tilt b* = 0.2\n(distance beyond what the reward needs)", 2, None))
    fig, axes = plt.subplots(1, 3, figsize=(13, 0.36 * len(ROWS) + 1.9), sharey=True)
    ys = np.arange(len(ROWS))[::-1]
    for ax, (title, k, ref) in zip(axes, panels):
        for y in ys[1::2]:
            ax.axhspan(y - 0.5, y + 0.5, color="#f6f6f6", zorder=0)
        if ref is not None:
            ax.axvline(ref, color=MUTED, lw=1, ls="--", zorder=1)
        for offset, (backbone, (marker, colour)) in zip((0.14, -0.14), BACKBONE.items()):
            for y, (label, _, _) in zip(ys, ROWS):
                v = data[backbone][label][k]
                if np.isfinite(v):
                    ax.plot(v, y + offset, marker, color=colour, ms=7, mec="white", mew=1.2, zorder=3)
                    ax.annotate(f"{v:.2f}", (v, y + offset), xytext=(6, 0), textcoords="offset points",
                                va="center", fontsize=6.5, color=MUTED)
        ax.set_title(title, fontsize=8.5, color=INK, loc="left")
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.tick_params(colors=MUTED, labelsize=7)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        lo, hi = ax.get_xlim()
        ax.set_xlim(min(0, lo), hi * 1.12)
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels([name for _, name, _ in ROWS], fontsize=7.5, color=INK)
    handles = [plt.Line2D([], [], marker=m, color=c, ls="", ms=7, mec="white", label=b) for b, (m, c) in BACKBONE.items()]
    fig.legend(handles=handles, loc="upper right", fontsize=7.5, frameon=False, ncol=2)
    fig.suptitle("TraFL and baselines on length-8 DNA (Potts + TFBind8), exact, β 0.5, 2 seeds: three questions, "
                 "both backbones", fontsize=9.5, color=INK, x=0.01, ha="left")
    fig.text(0.01, 0.005, "Missing point = undefined for that arm (no stated tilt; terminal law never settles) or not "
             "run on that backbone. Sources: results/port_grid{3,4}/summary.md, tilt.json, matched_tilt.md.",
             fontsize=6.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    Path("results/report").mkdir(parents=True, exist_ok=True)
    fig.savefig("results/report/headline.png", dpi=150)
    print("results/report/headline.png")


if __name__ == "__main__":
    main()
