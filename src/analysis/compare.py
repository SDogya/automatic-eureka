"""Comparisons across arms that do not depend on one setting's endpoint (ported from gfn_lab's S2 analyses).

matched      y at matched x: per seed, the FIRST time the run reaches the level, log y interpolated linearly in log x
             between the two consecutive checkpoints (in training order) that bracket it; then mean [min, max] over
             seeds. Default: path KL at matched terminal KL(p_ref || p_theta), the S2 reading. First crossing, not
             a window fit, because some arms (ESPO-PPO) move the terminal KL back down while the path KL keeps
             growing: a level is then visited twice, at different path distances.
ratio_windows median of y / x over checkpoints in update windows, keeping checkpoints whose x lies in a band (so
             early near-zero distances do not dominate), pooled over a group's seeds.
"""

from typing import NamedTuple

import numpy as np

from .runs import Run, by_group


class Cell(NamedTuple):
    mean: float
    low: float
    high: float
    seeds: int


def matched_value(x: np.ndarray, y: np.ndarray, level: float) -> float:
    """First crossing of x = level in training order, log-log interpolation of y; NaN if never reached or if the run
    did not record the metric (older runs lack the newer columns)."""
    if len(x) == 0 or len(x) != len(y):
        return float("nan")
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0)
    x, y = x[ok], np.maximum(y[ok], 1e-12)          # y can be ~ -1e-9 from float error: clamp, never drop the point
    for i in range(len(x) - 1):
        lo, hi = sorted((x[i], x[i + 1]))
        if lo <= level <= hi and hi > lo:
            w = (np.log(level) - np.log(x[i])) / (np.log(x[i + 1]) - np.log(x[i]))
            return float(np.exp(np.log(y[i]) + w * (np.log(y[i + 1]) - np.log(y[i]))))
    return float("nan")


def matched(runs: list[Run], levels: tuple[float, ...], x: str = "kl_from_ref",
            y: str = "kl_path_ref") -> dict[str, dict[float, Cell]]:
    table: dict[str, dict[float, Cell]] = {}
    for group in by_group(runs).values():
        row = {}
        for level in levels:
            values = [matched_value(r.evals.get(x, np.array([])), r.evals.get(y, np.array([])), level)
                      for r in group]
            values = [v for v in values if np.isfinite(v)]
            row[level] = Cell(float(np.mean(values)), min(values), max(values), len(values)) if values else \
                Cell(float("nan"), float("nan"), float("nan"), 0)
        table[group[0].label] = row
    return table


def ratio_windows(runs: list[Run], windows: tuple[tuple[int, int], ...], band: tuple[float, float],
                  x: str = "kl_from_ref", y: str = "kl_path_ref") -> dict[str, list[tuple[float, int]]]:
    """Per group: (median of y / x, number of checkpoints) in each update window, x restricted to the band."""
    out: dict[str, list[tuple[float, int]]] = {}
    for group in by_group(runs).values():
        cells = []
        for lo, hi in windows:
            ratios = []
            for r in group:
                step, xs, ys = r.evals["step"], r.evals.get(x), r.evals.get(y)
                if xs is None or ys is None:
                    continue
                keep = (step >= lo) & (step < hi) & (xs >= band[0]) & (xs <= band[1]) & np.isfinite(ys)
                ratios.extend((ys[keep] / xs[keep]).tolist())
            cells.append((float(np.median(ratios)) if ratios else float("nan"), len(ratios)))
        out[group[0].label] = cells
    return out


def format_matched(table: dict[str, dict[float, Cell]], digits: int = 4) -> str:
    """Markdown table: one row per group, one column per level, 'mean [min-max] (seeds)'."""
    levels = sorted({lv for row in table.values() for lv in row})
    lines = ["| configuration | " + " | ".join(f"{lv:g}" for lv in levels) + " |",
             "|---|" + "---|" * len(levels)]
    for label, row in table.items():
        cells = [f"{c.mean:.{digits}f} [{c.low:.{digits}f}-{c.high:.{digits}f}] ({c.seeds})" if c.seeds else "-"
                 for c in (row[lv] for lv in levels)]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)
