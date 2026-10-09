"""Comparisons across arms that do not depend on one setting's endpoint (ported from gfn_lab's S2 analyses).

matched      y at matched x: per seed, a least-squares line in log-log space through that seed's checkpoints with
             x within a factor `window` of the level, evaluated at the level; then mean [min, max] over seeds.
             Default: path KL at matched terminal KL(p_ref || p_theta), the S2 reading.
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


def matched_value(x: np.ndarray, y: np.ndarray, level: float, window: float = 3.0) -> float:
    """Local log-log fit of y on x through the points with level / window <= x <= level * window; NaN if < 2 points."""
    keep = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0) & (x >= level / window) & (x <= level * window)
    if keep.sum() < 2 or np.ptp(np.log(x[keep])) == 0:
        return float("nan")
    slope, intercept = np.polyfit(np.log(x[keep]), np.log(y[keep]), 1)
    return float(np.exp(intercept + slope * np.log(level)))


def matched(runs: list[Run], levels: tuple[float, ...], x: str = "kl_from_ref", y: str = "kl_path_ref",
            window: float = 3.0) -> dict[str, dict[float, Cell]]:
    table: dict[str, dict[float, Cell]] = {}
    for group in by_group(runs).values():
        row = {}
        for level in levels:
            values = [matched_value(r.evals.get(x, np.array([])), r.evals.get(y, np.array([])), level, window)
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
                keep = (step >= lo) & (step <= hi) & (xs >= band[0]) & (xs <= band[1]) & np.isfinite(ys)
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
