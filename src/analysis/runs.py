"""One reader for every post-training run: settings.json (identity, knobs) + evals.csv (exact metrics per checkpoint).

A run's *group* is everything that identifies a configuration except the seed, so seeds of one configuration are
compared with each other and never pooled across arms, knobs, architectures or references. Runs written before the
port lack the new metric columns and the `architecture` field; missing columns read as NaN and the architecture falls
back to the MLP of the recorded width.
"""

import csv
import json
from pathlib import Path
from typing import NamedTuple

import numpy as np

# knobs that identify a configuration, per arm (the rest of the TraFL config is recorded but not part of the label)
ARM_KNOBS = {
    "trafl": ("beta", "estimator", "mask_scheme", "mask_samples", "var_lambda", "normalization"),
    "tb": ("beta",),
    "espo": ("kappa", "mask_scheme", "mask_samples"),
    "espo_ppo": ("kappa", "mask_scheme", "mask_samples", "ppo_epochs", "eps_clip"),
    "grpo": (),
    "justgrpo": ("ppo_epochs", "eps_clip"),
    "entppo": ("beta", "gae_lambda", "ppo_epochs", "eps_clip"),
    "rspo": ("rspo_lambda", "advantage_std", "mask_scheme", "mask_samples"),
}
SHARED = ("context_source", "contexts", "group", "learning_rate", "reference")


class Run(NamedTuple):
    path: Path
    seed: int
    group: tuple            # hashable configuration identity without the seed
    label: str              # readable form of `group`
    arm: str
    architecture: str       # e.g. "mlp_w32" or "transformer_d32_l2_h4_f128"
    start: str | None       # start / reference checkpoint folder name, None = random init
    evals: dict[str, np.ndarray]


def _architecture(settings: dict) -> str:
    spec = settings.get("architecture") or {"family": "mlp", "width": settings["hidden_width"]}
    if spec["family"] == "mlp":
        return f"mlp_w{spec['width']}"
    return f"transformer_d{spec['d_model']}_l{spec['layers']}_h{spec['heads']}_f{spec['ff']}"


def read_run(path: Path) -> Run:
    settings = json.loads((path / "settings.json").read_text())
    trafl = settings["segments"][-1]["trafl"]          # resumed runs: the last segment's settings
    arm = trafl.get("arm", "trafl")
    knobs = tuple((k, trafl.get(k)) for k in ARM_KNOBS[arm])
    shared = tuple((k, trafl.get(k)) for k in SHARED)
    start = None if settings.get("start") is None else Path(settings["start"]).parent.name
    architecture = _architecture(settings)
    group = (arm, knobs, shared, architecture, start, settings["config"].get("reward"))
    label = f"{arm}[" + ", ".join(f"{k}={v}" for k, v in knobs) + f"] {architecture} from {start or 'random'}"
    with (path / "evals.csv").open() as f:
        rows = list(csv.DictReader(f))
    evals = {key: np.array([float(r[key]) if r.get(key) not in (None, "") else np.nan for r in rows])
             for key in (rows[0].keys() if rows else [])}
    return Run(path, int(trafl["seed"]), group, label, arm, architecture, start, evals)


def find_runs(root: Path) -> list[Run]:
    """Every run folder (one holding settings.json and evals.csv) below root, sorted by path."""
    return [read_run(p.parent) for p in sorted(root.rglob("settings.json")) if (p.parent / "evals.csv").exists()]


def by_group(runs: list[Run]) -> dict[tuple, list[Run]]:
    groups: dict[tuple, list[Run]] = {}
    for run in runs:
        groups.setdefault(run.group, []).append(run)
    return groups
