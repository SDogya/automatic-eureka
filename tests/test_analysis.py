"""Gates for the run adapter and the comparisons (src/analysis)."""

import csv
import json
from pathlib import Path

import numpy as np

from src.analysis.compare import matched, matched_value, ratio_windows
from src.analysis.report import report
from src.analysis.runs import find_runs, read_run


def write_run(root: Path, name: str, seed: int, trafl: dict, rows: list[dict], architecture: dict | None) -> Path:
    path = root / f"seed_{seed}" / name
    path.mkdir(parents=True)
    settings = {"name": name, "init": "finetune", "hidden_width": 32, "start": "models/pretrain/mlp_5/best.npz",
                "config": {"reward": "potts_tfbind8"}, "segments": [{"first_step": 0, "trafl": {"seed": seed, **trafl}}]}
    if architecture is not None:
        settings["architecture"] = architecture
    (path / "settings.json").write_text(json.dumps(settings))
    with (path / "evals.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def rows(scale: float) -> list[dict]:
    """Power-law synthetic run: terminal x grows with step, path y = scale * x^1.5, score = x."""
    out = []
    for step in (0, 250, 500, 1000, 2000, 4000):
        x = 0.01 * (1 + step / 100)
        out.append({"step": step, "kl": 1.0, "expected_score": x, "kl_to_ref": x, "kl_from_ref": x,
                    "kl_path_ref": scale * x**1.5, "llada_kl_to_ref": x, "llada_expected_score": x})
    return out


def test_adapter_round_trip_and_grouping(tmp_path: Path) -> None:
    mlp = {"family": "mlp", "width": 32}
    a0 = write_run(tmp_path, "a", 0, {"arm": "trafl", "beta": 1.5, "estimator": "square"}, rows(2.0), mlp)
    write_run(tmp_path, "a", 1, {"arm": "trafl", "beta": 1.5, "estimator": "square"}, rows(2.0), mlp)
    write_run(tmp_path, "b", 0, {"arm": "trafl", "beta": 1.5, "estimator": "split"}, rows(1.0), mlp)
    old = write_run(tmp_path, "old", 0, {"beta": 1.5}, [{"step": 0, "kl": 3.3, "expected_score": 0.35}], None)
    run = read_run(a0)
    with (a0 / "evals.csv").open() as f:
        first = next(csv.DictReader(f))
    assert run.evals["kl_path_ref"][0] == float(first["kl_path_ref"]) and run.seed == 0
    runs = find_runs(tmp_path)
    assert len(runs) == 4 and len({r.group for r in runs}) == 3          # seeds share a group; estimator splits it
    legacy = read_run(old)
    assert legacy.arm == "trafl" and legacy.architecture == "mlp_w32" and "kl_path_ref" not in legacy.evals


def test_matched_and_windows_recover_the_power_law(tmp_path: Path) -> None:
    x = np.array([0.01, 0.03, 0.1, 0.3, 1.0])
    assert abs(matched_value(x, 2 * x**1.5, 0.2) - 2 * 0.2**1.5) < 1e-9
    assert np.isnan(matched_value(x, 2 * x**1.5, 100.0))                  # never reached
    xs, ys = np.array([0.0, 0.1, 1.0, 0.5, 0.2]), np.array([0.0, 0.01, 0.1, 5.0, 9.0])   # rises, then falls back
    assert abs(matched_value(xs, ys, 0.3) - np.exp(np.log(0.01) + np.log(3) / np.log(10) * np.log(10))) < 1e-9
    mlp = {"family": "mlp", "width": 32}
    write_run(tmp_path, "a", 0, {"arm": "grpo"}, rows(2.0), mlp)
    write_run(tmp_path, "a", 1, {"arm": "grpo"}, rows(2.0), mlp)
    runs = find_runs(tmp_path)
    cell = matched(runs, (0.1,))[runs[0].label][0.1]
    assert cell.seeds == 2 and abs(cell.mean - 2 * 0.1**1.5) < 1e-9
    win = ratio_windows(runs, ((0, 600),), (0.0, 10.0))[runs[0].label][0]
    xs = np.array([0.01, 0.035, 0.06])
    assert win[1] == 6 and abs(win[0] - np.median(2 * xs**0.5)) < 1e-9
    out = tmp_path / "report"
    report(tmp_path, out, (0.1, 0.3), (0.0, 10.0))
    assert all((out / f).exists() for f in ("matched_path.md", "matched_score.md", "ratio_windows.md",
                                             "frontier.png", "path.png", "ratio.png", "decoder.png"))
