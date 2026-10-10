"""Post-hoc evaluation of each run's final checkpoint under the low-confidence-remasking decoder.

python -m src.analysis.endpoints --root models/port_grid2 --out results/port_grid2/endpoints.csv

Per run: the decoder's terminal and trajectory Hellinger distances from the run's own reference and their difference
(the path part, bounded; src/metrics/decoders.proposal_hellinger), at T 0.6 (TraFL's evaluation) and T 1.0, with the
decoder's entropy and expected TFBind8 score. Runs pulled from Kaggle keep only their last checkpoint, which is used.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from ..architecture import load as load_checkpoint
from ..data import tfbind8_scores
from ..evaluation import build_contexts
from ..metrics.decoders import proposal_hellinger, proposal_terminal_log_probs
from ..metrics.paths import log_conditionals
from .runs import find_runs

TEMPERATURES = (0.6, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    contexts, scores = build_contexts(), tfbind8_scores()
    reference_tables: dict[str, np.ndarray] = {}
    rows = []
    for run in find_runs(args.root):
        checkpoints = sorted(run.path.glob("step_*.npz"))
        if not checkpoints:
            continue
        settings = json.loads((run.path / "settings.json").read_text())
        start = settings["start"]
        if start not in reference_tables:
            arch, params = load_checkpoint(Path(start))
            reference_tables[start] = log_conditionals(arch.forward, params, contexts)
        arch, params = load_checkpoint(checkpoints[-1])
        cond = log_conditionals(arch.forward, params, contexts)
        row = {"configuration": run.label, "seed": run.seed, "step": int(checkpoints[-1].stem.split("_")[1])}
        for t in TEMPERATURES:
            h2_traj, h2_term, excess = proposal_hellinger(contexts, reference_tables[start], cond, t)
            log_p = proposal_terminal_log_probs(contexts, cond, t)
            p = np.exp(log_p)
            ok = p > 0
            row.update({f"T{t}_h2_term": h2_term, f"T{t}_h2_traj": h2_traj, f"T{t}_h2_path": excess,
                        f"T{t}_entropy": float(-p[ok] @ log_p[ok]), f"T{t}_score": float(p @ scores)})
        rows.append(row)
        print(row["configuration"][:60], row["seed"], {k: round(v, 4) for k, v in row.items() if k.startswith("T")},
              flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
