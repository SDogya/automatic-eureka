# python scripts/report_foundation_figure.py   (from the repo root)
"""Foundation figure: cumulative probability mass vs number of strings (sorted), for the pretraining laws, the k=5 references,
TraFL's targets and the decoder law. Writes results/report/foundation.png and the table behind it."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from src.architecture import load as load_checkpoint
from src.config import load_config
from src.data import build_target
from src.evaluation import build_contexts, joint_log_probs
from src.metrics.decoders import proposal_terminal_log_probs
from src.metrics.paths import log_conditionals

contexts = build_contexts()
reward = build_target(load_config(Path("configs/potts_tfbind8.json")))
curves = {}
for label, cfg in (("Potts law, beta 2 (current base)", "configs/potts.json"), ("Potts law, beta 0.75 (broader base)", "configs/potts_b0.75.json")):
    curves[label] = np.exp(build_target(load_config(Path(cfg))).log_pi)
for label, ck in (("reference k=5, beta-2 base", "models/pretrain/mlp_5/best.npz"), ("reference k=5, beta-0.75 base", "models/pretrain_b0.75/mlp_5/best.npz")):
    arch, params = load_checkpoint(Path(ck))
    cond = log_conditionals(arch.forward, params, contexts)
    log_p = joint_log_probs(contexts, cond)
    curves[label] = np.exp(log_p)
    if "0.75" in ck:
        for beta in (0.5, 1.5):
            w = log_p + beta * reward.log_reward
            curves[f"TraFL target beta {beta}, beta-0.75 reference"] = np.exp(w - np.logaddexp.reduce(w))
        curves["beta-0.75 reference, low-confidence decoder T 0.6"] = np.exp(proposal_terminal_log_probs(contexts, cond, 0.6))
palette = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7")
styles = ("-", "-", "--", "--", ":", ":", "-.")
fig, ax = plt.subplots(figsize=(9, 5.5))
table = {}
for (label, p), c, ls in zip(curves.items(), palette, styles):
    cum = np.cumsum(np.sort(p)[::-1])
    n99 = int(np.searchsorted(cum, 0.99) + 1)
    table[label] = {"strings_for_99pct": n99, "entropy": float(-(p[p > 0] * np.log(p[p > 0])).sum())}
    ax.plot(np.arange(1, len(cum) + 1), cum, ls, color=c, linewidth=2, label=f"{label}  (99 % in {n99:,})")
ax.set(xscale="log", xlabel="number of strings, most probable first (of 65,536)", ylabel="cumulative probability",
       title="How many strings carry the mass: pretraining laws, references, TraFL targets, decoder")
ax.grid(alpha=0.25, linewidth=0.5)
ax.legend(fontsize=7.5, loc="lower right")
fig.tight_layout()
fig.savefig("results/report/foundation.png", dpi=140)
Path("results/report/foundation_curves.json").write_text(json.dumps(table, indent=1) + "\n")
print(json.dumps(table, indent=1))
