# python scripts/decoder_true_law.py   (from the repo root)
"""Is the low-confidence decoder's collapse at the reference the reference's fault or the decoder's?

Exact over all 4^8 strings and 5^8 partial strings, broader base (pretraining Potts beta 0.75). The decoder (T 0.6, one
token per step, metrics.decoders) is run on four sets of conditionals:
  true law        conditionals of the Potts law itself (exact marginals): what a perfect reference would give
  reference net   the reference network's own conditionals (what every grid-3 evaluation uses)
  reference law   conditionals of the reference's random-order string law (order-consistent by construction):
                  separates the network's order inconsistency from the decoder's own mode-seeking
  TraFL target    conditionals of p_ref(y) R(y)^beta at beta 0.5: what the decoder would make of a perfectly fitted TraFL
Each law is also read under random order (for a consistent table, the law itself). Writes results/report/decoder_true_law.json.
"""

import json
from pathlib import Path

import numpy as np

from src.architecture import load as load_checkpoint
from src.config import LENGTH, MASK, Config
from src.data import build_target, potts_log_reward, tfbind8_scores
from src.evaluation import build_contexts, joint_log_probs
from src.metrics.decoders import proposal_terminal_log_probs
from src.metrics.paths import log_conditionals, string_law_log_conditionals
from src.rl.train import DECODER_T, diversity

REFERENCE = Path("models/pretrain_b0.75/mlp_5/best.npz")
contexts = build_contexts()
pi = build_target(Config.model_validate_json(Path("configs/potts_tfbind8.json").read_text()))   # post-training reward
scores = tfbind8_scores()
top = np.argsort(-pi.log_reward)[: len(pi.log_reward) // 100]
assert np.array_equal(contexts.tokens[contexts.full_ids], pi.tokens)                        # one string order


def describe(log_p: np.ndarray) -> dict:
    p = np.exp(log_p)
    live = p > 0
    s = np.sort(p)[::-1]
    d = diversity(p, top)
    return {"entropy": float(-p[live] @ log_p[live]), "strings_99": int(np.searchsorted(np.cumsum(s), 0.99) + 1),
            "top_string_mass": float(s[0]), "support": int(live.sum()), "expected_score": float(p @ scores),
            "distinct16": d[0], "top1pct_mass": d[1], "top1pct_distinct16": d[2]}


laws = {}
potts = potts_log_reward(pi.tokens, np.load("data/potts_b0.75/couplings.npy"), 0.75)
laws["true law (Potts beta 0.75)"] = (potts - np.logaddexp.reduce(potts), None)
arch, params = load_checkpoint(REFERENCE)
net_cond = log_conditionals(arch.forward, params, contexts)
log_ref = joint_log_probs(contexts, net_cond)
laws["reference net (mlp_5)"] = (log_ref, net_cond)
laws["reference law (consistent)"] = (log_ref, None)
tilt = log_ref + 0.5 * pi.log_reward
laws["TraFL target (beta 0.5)"] = (tilt - np.logaddexp.reduce(tilt), None)

out = {}
for name, (log_p, cond) in laws.items():
    cond = string_law_log_conditionals(contexts, log_p) if cond is None else cond
    row = {"random order": describe(log_p), "decoder T 0.6": describe(proposal_terminal_log_probs(contexts, cond, DECODER_T))}
    out[name] = row
    print(name, json.dumps({k: {m: round(v, 3) for m, v in r.items()} for k, r in row.items()}), flush=True)
Path("results/report").mkdir(parents=True, exist_ok=True)
Path("results/report/decoder_true_law.json").write_text(json.dumps(out, indent=1) + "\n")
