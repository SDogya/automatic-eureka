"""Trained tilt, distance to the target and per-string path drift of every run's final checkpoint (exact, empty prompt).

python -m src.analysis.tilt --root models/port_grid3 --out results/port_grid3

Per run, against the reference p_ref and the reward r = log R of TraFL's target family pi_b = p_ref e^{b r} / Z_b:
  b*          the tilt with the model's expected reward, E_{pi_b*}[r] = E_{p_theta}[r]; this is the minimiser of
              KL(p_theta || pi_b) over b, and b* = b exactly when p_theta = pi_b
  stated b    the tilt the arm's loss targets at its optimum: beta (1 - 1/G) for the group-centred arms (trafl, tb,
              tb_dec), beta for the raw-reward arms (db, subtb, entppo), none for the others
  slope       least-squares slope of log p_theta / p_ref on r, weighted by p_theta: the tilt where the model's mass is
              (b* also feels the low-reward tail; slope > b* means that tail is under-suppressed)
  shape KL    KL(p_theta || pi_b*): how far the law is from every tilt of the reference
  KL / TV     to pi at the stated tilt and at the nominal beta
  decoder TV  TV between the model's law and the stated target's law under the low-confidence decoder (the target
              decoded through its exact, order-consistent conditionals)
  per-string path KL  KL(c_ref(.|y) || c_theta(.|y)) for every y, by forward-backward on the 2^L reveal subsets; its
              p_ref-weighted mean is asserted to reproduce the gated aggregate (metrics.paths.path_kl)
Writes tilt.md, tilt.json and three figures: tilt.png (log p_theta / p_ref against r, coloured by model mass),
path_by_string.png (per-string path KL against r) and orders.png (reveal step x position for the top target string).
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

from ..architecture import load as load_checkpoint  # noqa: E402
from ..config import LENGTH, MASK, Config  # noqa: E402
from ..data import build_target, tfbind8_scores  # noqa: E402
from ..evaluation import build_contexts, joint_log_probs  # noqa: E402
from ..metrics.decoders import proposal_step_law, proposal_terminal_log_probs  # noqa: E402
from ..metrics.paths import log_conditionals, path_kl, string_law_log_conditionals  # noqa: E402
from ..rl.train import DECODER_T  # noqa: E402
from .report import ARM_ORDER, PALETTE, short_label  # noqa: E402
from .runs import by_group, find_runs  # noqa: E402

CENTRED, RAW = ("trafl", "tb", "tb_dec"), ("db", "subtb", "entppo")
SUBSETS = np.arange(2 ** LENGTH)
BITS = (SUBSETS[:, None] >> np.arange(LENGTH)) & 1                     # (2^L, L): position i revealed in subset S
NEXT = SUBSETS[:, None] | (1 << np.arange(LENGTH))                      # S + {i}
LAYERS = np.argsort(BITS.sum(1), kind="stable")                         # subsets by size


def tilted(log_p_ref: np.ndarray, r: np.ndarray, b: float) -> np.ndarray:
    w = log_p_ref + b * r
    return w - np.logaddexp.reduce(w)


def fitted_tilt(log_p: np.ndarray, log_p_ref: np.ndarray, r: np.ndarray) -> float:
    """b* with E_{pi_b*}[r] = E_p[r] (E_{pi_b}[r] increases in b), by bisection."""
    goal, lo, hi = float(np.exp(log_p) @ r), -5.0, 20.0
    for _ in range(80):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if float(np.exp(tilted(log_p_ref, r, mid)) @ r) < goal else (lo, mid)
    return (lo + hi) / 2


def kl(lp: np.ndarray, lq: np.ndarray) -> float:
    p = np.exp(lp)
    live = p > 0
    return float(p[live] @ (lp[live] - lq[live]))


def tv(lp: np.ndarray, lq: np.ndarray) -> float:
    return float(np.abs(np.exp(lp) - np.exp(lq)).sum() / 2)


def lattice_weights(contexts, log_cond: np.ndarray, ys: np.ndarray, decoder: bool = False) -> np.ndarray:
    """log weight of revealing y_i from the partial string (y on S, mask elsewhere), shape (B, 2^L, L); -inf for i in S.
    Random order: log q(y_i | s, i) - log(L - |S|). Decoder: the exact step law P(i, y_i | s)."""
    ids = (np.where(BITS[None] == 1, ys[:, None, :], MASK) * contexts.powers).sum(-1)          # (B, 2^L)
    if decoder:
        hidden = np.broadcast_to(BITS == 0, ids.shape + (LENGTH,)).reshape(-1, LENGTH)
        law = proposal_step_law(log_cond[ids.ravel()], hidden, DECODER_T).reshape(ids.shape + (LENGTH, MASK))
        with np.errstate(divide="ignore"):
            w = np.log(np.take_along_axis(law, ys[:, None, :, None], -1)[..., 0])
    else:
        lq = np.take_along_axis(log_cond[ids], ys[:, None, :, None], -1)[..., 0]
        with np.errstate(divide="ignore"):
            w = lq - np.log(LENGTH - BITS.sum(1))[None, :, None]
    return np.where(BITS[None] == 1, -np.inf, w)


def forward_backward(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """log alpha(S) = log P(reveal y_S first), log beta(S) = log P(complete y from S); alpha(full) = log p(y)."""
    n = w.shape[0]
    la = np.full((n, len(SUBSETS)), -np.inf)
    la[:, 0] = 0.0
    lb = np.full((n, len(SUBSETS)), -np.inf)
    lb[:, -1] = 0.0
    with np.errstate(invalid="ignore"):
        for s in LAYERS[1:]:
            prev = [s ^ (1 << i) for i in range(LENGTH) if s >> i & 1]
            la[:, s] = np.logaddexp.reduce(np.stack([la[:, p] + w[:, p, i] for p, i in
                                                     zip(prev, [i for i in range(LENGTH) if s >> i & 1])]), axis=0)
        for s in LAYERS[::-1][1:]:
            free = [i for i in range(LENGTH) if not s >> i & 1]
            lb[:, s] = np.logaddexp.reduce(np.stack([w[:, s, i] + lb[:, s | (1 << i)] for i in free]), axis=0)
    return la, lb


def edge_flow(w: np.ndarray, la: np.ndarray, lb: np.ndarray) -> np.ndarray:
    """P(the path given y crosses S -> S + {i}), shape (B, 2^L, L)."""
    with np.errstate(invalid="ignore"):
        f = np.exp(la[:, :, None] + w + lb[:, NEXT] - la[:, -1][:, None, None])
    return np.nan_to_num(f)


def string_path_kl(contexts, cond_ref: np.ndarray, cond_theta: np.ndarray, batch: int = 1024) -> np.ndarray:
    """KL(c_ref(. | y) || c_theta(. | y)) for every full string y (full_ids order)."""
    ys = contexts.tokens[contexts.full_ids]
    out = np.empty(len(ys))
    for start in range(0, len(ys), batch):
        y = ys[start:start + batch]
        w_ref, w_theta = lattice_weights(contexts, cond_ref, y), lattice_weights(contexts, cond_theta, y)
        la, lb = forward_backward(w_ref)
        la_theta, _ = forward_backward(w_theta)
        diff = np.where(np.isfinite(w_ref), w_ref - w_theta, 0.0)
        out[start:start + batch] = (edge_flow(w_ref, la, lb) * diff).sum((1, 2)) - (la[:, -1] - la_theta[:, -1])
    return out


def reveal_heatmap(contexts, log_cond: np.ndarray, y: np.ndarray, decoder: bool) -> np.ndarray:
    """P(step t reveals position i | y), shape (L steps, L positions)."""
    w = lattice_weights(contexts, log_cond, y[None], decoder)
    la, lb = forward_backward(w)
    f = edge_flow(w, la, lb)[0]                                          # (2^L, L)
    size = BITS.sum(1)
    return np.stack([f[size == t].sum(0) for t in range(LENGTH)])


def final_checkpoint(run_dir: Path) -> Path:
    steps = [p for p in run_dir.glob("step_*.npz")]
    return max(steps, key=lambda p: int(p.stem.split("_")[1]))


def load_reference(runs: list) -> tuple:
    """(contexts, settings per run, reward r = log R, reference conditionals, log p_ref) for runs that share one
    reference and one reward."""
    settings = [json.loads((r.path / "settings.json").read_text()) for r in runs]
    starts = {s["start"] for s in settings}
    configs = {json.dumps(s["config"], sort_keys=True) for s in settings}
    assert len(starts) == 1 and len(configs) == 1, "one reference and one reward per report"
    contexts = build_contexts()
    pi = build_target(Config.model_validate_json(configs.pop()))
    arch, params = load_checkpoint(Path(starts.pop()))
    cond_ref = log_conditionals(arch.forward, params, contexts)
    return contexts, settings, pi.log_reward, cond_ref, joint_log_probs(contexts, cond_ref)


def analyse(root: Path) -> tuple[list[dict], dict]:
    runs = find_runs(root)
    contexts, settings, r, cond_ref, log_ref = load_reference(runs)
    scores = tfbind8_scores()
    for b in (0.0, 0.37):                                                # the tilt estimator recovers a known tilt
        assert abs(fitted_tilt(tilted(log_ref, r, b), log_ref, r) - b) < 1e-6
    decoded_targets: dict[float, np.ndarray] = {}
    ref_dec = proposal_terminal_log_probs(contexts, cond_ref, DECODER_T)
    vary_arch = len({x.architecture for x in runs}) > 1
    rows, keep = [], {"contexts": contexts, "cond_ref": cond_ref, "log_ref": log_ref, "r": r, "runs": {}}
    for run, s in zip(runs, settings):
        trafl = s["segments"][-1]["trafl"]
        beta, group = trafl.get("beta"), trafl.get("group", 5)
        stated = beta * (1 - 1 / group) if run.arm in CENTRED else beta if run.arm in RAW else None
        a, p = load_checkpoint(final_checkpoint(run.path))
        cond = log_conditionals(a.forward, p, contexts)
        log_p = joint_log_probs(contexts, cond)
        log_dec = proposal_terminal_log_probs(contexts, cond, DECODER_T)
        per_string = string_path_kl(contexts, cond_ref, cond)
        gated = path_kl(contexts, cond_ref, cond).kl_path
        assert np.isclose(np.exp(log_ref) @ per_string, gated, rtol=1e-5, atol=1e-8), (run.path, gated)
        b_star = fitted_tilt(log_p, log_ref, r)
        p_theta, ratio = np.exp(log_p), log_p - log_ref                 # least-squares slope where the model's mass is
        slope = float(np.cov(r, ratio, aweights=p_theta)[0, 1] / np.cov(r, aweights=p_theta))
        row = {"label": short_label(run, vary_arch, False), "arm": run.arm, "seed": run.seed,
               "step": int(final_checkpoint(run.path).stem.split("_")[1]), "beta": beta, "stated": stated,
               "b_star": b_star, "slope_mass": slope, "shape_kl": kl(log_p, tilted(log_ref, r, b_star)),
               "path_kl": float(gated), "score": float(np.exp(log_p) @ scores),
               "score_decoder": float(np.exp(log_dec) @ scores), "tv_decoder_ref": tv(log_dec, ref_dec)}
        for name, b in (("stated", stated), ("nominal", beta)):
            if b is None or stated is None:
                continue
            target = tilted(log_ref, r, b)
            row[f"kl_{name}"], row[f"tv_{name}"] = kl(log_p, target), tv(log_p, target)
        if stated is not None:
            if stated not in decoded_targets:
                decoded_targets[stated] = proposal_terminal_log_probs(
                    contexts, string_law_log_conditionals(contexts, tilted(log_ref, r, stated)), DECODER_T)
            row["tv_decoder_stated"] = tv(log_dec, decoded_targets[stated])
        rows.append(row)
        keep["runs"][(row["label"], run.seed)] = {"log_p": log_p, "per_string": per_string, "cond": cond,
                                                  "arm": run.arm, "stated": stated, "b_star": b_star,
                                                  "slope": slope}
        print(f"{row['label']} s{run.seed}: b* {b_star:.3f} (stated {stated}) shape KL {row['shape_kl']:.3f}",
              flush=True)
    for b, log_dec_target in decoded_targets.items():                     # what a perfect fit of each target gives
        target = tilted(log_ref, r, b)
        rows.append({"label": f"target π_{b:g} itself (ideal)", "arm": "target", "seed": 0, "step": 0, "beta": None,
                     "stated": b, "b_star": b, "slope_mass": b, "shape_kl": 0.0, "kl_stated": 0.0, "tv_stated": 0.0,
                     "tv_decoder_stated": 0.0, "tv_decoder_ref": tv(log_dec_target, ref_dec),
                     "score": float(np.exp(target) @ scores), "score_decoder": float(np.exp(log_dec_target) @ scores)})
    return rows, keep


COLUMNS = (("stated", "stated b"), ("b_star", "fitted b*"), ("ratio", "b* / stated"), ("slope_mass", "mass-weighted slope"), ("shape_kl", "KL(p ‖ π_b*)"),
           ("kl_stated", "KL(p ‖ π_stated)"), ("kl_nominal", "KL(p ‖ π_β)"), ("tv_stated", "TV(p, π_stated)"),
           ("tv_decoder_stated", "decoder TV to decoded π_stated"), ("tv_decoder_ref", "decoder TV to decoded ref"),
           ("path_kl", "path KL (final)"), ("score", "score"), ("score_decoder", "score, decoder"))


def table(rows: list[dict]) -> tuple[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        row["ratio"] = row["b_star"] / row["stated"] if row["stated"] else None
        groups.setdefault(row["label"], []).append(row)
    order = sorted(groups, key=lambda k: (ARM_ORDER.index(groups[k][0]["arm"]) if groups[k][0]["arm"] in ARM_ORDER
                                          else -1, k))
    lines = ["Final checkpoint of every run; mean [min-max] over seeds. π_b = p_ref e^{b r} / Z_b, r = log R.\n",
             "| configuration | step | " + " | ".join(t for _, t in COLUMNS) + " |",
             "|---|---|" + "---|" * len(COLUMNS)]
    summary = []
    for label in order:
        g = groups[label]
        cells, entry = [], {"label": label, "arm": g[0]["arm"], "seeds": len(g)}
        for key, _ in COLUMNS:
            v = [x[key] for x in g if x.get(key) is not None]
            entry[key] = float(np.mean(v)) if v else None
            cells.append("-" if not v else f"{np.mean(v):.3g}" if len(v) == 1 or max(v) - min(v) < 5e-4 * max(1, abs(np.mean(v)))
                         else f"{np.mean(v):.3g} [{min(v):.3g}-{max(v):.3g}]")
        lines.append(f"| {label} | {g[0]['step']} | " + " | ".join(cells) + " |")
        summary.append(entry)
    return "\n".join(lines) + "\n", summary


def figures(keep: dict, out: Path, focus_beta: float) -> None:
    r, log_ref, contexts = keep["r"], keep["log_ref"], keep["contexts"]
    seed1 = {label: v for (label, seed), v in keep["runs"].items() if seed == min(s for (l, s) in keep["runs"] if l == label)}
    panels = [k for k in seed1 if f"β{focus_beta:g}" in k or seed1[k]["stated"] is None]
    panels.sort(key=lambda k: (ARM_ORDER.index(seed1[k]["arm"]), k))
    cols = 6
    rows_n = -(-len(panels) // cols)
    fig, axes = plt.subplots(rows_n, cols, figsize=(2.9 * cols, 2.6 * rows_n), sharex=True, squeeze=False)
    p_ref = np.exp(log_ref)
    lo_r, hi_r = np.quantile(r, [0.001, 0.999])
    for ax, label in zip(axes.flat, panels):
        v = seed1[label]
        p = np.exp(v["log_p"])
        ratio = v["log_p"] - log_ref
        hb = ax.hexbin(r, ratio, C=p, reduce_C_function=np.sum, gridsize=36, cmap="Blues",
                       norm=LogNorm(1e-4, 1), linewidths=0)
        centre_r, centre_y = float(p @ r), float(p @ ratio)
        xs = np.array([lo_r, hi_r])
        ax.plot(xs, centre_y + v["b_star"] * (xs - centre_r), color="#e34948", lw=1.2, label=f"fitted b* {v['b_star']:.2f}")
        ax.plot(xs, centre_y + v["slope"] * (xs - centre_r), color="#eda100", lw=1.2, ls=":",
                label=f"mass-weighted slope {v['slope']:.2f}")
        if v["stated"] is not None:
            ax.plot(xs, centre_y + v["stated"] * (xs - centre_r), color="#1a1a1a", lw=1, ls="--",
                    label=f"stated {v['stated']:.2f}")
        mix = (p + p_ref) / 2                                            # y-range of the strings that carry mass
        keep_y = ratio[mix > 1e-6]
        ax.set_ylim(np.quantile(keep_y, 0.001) - 1, np.quantile(keep_y, 0.999) + 1)
        ax.set_title(label, fontsize=6.5)
        ax.legend(fontsize=5.5, loc="lower right", frameon=False)
        ax.grid(alpha=0.2, lw=0.5)
    for ax in axes.flat[len(panels):]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("r(y) = log R", fontsize=7)
    for ax in axes[:, 0]:
        ax.set_ylabel("log p_θ(y) / p_ref(y)", fontsize=7)
    for ax in axes.flat:
        ax.tick_params(labelsize=6)
    fig.colorbar(hb, ax=axes, shrink=0.6, label="model mass in cell")
    fig.suptitle("Trained tilt: every string, coloured by the model's mass; a pure tilt of the reference is a straight "
                 "line of slope b", fontsize=9)
    fig.savefig(out / "tilt.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    edges = np.quantile(r, np.linspace(0, 1, 21))
    which = np.clip(np.searchsorted(edges, r, side="right") - 1, 0, 19)
    centres = np.array([r[which == k].mean() for k in range(20)])
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    styles, used = ("-", "--", ":", "-.", (0, (5, 1)), (0, (1, 1)), (0, (3, 1, 1, 1))), {}
    for label in [k for k in panels if seed1[k]["stated"] is not None]:
        v = seed1[label]
        w = p_ref * v["per_string"]
        mean = np.array([w[which == k].sum() / p_ref[which == k].sum() for k in range(20)])
        slot = ARM_ORDER.index(v["arm"])
        used[PALETTE[slot]] = used.get(PALETTE[slot], -1) + 1                # the palette repeats colours across arms
        ax.plot(centres, mean, color=PALETTE[slot], lw=1.3, ls=styles[used[PALETTE[slot]] % len(styles)], label=label)
    ax.set_yscale("log")
    ax.set_xlabel("r(y) = log R (20 equal-count bins)")
    ax.set_ylabel("KL(c_ref(τ|y) ‖ c_θ(τ|y)), p_ref-weighted bin mean")
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=6, ncol=2, frameon=False)
    ax.set_title("Where the path drift lives (arms with a stated tilt, final checkpoints): per-string order-law "
                 "distance from the reference", fontsize=8)
    fig.savefig(out / "path_by_string.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    top_b = next((v["stated"] for v in seed1.values() if v["stated"] is not None), 0.4)
    y_top = contexts.tokens[contexts.full_ids][int(np.argmax(tilted(log_ref, r, top_b)))]
    def pick(arm: str, decoder_rollout: bool) -> str | None:
        return next((k for k in panels if seed1[k]["arm"] == arm and ("decoder" in k) == decoder_rollout
                     and "λ" not in k and ("comp" in k or arm != "trafl")), None)

    show = [("reference\nread by random order", keep["cond_ref"], False)]
    show += [(f"{k}\nread by random order", seed1[k]["cond"], False) for k in (pick("trafl", False), pick("tb", False)) if k]
    show += [("reference\nread by the decoder", keep["cond_ref"], True)]
    show += [(f"{k}\nread by the decoder", seed1[k]["cond"], True)
             for k in (pick("trafl", False), pick("trafl", True), pick("espo", False), pick("espo", True)) if k]
    fig, axes = plt.subplots(2, 4, figsize=(12, 6), squeeze=False)
    for ax, (title, cond, dec) in zip(axes.flat, show):
        h = reveal_heatmap(contexts, cond, y_top, dec)
        im = ax.imshow(h, cmap="Blues", vmin=0, vmax=1, aspect="equal")
        for t in range(LENGTH):
            for i in range(LENGTH):
                if h[t, i] >= 0.2:
                    ax.text(i, t, f"{h[t, i]:.2f}", ha="center", va="center", fontsize=5.5,
                            color="white" if h[t, i] > 0.6 else "#1a1a1a")
        ax.set_title(title, fontsize=6.5)
        ax.set_xlabel("position", fontsize=7)
        ax.set_ylabel("reveal step", fontsize=7)
        ax.tick_params(labelsize=6)
    for ax in axes.flat[len(show):]:
        ax.axis("off")
    fig.colorbar(im, ax=axes, shrink=0.6, label="P(step reveals position | y)")
    letters = "".join("ACGT"[t] for t in y_top)
    fig.suptitle(f"Order law given the top target string y = {letters} (uniform order = 0.125 everywhere)", fontsize=9)
    fig.savefig(out / "orders.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--focus-beta", type=float, default=0.5, help="beta of the configurations shown in figures")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows, keep = analyse(args.root)
    text, summary = table(rows)
    (args.out / "tilt.md").write_text(text)
    (args.out / "tilt.json").write_text(json.dumps({"runs": rows, "configurations": summary}, indent=1) + "\n")
    figures(keep, args.out, args.focus_beta)
    print(text)


if __name__ == "__main__":
    main()
