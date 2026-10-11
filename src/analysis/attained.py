"""Attained tilt during training, and every metric at matched attained tilt (from evals.csv; no checkpoints needed).

python -m src.analysis.attained --root models/port_grid3 --out results/port_grid3 [--levels 0.1 0.2 0.3]

b*(t) solves E_{pi_b}[r] = E_{p_theta(t)}[r], with pi_b = p_ref e^{b r} / Z_b (src.analysis.tilt). It depends on the model
only through expected_log_reward, which every evaluation records. The distance from the tilt family follows from
kl_to_ref: KL(p_theta || pi_b*) = KL(p_theta || p_ref) - b* E_p[r] + log Z_b*.
At the final step these are asserted to reproduce tilt.json's checkpoint values when that file is present.
Matched readings take the first crossing of b* = level in training order (compare.matched_value), so like every
first-crossing comparison they read an arm passing through a tilt and an arm settling at it alike (REPORT §4b, confound 1);
the update at which the level is reached is reported next to them.
Writes attained.png (b*(t) and KL(p_theta || pi_b*)(t), seed means; stated tilts dashed) and matched_tilt.md.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .compare import matched_value  # noqa: E402
from .report import ARM_ORDER, PALETTE, short_label  # noqa: E402
from .runs import by_group, find_runs  # noqa: E402
from .tilt import CENTRED, RAW, load_reference  # noqa: E402

B_GRID = np.linspace(-2.0, 12.0, 14001)
METRICS = (("step", "update reached"), ("shape_kl", "KL(p ‖ π_b*)"), ("kl_to_ref", "KL(p ‖ p_ref)"),
           ("kl_path_ref", "path KL"), ("top_distinct16", "top-1 % distinct in 16, random"),
           ("llada_top_distinct16", "same, decoder"), ("expected_score", "score"),
           ("llada_expected_score", "score, decoder"))


def tilt_curve(log_ref: np.ndarray, r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """E_{pi_b}[r] and log Z_b on B_GRID (E increases in b)."""
    means, log_z = np.empty_like(B_GRID), np.empty_like(B_GRID)
    for k, b in enumerate(B_GRID):
        w = log_ref + b * r
        log_z[k] = np.logaddexp.reduce(w)
        means[k] = float(np.exp(w - log_z[k]) @ r)
    assert np.all(np.diff(means) > 0)
    return means, log_z


def attained(run, means: np.ndarray, log_z: np.ndarray) -> dict[str, np.ndarray]:
    e = dict(run.evals)
    m = e["expected_log_reward"]
    b = np.interp(m, means, B_GRID, left=np.nan, right=np.nan)
    e["b_star"] = b
    e["shape_kl"] = e["kl_to_ref"] - b * m + np.interp(b, B_GRID, log_z)
    e["step"] = e["step"].astype(float)
    return e


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--levels", type=float, nargs="+", default=(0.1, 0.2, 0.3))
    parser.add_argument("--focus-beta", type=float, default=0.5)
    args = parser.parse_args()
    runs = find_runs(args.root)
    _, settings, r, _, log_ref = load_reference(runs)
    means, log_z = tilt_curve(log_ref, r)
    vary_arch = len({x.architecture for x in runs}) > 1
    curves = {run.path: attained(run, means, log_z) for run in runs}
    stated = {}
    for run, s in zip(runs, settings):
        t = s["segments"][-1]["trafl"]
        stated[run.path] = (t["beta"] * (1 - 1 / t.get("group", 5)) if run.arm in CENTRED
                            else t["beta"] if run.arm in RAW else None)

    checks = Path(args.out) / "tilt.json"                       # the checkpoint analysis, if it was run
    if checks.exists():
        final = {(x["label"], x["seed"]): x for x in json.loads(checks.read_text())["runs"] if x["arm"] != "target"}
        for run in runs:
            x = final.get((short_label(run, vary_arch, False), run.seed))
            e = curves[run.path]
            if x and int(e["step"][-1]) == x["step"]:
                assert abs(e["b_star"][-1] - x["b_star"]) < 2e-3 and abs(e["shape_kl"][-1] - x["shape_kl"]) < 2e-3, \
                    (run.path, e["b_star"][-1], x["b_star"], e["shape_kl"][-1], x["shape_kl"])

    groups = sorted(by_group(runs).values(), key=lambda g: (ARM_ORDER.index(g[0].arm), short_label(g[0], vary_arch, False)))
    lines = ["Every metric at the first crossing of attained tilt b* = level (seed mean; '-' = never reached). "
             "b* from expected_log_reward; π_b = p_ref e^{b r} / Z_b.\n"]
    for level in args.levels:
        lines += [f"\n**b\\* = {level:g}**\n", "| configuration | stated b | " + " | ".join(t for _, t in METRICS) + " |",
                  "|---|---|" + "---|" * len(METRICS)]
        for g in groups:
            cells = []
            for key, _ in METRICS:
                v = [matched_value(curves[x.path]["b_star"], curves[x.path][key], level) for x in g
                     if key in curves[x.path]]
                v = [u for u in v if np.isfinite(u)]
                cells.append(f"{np.mean(v):.0f}" if v and key == "step" else f"{np.mean(v):.3g}" if v else "-")
            s0 = stated[g[0].path]
            lines.append(f"| {short_label(g[0], vary_arch, False)} | {'-' if s0 is None else f'{s0:g}'} | "
                         + " | ".join(cells) + " |")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "matched_tilt.md").write_text("\n".join(lines) + "\n")

    shown = [g for g in groups if stated[g[0].path] is None or f"β{args.focus_beta:g}" in short_label(g[0], vary_arch, False)]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    styles, used = ("-", "--", ":", "-.", (0, (5, 1)), (0, (1, 1)), (0, (3, 1, 1, 1))), {}
    for g in shown:
        arm, label = g[0].arm, short_label(g[0], vary_arch, False)
        steps = curves[g[0].path]["step"]
        b = np.nanmean([curves[x.path]["b_star"] for x in g], axis=0)
        shape = np.nanmean([curves[x.path]["shape_kl"] for x in g], axis=0)
        colour = PALETTE[ARM_ORDER.index(arm)]
        used[colour] = used.get(colour, -1) + 1                           # the palette repeats colours across arms
        kw = dict(color=colour, ls=styles[used[colour] % len(styles)], lw=1.3)
        axes[0].plot(steps[1:], b[1:], label=label, **kw)
        axes[1].plot(steps[1:], np.maximum(shape[1:], 1e-4), **kw)
    for value, text in ((args.focus_beta * 0.8, "stated, centred arms (0.8 β)"), (args.focus_beta, "stated, raw-reward arms (β)")):
        axes[0].axhline(value, color="#1a1a1a", lw=0.8, ls="--")
        axes[0].text(1.2, value * 1.04, text, fontsize=6.5)
    axes[0].set(xscale="log", yscale="log", ylim=(1e-2, 5), xlabel="update", ylabel="attained tilt b*")
    axes[1].set(xscale="log", yscale="log", xlabel="update", ylabel="KL(p_θ ‖ π_b*): distance from every tilt")
    for ax in axes:
        ax.grid(alpha=0.25, lw=0.5)
    axes[0].legend(fontsize=5.5, ncol=2, frameon=False, loc="upper left")
    fig.suptitle(f"Attained tilt during training (β {args.focus_beta:g} and arms without β; seed means)", fontsize=9)
    fig.tight_layout()
    fig.savefig(args.out / "attained.png", dpi=140)
    plt.close(fig)
    print((args.out / "matched_tilt.md").read_text()[:4000])


if __name__ == "__main__":
    main()
