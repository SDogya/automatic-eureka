"""Does a TraFL-trained model reach the trajectory-balance target? Exact chain-rule decomposition.

Target with reference p_ref (the step-0 model) and r = ln R:
    p*(tau | x) = p_ref(tau | x) exp(r(y)) / Z(x),
so p*(y | x) = p_ref(y | x) exp(r(y)) / Z(x) and p*(tau | x, y) = p_ref(tau | x, y). Then
    KL(p*(tau|x) || p(tau|x)) = KL(p*(y|x) || p(y|x)) + E_{y~p*} KL(p_ref(tau|x,y) || p(tau|x,y)).
"""

import json
import time
from pathlib import Path

import numpy as np

from ..config import MASK, Config
from ..data import build_target, token_ids
from ..model import forward, load_parameters
from .trajectories import Model, log_p_all_y, posterior_over_orders


def prompt_fit(reference: Model, trained: Model, prompt: np.ndarray, log_reward: np.ndarray,
               mass: float = 0.99, max_y: int = 400) -> dict:
    ys, log_ref = log_p_all_y(reference, prompt)
    _, log_p = log_p_all_y(trained, prompt)
    log_star = log_ref + log_reward[token_ids(ys)]
    log_star -= np.logaddexp.reduce(log_star)
    star = np.exp(log_star)
    order = np.argsort(-star)
    count = min(int(np.searchsorted(np.cumsum(star[order]), mass)) + 1, max_y, len(ys))
    chosen = order[:count]
    path_kl, path_tv = np.zeros(count), np.zeros(count)
    for n, index in enumerate(chosen):
        ref_post = posterior_over_orders(reference, prompt, ys[index]).log_posterior
        new_post = posterior_over_orders(trained, prompt, ys[index]).log_posterior
        path_kl[n] = np.exp(ref_post) @ (ref_post - new_post)
        path_tv[n] = np.abs(np.exp(ref_post) - np.exp(new_post)).sum() / 2
    covered = float(star[chosen].sum())
    weights = star[chosen] / covered
    kl_y = float(star @ (log_star - log_p))
    kl_paths = float(weights @ path_kl)
    result = {
        "prompt": "".join("?" if c == MASK else "ACGT"[c] for c in prompt), "hidden": int((prompt == MASK).sum()),
        "kl_y": kl_y, "kl_paths_given_y": kl_paths, "kl_tau": kl_y + kl_paths,
        "tv_y": float(np.abs(star - np.exp(log_p)).sum() / 2),
        "tv_paths_given_y": float(weights @ path_tv), "max_kl_paths_given_y": float(path_kl.max()),
        "y_evaluated": count, "target_mass_covered": covered,
    }
    return result


def direct_kl_tau(reference: Model, trained: Model, prompt: np.ndarray, log_reward: np.ndarray) -> float:
    """KL over every trajectory directly (no decomposition), for a sanity check on small prompts."""
    ys, log_ref = log_p_all_y(reference, prompt)
    log_z = np.logaddexp.reduce(log_ref + log_reward[token_ids(ys)])
    total = 0.0
    for index, y in enumerate(ys):
        ref_tau = posterior_over_orders(reference, prompt, y).log_p_tau
        new_tau = posterior_over_orders(trained, prompt, y).log_p_tau
        star_tau = ref_tau + log_reward[token_ids(y[None])[0]] - log_z
        total += float(np.exp(star_tau) @ (star_tau - new_tau))
    return total


def run_trajectory_fit(config: Config, runs: list[Path], checkpoint: str, output: Path,
                       prompts: int = 4, hidden: int = 5) -> list[dict]:
    output.mkdir(parents=True, exist_ok=True)
    target = build_target(config)
    generator = np.random.default_rng(0)
    prompt_list = [np.full(8, MASK, dtype=np.int32)]
    for _ in range(prompts):
        prompt = target.tokens[generator.choice(len(target.pi), p=target.pi)].copy()
        prompt[generator.choice(8, hidden, replace=False)] = MASK
        prompt_list.append(prompt)
    began = time.time()
    reports = []
    for run in runs:
        reference = Model(forward, load_parameters(run / "step_000.npz"))
        trained = Model(forward, load_parameters(run / checkpoint))
        row = {"run": str(run), "checkpoint": checkpoint, "prompts": []}
        for prompt in prompt_list:
            fit = prompt_fit(reference, trained, prompt, target.log_reward)
            row["prompts"].append(fit)
            print(f"[{run.parent.name}/{run.name}] x={fit['prompt']}  KL_tau={fit['kl_tau']:.4f} = "
                  f"KL_y {fit['kl_y']:.4f} + E KL_paths {fit['kl_paths_given_y']:.4f}  "
                  f"TV_y={fit['tv_y']:.3f}  TV_paths={fit['tv_paths_given_y']:.3f}  "
                  f"({fit['y_evaluated']} y, {fit['target_mass_covered']:.3f} mass, {time.time() - began:.0f}s)",
                  flush=True)
        reports.append(row)
        (output / f"trajectory_fit_{checkpoint.removesuffix('.npz')}.json").write_text(
            json.dumps(reports, indent=2) + "\n")
    return reports


if __name__ == "__main__":
    import argparse

    from ..config import load_config

    parser = argparse.ArgumentParser(description="Exact chain-rule checks for one TraFL run folder")
    parser.add_argument("run", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/potts_tfbind8.json"))
    parser.add_argument("--checkpoint", default="step_3000.npz")
    args = parser.parse_args()
    run_trajectory_fit(load_config(args.config), [args.run], args.checkpoint, args.run)
