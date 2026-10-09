"""One sweep job: TraFL from the Potts checkpoint and from random weights, for one seed and width.

python -m src.rl.sweep --config configs/potts_tfbind8.json --seed 1 --exponent 7 --output models/trafl
"""

import argparse
import os
from pathlib import Path

from ..config import load_config
from ..data import tfbind8_scores
from .train import LogFn, TraflConfig, train_trafl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--exponent", type=int, required=True)
    parser.add_argument("--steps", type=int, default=30000, help="Upper bound; training stops on a loss plateau")
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--schedule", choices=("constant", "cosine"), default="constant")
    parser.add_argument("--start-models", type=Path, default=Path("models/pretrain"))
    parser.add_argument("--start", type=Path, help="Any checkpoint (MLP or transformer) as start and reference; "
                        "overrides --start-models/mlp_<exponent>; implies --only finetune")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--beta", type=float, default=1.5, help="TraFL beta: target p_ref(tau|x) exp(beta r(y))")
    parser.add_argument("--mask-samples", type=int, default=32, help="K masks per completion in the surrogate")
    parser.add_argument("--plateau-window", type=int, default=2000)
    parser.add_argument("--context-source", choices=("uniform", "model"), default="uniform")
    parser.add_argument("--plateau-patience", type=int, default=2000)
    parser.add_argument("--no-plateau-stop", dest="stop_on_plateau", action="store_false")
    parser.add_argument("--only", choices=("finetune", "random_init"), help="Train one start only")
    parser.add_argument("--arm", choices=("trafl", "tb", "espo", "espo_ppo", "grpo", "justgrpo"), default="trafl")
    parser.add_argument("--mask-scheme", choices=("iid", "comp"), default="iid")
    parser.add_argument("--var-lambda", type=float, default=0.0, help="TraFL + across-order variance penalty")
    parser.add_argument("--kappa", type=float, default=0.05, help="ESPO k2 coefficient")
    parser.add_argument("--ppo-epochs", type=int, default=1, help="updates per rollout batch (ESPO's mu)")
    parser.add_argument("--eps-clip", type=float, default=0.2)
    parser.add_argument("--resume-from", type=Path, help="Folder with seed_s/<name>_k runs to continue")
    parser.add_argument("--resume-step", type=int)
    args = parser.parse_args()
    os.environ["JAX_PLATFORMS"] = "cpu"
    config = load_config(args.config)
    trafl = TraflConfig(steps=args.steps, save_every=args.save_every, schedule=args.schedule, seed=args.seed,
                        learning_rate=args.learning_rate, stop_on_plateau=args.stop_on_plateau,
                        beta=args.beta, mask_samples=args.mask_samples,
                        plateau_window=args.plateau_window, plateau_patience=args.plateau_patience,
                        context_source=args.context_source, arm=args.arm, mask_scheme=args.mask_scheme,
                        var_lambda=args.var_lambda, kappa=args.kappa, ppo_epochs=args.ppo_epochs,
                        eps_clip=args.eps_clip)
    root = args.output / f"seed_{args.seed}"
    log = LogFn(root / f"train_k{args.exponent}.log")
    scores = tfbind8_scores()
    starts = (args.start,) if args.start else (args.start_models / f"mlp_{args.exponent}" / "best.npz", None)
    for start in starts:
        tag = args.start.parent.name if args.start else str(args.exponent)
        name = f"{'finetune' if start else 'random_init'}_{tag}"
        if args.only and not name.startswith(args.only):
            continue
        resume = None if args.resume_from is None else (
            args.resume_from / f"seed_{args.seed}" / name, args.resume_step)
        report = train_trafl(config, trafl, name, 2**args.exponent, start, root / name, scores, log, resume)
        log(f"[{name}] done: KL(p*||p) {report.evaluations[0].kl:.4f} -> {report.evaluations[-1].kl:.4f}")


if __name__ == "__main__":
    main()
