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
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--schedule", choices=("constant", "cosine"), default="cosine")
    parser.add_argument("--start-models", type=Path, default=Path("models/pretrain"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--stop-on-plateau", action="store_true")
    parser.add_argument("--only", choices=("finetune", "random_init"), help="Train one start only")
    parser.add_argument("--resume-from", type=Path, help="Folder with seed_s/<name>_k runs to continue")
    parser.add_argument("--resume-step", type=int)
    args = parser.parse_args()
    os.environ["JAX_PLATFORMS"] = "cpu"
    config = load_config(args.config)
    trafl = TraflConfig(steps=args.steps, save_every=args.save_every, schedule=args.schedule, seed=args.seed,
                        learning_rate=args.learning_rate, stop_on_plateau=args.stop_on_plateau)
    root = args.output / f"seed_{args.seed}"
    log = LogFn(root / f"train_k{args.exponent}.log")
    scores = tfbind8_scores()
    for start in (args.start_models / f"mlp_{args.exponent}" / "best.npz", None):
        name = f"{'finetune' if start else 'random_init'}_{args.exponent}"
        if args.only and not name.startswith(args.only):
            continue
        resume = None if args.resume_from is None else (
            args.resume_from / f"seed_{args.seed}" / name, args.resume_step)
        report = train_trafl(config, trafl, name, 2**args.exponent, start, root / name, scores, log, resume)
        log(f"[{name}] done: KL(p*||p) {report.evaluations[0].kl:.4f} -> {report.evaluations[-1].kl:.4f}")


if __name__ == "__main__":
    main()
