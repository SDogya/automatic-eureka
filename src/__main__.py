"""Pretraining CLI: python -m src {generate,train,run,ablation} --config configs/potts.json.

`train` and `run` use config.architecture (default: the MLP); `ablation` runs the MLP width
ablation, or with --family transformer the transformer size ablation.
"""

import argparse
import os
from pathlib import Path

from .config import LENGTH, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "generate", "train", "ablation"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--family", choices=("mlp", "transformer"), default="mlp",
                        help="Architecture family of the ablation command")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing results")
    args = parser.parse_args()
    config = load_config(args.config)
    os.environ["JAX_PLATFORMS"] = "cpu"
    if args.command in ("run", "generate"):
        from .data import generate, load_dataset

        if args.command == "run" and not args.overwrite and (config.data_dir / f"data{LENGTH}.parquet").exists():
            load_dataset(config)
            print("Reusing validated dataset", flush=True)
        else:
            generate(config, overwrite=args.overwrite)
    if args.command == "ablation" and args.family == "transformer":
        from .ablation import run_transformer_ablation

        for size in run_transformer_ablation(config, overwrite=args.overwrite).rows:
            print(f"transformer_{size.tag}: params={size.parameter_count}, "
                  f"best KL={size.best_kl:.6f}, final KL={size.final_kl:.6f}, "
                  f"{size.wall_seconds:.0f} s", flush=True)
    elif args.command == "ablation":
        from .ablation import run_ablation

        for row in run_ablation(config, overwrite=args.overwrite).rows:
            print(f"mlp_{row.exponent}: params={row.parameter_count}, "
                  f"best KL={row.best_kl:.6f}, final KL={row.final_kl:.6f}", flush=True)
    if args.command in ("run", "train"):
        from .train import train

        report = train(config, overwrite=args.overwrite)
        print(f"Best validation checkpoint: epoch {report.best_epoch}", flush=True)


if __name__ == "__main__":
    main()
