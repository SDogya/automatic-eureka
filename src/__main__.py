"""CLI: python -m src {run,generate,train,sample}."""

import argparse
import os
from pathlib import Path

from .config import SampleRequest, decode, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "generate", "train", "sample"))
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    parser.add_argument("--overwrite", action="store_true", help="Replace existing results")
    parser.add_argument("--context", default="????????", help="Eight A/C/G/T/? symbols")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--seed", type=int, help="Optional sampling seed")
    parser.add_argument("--checkpoint", type=Path, help="Defaults to data_dir/best.npz")
    args = parser.parse_args()
    config = load_config(args.config)
    os.environ["JAX_PLATFORMS"] = "cpu"
    if args.command in ("run", "generate"):
        from .data import generate, load_dataset

        if args.command == "run" and not args.overwrite and (config.data_dir / "data8.parquet").exists():
            load_dataset(config)
            print("Reusing validated dataset", flush=True)
        else:
            generate(config, overwrite=args.overwrite)
    if args.command in ("run", "train"):
        from .train import train

        report = train(config, overwrite=args.overwrite)
        print(f"Best validation checkpoint: epoch {report.best_epoch}", flush=True)
    if args.command == "sample":
        from .model import load_parameters, sample

        request = SampleRequest(context=args.context, count=args.count,
                                seed=config.seed if args.seed is None else args.seed)
        params = load_parameters(args.checkpoint or config.data_dir / "best.npz")
        print("\n".join(decode(sample(params, request))))


if __name__ == "__main__":
    main()
