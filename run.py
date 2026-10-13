"""CLI entry point: python run.py --config config.yaml --mode train|eval|smoke.

The original orchestration is preserved in trafl.runner.main. This file only
imports and invokes it, including the usual __main__ guard so DataLoader worker
startup does not execute the training loop. Checkpoint/resume CLI is unchanged.
"""

from trafl.runner import main


if __name__ == "__main__":
    main()
