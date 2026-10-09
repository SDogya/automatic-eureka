"""Shared pieces: a TraFL run, its target, training-distribution batches and exact posteriors.

Target with reference p_ref (step-0 weights), r = ln R and the run's TraFL beta:
    p*(tau | x) = p_ref(tau | x) exp(beta r(y)) / Z(x),  so  p*(tau | x, y) = p_ref(tau | x, y).
"""

import json
from functools import cache
from itertools import permutations
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from ..config import LENGTH, MASK, Config
from ..data import build_target, token_ids
from ..metrics.trajectories import Model, log_p_all_y, posterior_over_orders
from ..model import complete, forward, load_parameters

RESULTS = Path("results/hypotheses")
EXPERIMENT = "uniform_contexts"  # figures go to RESULTS / EXPERIMENT
# The two runs the hypotheses are reported on (k = 5): pretrained start seed 1, random start seed 2;
# beta = 1.5, K = 32 masks, constant learning rate, stopped on a loss plateau, prompts drawn
# uniformly (see TraflConfig.context_source). Earlier model-generated prompts: models/trafl_b15_k32.
SELECTED = {"finetune": Path("models/trafl_uniform_x/seed_1/finetune_5"),
            "random_init": Path("models/trafl_uniform_x/seed_2/random_init_5")}


def _last_saved(path: Path) -> int:
    return max(int(p.stem.removeprefix("step_")) for p in path.glob("step_*.npz"))


@cache
def last_step() -> int:
    """Common horizon of the selected runs: the earlier of their last checkpoints."""
    return min(_last_saved(path) for path in SELECTED.values())


class Run(NamedTuple):
    path: Path
    config: Config
    reference: Model
    log_reward: np.ndarray  # beta * r(y), r = ln R, for all 4^8 strings (lexicographic); beta = TraFL beta
    beta: float = 1.0

    @property
    def label(self) -> str:
        return f"{self.path.parent.name}_{self.path.name}"

    def steps(self) -> list[int]:
        steps = (int(p.stem.removeprefix("step_")) for p in self.path.glob("step_*.npz"))
        return sorted(s for s in steps if s <= last_step())

    def model(self, step: int) -> Model:
        return Model(forward, load_parameters(self.path / f"step_{step:05d}.npz"))


def load_run(path: Path) -> Run:
    settings = json.loads((path / "settings.json").read_text())
    config = Config.model_validate_json(json.dumps(settings["config"]))
    beta = settings["segments"][-1]["trafl"]["beta"]
    return Run(path, config, Model(forward, load_parameters(path / "step_00000.npz")),
               beta * build_target(config).log_reward, beta)


def pick_steps(run: Run, requested: list[int] | None) -> list[int]:
    """Requested steps, or a log-spaced default: 0, 250, 500, 1000, 2000, 5000, 10000, ..., last."""
    available = run.steps()
    if requested:
        missing = set(requested) - set(available)
        if missing:
            raise ValueError(f"No checkpoints for steps {sorted(missing)} in {run.path}")
        return requested
    wanted = [0, 250, 500, 1000, 2000, 5000, 10000, 15000, 20000, available[-1]]
    return sorted({s for s in wanted if s in available})


class Batch(NamedTuple):
    contexts: np.ndarray  # (C, L) prompts x
    completions: np.ndarray  # (C, G, L) finals y
    orders: np.ndarray  # (C, G, L) reveal orders (tau), -1 after the last step


def training_batch(model: Model, key: jax.Array, contexts: int = 32, group: int = 4,
                   mask_probability: float = 0.5) -> Batch:
    """x, y, tau as in TraFL training: the model's own strings with each position hidden
    with probability 0.5 (at least one), then `group` completions of each prompt."""
    base_key, mask_key, fill_key = jax.random.split(key, 3)
    base, _ = complete(model.params, jnp.full((contexts, LENGTH), MASK, jnp.int32), base_key)
    hide = jax.random.bernoulli(mask_key, mask_probability, base.shape)
    hide = jnp.where(hide.any(axis=1, keepdims=True), hide, True)
    prompts = jnp.where(hide, MASK, base)
    finals, orders = complete(model.params, jnp.repeat(prompts, group, axis=0), fill_key)
    return Batch(np.asarray(prompts), np.asarray(finals).reshape(contexts, group, LENGTH),
                 np.asarray(orders).reshape(contexts, group, LENGTH))


@cache
def _order_index(hidden: tuple[int, ...]) -> dict[tuple[int, ...], int]:
    """Row of each reveal order in posterior_over_orders' enumeration (itertools order)."""
    return {order: i for i, order in enumerate(permutations(hidden))}


def order_row(prompt: np.ndarray, order: np.ndarray) -> int:
    hidden = tuple(int(i) for i in np.flatnonzero(prompt == MASK))
    return _order_index(hidden)[tuple(int(i) for i in order[:len(hidden)])]


def log_target_y(run: Run, prompt: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """All completions y of x, log p_ref(y | x) and exact log p*(y | x) = log p_ref + r - log Z."""
    ys, log_ref = log_p_all_y(run.reference, prompt)
    log_star = log_ref + run.log_reward[token_ids(ys)]
    return ys, log_ref, log_star - np.logaddexp.reduce(log_star)


def posterior_pair(run: Run, model: Model, prompt: np.ndarray, y: np.ndarray):
    """Exact posteriors over reveal orders for the model and the reference."""
    return posterior_over_orders(model, prompt, y), posterior_over_orders(run.reference, prompt, y)
