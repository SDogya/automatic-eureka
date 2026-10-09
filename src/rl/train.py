"""Post-training of the masked denoiser from random weights or a checkpoint, with exact evaluation.

arm="trafl" is this repo's TraFL (unchanged defaults); the other arms are the baselines TraFL is compared with
(losses/baselines.py, gated in tests/test_arms.py) and the exact trajectory-balance loss ("tb", RTB-type)."""

import json
import time
from pathlib import Path
from typing import Literal, NamedTuple

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import optax
from pydantic import Field

from ..config import LENGTH, MASK, Config, Record, rng
from ..data import Target, build_target
from ..evaluation import Contexts, build_contexts, evaluate
from ..architecture import Architecture
from ..architecture import load as load_checkpoint
from ..architecture import save as save_checkpoint
from ..model import Layer, Params, forward
from ..metrics.decoders import proposal_terminal_log_probs
from ..metrics.paths import log_conditionals, path_kl
from .losses.baselines import ar_token_log_probs, espo_loss, grpo_loss, justgrpo_loss, order_variance
from .losses.masks import Scheme
from .losses.trafl import Normalization, Reference, trafl_residuals
from .losses.trajectory_balance import trajectory_residuals
from .samplers import complete_ar, complete_uniform

ZParams = tuple[Layer, Layer]


class TraflConfig(Record):
    steps: int = Field(default=50, ge=1)
    save_every: int = Field(default=5, ge=1)
    contexts: int = Field(default=256, ge=1)
    group: int = Field(default=5, ge=2)
    mask_samples: int = Field(default=32, ge=1)
    context_mask_probability: float = Field(default=0.5, gt=0, le=1)  # only for context_source="model"
    # "uniform": u ~ U{1..L} hidden positions chosen uniformly, revealed letters uniform (prompts
    # independent of the model); "model": the model's own string with each position hidden w.p. 0.5.
    context_source: Literal["uniform", "model"] = "uniform"
    learning_rate: float = Field(default=1e-3, gt=0)
    schedule: Literal["constant", "cosine"] = "constant"
    final_lr_fraction: float = Field(default=0.01, gt=0, le=1)
    log_z_learning_rate: float = Field(default=1e-2, gt=0)
    log_z_width: int = Field(default=32, ge=1)
    beta: float = Field(default=1.5, gt=0)
    normalization: Normalization = "paper"
    reference: Literal["initial", "uniform"] = "initial"
    # arm and its knobs: beta (trafl, tb), kappa (espo, espo_ppo), none (grpo, justgrpo); var_lambda adds TraFL's
    # explicit across-order variance penalty; ppo_epochs = policy updates per rollout batch (ESPO's mu)
    arm: Literal["trafl", "tb", "espo", "espo_ppo", "grpo", "justgrpo"] = "trafl"
    mask_scheme: Scheme = "iid"
    var_lambda: float = Field(default=0.0, ge=0)
    kappa: float = Field(default=0.05, ge=0)
    ppo_epochs: int = Field(default=1, ge=1)
    eps_clip: float = Field(default=0.2, gt=0)
    stop_on_plateau: bool = True
    plateau_window: int = Field(default=2000, ge=1)
    plateau_patience: int = Field(default=2000, ge=1)
    plateau_tolerance: float = Field(default=0.01, ge=0)
    seed: int = 42


class StepMetric(Record):
    step: int
    loss: float
    delta_mean: float
    delta_abs: float
    log_z_mean: float
    rollout_score: float
    hidden_mean: float
    grad_norm: float
    learning_rate: float
    seconds: float


class EvalMetric(Record):
    step: int
    kl: float
    kl_to_pi: float
    reverse_kl: float
    tv: float
    expected_score: float
    probability_sum: float
    kl_to_ref: float = float("nan")          # KL(p_theta || p_ref), terminal, empty prompt
    kl_from_ref: float = float("nan")        # KL(p_ref || p_theta)
    kl_traj_ref: float = float("nan")        # KL(P_ref || P_theta) over trajectories
    kl_path_ref: float = float("nan")        # kl_traj_ref - kl_from_ref = E_{y~p_ref} KL(c_ref(tau|y) || c_theta(tau|y))
    expected_log_reward: float = float("nan")
    entropy: float = float("nan")
    ar_expected_score: float = float("nan")  # the same under the left-to-right decoder (JustGRPO's policy)
    ar_kl_to_ref: float = float("nan")       # KL(p_AR,theta || p_ref)
    # low-confidence-remasking decoder (LLaDA / Fast-dLLM, one token per step) at T = DECODER_T: the realistic decoder
    llada_expected_score: float = float("nan")
    llada_expected_log_reward: float = float("nan")
    llada_entropy: float = float("nan")
    llada_kl_to_ref: float = float("nan")    # KL(p_dec,theta || p_dec,ref), both under the same decoder


class TraflReport(Record):
    name: str
    init: Literal["random", "finetune"]
    hidden_width: int
    config: Config
    trafl: TraflConfig
    steps: list[StepMetric]
    evaluations: list[EvalMetric]


def init_log_z(key: jax.Array, width: int) -> ZParams:
    limit = np.sqrt(6 / (LENGTH * (MASK + 1) + width))
    weight = jax.random.uniform(key, (LENGTH * (MASK + 1), width), minval=-limit, maxval=limit)
    return (Layer(weight, jnp.zeros(width)), Layer(jnp.zeros((width, 1)), jnp.zeros(1)))


def log_z_forward(params: ZParams, contexts: jax.Array) -> jax.Array:
    hidden = jax.nn.one_hot(contexts, MASK + 1).reshape(len(contexts), -1)
    hidden = jax.nn.gelu(hidden @ params[0].weight + params[0].bias, approximate=False)
    return (hidden @ params[1].weight + params[1].bias)[:, 0]


def tilted_target(target: Target, log_p_reference: np.ndarray, beta: float) -> Target:
    """p*(y) proportional to p_ref(y) exp(beta r(y)), r = ln R: TraFL's target from the empty prompt."""
    log_weight = log_p_reference + beta * target.log_reward
    log_pi = log_weight - np.logaddexp.reduce(log_weight)
    return target._replace(log_pi=log_pi, pi=np.exp(log_pi))


DECODER_T = 0.6  # TraFL's evaluation temperature


class ReferenceTables(NamedTuple):
    log_cond: np.ndarray     # log q_ref(a | s, i) for every partial string
    log_p: np.ndarray        # log p_ref(y), empty prompt, random-order decoder
    log_p_llada: np.ndarray  # log p_ref(y) under the low-confidence-remasking decoder at DECODER_T


def ar_log_probs(forward_fn: object, params: object, full_tokens: np.ndarray, batch: int = 8192) -> np.ndarray:
    """Exact log p_AR(y) of every full string under the left-to-right decoder from the empty prompt."""
    empty = jnp.full((1, LENGTH), MASK, dtype=jnp.int32)
    fn = jax.jit(lambda p, y: ar_token_log_probs(forward_fn, p, jnp.broadcast_to(empty, y.shape), y).values.sum(-1))
    return np.concatenate([np.asarray(fn(params, jnp.asarray(full_tokens[s:s + batch])), dtype=np.float64)
                           for s in range(0, len(full_tokens), batch)])


def exact_evaluation(step: int, params: Params, contexts: Contexts, target: Target, pi: Target,
                     scores: np.ndarray, validation: np.ndarray, scratch: Path,
                     forward_fn: object = forward, ref: "ReferenceTables | None" = None) -> tuple[EvalMetric, np.ndarray]:
    """KL etc. against TraFL's target; kl_to_pi is against the reward distribution pi itself. With ref: terminal
    and path distances from the reference, and the left-to-right decoder's score and distance."""
    result = evaluate(params, contexts, target, validation, 0.5, 8192, scratch, forward=forward_fn)
    model = np.exp(result.log_probs)
    extra = {}
    if ref is not None:
        log_cond = log_conditionals(forward_fn, params, contexts)
        paths = path_kl(contexts, ref.log_cond, log_cond)
        log_ar = ar_log_probs(forward_fn, params, pi.tokens)
        log_dec = proposal_terminal_log_probs(contexts, log_cond, DECODER_T)
        dec = np.exp(log_dec)
        finite = dec > 0
        extra = dict(kl_to_ref=float(model @ (result.log_probs - ref.log_p)), kl_from_ref=paths.kl_term,
                     kl_traj_ref=paths.kl_traj, kl_path_ref=paths.kl_path,
                     expected_log_reward=float(model @ pi.log_reward), entropy=float(-model @ result.log_probs),
                     ar_expected_score=float(np.exp(log_ar) @ scores),
                     ar_kl_to_ref=float(np.exp(log_ar) @ (log_ar - ref.log_p)),
                     llada_expected_score=float(dec @ scores), llada_expected_log_reward=float(dec @ pi.log_reward),
                     llada_entropy=float(-dec[finite] @ log_dec[finite]),
                     llada_kl_to_ref=float(dec[finite] @ (log_dec[finite] - ref.log_p_llada[finite])))
    return EvalMetric(
        step=step, kl=result.kl, kl_to_pi=float(pi.pi @ (pi.log_pi - result.log_probs)),
        reverse_kl=float(model @ (result.log_probs - target.log_pi)),
        tv=float(np.abs(model - target.pi).sum() / 2),
        expected_score=float(model @ scores), probability_sum=result.probability_sum, **extra,
    ), result.log_probs


def load_log_z(path: Path) -> ZParams:
    with np.load(path) as saved:
        return tuple(Layer(jnp.asarray(saved[f"w{i}"]), jnp.asarray(saved[f"b{i}"])) for i in range(2))


def write_settings(output_dir: Path, name: str, init: str, width: int, config: Config,
                   trafl: TraflConfig, start: Path | None, first: int) -> None:
    """settings.json: the run identity plus one segment per (re)start of training."""
    path = output_dir / "settings.json"
    settings = json.loads(path.read_text()) if path.exists() else {
        "name": name, "init": init, "hidden_width": width, "start": None if start is None else str(start),
        "config": config.model_dump(mode="json"), "segments": []}
    settings["segments"].append({"first_step": first, "trafl": trafl.model_dump(mode="json"),
                                 **({"note": "resumed; Adam moments restarted"} if first else {})})
    path.write_text(json.dumps(settings, indent=2) + "\n")


def plateaued(steps: list["StepMetric"], trafl: TraflConfig) -> bool:
    """Mean loss over the last window improved by less than the tolerance versus `patience` steps ago."""
    window, patience = trafl.plateau_window, trafl.plateau_patience
    if len(steps) < window + patience:
        return False
    loss = np.array([m.loss for m in steps])
    recent, earlier = loss[-window:].mean(), loss[-window - patience:-patience].mean()
    return bool(recent > (1 - trafl.plateau_tolerance) * earlier)


def train_trafl(config: Config, trafl: TraflConfig, name: str, width: int,
                start: Path | None, output_dir: Path, scores: np.ndarray,
                log: "LogFn", resume: tuple[Path, int] | None = None,
                architecture: Architecture | None = None) -> TraflReport:
    """resume=(run_dir, step) continues that run: policy and log Z from the step checkpoint,
    reference from its step 0, step numbering continued; Adam moments restart."""
    output_dir.mkdir(parents=True, exist_ok=True)
    init: Literal["random", "finetune"] = "random" if start is None else "finetune"
    key = jax.random.fold_in(jax.random.key(trafl.seed), width)
    key, policy_key, z_key = jax.random.split(key, 3)
    first = 0
    # any architecture: a checkpoint carries its own (architecture.load); random init uses `architecture`, default
    # the MLP of this width (same key use as before the port, so MLP runs reproduce)
    if resume is None:
        if start is None:
            arch = architecture if architecture is not None else Architecture.mlp(width)
            initial = arch.init(policy_key)
        else:
            arch, initial = load_checkpoint(start)
        params = (initial, init_log_z(z_key, trafl.log_z_width))
    else:
        run_dir, first = resume
        arch, initial = load_checkpoint(run_dir / "step_00000.npz")
        params = (load_checkpoint(run_dir / f"step_{first:05d}.npz")[1],
                  load_log_z(run_dir / f"log_z_step_{first:05d}.npz"))
        key = jax.random.fold_in(key, first)
    forward = arch.forward  # noqa: F811  (local: every closure below uses the run's architecture)
    reference = Reference(forward, initial) if trafl.reference == "initial" else None
    if reference is None and (trafl.arm in ("espo", "espo_ppo") or trafl.var_lambda):
        raise ValueError(f"arm {trafl.arm} with var_lambda={trafl.var_lambda} needs reference='initial'")
    def rate(peak: float) -> optax.Schedule:
        if trafl.schedule == "constant":
            return optax.constant_schedule(peak)
        return optax.cosine_decay_schedule(peak, trafl.steps, alpha=trafl.final_lr_fraction)

    policy_rate = rate(trafl.learning_rate)

    optimizer = optax.multi_transform(
        {"policy": optax.adam(policy_rate), "log_z": optax.adam(rate(trafl.log_z_learning_rate))},
        ("policy", "log_z"))
    state = optimizer.init(params)
    pi = build_target(config)
    target = pi
    log_reward = jnp.asarray(target.log_reward, dtype=jnp.float32)
    score_table = jnp.asarray(scores, dtype=jnp.float32)
    powers = jnp.asarray(4 ** np.arange(LENGTH - 1, -1, -1, dtype=np.int32))
    contexts_all = build_contexts()
    validation = pi.tokens[rng(trafl.seed, 7).choice(len(pi.pi), 256, p=pi.pi)]
    ref_tables = None
    if reference is not None:
        _, log_p_initial = exact_evaluation(0, initial, contexts_all, pi, pi, scores, validation, output_dir,
                                            forward_fn=forward)
        target = tilted_target(pi, log_p_initial, trafl.beta)
        ref_cond = log_conditionals(forward, initial, contexts_all)
        ref_tables = ReferenceTables(ref_cond, log_p_initial,
                                     proposal_terminal_log_probs(contexts_all, ref_cond, DECODER_T))
    write_settings(output_dir, name, init, width, config, trafl, start, first)
    log(f"[{name}] init={init} architecture={arch.spec.model_dump()} start={start} reference={trafl.reference} "
        f"H(target)={float(-target.pi @ target.log_pi):.3f} E_target[y]={float(target.pi @ scores):.4f} -> {output_dir}")

    sampler = complete_ar if trafl.arm == "justgrpo" else complete_uniform

    @jax.jit
    def rollout(policy: Params, base_key: jax.Array, mask_key: jax.Array, fill_key: jax.Array):
        """Prompts (same distribution for every arm) and G completions each from the arm's own sampler."""
        shape = (trafl.contexts, LENGTH)
        if trafl.context_source == "uniform":
            count_key, order_key = jax.random.split(mask_key)
            base = jax.random.randint(base_key, shape, 0, 4, dtype=jnp.int32)
            hidden_count = jax.random.randint(count_key, (trafl.contexts, 1), 1, LENGTH + 1)
            rank = jnp.argsort(jnp.argsort(jax.random.uniform(order_key, shape), axis=1), axis=1)
            hide = rank < hidden_count
        else:
            base, _ = sampler(forward, policy, jnp.full(shape, MASK, dtype=jnp.int32), base_key)
            hide = jax.random.bernoulli(mask_key, trafl.context_mask_probability, shape)
            hide = jnp.where(hide.any(axis=1, keepdims=True), hide, True)
        contexts = jnp.where(hide, MASK, base)
        repeated = jnp.repeat(contexts, trafl.group, axis=0)
        completions, orders = sampler(forward, policy, repeated, fill_key)
        shape3 = (trafl.contexts, trafl.group, LENGTH)
        return contexts, completions.reshape(shape3), orders.reshape(shape3), hide.sum(axis=1)

    @jax.jit
    def update(params: tuple[Params, ZParams], state: optax.OptState, batch: tuple, key: jax.Array, old: Params):
        contexts, completions, orders, hidden = batch
        ids = completions @ powers
        rewards = log_reward[ids]
        loss_key, var_key = key, jax.random.fold_in(key, 7)  # loss_key as before the port: TraFL runs reproduce

        def loss_fn(p: tuple[Params, ZParams]):
            zero = jnp.zeros(rewards.shape)
            if trafl.arm == "trafl":
                beta = trafl.beta / hidden  # 1/l surrogate is log p / u, so beta / u keeps the target exp(beta r)
                delta = trafl_residuals(p, forward, log_z_forward, contexts, completions, rewards, loss_key, beta,
                                        reference=reference, samples=trafl.mask_samples,
                                        normalization=trafl.normalization, scheme=trafl.mask_scheme)
                loss = jnp.mean(delta**2)
            elif trafl.arm == "tb":  # exact log p(tau) is extensive: beta unscaled
                delta = trajectory_residuals(p, forward, log_z_forward, contexts, completions, orders, rewards,
                                             trafl.beta, reference=reference)
                loss = jnp.mean(delta**2)
            elif trafl.arm in ("espo", "espo_ppo"):
                delta = zero
                loss = espo_loss(p[0], forward, contexts, completions, rewards, loss_key, kappa=trafl.kappa,
                                 reference=reference, old=old if trafl.arm == "espo_ppo" else None,
                                 samples=trafl.mask_samples, scheme=trafl.mask_scheme, eps_clip=trafl.eps_clip)
            elif trafl.arm == "grpo":
                delta = zero
                loss = grpo_loss(p[0], forward, contexts, completions, orders, rewards)
            else:
                delta = zero
                loss = justgrpo_loss(p[0], forward, contexts, completions, rewards,
                                     old=old if trafl.ppo_epochs > 1 else None, eps_clip=trafl.eps_clip)
            if trafl.var_lambda:
                loss = loss + trafl.var_lambda * jnp.mean(order_variance(p[0], forward, reference, contexts,
                                                                         completions, var_key))
            return loss, delta

        (loss, delta), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, state = optimizer.update(grads, state, params)
        stats = {"loss": loss, "delta_mean": delta.mean(), "delta_abs": jnp.abs(delta).mean(),
                 "log_z_mean": log_z_forward(params[1], contexts).mean(),
                 "rollout_score": score_table[ids].mean(), "hidden_mean": hidden.mean(),
                 "grad_norm": optax.global_norm(grads[0])}
        return optax.apply_updates(params, updates), state, stats

    def train_step(params: tuple[Params, ZParams], state: optax.OptState, key: jax.Array):
        """One rollout batch, then ppo_epochs updates on it against the policy that sampled it."""
        base_key, mask_key, fill_key, loss_key = jax.random.split(key, 4)  # the pre-port split
        old = params[0]
        batch = rollout(old, base_key, mask_key, fill_key)
        for epoch in range(trafl.ppo_epochs):
            update_key = loss_key if epoch == 0 else jax.random.fold_in(loss_key, epoch)
            params, state, stats = update(params, state, batch, update_key, old)
        return params, state, stats

    steps: list[StepMetric] = []
    evaluations: list[EvalMetric] = []

    step_log = CsvLog(output_dir / "steps.csv", list(StepMetric.model_fields))
    eval_log = CsvLog(output_dir / "evals.csv", list(EvalMetric.model_fields))

    def checkpoint(step: int) -> None:
        save_checkpoint(output_dir / f"step_{step:05d}.npz", arch, params[0])
        np.savez(output_dir / f"log_z_step_{step:05d}.npz",
                 **{f"{n}{i}": np.asarray(a) for i, layer in enumerate(params[1]) for n, a in zip("wb", layer)})
        began = time.time()
        metric, _ = exact_evaluation(step, params[0], contexts_all, target, pi, scores, validation, output_dir,
                                     forward_fn=forward, ref=ref_tables)
        evaluations.append(metric)
        eval_log.write(metric.model_dump())
        plot_curves(name, steps, evaluations, output_dir / "curves.png")
        log(f"[{name}] eval step {step:3d}: KL(p*||p)={metric.kl:.4f}  KL(p||p*)={metric.reverse_kl:.4f}  "
            f"KL(pi||p)={metric.kl_to_pi:.4f}  "
            f"TV={metric.tv:.4f}  E_p[y]={metric.expected_score:.4f}  sum p={metric.probability_sum:.12f}  "
            f"({time.time() - began:.1f}s)")

    checkpoint(first)
    for step in range(first + 1, first + trafl.steps + 1):
        key, step_key = jax.random.split(key)
        began = time.time()
        params, state, stats = train_step(params, state, step_key)
        values = {k: float(v) for k, v in stats.items()}
        if not np.isfinite(values["loss"]):
            raise FloatingPointError(f"[{name}] non-finite loss at step {step}")
        metric = StepMetric(step=step, seconds=time.time() - began,
                            learning_rate=float(policy_rate(step - first - 1)), **values)
        steps.append(metric)
        step_log.write(metric.model_dump())
        stop = trafl.stop_on_plateau and step % trafl.save_every == 0 and plateaued(steps, trafl)
        if step % trafl.save_every == 0 or step == first + trafl.steps:
            checkpoint(step)
        if stop:
            log(f"[{name}] loss plateau at step {step}: mean of last {trafl.plateau_window} steps improved "
                f"< {trafl.plateau_tolerance:.0%} over {trafl.plateau_patience} steps; stopping")
            break
    return TraflReport(name=name, init=init, hidden_width=width, config=config, trafl=trafl,
                       steps=steps, evaluations=evaluations)


class CsvLog:
    """Append-only CSV, flushed after every row so it can be read during training."""

    def __init__(self, path: Path, fields: list[str]) -> None:
        self.fields = fields
        self.file = path.open("w")
        self.file.write(",".join(fields) + "\n")
        self.file.flush()

    def write(self, row: dict) -> None:
        self.file.write(",".join(repr(row[f]) for f in self.fields) + "\n")
        self.file.flush()


def plot_curves(name: str, steps: list[StepMetric], evals: list[EvalMetric], path: Path) -> None:
    """Live training curves: every logged step and every exact evaluation."""
    figure, axes = plt.subplots(2, 3, figsize=(15, 7.5))
    if steps:
        x = np.array([m.step for m in steps])
        loss = np.array([m.loss for m in steps])
        window = max(1, len(loss) // 50)
        smooth = np.convolve(loss, np.ones(window) / window, mode="valid")
        axes[0, 0].plot(x, loss, color="tab:gray", alpha=0.3, linewidth=0.8, label="per step")
        axes[0, 0].plot(x[window - 1:], smooth, color="tab:blue", linewidth=2, label=f"mean of {window} steps")
        axes[0, 0].set(yscale="log" if loss.min() > 0 else "linear", title="loss")
        axes[0, 0].legend(fontsize=8)
        axes[0, 1].plot(x, [m.delta_abs for m in steps], linewidth=0.8, label="mean |delta|")
        axes[0, 1].plot(x, [m.delta_mean for m in steps], linewidth=0.8, label="mean delta")
        axes[0, 1].set(title="Residuals")
        axes[0, 1].legend(fontsize=8)
        axes[0, 2].plot(x, [m.learning_rate for m in steps], color="tab:green", linewidth=2)
        axes[0, 2].set(yscale="log", title="Learning rate (policy)")
        twin = axes[0, 2].twinx()
        twin.plot(x, [m.grad_norm for m in steps], color="tab:red", alpha=0.4, linewidth=0.8)
        twin.set_ylabel("|grad|", color="tab:red")
    if evals:
        e = [m.step for m in evals]
        axes[1, 0].plot(e, [m.kl for m in evals], marker="o", markersize=4, linewidth=2, label="KL(p* || p)")
        axes[1, 0].plot(e, [m.reverse_kl for m in evals], marker="s", markersize=4, linewidth=2, label="KL(p || p*)")
        axes[1, 0].plot(e, [m.kl_to_pi for m in evals], linestyle=":", linewidth=2, label="KL(π || p)")
        axes[1, 0].set(yscale="log", title="Exact divergences")
        axes[1, 0].legend(fontsize=8)
        axes[1, 1].plot(e, [m.tv for m in evals], marker="o", markersize=4, linewidth=2)
        axes[1, 1].set(title="TV(p*, p)")
        axes[1, 2].plot(e, [m.expected_score for m in evals], marker="o", markersize=4, linewidth=2)
        axes[1, 2].set(title="E_p[y], TFBind8 score")
    for axis in axes.flat:
        axis.set_xlabel("step")
        axis.grid(alpha=0.3)
    figure.suptitle(name)
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


class LogFn:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = path.open("a")

    def __call__(self, message: str) -> None:
        line = f"{time.strftime('%H:%M:%S')} {message}"
        print(line, flush=True)
        self.file.write(line + "\n")
        self.file.flush()
