"""Reproducible masked-NLL training with exact periodic evaluation."""

import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .architecture import Architecture, save
from .config import ArchitectureSpec, Config, Metric, TrainingReport, rng, software_versions
from .data import build_target, load_dataset, save_model_distribution
from .evaluation import build_contexts, evaluate
from .plots import plot_distribution, plot_training


def resolve_architecture(config: Config, width: int | None,
                         architecture: ArchitectureSpec | None) -> Architecture:
    """An MLP of the given width, else the given spec, else the one chosen by the config."""
    if width is not None and architecture is not None:
        raise ValueError("Give either an MLP width or an architecture, not both")
    if width is not None:
        return Architecture.mlp(width)
    return Architecture(architecture or config.architecture)


def train(config: Config, width: int | None = None, output_dir: Path | None = None,
          overwrite: bool = False, architecture: ArchitectureSpec | None = None) -> TrainingReport:
    """Train one denoiser (config.architecture by default); checkpoints, report and plots go to output_dir.

    Training, optimiser, masks, validation split and exact evaluation do not depend on the family.
    """
    started = time.perf_counter()
    arch = resolve_architecture(config, width, architecture)
    output_dir = output_dir or config.data_dir
    best_path = output_dir / "best.npz"
    last_path = output_dir / "last.npz"
    report_path = output_dir / "training.json"
    if not overwrite and any(p.exists() for p in (best_path, last_path, report_path)):
        raise FileExistsError("Training results exist; use --overwrite to replace them")
    tokens, metadata = load_dataset(config)
    indices = rng(config.seed, 2).permutation(len(tokens))
    validation_size = round(len(tokens) * config.validation_fraction)
    validation = tokens[indices[:validation_size]]
    training = tokens[indices[validation_size:]]
    shuffle_rng = rng(config.seed, 3)
    key = jax.random.fold_in(jax.random.key(config.seed), 3)
    params = arch.init(jax.random.fold_in(jax.random.key(config.seed), 2))
    if arch.parameter_count(params) != arch.expected_parameter_count():
        raise ArithmeticError("Unexpected parameter count")
    optimizer = optax.adam(config.learning_rate)
    state = optimizer.init(params)
    contexts, target = build_contexts(), build_target(config)
    software = software_versions()
    masked_loss = arch.masked_loss
    loss_fn = jax.jit(masked_loss)

    @jax.jit
    def update(
        parameters: Any,
        optimizer_state: optax.OptState,
        batch: jax.Array,
        mask_key: jax.Array,
    ) -> tuple[Any, optax.OptState, jax.Array]:
        mask = jax.random.bernoulli(mask_key, config.mask_probability, batch.shape)
        loss, gradient = jax.value_and_grad(masked_loss)(parameters, batch, mask)
        updates, optimizer_state = optimizer.update(
            gradient, optimizer_state, parameters
        )
        return optax.apply_updates(parameters, updates), optimizer_state, loss

    total = 0.0
    for start in range(0, len(training), config.batch_size):
        batch = jnp.asarray(training[start : start + config.batch_size])
        key, mask_key = jax.random.split(key)
        mask = jax.random.bernoulli(mask_key, config.mask_probability, batch.shape)
        total += float(loss_fn(params, batch, mask)) * len(batch)
    initial = evaluate(
        params,
        contexts,
        target,
        validation,
        config.mask_probability,
        config.eval_batch_size,
        output_dir,
        forward=arch.forward,
    )
    metrics = [
        Metric(
            epoch=0,
            training_loss=total / len(training),
            validation_loss=initial.validation_loss,
            kl=initial.kl,
            probability_sum=initial.probability_sum,
        )
    ]
    best_loss, best_epoch = initial.validation_loss, 0
    best_log_probs = initial.log_probs
    save(best_path, arch, params)
    print(
        f"epoch 0: val={initial.validation_loss:.6f}, KL={initial.kl:.6f}", flush=True
    )

    for epoch in range(1, config.epochs + 1):
        training = training[shuffle_rng.permutation(len(training))]
        total = 0.0
        for start in range(0, len(training), config.batch_size):
            batch = jnp.asarray(training[start : start + config.batch_size])
            key, mask_key = jax.random.split(key)
            params, state, loss = update(params, state, batch, mask_key)
            value = float(loss)
            if not np.isfinite(value):
                raise FloatingPointError(f"Non-finite training loss at epoch {epoch}")
            total += value * len(batch)
        values: dict[str, float] = {}
        if epoch % config.eval_every == 0 or epoch == config.epochs:
            result = evaluate(
                params,
                contexts,
                target,
                validation,
                config.mask_probability,
                config.eval_batch_size,
                output_dir,
                forward=arch.forward,
            )
            values = {
                "validation_loss": result.validation_loss,
                "kl": result.kl,
                "probability_sum": result.probability_sum,
            }
            if result.validation_loss < best_loss:
                best_loss, best_epoch = result.validation_loss, epoch
                best_log_probs = result.log_probs
                save(best_path, arch, params)
            print(
                f"epoch {epoch}: train={total / len(training):.6f}, "
                f"val={result.validation_loss:.6f}, KL={result.kl:.6f}",
                flush=True,
            )
        metrics.append(
            Metric(epoch=epoch, training_loss=total / len(training), **values)
        )
        report = TrainingReport(
            config=config,
            software=software,
            hidden_width=arch.width,
            parameter_count=arch.parameter_count(params),
            dataset_sha256=metadata.dataset_sha256,
            best_epoch=best_epoch,
            metrics=metrics,
            architecture=arch.spec,
            wall_seconds=time.perf_counter() - started,
        )
        report.save(report_path)
    save(last_path, arch, params)
    plot_training(output_dir, report)
    save_model_distribution(output_dir, target, best_log_probs, best_epoch)
    plot_distribution(output_dir, target, best_log_probs, best_epoch, arch.width)
    return report
