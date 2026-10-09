"""Reproducible masked-NLL training with exact periodic evaluation."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .config import Config, Metric, TrainingReport, rng, software_versions
from .data import build_target, load_dataset, save_model_distribution
from .evaluation import build_contexts, evaluate
from .model import (
    DEFAULT_WIDTH,
    Params,
    expected_parameter_count,
    init_parameters,
    masked_loss,
    parameter_count,
    save_parameters,
)
from .plots import plot_distribution, plot_training


def train(config: Config, width: int = DEFAULT_WIDTH, output_dir: Path | None = None,
          overwrite: bool = False) -> TrainingReport:
    """Train one MLP; checkpoints, report and plots go to output_dir."""
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
    params = init_parameters(jax.random.fold_in(jax.random.key(config.seed), 2), width)
    if parameter_count(params) != expected_parameter_count(width):
        raise ArithmeticError("Unexpected parameter count")
    optimizer = optax.adam(config.learning_rate)
    state = optimizer.init(params)
    contexts, target = build_contexts(), build_target(config)
    software = software_versions()
    loss_fn = jax.jit(masked_loss)

    @jax.jit
    def update(
        parameters: Params,
        optimizer_state: optax.OptState,
        batch: jax.Array,
        mask_key: jax.Array,
    ) -> tuple[Params, optax.OptState, jax.Array]:
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
    save_parameters(best_path, params)
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
            )
            values = {
                "validation_loss": result.validation_loss,
                "kl": result.kl,
                "probability_sum": result.probability_sum,
            }
            if result.validation_loss < best_loss:
                best_loss, best_epoch = result.validation_loss, epoch
                best_log_probs = result.log_probs
                save_parameters(best_path, params)
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
            hidden_width=width,
            parameter_count=parameter_count(params),
            dataset_sha256=metadata.dataset_sha256,
            best_epoch=best_epoch,
            metrics=metrics,
        )
        report.save(report_path)
    save_parameters(last_path, params)
    plot_training(output_dir, report)
    save_model_distribution(output_dir, target, best_log_probs, best_epoch)
    plot_distribution(output_dir, target, best_log_probs, best_epoch, width)
    return report
