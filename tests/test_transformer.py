"""The transformer denoiser, the architecture registry and exact evaluation of either family."""

from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
pytest.importorskip("optax")
pytest.importorskip("pyarrow")

from pydantic import ValidationError

from src import model, transformer
from src.ablation import run_transformer_ablation
from src.architecture import Architecture, load, save
from src.config import LENGTH, MASK, TRANSFORMER_SIZES, Config, MLPSpec, TransformerSpec, load_config
from src.data import build_target
from src.evaluation import build_contexts, evaluate
from src.train import train

SIZES = ((8, 1, 2, 16), (16, 2, 4, 64), (12, 3, 3, 20), (4, 2, 1, 4))
SMALL = (8, 1, 2, 16)
MLP_CHECKPOINT = Path("models/pretrain/mlp_5/best.npz")


def randomized(params, seed: int = 0):
    """Nonzero biases and non-unit scales, so no part of the network is trivially inert."""
    leaves, treedef = jax.tree.flatten(params)
    keys = jax.random.split(jax.random.key(seed), len(leaves))
    return jax.tree.unflatten(treedef, [
        leaf + 0.3 * jax.random.normal(key, leaf.shape, dtype=jnp.float32)
        for key, leaf in zip(keys, leaves)])


def random_tokens(shape: tuple[int, ...], seed: int = 0, letters_only: bool = False) -> jax.Array:
    """Random tokens in 0..MASK, or in 0..3 (a fully revealed string) with letters_only."""
    return jnp.asarray(np.random.default_rng(seed).integers(0, MASK if letters_only else MASK + 1, shape),
                       dtype=jnp.int32)


def test_output_shape_dtype_and_batch_dimensions() -> None:
    params = transformer.init_parameters(jax.random.key(0), *SIZES[1])
    for shape in ((LENGTH,), (3, LENGTH), (2, 5, LENGTH)):
        logits = transformer.forward(params, random_tokens(shape))
        assert logits.shape == (*shape, 4) and logits.dtype == jnp.float32
        assert np.isfinite(np.asarray(logits)).all()
    with pytest.raises(ValueError, match="trailing dimension"):
        transformer.forward(params, jnp.zeros((3, LENGTH - 1), dtype=jnp.int32))
    # An out-of-range token must not be clamped silently to a real embedding.
    bad = jnp.array([[0, 1, 2, 3, 0, 1, 2, MASK + 1]], dtype=jnp.int32)
    assert np.isnan(np.asarray(transformer.forward(params, bad))).all()


@pytest.mark.parametrize("size", SIZES)
def test_parameter_count_formula_equals_actual_count(size: tuple[int, int, int, int]) -> None:
    d, layers, heads, ff = size
    params = transformer.init_parameters(jax.random.key(1), d, layers, heads, ff)
    actual = sum(int(a.size) for a in jax.tree.leaves(params))
    assert actual == transformer.expected_parameter_count(d, layers, heads, ff)
    assert actual == sum(int(np.prod(s)) for s in transformer.parameter_shapes(*size).values())
    arch = Architecture.transformer(*size)
    assert arch.parameter_count(params) == arch.expected_parameter_count() == actual
    assert transformer.dimensions(params) == size
    arrays = transformer.to_arrays(params)
    assert {n: a.shape for n, a in arrays.items()} == transformer.parameter_shapes(*size)
    # The head count is not part of the count.
    assert transformer.expected_parameter_count(12, 3, 1, 20) == transformer.expected_parameter_count(12, 3, 6, 20)
    with pytest.raises(ValueError, match="divisible"):
        transformer.expected_parameter_count(10, 1, 4, 8)


def test_default_ablation_sizes_span_the_mlp_range() -> None:
    counts = [Architecture(spec).expected_parameter_count() for spec in TRANSFORMER_SIZES]
    assert counts == [568, 1900, 6868, 26020, 101188, 398980]
    mlp = [model.expected_parameter_count(2**k) for k in range(1, 8)]
    assert min(counts) < 1000 and max(counts) > 10 * max(mlp)
    assert MLPSpec().width == model.DEFAULT_WIDTH


def test_initialisation_scheme() -> None:
    d, layers, heads, ff = 64, 2, 4, 256
    params = transformer.init_parameters(jax.random.key(7), d, layers, heads, ff)
    assert all(a.dtype == jnp.float32 for a in jax.tree.leaves(params))
    for embedding in (params.token_embedding, params.position_embedding):
        assert abs(float(embedding.mean())) < 0.01 and float(embedding.std()) == pytest.approx(0.02, rel=0.2)
    for block in params.blocks:
        assert all(np.all(np.asarray(b) == 0) for b in (block.ln1_bias, block.ln2_bias, block.qkv_bias,
                                                         block.proj_bias, block.ff_in_bias, block.ff_out_bias))
        assert all(np.all(np.asarray(s) == 1) for s in (block.ln1_scale, block.ln2_scale))
        for weight, fans in ((block.qkv_weight, d + 3 * d), (block.proj_weight, 2 * d),
                             (block.ff_in_weight, d + ff), (block.ff_out_weight, ff + d)):
            limit = np.sqrt(6 / fans)
            assert float(jnp.abs(weight).max()) <= limit
            assert float(jnp.abs(weight).max()) > 0.95 * limit  # fills the Glorot interval
    assert np.all(np.asarray(params.final_scale) == 1) and np.all(np.asarray(params.head_bias) == 0)


def test_attention_is_bidirectional_and_wired() -> None:
    params = transformer.init_parameters(jax.random.key(2), 16, 2, 4, 64)
    base = jnp.array([0, MASK, 1, MASK, 2, MASK, 3, MASK], dtype=jnp.int32)
    reference = np.asarray(transformer.forward(params, base))

    def change(position: int) -> np.ndarray:
        changed = transformer.forward(params, base.at[position].set((int(base[position]) + 1) % 4))
        assert changed.shape == reference.shape
        return np.abs(np.asarray(changed) - reference)

    for position in (0, 4, 6):  # revealed tokens
        difference = change(position)
        others = np.delete(difference, position, axis=0)
        assert others.max() > 1e-4
        assert difference[1].max() > 1e-6  # a masked position
    assert change(6)[0].max() > 1e-6  # the last token reaches the first position, and ...
    assert change(0)[7].max() > 1e-6  # ... the first reaches the last: no causal mask
    # Zeroing the attention output projection removes all cross-position influence.
    blocked = params._replace(blocks=tuple(
        block._replace(proj_weight=jnp.zeros_like(block.proj_weight)) for block in params.blocks))
    changed = transformer.forward(blocked, base.at[0].set(3))
    np.testing.assert_array_equal(np.delete(np.asarray(changed), 0, axis=0),
                                  np.delete(np.asarray(transformer.forward(blocked, base)), 0, axis=0))


def test_masked_nll_gradients_are_finite() -> None:
    arch = Architecture.transformer(*SMALL)
    params = arch.init(jax.random.key(3))
    tokens = random_tokens((16, LENGTH), letters_only=True)
    mask = jax.random.bernoulli(jax.random.key(4), 0.5, tokens.shape)
    loss, gradients = jax.value_and_grad(arch.masked_loss)(params, tokens, mask)
    assert np.isfinite(float(loss)) and float(loss) > 0
    assert all(np.isfinite(np.asarray(g)).all() for g in jax.tree.leaves(gradients))
    assert float(jnp.abs(gradients.blocks[0].qkv_weight).max()) > 0
    assert float(jnp.abs(gradients.token_embedding).max()) > 0
    # With a uniform output the masked NLL is log 4 per masked position, whatever the input.
    zero_head = params._replace(head_weight=jnp.zeros_like(params.head_weight))
    assert float(arch.masked_loss(zero_head, tokens, mask)) == pytest.approx(
        float(mask.sum(axis=1).mean()) * np.log(4), rel=1e-5)


def test_masked_loss_matches_the_mlp_loss_exactly() -> None:
    arch = Architecture.mlp(16)
    params = arch.init(jax.random.key(5))
    tokens = random_tokens((32, LENGTH), letters_only=True)
    mask = jax.random.bernoulli(jax.random.key(6), 0.5, tokens.shape)
    assert float(arch.masked_loss(params, tokens, mask)) == float(model.masked_loss(params, tokens, mask))


def test_checkpoint_roundtrip_is_bit_exact(tmp_path: Path) -> None:
    for size in (SMALL, (16, 2, 4, 64)):
        arch = Architecture.transformer(*size)
        params = randomized(arch.init(jax.random.key(8)))
        save(tmp_path / "t.npz", arch, params)
        restored_arch, restored = load(tmp_path / "t.npz")
        assert restored_arch == arch and restored_arch.family == "transformer"
        assert jax.tree.structure(restored) == jax.tree.structure(params)
        for a, b in zip(jax.tree.leaves(params), jax.tree.leaves(restored)):
            assert b.dtype == jnp.float32
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
        tokens = random_tokens((20, LENGTH))
        np.testing.assert_array_equal(np.asarray(arch.forward(params, tokens)),
                                      np.asarray(restored_arch.forward(restored, tokens)))
    # MLPs written through the registry stay in the legacy format that model.load_parameters reads.
    mlp = Architecture.mlp(8)
    mlp_params = randomized(mlp.init(jax.random.key(9)))
    save(tmp_path / "m.npz", mlp, mlp_params)
    for a, b in zip(jax.tree.leaves(mlp_params), jax.tree.leaves(model.load_parameters(tmp_path / "m.npz"))):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    assert load(tmp_path / "m.npz")[0] == mlp
    with pytest.raises(ValueError, match="do not match"):
        save(tmp_path / "bad.npz", Architecture.mlp(4), mlp_params)


def test_existing_mlp_checkpoint_loads_through_the_registry() -> None:
    arch, params = load(MLP_CHECKPOINT)
    assert arch == Architecture.mlp(32) and arch.family == "mlp" and arch.width == 32
    legacy = model.load_parameters(MLP_CHECKPOINT)
    tokens = random_tokens((64, LENGTH), seed=11)
    np.testing.assert_array_equal(np.asarray(arch.forward(params, tokens)), np.asarray(model.forward(legacy, tokens)))
    assert arch.parameter_count(params) == arch.expected_parameter_count() == 3424


@pytest.mark.parametrize("damage", ("float64", "nan", "shape", "missing", "extra", "bad_spec"))
def test_load_validates_transformer_checkpoints(damage: str, tmp_path: Path) -> None:
    arch = Architecture.transformer(*SMALL)
    save(tmp_path / "ok.npz", arch, arch.init(jax.random.key(10)))
    with np.load(tmp_path / "ok.npz", allow_pickle=False) as saved:
        fields = {name: saved[name] for name in saved.files}
    if damage == "float64":
        fields["head_weight"] = fields["head_weight"].astype(np.float64)
    elif damage == "nan":
        fields["head_bias"] = np.array([0, np.nan, 0, 0], dtype=np.float32)
    elif damage == "shape":
        fields["head_weight"] = fields["head_weight"][:, :3]
    elif damage == "missing":
        del fields["position_embedding"]
    elif damage == "extra":
        fields["surplus"] = np.zeros(1, dtype=np.float32)
    else:
        fields["architecture"] = np.asarray(TransformerSpec.model_construct(
            d_model=8, layers=1, heads=3, ff=16).model_dump_json())
    np.savez(tmp_path / "bad.npz", **fields)
    with pytest.raises(ValueError):
        load(tmp_path / "bad.npz")


def test_config_chooses_the_architecture(tmp_path: Path) -> None:
    assert Config().architecture == MLPSpec(width=model.DEFAULT_WIDTH)
    config = Config(architecture=TransformerSpec(d_model=8, layers=1, heads=2, ff=16))
    config.save(tmp_path / "c.json")
    assert load_config(tmp_path / "c.json") == config
    for spec in ({"d_model": 10, "layers": 1, "heads": 4, "ff": 8}, {"d_model": 8, "layers": 0, "heads": 2, "ff": 8}):
        with pytest.raises(ValidationError):
            TransformerSpec(**spec)
    with pytest.raises(ValidationError):
        Config(transformer_sizes=())
    with pytest.raises(ValidationError):
        Config(transformer_sizes=(TRANSFORMER_SIZES[0], TRANSFORMER_SIZES[0]))


def test_exact_evaluation_of_a_random_transformer() -> None:
    arch = Architecture.transformer(*SMALL)
    params = randomized(arch.init(jax.random.key(12)), seed=1)
    config = Config(reward="potts", beta=2.0)
    contexts, target = build_contexts(), build_target(config)
    validation = np.asarray(random_tokens((6, LENGTH), seed=3, letters_only=True))
    result = evaluate(params, contexts, target, validation, 0.5, 8192, forward=arch.forward)
    assert abs(result.probability_sum - 1) < 1e-6
    assert result.kl > 0 and np.isfinite(result.validation_loss)
    assert result.log_probs.shape == (4**LENGTH,)
    # The default is still the MLP: forward=None equals passing the MLP forward explicitly.
    mlp = Architecture.mlp(4)
    mlp_params = mlp.init(jax.random.key(13))
    default = evaluate(mlp_params, contexts, target, validation, 0.5, 8192)
    explicit = evaluate(mlp_params, contexts, target, validation, 0.5, 8192, forward=model.forward)
    assert default.kl == explicit.kl and default.validation_loss == explicit.validation_loss
    np.testing.assert_array_equal(default.log_probs, explicit.log_probs)


def test_transformer_pretraining_and_ablation_outputs(tmp_path: Path) -> None:
    sizes = (TransformerSpec(d_model=4, layers=1, heads=2, ff=8), TransformerSpec(d_model=8, layers=1, heads=2, ff=16))
    # The committed Potts dataset (generating one takes longer than the whole test); one epoch.
    config = load_config(Path("configs/potts.json")).model_copy(update={
        "epochs": 1, "eval_every": 1, "transformer_sizes": sizes, "models_dir": tmp_path / "models"})
    ablation = run_transformer_ablation(config)
    assert [row.tag for row in ablation.rows] == [spec.tag for spec in sizes]
    assert [row.parameter_count for row in ablation.rows] == [
        Architecture(spec).expected_parameter_count() for spec in sizes]
    for spec, row in zip(sizes, ablation.rows):
        folder = config.models_dir / f"transformer_{spec.tag}"
        assert (folder / "best.npz").exists() and (folder / "training.json").exists()
        assert load(folder / "best.npz")[0] == Architecture(spec) and row.wall_seconds > 0
        assert '"family": "transformer"' in (folder / "training.json").read_text()
    # The recorded best-epoch KL is the exact KL of the saved best checkpoint.
    arch, params = load(config.models_dir / f"transformer_{sizes[1].tag}" / "best.npz")
    validation = np.asarray(random_tokens((4, LENGTH), letters_only=True))
    again = evaluate(params, build_contexts(), build_target(config), validation, 0.5, 8192, forward=arch.forward)
    assert again.kl == pytest.approx(ablation.rows[1].best_kl, abs=1e-6)
    assert (config.models_dir / "transformer_ablation.json").exists()
    assert (config.models_dir / "transformer_ablation.png").exists()
    with pytest.raises(FileExistsError):
        run_transformer_ablation(config)
    with pytest.raises(FileExistsError):
        train(config, architecture=sizes[0], output_dir=config.models_dir / f"transformer_{sizes[0].tag}")
    with pytest.raises(ValueError, match="not both"):
        train(config, width=4, architecture=sizes[0], output_dir=tmp_path / "x")
