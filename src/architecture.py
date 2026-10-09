"""One interface over the MLP and the transformer denoiser: init, forward, loss, save, load.

An Architecture is a validated spec (config.MLPSpec / config.TransformerSpec); its parameters are
the pytree of model.py (MLP) or transformer.py. MLP checkpoints are written in the legacy
format of model.save_parameters (keys w0,b0,w1,b1,w2,b2, nothing else), so old and new code read
each other's files. Transformer checkpoints hold one float32 array per parameter plus an
`architecture` string array with the JSON spec (the head count cannot be read from the weights
alone).
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from pydantic import TypeAdapter

from . import model, transformer
from .config import MASK, ArchitectureSpec, MLPSpec, TransformerSpec

Forward = Callable[[Any, jax.Array], jax.Array]
LEGACY_MLP_FIELDS = frozenset(f"{name}{i}" for i in range(3) for name in ("w", "b"))
SPEC_FIELD = "architecture"


class Architecture(NamedTuple):
    spec: ArchitectureSpec

    @classmethod
    def mlp(cls, width: int) -> "Architecture":
        return cls(MLPSpec(width=width))

    @classmethod
    def transformer(cls, d_model: int, layers: int, heads: int, ff: int) -> "Architecture":
        return cls(TransformerSpec(d_model=d_model, layers=layers, heads=heads, ff=ff))

    @property
    def family(self) -> str:
        return self.spec.family

    @property
    def width(self) -> int:
        """MLP hidden width, or the transformer's d_model."""
        return self.spec.width if isinstance(self.spec, MLPSpec) else self.spec.d_model

    @property
    def forward(self) -> Forward:
        """forward(params, tokens) -> logits (..., LENGTH, 4); a stable function object for jit."""
        return model.forward if isinstance(self.spec, MLPSpec) else transformer.forward

    def init(self, key: jax.Array) -> Any:
        if isinstance(self.spec, MLPSpec):
            return model.init_parameters(key, self.spec.width)
        spec = self.spec
        return transformer.init_parameters(key, spec.d_model, spec.layers, spec.heads, spec.ff)

    def expected_parameter_count(self) -> int:
        if isinstance(self.spec, MLPSpec):
            return model.expected_parameter_count(self.spec.width)
        spec = self.spec
        return transformer.expected_parameter_count(spec.d_model, spec.layers, spec.heads, spec.ff)

    def parameter_count(self, params: Any) -> int:
        return sum(int(array.size) for array in jax.tree.leaves(params))

    def shapes(self) -> dict[str, tuple[int, ...]]:
        """Name -> shape of every parameter array, as stored in the checkpoint."""
        if isinstance(self.spec, MLPSpec):
            return {f"{name}{i}": shape if name == "w" else (shape[1],)
                    for i, shape in enumerate(model.shapes(self.spec.width)) for name in ("w", "b")}
        spec = self.spec
        return transformer.parameter_shapes(spec.d_model, spec.layers, spec.heads, spec.ff)

    def arrays(self, params: Any) -> dict[str, jax.Array]:
        """Flatten a parameter pytree to {name: array}, checking it has this architecture's shapes."""
        if isinstance(self.spec, MLPSpec):
            if len(params) != 3:
                raise ValueError("MLP parameters must be three layers")
            arrays = {f"{name}{i}": array for i, layer in enumerate(params)
                      for name, array in zip(("w", "b"), layer)}
        else:
            arrays = transformer.to_arrays(params)
        expected = self.shapes()
        if list(arrays) != list(expected) or any(
                tuple(array.shape) != expected[name] for name, array in arrays.items()):
            raise ValueError(f"Parameters do not match architecture {self.spec}")
        return arrays

    def masked_loss(self, params: Any, tokens: jax.Array, mask: jax.Array) -> jax.Array:
        """Mean over the batch of the summed NLL at masked positions (model.masked_nll)."""
        logits = self.forward(params, jnp.where(mask, MASK, tokens))
        return model.masked_nll(logits, tokens, mask)


def save(path: Path, arch: Architecture, params: Any) -> None:
    arrays = arch.arrays(params)
    if any(np.dtype(array.dtype) != np.float32 for array in arrays.values()):
        raise ValueError("Checkpoint must contain float32 parameters")
    if isinstance(arch.spec, MLPSpec):
        model.save_parameters(path, params)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{name: np.asarray(array) for name, array in arrays.items()},
             **{SPEC_FIELD: np.asarray(arch.spec.model_dump_json())})


def load(path: Path) -> tuple[Architecture, Any]:
    """Read a checkpoint of either family, validating fields, shapes, float32 and finiteness."""
    with np.load(path, allow_pickle=False) as saved:
        fields = set(saved.files)
        if fields == LEGACY_MLP_FIELDS:
            params = model.load_parameters(path)
            return Architecture.mlp(int(params[0].weight.shape[1])), params
        if SPEC_FIELD not in fields:
            raise ValueError("Invalid checkpoint fields")
        spec_array = saved[SPEC_FIELD]
        if spec_array.shape != () or spec_array.dtype.kind != "U":
            raise ValueError("Invalid checkpoint architecture field")
        spec = TypeAdapter(ArchitectureSpec).validate_json(str(spec_array))
        if not isinstance(spec, TransformerSpec):
            raise ValueError("Only transformer checkpoints carry an architecture field")
        arch = Architecture(spec)
        shapes = arch.shapes()
        if fields != set(shapes) | {SPEC_FIELD}:
            raise ValueError("Invalid checkpoint fields")
        arrays: dict[str, jax.Array] = {}
        for name, shape in shapes.items():
            array = saved[name]
            if array.shape != shape:
                raise ValueError("Invalid checkpoint shapes")
            if array.dtype != np.float32 or not np.isfinite(array).all():
                raise ValueError("Checkpoint must contain finite float32 parameters")
            arrays[name] = jnp.asarray(array)
    return arch, transformer.from_arrays(arrays, spec.layers)
