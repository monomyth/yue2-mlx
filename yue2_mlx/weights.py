"""Loading YuE2 checkpoints into MLX modules.

Published checkpoints are read-only. Nothing here writes to, requantizes, or rewrites a file under
the checkpoint directory: safetensors are memory-mapped and tensors are converted to MLX form in
memory as they are read.

The conversion that matters is convolution layout. PyTorch stores 1D kernels as ``(O, C, K)`` and
transposed 1D kernels as ``(Cin, Cout, K)``; MLX wants the input channel last in both. Rank alone
cannot tell those apart, and when a kernel's channel counts and width coincide, the unpermuted shape
already looks correct while the values are wrong — so modules declare their own layout through a
``TORCH_PERMUTATION`` class attribute rather than being guessed at.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

DTYPES = {"bfloat16": mx.bfloat16, "float16": mx.float16, "float32": mx.float32}


def load_safetensors(path: str | Path) -> dict[str, mx.array]:
    weights, _ = mx.load(str(path), return_metadata=True)
    return weights


# The published checkpoint is one model carrying both branches. Autoregressive layers use
# ``self_attn``/``mlp``; acoustic layers use the ``nar_``-prefixed equivalents. The final norm is
# shared, so it belongs to both.
_AR_LAYER_PARTS = (".self_attn.", ".mlp.", ".input_layernorm.", ".post_attention_layernorm.")
_SHARED_KEYS = ("model.norm.weight",)


def split_language_model(weights: dict[str, mx.array]) -> tuple[dict, dict]:
    """Split the combined YuE2 checkpoint into its autoregressive and acoustic halves."""
    ar, acoustic = {}, {}
    for key, value in weights.items():
        if key in _SHARED_KEYS:
            ar[key] = acoustic[key] = value
        elif ".nar_" in key:
            acoustic[key] = value
        elif key.startswith(("llm2vae.", "vae2llm.", "time_embedder.", "latent_pos_embed.")):
            acoustic[key] = value
        elif any(part in key for part in _AR_LAYER_PARTS) or key.startswith(
            ("model.embed_tokens.", "lm_head.")
        ):
            ar[key] = value

    if not ar or not acoustic:
        raise ValueError(
            f"could not split the checkpoint: {len(ar)} autoregressive and "
            f"{len(acoustic)} acoustic tensors"
        )
    return ar, acoustic


def fold_weight_norm(weights: dict[str, mx.array]) -> dict[str, mx.array]:
    """Collapse PyTorch weight-norm pairs into plain convolution weights.

    ``torch.nn.utils.weight_norm`` stores a direction ``weight_v`` and a per-output-channel
    magnitude ``weight_g``, and reconstructs ``g * v / ||v||`` on every forward pass. Inference does
    not need the parameterization, so it is folded once at load time.
    """
    folded = {key: value for key, value in weights.items() if not key.endswith(("_g", "_v"))}

    for key in weights:
        if not key.endswith("weight_v"):
            continue
        base = key[: -len("_v")]
        gain = weights.get(f"{base}_g")
        if gain is None:
            raise ValueError(f"{key} has no matching weight_g")

        direction = weights[key].astype(mx.float32)
        axes = tuple(range(1, direction.ndim))
        norm = mx.sqrt(mx.sum(mx.square(direction), axis=axes, keepdims=True))
        folded[base] = direction * (gain.astype(mx.float32) / norm)

    return folded


def declared_permutations(model: nn.Module) -> dict[str, tuple[int, ...]]:
    """Collect layouts that modules declare through ``TORCH_PERMUTATION``."""
    permutations: dict[str, tuple[int, ...]] = {}
    for path, module in model.named_modules():
        permutation = getattr(type(module), "TORCH_PERMUTATION", None)
        if permutation is not None:
            permutations[f"{path}.weight" if path else "weight"] = permutation
    return permutations


def apply_weights(
    model: nn.Module,
    weights: dict[str, mx.array],
    dtype: mx.Dtype,
    keep_float32: tuple[str, ...] = (),
    strict: bool = True,
) -> nn.Module:
    """Cast, relayout and load ``weights`` into ``model``, then materialize them."""
    expected = dict(tree_flatten(model.parameters()))

    missing = [key for key in expected if key not in weights]
    unexpected = [key for key in weights if key not in expected]
    if strict and missing:
        raise ValueError(f"checkpoint is missing {len(missing)} tensors, e.g. {missing[:5]}")

    declared = declared_permutations(model)
    resolved = []
    for key, target in expected.items():
        if key not in weights:
            continue
        permutation = declared.get(key)
        if permutation is not None:
            # Always applied, never conditioned on a shape mismatch: a cube-shaped kernel already
            # has the expected shape while still needing the permutation.
            array = weights[key].transpose(*permutation)
            if array.shape != tuple(target.shape):
                raise ValueError(
                    f"{key}: declared layout {permutation} gives {array.shape}, "
                    f"expected {tuple(target.shape)}"
                )
        elif weights[key].shape != tuple(target.shape):
            raise ValueError(
                f"{key}: checkpoint shape {weights[key].shape} does not match {tuple(target.shape)}"
            )
        else:
            array = weights[key]

        target_dtype = mx.float32 if any(part in key for part in keep_float32) else dtype
        resolved.append((key, array.astype(target_dtype)))

    model.load_weights(resolved)
    mx.eval(model.parameters())
    if unexpected:
        print(f"[yue2-mlx] ignored {len(unexpected)} unused tensors (e.g. {unexpected[:3]})")
    return model
