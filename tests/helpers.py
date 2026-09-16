"""Shared utilities for the parity suite."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from yue2_mlx.weights import apply_weights


def torch_state_to_mlx(state_dict) -> dict[str, mx.array]:
    return {
        key: mx.array(value.detach().to(dtype=_np_dtype(value)).numpy())
        for key, value in state_dict.items()
    }


def _np_dtype(tensor):
    import torch

    return torch.float32 if tensor.is_floating_point() else tensor.dtype


def copy_weights(torch_module, mlx_module: nn.Module, strict: bool = True) -> nn.Module:
    """Load a PyTorch module's parameters into the matching MLX module in fp32."""
    return apply_weights(
        mlx_module, torch_state_to_mlx(torch_module.state_dict()), mx.float32, strict=strict
    )


def max_abs_diff(a, b) -> float:
    left = np.asarray(a, dtype=np.float64)
    right = np.asarray(b, dtype=np.float64)
    assert left.shape == right.shape, f"shape mismatch {left.shape} vs {right.shape}"
    return float(np.max(np.abs(left - right)))


def relative_error(a, b) -> float:
    """Worst-element error relative to the reference's peak magnitude.

    The right metric for fp32 comparisons, where any structural mistake shows up as an outlier.
    """
    left = np.asarray(a, dtype=np.float64)
    right = np.asarray(b, dtype=np.float64)
    scale = max(float(np.max(np.abs(right))), 1e-8)
    return float(np.max(np.abs(left - right)) / scale)


def rms_relative_error(a, b) -> float:
    """Error energy relative to signal energy.

    For bf16 comparisons this is what matters: bf16 carries an 8-bit mantissa, so individual
    elements legitimately differ by a few tenths of a percent and a worst-element metric is
    dominated by one unlucky rounding. Structural mistakes move the whole distribution.
    """
    left = np.asarray(a, dtype=np.float64)
    right = np.asarray(b, dtype=np.float64)
    denominator = max(float(np.sqrt(np.mean(right**2))), 1e-12)
    return float(np.sqrt(np.mean((left - right) ** 2)) / denominator)


def cosine_similarity(a, b) -> float:
    left = np.asarray(a, dtype=np.float64).ravel()
    right = np.asarray(b, dtype=np.float64).ravel()
    norms = np.linalg.norm(left) * np.linalg.norm(right)
    return float(left @ right / max(norms, 1e-12))
