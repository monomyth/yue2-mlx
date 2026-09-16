"""Compare the MLX Oobleck audio decoder against an independent PyTorch implementation.

The real risk here is layout. PyTorch stores 1D kernels as ``(O, C, K)`` and transposed 1D kernels
as ``(Cin, Cout, K)``; MLX wants the input channel last in both, and a wrong permutation still gives
audio-shaped output. Comparing against PyTorch's own ``nn.Conv1d`` and ``nn.ConvTranspose1d`` makes
those permutations independently checkable, since nobody here wrote those.
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest
from helpers import copy_weights, relative_error

from yue2_mlx.models.vae import Conv1d, ConvTranspose1d, SnakeBeta, VAEConfig, YuE2VAE

pytest.importorskip("torch", reason="pip install -e '.[dev]' to run the reference comparisons")

# Same topology as the released decoder, scaled down so this runs in seconds.
CONFIG = VAEConfig(latent_dim=64, channels=4, c_mults=(1, 2, 4), strides=(2, 2, 4))


def _build_pair(seed: int = 0):
    import torch
    from reference_torch import RefOobleckDecoder

    torch.manual_seed(seed)
    ref = RefOobleckDecoder(
        latent_dim=CONFIG.latent_dim,
        out_channels=CONFIG.out_channels,
        channels=CONFIG.channels,
        c_mults=CONFIG.c_mults,
        strides=CONFIG.strides,
    ).eval()

    generator = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        for param in ref.parameters():
            param.copy_(torch.randn(param.shape, generator=generator) * 0.1)

    model = YuE2VAE(CONFIG)
    copy_weights(ref, model.decoder)
    return ref, model


def test_conv1d_layout_matches_torch():
    """A PyTorch ``(O, C, K)`` kernel must land as ``(O, K, C)`` and compute the same thing."""
    import torch

    rng = np.random.default_rng(0)
    signal = rng.standard_normal((1, 3, 9), dtype=np.float32)
    kernel = rng.standard_normal((5, 3, 7), dtype=np.float32)

    expected = torch.nn.functional.conv1d(
        torch.from_numpy(signal), torch.from_numpy(kernel), padding=3
    ).numpy()

    conv = Conv1d(3, 5, 7, padding=3, bias=False)
    conv.weight = mx.array(kernel.transpose(*Conv1d.TORCH_PERMUTATION))
    out = conv(mx.array(signal.transpose(0, 2, 1)))

    assert relative_error(np.asarray(out).transpose(0, 2, 1), expected) < 1e-5


@pytest.mark.parametrize("dilation", [1, 3, 9])
def test_dilated_conv1d_matches_torch(dilation):
    import torch

    rng = np.random.default_rng(dilation)
    signal = rng.standard_normal((1, 4, 24), dtype=np.float32)
    kernel = rng.standard_normal((4, 4, 7), dtype=np.float32)
    padding = dilation * 3

    expected = torch.nn.functional.conv1d(
        torch.from_numpy(signal), torch.from_numpy(kernel), padding=padding, dilation=dilation
    ).numpy()

    conv = Conv1d(4, 4, 7, padding=padding, dilation=dilation, bias=False)
    conv.weight = mx.array(kernel.transpose(*Conv1d.TORCH_PERMUTATION))
    out = conv(mx.array(signal.transpose(0, 2, 1)))

    assert relative_error(np.asarray(out).transpose(0, 2, 1), expected) < 1e-5


@pytest.mark.parametrize("stride", [2, 4, 5, 6])
def test_transposed_conv1d_layout_matches_torch(stride):
    """The transposed kernel needs a different permutation, and no kernel flip."""
    import torch

    rng = np.random.default_rng(stride)
    signal = rng.standard_normal((1, 4, 11), dtype=np.float32)
    kernel = rng.standard_normal((4, 6, 2 * stride), dtype=np.float32)
    padding = math.ceil(stride / 2)

    expected = torch.nn.functional.conv_transpose1d(
        torch.from_numpy(signal), torch.from_numpy(kernel), stride=stride, padding=padding
    ).numpy()

    conv = ConvTranspose1d(4, 6, 2 * stride, stride=stride, padding=padding)
    conv.weight = mx.array(
        np.ascontiguousarray(kernel.transpose(*ConvTranspose1d.TORCH_PERMUTATION))
    )
    conv.bias = mx.zeros((6,))
    out = conv(mx.array(signal.transpose(0, 2, 1)))

    assert relative_error(np.asarray(out).transpose(0, 2, 1), expected) < 1e-5


def test_cube_shaped_kernel_still_needs_its_permutation():
    """Equal channel counts and kernel width make the unpermuted shape look correct.

    This is the case that silently corrupted weights until modules declared their layout, so it is
    pinned here rather than left to chance.
    """
    import torch

    size = 4
    rng = np.random.default_rng(7)
    signal = rng.standard_normal((1, size, 6), dtype=np.float32)
    kernel = rng.standard_normal((size, size, size), dtype=np.float32)
    assert kernel.shape[0] == kernel.shape[1] == kernel.shape[2]

    stride, padding = 2, math.ceil(2 / 2)
    expected = torch.nn.functional.conv_transpose1d(
        torch.from_numpy(signal), torch.from_numpy(kernel), stride=stride, padding=padding
    ).numpy()

    conv = ConvTranspose1d(size, size, size, stride=stride, padding=padding)
    conv.weight = mx.array(
        np.ascontiguousarray(kernel.transpose(*ConvTranspose1d.TORCH_PERMUTATION))
    )
    conv.bias = mx.zeros((size,))
    correct = np.asarray(conv(mx.array(signal.transpose(0, 2, 1)))).transpose(0, 2, 1)

    conv.weight = mx.array(kernel)  # the shape fits, the values do not
    wrong = np.asarray(conv(mx.array(signal.transpose(0, 2, 1)))).transpose(0, 2, 1)

    assert relative_error(correct, expected) < 1e-5
    assert relative_error(wrong, expected) > 0.1, "an unpermuted cube kernel must not pass silently"


def test_snake_beta_matches_its_formula():
    snake = SnakeBeta(3)
    snake.alpha = mx.array([0.0, 0.5, -0.5])
    snake.beta = mx.array([0.0, 0.5, -0.5])
    x = mx.array([[[0.3, -1.2, 2.0]]])

    alpha = beta = np.exp([0.0, 0.5, -0.5])
    values = np.array([0.3, -1.2, 2.0])
    expected = values + np.sin(values * alpha) ** 2 / (beta + 1e-9)

    assert relative_error(snake(x)[0, 0], expected) < 1e-5


@pytest.mark.parametrize("frames", [8, 32])
def test_decoder_matches_the_reference(frames):
    import torch

    ref, model = _build_pair()
    latent = np.random.default_rng(2).standard_normal(
        (1, CONFIG.latent_dim, frames), dtype=np.float32
    )

    with torch.no_grad():
        expected = ref(torch.from_numpy(latent.copy())).numpy()
    out = model.decode(mx.array(latent))

    assert out.shape == tuple(expected.shape)
    assert out.shape[1] == CONFIG.out_channels
    assert relative_error(out, expected) < 1e-4


def test_tiled_decode_equals_full_decode():
    """Halo size comes from the receptive field, and ``decode_tiled`` refuses anything smaller."""
    _, model = _build_pair()
    latent = mx.array(
        np.random.default_rng(4).standard_normal((1, CONFIG.latent_dim, 40), dtype=np.float32)
    )

    core = 8
    halo = model.required_halo(core)
    assert halo > 0

    full = model.decode(latent)
    tiled = model.decode_tiled(latent, core_frames=core, halo_frames=halo)

    assert tiled.shape[-1] <= full.shape[-1]
    assert relative_error(tiled, full[..., : tiled.shape[-1]]) < 1e-4

    with pytest.raises(ValueError, match="halo_frames must be at least"):
        model.decode_tiled(latent, core_frames=core, halo_frames=halo - 1)


def test_output_length_matches_actual_decode():
    _, model = _build_pair()
    for frames in (5, 17, 33):
        latent = mx.zeros((1, CONFIG.latent_dim, frames))
        assert model.decode(latent).shape[-1] == model.output_length(frames)


def test_upsampling_ratio_is_the_product_of_strides():
    _, model = _build_pair()
    assert model.config.downsampling_ratio == math.prod(CONFIG.strides)

    latent = mx.zeros((1, CONFIG.latent_dim, 16))
    samples = model.decode(latent).shape[-1]
    # Edge effects shorten the tail slightly, so this is approximate by design.
    assert abs(samples - 16 * model.config.downsampling_ratio) < model.config.downsampling_ratio
