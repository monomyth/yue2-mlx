"""YuE2's Oobleck audio decoder in MLX: 64-dim latents to 48 kHz stereo.

Derived from stable-audio-tools. A stack of transposed convolutions upsamples by 1920x in total
(strides 2, 2, 4, 4, 5, 6), each followed by dilated residual units, with SnakeBeta activations
throughout — a periodic nonlinearity that helps a network produce audio that actually oscillates.

Only the decoder is ported; generation never needs the encoder. MLX convolutions are channels-last,
so tensors here are ``[B, T, C]`` and the public ``decode`` converts at the boundary. Module
attribute names track the checkpoint's ``nn.Sequential`` indices, so weights load unrenamed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn


@dataclass(frozen=True)
class VAEConfig:
    latent_dim: int = 64
    out_channels: int = 2
    channels: int = 64
    c_mults: tuple[int, ...] = (1, 2, 4, 8, 16, 32)
    strides: tuple[int, ...] = (2, 2, 4, 4, 5, 6)
    sample_rate: int = 48000
    audio_channels: int = 2
    final_tanh: bool = False

    @property
    def downsampling_ratio(self) -> int:
        return math.prod(self.strides)


class SnakeBeta(nn.Module):
    """``x + sin(alpha*x)^2 / beta``, with both parameters stored in log space."""

    def __init__(self, channels: int):
        super().__init__()
        self.alpha = mx.zeros((channels,))
        self.beta = mx.zeros((channels,))

    def __call__(self, x: mx.array) -> mx.array:
        alpha = mx.exp(self.alpha.astype(x.dtype))
        beta = mx.exp(self.beta.astype(x.dtype))
        return x + mx.square(mx.sin(x * alpha)) / (beta + 1e-9)


class Conv1d(nn.Module):
    """1D convolution in channels-last layout, weights stored as ``(O, K, C)``."""

    TORCH_PERMUTATION = (0, 2, 1)  # from PyTorch's (O, C, K)

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.weight = mx.zeros((out_channels, kernel_size, in_channels))
        if bias:
            self.bias = mx.zeros((out_channels,))
        self.padding = padding
        self.dilation = dilation

    def __call__(self, x: mx.array) -> mx.array:
        out = mx.conv1d(
            x, self.weight.astype(x.dtype), padding=self.padding, dilation=self.dilation
        )
        return out + self.bias.astype(out.dtype) if "bias" in self else out


class ConvTranspose1d(nn.Module):
    """Transposed 1D convolution: PyTorch keeps ``(Cin, Cout, K)``, MLX ``(Cout, K, Cin)``."""

    TORCH_PERMUTATION = (1, 2, 0)

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
    ):
        super().__init__()
        self.weight = mx.zeros((out_channels, kernel_size, in_channels))
        self.bias = mx.zeros((out_channels,))
        self.stride = stride
        self.padding = padding

    def __call__(self, x: mx.array) -> mx.array:
        out = mx.conv_transpose1d(
            x, self.weight.astype(x.dtype), stride=self.stride, padding=self.padding
        )
        return out + self.bias.astype(out.dtype)


class ResidualUnit(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dilation: int):
        super().__init__()
        self.layers = [
            SnakeBeta(out_channels),
            Conv1d(in_channels, out_channels, 7, padding=(dilation * 6) // 2, dilation=dilation),
            SnakeBeta(out_channels),
            Conv1d(out_channels, out_channels, 1),
        ]

    def __call__(self, x: mx.array) -> mx.array:
        out = x
        for layer in self.layers:
            out = layer(out)
        return out + x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.layers = [
            SnakeBeta(in_channels),
            ConvTranspose1d(
                in_channels, out_channels, 2 * stride, stride=stride, padding=math.ceil(stride / 2)
            ),
            ResidualUnit(out_channels, out_channels, 1),
            ResidualUnit(out_channels, out_channels, 3),
            ResidualUnit(out_channels, out_channels, 9),
        ]

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.layers:
            x = layer(x)
        return x


class _Tanh(nn.Module):
    def __call__(self, x: mx.array) -> mx.array:
        return mx.tanh(x)


class _Identity(nn.Module):
    def __call__(self, x: mx.array) -> mx.array:
        return x


class OobleckDecoder(nn.Module):
    def __init__(self, config: VAEConfig):
        super().__init__()
        mults = [1] + list(config.c_mults)
        depth = len(mults)

        layers: list[nn.Module] = [
            Conv1d(config.latent_dim, mults[-1] * config.channels, 7, padding=3)
        ]
        for index in range(depth - 1, 0, -1):
            layers.append(
                DecoderBlock(
                    mults[index] * config.channels,
                    mults[index - 1] * config.channels,
                    config.strides[index - 1],
                )
            )
        layers.append(SnakeBeta(mults[0] * config.channels))
        layers.append(
            Conv1d(mults[0] * config.channels, config.out_channels, 7, padding=3, bias=False)
        )
        layers.append(_Tanh() if config.final_tanh else _Identity())
        self.layers = layers

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.layers:
            x = layer(x)
        return x


def _dependency_interval(module, low: int, high: int) -> tuple[int, int]:
    """Which input positions an output interval depends on, walking the stack backwards.

    Used to size tile halos exactly. Ported from upstream so the two agree on what "enough
    context" means.
    """
    if isinstance(module, list):
        for child in reversed(module):
            low, high = _dependency_interval(child, low, high)
        return low, high
    if isinstance(module, (OobleckDecoder, DecoderBlock)):
        return _dependency_interval(module.layers, low, high)
    if isinstance(module, ResidualUnit):
        # The residual path means the identity's support counts too.
        inner_low, inner_high = _dependency_interval(module.layers, low, high)
        return min(inner_low, low), max(inner_high, high)
    if isinstance(module, ConvTranspose1d):
        stride, padding, kernel = module.stride, module.padding, module.weight.shape[1]
        return -(-(low + padding - (kernel - 1)) // stride), (high + padding) // stride
    if isinstance(module, Conv1d):
        padding, dilation, kernel = module.padding, module.dilation, module.weight.shape[1]
        return low - padding, high - padding + dilation * (kernel - 1)
    if isinstance(module, (SnakeBeta, _Tanh, _Identity)):
        return low, high
    raise TypeError(f"no receptive-field rule for {type(module).__name__}")


def _output_length(module, length: int) -> int:
    if isinstance(module, list):
        for child in module:
            length = _output_length(child, length)
        return length
    if isinstance(module, (OobleckDecoder, DecoderBlock)):
        return _output_length(module.layers, length)
    if isinstance(module, ConvTranspose1d):
        return (length - 1) * module.stride - 2 * module.padding + module.weight.shape[1]
    if isinstance(module, Conv1d):
        return length + 2 * module.padding - module.dilation * (module.weight.shape[1] - 1)
    if isinstance(module, (ResidualUnit, SnakeBeta, _Tanh, _Identity)):
        return length
    raise TypeError(f"no length rule for {type(module).__name__}")


class YuE2VAE(nn.Module):
    def __init__(self, config: VAEConfig | None = None):
        super().__init__()
        self.config = config or VAEConfig()
        self.decoder = OobleckDecoder(self.config)

    def output_length(self, frames: int) -> int:
        """Samples produced for ``frames`` latent frames -- slightly under ``1920 * frames``."""
        return _output_length(self.decoder, frames)

    def required_halo(self, core_frames: int) -> int:
        """Smallest halo that makes each tile's core identical to a full decode."""
        ratio = self.config.downsampling_ratio
        low, high = _dependency_interval(self.decoder, 0, core_frames * ratio - 1)
        return max(0, -low, high - core_frames + 1)

    def decode(self, latent: mx.array) -> mx.array:
        """``[B, latent_dim, T]`` latents (PyTorch axis order) → ``[B, channels, samples]``."""
        x = latent.transpose(0, 2, 1).astype(self.decoder.layers[0].weight.dtype)
        return self.decoder(x).transpose(0, 2, 1)

    def decode_tiled(
        self,
        latent: mx.array,
        core_frames: int = 1024,
        halo_frames: int = 16,
        report=None,
    ) -> mx.array:
        """Decode in tiles with enough context on each side that cores are exact.

        The receptive field is finite, so a tile decoded with `halo_frames` of extra context on each
        side yields a core identical to the full decode. No crossfading is involved.
        """
        frames = latent.shape[-1]
        if core_frames < 1 or halo_frames < 0:
            raise ValueError("core_frames must be positive and halo_frames non-negative")
        required = self.required_halo(core_frames)
        if halo_frames < required:
            raise ValueError(
                f"halo_frames must be at least {required} for this decoder, got {halo_frames}"
            )

        ratio = self.config.downsampling_ratio
        tiles = (frames + core_frames - 1) // core_frames
        pieces = []
        for index, start in enumerate(range(0, frames, core_frames)):
            end = min(frames, start + core_frames)
            left, right = max(0, start - halo_frames), min(frames, end + halo_frames)
            decoded = self.decode(latent[..., left:right])
            crop_start = (start - left) * ratio
            pieces.append(decoded[..., crop_start : crop_start + (end - start) * ratio])
            mx.eval(pieces[-1])
            if report is not None:
                report(index + 1, tiles)
        return mx.concatenate(pieces, axis=-1)
