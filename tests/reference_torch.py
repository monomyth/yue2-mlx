"""Independent PyTorch reference implementations of the YuE2 stages.

Written from the architecture in idiomatic, channels-first PyTorch, using PyTorch's own
convolutions and attention. They exist so the MLX port can be diffed against a second expression of
the same maths without depending on any other repository.

What this catches, which is where porting bugs actually live: channels-last conversions, kernel
layout permutations for 1D and transposed 1D convolutions, the half-split RoPE convention,
grouped-query head expansion, the order of the residual and norm operations, and the midpoint
integrator's timestep schedule.

What it cannot catch: a misunderstanding of the architecture shared by both implementations, since
both were written by the same author from the same reading. Two other things cover that gap. The
autoregressive stage is checked against HuggingFace's ``Qwen3ForCausalLM``, which nobody here wrote.
And the published checkpoints are loaded and asked what they predict, which only agrees with the
model's own training if the port is right (see ``scripts/validate_yue2_weights.py``).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class RefRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        scale = torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x.float() * scale).to(x.dtype) * self.weight


def ref_apply_rope(x, cos, sin):
    """Half-split rotation on ``[B, T, H, D]``; cos and sin are ``[T, D/2]``."""
    half = x.shape[-1] // 2
    first, second = x[..., :half], x[..., half:]
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]
    return torch.cat([first * cos - second * sin, second * cos + first * sin], dim=-1)


def ref_rope_tables(positions, head_dim: int, base: float):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    angles = positions.float().unsqueeze(-1) * inv_freq.unsqueeze(0)
    return angles.cos(), angles.sin()


class RefAcousticAttention(nn.Module):
    """Self-attention over latent frames, with the AR stage's keys and values prepended."""

    def __init__(self, hidden, heads, kv_heads, head_dim, eps):
        super().__init__()
        self.heads, self.kv_heads, self.head_dim = heads, kv_heads, head_dim
        self.q_proj = nn.Linear(hidden, heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(heads * head_dim, hidden, bias=False)
        self.q_norm = RefRMSNorm(head_dim, eps)
        self.k_norm = RefRMSNorm(head_dim, eps)

    def forward(self, x, memory, cos, sin):
        batch, length, _ = x.shape
        heads, kv_heads, dim = self.heads, self.kv_heads, self.head_dim

        query = self.q_norm(self.q_proj(x).view(batch, length, heads, dim))
        key = self.k_norm(self.k_proj(x).view(batch, length, kv_heads, dim))
        value = self.v_proj(x).view(batch, length, kv_heads, dim)

        query = ref_apply_rope(query, cos, sin)
        key = ref_apply_rope(key, cos, sin)

        prefix_key, prefix_value = memory
        key = torch.cat([prefix_key.unsqueeze(0), key], dim=1)
        value = torch.cat([prefix_value.unsqueeze(0), value], dim=1)

        groups = heads // kv_heads
        if groups > 1:
            key = key.repeat_interleave(groups, dim=2)
            value = value.repeat_interleave(groups, dim=2)

        out = F.scaled_dot_product_attention(
            query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
        )
        return self.o_proj(out.transpose(1, 2).reshape(batch, length, heads * dim))


class RefMLP(nn.Module):
    def __init__(self, hidden, intermediate):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class RefAcousticLayer(nn.Module):
    def __init__(self, hidden, heads, kv_heads, head_dim, intermediate, eps):
        super().__init__()
        self.nar_input_layernorm = RefRMSNorm(hidden, eps)
        self.nar_self_attn = RefAcousticAttention(hidden, heads, kv_heads, head_dim, eps)
        self.nar_pre_mlp_layernorm = RefRMSNorm(hidden, eps)
        self.nar_mlp = RefMLP(hidden, intermediate)

    def forward(self, x, memory, cos, sin):
        x = x + self.nar_self_attn(self.nar_input_layernorm(x), memory, cos, sin)
        return x + self.nar_mlp(self.nar_pre_mlp_layernorm(x))


class RefTimestepEmbedder(nn.Module):
    def __init__(self, hidden, frequency_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_size, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )
        self.frequency_size = frequency_size

    def forward(self, t):
        half = self.frequency_size // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32) / half)
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        return self.mlp(torch.cat([args.cos(), args.sin()], dim=-1))


class RefAcousticModel(nn.Module):
    """Flow-matching denoiser over audio latents."""

    def __init__(
        self,
        hidden=64,
        layers=3,
        heads=4,
        kv_heads=2,
        head_dim=16,
        intermediate=128,
        latent_dim=8,
        max_frames=256,
        eps=1e-6,
        rope_theta=1000000.0,
        timestep_shift=1.0,
    ):
        super().__init__()
        self.head_dim, self.rope_theta = head_dim, rope_theta
        self.timestep_shift = timestep_shift
        self.layers = nn.ModuleList(
            RefAcousticLayer(hidden, heads, kv_heads, head_dim, intermediate, eps)
            for _ in range(layers)
        )
        self.norm = RefRMSNorm(hidden, eps)
        self.llm2vae = nn.Linear(hidden, latent_dim)
        self.vae2llm = nn.Linear(latent_dim, hidden)
        self.time_embedder = RefTimestepEmbedder(hidden)
        self.register_buffer("pe", torch.zeros(max_frames, hidden))

    def velocity(self, state, raw_t, memory, cos, sin):
        t = torch.sigmoid(raw_t.float())
        shift = self.timestep_shift
        t = shift * t / (1 + (shift - 1) * t)

        padded = F.pad(state, (0, 0, 1, 1)).unsqueeze(0)
        x = self.vae2llm(padded)
        x = x + self.time_embedder(t.reshape(1)) + self.pe[: padded.shape[1]].unsqueeze(0)

        # Each layer attends into its own slice of the autoregressive stage's memory.
        for layer, layer_memory in zip(self.layers, memory, strict=True):
            x = layer(x, layer_memory, cos, sin)
        return self.llm2vae(self.norm(x))[0, 1:-1]

    def synthesize(self, noise, memory, ar_length, steps):
        """Midpoint integration, two evaluations per step."""
        length = noise.shape[0] + 2
        positions = torch.arange(ar_length, ar_length + length)
        cos, sin = ref_rope_tables(positions, self.head_dim, self.rope_theta)

        grid = torch.arange(2 * steps, 0, -1, dtype=torch.float64) / (2 * steps)
        raw = torch.logit(grid).clamp(-20, 20).float()

        state = noise
        for step in range(steps):
            first = self.velocity(state, raw[2 * step], memory, cos, sin)
            midpoint = state - first / (2 * steps)
            second = self.velocity(midpoint, raw[2 * step + 1], memory, cos, sin)
            state = state - second / steps
        return state


class RefSnakeBeta(nn.Module):
    """``x + sin(alpha*x)^2 / beta``, both parameters in log space."""

    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        alpha = self.alpha.exp().view(1, -1, 1)
        beta = self.beta.exp().view(1, -1, 1)
        return x + torch.sin(x * alpha).pow(2) / (beta + 1e-9)


class RefResidualUnit(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__()
        self.layers = nn.Sequential(
            RefSnakeBeta(out_channels),
            nn.Conv1d(in_channels, out_channels, 7, dilation=dilation, padding=dilation * 3),
            RefSnakeBeta(out_channels),
            nn.Conv1d(out_channels, out_channels, 1),
        )

    def forward(self, x):
        return x + self.layers(x)


class RefDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.layers = nn.Sequential(
            RefSnakeBeta(in_channels),
            nn.ConvTranspose1d(
                in_channels, out_channels, 2 * stride, stride=stride, padding=math.ceil(stride / 2)
            ),
            RefResidualUnit(out_channels, out_channels, 1),
            RefResidualUnit(out_channels, out_channels, 3),
            RefResidualUnit(out_channels, out_channels, 9),
        )

    def forward(self, x):
        return self.layers(x)


class RefOobleckDecoder(nn.Module):
    """Latents to waveform: one convolution in, transposed-convolution upsampling out."""

    def __init__(
        self, latent_dim=64, out_channels=2, channels=64, c_mults=(1, 2, 4), strides=(2, 2, 4)
    ):
        super().__init__()
        mults = [1, *c_mults]
        depth = len(mults)

        layers: list[nn.Module] = [nn.Conv1d(latent_dim, mults[-1] * channels, 7, padding=3)]
        for index in range(depth - 1, 0, -1):
            layers.append(
                RefDecoderBlock(
                    mults[index] * channels, mults[index - 1] * channels, strides[index - 1]
                )
            )
        layers.append(RefSnakeBeta(mults[0] * channels))
        layers.append(nn.Conv1d(mults[0] * channels, out_channels, 7, padding=3, bias=False))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
