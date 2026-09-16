"""YuE2's acoustic stage: flow matching from codec tokens to audio latents, in MLX.

Same transformer shape as the AR stage, used differently. There is no cross-attention: each layer
prepends the AR model's keys and values for this chunk to its own, so the latents attend over the
whole conditioning sequence and over each other. Attention is bidirectional over latents — this is
not a language model, it is a denoiser over a fixed-length latent sequence.

Two details that matter and are easy to get wrong:

- The latent sequence is padded by one frame at each end before the transformer and cropped after.
  Positions continue from the end of the AR sequence rather than restarting at zero.
- The integrator is midpoint, not Euler: two model calls per step, with timesteps drawn from a
  logit-spaced schedule and passed through a sigmoid inside the model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from .ar import MLP, QwenRMSNorm


@dataclass(frozen=True)
class AcousticConfig:
    hidden_size: int = 2048
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    intermediate_size: int = 6144
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    latent_dim: int = 64
    max_latent_frames: int = 24576
    timestep_shift: float = 1.0

    @property
    def kv_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads


class AcousticAttention(nn.Module):
    def __init__(self, config: AcousticConfig):
        super().__init__()
        self.config = config
        heads, kv_heads, dim = (
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim,
        )
        self.q_proj = nn.Linear(config.hidden_size, heads * dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, kv_heads * dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, kv_heads * dim, bias=False)
        self.o_proj = nn.Linear(heads * dim, config.hidden_size, bias=False)
        self.q_norm = QwenRMSNorm(dim, config.rms_norm_eps)
        self.k_norm = QwenRMSNorm(dim, config.rms_norm_eps)
        self.scale = dim**-0.5

    def __call__(
        self,
        x: mx.array,
        memory: tuple[mx.array, mx.array],
        cos: mx.array,
        sin: mx.array,
    ) -> mx.array:
        config = self.config
        batch, seq_len, _ = x.shape
        dim = config.head_dim

        q = self.q_norm(self.q_proj(x).reshape(batch, seq_len, config.num_attention_heads, dim))
        k = self.k_norm(self.k_proj(x).reshape(batch, seq_len, config.num_key_value_heads, dim))
        v = self.v_proj(x).reshape(batch, seq_len, config.num_key_value_heads, dim)

        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)

        # Prepend the AR memory, then expand key/value heads to query heads. Repeating each kv head
        # consecutively is what makes query head h read kv head h // kv_groups.
        prefix_k, prefix_v = memory
        k = mx.concatenate([prefix_k.astype(k.dtype)[None], k], axis=1)
        v = mx.concatenate([prefix_v.astype(v.dtype)[None], v], axis=1)
        if config.kv_groups > 1:
            k = mx.repeat(k, config.kv_groups, axis=2)
            v = mx.repeat(v, config.kv_groups, axis=2)

        out = mx.fast.scaled_dot_product_attention(
            q.transpose(0, 2, 1, 3),
            k.transpose(0, 2, 1, 3),
            v.transpose(0, 2, 1, 3),
            scale=self.scale,
        )
        out = out.transpose(0, 2, 1, 3).reshape(batch, seq_len, -1)
        return self.o_proj(out)


def _apply_rope(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """Half-split rotation on ``[B, T, H, D]``, the Qwen/NeoX convention."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    cos = cos.astype(x.dtype)[None, :, None, :]
    sin = sin.astype(x.dtype)[None, :, None, :]
    return mx.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)


def rope_tables(positions: mx.array, head_dim: int, base: float) -> tuple[mx.array, mx.array]:
    inv_freq = 1.0 / mx.power(base, mx.arange(0, head_dim, 2, dtype=mx.float32) / head_dim)
    angles = positions.astype(mx.float32).reshape(-1, 1) * inv_freq.reshape(1, -1)
    return mx.cos(angles), mx.sin(angles)


class AcousticLayer(nn.Module):
    def __init__(self, config: AcousticConfig):
        super().__init__()
        self.nar_input_layernorm = QwenRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.nar_self_attn = AcousticAttention(config)
        self.nar_pre_mlp_layernorm = QwenRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.nar_mlp = MLP(config)

    def __call__(self, x, memory, cos, sin):
        x = x + self.nar_self_attn(self.nar_input_layernorm(x), memory, cos, sin)
        return x + self.nar_mlp(self.nar_pre_mlp_layernorm(x))


class AcousticBackbone(nn.Module):
    def __init__(self, config: AcousticConfig):
        super().__init__()
        self.layers = [AcousticLayer(config) for _ in range(config.num_hidden_layers)]
        self.norm = QwenRMSNorm(config.hidden_size, config.rms_norm_eps)


class _SiLU(nn.Module):
    """Parameter-free slot that keeps ``mlp.0`` / ``mlp.2`` aligned with the checkpoint."""

    def __call__(self, x: mx.array) -> mx.array:
        return nn.silu(x)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = [
            nn.Linear(frequency_embedding_size, hidden_size),
            _SiLU(),
            nn.Linear(hidden_size, hidden_size),
        ]
        self.frequency_embedding_size = frequency_embedding_size

    def __call__(self, t: mx.array) -> mx.array:
        half = self.frequency_embedding_size // 2
        freqs = mx.exp(-math.log(10000) * mx.arange(half, dtype=mx.float32) / half)
        args = t.astype(mx.float32).reshape(-1, 1) * freqs.reshape(1, -1)
        emb = mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)
        emb = emb.astype(self.mlp[0].weight.dtype)
        return self.mlp[2](self.mlp[1](self.mlp[0](emb)))


class AudioPositionEmbedding(nn.Module):
    """Fixed sinusoidal table, stored in the checkpoint rather than recomputed."""

    def __init__(self, max_frames: int, hidden_size: int):
        super().__init__()
        self.pe = mx.zeros((max_frames, hidden_size))

    def __call__(self, count: int) -> mx.array:
        return self.pe[:count]


class YuE2Acoustic(nn.Module):
    def __init__(self, config: AcousticConfig):
        super().__init__()
        self.config = config
        self.model = AcousticBackbone(config)
        self.llm2vae = nn.Linear(config.hidden_size, config.latent_dim)
        self.vae2llm = nn.Linear(config.latent_dim, config.hidden_size)
        self.time_embedder = TimestepEmbedder(config.hidden_size)
        self.latent_pos_embed = AudioPositionEmbedding(config.max_latent_frames, config.hidden_size)

    def velocity(
        self,
        state: mx.array,
        raw_t: mx.array,
        memory: list[tuple[mx.array, mx.array]],
        cos: mx.array,
        sin: mx.array,
        position_embedding: mx.array,
        eval_every: int = 0,
    ) -> mx.array:
        """One denoiser evaluation: ``[T, latent_dim]`` in, same shape out.

        ``eval_every`` splits the lazy graph into shorter Metal command buffers. macOS kills a
        command buffer that occupies the GPU long enough to stall the display, and 28 layers over
        thousands of latent frames is comfortably past that threshold.
        """
        config = self.config
        t = mx.sigmoid(raw_t.astype(mx.float32))
        shift = config.timestep_shift
        t = shift * t / (1 + (shift - 1) * t)

        padded = mx.pad(state, [(1, 1), (0, 0)])
        x = self.vae2llm(padded.astype(self.vae2llm.weight.dtype))[None]
        x = x + self.time_embedder(t.reshape(1)) + position_embedding

        for index, (layer, layer_memory) in enumerate(
            zip(self.model.layers, memory, strict=True), start=1
        ):
            x = layer(x, layer_memory, cos, sin)
            if eval_every and index % eval_every == 0:
                mx.eval(x)

        return self.llm2vae(self.model.norm(x))[0, 1:-1]

    def synthesize(
        self,
        noise: mx.array,
        memory: list[tuple[mx.array, mx.array]],
        ar_length: int,
        steps: int,
        report=None,
        eval_every: int = 4,
    ) -> mx.array:
        """Integrate noise to audio latents with a midpoint solver, two calls per step."""
        config = self.config
        state = noise.astype(self.vae2llm.weight.dtype)
        length = state.shape[0] + 2

        positions = mx.arange(ar_length, ar_length + length)
        cos, sin = rope_tables(positions, config.head_dim, config.rope_theta)
        position_embedding = self.latent_pos_embed(length)[None]

        # Logit-spaced schedule over 2*steps values; the model applies the sigmoid itself.
        grid = mx.arange(2 * steps, 0, -1, dtype=mx.float32) / (2 * steps)
        raw_steps = mx.clip(mx.log(grid / (1 - grid)), -20, 20)

        for step in range(steps):
            first = self.velocity(
                state, raw_steps[2 * step], memory, cos, sin, position_embedding, eval_every
            )
            midpoint = state - first / (2 * steps)
            second = self.velocity(
                midpoint, raw_steps[2 * step + 1], memory, cos, sin, position_embedding, eval_every
            )
            state = state - second / steps
            mx.eval(state)
            if report is not None:
                report(step + 1, steps)
        return state.astype(mx.float32)
