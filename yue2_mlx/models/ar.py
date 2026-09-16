"""YuE2's autoregressive stage: a Qwen3-architecture language model, in MLX.

This is the model that turns lyrics and a style description into an optional ABC score and then a
stream of audio codec tokens at 25 per second. Architecturally it is Qwen3 — grouped-query
attention with per-head RMS norm on q and k, SwiGLU, half-split RoPE at theta 1e6 — over a
184,704-token vocabulary whose upper block is audio codec ids.

It has a second job beyond generating tokens. ``condition`` replays a token sequence and hands back
every layer's post-RoPE keys and values, which the acoustic stage prepends to its own attention.
That is how conditioning flows between the two models: not through final hidden states, but through
each layer's attention memory.

Attribute names mirror the checkpoint (``model.layers.N.self_attn.q_proj`` and friends), so weights
load without renaming.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn


@dataclass(frozen=True)
class ARConfig:
    hidden_size: int = 2048
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    intermediate_size: int = 6144
    vocab_size: int = 184704
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    max_position_embeddings: int = 24576

    @property
    def kv_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads


class KVCache:
    """Append-only key/value cache that grows in blocks to avoid reallocating every step."""

    STEP = 512

    def __init__(self):
        self.keys: mx.array | None = None
        self.values: mx.array | None = None
        self.offset = 0

    def update_and_fetch(self, keys: mx.array, values: mx.array) -> tuple[mx.array, mx.array]:
        added = keys.shape[2]
        if self.keys is None or self.offset + added > self.keys.shape[2]:
            batch, heads, _, dim = keys.shape
            grow = ((added + self.STEP - 1) // self.STEP) * self.STEP
            spare = mx.zeros((batch, heads, grow, dim), keys.dtype)
            if self.keys is None:
                self.keys, self.values = spare, mx.zeros_like(spare)
            else:
                self.keys = mx.concatenate([self.keys[..., : self.offset, :], spare], axis=2)
                self.values = mx.concatenate([self.values[..., : self.offset, :], spare], axis=2)

        self.keys[..., self.offset : self.offset + added, :] = keys
        self.values[..., self.offset : self.offset + added, :] = values
        self.offset += added
        return self.keys[..., : self.offset, :], self.values[..., : self.offset, :]


class QwenRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((dim,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, self.weight, self.eps)


class Attention(nn.Module):
    def __init__(self, config: ARConfig):
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
        # Qwen3 normalizes each head independently, unlike Wan which norms the whole concatenation.
        self.q_norm = QwenRMSNorm(dim, config.rms_norm_eps)
        self.k_norm = QwenRMSNorm(dim, config.rms_norm_eps)
        self.scale = dim**-0.5

    def project(self, x: mx.array, offset: int) -> tuple[mx.array, mx.array, mx.array]:
        """Return rotated q/k and v in ``[B, heads, T, D]`` layout."""
        config = self.config
        batch, seq_len, _ = x.shape
        dim = config.head_dim

        q = self.q_norm(self.q_proj(x).reshape(batch, seq_len, config.num_attention_heads, dim))
        k = self.k_norm(self.k_proj(x).reshape(batch, seq_len, config.num_key_value_heads, dim))
        v = self.v_proj(x).reshape(batch, seq_len, config.num_key_value_heads, dim)

        q, k, v = (a.transpose(0, 2, 1, 3) for a in (q, k, v))
        q = mx.fast.rope(
            q, dim, traditional=False, base=config.rope_theta, scale=1.0, offset=offset
        )
        k = mx.fast.rope(
            k, dim, traditional=False, base=config.rope_theta, scale=1.0, offset=offset
        )
        return q, k, v

    def __call__(self, x: mx.array, cache: KVCache | None, mask) -> mx.array:
        offset = cache.offset if cache is not None else 0
        q, k, v = self.project(x, offset)
        if cache is not None:
            k, v = cache.update_and_fetch(k, v)

        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(x.shape[0], x.shape[1], -1)
        return self.o_proj(out)


class MLP(nn.Module):
    def __init__(self, config: ARConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(self, config: ARConfig):
        super().__init__()
        self.self_attn = Attention(config)
        self.mlp = MLP(config)
        self.input_layernorm = QwenRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = QwenRMSNorm(config.hidden_size, config.rms_norm_eps)

    def __call__(self, x: mx.array, cache: KVCache | None, mask) -> mx.array:
        x = x + self.self_attn(self.input_layernorm(x), cache, mask)
        return x + self.mlp(self.post_attention_layernorm(x))


class Backbone(nn.Module):
    def __init__(self, config: ARConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        self.norm = QwenRMSNorm(config.hidden_size, config.rms_norm_eps)


class YuE2AR(nn.Module):
    def __init__(self, config: ARConfig):
        super().__init__()
        self.config = config
        self.model = Backbone(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def make_cache(self) -> list[KVCache]:
        return [KVCache() for _ in range(self.config.num_hidden_layers)]

    def __call__(
        self,
        tokens: mx.array,
        cache: list[KVCache] | None = None,
        eval_every: int = 0,
    ) -> mx.array:
        """Logits for the last position of ``tokens`` (``[B, T]``); advances ``cache`` in place.

        ``eval_every`` matters for prefill, where a 24k-token prompt through 28 layers is a single
        long Metal command buffer that macOS may kill for stalling the display. Single-token decode
        steps are short enough not to need it.
        """
        x = self.model.embed_tokens(tokens)
        mask = "causal" if x.shape[1] > 1 else None
        for index, (layer, layer_cache) in enumerate(
            zip(self.model.layers, cache or [None] * len(self.model.layers), strict=True), start=1
        ):
            x = layer(x, layer_cache, mask)
            if eval_every and index % eval_every == 0:
                mx.eval(x)
        return self.lm_head(self.model.norm(x[:, -1:]))

    def condition(self, token_ids: list[int]) -> list[tuple[mx.array, mx.array]]:
        """Replay ``token_ids`` and return each layer's post-RoPE keys and values.

        Returned as ``[S, kv_heads, head_dim]`` per layer, the layout the acoustic stage prepends to
        its own attention. This is a full causal forward, so cost matches one prefill.
        """
        x = self.model.embed_tokens(mx.array(token_ids)[None])
        memory = []
        for layer in self.model.layers:
            attn = layer.self_attn
            q, k, v = attn.project(layer.input_layernorm(x), offset=0)
            memory.append((k[0].transpose(1, 0, 2), v[0].transpose(1, 0, 2)))

            out = mx.fast.scaled_dot_product_attention(q, k, v, scale=attn.scale, mask="causal")
            out = out.transpose(0, 2, 1, 3).reshape(x.shape[0], x.shape[1], -1)
            x = x + attn.o_proj(out)
            x = x + layer.mlp(layer.post_attention_layernorm(x))
            mx.eval(x)
        return memory
