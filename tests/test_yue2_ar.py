"""Compare the MLX AR stage against HuggingFace's ``Qwen3ForCausalLM``.

YuE2's AR stage *is* Qwen3, and the checkpoint uses HuggingFace's key names, so HuggingFace is both
an independent implementation and the natural reference.

``condition`` is checked against HuggingFace's ``past_key_values``, which is exactly the per-layer
post-RoPE keys and values the acoustic stage consumes.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest
from helpers import copy_weights, relative_error

from yue2_mlx.models.ar import ARConfig, YuE2AR

transformers = pytest.importorskip("transformers", reason="pip install transformers for AR parity")

CONFIG = ARConfig(
    hidden_size=64,
    num_hidden_layers=3,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    intermediate_size=128,
    vocab_size=256,
    rope_theta=1000000.0,
)


def _build_pair(seed: int = 0):
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM

    hf_config = Qwen3Config(
        hidden_size=CONFIG.hidden_size,
        num_hidden_layers=CONFIG.num_hidden_layers,
        num_attention_heads=CONFIG.num_attention_heads,
        num_key_value_heads=CONFIG.num_key_value_heads,
        head_dim=CONFIG.head_dim,
        intermediate_size=CONFIG.intermediate_size,
        vocab_size=CONFIG.vocab_size,
        rms_norm_eps=CONFIG.rms_norm_eps,
        rope_theta=CONFIG.rope_theta,
        max_position_embeddings=CONFIG.max_position_embeddings,
        tie_word_embeddings=False,
        attn_implementation="eager",
    )
    torch.manual_seed(seed)
    ref = Qwen3ForCausalLM(hf_config).eval()

    generator = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        for param in ref.parameters():
            param.copy_(torch.randn(param.shape, generator=generator) * 0.08)

    model = copy_weights(ref, YuE2AR(CONFIG))
    return ref, model


def _tokens(length: int, seed: int = 3) -> np.ndarray:
    return (
        np.random.default_rng(seed)
        .integers(0, CONFIG.vocab_size, size=(1, length))
        .astype(np.int32)
    )


@pytest.mark.parametrize("length", [1, 7, 32])
def test_prefill_logits_match_huggingface(length):
    import torch

    ref, model = _build_pair()
    tokens = _tokens(length)

    with torch.no_grad():
        expected = ref(torch.from_numpy(tokens.astype(np.int64))).logits[:, -1:]
    out = model(mx.array(tokens), model.make_cache())

    assert out.shape == tuple(expected.shape)
    assert relative_error(out, expected.numpy()) < 2e-4


def test_incremental_decoding_matches_full_prefill():
    """Feeding tokens one at a time through the cache must equal prefilling them together."""
    _, model = _build_pair()
    tokens = _tokens(12)

    prefilled = model(mx.array(tokens), model.make_cache())

    cache = model.make_cache()
    stepwise = None
    for index in range(tokens.shape[1]):
        stepwise = model(mx.array(tokens[:, index : index + 1]), cache)
    mx.eval(stepwise)

    assert cache[0].offset == tokens.shape[1]
    assert relative_error(stepwise, prefilled) < 2e-4


def test_condition_matches_huggingface_key_values():
    """The per-layer memory handed to the acoustic stage is HuggingFace's past_key_values."""
    import torch

    ref, model = _build_pair()
    tokens = _tokens(9)

    with torch.no_grad():
        outputs = ref(torch.from_numpy(tokens.astype(np.int64)), use_cache=True)
    past = outputs.past_key_values

    memory = model.condition(tokens[0].tolist())
    assert len(memory) == CONFIG.num_hidden_layers

    for layer in range(CONFIG.num_hidden_layers):
        ref_k, ref_v = past.layers[layer].keys, past.layers[layer].values
        # HuggingFace keeps [B, kv_heads, S, D]; this port hands over [S, kv_heads, D].
        keys, values = memory[layer]
        assert keys.shape == (tokens.shape[1], CONFIG.num_key_value_heads, CONFIG.head_dim)
        assert relative_error(keys, ref_k[0].permute(1, 0, 2).numpy()) < 2e-4, (
            f"keys, layer {layer}"
        )
        assert relative_error(values, ref_v[0].permute(1, 0, 2).numpy()) < 2e-4, (
            f"values, layer {layer}"
        )
