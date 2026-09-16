"""Compare the MLX acoustic stage against an independent PyTorch implementation.

The reference in ``reference_torch`` is a second expression of the same architecture, written in
channels-first PyTorch with PyTorch's own attention. Agreement means the MLX version handles layout,
the half-split RoPE convention, grouped-query head expansion, the residual ordering and the
integrator's timestep schedule correctly.

Both were written by the same author from the same reading of the architecture, so this cannot catch
a misunderstanding common to both. Two other things cover that: the autoregressive stage is checked
against HuggingFace's Qwen3, and ``scripts/validate_yue2_weights.py`` asks the real checkpoint what
it predicts.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest
from helpers import relative_error, torch_state_to_mlx

from yue2_mlx.models.acoustic import AcousticConfig, YuE2Acoustic, rope_tables
from yue2_mlx.weights import apply_weights

pytest.importorskip("torch", reason="pip install -e '.[dev]' to run the reference comparisons")

CONFIG = AcousticConfig(
    hidden_size=64,
    num_hidden_layers=3,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    intermediate_size=128,
    latent_dim=8,
    max_latent_frames=256,
)
FRAMES = 6
AR_LENGTH = 11


def _build_pair(seed: int = 0):
    import torch
    from reference_torch import RefAcousticModel

    torch.manual_seed(seed)
    ref = RefAcousticModel(
        hidden=CONFIG.hidden_size,
        layers=CONFIG.num_hidden_layers,
        heads=CONFIG.num_attention_heads,
        kv_heads=CONFIG.num_key_value_heads,
        head_dim=CONFIG.head_dim,
        intermediate=CONFIG.intermediate_size,
        latent_dim=CONFIG.latent_dim,
        max_frames=CONFIG.max_latent_frames,
        eps=CONFIG.rms_norm_eps,
        rope_theta=CONFIG.rope_theta,
        timestep_shift=CONFIG.timestep_shift,
    ).eval()

    generator = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        for param in ref.parameters():
            param.copy_(torch.randn(param.shape, generator=generator) * 0.08)
        ref.pe.copy_(torch.randn(ref.pe.shape, generator=generator) * 0.08)

    # The reference nests differently, so map its names onto the checkpoint's layout.
    state = {}
    for name, tensor in list(ref.state_dict().items()):
        if name == "pe":
            state["latent_pos_embed.pe"] = tensor
        elif name.startswith("layers."):
            state[f"model.{name}"] = tensor
        elif name.startswith("norm."):
            state[f"model.{name}"] = tensor
        else:
            state[name] = tensor

    model = apply_weights(YuE2Acoustic(CONFIG), torch_state_to_mlx(state), mx.float32)
    return ref, model


def _memory(seed: int = 5):
    """Stand-in for the autoregressive stage's per-layer keys and values."""
    rng = np.random.default_rng(seed)
    shape = (AR_LENGTH, CONFIG.num_key_value_heads, CONFIG.head_dim)
    return [
        (
            rng.standard_normal(shape, dtype=np.float32),
            rng.standard_normal(shape, dtype=np.float32),
        )
        for _ in range(CONFIG.num_hidden_layers)
    ]


def test_rope_tables_match_the_reference():
    import torch
    from reference_torch import ref_rope_tables

    positions = mx.arange(AR_LENGTH, AR_LENGTH + FRAMES + 2)
    cos, sin = rope_tables(positions, CONFIG.head_dim, CONFIG.rope_theta)
    ref_cos, ref_sin = ref_rope_tables(
        torch.arange(AR_LENGTH, AR_LENGTH + FRAMES + 2), CONFIG.head_dim, CONFIG.rope_theta
    )

    assert cos.shape == (FRAMES + 2, CONFIG.head_dim // 2), "half-split rope halves the table width"
    assert relative_error(cos, ref_cos.numpy()) < 1e-5
    assert relative_error(sin, ref_sin.numpy()) < 1e-5


@pytest.mark.parametrize("raw_t", [8.0, 0.0, -3.5])
def test_single_velocity_evaluation_matches_the_reference(raw_t):
    import torch
    from reference_torch import ref_rope_tables

    ref, model = _build_pair()
    memory = _memory()
    state = np.random.default_rng(1).standard_normal((FRAMES, CONFIG.latent_dim), dtype=np.float32)

    positions = mx.arange(AR_LENGTH, AR_LENGTH + FRAMES + 2)
    cos, sin = rope_tables(positions, CONFIG.head_dim, CONFIG.rope_theta)
    ref_cos, ref_sin = ref_rope_tables(
        torch.arange(AR_LENGTH, AR_LENGTH + FRAMES + 2), CONFIG.head_dim, CONFIG.rope_theta
    )

    with torch.no_grad():
        expected = ref.velocity(
            torch.from_numpy(state.copy()),
            torch.tensor(raw_t),
            [
                (torch.from_numpy(keys.copy()), torch.from_numpy(values.copy()))
                for keys, values in memory
            ],
            ref_cos,
            ref_sin,
        )

    out = model.velocity(
        mx.array(state),
        mx.array(raw_t),
        [(mx.array(keys), mx.array(values)) for keys, values in memory],
        cos,
        sin,
        model.latent_pos_embed(FRAMES + 2)[None],
    )

    assert out.shape == tuple(expected.shape)
    assert relative_error(out, expected.numpy()) < 2e-4


@pytest.mark.parametrize("steps", [1, 4])
def test_midpoint_integrator_matches_the_reference(steps):
    import torch

    ref, model = _build_pair()
    memory = _memory()
    noise = np.random.default_rng(9).standard_normal((FRAMES, CONFIG.latent_dim), dtype=np.float32)

    with torch.no_grad():
        expected = ref.synthesize(
            torch.from_numpy(noise.copy()),
            [
                (torch.from_numpy(keys.copy()), torch.from_numpy(values.copy()))
                for keys, values in memory
            ],
            AR_LENGTH,
            steps,
        )

    out = model.synthesize(
        mx.array(noise),
        [(mx.array(keys), mx.array(values)) for keys, values in memory],
        AR_LENGTH,
        steps,
    )

    assert out.shape == tuple(expected.shape)
    assert relative_error(out, expected.numpy()) < 5e-4


def test_timestep_schedule_is_logit_spaced_and_clamped():
    """Independent of any reference: the schedule is a closed form worth pinning down."""
    steps = 8
    grid = mx.arange(2 * steps, 0, -1, dtype=mx.float32) / (2 * steps)
    schedule = mx.clip(mx.log(grid / (1 - grid)), -20, 20)

    values = np.asarray(schedule, dtype=np.float64)
    assert values.shape == (2 * steps,), "two timesteps per solver step, for the midpoint method"
    assert values[0] == 20.0, "the first timestep saturates at the clamp"
    assert np.all(np.diff(values) < 0), "the schedule descends monotonically"
    # The grid lands exactly on p=0.5 halfway through, where logit is zero, and is antisymmetric
    # about it.
    assert values[steps - 1] > 0
    assert values[steps] == pytest.approx(0.0, abs=1e-6)
    assert values[steps + 1] < 0


def test_grouped_query_expansion_repeats_each_kv_head_consecutively():
    """Query head h must read kv head h // groups; interleaved repetition is what does that."""
    groups = CONFIG.kv_groups
    assert groups == 2

    keys = mx.arange(CONFIG.num_key_value_heads * 3).reshape(1, 1, CONFIG.num_key_value_heads, 3)
    expanded = mx.repeat(keys, groups, axis=2)

    assert expanded.shape[2] == CONFIG.num_attention_heads
    for head in range(CONFIG.num_attention_heads):
        source = head // groups
        assert bool(mx.all(expanded[0, 0, head] == keys[0, 0, source]))
