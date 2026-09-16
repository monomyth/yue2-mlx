"""End-to-end checks for the YuE2 pipeline that need no checkpoints and no upstream tree.

Covers the plumbing the per-stage parity tests do not: the token protocol, the sampling constraints,
chunking, and that the three stages compose into audio at miniature scale.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest
from helpers import relative_error

from yue2_mlx.models.acoustic import AcousticConfig, YuE2Acoustic
from yue2_mlx.models.ar import ARConfig, YuE2AR
from yue2_mlx.models.vae import VAEConfig, YuE2VAE
from yue2_mlx.protocol import (
    ABC_END,
    ABC_START,
    CODEC_OFFSET,
    CODEC_SIZE,
    CONTEXT,
    EOD,
    MUSIC_END,
    MUSIC_START,
    Sampling,
    SongRequest,
    chunk_ranges,
    fit_to_context,
    negative_prefix,
    token_prefix,
)
from yue2_mlx.sampling import PhaseMask, constrain_and_sample


class FakeTokenizer:
    """Deterministic stand-in: one token per character, inside the text vocabulary."""

    def encode(self, text: str) -> list[int]:
        return [ord(character) % 1000 for character in text]

    def decode(self, ids) -> str:
        return "".join(chr(int(token) % 128) for token in ids)


def test_prompt_carries_instruction_style_and_lyrics():
    request = SongRequest(style="dream pop", lyrics="[Verse]\nsunlight", cot="full")
    prefix = token_prefix(request, FakeTokenizer())

    assert prefix[0] == EOD
    assert prefix[-1] == ABC_START, "with no score supplied the model is left to write one"

    text = request.text()
    assert "[Tags]" in text and "dream pop" in text and "sunlight" in text


def test_score_supplied_closes_the_abc_block_and_opens_music():
    request = SongRequest(style="folk", lyrics="la la", cot="full")
    prefix = token_prefix(request, FakeTokenizer(), abc_ids=[1, 2, 3])

    assert prefix[-5:] == [ABC_START, 1, 2, 3, ABC_END, MUSIC_START][-5:]
    assert prefix[-1] == MUSIC_START


def test_negative_prompt_drops_style_and_lyrics_but_keeps_the_score():
    tokenizer = FakeTokenizer()
    request = SongRequest(style="industrial techno", lyrics="secret words", cot="full")
    abc_ids = [7, 8]

    positive = token_prefix(request, tokenizer, abc_ids)
    negative = negative_prefix(request, tokenizer, abc_ids)

    def contains(sequence, pattern):
        return any(
            sequence[i : i + len(pattern)] == pattern
            for i in range(len(sequence) - len(pattern) + 1)
        )

    style_tokens = tokenizer.encode(request.style)
    lyric_tokens = tokenizer.encode(request.lyrics)

    assert contains(positive, style_tokens) and contains(positive, lyric_tokens)
    assert not contains(negative, style_tokens), "style must not survive into the negative branch"
    assert not contains(negative, lyric_tokens), "lyrics must not survive either"
    assert contains(negative, abc_ids), "the score is shared by both branches"
    assert len(negative) < len(positive)


def test_mode_off_skips_the_score_entirely():
    request = SongRequest(style="ambient", lyrics="hush", cot="off")
    prefix = token_prefix(request, FakeTokenizer())
    assert prefix[-3:] == [ABC_START, ABC_END, MUSIC_START]
    assert request.guidance == pytest.approx(1.01), "mode off nudges guidance above 1"


def test_chunking_leaves_room_for_prefix_and_latents():
    ranges = chunk_ranges(5000, prefix_tokens=1000)
    assert ranges[0][0] == 0 and ranges[-1][1] == 5000
    assert all(end > start for start, end in ranges)
    # Each chunk needs prefix + codec + one latent frame per codec token inside the context.
    longest = max(end - start for start, end in ranges)
    assert 1000 + 2 * longest + 3 <= CONTEXT


def test_budget_is_clamped_to_the_context_window():
    sampling = fit_to_context(Sampling(max_tokens=CONTEXT), prefix_len=1000, negative_len=900)
    assert sampling.max_tokens == CONTEXT - 1000
    assert sampling.min_tokens < sampling.max_tokens

    with pytest.raises(ValueError, match="no room for audio"):
        fit_to_context(Sampling(), prefix_len=CONTEXT, negative_len=CONTEXT)


def _sample(logits, history, sampling, phase, seed=0):
    mask = PhaseMask.build(phase, logits.size)
    return constrain_and_sample(logits, history, sampling, mask, mx.random.key(seed))


def test_audio_phase_can_only_emit_codec_tokens():
    vocab = CODEC_OFFSET + CODEC_SIZE + 64
    logits = mx.ones((1, vocab)) * 5.0
    # min_tokens keeps the end token out of the running so this isolates the vocabulary constraint.
    sampling = Sampling(temperature=0, min_tokens=10)

    # Bias a text token far above everything else; the constraint must still refuse it.
    logits[0, 100] = 500.0
    token = _sample(logits, [], sampling, "semantic")
    assert CODEC_OFFSET <= token < CODEC_OFFSET + CODEC_SIZE


def test_end_token_is_withheld_until_the_minimum_length():
    vocab = CODEC_OFFSET + CODEC_SIZE + 64
    logits = mx.ones((1, vocab)) * -10.0
    logits[0, MUSIC_END] = 100.0
    sampling = Sampling(temperature=0, min_tokens=5)

    early = _sample(logits, [CODEC_OFFSET] * 2, sampling, "semantic")
    assert early != MUSIC_END, "ending before min_tokens must be impossible"

    allowed = _sample(logits, [CODEC_OFFSET] * 5, sampling, "semantic")
    assert allowed == MUSIC_END


def test_score_phase_can_only_emit_text_tokens():
    vocab = CODEC_OFFSET + CODEC_SIZE + 64
    logits = mx.ones((1, vocab)) * -10.0
    logits[0, CODEC_OFFSET + 5] = 100.0
    logits[0, 42] = 50.0

    token = _sample(logits, [], Sampling(temperature=0, min_tokens=0), "abc")
    assert token < EOD


def test_repetition_penalty_suppresses_recent_tokens():
    vocab = CODEC_OFFSET + CODEC_SIZE + 64
    logits = mx.ones((1, vocab)) * -20.0
    first, second = CODEC_OFFSET + 1, CODEC_OFFSET + 2
    logits[0, first] = 10.0
    logits[0, second] = 9.0
    sampling = Sampling(temperature=0, min_tokens=0, repetition_penalty=4.0, penalty_window=50)

    assert _sample(logits, [], sampling, "semantic") == first
    # After repeating it, the slightly weaker alternative should win.
    assert _sample(logits, [first] * 3, sampling, "semantic") == second


def test_stages_compose_into_stereo_audio():
    ar_config = ARConfig(
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        intermediate_size=64,
        vocab_size=256,
    )
    acoustic_config = AcousticConfig(
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        intermediate_size=64,
        latent_dim=8,
        max_latent_frames=64,
    )
    vae_config = VAEConfig(latent_dim=8, channels=4, c_mults=(1, 2), strides=(2, 2))

    ar, acoustic, vae = YuE2AR(ar_config), YuE2Acoustic(acoustic_config), YuE2VAE(vae_config)
    mx.eval(ar.parameters(), acoustic.parameters(), vae.parameters())

    token_ids = list(range(12))
    memory = ar.condition(token_ids)
    assert len(memory) == ar_config.num_hidden_layers

    frames = 6
    noise = mx.array(
        np.random.default_rng(0).standard_normal(
            (frames, acoustic_config.latent_dim), dtype=np.float32
        )
    )
    latents = acoustic.synthesize(noise, memory, len(token_ids), steps=2)
    assert latents.shape == (frames, acoustic_config.latent_dim)
    assert bool(mx.all(mx.isfinite(latents)))

    audio = vae.decode(latents.T[None])
    assert audio.shape[0] == 1 and audio.shape[1] == vae_config.out_channels
    assert audio.shape[2] == vae.output_length(frames)
    assert bool(mx.all(mx.isfinite(audio)))


def test_wav_writing_roundtrips():
    import wave
    from tempfile import TemporaryDirectory

    from yue2_mlx.pipeline import save_wav

    audio = np.stack([np.linspace(-1, 1, 480, dtype=np.float32)] * 2)
    with TemporaryDirectory() as directory:
        path = save_wav(audio, f"{directory}/out.wav", sample_rate=48000)
        with wave.open(str(path)) as handle:
            assert handle.getnchannels() == 2
            assert handle.getframerate() == 48000
            assert handle.getnframes() == 480
            assert handle.getsampwidth() == 2


def test_language_model_split_separates_the_two_branches():
    """The published checkpoint is one model; the branches share only the final norm."""
    from yue2_mlx.weights import split_language_model

    combined = {
        "model.embed_tokens.weight": mx.zeros((4, 2)),
        "lm_head.weight": mx.zeros((4, 2)),
        "model.norm.weight": mx.zeros((2,)),
        "model.layers.0.self_attn.q_proj.weight": mx.zeros((2, 2)),
        "model.layers.0.mlp.gate_proj.weight": mx.zeros((2, 2)),
        "model.layers.0.input_layernorm.weight": mx.zeros((2,)),
        "model.layers.0.post_attention_layernorm.weight": mx.zeros((2,)),
        "model.layers.0.nar_self_attn.q_proj.weight": mx.zeros((2, 2)),
        "model.layers.0.nar_mlp.gate_proj.weight": mx.zeros((2, 2)),
        "model.layers.0.nar_input_layernorm.weight": mx.zeros((2,)),
        "model.layers.0.nar_pre_mlp_layernorm.weight": mx.zeros((2,)),
        "llm2vae.weight": mx.zeros((2, 2)),
        "vae2llm.weight": mx.zeros((2, 2)),
        "time_embedder.mlp.0.weight": mx.zeros((2, 2)),
        "latent_pos_embed.pe": mx.zeros((4, 2)),
    }
    ar, acoustic = split_language_model(combined)

    assert "model.embed_tokens.weight" in ar and "lm_head.weight" in ar
    assert not any(".nar_" in key for key in ar)
    assert not any(key.startswith(("llm2vae", "vae2llm", "time_embedder")) for key in ar)

    assert "latent_pos_embed.pe" in acoustic and "llm2vae.weight" in acoustic
    assert not any(".self_attn." in key for key in acoustic)

    assert "model.norm.weight" in ar and "model.norm.weight" in acoustic, "final norm is shared"
    assert set(ar) | set(acoustic) == set(combined), "every tensor lands somewhere"


def test_weight_norm_folding_matches_pytorch():
    """Folding must reproduce what torch.nn.utils.weight_norm computes on each forward."""
    torch = pytest.importorskip("torch")

    from yue2_mlx.weights import fold_weight_norm

    conv = torch.nn.Conv1d(5, 3, kernel_size=7)
    normed = torch.nn.utils.weight_norm(conv)
    with torch.no_grad():
        normed.weight_g.copy_(torch.randn_like(normed.weight_g))
        normed.weight_v.copy_(torch.randn_like(normed.weight_v))

    published = {
        "layers.0.weight_g": mx.array(normed.weight_g.detach().numpy()),
        "layers.0.weight_v": mx.array(normed.weight_v.detach().numpy()),
        "layers.0.bias": mx.array(normed.bias.detach().numpy()),
    }
    folded = fold_weight_norm(published)

    assert set(folded) == {"layers.0.weight", "layers.0.bias"}, "the pair collapses into one weight"

    # weight_norm recomputes `weight` in a pre-forward hook, so `.weight` is stale until the module
    # runs. Compare against what it actually convolves with.
    with torch.no_grad():
        normed(torch.zeros(1, 5, 16))
    expected = normed.weight.detach().numpy()

    assert folded["layers.0.weight"].shape == expected.shape
    assert relative_error(folded["layers.0.weight"], expected) < 1e-5


def test_weight_norm_folding_rejects_an_orphan_direction():
    from yue2_mlx.weights import fold_weight_norm

    with pytest.raises(ValueError, match="no matching weight_g"):
        fold_weight_norm({"conv.weight_v": mx.zeros((2, 2, 3))})
