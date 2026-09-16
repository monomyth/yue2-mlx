"""Check that the real YuE2 checkpoints load and generate, on any MLX backend.

The per-stage parity tests use small random weights, which proves the architecture but not the key
mapping. This script proves the rest: that the published checkpoints map onto these modules, that
the model is semantically sane with them, and that all three stages compose into audio.

The interesting check is the second one. Given a prompt that ends with ``MUSIC_START``, a correctly
ported model should want to emit audio codec tokens at that position — with no constraints applied
at all. A wrong key mapping, a broken RoPE or a mis-transposed projection still yields finite
logits, but it does not yield that.

Sized to finish on MLX's CPU backend, so the generated clip is a fraction of a second. On a Mac,
prefer a real run through ``yue2-mlx song``.

    python scripts/validate_yue2_weights.py --ckpt-dir ~/.cache/yue2-mlx
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from yue2_mlx import protocol  # noqa: E402
from yue2_mlx.checkpoints import resolve  # noqa: E402
from yue2_mlx.models.ar import ARConfig, YuE2AR  # noqa: E402
from yue2_mlx.pipeline import SongConfig, YuE2Pipeline, save_wav  # noqa: E402
from yue2_mlx.protocol import (  # noqa: E402
    CODEC_OFFSET,
    CODEC_SIZE,
    MUSIC_END,
    SongRequest,
    token_prefix,
)
from yue2_mlx.tokenizer import YuE2Tokenizer  # noqa: E402
from yue2_mlx.weights import (  # noqa: E402
    DTYPES,
    apply_weights,
    load_safetensors,
    split_language_model,
)

STYLE = "warm acoustic pop, female vocal, fingerpicked guitar, 90 BPM"
LYRICS = "[Verse]\nMorning light across the bay"


def check_tokenizer(paths) -> YuE2Tokenizer:
    print("\n[1/3] tokenizer")
    tokenizer = YuE2Tokenizer(paths.tokenizer)
    text = "Generate music with codec tokens from the given conditions."
    ids = tokenizer.encode(text)
    print(f"  {len(ids)} tokens, first few {ids[:6]}")
    assert tokenizer.decode(ids) == text, "tokenizer does not round-trip"
    return tokenizer


def check_ar_prefers_codec_tokens(paths, tokenizer, dtype) -> None:
    print("\n[2/3] AR stage: does it want to emit audio at MUSIC_START?")
    start = time.perf_counter()
    ar_weights, _ = split_language_model(load_safetensors(paths.language_model))
    model = apply_weights(YuE2AR(ARConfig()), ar_weights, dtype)
    del ar_weights
    print(f"  loaded {time.perf_counter() - start:.1f}s")

    request = SongRequest(style=STYLE, lyrics=LYRICS, cot="off")
    prefix = token_prefix(request, tokenizer)
    print(f"  prompt {len(prefix)} tokens, ending in ABC_START, ABC_END, MUSIC_START")

    start = time.perf_counter()
    logits = model(mx.array(prefix)[None], model.make_cache(), eval_every=4)
    values = np.asarray(logits.astype(mx.float32), dtype=np.float32).ravel()
    print(f"  prefill {time.perf_counter() - start:.1f}s")
    assert np.isfinite(values).all(), "AR produced non-finite logits"

    codec = slice(CODEC_OFFSET, CODEC_OFFSET + CODEC_SIZE)
    top = np.argsort(-values)[:10]
    in_codec = [bool(codec.start <= int(token) < codec.stop) for token in top]
    shifted = np.exp(values - values.max())
    mass = float(shifted[codec].sum() / shifted.sum())

    print(f"  top-10 predictions in the codec block: {sum(in_codec)}/10")
    print(f"  probability mass on codec tokens:      {mass:.4f}")
    first_miss = in_codec.index(False) if False in in_codec else None
    print(
        f"  first non-audio candidate:             rank {first_miss}"
        if first_miss is not None
        else "  first non-audio candidate:             none in the top 10"
    )

    assert all(in_codec), "unconstrained predictions left the codec block; check the key mapping"
    assert mass > 0.99, f"only {mass:.3f} of the mass is on audio tokens"
    assert int(top[0]) != MUSIC_END, "the model wants to end before emitting anything"


def check_end_to_end(paths, dtype, output: Path, frames: int, steps: int) -> None:
    print(f"\n[3/3] end to end: {frames} latent frames, {steps} acoustic steps")
    # A tiny clip needs to be allowed to stop early; the shipped minimum is 200 tokens.
    protocol.SEMANTIC_SAMPLING = replace(protocol.SEMANTIC_SAMPLING, min_tokens=1)
    import yue2_mlx.pipeline as pipeline_module

    pipeline_module.SEMANTIC_SAMPLING = protocol.SEMANTIC_SAMPLING

    config = SongConfig(
        lyrics=LYRICS,
        style=STYLE,
        duration_seconds=frames / protocol.FRAME_RATE,
        steps=steps,
        guidance=1.0,
        mode="off",
    )
    pipeline = YuE2Pipeline(paths, dtype=dtype)

    start = time.perf_counter()
    result = pipeline.generate(config)
    elapsed = time.perf_counter() - start

    audio = result.audio
    seconds = audio.shape[-1] / result.sample_rate
    breakdown = "  ".join(f"{name}={value:.1f}s" for name, value in result.timings.items())
    print(f"  generated in {elapsed:.1f}s   {breakdown}")
    print(f"  audio {audio.shape} = {seconds:.3f}s at {result.sample_rate} Hz")
    print(
        f"  stats min={audio.min():+.4f} max={audio.max():+.4f} "
        f"rms={np.sqrt((audio**2).mean()):.4f}"
    )

    assert np.isfinite(audio).all(), "decoded non-finite audio"
    assert audio.shape[0] == 2, "expected stereo"
    assert np.sqrt((audio**2).mean()) > 1e-4, "decoded silence"
    print(f"  wrote {save_wav(audio, output, result.sample_rate)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt-dir", default=None)
    parser.add_argument("--dtype", default="bfloat16", choices=sorted(DTYPES))
    parser.add_argument("--frames", type=int, default=12, help="latent frames, 25 per second")
    parser.add_argument("--steps", type=int, default=2, help="acoustic solver steps")
    parser.add_argument("-o", "--output", default="outputs/yue2_validation.wav")
    args = parser.parse_args()

    paths = resolve(args.ckpt_dir, download=False)
    dtype = DTYPES[args.dtype]
    print(f"backend {mx.default_device()}   dtype {args.dtype}")

    started = time.perf_counter()
    tokenizer = check_tokenizer(paths)
    check_ar_prefers_codec_tokens(paths, tokenizer, dtype)
    check_end_to_end(paths, dtype, Path(args.output), args.frames, args.steps)
    print(f"\nall checks passed in {time.perf_counter() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
