"""YuE2 song generation on MLX: lyrics and a style description in, 48 kHz stereo out.

Four stages, following upstream's ``models/TTS/yue2/pipeline.py``:

1. Optionally, the AR model writes an ABC-notation score — chords and melody — as a plan.
2. The AR model generates audio codec tokens at 25 per second, guided against a prompt that keeps
   the instruction and score but drops the style and lyrics.
3. For each chunk of codec tokens, the AR model's per-layer keys and values become the conditioning
   memory, and the acoustic model integrates noise into 64-dim audio latents.
4. The Oobleck decoder turns latents into stereo audio.

Memory strategy on Apple Silicon: both transformers are about 3B parameters, roughly 7.4 GB together
at bf16, and stage 3 needs them at the same time. On a 96 GB machine everything stays resident and
nothing is paged.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import mlx.core as mx
import numpy as np

from .checkpoints import CheckpointPaths
from .models.acoustic import AcousticConfig, YuE2Acoustic
from .models.ar import ARConfig, YuE2AR
from .models.vae import VAEConfig, YuE2VAE
from .protocol import (
    ABC_END,
    ABC_SAMPLING,
    FRAME_RATE,
    MUSIC_END,
    SAMPLE_RATE,
    SEMANTIC_SAMPLING,
    Sampling,
    SongRequest,
    chunk_ranges,
    fit_to_context,
    negative_prefix,
    token_prefix,
)
from .sampling import PhaseMask, constrain_and_sample
from .tokenizer import YuE2Tokenizer
from .weights import apply_weights, fold_weight_norm, load_safetensors, split_language_model


@dataclass
class SongConfig:
    lyrics: str
    style: str = "warm acoustic pop, female vocal, fingerpicked guitar, hopeful, 90 BPM"
    # An upper bound, not a target: the model may emit its end token earlier, though never before
    # the minimum length. Audio runs at 25 codec tokens per second.
    duration_seconds: float = 30.0
    steps: int = 32
    guidance: float = 1.0
    temperature: float = 1.0
    top_k: int = 100
    top_p: float = 0.95
    seed: int = 831001
    mode: str = "full"
    abc: str | None = None
    # Split the lazy graph into shorter Metal command buffers; see WanModel.__call__ for why.
    eval_every: int = 4

    def request(self) -> SongRequest:
        return SongRequest(
            style=self.style,
            lyrics=self.lyrics,
            cot=self.mode,
            abc=self.abc,
            cfg_scale=self.guidance,
            seed=self.seed,
        )

    def semantic_sampling(self) -> Sampling:
        maximum = max(1, int(self.duration_seconds * FRAME_RATE))
        return replace(
            SEMANTIC_SAMPLING,
            max_tokens=maximum,
            min_tokens=min(SEMANTIC_SAMPLING.min_tokens, maximum - 1),
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
        )


@dataclass
class SongResult:
    audio: np.ndarray  # float32 [channels, samples] in [-1, 1]
    sample_rate: int = SAMPLE_RATE
    abc: str = ""
    codec_tokens: int = 0
    timings: dict[str, float] = field(default_factory=dict)


class YuE2Pipeline:
    def __init__(self, paths: CheckpointPaths, dtype: mx.Dtype = mx.bfloat16, verbose: bool = True):
        self.paths = paths
        self.dtype = dtype
        self.verbose = verbose
        self.tokenizer = YuE2Tokenizer(paths.tokenizer)
        self._ar: YuE2AR | None = None
        self._acoustic: YuE2Acoustic | None = None
        self._vae: YuE2VAE | None = None
        self._lm_halves: tuple[dict, dict] | None = None

    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"[yue2-mlx] {message}", flush=True)

    def _language_model_halves(self) -> tuple[dict, dict]:
        """Read the combined checkpoint once and split it into its two branches."""
        if self._lm_halves is None:
            start = time.perf_counter()
            self._lm_halves = split_language_model(load_safetensors(self.paths.language_model))
            self._log(
                f"checkpoint read in {time.perf_counter() - start:.1f}s "
                f"({len(self._lm_halves[0])} autoregressive, {len(self._lm_halves[1])} acoustic)"
            )
        return self._lm_halves

    @property
    def ar(self) -> YuE2AR:
        if self._ar is None:
            start = time.perf_counter()
            self._ar = apply_weights(
                YuE2AR(ARConfig()), self._language_model_halves()[0], self.dtype
            )
            self._log(f"autoregressive stage ready in {time.perf_counter() - start:.1f}s")
        return self._ar

    @property
    def acoustic(self) -> YuE2Acoustic:
        if self._acoustic is None:
            start = time.perf_counter()
            self._acoustic = apply_weights(
                YuE2Acoustic(AcousticConfig()),
                self._language_model_halves()[1],
                self.dtype,
                # The sinusoidal position table and the timestep MLP feed every layer's input.
                keep_float32=("latent_pos_embed", "time_embedder"),
            )
            self._log(f"acoustic stage ready in {time.perf_counter() - start:.1f}s")
        return self._acoustic

    @property
    def vae(self) -> YuE2VAE:
        if self._vae is None:
            start = time.perf_counter()
            # Published in fp32 with weight-norm parameterization, and carrying encoder tensors
            # this port never uses. Folding collapses the parameterization into plain kernels.
            weights = fold_weight_norm(load_safetensors(self.paths.vae))
            self._vae = apply_weights(YuE2VAE(VAEConfig()), weights, mx.float32)
            self._log(f"VAE ready in {time.perf_counter() - start:.1f}s")
        return self._vae

    def generate_tokens(
        self,
        prefix: list[int],
        sampling: Sampling,
        phase: str,
        seed: int,
        negative: list[int] | None = None,
        guidance: float = 1.0,
        direct: bool = False,
        report=None,
        eval_every: int = 4,
    ) -> list[int]:
        """Sample from the AR model until its end token or the budget runs out.

        With guidance above 1 a second sequence runs in lockstep on the negative prompt, and the two
        logit vectors are combined before constraints are applied. Both sequences are then fed the
        same sampled token, so they stay aligned.
        """
        model = self.ar
        end_token = ABC_END if phase == "abc" else MUSIC_END
        mask = PhaseMask.build(phase, model.config.vocab_size)
        guided = guidance != 1.0 and negative is not None

        cache = model.make_cache()
        logits = model(mx.array(prefix)[None], cache, eval_every)
        negative_cache = None
        if guided:
            negative_cache = model.make_cache()
            negative_logits = model(mx.array(negative)[None], negative_cache, eval_every)

        tokens: list[int] = []
        key = mx.random.key(seed)
        started = time.perf_counter()
        for step in range(sampling.max_tokens):
            combined = logits
            if guided:
                combined = negative_logits + guidance * (logits - negative_logits)

            key, subkey = mx.random.split(key)
            token = constrain_and_sample(combined, tokens, sampling, mask, subkey, direct)
            if token == end_token:
                self._log(
                    f"{phase} ended after {len(tokens)} tokens "
                    f"({time.perf_counter() - started:.1f}s)"
                )
                return tokens

            tokens.append(token)
            if report is not None and step % 25 == 0:
                report(phase, len(tokens), sampling.max_tokens)

            step_input = mx.array([[token]])
            logits = model(step_input, cache)
            if guided:
                negative_logits = model(step_input, negative_cache)

        self._log(f"{phase} hit its {sampling.max_tokens}-token budget; output may be cut short")
        return tokens

    def generate(self, config: SongConfig, report=None) -> SongResult:
        timings: dict[str, float] = {}
        request = config.request()

        abc, abc_ids = config.abc or "", []
        if request.cot != "off":
            if config.abc:
                abc_ids = self.tokenizer.encode(config.abc)
            else:
                start = time.perf_counter()
                abc_ids = self.generate_tokens(
                    token_prefix(request, self.tokenizer),
                    ABC_SAMPLING,
                    "abc",
                    config.seed,
                    report=report,
                    eval_every=config.eval_every,
                )
                abc = self.tokenizer.decode(abc_ids)
                timings["score"] = time.perf_counter() - start

        prefix = token_prefix(request, self.tokenizer, abc_ids)
        guidance = request.guidance
        negative = negative_prefix(request, self.tokenizer, abc_ids) if guidance != 1.0 else None
        sampling = fit_to_context(config.semantic_sampling(), len(prefix), len(negative or prefix))
        self._log(
            f"prompt {len(prefix)} tokens, up to {sampling.max_tokens} audio tokens "
            f"({sampling.max_tokens / FRAME_RATE:.1f}s), guidance {guidance}"
        )

        start = time.perf_counter()
        codec = self.generate_tokens(
            prefix,
            sampling,
            "semantic",
            config.seed,
            negative,
            guidance,
            direct=request.cot == "off",
            report=report,
            eval_every=config.eval_every,
        )
        timings["codec"] = time.perf_counter() - start
        if not codec:
            raise RuntimeError("the model produced no audio tokens")

        start = time.perf_counter()
        latents = self._synthesize(prefix, codec, config, report)
        timings["acoustic"] = time.perf_counter() - start

        start = time.perf_counter()
        audio = self._decode(latents, report)
        timings["decode"] = time.perf_counter() - start

        return SongResult(audio=audio, abc=abc, codec_tokens=len(codec), timings=timings)

    def _synthesize(self, prefix, codec, config: SongConfig, report=None) -> mx.array:
        chunks = chunk_ranges(len(codec), len(prefix))
        noise = np.random.default_rng(config.seed).standard_normal(
            (len(codec), self.acoustic.config.latent_dim), dtype=np.float32
        )
        self._log(
            f"synthesizing {len(codec)} latent frames in {len(chunks)} chunk(s), "
            f"{config.steps} steps each"
        )

        parts = []
        for index, (start, end) in enumerate(chunks):
            token_ids = prefix + codec[start:end] + [MUSIC_END]
            memory = self.ar.condition(token_ids)
            parts.append(
                self.acoustic.synthesize(
                    mx.array(noise[start:end]),
                    memory,
                    len(token_ids),
                    config.steps,
                    eval_every=config.eval_every,
                    report=(
                        lambda step, total, i=index: report(
                            "acoustic", i * config.steps + step, len(chunks) * config.steps
                        )
                    )
                    if report
                    else None,
                )
            )
            del memory
            mx.eval(parts[-1])
        return mx.concatenate(parts) if len(parts) > 1 else parts[0]

    def _decode(self, latents: mx.array, report=None) -> np.ndarray:
        latent = latents.T[None]
        core = 1024
        halo = max(self.vae.required_halo(core), 16)
        audio = self.vae.decode_tiled(
            latent,
            core_frames=core,
            halo_frames=halo,
            report=(lambda done, total: report("decode", done, total)) if report else None,
        )
        mx.eval(audio)
        clipped = mx.clip(audio[0].astype(mx.float32), -1.0, 1.0)
        return np.asarray(clipped, dtype=np.float32)


def save_wav(audio: np.ndarray, path: str | Path, sample_rate: int = SAMPLE_RATE) -> Path:
    """Write float audio ``[channels, samples]`` as 16-bit PCM."""
    import wave

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = np.clip(audio.T, -1.0, 1.0)
    pcm = (samples * 32767.0).round().astype("<i2")

    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(samples.shape[1])
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())
    return path
