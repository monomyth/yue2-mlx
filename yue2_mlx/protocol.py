"""YuE2's token protocol: special ids, prompt construction, sampling defaults, chunking.

Ported from upstream's ``models/TTS/yue2/protocol.py``. The numbers here are checkpoint-native --
they are baked into the weights, not conventions we get to choose.

The vocabulary is Qwen's 151,643 ordinary text tokens, then special markers, then a 32,768-entry
block of audio codec tokens. A song prompt is text (instruction, style tags, lyrics) followed by an
optional ABC-notation score, and the model answers in codec tokens at 25 per second.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

EOD = 151643
ABC_START, ABC_END = 151847, 151848
MUSIC_START, MUSIC_END = 151851, 151852
CODEC_OFFSET, CODEC_SIZE = 151853, 32768
VOCAB_SIZE, CONTEXT = 184704, 24576
FRAME_RATE = 25
SAMPLE_RATE = 48000

INSTRUCTIONS = {
    "off": "Generate music with codec tokens from the given conditions.",
    "melody": (
        "Generate a melody-only ABC transcription without chord symbols, then generate music with "
        "codec tokens from the given conditions."
    ),
    "full": (
        "Generate a chord-annotated ABC transcription, then generate music with codec tokens from "
        "the given conditions."
    ),
}


@dataclass(frozen=True)
class Sampling:
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 100
    repetition_penalty: float = 1.2
    penalty_window: int = 50
    min_tokens: int = 200
    max_tokens: int = 9000

    def __post_init__(self):
        if not 0 <= self.temperature <= 5 or not 0 <= self.top_p <= 1 or self.top_k < 0:
            raise ValueError("invalid sampling temperature/top_p/top_k")
        if self.repetition_penalty <= 0 or not 1 <= self.penalty_window <= 100:
            raise ValueError("invalid repetition penalty or window")
        if not 0 <= self.min_tokens <= self.max_tokens or self.max_tokens < 1:
            raise ValueError("require 0 <= min_tokens <= max_tokens")


# Upstream's defaults: the ABC stage is tighter and shorter than the audio stage.
ABC_SAMPLING = Sampling(
    temperature=0.7,
    top_p=0.9,
    top_k=30,
    repetition_penalty=1.005,
    penalty_window=100,
    min_tokens=32,
    max_tokens=4096,
)
SEMANTIC_SAMPLING = Sampling()


@dataclass(frozen=True)
class SongRequest:
    style: str
    lyrics: str
    cot: str = "full"
    abc: str | None = None
    cfg_scale: float | None = None
    seed: int = 831001

    def __post_init__(self):
        if self.cot not in INSTRUCTIONS:
            raise ValueError("cot must be one of off, melody, full")
        if self.cfg_scale is not None and (
            not math.isfinite(self.cfg_scale) or not 0 <= self.cfg_scale <= 20
        ):
            raise ValueError("cfg_scale must be finite and within [0, 20]")

    @property
    def guidance(self) -> float:
        if self.cfg_scale is not None:
            return self.cfg_scale
        return 1.01 if self.cot == "off" else 1.0

    def text(self) -> str:
        return f"{INSTRUCTIONS[self.cot]}\n[Tags]\n{self.style}\n[Lyrics]\n{self.lyrics}\n"


def token_prefix(request: SongRequest, tokenizer, abc_ids: list[int] | None = None) -> list[int]:
    """The conditional prompt: instruction, tags and lyrics, then the score slot."""
    base = [EOD] + tokenizer.encode(request.text())
    if request.cot == "off":
        return base + [ABC_START, ABC_END, MUSIC_START]
    if abc_ids is None:
        if request.abc is None:
            return base + [ABC_START]  # the model writes the score itself
        abc_ids = tokenizer.encode(request.abc)
    _check_abc_ids(abc_ids)
    return base + [ABC_START] + list(abc_ids) + [ABC_END, MUSIC_START]


def negative_prefix(request: SongRequest, tokenizer, abc_ids: list[int] | None = None) -> list[int]:
    """The unconditional prompt: the instruction and score survive, style and lyrics do not.

    Guidance therefore pushes toward the requested style and lyrics while leaving the musical
    structure the score already fixed alone.
    """
    base = [EOD] + tokenizer.encode(INSTRUCTIONS[request.cot])
    if request.cot == "off":
        return base + [MUSIC_START]
    if abc_ids is None:
        raise ValueError("guided generation needs the same ABC ids the positive branch used")
    _check_abc_ids(abc_ids)
    return base + [ABC_START] + list(abc_ids) + [ABC_END, MUSIC_START]


def _check_abc_ids(abc_ids) -> None:
    if any(not isinstance(token, int) or not 0 <= token < EOD for token in abc_ids):
        raise ValueError("ABC ids must stay inside the ordinary text vocabulary")


def chunk_ranges(frames: int, prefix_tokens: int, context: int = CONTEXT) -> list[tuple[int, int]]:
    """Split the audio timeline so each acoustic pass fits the context window.

    The acoustic model attends over the AR prefix plus the codec tokens plus one latent frame per
    codec token, hence halving the remaining budget.
    """
    size = min((context - prefix_tokens - 3) // 2, context)
    if frames < 1 or size < 1:
        raise ValueError("no room left for audio: shorten the lyrics or score")
    return [(start, min(start + size, frames)) for start in range(0, frames, size)]


def fit_to_context(sampling: Sampling, prefix_len: int, negative_len: int) -> Sampling:
    """Clamp the token budget to what the context window leaves after prompting."""
    available = CONTEXT - max(prefix_len, negative_len)
    if available < 5:
        raise ValueError("lyrics or score leave no room for audio; shorten them")
    if sampling.max_tokens <= available:
        return sampling
    return replace(
        sampling, max_tokens=available, min_tokens=min(sampling.min_tokens, available - 1)
    )
