"""Token constraints and sampling for YuE2's autoregressive stage.

Ported from upstream's ``sampling.py``. The order of operations is load-bearing and matches upstream
exactly: mask the phase's illegal tokens, then decide whether the end token is allowed yet, then
apply the repetition penalty, then temperature, then top-k, then top-p.

The end token gets restored after masking rather than exempted from it, because in the audio phase
``MUSIC_END`` sits just below the codec block and would otherwise be masked out with the text
vocabulary.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx

from .protocol import ABC_END, CODEC_OFFSET, CODEC_SIZE, EOD, MUSIC_END, Sampling

NEG_INF = -float("inf")


@dataclass
class PhaseMask:
    """The static part of the constraint, built once per phase."""

    allowed: mx.array
    end_token: int

    @classmethod
    def build(cls, phase: str, vocab_size: int) -> PhaseMask:
        index = mx.arange(vocab_size)
        if phase == "abc":
            allowed = index < EOD
            end_token = ABC_END
        else:
            allowed = (index >= CODEC_OFFSET) & (index < CODEC_OFFSET + CODEC_SIZE)
            end_token = MUSIC_END
        return cls(allowed=allowed, end_token=end_token)


def constrain_and_sample(
    logits: mx.array,
    history: list[int],
    sampling: Sampling,
    mask: PhaseMask,
    key: mx.array,
    direct: bool = False,
) -> int:
    """Apply YuE2's constraints to one step's logits and draw a token."""
    scores = logits.astype(mx.float32).reshape(-1)
    end_score = scores[mask.end_token]

    scores = mx.where(mask.allowed, scores, NEG_INF)
    # Ending early is disallowed until the minimum length is reached.
    allow_end = len(history) >= sampling.min_tokens
    scores[mask.end_token] = end_score if allow_end else NEG_INF

    if history:
        recent = mx.array(history[-sampling.penalty_window :])
        frequencies = mx.zeros(scores.shape, dtype=mx.float32)
        frequencies = frequencies.at[recent].add(mx.ones(recent.shape, dtype=mx.float32))
        alpha = mx.power(sampling.repetition_penalty, frequencies)
        # Penalizing means pushing scores toward zero, which is a divide or a multiply by sign.
        scores = mx.where(scores < 0, scores * alpha, scores / alpha)

    if sampling.temperature == 0:
        return int(mx.argmax(scores).item())

    scores = scores / sampling.temperature

    if sampling.top_k:
        kth = mx.sort(scores)[-sampling.top_k]
        scores = mx.where(scores < kth, NEG_INF, scores)

    if sampling.top_p < 1:
        order = mx.argsort(-scores)
        ordered = scores[order]
        probabilities = mx.softmax(ordered)
        cumulative = mx.cumsum(probabilities) - probabilities
        drop = cumulative > sampling.top_p
        # Always keep the leading candidates so the distribution cannot become empty.
        drop[: 3 if direct else 1] = False
        ordered = mx.where(drop, NEG_INF, ordered)
        scores = ordered[mx.argsort(order)]  # undo the sort rather than scatter into it

    return int(mx.random.categorical(scores, key=key).item())
