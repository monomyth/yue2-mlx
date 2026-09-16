"""YuE2's text tokenizer.

A tiktoken BPE built from the checkpoint's ``qwen.tiktoken`` (151,643 ordinary tokens) plus 208
special markers, two of which are the ABC score delimiters. Text is NFC-normalized first and encoded
without special-token handling, so a prompt can never inject a control marker.
"""

from __future__ import annotations

import base64
import unicodedata
from pathlib import Path

ORDINARY_TOKENS = 151643

PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*"
    r"|\s*[\r\n]+|\s+(?!\S)|\s+"
)


def _special_tokens() -> list[str]:
    specials = [
        "<|endoftext|>",
        "<|im_start|>",
        "<|im_end|>",
        "<R>",
        "<S>",
        "<X>",
        "<mask>",
        "<sep>",
    ]
    specials += [f"<extra_{i}>" for i in range(200)]
    specials[204:206] = ["<abc>", "</abc>"]
    return specials


class YuE2Tokenizer:
    def __init__(self, merge_file: str | Path):
        import tiktoken

        path = Path(merge_file)
        ranks = {
            base64.b64decode(token): int(rank)
            for token, rank in (line.split() for line in path.read_bytes().splitlines() if line)
        }
        if len(ranks) != ORDINARY_TOKENS:
            raise ValueError(
                f"expected the checkpoint's qwen.tiktoken with {ORDINARY_TOKENS} ordinary tokens, "
                f"found {len(ranks)}"
            )

        specials = _special_tokens()
        self._encoding = tiktoken.Encoding(
            "YuE2",
            pat_str=PATTERN,
            mergeable_ranks=ranks,
            special_tokens={name: index + len(ranks) for index, name in enumerate(specials)},
        )

    def encode(self, text: str) -> list[int]:
        return self._encoding.encode_ordinary(unicodedata.normalize("NFC", text))

    def decode(self, ids) -> str:
        vocab = self._encoding.n_vocab
        return self._encoding.decode(
            [int(token) for token in ids if 0 <= token < vocab], errors="replace"
        )
