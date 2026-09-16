"""YuE2 checkpoint locations.

Weights are fetched from the YuE2 authors' own Hugging Face repositories, not from any third-party
repack, so this port depends on nothing but the original publisher.

They arrive in the published shape and are adapted in memory: the language model ships as a single
file carrying both the autoregressive and acoustic branches, and the VAE ships with weight-norm
parameterization and in fp32. Files are opened read-only and never rewritten.

The weights are CC BY-NC 4.0 — non-commercial. That is a different licence from this code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

LM_REPO = "m-a-p/YuE2-3B"
VAE_REPO = "m-a-p/YuE2-Vae"

LM_FILE = "model.safetensors"
TOKENIZER_FILE = "qwen.tiktoken"
VAE_FILE = "model.safetensors"

APPROX_GB = {"lm": 7.26, "vae": 0.53, "tokenizer": 0.01}


@dataclass
class CheckpointPaths:
    """One file holds both transformers; the VAE and tokenizer are separate."""

    language_model: Path
    vae: Path
    tokenizer: Path


def default_ckpt_dir() -> Path:
    return Path(os.environ.get("YUE2_MLX_CKPT_DIR", Path.home() / ".cache" / "yue2-mlx"))


def _layout(root: Path) -> dict[str, Path]:
    return {
        "lm": root / LM_REPO.replace("/", "--") / LM_FILE,
        "vae": root / VAE_REPO.replace("/", "--") / VAE_FILE,
        "tokenizer": root / LM_REPO.replace("/", "--") / TOKENIZER_FILE,
    }


def resolve(ckpt_dir: Path | None = None, download: bool = True) -> CheckpointPaths:
    root = Path(ckpt_dir) if ckpt_dir else default_ckpt_dir()
    paths = _layout(root)
    missing = [name for name, path in paths.items() if not path.exists()]

    if missing:
        if not download:
            raise FileNotFoundError(
                f"missing {len(missing)} checkpoint file(s) under {root}: "
                + ", ".join(str(paths[name].relative_to(root)) for name in missing)
            )
        _download(missing, root)

    return CheckpointPaths(
        language_model=paths["lm"], vae=paths["vae"], tokenizer=paths["tokenizer"]
    )


def _download(names: list[str], root: Path) -> None:
    from huggingface_hub import hf_hub_download

    jobs = {
        "lm": (LM_REPO, LM_FILE),
        "vae": (VAE_REPO, VAE_FILE),
        "tokenizer": (LM_REPO, TOKENIZER_FILE),
    }
    total = sum(APPROX_GB[name] for name in names)
    print(
        f"[yue2-mlx] fetching {len(names)} file(s) (~{total:.1f} GB) from Hugging Face into {root}"
    )

    for name in names:
        repo, filename = jobs[name]
        target = root / repo.replace("/", "--")
        target.mkdir(parents=True, exist_ok=True)
        hf_hub_download(repo_id=repo, filename=filename, local_dir=str(target))
