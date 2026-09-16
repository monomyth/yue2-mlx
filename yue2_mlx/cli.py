"""Command line entry point: ``yue2-mlx song --lyrics-file song.txt``."""

from __future__ import annotations

import argparse
import platform
import sys
import time
from pathlib import Path

import mlx.core as mx

from .checkpoints import APPROX_GB, LM_REPO, VAE_REPO, default_ckpt_dir, resolve
from .pipeline import SongConfig, YuE2Pipeline, save_wav
from .weights import DTYPES

EXAMPLE_LYRICS = """[Verse]
Morning light across the bay
We watch the shadows drift away
[Chorus]
Stay with me until the dawn
Let our little song go on"""


def _progress_printer():
    state = {"stage": None, "start": 0.0}

    def report(stage: str, current: int, total: int) -> None:
        if state["stage"] != stage:
            state.update(stage=stage, start=time.perf_counter())
            print()
        elapsed = time.perf_counter() - state["start"]
        rate = f"{elapsed / current:.2f}s/it" if current else "-"
        print(
            f"\r  {stage:9s} {current:5d}/{total:5d}  {elapsed:6.1f}s  {rate}   ",
            end="",
            flush=True,
        )

    return report


def _read_text(path: str, what: str, encoding: str = "utf-8") -> str:
    try:
        return Path(path).read_text(encoding=encoding)
    except FileNotFoundError:
        raise SystemExit(f"no {what} file at {path!r}") from None
    except OSError as error:
        raise SystemExit(f"could not read the {what} file {path!r}: {error}") from None


def cmd_song(args: argparse.Namespace) -> int:
    lyrics = _read_text(args.lyrics_file, "lyrics") if args.lyrics_file else args.lyrics
    if not lyrics or not lyrics.strip():
        raise SystemExit("provide lyrics with --lyrics or --lyrics-file")

    config = SongConfig(
        lyrics=lyrics,
        style=args.style,
        duration_seconds=args.duration,
        steps=args.steps,
        guidance=args.guidance,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        seed=args.seed,
        mode=args.mode,
        abc=_read_text(args.score, "score", encoding="utf-8-sig") if args.score else None,
    )

    try:
        paths = resolve(args.ckpt_dir, download=not args.no_download)
    except FileNotFoundError as error:
        raise SystemExit(f"{error}\nRun 'yue2-mlx download' to fetch them.") from None
    pipeline = YuE2Pipeline(paths, dtype=DTYPES[args.dtype])
    result = pipeline.generate(config, report=_progress_printer())

    output = Path(args.output) if args.output else Path("outputs") / f"{int(time.time())}.wav"
    save_wav(result.audio, output, result.sample_rate)

    seconds = result.audio.shape[-1] / result.sample_rate
    total = sum(result.timings.values())
    breakdown = "  ".join(f"{name}={value:.1f}s" for name, value in result.timings.items())
    print(f"\nwrote {output}  ({seconds:.1f}s of audio, {result.codec_tokens} codec tokens)")
    print(f"total {total:.1f}s   {breakdown}")
    if result.abc and args.save_score:
        score_path = output.with_suffix(".abc")
        score_path.write_text(result.abc, encoding="utf-8")
        print(f"score  {score_path}")
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    print(f"platform      {platform.platform()}  python {platform.python_version()}")
    print(f"mlx           {mx.__version__}   default device {mx.default_device()}")
    if "gpu" not in str(mx.default_device()).lower():
        print("              WARNING: not on Metal; generation will be impractically slow")

    root = Path(args.ckpt_dir) if args.ckpt_dir else default_ckpt_dir()
    print(f"\ncheckpoints   {root}")
    print(f"  sources     {LM_REPO}, {VAE_REPO}")
    try:
        paths = resolve(root, download=False)
        present = {
            "language model": paths.language_model,
            "vae": paths.vae,
            "tokenizer": paths.tokenizer,
        }
        for label, path in present.items():
            print(f"  {label:14s} present  {path.stat().st_size / 1e9:5.2f} GB")
    except FileNotFoundError as error:
        print(f"  {error}")
        print("  run 'yue2-mlx download' to fetch them")

    resident = APPROX_GB["lm"] + APPROX_GB["vae"]
    print(
        f"\nBoth transformers ship in one file and are needed together during synthesis: "
        f"about {resident:.1f} GB resident at bf16."
    )
    try:
        import tiktoken  # noqa: F401
    except ImportError:
        print("tiktoken is not installed; run: pip install -e '.[yue2]'")
    return 0


def cmd_download(args: argparse.Namespace) -> int:
    paths = resolve(args.ckpt_dir, download=True)
    print(f"language model  {paths.language_model}")
    print(f"vae             {paths.vae}")
    print(f"tokenizer       {paths.tokenizer}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="yue2-mlx", description="YuE2 song generation on Apple Silicon via MLX"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    song = sub.add_parser("song", help="generate a song from lyrics and a style description")
    song.add_argument("--lyrics", default=EXAMPLE_LYRICS, help="lyrics, with [Verse]/[Chorus] tags")
    song.add_argument("--lyrics-file", default=None, help="read lyrics from a file instead")
    song.add_argument(
        "--style",
        default="warm acoustic pop, female vocal, fingerpicked guitar, hopeful, 90 BPM",
        help="genre, instrumentation, mood, tempo",
    )
    song.add_argument(
        "--duration",
        type=float,
        default=30.0,
        help="upper bound in seconds (25 codec tokens per second). The model may end earlier once "
        "it passes the minimum length, so treat this as a ceiling, not a target",
    )
    song.add_argument("--steps", type=int, default=32, help="acoustic solver steps")
    song.add_argument("--guidance", type=float, default=1.0)
    song.add_argument("--temperature", type=float, default=1.0)
    song.add_argument("--top-k", type=int, default=100)
    song.add_argument("--top-p", type=float, default=0.95)
    song.add_argument("--seed", type=int, default=831001)
    song.add_argument(
        "--mode",
        default="full",
        choices=["full", "melody", "off"],
        help="full writes a chord-annotated score first, melody a melody-only one, off skips it",
    )
    song.add_argument("--score", default=None, help="use an existing ABC score file")
    song.add_argument("--save-score", action="store_true", help="write the generated score as .abc")
    song.add_argument("-o", "--output", default=None, help="output .wav path")
    song.add_argument("--dtype", default="bfloat16", choices=sorted(DTYPES))
    song.add_argument("--ckpt-dir", default=None, help="defaults to ~/.cache/yue2-mlx")
    song.add_argument("--no-download", action="store_true")
    song.set_defaults(func=cmd_song)

    doctor = sub.add_parser("doctor", help="check backend and checkpoint state")
    doctor.add_argument("--ckpt-dir", default=None)
    doctor.set_defaults(func=cmd_doctor)

    download = sub.add_parser("download", help="fetch weights without generating")
    download.add_argument("--ckpt-dir", default=None)
    download.set_defaults(func=cmd_download)

    args = parser.parse_args(argv)
    if hasattr(args, "seed"):
        mx.random.seed(args.seed)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
