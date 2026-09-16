# YuE2 on MLX

Song generation running natively on Apple Silicon: lyrics and a style description in, 48 kHz stereo
out, via [Apple MLX](https://github.com/ml-explore/mlx).

No CUDA, no PyTorch at runtime. The GPU work goes through Metal.

```bash
pip install -e .
yue2-mlx song --duration 10
```

The first run downloads ~7.4 GB of weights into `~/.cache/yue2-mlx` and writes a WAV to `outputs/`.

## What this is

A standalone MLX implementation of YuE2 song generation:

| Component | What it does |
| --- | --- |
| `models/ar.py` | Qwen3-architecture transformer, 28 layers: lyrics and style to audio codec tokens at 25/s |
| `models/acoustic.py` | Flow-matching transformer over 64-dim audio latents, midpoint solver |
| `models/vae.py` | Oobleck decoder, 1920x upsampling to 48 kHz stereo, SnakeBeta activations |
| `protocol.py` | Token layout, prompt construction, context chunking |
| `sampling.py` | Phase constraints, windowed repetition penalty, top-k/top-p |
| `pipeline.py` | The four stages, wired together |

MLX arrays are immutable and lazily evaluated — the graph, not the buffer, is the optimization
surface — so the transformers are written for that model, not translated from in-place CUDA code.

**Weights are never modified.** The published Hugging Face safetensors are opened read-only,
memory-mapped, and converted to MLX layout in memory. Nothing here rewrites, requantizes, or
re-uploads an upstream checkpoint file.

## How it works

Four stages:

1. **Score.** The AR model writes an ABC-notation score — chords and melody — as a plan.
2. **Codec tokens.** The same model generates audio tokens at 25 per second, guided against a prompt
   that keeps the instruction and score but drops the style and lyrics, so guidance pushes toward
   *this* song rather than away from music in general.
3. **Acoustic.** For each chunk, the AR model's per-layer keys and values become conditioning memory,
   and a flow-matching transformer integrates noise into 64-dim audio latents with a midpoint solver.
4. **Decode.** The Oobleck decoder upsamples 1920x into 48 kHz stereo.

Stage 3 is where the two transformers couple, and it is unusual: they do not communicate through
hidden states. The AR model's attention memory is *prepended* to the acoustic model's own keys and
values, layer by layer, so both models must be resident at the same time.

## Requirements

- Apple Silicon Mac (M1 or newer) on macOS 13.5+
- Python 3.10+
- ~8 GB of disk for weights, ~7.4 GB of memory resident during synthesis

MLX also publishes a Linux CPU backend (`pip install 'mlx[cpu]'`), which is how the test suite runs
in CI. It is far too slow for real generation.

## Usage

```bash
# built-in example lyrics, good for a first run
yue2-mlx song --duration 10

# your own words, inline
yue2-mlx song --duration 30 --style "synthwave, male vocal, analog bass, 110 BPM" --lyrics "[Verse]
Neon on the pavement, engine running low
[Chorus]
Take me where the night lights go"

# or from a file
yue2-mlx song --lyrics-file song.txt --style "dream pop, breathy female vocal, 100 BPM"

yue2-mlx doctor      # backend, and which weights are present
yue2-mlx download    # pre-fetch weights without generating
```

The style string carries genre, instrumentation, mood and tempo. The lyrics carry structure through
`[Verse]`, `[Chorus]` and similar tags.

`--duration` is an upper bound rather than a target. It converts to a token budget at 25 codec tokens
per second, and the model may emit its end token earlier once past the 200-token minimum — so
`--duration 30` yields somewhere between 8 and 30 seconds, wherever the model finds an ending. Below
8 seconds the minimum clamps to just under your budget, so short requests come out near full length.

Other flags: `--mode melody` writes a melody-only score, `--mode off` skips the score entirely;
`--save-score` keeps the generated score as `.abc`; `--score file.abc` performs a score you supply;
`--steps` sets acoustic solver steps (32 by default); `--guidance`, `--temperature`, `--top-k`,
`--top-p`, `--seed` control sampling; `--eval-every` is described below.

## If a run dies with "Impacting Interactivity"

```
[METAL] Command buffer execution failed: Impacting Interactivity
(0000000e:kIOGPUCommandBufferCallbackErrorImpactingInteractivity)
```

macOS killing a GPU command buffer for occupying the GPU long enough to stall the display. A
watchdog, not an out-of-memory condition, and load-dependent — the same command can succeed once and
abort the next time.

MLX builds a lazy graph and submits it as a single command buffer, so an unbroken 28-layer forward
over thousands of latent frames is one long submission. `--eval-every N` forces evaluation every N
transformer blocks, keeping each submission short; it defaults to 4 and does not change the output.
Lower it if you still hit the watchdog. Quitting other GPU-heavy apps helps too.

## Testing

```bash
pip install -e '.[dev]'
pytest tests/
```

No other repository, no network access, no downloads. Correctness is established by numerical
comparison rather than by listening, because a wrong RoPE convention or a mis-transposed kernel still
produces audio-shaped output.

| Stage | Checked against | What |
| --- | --- | --- |
| Autoregressive transformer | `transformers.Qwen3ForCausalLM` | Logits at three prompt lengths, incremental decoding against full prefill, and the per-layer keys and values `condition` produces against HuggingFace's `past_key_values` |
| Acoustic transformer | `tests/reference_torch.py` | Single denoiser evaluations at three timesteps, the midpoint integrator over full trajectories, RoPE tables, grouped-query head expansion, the timestep schedule |
| Oobleck decoder | `torch.nn.Conv1d` / `ConvTranspose1d`, and `tests/reference_torch.py` | Kernel layout at four strides and three dilations, decode at two lengths, tiled against full decode, output length, receptive field, the SnakeBeta formula |
| Checkpoint handling | `torch.nn.utils.weight_norm` | Branch splitting of the combined checkpoint, weight-norm folding |

Two of those references deserve a caveat, stated plainly. `tests/reference_torch.py` is a second
implementation of the same architecture written in channels-first PyTorch by the same author, so it
cannot catch a misunderstanding common to both. It is good at what porting actually gets wrong —
layout, RoPE convention, head expansion, operation order — and PyTorch's own convolutions and
`weight_norm` are genuine third-party ground truth for the layout conversions.

The gap is covered from two other directions. HuggingFace's Qwen3 is an implementation nobody here
wrote. And the real checkpoints answer the question directly:

```bash
python scripts/validate_yue2_weights.py --ckpt-dir ~/.cache/yue2-mlx
```

Given a prompt ending in `MUSIC_START`, it asks what the model predicts with **no constraints applied
at all**. On the published checkpoint, 10 of the top 10 predictions are audio codec tokens and the
probability mass on the codec block is 1.0000 — the model knows it should be making sound there.
Weights only agree with their own training if the port is right, which is what makes that check worth
more than any hand-written reference.

## Provenance

Architecture and token protocol follow the YuE2 authors' published model. Weights download from
[m-a-p/YuE2-3B](https://huggingface.co/m-a-p/YuE2-3B) and
[m-a-p/YuE2-Vae](https://huggingface.co/m-a-p/YuE2-Vae). The AR stage is Qwen3. The decoder derives
from stable-audio-tools (MIT) and SnakeBeta from NVIDIA BigVGAN (MIT). See `NOTICE`.

This repository is self-contained: clone, install, run. No other project is required to build, test,
or generate.

## Licensing

Read `NOTICE` before redistributing or before using generated audio commercially. Licences differ per
component, and one of them is non-commercial.

- **This project's code: Apache-2.0.**
- **YuE2 weights: CC BY-NC 4.0 — non-commercial.** That covers audio you generate with them. No
  weights are in this repository; they are downloaded unmodified from
  [m-a-p/YuE2-3B](https://huggingface.co/m-a-p/YuE2-3B) and
  [m-a-p/YuE2-Vae](https://huggingface.co/m-a-p/YuE2-Vae). Attribution requires naming YuE2, the
  model, and those repositories.
- **Oobleck decoder:** stable-audio-tools (Stability AI, MIT).
- **SnakeBeta:** NVIDIA BigVGAN (MIT).

Nothing here is legal advice.
