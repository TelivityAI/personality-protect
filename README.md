# PersonalityProtect

Local-only CLI that turns **your** writing into a small voice adapter on **Qwen3.5-9B**, then rewrites frontier drafts through it before you publish.

**Download target: one quantized model, ~5–7 GB on disk** — not an 18–20 GB full-precision dump.

| Artifact | Role | Approx. size |
| --- | --- | --- |
| Qwen3.5-9B **Q4_K_M GGUF** | Default local runtime (`filter` via llama.cpp) | **~5.6 GB** |
| Qwen3.5-9B **MLX 4-bit** | Apple Silicon train + filter | **~6 GB** |
| Your LoRA / adapter | After `train` | small (MBs) |

`demo` / `mock` need **no** model download (pipeline smoke only).

**Your corpus, indexes, SFT JSONL, and adapters never leave this machine.**  
No Kaggle. No Colab. No cloud train. No uploading personal weights.

Built by [Telivity](https://telivity.com). Apache-2.0.

```text
sources ──► ingest ──► select ──► train (local LoRA) ──► filter ──► your voice
                 ▲                      │
                 └── demo (synthetic) ──┘
```

## Install

Requires Python 3.10+.

```bash
git clone https://github.com/TelivityAI/personality-protect.git
cd personality-protect
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Optional extras (pick what matches your machine):

```bash
# Hugging Face download helper (GGUF / MLX prefetch)
pip install -e ".[models]"

# llama.cpp GGUF runtime (recommended for filter)
pip install -e ".[gguf]"

# Apple Silicon MLX LoRA train + filter
pip install -e ".[mlx]"

# NVIDIA CUDA QLoRA
pip install -e ".[cuda]"
```

## Quick start (synthetic demo — no download)

```bash
personality-protect demo
personality-protect demo --json    # machine-readable; no logo
```

This runs: init → ingest synthetic docs → select → mock train → filter.

## Average Joe path (real local model)

1. Install + optional extras above.
2. Init a profile and **download one quantized artifact** (~5–7 GB once):

```bash
personality-protect init
personality-protect download                 # GGUF Q4_K_M ~5.6 GB (default)
# Apple Silicon trainers may also want:
personality-protect download --format mlx    # MLX 4-bit ~6 GB
```

3. Ingest → select → train → filter:

```bash
personality-protect ingest --linkedin ~/Downloads/LinkedInExport
personality-protect select
personality-protect train --backend mlx      # Mac: train on 4-bit MLX base
# or: personality-protect train --backend cuda
personality-protect filter --text "In today's fast-paced world we must leverage synergies."
```

Filter auto-prefers a local GGUF when present (`--backend llama`), then MLX 4-bit, then mock.

## Real workflow details

State lives in `~/.personality-protect/` (override with `--home` or `PERSONALITY_PROTECT_HOME`).

### 1. Init

```bash
personality-protect init
personality-protect init --profile work
```

### 2. Download (quantized only)

```bash
personality-protect download --format gguf   # → ~/.personality-protect/models/*.gguf
personality-protect download --format mlx    # → Hugging Face cache, ~6 GB
```

### 3. Ingest

LinkedIn export (folder or `.zip`) — reads Shares*/Comments*/Articles* in place; zips unpack only into the profile cache:

```bash
personality-protect ingest --linkedin ~/Downloads/LinkedInExport
personality-protect ingest --linkedin ~/Downloads/LinkedInExport.zip
```

Local emails / docs / notes (read in place — **no mandatory copy**):

```bash
personality-protect ingest --path ~/Documents/notes --source note
personality-protect ingest --path ~/Mail/archive --source email
```

### 4. Select

Defaults: **>50 words**, dates **through 2024** (overridable):

```bash
personality-protect select
personality-protect select --min-words 75 --through-year 2023
personality-protect select --include-undated
```

### 5. Train

Builds local SFT JSONL, then fine-tunes a **small adapter** on top of the quantized base.

Full-train defaults scale steps from your SFT example count (≈3 epochs, clamped). Use `--smoke` for CI/low-step. Mock is never a silent fallback — pass `--backend mock` or `--allow-mock` explicitly.

Corpus gates: **warn** below 50 selected pieces; **block** below 20 unless `--force`.

```bash
personality-protect train --sft-only
personality-protect train --backend mock --smoke --force   # CI / pipeline only
personality-protect train --backend mlx                    # auto steps from corpus size (fresh)
personality-protect train --backend mlx --proof            # bounded real train (receipts)
personality-protect train --backend mlx --chunk-steps 50 --memory-gb 16
personality-protect train --backend mlx --resume --max-steps 750 --chunk-steps 50 --memory-gb 16
personality-protect train --backend mlx --force-retrain --max-steps 750  # wipe adapters, start clean
personality-protect train --backend cuda --max-steps 200
```

Each successful MLX chunk writes `adapters.safetensors` (plus a numbered `NNNNNNN_adapters.safetensors`) and updates `train_chunks.json` (`completed_steps`, `total_steps`, `last_chunk`). If a run dies, `--resume` continues from the last good chunk instead of restarting from zero.
Adapters land under `~/.personality-protect/profiles/<name>/adapters/` only.

**Honest hardware note:** MLX train **and** filter/compare apply a **Metal wired-memory cap** (default ~40% of RAM, max 20 GB). Stock `mlx-lm` `generate`/`train` call `set_wired_limit(~40 GB)` on a 48 GB Mac and jetsam-kill Python ("quit unexpectedly"). Train additionally runs in subprocess chunks. Peak RAM is still higher than the ~5–7 GB on-disk footprint. CUDA’s first HF fetch can cache extra shards; day-to-day `filter` should use the GGUF under `models/` when available.

### 6. Filter

```bash
personality-protect filter --text "In today's fast-paced world we must leverage synergies."
personality-protect filter --backend llama --text "…"
personality-protect filter --file draft.txt --out voice.txt
personality-protect filter --backend mock --text "…"   # after mock train / demo
```

On Apple Silicon with an MLX adapter, `filter --backend auto` prefers MLX.

### 7. Eval / compare

Synthetic slop drafts ship under package `data/evals/`. Receipts write to the local profile `evals/` directory (gitignored — never commit).

```bash
personality-protect compare --synthetic slop_branding
personality-protect eval --synthetic slop_branding
personality-protect compare --text "It is important to note that we must leverage synergies."
```

Three-way compare: **raw** vs **prompt few-shot baseline** vs **LoRA/adapter filter**.

### 8. Local API stub (future browser extension)

Loopback only (`127.0.0.1`). Refuses non-local binds.

```bash
personality-protect api
# GET  http://127.0.0.1:8765/health
# POST http://127.0.0.1:8765/v1/filter  {"text":"…"}
```

## Launch / impress workflow

See [docs/LAUNCH.md](docs/LAUNCH.md) for the operator checklist (hardware, privacy, steps).

```bash
chmod +x scripts/beast_demo.sh
./scripts/beast_demo.sh --linkedin ~/Downloads/LinkedInExport
# synthetic smoke (no personal data, no multi-GB download):
./scripts/beast_demo.sh --skip-download
```

## CLI branding

Telivity terminal logo on interactive welcome / help surfaces. Never on `--json`.

```bash
personality-protect                 # banner + welcome
personality-protect logo
personality-protect --color never --logo mark status
```

Flags: `--color auto|always|never`, `--logo full|mark|off`, `--logo-mode auto|truecolor|color|plain|ascii`.  
Honors `NO_COLOR` and non-TTY (no ANSI).

## Privacy hard rules

| Stays on your machine | Never commit / never upload |
| --- | --- |
| Writing corpus & index | Real LinkedIn exports |
| SFT JSONL | Emails, notes, personal paths |
| LoRA / GGUF adapters | API keys, `.env`, tokens |
| Profile `evals/` receipts | Personal adapters / weights |
| Profile cache + `models/*.gguf` | Cloud train uploads |

Public git contains **code + synthetic demo/eval fixtures only**.  
Hugging Face is used only to **download public quantized base weights** (GGUF / MLX 4-bit).

## Hardware notes

| Backend | When | Disk download |
| --- | --- | --- |
| `mock` | Demo, CI, pipeline smoke | None |
| `llama` / GGUF | Default local filter | Q4_K_M **~5.6 GB** |
| `mlx` | Apple Silicon train/filter | MLX 4-bit **~6 GB** |
| `cuda` | NVIDIA QLoRA train | Prefer GGUF for filter; train uses 4-bit in VRAM |
| `cpu` | Impractical for 9B SFT | Prefer mock / GGUF filter on another machine |

## Develop

```bash
pip install -e ".[dev]"
pytest
ruff check src tests
```

### CI / branch protection

Workflow: `.github/workflows/ci.yml`. Required check names:

| Check name | What it runs |
| --- | --- |
| `lint` | `ruff check src tests` |
| `test (3.11)` | `pytest` on Python 3.11 |
| `test (3.12)` | `pytest` on Python 3.12 |
| `sanitize` | Block private-path / discussion leaks in tracked files |
| `cli-smoke` | `demo` + `compare` + mock `--smoke` train (no model download) |

## License

Apache-2.0. See [LICENSE](LICENSE).
