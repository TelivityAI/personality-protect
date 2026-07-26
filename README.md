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

Builds local SFT JSONL, then fine-tunes a **small adapter** on top of the quantized base:

```bash
personality-protect train --sft-only
personality-protect train --backend mock              # no multi-GB download
personality-protect train --backend mlx --max-steps 200   # MLX 4-bit base ~6 GB
personality-protect train --backend cuda --max-steps 200  # QLoRA (4-bit in VRAM)
```

Adapters land under `~/.personality-protect/profiles/<name>/adapters/` only.

**Honest hardware note:** train may briefly need more **RAM** than the ~5–7 GB on-disk footprint. The **download** story stays quantized — you should not need to keep a full BF16 copy as the happy path. CUDA’s first HF fetch can cache extra shards; day-to-day `filter` should use the GGUF under `models/`.

### 6. Filter

```bash
personality-protect filter --text "In today's fast-paced world we must leverage synergies."
personality-protect filter --backend llama --text "…"
personality-protect filter --file draft.txt --out voice.txt
personality-protect filter --backend mock --text "…"   # after mock train / demo
```

### 7. Local API stub (future browser extension)

Loopback only (`127.0.0.1`). Refuses non-local binds.

```bash
personality-protect api
# GET  http://127.0.0.1:8765/health
# POST http://127.0.0.1:8765/v1/filter  {"text":"…"}
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
| Profile cache + `models/*.gguf` | Personal adapters / weights |

Public git contains **code + synthetic demo fixtures only**.  
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
```

## License

Apache-2.0. See [LICENSE](LICENSE).
