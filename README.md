# PersonalityProtect

**Stop sounding like everyone else's AI.**

Draft LinkedIn posts and articles in *your* voice on your machine. Index your writing, measure cadence, retrieve short rhythm references, and generate with a local quantized model (~5–7 GB). Optional writer LoRA deepens voice after it beats RAG-alone on holdouts. Your corpus never leaves this Mac.

Built by [Telivity](https://telivity.com). Apache-2.0.

```text
your writing ──► ingest ──► index-voice ──► build-style-profile ──► write
                                      └── optional: train --writer ──► write --adapter
```

<p align="center">
  <img src="docs/images/cli-shipped.png" alt="personality-protect write path" width="760" />
</p>

---

## Why

The feed is drowning in AI writing that all sounds the same — *“In today’s fast-paced world, we must leverage synergies…”* A pasted style guide helps a little. It still isn’t **you**.

Cloud fine-tunes are a non-option for personal writing. Notes, emails, and posts are biometric-adjacent. Shipping them to a rented GPU so someone else’s stack can imitate you is a strange bargain.

PersonalityProtect keeps the corpus on disk, measures your cadence, retrieves short rhythm references from your own writing, and drafts locally with MLX on Apple Silicon. Treat outputs as drafts you still own.

---

## How you get voice

1. **Ingest** your LinkedIn export and/or local notes (stays on disk).
2. **`index-voice`** builds a local retrieval index.
3. **`build-style-profile`** measures cadence (sentence length, short lines, post length band, banned filler).
4. **`write --topic --points`** drafts from the brief only; retrieved posts are rhythm reference.

Optional:

- **`train --writer`** trains a brief→post LoRA from your posts (holdouts excluded).
- **`write --adapter`** loads that LoRA when `adapters.safetensors` is present.
- **`write --channel article`** runs outline → sections → stitch (needs enough `linkedin_article` pieces).

---

## Quick start

Requires Python 3.10+ and an Apple Silicon Mac for the happy path.

```bash
git clone https://github.com/TelivityAI/personality-protect.git
cd personality-protect
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,mlx]"

personality-protect init
personality-protect download --format mlx    # ~6 GB, once

personality-protect ingest --linkedin ~/path/to/linkedin-export
personality-protect ingest --path ~/path/to/notes --source note

personality-protect index-voice
personality-protect build-style-profile

# LinkedIn post (targets your long-post band, up to ~550 words / ~3k chars)
personality-protect write \
  --topic "Contoso Ledger exceptions" \
  --points "- Name one owner\n- Keep the rollout boring"

# Article (outline → sections → stitch)
personality-protect write \
  --channel article \
  --topic "Contoso Ledger guide" \
  --points "- Name one owner\n- Cut exceptions\n- Keep rollbacks boring"

personality-protect status
```

Optional writer LoRA:

```bash
personality-protect build-writer-sft
personality-protect train --writer --backend mlx
personality-protect write --adapter --topic "…" --points "…"
```

Optional extras:

```bash
pip install -e ".[models]"   # Hugging Face download helper
pip install -e ".[gguf]"     # llama.cpp GGUF (optional)
pip install -e ".[cuda]"     # NVIDIA path (optional)
```

---

## Screenshots

Public docs use **synthetic Contoso / synergy-slop text only**. No personal corpus.

| Write path | Status | Mark |
| --- | --- | --- |
| <img src="docs/images/cli-shipped.png" alt="write path" width="360" /> | <img src="docs/images/cli-status.png" alt="status" width="280" /> | <img src="docs/images/cli-logo.png" alt="Telivity CLI logo" width="280" /> |

```bash
personality-protect index-voice
personality-protect build-style-profile
personality-protect write --topic "Contoso pricing" --points "Name one owner."
personality-protect status
```

Optional smoke tour (no model download; synthetic only):

```bash
personality-protect demo
```

---

## Privacy

**Hard rule: personal writing stays on your machine.**

| Stays in `~/.personality-protect/` | Never commit / never upload |
| --- | --- |
| Profiles, corpus index, voice index, style profile | Real LinkedIn / email / note exports |
| Writer LoRA adapters under `adapters/` | Profile URLs, personal paths |
| Downloaded weights under `models/` / HF cache | API keys, `.env`, tokens |
| Local eval receipts | Cloud train uploads |

Override the home directory with `--home` or `PERSONALITY_PROTECT_HOME`.

`.gitignore` blocks profiles, adapters, SFT, exports, weights, and secrets. Public git ships **code + synthetic demo/eval fixtures only**. Hugging Face is used only to **download public quantized base weights**.

This README uses synthetic examples only (e.g. Contoso, “leverage synergies”). Do not paste real posts, profile URLs, or personal before/after samples into docs or PRs.

---

## Hardware

**Happy path: Apple Silicon Mac** (MLX).

| What | Size / note |
| --- | --- |
| MLX 4-bit base (default write) | ~6 GB download |
| GGUF Q4_K_M (optional) | ~5.6 GB download |
| Writer LoRA | Small (MBs) under the profile |
| Peak RAM while writing / training | Memory-capped; 16 GB+ recommended |

MLX applies a wired-memory cap so Metal does not jetsam-kill Python on mid-size Macs.

---

## Workflow

State lives in `~/.personality-protect/profiles/<name>/`.

### Init / download / ingest

```bash
personality-protect init
personality-protect download --format mlx
personality-protect ingest --linkedin ~/path/to/linkedin-export.zip
personality-protect ingest --path ~/path/to/notes --source note
```

### Index and style

```bash
personality-protect index-voice
personality-protect build-style-profile
```

Post length targets come from `linkedin_post` pieces (p75/p90), clamped to the LinkedIn ~3000-character band (~550 words).

### Write

```bash
personality-protect write --topic "…" --points "…"
personality-protect write --channel article --topic "…" --points "…"
personality-protect write --adapter --topic "…" --points "…"
personality-protect write --topic "…" --points "…" --json
```

`--topic` and `--points` are the only content the draft may use. Article channel requires at least five `linkedin_article` pieces in the corpus.

### Writer LoRA

```bash
personality-protect build-writer-sft
personality-protect train --writer --backend mlx
```

Keep the adapter only if it improves holdout drafts vs RAG-alone (`eval-write-holdout`). Delete it if it does not.

### Status / API

```bash
personality-protect status
personality-protect api   # loopback 127.0.0.1 only
```

---

## CLI reference

Global flags (most commands): `--profile`, `--home`, `--json`, plus branding `--color`, `--logo`, `--logo-mode`.

| Command | Purpose |
| --- | --- |
| `init` | Create profile under `~/.personality-protect/` |
| `download` | Prefetch quantized MLX or GGUF base |
| `ingest` | Index LinkedIn export and/or local paths |
| `index-voice` | Build local voice retrieval index |
| `build-style-profile` | Build cadence / length / banned-filler style card |
| `build-writer-sft` | Build brief→post SFT for writer LoRA |
| `write` | Draft a post or article (`--channel`, optional `--adapter`) |
| `train` | Local LoRA train (`--writer` for writer LoRA) |
| `eval-write-holdout` | Score write quality on held-out pieces (local receipt) |
| `status` | Show profile state |
| `demo` | Optional synthetic smoke tour (no download) |
| `api` | Loopback HTTP stub |
| `logo` | Print Telivity CLI mark |

### `write` flags

| Flag | Meaning |
| --- | --- |
| `--topic` | What the piece is about |
| `--points` | Facts/claims the draft may use |
| `--channel post\|article` | Post (default) or article outline→sections→stitch |
| `--k` | Rhythm exemplars to retrieve |
| `--adapter` / `--no-adapter` | Load writer LoRA when present (default off) |
| `--json` | Machine-readable receipt |

### `eval-write-holdout` flags

| Flag | Meaning |
| --- | --- |
| `--holdout-id` | Piece id never indexed (repeatable) |
| `--save-raw` | Local prompts/drafts under the profile (never commit) |
| `--out PATH` | Contoso-safe aggregate receipt JSON |

---

## Advanced (optional)

Other experiment commands (`select`, `filter`, `compare`, translator pairs) remain in the CLI. The shipped path above does not require them.

---

## Launch script

Operator checklist: [docs/LAUNCH.md](docs/LAUNCH.md).

```bash
chmod +x scripts/beast_demo.sh
./scripts/beast_demo.sh --linkedin ~/path/to/linkedin-export
./scripts/beast_demo.sh --skip-download   # synthetic smoke
```

---

## Develop

```bash
pip install -e ".[dev]"
pytest
ruff check src tests
```

CI (`.github/workflows/ci.yml`) required checks: `lint`, `test (3.11)`, `test (3.12)`, `sanitize`, `cli-smoke`.

---

## License

Apache-2.0. See [LICENSE](LICENSE).
