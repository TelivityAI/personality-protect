# PersonalityProtect

**Stop sounding like everyone else's AI.**

Draft LinkedIn posts in *your* voice on your machine. Index your writing, retrieve a few of your own posts for rhythm, and generate from a quantized local model (~5–7 GB). Your corpus never leaves this Mac.

Built by [Telivity](https://telivity.com). Apache-2.0.

```text
your writing ──► ingest ──► index-voice ──► build-style-profile ──► write
```

<p align="center">
  <img src="docs/images/cli-shipped.png" alt="personality-protect write path" width="760" />
</p>

---

## Why

The feed is drowning in AI writing that all sounds the same — *“In today’s fast-paced world, we must leverage synergies…”* A pasted style guide helps a little. It still isn’t **you**.

Cloud fine-tunes are a non-option for personal writing. Notes, emails, and posts are biometric-adjacent. Shipping them to a rented GPU so someone else’s stack can imitate you is a strange bargain.

PersonalityProtect keeps the corpus on disk, measures your cadence, retrieves short rhythm references from your own posts, and drafts locally with MLX on Apple Silicon. Treat outputs as drafts you still own.

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

# Your exports stay on this machine
personality-protect ingest --linkedin ~/path/to/linkedin-export
personality-protect ingest --path ~/path/to/notes --source note

personality-protect index-voice
personality-protect build-style-profile
personality-protect write \
  --topic "Contoso Ledger exceptions" \
  --points "Name one owner. Keep the rollout boring."

personality-protect status
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
| Downloaded weights under `models/` / HF cache | Profile URLs, personal paths |
| Local eval receipts | API keys, `.env`, tokens |

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
| Peak RAM while writing | Memory-capped; typically comfortable on 16 GB+ |

MLX applies a wired-memory cap so Metal does not jetsam-kill Python on mid-size Macs.

---

## Workflow

State lives in `~/.personality-protect/profiles/<name>/`.

### Init

```bash
personality-protect init
personality-protect init --profile work
```

### Download

```bash
personality-protect download --format mlx    # → Hugging Face cache, ~6 GB
personality-protect download --format gguf   # → ~/.personality-protect/models/*.gguf
```

### Ingest

LinkedIn export (folder or `.zip`) — CSV/HTML read in place; zips unpack only into the profile cache:

```bash
personality-protect ingest --linkedin ~/path/to/linkedin-export
personality-protect ingest --linkedin ~/path/to/linkedin-export.zip
```

Local docs / notes / mail archives (read in place — **no mandatory copy**):

```bash
personality-protect ingest --path ~/path/to/notes --source note
personality-protect ingest --path ~/path/to/mail-archive --source email
```

### Index and style

```bash
personality-protect index-voice
personality-protect build-style-profile
```

`index-voice` builds a local retrieval index from your corpus. `build-style-profile` measures cadence targets (sentence length, short lines, typical post length, banned filler) used by `write`.

### Write

```bash
personality-protect write \
  --topic "Contoso Ledger exceptions" \
  --points "Name one owner. Keep the rollout boring."
personality-protect write --topic "…" --points "…" --json
```

`--topic` and `--points` are the only content the draft may use. Retrieved posts are rhythm reference only — facts come from the brief.

### Status

```bash
personality-protect status
```

### Local API stub

Loopback only (`127.0.0.1`). Refuses non-local binds. Future browser-extension hook.

```bash
personality-protect api
# GET  http://127.0.0.1:8765/health
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
| `build-style-profile` | Build cadence / banned-filler style card |
| `write` | Draft a post from topic + points |
| `eval-write-holdout` | Score write quality on held-out pieces (local receipt) |
| `status` | Show profile state |
| `demo` | Optional synthetic smoke tour (no download) |
| `api` | Loopback HTTP stub |
| `logo` | Print Telivity CLI mark |

### Important flags

**`download`**

| Flag | Meaning |
| --- | --- |
| `--format mlx\|gguf` | Which quantized artifact to fetch |

**`ingest`**

| Flag | Meaning |
| --- | --- |
| `--linkedin PATH` | LinkedIn export folder or `.zip` |
| `--path PATH` | Local docs/notes/mail (repeatable) |
| `--source NAME` | Label for `--path` sources |

**`write`**

| Flag | Meaning |
| --- | --- |
| `--topic` | What the post is about |
| `--points` | Facts/claims the draft may use |
| `--k` | How many rhythm exemplars to retrieve |
| `--json` | Machine-readable receipt |

**`eval-write-holdout`**

| Flag | Meaning |
| --- | --- |
| `--holdout-id` | Piece id never indexed (repeatable) |
| `--save-raw` | Local prompts/drafts under the profile (never commit) |
| `--out PATH` | Contoso-safe aggregate receipt JSON |

---

## Advanced (optional)

These commands are available for experimentation. The shipped path above does not require them.

```bash
personality-protect select
personality-protect train --backend mlx
personality-protect filter --text "…"
personality-protect compare --synthetic slop_branding
```

See `personality-protect train --help` and `filter --help` for flags. Adapters, when used, stay under `~/.personality-protect/profiles/<name>/adapters/`.

---

## Launch script

Operator checklist: [docs/LAUNCH.md](docs/LAUNCH.md).

```bash
chmod +x scripts/beast_demo.sh
./scripts/beast_demo.sh --linkedin ~/path/to/linkedin-export
# synthetic smoke (no personal data, no multi-GB download):
./scripts/beast_demo.sh --skip-download
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
