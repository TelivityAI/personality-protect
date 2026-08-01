# PersonalityProtect

**Stop sounding like everyone else's AI.**

Draft LinkedIn posts and articles in *your* voice on your machine. Index your writing, measure your cadence, retrieve short rhythm references from your own pieces, and generate with a local quantized model (~5–7 GB). Your corpus never leaves your Mac.

Built by [Telivity](https://telivity.com). Apache-2.0.

```text
your writing ──► ingest ──► index-voice ──► build-style-profile ──► write [--channel post|article]
```

No fine-tune is required to get a draft. `write` runs the local base model with your retrieval index plus your measured style card (`adapter=none`).

<p align="center">
  <img src="docs/images/cli-shipped.png" alt="personality-protect write drafting a Contoso post locally with adapter=none" width="800" />
</p>

---

## Why

The feed is drowning in AI writing that all sounds the same — *“In today’s fast-paced world, we must leverage synergies…”* A pasted style guide helps a little. It still isn’t **you**.

Cloud fine-tunes are a non-option for personal writing. Notes, emails, and posts are biometric-adjacent. Shipping them to a rented GPU so someone else’s stack can imitate you is a strange bargain.

PersonalityProtect keeps the corpus on disk, measures your cadence, retrieves short rhythm references from your own writing, and drafts locally with MLX on Apple Silicon. Treat outputs as drafts you still own.

---

## How you get voice

1. **Ingest** your LinkedIn export and/or local notes (stays on disk).
2. **`select`** gates the corpus by length (`--min-words`, default 50) and an optional year cap (`--through-year`, default: current year).
3. **`index-voice`** builds a local retrieval index.
4. **`build-style-profile`** measures cadence from the selection (sentence length, short lines, post length band, banned filler).
5. **`write --topic --points`** drafts from the brief only; retrieved pieces are rhythm reference.

Two channels come out of step 5, and each takes its length from its own pieces:

- **`--channel post`** (default) targets your long-post band, up to the LinkedIn ~3000-character limit (~550 words).
- **`--channel article`** runs outline → sections → stitch. Total length comes from the p50/p75/p90 of your `linkedin_article` pieces, split across the outline; the post ceiling never applies. Needs at least five `linkedin_article` pieces both in the corpus and in the voice index.

Local LoRA training stays in the CLI as an experiment, not as the path to a first draft — see [Advanced](#advanced-optional).

---

## Quick start

Requires Python 3.10+ and an Apple Silicon Mac: `write` runs on MLX and needs Metal.

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

personality-protect select
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

That is the whole path to a draft — no training step.

Optional extras:

```bash
pip install -e ".[models]"   # Hugging Face download helper
pip install -e ".[gguf]"     # llama.cpp GGUF (optional)
pip install -e ".[cuda]"     # NVIDIA path (optional)
```

---

## Screenshots

Public docs use **synthetic Contoso / synergy-slop text only**. No personal corpus.

| `write` (post + article) | `status` | Setup |
| --- | --- | --- |
| <img src="docs/images/cli-shipped.png" alt="write drafting a Contoso post with adapter=none" width="360" /> | <img src="docs/images/cli-status.png" alt="status output for the synthetic demo profile" width="280" /> | <img src="docs/images/cli-setup.png" alt="personality-protect setup / logo" width="280" /> |

```bash
personality-protect select
personality-protect index-voice
personality-protect build-style-profile
personality-protect write --topic "Contoso pricing" --points "Name one owner."
personality-protect status
```

Optional smoke tour — runs the write path with a stubbed model call (no download; synthetic Contoso only):

| Smoke tour (`demo`) | Mark |
| --- | --- |
| <img src="docs/images/cli-demo.png" alt="personality-protect demo smoke tour of the write path" width="360" /> | <img src="docs/images/cli-logo.png" alt="Telivity CLI logo" width="280" /> |

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

**`write` requires an Apple Silicon Mac.** Drafting runs Qwen3.5-9B 4-bit through MLX/Metal; there is no cloud fallback.

| What | Size / note |
| --- | --- |
| MLX 4-bit base (what `write` loads) | ~6 GB download, once |
| Peak RAM while writing | Memory-capped; 16 GB+ recommended |
| GGUF Q4_K_M (optional, for `filter`) | ~5.6 GB download |
| Writer LoRA (optional) | Small (MBs) under the profile |

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

### Select, index and style

`select` is required before `build-style-profile` (the style card reads `selection.json`). It is a length gate plus an optional year cap:

```bash
personality-protect select
personality-protect select --min-words 75 --include-undated
personality-protect select --through-year 2024   # deliberate narrowing only
personality-protect index-voice
personality-protect build-style-profile
```

Defaults: **≥50 words**, dates through the **current year**. Use `--through-year` when you intentionally want an older slice. Corpus gates: **warn** below 50 selected pieces; **block** below 20 unless `--force`. Holding pieces back from retrieval is separate — `index-voice --holdout-id`, scored by `eval-write-holdout`.

Length targets are per channel and never borrow across channels:

| Channel | Measured from | Aim | Ceiling |
| --- | --- | --- | --- |
| post | `linkedin_post` pieces | p75 (floor 300) | p90, clamped to ~550 words (~3000 chars) |
| article | `linkedin_article` pieces | median, clamped to 600–3000 words | p90, clamped to 3000 words |

The article aim is divided across the outline to get a per-section budget (clamped to 180–600 words), so a five-section article asks for five short sections rather than five posts. With no `linkedin_article` pieces in the corpus, the article aim falls back to a stated default of 1100 words instead of borrowing the post band.

### Write

```bash
personality-protect write --topic "…" --points "…"
personality-protect write --channel article --topic "…" --points "…"
personality-protect write --topic "…" --points "…" --json
```

`--topic` and `--points` are the only content the draft may use; retrieved pieces supply rhythm, not facts. Every `write` above runs base weights (`adapter=none`).

On `--channel article`, each `--points` bullet becomes a section (2–8), retrieval is restricted to `linkedin_article` pieces so posts cannot become the rhythm reference, and sections that restate each other (including paraphrases) are dropped before stitching. Each section prompt sees only its own bullet — not the full brief dumped into every call — plus titles of the other sections so it does not rewrite them. Exemplar clips strip export chrome (duplicate titles, Created/Published lines) so the word budget is real prose. The style card for this channel uses article-only cadence when enough articles were measured, not the comment-dominated corpus card. Section prompts also use an article-specific system rule set: the BRIEF is the only fact source, allowed names/figures from the BRIEF are listed explicitly, and thin briefs lower the per-section word aim. A section that invents is repaired before it is dropped: the regenerate names the fabricated entities and figures it has to lose, and if they survive that, the sentences carrying them are cut and the section is re-checked. Only a section with nothing left is omitted, and the whole article is flagged for invention only when the stitch is empty or the stitched text still invents. The channel refuses to draft unless at least five `linkedin_article` pieces are in the corpus *and* five are in the voice index — a large carve that leaves retrieval empty is an error, not a silently thinner draft.

### Article holdout eval

The post channel is scored by `eval-write-holdout`. The article channel has its own carve and eval:

```bash
personality-protect select-article-holdouts            # report only
personality-protect select-article-holdouts --apply
personality-protect index-voice --from-carve           # holdouts leave retrieval
PP_MLX_ALLOW=1 personality-protect eval-write-article --out receipt.json
```

The carve is deterministic (`blake2b(piece_id)` order), keeps previously carved ids pinned, and never drops the voice index below the five-article floor. Each holdout is reduced to a lossy brief — a topic plus 3–6 section bullets drawn one per segment of the piece, capped at 60 words and 10% of the source — so neither arm is handed the article back to paraphrase.

Two arms then write the same brief with the same outline, per-section budget, trim, and invent repair. The product arm gets retrieved exemplars and the measured style card; the control arm gets neither. Invention is judged against the visible brief both arms saw (not the full source article). Drafts are scored on distance to the holdout's own cadence axes, and a draft that parrots its exemplars, echoes the brief, or invents entities or figures is disqualified regardless of distance. Receipts carry ids, distances, and flags — never draft or corpus text.

The verdict needs all three of: the article arm wins the majority, the margin clears `--alpha` (default 0.10) on a one-sided sign test, and it is not disqualified more often than the control. When both arms are disqualified on every holdout, distance never decided anything, and the receipt says so (`distance_ever_decided: false`) rather than reporting it as a cadence loss.

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
| `dedupe-index` | Report pieces repeating another piece's text; `--apply` backs up then rewrites the index |
| `index-voice` | Build local voice retrieval index |
| `build-style-profile` | Build cadence / length / banned-filler style card |
| `write` | Draft a post or article (`--channel post\|article`) |
| `eval-write-holdout` | Score post-channel writes on held-out pieces (local receipt) |
| `select-article-holdouts` | Deterministic article carve that respects the retrieval floor |
| `eval-write-article` | Score article-channel writes against a no-voice control (local receipt) |
| `status` | Show profile state |
| `demo` | Optional synthetic smoke tour of the write path (no download) |
| `api` | Loopback HTTP stub |
| `logo` | Print Telivity CLI mark |
| `build-writer-sft`, `train` | Optional LoRA experiments — see [Advanced](#advanced-optional) |

### `write` flags

| Flag | Meaning |
| --- | --- |
| `--topic` | What the piece is about |
| `--points` | Facts/claims the draft may use |
| `--channel post\|article` | Post (default) or article outline→sections→stitch |
| `--k` | Rhythm exemplars to retrieve |
| `--adapter` / `--no-adapter` | Default `--no-adapter` (base weights); `--adapter` needs a trained LoRA |
| `--json` | Machine-readable receipt |

### `eval-write-holdout` flags

| Flag | Meaning |
| --- | --- |
| `--holdout-id` | Piece id never indexed (repeatable) |
| `--save-raw` | Local prompts/drafts under the profile (never commit) |
| `--out PATH` | Contoso-safe aggregate receipt JSON |

### `eval-write-article` flags

| Flag | Meaning |
| --- | --- |
| `--holdout-id` | Article id to score (repeatable); defaults to the saved carve |
| `--k` | Article exemplars retrieved per section |
| `--alpha` | One-sided significance the run must reach (default 0.10) |
| `--save-raw` | Local prompts/drafts under the profile (never commit) |
| `--out PATH` | Contoso-safe receipt JSON |

### `select-article-holdouts` flags

| Flag | Meaning |
| --- | --- |
| `--apply` | Write the carve (default is report-only) |
| `--fraction` | Share of briefable articles to reserve |
| `--min` / `--max` | Carve size band (4–5; three cannot reach `--alpha` on a sign test) |
| `--keep-indexed` | Articles the carve must leave in retrieval (default 5) |

---

## Advanced (optional)

Nothing here is needed for a draft. These commands stay in the CLI for local experiments and receipts.

### Writer LoRA (experimental plumbing)

`write` defaults to base weights. The adapter path exists so a trained LoRA *can* be loaded, and `--adapter` errors out when no adapter is present:

```bash
personality-protect build-writer-sft
personality-protect select-writer-holdouts --apply
personality-protect index-voice --from-carve
personality-protect train --writer --backend mlx
personality-protect eval-writer-adapter --archive-on-fail
personality-protect write --adapter --topic "…" --points "…"
```

`build-writer-sft` builds de-voiced brief→post pairs. `train --writer` uses a short writer recipe (3 epochs) and keeps per-chunk checkpoints under `adapters/latest/checkpoints/`. Keep an adapter only when `eval-writer-adapter` decides `keep`; otherwise archive it and stay on `adapter=none`. Training is not a prerequisite for `write`.

### Other experiment commands

`filter`, `compare`, `eval`, and the translator-pair commands remain available. They score or rewrite existing text and are not part of the drafting path above. (`select` *is* part of the drafting path — see [Select, index and style](#select-index-and-style).)

### Operator script

`scripts/beast_demo.sh` drives the older `select` → `train` → `compare` → `eval` sequence, not `write`. Use it for train/compare runs only:

```bash
chmod +x scripts/beast_demo.sh
./scripts/beast_demo.sh --linkedin ~/path/to/linkedin-export
./scripts/beast_demo.sh --skip-download   # synthetic smoke
```

Operator checklist: [docs/LAUNCH.md](docs/LAUNCH.md).

---

## Develop

```bash
pip install -e ".[dev]"
pytest
ruff check src tests scripts
```

CI (`.github/workflows/ci.yml`) required checks: `lint`, `test (3.11)`, `test (3.12)`, `sanitize`, `cli-smoke`.

### Regenerating screenshots

`scripts/shot.py` renders captured ANSI terminal bytes to PNG on a fixed character grid (so Rich box-drawing lines up). Needs `pillow`.

```bash
export PERSONALITY_PROTECT_HOME=/tmp/shots COLUMNS=94 TERM=xterm-256color
pip install pillow
script -qec "personality-protect --logo off demo" /dev/null \
  | python3 scripts/shot.py docs/images/cli-demo.png "personality-protect demo"
script -qec "personality-protect --logo off status" /dev/null \
  | python3 scripts/shot.py docs/images/cli-status.png "personality-protect status"
```

Do not regenerate `docs/images/cli-shipped.png` off Apple Silicon — it shows a real `write` against MLX weights.

---

## License

Apache-2.0. See [LICENSE](LICENSE).
