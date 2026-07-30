# Launch checklist

Operator guide for a local PersonalityProtect run. Corpus, SFT JSONL, adapters, and eval receipts stay on this machine.

## Hardware

| Path | Machine | Disk download |
| --- | --- | --- |
| `write` (post + article) | Apple Silicon | MLX 4-bit **~6 GB** |
| GGUF `filter` (optional) | Any (llama.cpp) | Q4_K_M **~5.6 GB** |
| MLX LoRA train (optional) | Apple Silicon | Reuses the MLX 4-bit base |
| CUDA QLoRA train (optional) | NVIDIA 24GB+ VRAM | Prefer GGUF for day-to-day `filter` |
| Mock / smoke | CI or pipeline check | None |

Quantized defaults stay in the **~5–7 GB** range. Full BF16 is not the happy path.

## Privacy

- State directory: `~/.personality-protect/` (override with `--home` / `PERSONALITY_PROTECT_HOME`)
- Never commit: profiles, adapters, SFT JSONL, eval receipts, LinkedIn exports, emails, notes
- Never upload personal weights or corpus to cloud train / Colab / Kaggle
- Hugging Face is used only to download **public quantized base** weights

## Operator steps

1. Install: `pip install -e ".[dev,mlx]"` plus extras (`gguf`, `cuda`, `models`) as needed.
2. Init: `personality-protect init`
3. Download the MLX base: `personality-protect download --format mlx` (add `--format gguf` only if you want `filter`).
4. Ingest local writing: `personality-protect ingest --linkedin <export> --path <docs>`
5. Index: `personality-protect index-voice`
6. Style card: `personality-protect build-style-profile`
7. Draft: `personality-protect write --topic "…" --points "…"` (add `--channel article` with 5+ `linkedin_article` pieces)
8. Receipt: `personality-protect eval-write-holdout --out receipt.json`
9. State check: `personality-protect status`

Steps 1–9 need no training run. `write` uses base weights (`adapter=none`) plus the retrieval index and style card.

## Optional experiments

- Select + train a LoRA: `personality-protect select`, then `personality-protect train` (`--writer` for the brief→post writer LoRA). Useful flags: `--proof`, `--resume`, `--chunk-steps`, `--memory-gb`. CI uses `--smoke` / `--backend mock`.
- Load an adapter for a draft: `personality-protect write --adapter …` — only after `eval-write-holdout` shows it beating the default on holdouts.
- Rewrite/score existing text: `personality-protect filter --text "…"`, `personality-protect compare --synthetic slop_branding`, `personality-protect eval --synthetic slop_branding`.

MLX train is chunked and checkpointed — a crash does not wipe a full run; use `--resume` (incomplete runs also auto-resume).

## One-shot script

```bash
# Full local run (real train when backend allows):
./scripts/beast_demo.sh --linkedin ~/path/to/linkedin-export
# Synthetic smoke only (mock — not the shipped mlx/llama path):
./scripts/beast_demo.sh --smoke --allow-mock --backend mock --skip-download
```

`scripts/beast_demo.sh` is an operator helper for PersonalityProtect. It is not a separate “PersonalityProtect demo” product.

## Branch protection checks

GitHub Actions workflow `.github/workflows/ci.yml` publishes these check names:

- `lint`
- `test (3.11)`
- `test (3.12)`
- `sanitize`
- `cli-smoke`
