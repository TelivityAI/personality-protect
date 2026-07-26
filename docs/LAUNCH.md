# Launch checklist

Operator guide for a local PersonalityProtect run. Corpus, SFT JSONL, adapters, and eval receipts stay on this machine.

## Hardware

| Path | Machine | Disk download |
| --- | --- | --- |
| MLX train + filter | Apple Silicon | MLX 4-bit **~6 GB** |
| GGUF filter | Any (llama.cpp) | Q4_K_M **~5.6 GB** |
| CUDA QLoRA train | NVIDIA 24GB+ VRAM | Prefer GGUF for day-to-day filter |
| Mock / smoke | CI or pipeline check | None |

Quantized defaults stay in the **~5–7 GB** range. Full BF16 is not the happy path.

## Privacy

- State directory: `~/.personality-protect/` (override with `--home` / `PERSONALITY_PROTECT_HOME`)
- Never commit: profiles, adapters, SFT JSONL, eval receipts, LinkedIn exports, emails, notes
- Never upload personal weights or corpus to cloud train / Colab / Kaggle
- Hugging Face is used only to download **public quantized base** weights

## Operator steps

1. Install: `pip install -e ".[dev]"` plus extras (`mlx`, `gguf`, `cuda`, `models`) as needed.
2. Init: `personality-protect init`
3. Download one quantized artifact: `personality-protect download` (GGUF) and/or `--format mlx` on Apple Silicon.
4. Ingest local writing: `personality-protect ingest --linkedin <export> --path <docs>`
5. Select: `personality-protect select` (warns below 50 pieces; blocks below 20 unless `--force`)
6. Full train: `personality-protect train` (auto steps from SFT count). CI uses `--smoke` / `--backend mock`.
7. Filter: `personality-protect filter --text "…"`
8. Compare: `personality-protect compare --synthetic slop_branding`
9. Eval: `personality-protect eval --synthetic slop_branding`

## One-shot script

```bash
./scripts/beast_demo.sh --linkedin ~/Downloads/LinkedInExport
# or synthetic smoke (no personal data):
./scripts/beast_demo.sh --smoke --allow-mock --backend mock --skip-download
```

## Branch protection checks

GitHub Actions workflow `.github/workflows/ci.yml` publishes these check names:

- `lint`
- `test (3.11)`
- `test (3.12)`
- `sanitize`
- `cli-smoke`
