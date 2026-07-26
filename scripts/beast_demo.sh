#!/usr/bin/env bash
# Beast demo: quantized download → ingest → select → full train → eval/compare.
# Personal corpus and adapters stay on this machine only. Never upload.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PROFILE="${PROFILE:-beast}"
HOME_DIR="${PERSONALITY_PROTECT_HOME:-$HOME/.personality-protect}"
BACKEND="${BACKEND:-auto}"

usage() {
  cat <<'EOF'
Usage: scripts/beast_demo.sh [options]

  --profile NAME     Profile name (default: beast)
  --home DIR         State directory (default: ~/.personality-protect)
  --backend NAME     Train backend: auto|mlx|cuda|mock (default: auto)
  --linkedin PATH    LinkedIn export dir or zip (optional)
  --path PATH        Local writing path (repeatable; optional)
  --source LABEL     Source label for --path (email|doc|note)
  --force            Allow small corpora (<20 pieces)
  --smoke            Low-step / CI path (does not silently use mock)
  --allow-mock       Permit mock when real backend unavailable
  --skip-download    Skip quantized model download
  -h, --help         Show this help

Environment:
  PERSONALITY_PROTECT_HOME  Override state directory
  PROFILE / BACKEND         Same as flags

Privacy: corpus, SFT JSONL, adapters, and eval receipts never leave this machine.
EOF
}

LINKEDIN=""
PATHS=()
SOURCE=""
SMOKE=0
ALLOW_MOCK=0
SKIP_DOWNLOAD=0
FORCE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile) PROFILE="$2"; shift 2 ;;
    --home) HOME_DIR="$2"; shift 2 ;;
    --backend) BACKEND="$2"; shift 2 ;;
    --linkedin) LINKEDIN="$2"; shift 2 ;;
    --path) PATHS+=("$2"); shift 2 ;;
    --source) SOURCE="$2"; shift 2 ;;
    --force) FORCE=1; shift ;;
    --smoke) SMOKE=1; shift ;;
    --allow-mock) ALLOW_MOCK=1; shift ;;
    --skip-download) SKIP_DOWNLOAD=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

export PERSONALITY_PROTECT_HOME="$HOME_DIR"

echo "== PersonalityProtect beast demo =="
echo "home=$HOME_DIR profile=$PROFILE backend=$BACKEND"
echo "Privacy: all corpus/adapters stay under $HOME_DIR"

personality-protect --logo off init --home "$HOME_DIR" --profile "$PROFILE"

if [[ "$SKIP_DOWNLOAD" -eq 0 && "$BACKEND" != "mock" ]]; then
  echo "== Download quantized artifact (~5–7 GB) =="
  if [[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "arm64" ]]; then
    personality-protect --logo off download --home "$HOME_DIR" --profile "$PROFILE" --format mlx || true
  fi
  personality-protect --logo off download --home "$HOME_DIR" --profile "$PROFILE" --format gguf || true
fi

if [[ -n "$LINKEDIN" || ${#PATHS[@]} -gt 0 ]]; then
  echo "== Ingest =="
  INGEST_ARGS=(--home "$HOME_DIR" --profile "$PROFILE")
  [[ -n "$LINKEDIN" ]] && INGEST_ARGS+=(--linkedin "$LINKEDIN")
  for p in "${PATHS[@]+"${PATHS[@]}"}"; do
    INGEST_ARGS+=(--path "$p")
  done
  [[ -n "$SOURCE" ]] && INGEST_ARGS+=(--source "$SOURCE")
  personality-protect --logo off ingest "${INGEST_ARGS[@]}"
else
  echo "== No --linkedin/--path; using synthetic demo corpus =="
  DEMO_CORPUS="$HOME_DIR/profiles/$PROFILE/cache/demo_corpus"
  mkdir -p "$DEMO_CORPUS"
  PROFILE="$PROFILE" PERSONALITY_PROTECT_HOME="$HOME_DIR" python - <<'PY'
import os
from pathlib import Path
from personality_protect.demo import ensure_demo_corpus

home = Path(os.environ["PERSONALITY_PROTECT_HOME"])
profile = os.environ["PROFILE"]
ensure_demo_corpus(home / "profiles" / profile / "cache" / "demo_corpus")
print("demo corpus ready")
PY
  personality-protect --logo off ingest \
    --home "$HOME_DIR" --profile "$PROFILE" \
    --path "$DEMO_CORPUS" \
    --source demo
  FORCE=1
  SMOKE=1
  ALLOW_MOCK=1
  BACKEND=mock
fi

echo "== Select =="
SELECT_ARGS=(--home "$HOME_DIR" --profile "$PROFILE" --include-undated)
[[ "$FORCE" -eq 1 ]] && SELECT_ARGS+=(--force)
# Synthetic pieces are shorter than the default 50-word gate
if [[ "$BACKEND" == "mock" || "$SMOKE" -eq 1 ]]; then
  SELECT_ARGS+=(--min-words 20)
fi
personality-protect --logo off select "${SELECT_ARGS[@]}"

echo "== Train =="
TRAIN_ARGS=(--home "$HOME_DIR" --profile "$PROFILE" --backend "$BACKEND")
[[ "$SMOKE" -eq 1 ]] && TRAIN_ARGS+=(--smoke)
[[ "$ALLOW_MOCK" -eq 1 ]] && TRAIN_ARGS+=(--allow-mock)
[[ "$FORCE" -eq 1 ]] && TRAIN_ARGS+=(--force)
personality-protect --logo off train "${TRAIN_ARGS[@]}"

echo "== Compare (synthetic slop) =="
personality-protect --logo off compare \
  --home "$HOME_DIR" --profile "$PROFILE" \
  --synthetic slop_branding \
  --backend mock \
  --json

echo ""
echo "== Eval =="
personality-protect --logo off eval \
  --home "$HOME_DIR" --profile "$PROFILE" \
  --synthetic slop_branding \
  --backend mock \
  --json

echo ""
echo "Done. Receipts under: $HOME_DIR/profiles/$PROFILE/evals/"
echo "Adapters under: $HOME_DIR/profiles/$PROFILE/adapters/"
echo "Do not commit those directories."
