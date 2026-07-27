#!/bin/bash
# Re-voice Claude drafts with local PersonalityProtect (Mac Studio).
# Tracked outside private/ so git pull brings it (private/ is gitignored).
#
# Usage:
#   ./scripts/refilter-run.sh
#   ./scripts/refilter-run.sh /path/to/repo private/runs/20260727T030224Z
set -euo pipefail
ROOT="${1:-.}"
RUN="${2:-$ROOT/private/runs/20260727T030224Z}"
cd "$ROOT"

if [ ! -d .venv ]; then
  echo "No .venv — create one: python3 -m venv .venv && source .venv/bin/activate && pip install -e \".[mlx]\""
  exit 1
fi
# shellcheck disable=SC1091
source .venv/bin/activate
export PATH="$VIRTUAL_ENV/bin:$HOME/.local/bin:/opt/homebrew/bin:$PATH"

if [ ! -d "$RUN" ]; then
  echo "Run dir not found: $RUN"
  exit 1
fi
shopt -s nullglob
files=("$RUN"/*-claude.md)
if [ ${#files[@]} -eq 0 ]; then
  echo "No *-claude.md files in $RUN"
  exit 1
fi

which personality-protect
# Don't let status|head SIGPIPE abort the script under pipefail.
personality-protect status | head -20 || true

filter_one() {
  local f="$1"
  local base chars budget attempt out json
  base=$(basename "$f" -claude.md)
  chars=$(wc -c <"$f" | tr -d ' ')
  # ~3 chars/token + margin; articles need far more than the old 480 hard cap.
  budget=$(( chars / 3 + 256 ))
  if [ "$budget" -lt 512 ]; then budget=512; fi
  if [ "$budget" -gt 4096 ]; then budget=4096; fi
  out="$RUN/${base}-voiced.md"
  json="$RUN/${base}-filter.json"
  for attempt in 1 2; do
    echo "→ filter $base (chars=$chars max_tokens=$budget --force attempt=$attempt)"
    personality-protect filter \
      --file "$f" \
      --out "$out" \
      --backend auto \
      --max-tokens "$budget" \
      --force \
      --json >"$json" || true
    python3 - <<PY
import json, sys
from pathlib import Path
meta = json.loads(Path("$json").read_text())
draft = Path("$f").read_text().strip()
voiced = Path("$out").read_text().strip() if Path("$out").is_file() else ""
# JSON text is authoritative when --out was wiped empty by a bad sample.
text = (meta.get("text") or "").strip()
if text and not voiced:
    Path("$out").write_text(text + "\n", encoding="utf-8")
    voiced = text
print(
    f"  backend={meta.get('backend')} max_tokens={meta.get('max_tokens')} "
    f"force={meta.get('force')} unchanged={meta.get('unchanged')} "
    f"truncated={meta.get('likely_truncated')} len={len(draft)}→{len(voiced)}"
)
ok = bool(voiced) and len(voiced) >= len(draft) * 0.55
if meta.get("unchanged") or draft == voiced:
    print("  WARN: still leave-alone even with --force (adapter may be copy-biased)")
if not ok:
    print("  WARN: empty/truncated rewrite")
    sys.exit(2)
sys.exit(0)
PY
    if [ $? -eq 0 ]; then
      return 0
    fi
    echo "  retrying $base…"
  done
  echo "  FAIL: $base still empty/truncated after retry"
  return 1
}

fail=0
for f in "${files[@]}"; do
  filter_one "$f" || fail=1
done
if [ "$fail" -ne 0 ]; then
  echo "Some filters failed — re-run or raise --max-tokens for those pieces."
fi
echo "Done. Voiced files in $RUN"
ls -la "$RUN"/*-voiced.md
