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

for f in "${files[@]}"; do
  base=$(basename "$f" -claude.md)
  chars=$(wc -c <"$f" | tr -d ' ')
  # ~3 chars/token + margin; articles need far more than the old 480 hard cap.
  budget=$(( chars / 3 + 256 ))
  if [ "$budget" -lt 512 ]; then budget=512; fi
  if [ "$budget" -gt 4096 ]; then budget=4096; fi
  echo "→ filter $base (chars=$chars max_tokens=$budget)"
  personality-protect filter \
    --file "$f" \
    --out "$RUN/${base}-voiced.md" \
    --backend auto \
    --max-tokens "$budget" \
    --json | tee "$RUN/${base}-filter.json" | head -c 400
  echo
  python3 - <<PY
import json
from pathlib import Path
meta = json.loads(Path("$RUN/${base}-filter.json").read_text())
draft = Path("$f").read_text().strip()
voiced = Path("$RUN/${base}-voiced.md").read_text().strip()
print(
    f"  backend={meta.get('backend')} max_tokens={meta.get('max_tokens')} "
    f"unchanged={meta.get('unchanged')} truncated={meta.get('likely_truncated')} "
    f"len={len(draft)}→{len(voiced)}"
)
if meta.get("unchanged") or draft == voiced:
    print("  WARN: leave-alone / no-op")
if meta.get("likely_truncated") or (draft and len(voiced) < len(draft) * 0.55):
    print("  WARN: likely truncated — raise --max-tokens")
PY
done
echo "Done. Voiced files in $RUN"
ls -la "$RUN"/*-voiced.md
