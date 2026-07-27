#!/usr/bin/env bash
# Fail CI if private paths or discussion-only language leak into tracked files.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Portable file list (bash 3.2+ / Ubuntu CI)
FILES=()
while IFS= read -r line; do
  FILES+=("$line")
done < <(git ls-files \
  '*.py' '*.md' '*.yml' '*.yaml' '*.toml' '*.sh' '*.txt' '*.json' '*.jsonl' \
  | grep -v '^LICENSE$' \
  | grep -v '^scripts/sanitize_check\.sh$' \
  || true)

if [[ ${#FILES[@]} -eq 0 ]]; then
  echo "No files to sanitize."
  exit 0
fi

FAIL=0

check_pattern() {
  local label="$1"
  local pattern="$2"
  local hits
  hits="$(rg -n --no-heading -S "$pattern" "${FILES[@]}" 2>/dev/null || true)"
  if [[ -n "$hits" ]]; then
    echo "SANITIZE FAIL ($label):"
    echo "$hits"
    FAIL=1
  fi
}

# Absolute personal cloud / export paths
check_pattern "dropbox-path" '/Users/[^/]+/Dropbox|/Users/[^/]+/Library/CloudStorage/Dropbox'
check_pattern "linkedin-personal-path" '/Users/[^/]+/.*LinkedIn|/Users/[^/]+/Downloads/.*[Ee]xport'
check_pattern "home-personal-notes" '/Users/[^/]+/(Documents|Desktop)/.*(notes|emails|linkedin)'

# Discussion / planning language that must not ship in product docs
check_pattern "discussion-leak" '(?i)\b(thread discussion|offline discussion|beast upgrade plan|do not tell the user|as we discussed in chat)\b'

# Accidental secret filenames committed
check_pattern "secret-files" '(^|/)\.env($|\.)|credentials\.json|huggingface_token|HF_TOKEN'

# Local-only / unpublished workflow paths must not appear in the public tree
check_pattern "private-dir-path" 'private/(studio|seeds|runs|prompts|\.env)'
check_pattern "private-voice-studio" '(?i)private voice studio|voice studio directory'
check_pattern "orchestration-leak" '(?i)Kimi\s*→\s*Claude|Kimi->Claude|private/studio\.py'

if [[ "$FAIL" -ne 0 ]]; then
  echo "Sanitize check failed. Remove private paths and discussion language from tracked files."
  exit 1
fi

echo "Sanitize OK (${#FILES[@]} files)."
