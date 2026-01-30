#!/usr/bin/env bash
set -euo pipefail

# Ensures the repo-local .local/bin is on PATH by appending a snippet
# to the user's shell rc file. Supports bash and zsh.

REPO_DIR=$(cd "$(dirname "$0")/.." && pwd)
BIN_DIR="$REPO_DIR/.local/bin"
SNIPPET="# Open R1 local bin (added by ensure_local_path.sh)
if [ -d \"$BIN_DIR\" ]; then
  case \"$PATH\" in
    *\"$BIN_DIR\"*) : ;; # already present
    *) export PATH=\"$BIN_DIR:$PATH\" ;;
  esac
fi"

detect_rc() {
  local shell_name rc
  shell_name=$(basename "${SHELL:-bash}")
  case "$shell_name" in
    zsh) rc="$HOME/.zshrc" ;;
    bash) rc="$HOME/.bashrc" ;;
    *) rc="$HOME/.profile" ;;
  esac
  printf '%s\n' "$rc"
}

apply=0
if [[ "${1:-}" == "--apply" ]]; then
  apply=1
fi

RC_FILE=$(detect_rc)

echo "Proposed PATH snippet for $RC_FILE:" >&2
echo "-------------------------------------" >&2
echo "$SNIPPET" >&2
echo "-------------------------------------" >&2

if [[ $apply -eq 1 ]]; then
  if ! grep -Fq "$BIN_DIR" "$RC_FILE" 2>/dev/null; then
    echo >> "$RC_FILE"
    echo "$SNIPPET" >> "$RC_FILE"
    echo "Appended snippet to $RC_FILE" >&2
  else
    echo "PATH already contains $BIN_DIR in $RC_FILE" >&2
  fi
  echo "Open a new shell or 'source $RC_FILE' to apply." >&2
else
  echo "Dry run. Re-run with --apply to modify $RC_FILE." >&2
fi
