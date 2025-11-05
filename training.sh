#!/usr/bin/env bash
set -euo pipefail

# Simple wrapper to match README instructions; delegates to the main launcher.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/training-math-grpo.sh" "$@"

