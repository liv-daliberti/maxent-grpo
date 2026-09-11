#!/usr/bin/env bash
# Compatibility wrapper for the scale-aware E4 workflow. Submits jobs.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/launch_e4_tau_control.sh" 3b
