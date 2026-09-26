#!/usr/bin/env bash
# Operational 7B smoke: one seed per arm, 256 updates. Submits three jobs.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/launch_e4_tau_control.sh" 7b-smoke
