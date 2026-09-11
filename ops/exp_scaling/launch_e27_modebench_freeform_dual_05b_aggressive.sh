#!/usr/bin/env bash
# Launch E27's aggressive treatment-only 0.5B ModeBench cohorts.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export OAT_ZERO_E27_AGGRESSIVE_05B=1
exec "$ROOT_DIR/ops/exp_scaling/launch_e22_modebench_freeform_token_maxent_v2.sh" "$@"
