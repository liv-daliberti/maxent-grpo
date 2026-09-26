#!/usr/bin/env bash
# Launch the route-wise E49E finite-kernel iteration after v1 is terminal.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export E49E_KERNEL_VARIANT=v2
exec bash \
  "$ROOT_DIR/ops/math_strategy_calibration/launch_e49e_kernel_augmentation_job.sh" \
  "$@"
