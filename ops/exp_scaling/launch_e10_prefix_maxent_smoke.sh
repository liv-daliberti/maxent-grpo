#!/usr/bin/env bash
# Historical tombstone: E10 optimized H/T_max and must not use E11 source.
set -euo pipefail

echo "E10 is a completed normalized-entropy smoke and cannot be relaunched from the standard-MaxEnt checkout." >&2
echo "Use ops/exp_scaling/launch_e11_standard_maxent_smoke.sh instead." >&2
exit 2
