#!/usr/bin/env bash
# Historical compatibility tombstone. E7 was cancelled after its objective
# audit; keeping this path prevents old notes from silently launching it.
set -euo pipefail

echo "E7 candidate projection is retired and cannot be submitted." >&2
echo "Use ops/exp_scaling/launch_on_policy_maxent_extension.sh for E9." >&2
exit 2
