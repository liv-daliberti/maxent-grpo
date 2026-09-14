#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/n/fs/similarity/maxent-grpo"
cd "$ROOT_DIR"
exec "$ROOT_DIR/var/seed_paper_eval/paper310/bin/python" \
  "$ROOT_DIR/ops/exp_scaling/monitor_e113r3_failed_allocations.py" \
  --poll-seconds 30 --max-seconds 601200
