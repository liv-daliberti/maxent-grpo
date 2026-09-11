#!/usr/bin/env bash
# Retry only E65's atomic held-cohort submission while Slurm is unavailable.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IDENTITY="$ROOT_DIR/var/artifacts/e65_entropy_gated_singleton_confirmation_identity.json"
LAUNCHER="$ROOT_DIR/ops/exp_scaling/launch_e65_entropy_gated_singleton_confirmation.sh"
LOG="$ROOT_DIR/var/artifacts/e65_submission_watch.log"

exec 9>"$ROOT_DIR/var/artifacts/e65_submission_watch.lock"
if ! flock -n 9; then
  echo "[e65-submit-watch] another watcher owns the lock"
  exit 0
fi

while [[ ! -f "$IDENTITY" ]]; do
  if scontrol ping 2>/dev/null | grep -q "is UP"; then
    echo "[$(date -Is)] Slurm is up; attempting atomic E65 submission" >> "$LOG"
    if bash "$LAUNCHER" full >> "$LOG" 2>&1; then
      echo "[$(date -Is)] E65 submission completed" >> "$LOG"
      exit 0
    fi
    echo "[$(date -Is)] E65 submission attempt failed; retrying" >> "$LOG"
  fi
  sleep 60
done
