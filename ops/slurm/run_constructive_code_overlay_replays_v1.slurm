#!/usr/bin/env bash
# Frozen four-family released-checker replay on the CodeContests-O overlay.
#SBATCH --job-name=constructive-overlay-v1
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=var/artifacts/logs/constructive-overlay-v1-%j.out
#SBATCH --error=var/artifacts/logs/constructive-overlay-v1-%j.err

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-/n/fs/similarity/maxent-grpo}"
LOCAL_ROOT="$ROOT_DIR/var/tmp/constructive-overlay-v1-${SLURM_JOB_ID:?SLURM_JOB_ID is required}"
mkdir -p "$ROOT_DIR/var/artifacts/logs" "$LOCAL_ROOT/scratch"
cd "$ROOT_DIR"
exec "$ROOT_DIR/var/seed_paper_eval/paper310/bin/python" \
  "$ROOT_DIR/ops/run_constructive_code_overlay_replays.py" \
  --runtime-root "$LOCAL_ROOT/runtime" \
  --launcher "$LOCAL_ROOT/constructive_code_sandbox" \
  --scratch-root "$LOCAL_ROOT/scratch" \
  --workers 12
