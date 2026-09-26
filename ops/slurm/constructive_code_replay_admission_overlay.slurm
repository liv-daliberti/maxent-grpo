#!/bin/bash
#SBATCH --job-name=constructive_code_overlay_admit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --partition=all
#SBATCH --time=01:00:00
#SBATCH --output=var/artifacts/logs/constructive-code-overlay-admission-%j.out
#SBATCH --error=var/artifacts/logs/constructive-code-overlay-admission-%j.err

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
SCRATCH_ROOT="${SLURM_TMPDIR:-/tmp}/constructive-code-admission-$SLURM_JOB_ID"
mkdir -p "$SCRATCH_ROOT/candidates"

exec /n/fs/similarity/maxent-grpo/var/seed_paper_eval/paper310/bin/python \
  "$ROOT_DIR/ops/replay_constructive_code_review_slate.py" \
  --runtime-root "$SCRATCH_ROOT/python-root" \
  --launcher "$SCRATCH_ROOT/constructive_code_sandbox" \
  --build-root "$SCRATCH_ROOT/checkers" \
  --scratch-root "$SCRATCH_ROOT/candidates" \
  --replays-output \
    "$ROOT_DIR/var/artifacts/constructive_code_checker_replays_admission_overlay_20x.jsonl" \
  --manifest-output \
    "$ROOT_DIR/var/artifacts/constructive_code_admission_manifest_overlay_20x.json" \
  --audit-output \
    "$ROOT_DIR/var/artifacts/constructive_code_checker_equivalence_admission_overlay_20x.json" \
  --run-audit-output \
    "$ROOT_DIR/var/artifacts/constructive_code_replay_run_admission_overlay_20x.json" \
  --correct-limit 20 \
  --incorrect-limit 20 \
  --verified-threshold 0.9 \
  --replay-stage admission \
  --workers 16
