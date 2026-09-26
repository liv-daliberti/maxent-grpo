#!/bin/bash
#SBATCH --job-name=constructive_code_replay_repair
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --partition=all
#SBATCH --time=00:30:00
#SBATCH --output=var/artifacts/logs/constructive-code-replay-repair-%j.out
#SBATCH --error=var/artifacts/logs/constructive-code-replay-repair-%j.err

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
SCRATCH_ROOT="${SLURM_TMPDIR:-/tmp}/constructive-code-repair-$SLURM_JOB_ID"
mkdir -p "$SCRATCH_ROOT/candidates"

exec /n/fs/similarity/maxent-grpo/var/seed_paper_eval/paper310/bin/python \
  "$ROOT_DIR/ops/replay_constructive_code_review_slate.py" \
  --runtime-root "$SCRATCH_ROOT/python-root" \
  --launcher "$SCRATCH_ROOT/constructive_code_sandbox" \
  --build-root "$SCRATCH_ROOT/checkers" \
  --scratch-root "$SCRATCH_ROOT/candidates" \
  --replays-output \
    "$ROOT_DIR/var/artifacts/constructive_code_checker_replays_compat_repair_5x.jsonl" \
  --manifest-output \
    "$ROOT_DIR/var/artifacts/constructive_code_admission_manifest_compat_repair_5x.json" \
  --audit-output \
    "$ROOT_DIR/var/artifacts/constructive_code_checker_equivalence_compat_repair_5x.json" \
  --run-audit-output \
    "$ROOT_DIR/var/artifacts/constructive_code_replay_run_compat_repair_5x.json" \
  --correct-limit 5 \
  --incorrect-limit 5 \
  --verified-threshold 0.9 \
  --workers 16 \
  --task 1208_C \
  --task 1516_C
