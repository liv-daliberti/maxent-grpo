#!/bin/bash
#SBATCH --job-name=audit-pantry-drgrpo-r1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=all
#SBATCH --time=00:30:00
#SBATCH --output=var/artifacts/logs/audit-pantry-drgrpo-r1-%j.out
#SBATCH --error=var/artifacts/logs/audit-pantry-drgrpo-r1-%j.err

set -euo pipefail
ROOT_DIR="${ROOT_DIR:?ROOT_DIR is required}"
EXECUTION_ROOT="${OAT_ZERO_EXECUTION_ROOT:?execution root is required}"
TRAIN_JOB_ID="${PANTRY_R1_TRAIN_JOB_ID:?training job ID is required}"
exec "$ROOT_DIR/var/seed_paper_eval/paper310/bin/python" \
  "$EXECUTION_ROOT/audit_pantry_support_mask_drgrpo_smoke_v1_r1.py" \
  --repo-root "$ROOT_DIR" \
  --identity "$ROOT_DIR/var/artifacts/pantry_support_mask_drgrpo_smoke_v1_r1_identity.json" \
  --submission "$ROOT_DIR/var/artifacts/pantry_support_mask_drgrpo_smoke_v1_r1_submission.json" \
  --manifest "$ROOT_DIR/var/artifacts/ppsmoke_support_mask_drgrpo_v1_r1_comparative_jobs.tsv" \
  --job-id "$TRAIN_JOB_ID" \
  --output "$ROOT_DIR/var/artifacts/pantry_support_mask_drgrpo_smoke_v1_r1_audit.json"
