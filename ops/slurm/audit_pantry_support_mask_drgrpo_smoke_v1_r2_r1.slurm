#!/bin/bash
#SBATCH --job-name=audit-pantry-r2-r1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=all
#SBATCH --account=allcs
#SBATCH --time=00:30:00
#SBATCH --output=var/artifacts/logs/%x-%j.out
#SBATCH --error=var/artifacts/logs/%x-%j.err

set -euo pipefail
ROOT_DIR="${ROOT_DIR:?ROOT_DIR is required}"
EXECUTION_ROOT="${OAT_ZERO_EXECUTION_ROOT:?execution root is required}"
exec "$ROOT_DIR/var/seed_paper_eval/paper310/bin/python" \
  "$EXECUTION_ROOT/audit_pantry_support_mask_drgrpo_smoke_v1_r2_r1.py" \
  --repo-root "$ROOT_DIR" \
  --identity "$ROOT_DIR/var/artifacts/pantry_support_mask_drgrpo_smoke_v1_r2_identity.json" \
  --submission "$ROOT_DIR/var/artifacts/pantry_support_mask_drgrpo_smoke_v1_r2_submission.json" \
  --manifest "$ROOT_DIR/var/artifacts/ppsmoke_support_mask_drgrpo_v1_r2_comparative_jobs.tsv" \
  --job-id 30199933 \
  --output "$ROOT_DIR/var/artifacts/pantry_support_mask_drgrpo_smoke_v1_r2_r1_audit.json"
