#!/bin/bash
#SBATCH --job-name=pantry-retention-final-audit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=all
#SBATCH --account=allcs
#SBATCH --time=01:00:00
#SBATCH --output=var/artifacts/logs/%x-%j.out
#SBATCH --error=var/artifacts/logs/%x-%j.err

set -euo pipefail
ROOT_DIR="${ROOT_DIR:?ROOT_DIR is required}"
EXECUTION_ROOT="${OAT_ZERO_EXECUTION_ROOT:?execution root is required}"
export PYTHONPATH="$EXECUTION_ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec "$ROOT_DIR/var/seed_paper_eval/paper310/bin/python" \
  "$EXECUTION_ROOT/audit_pantry_support_retention_final_v1.py" \
  --repo-root "$ROOT_DIR" \
  --identity "$ROOT_DIR/var/artifacts/pantry_support_retention_final_v1_identity.json" \
  --submission "$ROOT_DIR/var/artifacts/pantry_support_retention_final_v1_submission.json" \
  --manifest "$ROOT_DIR/var/artifacts/pprepair_support_retention_final_v1_comparative_jobs.tsv" \
  --output "$ROOT_DIR/var/artifacts/pantry_support_retention_final_v1_audit.json"
