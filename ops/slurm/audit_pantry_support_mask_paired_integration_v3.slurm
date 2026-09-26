#!/bin/bash
#SBATCH --job-name=audit-pantry-integration-v3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=var/artifacts/logs/%x-%j.out
#SBATCH --error=var/artifacts/logs/%x-%j.err

set -euo pipefail
ROOT_DIR="${ROOT_DIR:?ROOT_DIR is required}"
EXECUTION_ROOT="${OAT_ZERO_EXECUTION_ROOT:?execution root is required}"
source "$ROOT_DIR/ops/repo_env.sh"
export PYTHONPATH="$EXECUTION_ROOT:$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$ROOT_DIR/var/seed_paper_eval/paper310/bin/python" \
  "$EXECUTION_ROOT/audit_pantry_support_mask_paired_integration_v3.py" \
  --repo-root "$ROOT_DIR" \
  --identity "$ROOT_DIR/var/artifacts/pantry_support_mask_paired_integration_v3_identity.json" \
  --submission "$ROOT_DIR/var/artifacts/pantry_support_mask_paired_integration_v3_submission.json" \
  --manifest "$ROOT_DIR/var/artifacts/ppsmoke_support_mask_paired_integration_v3_comparative_jobs.tsv" \
  --output "$ROOT_DIR/var/artifacts/pantry_support_mask_paired_integration_v3_audit.json"
