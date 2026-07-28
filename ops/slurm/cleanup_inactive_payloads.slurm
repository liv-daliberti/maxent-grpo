#!/bin/bash
# Remove model/checkpoint payloads after a campaign dependency set terminates.
# Live Slurm run stamps are rechecked at execution time, so replacements that
# share a run stamp remain protected even if they have a different job ID.
#SBATCH --job-name=xdr_cleanup
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --partition=cs
#SBATCH --account=allcs
#SBATCH --time=01:00:00
#SBATCH --output=var/artifacts/logs/%x-%j.out
#SBATCH --error=var/artifacts/logs/%x-%j.err

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
exec "$ROOT_DIR/var/seed_paper_eval/paper310/bin/python" \
  "$ROOT_DIR/ops/cleanup_inactive_run_payloads.py" \
  --min-age-minutes 0 \
  --apply
