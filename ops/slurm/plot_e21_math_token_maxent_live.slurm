#!/bin/bash
# Refresh the E21 live figure while any cohort job remains queued or running.

#SBATCH --job-name=e21_live_plot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --partition=cs
#SBATCH --account=allcs
#SBATCH --time=7-00:00:00
#SBATCH --output=var/artifacts/logs/%x-%j.out
#SBATCH --error=var/artifacts/logs/%x-%j.err

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
PREFIX="${E21_PREFIX:-mte21_math_conditional_token_05b_v4}"
MANIFEST="${E21_MANIFEST:-$ROOT_DIR/var/artifacts/${PREFIX}_comparative_jobs.tsv}"
REFRESH_SECONDS="${E21_PLOT_REFRESH_SECONDS:-300}"

if [[ ! -f "$MANIFEST" ]]; then
  echo "Missing E21 manifest: $MANIFEST" >&2
  exit 1
fi
job_csv="$(tail -n +2 "$MANIFEST" | cut -f3 | paste -sd, -)"
if [[ -z "$job_csv" ]]; then
  echo "E21 manifest has no jobs: $MANIFEST" >&2
  exit 1
fi

cd "$ROOT_DIR"
while squeue -h -j "$job_csv" | grep -q .; do
  python ops/plot_e21_math_token_maxent_live.py --prefix "$PREFIX"
  sleep "$REFRESH_SECONDS"
done
python ops/plot_e21_math_token_maxent_live.py --prefix "$PREFIX"
