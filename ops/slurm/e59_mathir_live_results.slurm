#!/bin/bash
# Refresh the shared E58+E59 tracker and figure every 60 seconds.

#SBATCH --job-name=e59_mathir_plot
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
PREFIX=mie59_mathir_global_verified_replay_05b_50ep
MANIFEST="$ROOT_DIR/var/artifacts/${PREFIX}_comparative_jobs.tsv"
PYTHON_BIN="${E59_PLOT_PYTHON:-/usr/local/anaconda3/2024.02/bin/python}"
REFRESH_SECONDS="${E59_PLOT_REFRESH_SECONDS:-60}"

if [[ ! -f "$MANIFEST" ]]; then
  echo "Missing E59 manifest: $MANIFEST" >&2
  exit 1
fi
job_csv="$(tail -n +2 "$MANIFEST" | cut -f3 | paste -sd, -)"
if [[ -z "$job_csv" ]]; then
  echo "E59 manifest has no jobs: $MANIFEST" >&2
  exit 1
fi

export MPLCONFIGDIR="${TMPDIR:-/tmp}/e59-matplotlib-${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"
cd "$ROOT_DIR"

refresh() {
  "$PYTHON_BIN" ops/exp_scaling/refresh_latest_freeform_05b.py \
    --current-canonical-only
  "$PYTHON_BIN" ops/plot_modebench_paper.py
  "$PYTHON_BIN" ops/exp_scaling/monitor_campaign.py \
    --current-canonical-only --once --no-clear
}

while squeue -h -j "$job_csv" | grep -q .; do
  refresh
  sleep "$REFRESH_SECONDS"
done
refresh
