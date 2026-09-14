#!/usr/bin/env bash
#SBATCH --job-name=mb-l2-admit
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=var/logs/modebench_level2_viability/%j_aggregate.out
#SBATCH --error=var/logs/modebench_level2_viability/%j_aggregate.err
set -euo pipefail
ROOT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
source "$ROOT_DIR/ops/repo_env.sh"
exec "$ROOT_DIR/var/seed_paper_eval/paper310/bin/python" "$ROOT_DIR/ops/aggregate_modebench_level2_admission.py" --data-root "$ROOT_DIR/var/data/modebench_harder_v2_matched_r1" --receipts-dir "$ROOT_DIR/var/results/modebench_level2_viability_r1_frozen"
