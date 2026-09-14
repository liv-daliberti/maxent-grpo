#!/bin/bash
#SBATCH --job-name=e105-analysis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --partition=all
#SBATCH --account=allcs
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=var/artifacts/logs/%x-%j.out
#SBATCH --error=var/artifacts/logs/%x-%j.err

set -euo pipefail

ROOT_DIR="${OAT_ZERO_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$ROOT_DIR"
SCRIPT_PATH="$ROOT_DIR/ops/slurm/e105_post_campaign_analysis.slurm"
EXPECTED_SHA="${OAT_ZERO_ANALYSIS_SCRIPT_SHA256:?missing frozen analysis-script digest}"
OBSERVED_SHA="$(sha256sum "$SCRIPT_PATH" | awk '{print $1}')"
if [[ "$OBSERVED_SHA" != "$EXPECTED_SHA" ]]; then
  echo "post-campaign analysis script digest drifted" >&2
  exit 1
fi
PYTHON_BIN="${OAT_ZERO_ANALYSIS_PYTHON:-$(command -v python)}"

"$PYTHON_BIN" ops/exp_scaling/build_e105_group_centered_semantic_repair_results.py
"$PYTHON_BIN" ops/exp_scaling/plot_e105_group_centered_semantic_endpoint_effects.py
"$PYTHON_BIN" ops/exp_scaling/campaign_stats.py --markdown
