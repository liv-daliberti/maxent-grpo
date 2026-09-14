#!/bin/bash
#SBATCH --job-name=e105-post-gate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --partition=all
#SBATCH --account=allcs
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=var/artifacts/logs/%x-%j.out
#SBATCH --error=var/artifacts/logs/%x-%j.err

set -euo pipefail

ROOT_DIR="${OAT_ZERO_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$ROOT_DIR"
SCRIPT_PATH="$ROOT_DIR/ops/slurm/e105_post_gate_release.slurm"
EXPECTED_SHA="${OAT_ZERO_RELEASE_SCRIPT_SHA256:?missing frozen release-script digest}"
OBSERVED_SHA="$(sha256sum "$SCRIPT_PATH" | awk '{print $1}')"
if [[ "$OBSERVED_SHA" != "$EXPECTED_SHA" ]]; then
  echo "post-gate release script digest drifted" >&2
  exit 1
fi
PYTHON_BIN="${OAT_ZERO_RELEASE_PYTHON:-$(command -v python)}"

"$PYTHON_BIN" ops/exp_scaling/audit_e106_python_lambda_normalization.py
"$PYTHON_BIN" ops/exp_scaling/plot_e106_group_centered_mechanism_diagnostic.py
"$PYTHON_BIN" ops/exp_scaling/apply_e105_qwen3_paired_a6000_placement_amendment.py --apply
"$PYTHON_BIN" ops/exp_scaling/launch_e109_repaired_python_replay_comparators.py --submit
"$PYTHON_BIN" ops/exp_scaling/launch_e105_group_centered_semantic_repair_full_three_scale.py --submit
"$PYTHON_BIN" ops/exp_scaling/campaign_stats.py --markdown
