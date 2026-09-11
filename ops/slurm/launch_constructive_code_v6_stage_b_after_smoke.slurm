#!/bin/bash
#SBATCH --job-name=launch-constructive-v6-stage-b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --partition=all
#SBATCH --account=allcs
#SBATCH --time=00:30:00
#SBATCH --output=var/artifacts/logs/%x-%j.out
#SBATCH --error=var/artifacts/logs/%x-%j.err

set -euo pipefail
ROOT_DIR="${ROOT_DIR:?ROOT_DIR is required}"
EXPECTED="${OAT_ZERO_EXPECTED_LAUNCHER_SHA256:?launcher hash is required}"
LAUNCHER="$ROOT_DIR/ops/exp_scaling/launch_constructive_code_v6_experiment.py"
[[ "$(sha256sum "$LAUNCHER" | cut -d' ' -f1)" == "$EXPECTED" ]] || {
  echo "ConstructiveCode Stage-B launcher hash drift" >&2
  exit 1
}
exec "$ROOT_DIR/var/seed_paper_eval/paper310/bin/python" \
  "$LAUNCHER" stage_b run
