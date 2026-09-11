#!/bin/bash
#SBATCH --job-name=prepare-constructive-v5-paired
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --partition=all
#SBATCH --account=allcs
#SBATCH --time=00:15:00
#SBATCH --output=var/artifacts/logs/%x-%j.out
#SBATCH --error=var/artifacts/logs/%x-%j.err

set -euo pipefail
ROOT_DIR="${ROOT_DIR:?ROOT_DIR is required}"
EXPECTED="${OAT_ZERO_EXPECTED_DEPENDENCY_LAUNCHER_SHA256:?dependency launcher hash is required}"
LAUNCHER="$ROOT_DIR/ops/exp_scaling/launch_constructive_code_v5_paired_dependency.sh"
[[ "$(sha256sum "$LAUNCHER" | cut -d' ' -f1)" == "$EXPECTED" ]] || {
  echo "ConstructiveCode paired dependency launcher hash drift" >&2
  exit 1
}
exec bash "$LAUNCHER" run
