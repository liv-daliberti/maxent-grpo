#!/bin/bash
# Advance E76 only after the completed stage passes its frozen selector.
#SBATCH --job-name=e76ctl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --partition=cs
#SBATCH --account=allcs
#SBATCH --time=00:30:00
#SBATCH --output=var/artifacts/logs/%x-%j.out
#SBATCH --error=var/artifacts/logs/%x-%j.err

set -euo pipefail

ROOT_DIR="${E76_REPO_ROOT:?E76_REPO_ROOT is required}"
STAGE="${E76_COMPLETED_STAGE:?E76_COMPLETED_STAGE is required}"
SNAPSHOT_ROOT="${E76_SNAPSHOT_ROOT:?E76_SNAPSHOT_ROOT is required}"
OPS_ROOT="${E76_OPS_SNAPSHOT_ROOT:?E76_OPS_SNAPSHOT_ROOT is required}"
PYTHON_BIN="${ROOT_DIR}/var/seed_paper_eval/paper310/bin/python"

cd "$ROOT_DIR"
mkdir -p var/artifacts/logs

case "$STAGE" in
  a) NEXT_STAGE=b ;;
  b) NEXT_STAGE=c ;;
  *) echo "E76 controller cannot advance completed stage: $STAGE" >&2; exit 2 ;;
esac

export E76_REPO_ROOT="$ROOT_DIR"
"$PYTHON_BIN" "$OPS_ROOT/exp_scaling/select_e76_tuned_scale.py" --stage "$STAGE"

controller_args=()
if [[ "$NEXT_STAGE" != c ]]; then
  controller_args+=(--queue-controller)
fi
"$PYTHON_BIN" "$OPS_ROOT/exp_scaling/launch_e76_tuned_scale.py" \
  --stage "$NEXT_STAGE" --submit --snapshot-root "$SNAPSHOT_ROOT" \
  "${controller_args[@]}"
