#!/bin/bash
#SBATCH --job-name=audit-point-stage-b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --partition=cs
#SBATCH --time=12:00:00
#SBATCH --output=var/artifacts/logs/%x-%j.out
#SBATCH --error=var/artifacts/logs/%x-%j.err

set -euo pipefail
ROOT_DIR="${ROOT_DIR:?ROOT_DIR is required}"
EXECUTION_ROOT="${OAT_ZERO_EXECUTION_ROOT:?execution root is required}"
SOURCE_ROOT="${OAT_ZERO_SOURCE_ROOT:?source root is required}"
VARIANT="${OAT_ZERO_POINT_STAGE_B_VARIANT:-standard}"
[[ "$VARIANT" == "standard" || "$VARIANT" == "geometry_shift" ]] || { echo "invalid PointMaze Stage-B variant" >&2; exit 1; }
STEM="point_maze_stage_b_05b_12pass"
if [[ "$VARIANT" == "geometry_shift" ]]; then STEM="point_maze_geometry_shift_stage_b_05b_12pass"; fi
export PYTHONPATH="$EXECUTION_ROOT:$SOURCE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OAT_ZERO_SOURCE_ROOT="$SOURCE_ROOT" MUJOCO_GL=egl
exec "$ROOT_DIR/var/seed_paper_eval/paper310/bin/python" \
  "$EXECUTION_ROOT/audit_point_maze_stage_b_05b_12pass.py" \
  --repo-root "$ROOT_DIR" \
  --identity "$ROOT_DIR/var/artifacts/${STEM}_identity.json" \
  --submission "$ROOT_DIR/var/artifacts/${STEM}_submission.json" \
  --manifest "$ROOT_DIR/var/artifacts/${STEM}_jobs.tsv" \
  --output "$ROOT_DIR/var/artifacts/${STEM}_audit.json"
