#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=all
#SBATCH --account=mltheory
#SBATCH --time=3-00:00:00
#SBATCH --output=var/artifacts/logs/%x-%j.out
#SBATCH --error=var/artifacts/logs/%x-%j.err

set -euo pipefail
ROOT_DIR="${ROOT_DIR:?ROOT_DIR is required}"
SOURCE_ROOT="${OAT_ZERO_SOURCE_ROOT:?source root is required}"
EXECUTION_ROOT="${OAT_ZERO_EXECUTION_ROOT:?execution root is required}"
ARM="${OAT_ZERO_ARM:?arm is required}"
SEED="${OAT_ZERO_SEED:?seed is required}"
SOURCE_HASH="${OAT_ZERO_SOURCE_HASH:?source hash is required}"
EXECUTION_HASH="${OAT_ZERO_EXECUTION_HASH:?execution hash is required}"
VARIANT="${OAT_ZERO_POINT_STAGE_B_VARIANT:-standard}"
[[ "$VARIANT" == "standard" || "$VARIANT" == "geometry_shift" ]] || { echo "invalid PointMaze Stage-B variant" >&2; exit 1; }
source "$ROOT_DIR/ops/repo_env.sh"
export PYTHONPATH="$EXECUTION_ROOT:$SOURCE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OAT_ZERO_REPO_ROOT="$ROOT_DIR" OAT_ZERO_SOURCE_ROOT="$SOURCE_ROOT"
export OAT_ZERO_EXECUTION_ROOT="$EXECUTION_ROOT"
export OAT_ZERO_MAZE_WORKER_PYTHON="$ROOT_DIR/var/maze_runtime/venv/bin/python"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false MUJOCO_GL=egl

ARTIFACT_STEM="point_maze_stage_b_05b_12pass"
DATA_ROOT="$ROOT_DIR/var/data/point_maze_modebench_v1"
PROTOCOL="point_maze_stage_b_05b_12pass_v3_20260730.md"
QUALIFICATION="$ROOT_DIR/var/artifacts/point_maze_interactive_paired_smoke_v3_audit.json"
IDENTITY="$ROOT_DIR/var/artifacts/point_maze_stage_b_05b_12pass_identity.json"
if [[ "$VARIANT" == "geometry_shift" ]]; then
  ARTIFACT_STEM="point_maze_geometry_shift_stage_b_05b_12pass"
  DATA_ROOT="$ROOT_DIR/var/data/point_maze_geometry_shift_v1"
  PROTOCOL="point_maze_geometry_shift_stage_b_05b_12pass_v1_20260730.md"
  QUALIFICATION="$ROOT_DIR/var/artifacts/point_maze_geometry_shift_paired_smoke_v1_audit.json"
  IDENTITY="$ROOT_DIR/var/artifacts/point_maze_geometry_shift_stage_b_05b_12pass_identity.json"
fi
STEM="$ROOT_DIR/var/artifacts/${ARTIFACT_STEM}_${ARM}_s${SEED}"
exec "$ROOT_DIR/var/seed_paper_eval/paper310/bin/python" \
  "$EXECUTION_ROOT/train_point_maze_stage_b_05b_12pass.py" \
  --arm "$ARM" --seed "$SEED" \
  --model "$ROOT_DIR/var/models/point_maze_interactive_warmstart_v3" \
  --data-root "$DATA_ROOT" \
  --train-root "$DATA_ROOT/train" \
  --evaluation-root "$DATA_ROOT/eval" \
  --worker-python "$OAT_ZERO_MAZE_WORKER_PYTHON" \
  --protocol "$EXECUTION_ROOT/$PROTOCOL" \
  --qualification-audit "$QUALIFICATION" \
  --identity "$IDENTITY" \
  --output-receipt "${STEM}.json" --output-metrics "${STEM}.metrics.jsonl" \
  --output-replay "${STEM}.state_replay.jsonl" \
  --learning-rate 0.0000002 --microbatch-size 16 --evaluation-batch-size 16 \
  --max-length 1536 --job-id "${SLURM_JOB_ID:?job ID is required}" \
  --source-hash "$SOURCE_HASH" --execution-hash "$EXECUTION_HASH"
