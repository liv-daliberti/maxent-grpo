#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=all
#SBATCH --account=mltheory
#SBATCH --time=12:00:00
#SBATCH --output=var/artifacts/logs/%x-%j.out
#SBATCH --error=var/artifacts/logs/%x-%j.err

set -euo pipefail
ROOT_DIR="${ROOT_DIR:?ROOT_DIR is required}"
SOURCE_ROOT="${OAT_ZERO_SOURCE_ROOT:?source root is required}"
EXECUTION_ROOT="${OAT_ZERO_EXECUTION_ROOT:?execution root is required}"
ARM="${OAT_ZERO_ARM:?arm is required}"
SOURCE_HASH="${OAT_ZERO_SOURCE_HASH:?source hash is required}"
EXECUTION_HASH="${OAT_ZERO_EXECUTION_HASH:?execution hash is required}"
COHORT="${OAT_ZERO_POINT_SMOKE_COHORT:-v1}"
SEED="${OAT_ZERO_POINT_SMOKE_SEED:-75301}"
[[ "$COHORT" == "v1" || "$COHORT" == "v2" || "$COHORT" == "v3" || "$COHORT" == "v4" ]] || { echo "invalid PointMaze smoke cohort" >&2; exit 1; }
MICROBATCH=4
[[ "$COHORT" == "v3" || "$COHORT" == "v4" ]] && MICROBATCH=16
STEM="point_maze_interactive_paired_smoke_${COHORT}"
DATA_ROOT="$ROOT_DIR/var/data/point_maze_modebench_v1"
VIABILITY="$ROOT_DIR/var/artifacts/point_maze_interactive_05b_viability_warmstart_v3.json"
ADMISSION="$ROOT_DIR/var/artifacts/point_maze_modebench_v1_admission_audit.json"
PROTOCOL="point_maze_interactive_paired_smoke_${COHORT}_20260730.md"
if [[ "$COHORT" == "v4" ]]; then
  STEM="point_maze_geometry_shift_paired_smoke_v1"
  DATA_ROOT="$ROOT_DIR/var/data/point_maze_geometry_shift_v1"
  VIABILITY="$ROOT_DIR/var/artifacts/point_maze_geometry_shift_05b_viability_v1.json"
  ADMISSION="$ROOT_DIR/var/artifacts/point_maze_geometry_shift_v1_admission_audit.json"
  PROTOCOL="point_maze_geometry_shift_paired_smoke_v1_20260730.md"
fi
source "$ROOT_DIR/ops/repo_env.sh"
export PYTHONPATH="$EXECUTION_ROOT:$SOURCE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OAT_ZERO_REPO_ROOT="$ROOT_DIR" OAT_ZERO_SOURCE_ROOT="$SOURCE_ROOT"
export OAT_ZERO_MAZE_WORKER_PYTHON="$ROOT_DIR/var/maze_runtime/venv/bin/python"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false MUJOCO_GL=egl

exec "$ROOT_DIR/var/seed_paper_eval/paper310/bin/python" \
  "$EXECUTION_ROOT/train_point_maze_interactive_paired_smoke_v1.py" \
  --arm "$ARM" \
  --model "$ROOT_DIR/var/models/point_maze_interactive_warmstart_v3" \
  --data-root "$DATA_ROOT" \
  --data-split-root "$DATA_ROOT/train" \
  --worker-python "$OAT_ZERO_MAZE_WORKER_PYTHON" \
  --protocol "$EXECUTION_ROOT/$PROTOCOL" \
  --viability-receipt "$VIABILITY" \
  --admission-audit "$ADMISSION" \
  --warmstart-receipt "$ROOT_DIR/var/artifacts/point_maze_interactive_warmstart_sft_v3.json" \
  --identity "$ROOT_DIR/var/artifacts/${STEM}_identity.json" \
  --output-receipt "$ROOT_DIR/var/artifacts/${STEM}_${ARM}.json" \
  --output-metrics "$ROOT_DIR/var/artifacts/${STEM}_${ARM}.metrics.jsonl" \
  --output-replay "$ROOT_DIR/var/artifacts/${STEM}_${ARM}.state_replay.jsonl" \
  --seed "$SEED" --learning-rate 0.0000002 --microbatch-size "$MICROBATCH" --max-length 1536 \
  --job-id "${SLURM_JOB_ID:?job ID is required}" \
  --source-hash "$SOURCE_HASH" --execution-hash "$EXECUTION_HASH"
