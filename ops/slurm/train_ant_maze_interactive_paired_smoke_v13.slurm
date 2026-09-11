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
COHORT="${OAT_ZERO_ANT_SMOKE_COHORT:-v13}"
[[ "$COHORT" == "v13" || "$COHORT" == "v13r1" || "$COHORT" == "v13r2" ]] || { echo "invalid AntMaze smoke cohort" >&2; exit 1; }
MICROBATCH=4
[[ "$COHORT" == "v13r2" ]] && MICROBATCH=16
PROTOCOL_NAME="ant_maze_interactive_paired_smoke_v13_20260730.md"
[[ "$COHORT" == "v13r2" ]] && PROTOCOL_NAME="ant_maze_interactive_paired_smoke_v13r2_20260730.md"
source "$ROOT_DIR/ops/repo_env.sh"
export PYTHONPATH="$EXECUTION_ROOT:$SOURCE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OAT_ZERO_REPO_ROOT="$ROOT_DIR" OAT_ZERO_SOURCE_ROOT="$SOURCE_ROOT"
export OAT_ZERO_EXECUTION_ROOT="$EXECUTION_ROOT"
export OAT_ZERO_MAZE_WORKER_PYTHON="$ROOT_DIR/var/maze_runtime/venv/bin/python"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false MUJOCO_GL=egl
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

STEM="$ROOT_DIR/var/artifacts/ant_maze_interactive_paired_smoke_${COHORT}_${ARM}"
exec "$ROOT_DIR/var/seed_paper_eval/paper310/bin/python" \
  "$EXECUTION_ROOT/train_ant_maze_interactive_paired_smoke_v13.py" \
  --arm "$ARM" --model "$ROOT_DIR/var/models/ant_maze_interactive_warmstart_v13" \
  --data-root "$ROOT_DIR/var/data/ant_maze_modebench_v12" \
  --train-root "$ROOT_DIR/var/data/ant_maze_modebench_v12/train" \
  --worker-python "$OAT_ZERO_MAZE_WORKER_PYTHON" \
  --protocol "$EXECUTION_ROOT/$PROTOCOL_NAME" \
  --viability-receipt "$ROOT_DIR/var/artifacts/ant_maze_interactive_05b_viability_v13.json" \
  --admission-audit "$ROOT_DIR/var/artifacts/ant_maze_modebench_v12_admission_audit.json" \
  --warmstart-receipt "$ROOT_DIR/var/artifacts/ant_maze_interactive_warmstart_sft_v13.json" \
  --identity "$ROOT_DIR/var/artifacts/ant_maze_interactive_paired_smoke_${COHORT}_identity.json" \
  --output-receipt "${STEM}.json" --output-metrics "${STEM}.metrics.jsonl" \
  --output-replay "${STEM}.state_replay.jsonl" \
  --seed 76313 --learning-rate 0.0000002 --microbatch-size "$MICROBATCH" --max-length 1536 \
  --job-id "${SLURM_JOB_ID:?job ID is required}" \
  --source-hash "$SOURCE_HASH" --execution-hash "$EXECUTION_HASH"
