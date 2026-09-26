#!/bin/bash
#SBATCH --job-name=audit-ant-paired-v13
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=cs
#SBATCH --time=12:00:00
#SBATCH --output=var/artifacts/logs/%x-%j.out
#SBATCH --error=var/artifacts/logs/%x-%j.err

set -euo pipefail
ROOT_DIR="${ROOT_DIR:?ROOT_DIR is required}"
SOURCE_ROOT="${OAT_ZERO_SOURCE_ROOT:?source root is required}"
EXECUTION_ROOT="${OAT_ZERO_EXECUTION_ROOT:?execution root is required}"
COHORT="${OAT_ZERO_ANT_SMOKE_COHORT:-v13}"
[[ "$COHORT" == "v13" || "$COHORT" == "v13r1" || "$COHORT" == "v13r2" ]] || { echo "invalid AntMaze smoke cohort" >&2; exit 1; }
STEM="ant_maze_interactive_paired_smoke_${COHORT}"
export PYTHONPATH="$EXECUTION_ROOT:$SOURCE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OAT_ZERO_REPO_ROOT="$ROOT_DIR" OAT_ZERO_SOURCE_ROOT="$SOURCE_ROOT"
export OAT_ZERO_EXECUTION_ROOT="$EXECUTION_ROOT" MUJOCO_GL=egl
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
exec "$ROOT_DIR/var/seed_paper_eval/paper310/bin/python" \
  "$EXECUTION_ROOT/audit_ant_maze_interactive_paired_smoke_v13.py" \
  --repo-root "$ROOT_DIR" \
  --identity "$ROOT_DIR/var/artifacts/${STEM}_identity.json" \
  --manifest "$ROOT_DIR/var/artifacts/${STEM}_jobs.tsv" \
  --model "$ROOT_DIR/var/models/ant_maze_interactive_warmstart_v13" \
  --train-root "$ROOT_DIR/var/data/ant_maze_modebench_v12/train" \
  --worker-python "$ROOT_DIR/var/maze_runtime/venv/bin/python" \
  --output "$ROOT_DIR/var/artifacts/${STEM}_audit.json"
