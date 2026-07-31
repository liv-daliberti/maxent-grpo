#!/bin/bash
#SBATCH --job-name=antmaze-v9-crossnode
#SBATCH --nodes=3
#SBATCH --ntasks=3
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --partition=all
#SBATCH --account=allcs
#SBATCH --time=02:00:00
#SBATCH --output=var/artifacts/logs/%x-%j.out
#SBATCH --error=var/artifacts/logs/%x-%j.err

set -euo pipefail

ROOT_DIR="${ROOT_DIR:?ROOT_DIR is required}"
SOURCE_ROOT="${OAT_ZERO_SOURCE_ROOT:?OAT_ZERO_SOURCE_ROOT is required}"
EXECUTION_ROOT="${OAT_ZERO_EXECUTION_ROOT:?OAT_ZERO_EXECUTION_ROOT is required}"
SOURCE_HASH="${OAT_ZERO_SOURCE_HASH:?OAT_ZERO_SOURCE_HASH is required}"
EXECUTION_HASH="${OAT_ZERO_EXECUTION_HASH:?OAT_ZERO_EXECUTION_HASH is required}"
ROUTE_IDENTITY="${OAT_ZERO_ROUTE_IDENTITY:?OAT_ZERO_ROUTE_IDENTITY is required}"
SPECS="${OAT_ZERO_CROSS_NODE_SPECS:?OAT_ZERO_CROSS_NODE_SPECS is required}"
PROTOCOL="$EXECUTION_ROOT/ant_maze_v9_cross_node_audit_20260729.md"
PREFIX="$ROOT_DIR/var/artifacts/ant_maze_v9_cross_node_v1"

source "$ROOT_DIR/ops/repo_env.sh"
export OAT_ZERO_REPO_ROOT="$ROOT_DIR"
export OAT_ZERO_SOURCE_ROOT="$SOURCE_ROOT"
export OAT_ZERO_PROTOCOL_IDENTITY="$ROUTE_IDENTITY"
export PYTHONPATH="$SOURCE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MUJOCO_GL=egl
export ROOT_DIR SOURCE_ROOT EXECUTION_ROOT SOURCE_HASH EXECUTION_HASH PROTOCOL PREFIX SPECS

srun --ntasks=3 --ntasks-per-node=1 bash -lc '
  set -euo pipefail
  exec "$ROOT_DIR/var/maze_runtime/venv/bin/python" \
    "$EXECUTION_ROOT/audit_ant_maze_cross_node_v9.py" \
    --specs "$SPECS" \
    --output "${PREFIX}_replica_${SLURM_PROCID}.json" \
    --protocol "$PROTOCOL" \
    --replica "$SLURM_PROCID" \
    --repetitions 3 \
    --source-hash "$SOURCE_HASH" \
    --execution-hash "$EXECUTION_HASH" \
    --job-id "$SLURM_JOB_ID"
'

exec "$ROOT_DIR/var/seed_paper_eval/paper310/bin/python" \
  "$EXECUTION_ROOT/aggregate_ant_maze_cross_node_v9.py" \
  --replicas \
    "${PREFIX}_replica_0.json" \
    "${PREFIX}_replica_1.json" \
    "${PREFIX}_replica_2.json" \
  --output "$ROOT_DIR/var/artifacts/ant_maze_v9_cross_node_v1.json" \
  --protocol "$PROTOCOL" \
  --specs "$SPECS" \
  --source-hash "$SOURCE_HASH" \
  --execution-hash "$EXECUTION_HASH" \
  --job-id "$SLURM_JOB_ID"
