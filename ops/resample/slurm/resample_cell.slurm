#!/bin/bash
# Re-measure one terminal cell's mode coverage under disjoint sampling streams.
#
#   sbatch --gres=gpu:a100:1 --export=ALL,CELL=205,MODE=reproduce \
#          ops/resample/slurm/resample_cell.slurm
#
# The GPU model is chosen by the caller because each cell must run on the model
# it trained on; the worker refuses to start if the allocation disagrees.

#SBATCH --job-name=pmd-resample
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --partition=lowprio
#SBATCH --output=var/artifacts/pmd_independent_resample_20260915/logs/%x-%j.out
#SBATCH --error=var/artifacts/pmd_independent_resample_20260915/logs/%x-%j.err

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$ROOT_DIR"

RUN_ROOT="$ROOT_DIR/var/artifacts/pmd_independent_resample_20260915"
# A self-consistency check runs the same cell twice; let it write elsewhere so it
# never collides with, or overwrites, the receipt the cohort actually uses.
RECEIPT_ROOT="${RECEIPT_ROOT:-$RUN_ROOT/receipts}"
PYTHON="$ROOT_DIR/var/seed_paper_eval/paper310/bin/python"
CELL_INDEX="${CELL:-${SLURM_ARRAY_TASK_ID:?CELL or array task id required}}"

# oat pulls in launchpad/courier, whose extension modules link against the
# interpreter's shared library; the training jobs put it on the loader path too.
export LD_LIBRARY_PATH="$ROOT_DIR/var/seed_paper_eval/paper310/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# The child-seed expansion the whole correction depends on is a V0 behaviour.
export VLLM_USE_V1=0
export HF_HUB_OFFLINE=0
export TOKENIZERS_PARALLELISM=false
# Several of these run concurrently on one node, and the engine still brings up
# a single-rank distributed group. Left to the default port they collide with
# EADDRINUSE, so derive a per-job port from the job id.
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$(( 20000 + (${SLURM_JOB_ID:-$$} % 40000) ))
export VLLM_PORT=$(( 20000 + ((${SLURM_JOB_ID:-$$} + 7919) % 40000) ))
export HF_HOME="${HF_HOME:-$ROOT_DIR/var/cache/huggingface}"
# Weights and engine scratch stay on node-local disk: this is high-volume I/O
# that has no business touching shared storage.
export STAGING="${STAGING:-/tmp/pmd_resample_${SLURM_JOB_ID:-$$}}"
mkdir -p "$STAGING" "$RUN_ROOT/logs"
trap 'rm -rf "$STAGING"' EXIT

"$PYTHON" -B ops/resample/resample_terminal_mode_coverage.py \
  --manifest "${MANIFEST:-$RUN_ROOT/cohort_manifest.json}" \
  --cell "$CELL_INDEX" \
  --mode "${MODE:-independent}" \
  --output-root "$RECEIPT_ROOT" \
  --staging "$STAGING"
