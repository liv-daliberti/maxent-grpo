#!/bin/bash
#SBATCH --job-name=constructive_code_isolation
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --partition=all
#SBATCH --time=00:15:00
#SBATCH --output=var/artifacts/logs/constructive-code-isolation-%j.out
#SBATCH --error=var/artifacts/logs/constructive-code-isolation-%j.err

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
IMAGE="$ROOT_DIR/var/images/python-3.10-slim-c1e4e6c01eb4.sqsh"
SCRATCH_ROOT="${SLURM_TMPDIR:-/tmp}/constructive-code-isolation-$SLURM_JOB_ID"
LAUNCHER="$SCRATCH_ROOT/constructive_code_sandbox"
RUNTIME_ROOT="$SCRATCH_ROOT/python-root"

if [[ ! -f "$IMAGE" ]]; then
  echo "missing pinned SquashFS runtime: $IMAGE" >&2
  exit 1
fi
mkdir -p "$SCRATCH_ROOT"

exec /n/fs/similarity/maxent-grpo/var/seed_paper_eval/paper310/bin/python \
  "$ROOT_DIR/ops/smoke_constructive_code_isolation.py" \
  --image "$IMAGE" \
  --launcher "$LAUNCHER" \
  --runtime-root "$RUNTIME_ROOT" \
  --output "$ROOT_DIR/var/artifacts/constructive_code_isolation_smoke.json"
