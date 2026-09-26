#!/bin/bash
#SBATCH --job-name=constructive_code_v2_gate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --partition=all
#SBATCH --time=04:00:00
#SBATCH --output=var/artifacts/logs/constructive-code-v2-gate-%j.out
#SBATCH --error=var/artifacts/logs/constructive-code-v2-gate-%j.err

set -euo pipefail
ROOT_DIR="${ROOT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
OPS_ROOT="${OAT_ZERO_OPS_ROOT:?missing OAT_ZERO_OPS_ROOT}"
SOURCE_ROOT="${OAT_ZERO_SOURCE_ROOT:?missing OAT_ZERO_SOURCE_ROOT}"
TESTLIB_ROOT="${OAT_ZERO_TESTLIB_ROOT:?missing OAT_ZERO_TESTLIB_ROOT}"
PROTOCOL="${OAT_ZERO_CONSTRUCTIVE_V2_PROTOCOL:?missing protocol snapshot}"
SANDBOX_SOURCE="${OAT_ZERO_SANDBOX_SOURCE:?missing sandbox source}"
SLATE_ROOT="${OAT_ZERO_CONSTRUCTIVE_V2_SLATE:?missing frozen v2 slate}"
V1_ROOT="${OAT_ZERO_CONSTRUCTIVE_V2_V1_ROOT:?missing frozen v1 ledger}"
INDEX="${OAT_ZERO_CONSTRUCTIVE_V2_INDEX:?missing frozen candidate index}"
IDENTITY="${OAT_ZERO_PROTOCOL_IDENTITY:?missing protocol identity}"
for required in \
  "$IDENTITY" "$PROTOCOL" "$SANDBOX_SOURCE" "$TESTLIB_ROOT/testlib.h" \
  "$SOURCE_ROOT/oat_drgrpo/__init__.py" \
  "$OPS_ROOT/replay_constructive_code_v2.py" "$SLATE_ROOT/manifest.json" "$V1_ROOT/manifest.json" "$INDEX"; do
  [[ -f "$required" ]] || { echo "Missing frozen ConstructiveCode v2 input: $required" >&2; exit 1; }
done
export OAT_ZERO_REPO_ROOT="$ROOT_DIR"
export OAT_ZERO_SOURCE_ROOT="$SOURCE_ROOT"
export OAT_ZERO_TESTLIB_ROOT="$TESTLIB_ROOT"
export OAT_ZERO_CONSTRUCTIVE_V2_PROTOCOL="$PROTOCOL"
export OAT_ZERO_SANDBOX_SOURCE="$SANDBOX_SOURCE"
export PYTHONPATH="$SOURCE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
RUN_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/constructive-code-v2-${SLURM_JOB_ID}.XXXXXX")"

exec "$ROOT_DIR/var/seed_paper_eval/paper310/bin/python" \
  "$OPS_ROOT/replay_constructive_code_v2.py" \
  --slate-root "$SLATE_ROOT" \
  --v1-root "$V1_ROOT" \
  --runtime-root "$RUN_ROOT/runtime" \
  --launcher "$RUN_ROOT/constructive_code_sandbox" \
  --build-root "$RUN_ROOT/checkers" \
  --scratch-root "$RUN_ROOT/scratch" \
  --workers "${SLURM_CPUS_PER_TASK:-16}"
