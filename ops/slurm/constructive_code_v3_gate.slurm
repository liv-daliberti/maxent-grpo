#!/bin/bash
#SBATCH --job-name=constructive_code_v3_gate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --partition=all
#SBATCH --time=12:00:00
#SBATCH --output=var/artifacts/logs/constructive-code-v3-gate-%j.out
#SBATCH --error=var/artifacts/logs/constructive-code-v3-gate-%j.err

set -euo pipefail

ROOT_DIR="${ROOT_DIR:?missing ROOT_DIR}"
OPS_ROOT="${OAT_ZERO_OPS_ROOT:?missing OAT_ZERO_OPS_ROOT}"
SOURCE_ROOT="${OAT_ZERO_SOURCE_ROOT:?missing OAT_ZERO_SOURCE_ROOT}"
TESTLIB_ROOT="${OAT_ZERO_TESTLIB_ROOT:?missing OAT_ZERO_TESTLIB_ROOT}"
PROTOCOL="${OAT_ZERO_CONSTRUCTIVE_V3_PROTOCOL:?missing v3 protocol}"
INDEX="${OAT_ZERO_CONSTRUCTIVE_V3_INDEX:?missing candidate index}"
V1_ROOT="${OAT_ZERO_CONSTRUCTIVE_V3_V1_ROOT:?missing v1 exclusion ledger}"
SANDBOX_SOURCE="${OAT_ZERO_SANDBOX_SOURCE:?missing sandbox source}"
IDENTITY="${OAT_ZERO_PROTOCOL_IDENTITY:?missing protocol identity}"
PYTHON_BIN="$ROOT_DIR/var/seed_paper_eval/paper310/bin/python"
SLATE_ROOT="$ROOT_DIR/var/data/constructive_code_v3"

for required in \
  "$PYTHON_BIN" "$IDENTITY" "$PROTOCOL" "$INDEX" "$V1_ROOT/manifest.json" \
  "$TESTLIB_ROOT/testlib.h" "$SANDBOX_SOURCE" \
  "$OPS_ROOT/materialize_constructive_code_v3.py" \
  "$OPS_ROOT/replay_constructive_code_v3.py" \
  "$SOURCE_ROOT/oat_drgrpo/__init__.py"; do
  [[ -e "$required" ]] || {
    echo "Missing frozen ConstructiveCode v3 input: $required" >&2
    exit 1
  }
done

export OAT_ZERO_REPO_ROOT="$ROOT_DIR"
export OAT_ZERO_SOURCE_ROOT="$SOURCE_ROOT"
export OAT_ZERO_TESTLIB_ROOT="$TESTLIB_ROOT"
export OAT_ZERO_CONSTRUCTIVE_V3_PROTOCOL="$PROTOCOL"
export OAT_ZERO_CONSTRUCTIVE_V3_INDEX="$INDEX"
export OAT_ZERO_CONSTRUCTIVE_V3_V1_ROOT="$V1_ROOT"
export OAT_ZERO_SANDBOX_SOURCE="$SANDBOX_SOURCE"
export PYTHONPATH="$OPS_ROOT:$SOURCE_ROOT${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON_BIN" "$OPS_ROOT/materialize_constructive_code_v3.py" \
  --index "$INDEX" \
  --v1-root "$V1_ROOT" \
  --output "$SLATE_ROOT"

RUN_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/constructive-code-v3-${SLURM_JOB_ID}.XXXXXX")"
exec "$PYTHON_BIN" "$OPS_ROOT/replay_constructive_code_v3.py" \
  --slate-root "$SLATE_ROOT" \
  --v1-root "$V1_ROOT" \
  --runtime-root "$RUN_ROOT/runtime" \
  --launcher "$RUN_ROOT/constructive_code_sandbox" \
  --build-root "$RUN_ROOT/checkers" \
  --scratch-root "$RUN_ROOT/scratch" \
  --workers "${SLURM_CPUS_PER_TASK:-16}"
