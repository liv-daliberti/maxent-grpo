#!/usr/bin/env bash
# CPU-only E49E trace certification on the node105-reachable private fabric.
#SBATCH --job-name=e49e-trace-banks
#SBATCH --partition=mltheory
#SBATCH --account=mltheory
#SBATCH --nodelist=node302
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00

set -euo pipefail

ROOT_DIR="${OAT_ZERO_REPO_ROOT:-${SLURM_SUBMIT_DIR:-}}"
STAGE="${E49E_STAGE:-}"
SNAPSHOT_ROOT="${E49E_SOURCE_SNAPSHOT:-}"
SNAPSHOT_HASH="${E49E_SOURCE_HASH:-}"
FROZEN_INPUT="${E49E_FROZEN_INPUT:-}"
ENDPOINT="${E49E_FROZEN_ENDPOINT:-}"
IDENTITY="${E49E_FROZEN_IDENTITY:-}"
IDENTITY_HASH="${E49E_FROZEN_IDENTITY_HASH:-}"
CONTROLS="${E49E_KNOWN_INVALID_CONTROLS:-}"
EQUIVALENT_CONTROLS="${E49E_KNOWN_EQUIVALENT_CONTROLS:-}"
INHERITED_CACHE_COUNT="${E49E_INHERITED_CACHE_COUNT:-}"
INHERITED_CACHE_HASH="${E49E_INHERITED_CACHE_HASH:-}"
if [[ "$ROOT_DIR" != "/n/fs/similarity/maxent-grpo" ]] \
  || [[ "$STAGE" != toy && "$STAGE" != full ]] \
  || [[ "$(hostname -s)" != node302 ]] \
  || [[ -z "$SNAPSHOT_ROOT" || -z "$SNAPSHOT_HASH" ]] \
  || [[ -z "$FROZEN_INPUT" || -z "$ENDPOINT" ]] \
  || [[ -z "$IDENTITY" || -z "$IDENTITY_HASH" ]] \
  || [[ -z "$CONTROLS" || -z "$EQUIVALENT_CONTROLS" ]] \
  || [[ ! "$INHERITED_CACHE_COUNT" =~ ^[0-9]+$ ]] \
  || [[ -z "$INHERITED_CACHE_HASH" ]]; then
  echo "E49E trace certification requires stage toy/full on node302" >&2
  exit 2
fi

source "$ROOT_DIR/ops/repo_env.sh"
export PYTHONDONTWRITEBYTECODE=1
PYTHON_BIN="$ROOT_DIR/var/seed_paper_eval/paper310/bin/python"
export PYTHONPATH="$SNAPSHOT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$ROOT_DIR/var/seed_paper_eval/paper310/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
MATERIALIZER="$SNAPSHOT_ROOT/ops/math_strategy_calibration/materialize_e49e_trace_bank_data.py"

if [[ "$STAGE" == toy ]]; then
  SOURCE="$ROOT_DIR/var/data/e49b_math_strategy_toy"
  OUTPUT="$ROOT_DIR/var/data/e49e_trace_bank_math_toy"
  EVIDENCE="$ROOT_DIR/var/artifacts/e49e_trace_bank_math_toy_v1"
else
  SOURCE="$ROOT_DIR/var/data/math12k_384_math500"
  OUTPUT="$ROOT_DIR/var/data/e49e_trace_bank_math_full"
  EVIDENCE="$ROOT_DIR/var/artifacts/e49e_trace_bank_math_full_v1"
fi
CACHE_ROOT="$EVIDENCE/request_cache"

hash_tree() {
  "$PYTHON_BIN" - "$1" <<'PY'
import hashlib
import pathlib
import sys
root = pathlib.Path(sys.argv[1])
digest = hashlib.sha256()
for item in sorted(path for path in root.rglob("*") if path.is_file()):
    digest.update(str(item.relative_to(root)).encode("utf-8"))
    digest.update(b"\0")
    digest.update(hashlib.sha256(item.read_bytes()).digest())
print(digest.hexdigest())
PY
}

if [[ ! -f "$MATERIALIZER" ]] \
  || [[ ! -f "$ENDPOINT" ]] \
  || [[ ! -f "$FROZEN_INPUT/menu_records.jsonl" ]] \
  || [[ ! -f "$CONTROLS" ]] \
  || [[ ! -f "$EQUIVALENT_CONTROLS" ]] \
  || [[ ! -f "$IDENTITY" ]] \
  || [[ "$(hash_tree "$SNAPSHOT_ROOT")" != "$SNAPSHOT_HASH" ]] \
  || [[ "$(find "$CACHE_ROOT" -type f -name '*.json' | wc -l | tr -d ' ')" != "$INHERITED_CACHE_COUNT" ]] \
  || [[ "$(hash_tree "$CACHE_ROOT")" != "$INHERITED_CACHE_HASH" ]] \
  || [[ "$(sha256sum "$IDENTITY" | cut -d' ' -f1)" != "$IDENTITY_HASH" ]]; then
  echo "E49E frozen implementation or evidence identity failed" >&2
  exit 3
fi

exec "$PYTHON_BIN" "$MATERIALIZER" \
  --source "$SOURCE" \
  --input-evidence "$FROZEN_INPUT" \
  --output "$OUTPUT" \
  --evidence "$EVIDENCE" \
  --endpoint "$ENDPOINT" \
  --known-invalid-controls "$CONTROLS" \
  --known-equivalent-controls "$EQUIVALENT_CONTROLS" \
  --workers 8 --timeout 900
