#!/usr/bin/env bash
#SBATCH --job-name=e49e-repair-singletons-v3
#SBATCH --partition=mltheory
#SBATCH --account=mltheory
#SBATCH --nodelist=node915
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=1-00:00:00

set -euo pipefail
ROOT_DIR="${OAT_ZERO_REPO_ROOT:-${SLURM_SUBMIT_DIR:-}}"
SNAPSHOT_ROOT="${E49E_REPAIR_SOURCE_SNAPSHOT:-}"
SNAPSHOT_HASH="${E49E_REPAIR_SOURCE_HASH:-}"
FROZEN_E49D="${E49E_REPAIR_FROZEN_E49D:-}"
FROZEN_E49E="${E49E_REPAIR_FROZEN_E49E:-}"
FROZEN_AUGMENT="${E49E_REPAIR_FROZEN_AUGMENT:-}"
ENDPOINT="${E49E_REPAIR_ENDPOINT:-}"
INVALID="${E49E_REPAIR_INVALID_CONTROLS:-}"
EQUIVALENT="${E49E_REPAIR_EQUIVALENT_CONTROLS:-}"
PRIOR="${E49E_REPAIR_PRIOR_V2B:-}"
CONTRACTS="${E49E_REPAIR_CURATED_CONTRACTS:-}"
IDENTITY="${E49E_REPAIR_IDENTITY:-}"
IDENTITY_HASH="${E49E_REPAIR_IDENTITY_HASH:-}"
if [[ "$ROOT_DIR" != "/n/fs/similarity/maxent-grpo" ]] \
  || [[ "$(hostname -s)" != node915 ]] \
  || [[ -z "$SNAPSHOT_ROOT" || -z "$SNAPSHOT_HASH" ]] \
  || [[ -z "$FROZEN_E49D" || -z "$FROZEN_E49E" || -z "$FROZEN_AUGMENT" ]] \
  || [[ -z "$ENDPOINT" || -z "$INVALID" || -z "$EQUIVALENT" ]] \
  || [[ -z "$PRIOR" || -z "$CONTRACTS" ]] \
  || [[ -z "$IDENTITY" || -z "$IDENTITY_HASH" ]]; then
  echo "invalid E49E curated-repair V3 environment" >&2
  exit 2
fi

source "$ROOT_DIR/ops/repo_env.sh"
PYTHON_BIN="$ROOT_DIR/var/seed_paper_eval/paper310/bin/python"
export PYTHONPATH="$SNAPSHOT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$ROOT_DIR/var/seed_paper_eval/paper310/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export E49E_PRIOR_REPAIR_V2B_RECORDS="$PRIOR"
export E49E_CURATED_CONTRACTS="$CONTRACTS"
REPAIR="$SNAPSHOT_ROOT/ops/math_strategy_calibration/repair_e49e_singleton_gaps_v3_curated.py"
SOURCE="$ROOT_DIR/var/data/e49b_math_strategy_toy"
OUTPUT="$ROOT_DIR/var/data/e49e_trace_bank_math_toy_repaired_v3"
EVIDENCE="$ROOT_DIR/var/artifacts/e49e_trace_bank_math_toy_repair_v3"

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
if [[ ! -f "$REPAIR" || ! -f "$CONTRACTS" || ! -f "$PRIOR" ]] \
  || [[ ! -f "$FROZEN_E49D/menu_records.jsonl" ]] \
  || [[ ! -f "$FROZEN_E49E/trace_bank_records.jsonl" ]] \
  || [[ ! -f "$FROZEN_AUGMENT/augmentation_records.jsonl" ]] \
  || [[ ! -f "$ENDPOINT" || ! -f "$INVALID" || ! -f "$EQUIVALENT" ]] \
  || [[ ! -f "$IDENTITY" ]] \
  || [[ "$(hash_tree "$SNAPSHOT_ROOT")" != "$SNAPSHOT_HASH" ]] \
  || [[ "$(sha256sum "$IDENTITY" | cut -d' ' -f1)" != "$IDENTITY_HASH" ]]; then
  echo "E49E curated-repair V3 frozen identity failed" >&2
  exit 3
fi

exec "$PYTHON_BIN" "$REPAIR" \
  --source "$SOURCE" \
  --input-e49d "$FROZEN_E49D" \
  --input-e49e "$FROZEN_E49E" \
  --input-augmentation "$FROZEN_AUGMENT" \
  --output "$OUTPUT" \
  --evidence "$EVIDENCE" \
  --endpoint "$ENDPOINT" \
  --known-invalid-controls "$INVALID" \
  --known-equivalent-controls "$EQUIVALENT" \
  --repair-workers 4 \
  --timeout 900
