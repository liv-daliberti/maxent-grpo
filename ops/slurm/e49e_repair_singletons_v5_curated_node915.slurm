#!/usr/bin/env bash
#SBATCH --job-name=e49e-repair-singletons-v5
#SBATCH --partition=mltheory
#SBATCH --account=mltheory
#SBATCH --nodelist=node915
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
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
PRIOR_V4="${E49E_REPAIR_PRIOR_V4:-}"
V3_RECORDS="${E49E_REPAIR_V3_RECORDS:-}"
V3_CONTRACTS="${E49E_REPAIR_V3_CONTRACTS:-}"
V4_CONTRACTS="${E49E_REPAIR_V4_CONTRACTS:-}"
V5_CONTRACTS="${E49E_REPAIR_V5_CONTRACTS:-}"
IDENTITY="${E49E_REPAIR_IDENTITY:-}"
IDENTITY_HASH="${E49E_REPAIR_IDENTITY_HASH:-}"
if [[ "$ROOT_DIR" != "/n/fs/similarity/maxent-grpo" ]] \
  || [[ "$(hostname -s)" != node915 ]] \
  || [[ -z "$SNAPSHOT_ROOT" || -z "$SNAPSHOT_HASH" ]] \
  || [[ -z "$FROZEN_E49D" || -z "$FROZEN_E49E" || -z "$FROZEN_AUGMENT" ]] \
  || [[ -z "$ENDPOINT" || -z "$INVALID" || -z "$EQUIVALENT" ]] \
  || [[ -z "$PRIOR_V4" || -z "$V3_RECORDS" ]] \
  || [[ -z "$V3_CONTRACTS" || -z "$V4_CONTRACTS" || -z "$V5_CONTRACTS" ]] \
  || [[ -z "$IDENTITY" || -z "$IDENTITY_HASH" ]]; then
  echo "invalid E49E curated-repair V5 environment" >&2
  exit 2
fi

source "$ROOT_DIR/ops/repo_env.sh"
PYTHON_BIN="$ROOT_DIR/var/seed_paper_eval/paper310/bin/python"
export PYTHONPATH="$SNAPSHOT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$ROOT_DIR/var/seed_paper_eval/paper310/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export E49E_PRIOR_REPAIR_V4_RECORDS="$PRIOR_V4"
export E49E_V3_REPAIR_RECORDS="$V3_RECORDS"
export E49E_V3_CURATED_CONTRACTS="$V3_CONTRACTS"
export E49E_V4_CURATED_CONTRACTS="$V4_CONTRACTS"
export E49E_V5_CURATED_CONTRACTS="$V5_CONTRACTS"
REPAIR="$SNAPSHOT_ROOT/ops/math_strategy_calibration/repair_e49e_singleton_gaps_v5_curated.py"
SOURCE="$ROOT_DIR/var/data/e49b_math_strategy_toy"
OUTPUT="$ROOT_DIR/var/data/e49e_trace_bank_math_toy_repaired_v5"
EVIDENCE="$ROOT_DIR/var/artifacts/e49e_trace_bank_math_toy_repair_v5"

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

if [[ ! -f "$REPAIR" || ! -f "$PRIOR_V4" || ! -f "$V3_RECORDS" ]] \
  || [[ ! -f "$V3_CONTRACTS" || ! -f "$V4_CONTRACTS" || ! -f "$V5_CONTRACTS" ]] \
  || [[ ! -f "$FROZEN_E49D/menu_records.jsonl" ]] \
  || [[ ! -f "$FROZEN_E49E/trace_bank_records.jsonl" ]] \
  || [[ ! -f "$FROZEN_AUGMENT/augmentation_records.jsonl" ]] \
  || [[ ! -f "$ENDPOINT" || ! -f "$INVALID" || ! -f "$EQUIVALENT" ]] \
  || [[ ! -f "$IDENTITY" ]] \
  || [[ "$(hash_tree "$SNAPSHOT_ROOT")" != "$SNAPSHOT_HASH" ]] \
  || [[ "$(sha256sum "$IDENTITY" | cut -d' ' -f1)" != "$IDENTITY_HASH" ]]; then
  echo "E49E curated-repair V5 frozen identity failed" >&2
  exit 3
fi

"$PYTHON_BIN" - "$IDENTITY" \
  "$PRIOR_V4" "$V3_RECORDS" \
  "$V3_CONTRACTS" "$V4_CONTRACTS" "$V5_CONTRACTS" \
  "$ENDPOINT" \
  "$(hash_tree "$FROZEN_E49D")" \
  "$(hash_tree "$FROZEN_E49E")" \
  "$(hash_tree "$FROZEN_AUGMENT")" \
  "$(hash_tree "$SOURCE/train")" \
  "$(hash_tree "$SOURCE/eval")" <<'PY'
import hashlib
import json
import pathlib
import sys

identity = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
files = (
    ("prior_v4_repair_records_sha256", sys.argv[2]),
    ("embedded_v3_repair_records_sha256", sys.argv[3]),
    ("v3_contracts_sha256", sys.argv[4]),
    ("v4_contracts_sha256", sys.argv[5]),
    ("v5_contracts_sha256", sys.argv[6]),
    ("endpoint_record_sha256", sys.argv[7]),
)
for key, path in files:
    observed = hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
    if identity.get(key) != observed:
        raise SystemExit(f"frozen V5 file binding failed: {key}")
trees = (
    ("frozen_e49d_tree_sha256", sys.argv[8]),
    ("frozen_e49e_tree_sha256", sys.argv[9]),
    ("frozen_augmentation_tree_sha256", sys.argv[10]),
    ("source_train_tree_sha256", sys.argv[11]),
    ("source_eval_tree_sha256", sys.argv[12]),
)
for key, observed in trees:
    if identity.get(key) != observed:
        raise SystemExit(f"frozen V5 tree binding failed: {key}")
PY

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
  --repair-workers 1 \
  --timeout 900
