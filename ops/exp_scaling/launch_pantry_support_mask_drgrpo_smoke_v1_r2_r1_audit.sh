#!/usr/bin/env bash
# Snapshot and submit the audit-only Pantry r2-r1 schema repair.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
phase="${1:-}"
case "$phase" in config|run) ;; *) echo "Usage: $0 {config|run}" >&2; exit 1 ;; esac
PYTHON_BIN="$ROOT_DIR/var/seed_paper_eval/paper310/bin/python"
PROTOCOL="$ROOT_DIR/paper/preregistration/pantry_support_mask_drgrpo_smoke_v1_r2_r1_audit_schema_repair_20260730.md"
SLURM_SCRIPT="$ROOT_DIR/ops/slurm/audit_pantry_support_mask_drgrpo_smoke_v1_r2_r1.slurm"
OUTPUT="$ROOT_DIR/var/artifacts/pantry_support_mask_drgrpo_smoke_v1_r2_r1_audit.json"
IDENTITY="$ROOT_DIR/var/artifacts/pantry_support_mask_drgrpo_smoke_v1_r2_r1_audit_runner_identity.json"
FAILED="$ROOT_DIR/var/artifacts/pantry_support_mask_drgrpo_smoke_v1_r2_audit.json"
FILES=(audit_pantry_support_mask_drgrpo_smoke_v1.py audit_pantry_support_mask_drgrpo_smoke_v1_r2.py audit_pantry_support_mask_drgrpo_smoke_v1_r2_r1.py)

for required in "$PYTHON_BIN" "$PROTOCOL" "$SLURM_SCRIPT" "$FAILED"; do
  [[ -f "$required" ]] || { echo "Missing Pantry r2-r1 audit prerequisite: $required" >&2; exit 1; }
done
hash_tree() { local tree="$1"; (cd "$tree"; find . -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1); }

if [[ "$phase" == config ]]; then
  for file in "${FILES[@]}"; do "$PYTHON_BIN" -m py_compile "$ROOT_DIR/ops/$file"; done
  bash -n "$SLURM_SCRIPT"
  echo "[pantry-r2-r1-audit] configuration passed; no audit submitted"
  exit 0
fi
for fresh in "$OUTPUT" "$IDENTITY"; do [[ ! -e "$fresh" ]] || { echo "Fresh Pantry r2-r1 audit target required: $fresh" >&2; exit 1; }; done
STAGING="$(mktemp -d "$ROOT_DIR/var/artifacts/source_snapshots/.pantry-r2-r1-audit.XXXXXX")"
for file in "${FILES[@]}"; do cp "$ROOT_DIR/ops/$file" "$STAGING/$file"; done
cp "$SLURM_SCRIPT" "$STAGING/audit_pantry_support_mask_drgrpo_smoke_v1_r2_r1.slurm"
EXEC_HASH="$(hash_tree "$STAGING")"
EXEC_ROOT="$ROOT_DIR/var/artifacts/source_snapshots/pantry_support_mask_drgrpo_smoke_v1_r2_r1_audit_${EXEC_HASH}"
mv "$STAGING" "$EXEC_ROOT"
job_id="$(sbatch --parsable --partition=all --account=allcs --export="ALL,ROOT_DIR=$ROOT_DIR,OAT_ZERO_EXECUTION_ROOT=$EXEC_ROOT" "$EXEC_ROOT/audit_pantry_support_mask_drgrpo_smoke_v1_r2_r1.slurm")"
job_id="${job_id%%;*}"
"$PYTHON_BIN" - "$IDENTITY" "$job_id" "$EXEC_HASH" "$PROTOCOL" "$FAILED" <<'PY'
import hashlib,json,os,pathlib,sys,tempfile
def digest(path): return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
path=pathlib.Path(sys.argv[1]); payload={"schema_version":"pantry-r2-r1-audit-runner-v1","audit_job_id":int(sys.argv[2]),"training_job_id":30199933,"execution_hash":sys.argv[3],"protocol_sha256":digest(sys.argv[4]),"failed_r2_audit_sha256":digest(sys.argv[5]),"training_rerun":False}
path.parent.mkdir(parents=True,exist_ok=True); fd,tmp=tempfile.mkstemp(prefix=f".{path.name}.",dir=path.parent)
with os.fdopen(fd,"w",encoding="utf-8") as handle: json.dump(payload,handle,indent=2,sort_keys=True); handle.write("\n")
os.replace(tmp,path)
PY
echo "[pantry-r2-r1-audit] submitted audit job $job_id"
