#!/usr/bin/env bash
# Submit the fail-closed post-v5-gate Coder viability launcher dependency.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="$ROOT_DIR/var/seed_paper_eval/paper310/bin/python"
GATE_IDENTITY="$ROOT_DIR/var/artifacts/constructive_code_v5_gate_identity.json"
LAUNCHER="$ROOT_DIR/ops/exp_scaling/launch_constructive_code_v5_coder_viability.sh"
BATCH="$ROOT_DIR/ops/slurm/launch_constructive_code_v5_coder_after_gate.slurm"
OUTPUT="$ROOT_DIR/var/artifacts/constructive_code_v5_coder_dependency_identity.json"
phase="${1:-}"
case "$phase" in config|run) ;; *) echo "Usage: $0 {config|run}" >&2; exit 1 ;; esac
for required in "$PYTHON_BIN" "$GATE_IDENTITY" "$LAUNCHER" "$BATCH"; do
  [[ -e "$required" ]] || { echo "Missing v5 Coder dependency prerequisite: $required" >&2; exit 1; }
done
bash -n "$BATCH" "$LAUNCHER"
gate_job_id="$(jq -er '.job_id' "$GATE_IDENTITY")"
[[ "$gate_job_id" =~ ^[0-9]+$ ]] || { echo "Invalid v5 gate job ID" >&2; exit 1; }
launcher_sha256="$(sha256sum "$LAUNCHER" | cut -d' ' -f1)"
if [[ "$phase" == config ]]; then
  echo "[constructive-v5-coder-dependency] configuration passed; no job submitted"
  exit 0
fi
[[ ! -e "$OUTPUT" ]] || { echo "Fresh v5 Coder dependency identity required" >&2; exit 1; }
job_id="$(sbatch --parsable --dependency="afterok:$gate_job_id" \
  --export="ALL,ROOT_DIR=$ROOT_DIR,OAT_ZERO_EXPECTED_LAUNCHER_SHA256=$launcher_sha256,OAT_ZERO_GATE_JOB_ID=$gate_job_id" \
  "$BATCH")"
job_id="${job_id%%;*}"
[[ "$job_id" =~ ^[0-9]+$ ]] || { echo "Invalid v5 Coder dependency job ID" >&2; exit 1; }
"$PYTHON_BIN" - "$OUTPUT" "$job_id" "$gate_job_id" "$launcher_sha256" "$LAUNCHER" "$BATCH" "$GATE_IDENTITY" <<'PY'
import hashlib,json,os,pathlib,sys,tempfile
def sha(path): return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
path=pathlib.Path(sys.argv[1]); payload={
 "schema":"constructive-code-v5-coder-dependency-identity-v1",
 "job_id":int(sys.argv[2]),"dependency":f"afterok:{sys.argv[3]}","gate_job_id":int(sys.argv[3]),
 "expected_viability_launcher_sha256":sys.argv[4],"dependency_launcher_sha256":sha(sys.argv[6]),
 "gate_identity_sha256":sha(sys.argv[7]),"fail_closed":True,"model_sampling_before_gate_pass":False
}
path.parent.mkdir(parents=True,exist_ok=True); fd,tmp=tempfile.mkstemp(prefix=f".{path.name}.",dir=path.parent)
with os.fdopen(fd,"w") as handle: json.dump(payload,handle,indent=2,sort_keys=True); handle.write("\n")
os.replace(tmp,path)
PY
echo "[constructive-v5-coder-dependency] submitted job $job_id afterok:$gate_job_id"
