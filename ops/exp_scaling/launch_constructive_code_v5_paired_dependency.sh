#!/usr/bin/env bash
# Submit a fail-closed paired-launch wrapper after the actual viability job.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
phase="${1:-}"
case "$phase" in config|run) ;; *) echo "Usage: $0 {config|run}" >&2; exit 1 ;; esac
IDENTITY="$ROOT_DIR/var/artifacts/constructive_code_v5_coder_05b_viability_identity.json"
LAUNCHER="$ROOT_DIR/ops/exp_scaling/launch_constructive_code_v5_experiment.py"
BATCH="$ROOT_DIR/ops/slurm/launch_constructive_code_v5_paired_after_viability.slurm"
OUTPUT="$ROOT_DIR/var/artifacts/constructive_code_v5_paired_dependency_identity.json"
bash -n "$BATCH"
"$ROOT_DIR/var/seed_paper_eval/paper310/bin/python" -m py_compile "$LAUNCHER"
if [[ "$phase" == config ]]; then
  echo "[constructive-v5-paired-dependency] configuration passed; no job submitted"
  exit 0
fi
[[ -f "$IDENTITY" ]] || { echo "ConstructiveCode viability identity is missing" >&2; exit 1; }
[[ ! -e "$OUTPUT" ]] || { echo "Fresh paired dependency identity required" >&2; exit 1; }
viability_job="$(jq -er '.job_id' "$IDENTITY")"
[[ "$viability_job" =~ ^[0-9]+$ ]] || { echo "Invalid viability job id" >&2; exit 1; }
launcher_sha="$(sha256sum "$LAUNCHER" | cut -d' ' -f1)"
job_id="$(sbatch --parsable --dependency="afterok:$viability_job" \
  --export="ALL,ROOT_DIR=$ROOT_DIR,OAT_ZERO_EXPECTED_LAUNCHER_SHA256=$launcher_sha" \
  "$BATCH")"
job_id="${job_id%%;*}"
[[ "$job_id" =~ ^[0-9]+$ ]] || { echo "Invalid paired-launch dependency job id" >&2; exit 1; }
"$ROOT_DIR/var/seed_paper_eval/paper310/bin/python" - "$OUTPUT" "$job_id" "$viability_job" "$launcher_sha" "$IDENTITY" "$BATCH" <<'PY'
import hashlib,json,os,pathlib,sys,tempfile
def sha(path): return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
path=pathlib.Path(sys.argv[1]); payload={
 "schema":"constructive-code-v5-paired-dependency-identity-v1",
 "job_id":int(sys.argv[2]),"dependency":f"afterok:{sys.argv[3]}",
 "viability_job_id":int(sys.argv[3]),"expected_launcher_sha256":sys.argv[4],
 "viability_identity_sha256":sha(sys.argv[5]),"batch_sha256":sha(sys.argv[6]),
 "fail_closed":True
}
path.parent.mkdir(parents=True,exist_ok=True); fd,tmp=tempfile.mkstemp(prefix=f".{path.name}.",dir=path.parent)
with os.fdopen(fd,"w") as handle: json.dump(payload,handle,indent=2,sort_keys=True); handle.write("\n")
os.replace(tmp,path)
PY
echo "[constructive-v5-paired-dependency] submitted job $job_id afterok:$viability_job"
