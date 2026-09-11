#!/usr/bin/env bash
# Submit the dependency preparer after the v5 gate's viability-launch job.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
phase="${1:-}"
case "$phase" in config|run) ;; *) echo "Usage: $0 {config|run}" >&2; exit 1 ;; esac
UPSTREAM="$ROOT_DIR/var/artifacts/constructive_code_v5_coder_dependency_identity.json"
DEPENDENCY_LAUNCHER="$ROOT_DIR/ops/exp_scaling/launch_constructive_code_v5_paired_dependency.sh"
BATCH="$ROOT_DIR/ops/slurm/prepare_constructive_code_v5_paired_dependency.slurm"
OUTPUT="$ROOT_DIR/var/artifacts/constructive_code_v5_paired_dependency_preparer_identity.json"
for required in "$UPSTREAM" "$DEPENDENCY_LAUNCHER" "$BATCH"; do
  [[ -e "$required" ]] || { echo "Missing paired dependency-preparer prerequisite: $required" >&2; exit 1; }
done
bash -n "$DEPENDENCY_LAUNCHER" "$BATCH"
if [[ "$phase" == config ]]; then
  bash "$DEPENDENCY_LAUNCHER" config
  echo "[constructive-v5-paired-preparer] configuration passed; no job submitted"
  exit 0
fi
[[ ! -e "$OUTPUT" ]] || { echo "Fresh paired dependency-preparer identity required" >&2; exit 1; }
upstream_job="$(jq -er '.job_id' "$UPSTREAM")"
[[ "$upstream_job" =~ ^[0-9]+$ ]] || { echo "Invalid upstream dependency job id" >&2; exit 1; }
launcher_sha="$(sha256sum "$DEPENDENCY_LAUNCHER" | cut -d' ' -f1)"
job_id="$(sbatch --parsable --dependency="afterok:$upstream_job" \
  --export="ALL,ROOT_DIR=$ROOT_DIR,OAT_ZERO_EXPECTED_DEPENDENCY_LAUNCHER_SHA256=$launcher_sha" \
  "$BATCH")"
job_id="${job_id%%;*}"
[[ "$job_id" =~ ^[0-9]+$ ]] || { echo "Invalid dependency-preparer job id" >&2; exit 1; }
"$ROOT_DIR/var/seed_paper_eval/paper310/bin/python" - "$OUTPUT" "$job_id" "$upstream_job" "$launcher_sha" "$UPSTREAM" "$BATCH" <<'PY'
import hashlib,json,os,pathlib,sys,tempfile
def sha(path): return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
path=pathlib.Path(sys.argv[1]); payload={
 "schema":"constructive-code-v5-paired-dependency-preparer-identity-v1",
 "job_id":int(sys.argv[2]),"dependency":f"afterok:{sys.argv[3]}",
 "upstream_launcher_job_id":int(sys.argv[3]),"expected_dependency_launcher_sha256":sys.argv[4],
 "upstream_identity_sha256":sha(sys.argv[5]),"batch_sha256":sha(sys.argv[6]),"fail_closed":True
}
path.parent.mkdir(parents=True,exist_ok=True); fd,tmp=tempfile.mkstemp(prefix=f".{path.name}.",dir=path.parent)
with os.fdopen(fd,"w") as handle: json.dump(payload,handle,indent=2,sort_keys=True); handle.write("\n")
os.replace(tmp,path)
PY
echo "[constructive-v5-paired-preparer] submitted job $job_id afterok:$upstream_job"
