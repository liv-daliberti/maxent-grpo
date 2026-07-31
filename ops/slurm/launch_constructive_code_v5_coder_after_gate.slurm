#!/bin/bash
#SBATCH --job-name=launch-constructive-v5-coder
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --partition=cs
#SBATCH --time=00:15:00
#SBATCH --output=var/artifacts/logs/%x-%j.out
#SBATCH --error=var/artifacts/logs/%x-%j.err

set -euo pipefail
ROOT_DIR="${ROOT_DIR:?ROOT_DIR is required}"
LAUNCHER="$ROOT_DIR/ops/exp_scaling/launch_constructive_code_v5_coder_viability.sh"
EXPECTED_SHA256="${OAT_ZERO_EXPECTED_LAUNCHER_SHA256:?launcher hash is required}"
GATE_JOB_ID="${OAT_ZERO_GATE_JOB_ID:?gate job ID is required}"
IDENTITY="$ROOT_DIR/var/artifacts/constructive_code_v5_gate_identity.json"
AUDIT="$ROOT_DIR/var/artifacts/constructive_code_v5_gate_audit.json"

[[ "$(sha256sum "$LAUNCHER" | cut -d' ' -f1)" == "$EXPECTED_SHA256" ]] || {
  echo "ConstructiveCode v5 Coder launcher changed after dependency submission" >&2
  exit 1
}
"$ROOT_DIR/var/seed_paper_eval/paper310/bin/python" - "$IDENTITY" "$AUDIT" "$GATE_JOB_ID" <<'PY'
import json,pathlib,sys
identity=json.loads(pathlib.Path(sys.argv[1]).read_text())
audit=json.loads(pathlib.Path(sys.argv[2]).read_text())
if identity.get("job_id") != int(sys.argv[3]): raise SystemExit("v5 gate job identity mismatch")
if audit.get("status") != "pass": raise SystemExit("v5 gate is not pass")
if audit.get("expected_replay_count") != 2304 or audit.get("observed_replay_count") != 2304: raise SystemExit("v5 replay count mismatch")
if audit.get("violations") not in ([],None) or audit.get("checker_equivalence_violations") not in ([],None): raise SystemExit("v5 gate has hard violations")
PY
exec bash "$LAUNCHER" run
