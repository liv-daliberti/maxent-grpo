#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
mode="${1:-run}"
if [[ "$mode" != config && "$mode" != run ]]; then
  echo "usage: $0 [config|run]" >&2
  exit 1
fi

: "${E50E_ENDPOINT_RECORD:?missing E50E_ENDPOINT_RECORD}"
: "${E50E_ROUTE_CALIBRATION:?missing E50E_ROUTE_CALIBRATION}"
: "${E50E_DECLARATION_CALIBRATION:?missing E50E_DECLARATION_CALIBRATION}"
: "${E50E_CONTINUITY_CERTIFICATE:?missing E50E_CONTINUITY_CERTIFICATE}"
"$ROOT_DIR/var/seed_paper_eval/paper310/bin/python" \
  "$ROOT_DIR/ops/math_strategy_calibration/certify_e50e_qwen72_continuity.py" \
  --endpoint-record "$E50E_ENDPOINT_RECORD" \
  --route-result "$E50E_ROUTE_CALIBRATION" \
  --declaration-result "$E50E_DECLARATION_CALIBRATION" \
  --expected-node node302 \
  --expected-port 8771 \
  --output "$E50E_CONTINUITY_CERTIFICATE" \
  --audit-only >/dev/null

toy_advancement="$ROOT_DIR/var/artifacts/e50d_matched_math_toy_advancement_v1.json"
if [[ ! -f "$toy_advancement" ]] || ! jq -e \
  '.schema == "e50_matched_math_toy_advancement_v1"
   and .variant == "e50d"
   and .complete_evidence == true
   and .advance_to_exact_oat_full == true
   and ([.checks[]] | all)' \
  "$toy_advancement" >/dev/null; then
  echo "E50E requires a passing E50D matched toy" >&2
  exit 1
fi

source_hash=3cf26e564b3cafbdc4d0a819febe6d0a61d443746750931c87250f662fcec0a6
ops_hash=782fbefe34db2a0053dcd79d76542b58b31ed70e260518302af71f5a7281c867
source_base="$ROOT_DIR/var/artifacts/source_snapshots/e49t_natural_menu_${source_hash}/src"
source_overlay="$ROOT_DIR/var/artifacts/source_snapshots/e50_math_corrected_overlay_v1/src"
expected_overlay_hash=ba032e9ca300c25385be9650582556f6c8f833ce1f4f5a7197c4259ce5e44a1e
observed_overlay_hash="$(
  "$ROOT_DIR/var/seed_paper_eval/paper310/bin/python" - "$source_overlay" <<'PY'
import hashlib
import pathlib
import sys
root = pathlib.Path(sys.argv[1])
digest = hashlib.sha256()
for path in sorted(
    item
    for item in root.rglob("*")
    if item.is_file()
    and "__pycache__" not in item.parts
    and item.suffix != ".pyc"
):
    digest.update(str(path.relative_to(root)).encode("utf-8"))
    digest.update(b"\0")
    digest.update(hashlib.sha256(path.read_bytes()).digest())
print(digest.hexdigest())
PY
)"
if [[ "$observed_overlay_hash" != "$expected_overlay_hash" ]]; then
  echo "E50E corrected source overlay identity mismatch" >&2
  exit 1
fi
export PYTHONDONTWRITEBYTECODE=1
export E49V_PROTOCOL="$ROOT_DIR/paper/preregistration/e50e_exact_oat_math_full_20260726.md"
export E49V_DATA_ROOT="$ROOT_DIR/var/data/e50e_exact_oat_math_full"
export E49V_DATA_EVIDENCE="$ROOT_DIR/var/artifacts/e50e_exact_oat_math_full_v1"
export E49V_MATERIALIZER="$ROOT_DIR/ops/math_strategy_calibration/materialize_e50e_exact_oat_math.py"
export E49V_TOY_ADVANCEMENT="$toy_advancement"
export E49V_TOY_ADVANCEMENT_SCHEMA=e50_matched_math_toy_advancement_v1
export E49V_DATA_SCHEMA=e50e_exact_oat_math_materialization_v1
export E49V_SINGLETON_TOTAL=873
export E49V_MULTI_TOTAL=11
export E49V_TRAIN_MULTI=10
export E49V_EVAL_MULTI=1
export E49V_PREFIX=e50e_exact_oat_math_05b_3ep_v1
export E49V_QWEN72_ENDPOINT_RECORD="$E50E_ENDPOINT_RECORD"
export E49V_QWEN72_NODE=node302
export E49V_QWEN72_PORT=8771
export E49V_ROUTE_CALIBRATION_RESULT="$E50E_ROUTE_CALIBRATION"
export E49V_ROUTE_CALIBRATION_IDENTITY="$ROOT_DIR/var/artifacts/e49t_route_confusion_calibration_v1/frozen_identity.json"
export E49V_DECLARATION_CALIBRATION_RESULT="$E50E_DECLARATION_CALIBRATION"
export E49V_DECLARATION_CALIBRATION_IDENTITY="$ROOT_DIR/var/artifacts/e49t_declaration_mismatch_calibration_v1/frozen_identity.json"
export E49V_CONTINUITY_CERTIFICATE="$E50E_CONTINUITY_CERTIFICATE"
export E49V_SOURCE_ROOT_OVERRIDE="$source_overlay"
export E49V_CANONICALIZER_SOURCE_OVERRIDE="$source_base/oat_drgrpo/math_strategy_canonicalizer.py"
export E49V_OPS_ROOT_OVERRIDE="$ROOT_DIR/var/artifacts/source_snapshots/e49t_natural_menu_ops_${ops_hash}/ops"
export E49V_WRAPPER_HASH="$(sha256sum "${BASH_SOURCE[0]}" | cut -d' ' -f1)"

bash "$ROOT_DIR/ops/exp_scaling/launch_e49v_exact_oat_natural_menu_math_05b.sh" \
  "$mode"
if [[ "$mode" == config ]]; then
  exit 0
fi

manifest="$ROOT_DIR/var/artifacts/e50e_exact_oat_math_05b_3ep_v1_comparative_jobs.tsv"
mapfile -t job_ids < <(awk -F '\t' 'NR > 1 {print $3}' "$manifest")
if [[ ${#job_ids[@]} != 2 ]]; then
  echo "E50E matched manifest does not contain two jobs" >&2
  exit 1
fi
terminal_job="$(sbatch --parsable \
  --dependency="afterany:${job_ids[0]}:${job_ids[1]}" \
  --export="ALL,E50E_ENDPOINT_RECORD=$E50E_ENDPOINT_RECORD,E50E_ROUTE_CALIBRATION=$E50E_ROUTE_CALIBRATION,E50E_DECLARATION_CALIBRATION=$E50E_DECLARATION_CALIBRATION,E50E_CONTINUITY_CERTIFICATE=$E50E_CONTINUITY_CERTIFICATE" \
  "$ROOT_DIR/ops/slurm/e50e_terminal_probes_after_training.slurm")"
printf '%s\n' "$terminal_job" \
  > "$ROOT_DIR/var/artifacts/e50e_terminal_probe_orchestrator_job.txt"
printf 'queued E50E terminal route-probe orchestrator job %s\n' \
  "$terminal_job"
