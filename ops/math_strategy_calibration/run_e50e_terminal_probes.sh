#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=/n/fs/similarity/maxent-grpo
cd "$ROOT_DIR"
prefix=e50e_exact_oat_math_05b_3ep_v1
manifest="$ROOT_DIR/var/artifacts/${prefix}_comparative_jobs.tsv"
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

route_variant=e50d
advancement="$ROOT_DIR/var/artifacts/e50d_matched_math_toy_advancement_v1.json"
if [[ ! -f "$advancement" ]] || ! jq -e \
  '.schema == "e50_matched_math_toy_advancement_v1"
   and .variant == "e50d"
   and .complete_evidence == true
   and .advance_to_exact_oat_full == true
   and ([.checks[]] | all)' \
  "$advancement" >/dev/null; then
  echo "cannot verify the passing E50D route-probe toy" >&2
  exit 1
fi

control_model=""
treatment_model=""
while IFS=$'\t' read -r arm seed job_id run_stamp; do
  [[ "$arm" == arm ]] && continue
  matches=(
    "$ROOT_DIR"/var/data/xdr_qwen25_0p5b_instruct_*_"$run_stamp"
  )
  if [[ ${#matches[@]} != 1 || ! -d "${matches[0]}" ]]; then
    echo "could not resolve E50E run root for $run_stamp" >&2
    exit 1
  fi
  run_root="${matches[0]}"
  debug="$run_root/debug_job$job_id"
  if [[ ! -f "$run_root/TRAINING_COMPLETE.json" ]]; then
    echo "E50E arm did not complete: $arm" >&2
    exit 1
  fi
  model="$(find "$debug/saved_models" -mindepth 1 -maxdepth 1 \
    -type d -name 'step_*' | sort | tail -1)"
  if [[ -z "$model" || ! -f "$model/model.safetensors" ]]; then
    echo "E50E terminal model missing: $arm" >&2
    exit 1
  fi
  case "$arm" in
    grpo) control_model="$model" ;;
    online_canonical_haarnoja) treatment_model="$model" ;;
    *) echo "unexpected E50E arm: $arm" >&2; exit 1 ;;
  esac
done < "$manifest"
if [[ -z "$control_model" || -z "$treatment_model" ]]; then
  echo "E50E matched terminal models are incomplete" >&2
  exit 1
fi

bash "$ROOT_DIR/ops/math_strategy_calibration/launch_e50_route_probe.sh" \
  "$route_variant" full_control_terminal "$control_model" \
  "$E50E_ENDPOINT_RECORD" node302 8771
bash "$ROOT_DIR/ops/math_strategy_calibration/launch_e50_route_probe.sh" \
  "$route_variant" full_treatment_terminal "$treatment_model" \
  "$E50E_ENDPOINT_RECORD" node302 8771

control_job="$(<"$ROOT_DIR/var/artifacts/${route_variant}_full_control_terminal_route_probe_v1.job.txt")"
treatment_job="$(<"$ROOT_DIR/var/artifacts/${route_variant}_full_treatment_terminal_route_probe_v1.job.txt")"
analysis_job="$(sbatch --parsable \
  --dependency="afterok:$control_job:$treatment_job" \
  --export="ALL,E50E_ROUTE_VARIANT=$route_variant,E50E_ENDPOINT_RECORD=$E50E_ENDPOINT_RECORD,E50E_ROUTE_CALIBRATION=$E50E_ROUTE_CALIBRATION,E50E_DECLARATION_CALIBRATION=$E50E_DECLARATION_CALIBRATION,E50E_CONTINUITY_CERTIFICATE=$E50E_CONTINUITY_CERTIFICATE" \
  "$ROOT_DIR/ops/slurm/e50e_analyze_after_probes.slurm")"
printf '%s\n' "$analysis_job" \
  > "$ROOT_DIR/var/artifacts/e50e_exact_oat_analysis_job.txt"
printf 'queued E50E exact analysis job %s\n' "$analysis_job"
