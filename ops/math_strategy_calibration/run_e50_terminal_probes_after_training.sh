#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=/n/fs/similarity/maxent-grpo
cd "$ROOT_DIR"
variant="${E50_TERMINAL_VARIANT:?missing E50_TERMINAL_VARIANT}"
case "$variant" in
  e50b) prefix=e50b_observed_route_math_toy_05b_3ep_v1 ;;
  e50d) prefix=e50d_teacher_route_math_toy_05b_3ep_v1 ;;
  *) echo "invalid E50 terminal variant: $variant" >&2; exit 1 ;;
esac
manifest="$ROOT_DIR/var/artifacts/${prefix}_comparative_jobs.tsv"
if [[ ! -f "$manifest" ]]; then
  echo "missing matched manifest: $manifest" >&2
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
    echo "could not resolve run root for $run_stamp" >&2
    exit 1
  fi
  run_root="${matches[0]}"
  debug="$run_root/debug_job$job_id"
  if [[ ! -f "$run_root/TRAINING_COMPLETE.json" ]]; then
    echo "matched arm did not complete: $arm job $job_id" >&2
    exit 1
  fi
  model="$(find "$debug/saved_models" -mindepth 1 -maxdepth 1 \
    -type d -name 'step_*' | sort | tail -1)"
  if [[ -z "$model" || ! -f "$model/model.safetensors" ]]; then
    echo "matched arm has no terminal exported model: $arm" >&2
    exit 1
  fi
  case "$arm" in
    grpo) control_model="$model" ;;
    online_canonical_haarnoja) treatment_model="$model" ;;
    *) echo "unexpected matched arm: $arm" >&2; exit 1 ;;
  esac
done < "$manifest"
if [[ -z "$control_model" || -z "$treatment_model" ]]; then
  echo "matched terminal models are incomplete" >&2
  exit 1
fi

bash "$ROOT_DIR/ops/math_strategy_calibration/launch_e50_route_probe.sh" \
  "$variant" control_terminal "$control_model"
bash "$ROOT_DIR/ops/math_strategy_calibration/launch_e50_route_probe.sh" \
  "$variant" treatment_terminal "$treatment_model"

control_job="$(<"$ROOT_DIR/var/artifacts/${variant}_control_terminal_route_probe_v1.job.txt")"
treatment_job="$(<"$ROOT_DIR/var/artifacts/${variant}_treatment_terminal_route_probe_v1.job.txt")"
analysis_job="$(sbatch --parsable \
  --dependency="afterok:$control_job:$treatment_job" \
  --export="ALL,E50_ANALYSIS_VARIANT=$variant" \
  "$ROOT_DIR/ops/slurm/e50_analyze_matched_after_probes.slurm")"
printf '%s\n' "$analysis_job" \
  > "$ROOT_DIR/var/artifacts/${variant}_matched_analysis_job.txt"
printf 'queued %s matched analysis job %s\n' "$variant" "$analysis_job"
