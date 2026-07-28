#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=/n/fs/similarity/maxent-grpo
cd "$ROOT_DIR"

variant="${1:-}"
stage="${2:-baseline}"
model_root="${3:-$ROOT_DIR/var/cache/huggingface/transformers/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775}"
endpoint_record="${4:-$ROOT_DIR/var/artifacts/e49t_qwen72_node302_v1/qwen72_endpoint.json}"
endpoint_node="${5:-node302}"
endpoint_port="${6:-8770}"
case "$variant" in
  e50b)
    data="$ROOT_DIR/var/data/e50b_observed_route_math_toy"
    schema=e50b_observed_route_math_toy_v1
    ;;
  e50d)
    data="$ROOT_DIR/var/data/e50d_teacher_route_math_toy"
    schema=e50d_teacher_route_math_toy_v1
    ;;
  *)
    echo "usage: $0 {e50b|e50d} {baseline|control_terminal|treatment_terminal} [model_root]" >&2
    exit 1
    ;;
esac
case "$stage" in
  baseline|control_terminal|treatment_terminal|full_control_terminal|full_treatment_terminal) ;;
  *)
    echo "invalid route-probe stage: $stage" >&2
    exit 1
    ;;
esac

output="$ROOT_DIR/var/artifacts/${variant}_${stage}_route_probe_v1"
if [[ ! -f "$data/MATERIALIZATION_MANIFEST.json" ]]; then
  echo "route-probe data is not materialized: $data" >&2
  exit 1
fi
if [[ ! -f "$model_root/config.json" ]]; then
  echo "route-probe model is incomplete: $model_root" >&2
  exit 1
fi
if [[ ! -f "$endpoint_record" ]]; then
  echo "route-probe endpoint record is missing: $endpoint_record" >&2
  exit 1
fi
if [[ -e "$output/result.json" ]]; then
  echo "fresh route-probe result required: $output/result.json" >&2
  exit 1
fi

require_baseline=0
if [[ "$stage" == baseline ]]; then
  require_baseline=1
fi
export_string="ALL,E50_ROUTE_DATA=$data,E50_ROUTE_MODEL=$model_root"
export_string+=",E50_ROUTE_OUT=$output/result.json,E50_ROUTE_SCHEMA=$schema"
export_string+=",E50_ROUTE_LABEL=${variant}_${stage}"
export_string+=",E50_ROUTE_REQUIRE_BASELINE=$require_baseline"
export_string+=",E50_ROUTE_ENDPOINT_RECORD=$endpoint_record"
export_string+=",E50_ROUTE_ENDPOINT_SHA256=$(sha256sum "$endpoint_record" | cut -d' ' -f1)"
export_string+=",E50_ROUTE_ENDPOINT_NODE=$endpoint_node"
export_string+=",E50_ROUTE_ENDPOINT_PORT=$endpoint_port"
job_id="$(sbatch --parsable --export="$export_string" \
  "$ROOT_DIR/ops/slurm/e50_route_probe_a6000.slurm")"
printf '%s\n' "$job_id" > "$output.job.txt"
printf 'submitted %s %s route probe job %s\n' "$variant" "$stage" "$job_id"
if [[ "$stage" == baseline ]]; then
  matched_job_id="$(sbatch --parsable \
    --dependency="afterok:$job_id" \
    --export="ALL,E50_MATCHED_VARIANT=$variant" \
    "$ROOT_DIR/ops/slurm/e50_launch_matched_after_probe.slurm")"
  printf '%s\n' "$matched_job_id" > "$output.matched_launch_job.txt"
  printf 'queued %s matched launch job %s after route probe\n' \
    "$variant" "$matched_job_id"
fi
