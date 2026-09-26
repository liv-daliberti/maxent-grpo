#!/usr/bin/env bash
# Recover every deficient graph-coloring 7B E4/E5 cell after the deterministic
# vLLM 0.8.4 CuMem sleep-allocator crash at the 1,024th sleep cycle.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
MODE="${1:-config}"

case "$MODE" in
  config|full) ;;
  *) echo "Usage: $0 {config|full}" >&2; exit 2 ;;
esac

if [[ "$MODE" == "full" && "${OAT_ZERO_RETRY_GRAPH_7B_APPROVED:-0}" != "1" ]]; then
  echo "Set OAT_ZERO_RETRY_GRAPH_7B_APPROVED=1 to submit recovery jobs." >&2
  exit 2
fi

# Keeping vLLM resident avoids the defective CuMem sleep path entirely.  The
# smaller KV reservation plus the already-proven microbatch/offload layout fit
# the collocated 7B actor and learner on two 48-GiB A6000s without changing the
# data, group size, optimizer batch, objective, or evaluation cadence.
common_env=(
  "OAT_ZERO_TRAIN_NODELIST=node103,node104,node208"
  "OAT_ZERO_TRAIN_GRES=gpu:a6000:2"
  "OAT_ZERO_TRAIN_PARTITION=lowprio"
  "OAT_ZERO_TRAIN_ACCOUNT=mltheory"
  "OAT_ZERO_TRAIN_MEMORY=192G"
  "OAT_ZERO_N_GPU=2"
  "OAT_ZERO_NUM_GPUS_PER_ACTOR=2"
  "OAT_ZERO_ROLLOUT_BATCH_SIZE=2"
  "OAT_ZERO_ROLLOUT_BATCH_SIZE_PER_DEVICE=1"
  "OAT_ZERO_TRAIN_BATCH_SIZE=32"
  "OAT_ZERO_TRAIN_BATCH_SIZE_PER_DEVICE=4"
  "OAT_ZERO_VLLM_GPU_RATIO=0.10"
  "OAT_ZERO_VLLM_SLEEP=0"
  "OAT_ZERO_ADAM_OFFLOAD=1"
  "OAT_ZERO_ACTIVATION_OFFLOADING=1"
  "OAT_ZERO_SAVE_CKPT=0"
  "OAT_ZERO_AUTO_RESUME=0"
  "OAT_ZERO_WATCHDOG_REQUEUE=0"
  "OAT_ZERO_WATCHDOG_STALE_SECONDS=0"
  "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
  "OAT_ZERO_REQUIRE_EXISTING_DATA=1"
  "OAT_ZERO_APPEND_MANIFEST=1"
)

if [[ "$MODE" == "config" ]]; then
  common_env+=("OAT_ZERO_COMPARATIVE_CONFIG_ONLY=1")
else
  common_env+=("OAT_ZERO_SBATCH_HOLD=1")
fi

echo "[graph-7b-recovery] mode=${MODE} vllm_sleep=0 vllm_ratio=0.10 checkpoints=0"
echo "[graph-7b-recovery] placement=lowprio/mltheory nodes=node103,node104,node208"

e4_manifest="$ROOT_DIR/var/artifacts/gce4_taucontrol_7b_full_2xa6000_v1_comparative_jobs.tsv"
e5_manifest="$ROOT_DIR/var/artifacts/gce5_haarnoja_7b_v1_comparative_jobs.tsv"
e4_before=$(wc -l < "$e4_manifest")
e5_before=$(wc -l < "$e5_manifest")

env "${common_env[@]}" \
  OAT_ZERO_7B_ANALYTICAL_APPROVED=1 \
  RUN_STAMP_PREFIX=gce4_taucontrol_7b_full_2xa6000_v1 \
  OAT_ZERO_TRAIN_SEEDS=43,44,45 \
  OAT_ZERO_ONLY_ARMS=grpo,xdr_tau0p05,xdr_tau_control \
  bash "$SCRIPT_DIR/launch_e4_tau_control_7b.sh"

env "${common_env[@]}" \
  OAT_ZERO_7B_ANALYTICAL_APPROVED=1 \
  RUN_STAMP_PREFIX=gce5_haarnoja_7b_v1 \
  OAT_ZERO_TRAIN_SEEDS=43,44,45 \
  OAT_ZERO_ONLY_ARMS=xdr_sac_dual \
  OAT_ZERO_INCLUDE_XDR_SAC_DUAL_ARM=1 \
  OAT_ZERO_XDR_SAC_DUAL_BASE=0.05 \
  OAT_ZERO_XDR_SAC_DUAL_RATIO=0.8 \
  OAT_ZERO_XDR_SAC_DUAL_WARMUP_STEPS=64 \
  OAT_ZERO_XDR_SAC_DUAL_MIN_TAU=0.005 \
  OAT_ZERO_XDR_SAC_DUAL_MAX_TAU=0.5 \
  OAT_ZERO_XDR_SAC_DUAL_ALPHA_LR=0.003 \
  bash "$SCRIPT_DIR/launch_e4_tau_control.sh" 7b

if [[ "$MODE" == "config" ]]; then
  echo "[graph-7b-recovery] configuration validated; no jobs submitted"
  exit 0
fi

mapfile -t new_ids < <(
  {
    tail -n "+$((e4_before + 1))" "$e4_manifest"
    tail -n "+$((e5_before + 1))" "$e5_manifest"
  } | awk -F '\t' 'NF >= 3 {print $3}'
)
if [[ "${#new_ids[@]}" -ne 12 ]]; then
  echo "Expected 12 held recovery jobs, found ${#new_ids[@]}; leaving them held." >&2
  exit 1
fi

for job_id in "${new_ids[@]}"; do
  job_dump="$(scontrol show job -o "$job_id")"
  for required in \
    'JobState=PENDING' \
    'OAT_ZERO_VLLM_SLEEP=0' \
    'OAT_ZERO_SAVE_CKPT=0' \
    'gres/gpu:a6000=2'
  do
    if [[ "$job_dump" != *"$required"* ]]; then
      echo "Held-job audit failed for ${job_id}: missing ${required}; leaving cohort held." >&2
      exit 1
    fi
  done
done

old_jobs=(30012414 30012415 30012416 30012417 30012418 30012419)
scancel "${old_jobs[@]}"
job_csv="$(IFS=,; echo "${new_ids[*]}")"
scontrol release "$job_csv"

receipt="$ROOT_DIR/var/artifacts/graph_7b_allocator_recovery_$(date +%Y%m%d_%H%M%S).tsv"
{
  printf 'status\tjob_id\n'
  printf 'cancelled_superseded\t%s\n' "${old_jobs[@]}"
  printf 'submitted_released\t%s\n' "${new_ids[@]}"
} > "$receipt"
echo "[graph-7b-recovery] released=${job_csv}"
echo "[graph-7b-recovery] receipt=${receipt}"
