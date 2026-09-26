#!/usr/bin/env bash
# Use otherwise-idle GPU pools for two narrow graph-coloring 3B recoveries.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
MODE="${1:-config}"
case "$MODE" in
  config|full|dual-config|dual-retry) ;;
  *) echo "Usage: $0 {config|full|dual-config|dual-retry}" >&2; exit 2 ;;
esac

if [[ "$MODE" == full || "$MODE" == dual-retry ]] \
  && [[ "${OAT_ZERO_RETRY_GRAPH_3B_APPROVED:-0}" != "1" ]]; then
  echo "Set OAT_ZERO_RETRY_GRAPH_3B_APPROVED=1 to submit the two canaries." >&2
  exit 2
fi

ops_root="${OAT_ZERO_RECOVERY_OPS_ROOT:-$ROOT_DIR/var/artifacts/source_snapshots/cde4_7b_fixed_recovery_ops_dbc15d217ae3145a1e48a885ef8426428f518b33b46a7e91d8f77f70f4204cfd/ops}"
for required in \
  repo_env.sh \
  run_experiment.sh \
  train.sh \
  resolve_eval_cadence.py \
  slurm/train_node302.slurm; do
  if [[ ! -f "$ops_root/$required" ]]; then
    echo "Missing frozen recovery ops file: $ops_root/$required" >&2
    exit 1
  fi
done

feedback_manifest="$ROOT_DIR/var/artifacts/gce4_taucontrol_3b_feedback_2xa6000_cs_v1_comparative_jobs.tsv"
dual_manifest="$ROOT_DIR/var/artifacts/gce5_haarnoja_3b_2xa5000_v1_comparative_jobs.tsv"
for manifest in "$feedback_manifest" "$dual_manifest"; do
  if [[ ! -f "$manifest" ]]; then
    echo "Missing recovery manifest: $manifest" >&2
    exit 1
  fi
done

common_env=(
  "OAT_ZERO_OPS_SNAPSHOT_ROOT=${ops_root}"
  "OAT_ZERO_APPEND_MANIFEST=1"
  "OAT_ZERO_SBATCH_HOLD=1"
  "OAT_ZERO_TRAIN_MEMORY=96G"
  "OAT_ZERO_N_GPU=2"
  "OAT_ZERO_NUM_GPUS_PER_ACTOR=2"
  "OAT_ZERO_ROLLOUT_BATCH_SIZE=2"
  "OAT_ZERO_ROLLOUT_BATCH_SIZE_PER_DEVICE=1"
  "OAT_ZERO_TRAIN_BATCH_SIZE=32"
  "OAT_ZERO_TRAIN_BATCH_SIZE_PER_DEVICE=16"
  "OAT_ZERO_VLLM_GPU_RATIO=0.25"
  "OAT_ZERO_VLLM_SLEEP=1"
  "OAT_ZERO_ADAM_OFFLOAD=1"
  "OAT_ZERO_ACTIVATION_OFFLOADING=1"
  "OAT_ZERO_SAVE_CKPT=1"
  "OAT_ZERO_SAVE_STEPS=1024"
  "OAT_ZERO_SAVE_FROM=1024"
  "OAT_ZERO_MAX_SAVE_NUM=1"
  "OAT_ZERO_AUTO_RESUME=1"
  "OAT_ZERO_WATCHDOG_STALE_SECONDS=2700"
  "OAT_ZERO_WATCHDOG_STARTUP_GRACE_SECONDS=3600"
  "OAT_ZERO_WATCHDOG_POLL_SECONDS=30"
  "OAT_ZERO_WATCHDOG_REQUEUE=1"
  "OAT_ZERO_WATCHDOG_MAX_RESTARTS=8"
  "PYTORCH_CUDA_ALLOC_CONF="
)
if [[ "$MODE" == config || "$MODE" == dual-config ]]; then
  common_env+=("OAT_ZERO_COMPARATIVE_CONFIG_ONLY=1")
fi

submit_feedback() {
  env "${common_env[@]}" \
    RUN_STAMP_PREFIX=gce4_taucontrol_3b_feedback_2xa6000_cs_v1 \
    OAT_ZERO_TRAIN_SEEDS=45 \
    OAT_ZERO_ONLY_ARMS=xdr_tau_control \
    OAT_ZERO_INCLUDE_XDR_TAU_CONTROL_ARM=1 \
    OAT_ZERO_TRAIN_NODELIST=node302 \
    OAT_ZERO_TRAIN_GRES=gpu:a100:2 \
    OAT_ZERO_TRAIN_PARTITION=mltheory \
    OAT_ZERO_TRAIN_ACCOUNT=mltheory \
    bash "$SCRIPT_DIR/launch_e1_3b.sh"
}

submit_dual_canary() {
  env "${common_env[@]}" \
    RUN_STAMP_PREFIX=gce5_haarnoja_3b_2xa5000_v1 \
    OAT_ZERO_TRAIN_SEEDS=43 \
    OAT_ZERO_ONLY_ARMS=xdr_sac_dual \
    OAT_ZERO_INCLUDE_XDR_SAC_DUAL_ARM=1 \
    OAT_ZERO_XDR_SAC_DUAL_BASE=0.05 \
    OAT_ZERO_XDR_SAC_DUAL_RATIO=0.8 \
    OAT_ZERO_XDR_SAC_DUAL_WARMUP_STEPS=64 \
    OAT_ZERO_XDR_SAC_DUAL_MIN_TAU=0.005 \
    OAT_ZERO_XDR_SAC_DUAL_MAX_TAU=0.5 \
    OAT_ZERO_XDR_SAC_DUAL_ALPHA_LR=0.003 \
    OAT_ZERO_TRAIN_BATCH_SIZE_PER_DEVICE=4 \
    OAT_ZERO_TRAIN_NODELIST=node204 \
    OAT_ZERO_TRAIN_GRES=gpu:a5000:2 \
    OAT_ZERO_TRAIN_PARTITION=lowprio \
    OAT_ZERO_TRAIN_ACCOUNT=mltheory \
    bash "$SCRIPT_DIR/launch_e1_3b.sh"
}

if [[ "$MODE" == config ]]; then
  submit_feedback
  submit_dual_canary
  echo "[graph-3b-canaries] configurations passed; no jobs submitted"
  exit 0
fi

if [[ "$MODE" == dual-config ]]; then
  submit_dual_canary
  echo "[graph-3b-canaries] dual microbatch-4 configuration passed; no job submitted"
  exit 0
fi

if [[ "$MODE" == dual-retry ]]; then
  dual_before="$(wc -l < "$dual_manifest")"
  submit_dual_canary
  mapfile -t dual_ids < <(
    tail -n "+$((dual_before + 1))" "$dual_manifest" |
      awk -F '\t' '$1 == "xdr_sac_dual" && $2 == "43" && $3 ~ /^[0-9]+$/ {print $3}'
  )
  if (( ${#dual_ids[@]} != 1 )); then
    echo "Dual recovery submission incomplete; any created job remains held." >&2
    exit 1
  fi
  dual_id="${dual_ids[0]}"
  dual_dump="$(scontrol show job -o "$dual_id")"
  for required in \
    'JobState=PENDING' \
    'ReqNodeList=node204' \
    'gres/gpu:a5000:2' \
    'OAT_ZERO_TRAIN_BATCH_SIZE_PER_DEVICE=4' \
    "OAT_ZERO_OPS_SNAPSHOT_ROOT=${ops_root}"; do
    if [[ "$dual_dump" != *"$required"* ]]; then
      echo "Dual held-job audit failed: missing $required" >&2
      exit 1
    fi
  done
  scontrol release "$dual_id"
  echo "[graph-3b-canaries] released microbatch-4 dual-s43=$dual_id"
  exit 0
fi

feedback_before="$(wc -l < "$feedback_manifest")"
dual_before="$(wc -l < "$dual_manifest")"
submit_feedback
submit_dual_canary

mapfile -t feedback_ids < <(
  tail -n "+$((feedback_before + 1))" "$feedback_manifest" |
    awk -F '\t' '$1 == "xdr_tau_control" && $2 == "45" && $3 ~ /^[0-9]+$/ {print $3}'
)
mapfile -t dual_ids < <(
  tail -n "+$((dual_before + 1))" "$dual_manifest" |
    awk -F '\t' '$1 == "xdr_sac_dual" && $2 == "43" && $3 ~ /^[0-9]+$/ {print $3}'
)
if (( ${#feedback_ids[@]} != 1 || ${#dual_ids[@]} != 1 )); then
  echo "Recovery submission incomplete; any created jobs remain held." >&2
  exit 1
fi

feedback_id="${feedback_ids[0]}"
dual_id="${dual_ids[0]}"
feedback_dump="$(scontrol show job -o "$feedback_id")"
dual_dump="$(scontrol show job -o "$dual_id")"
for required in \
  'JobState=PENDING' \
  'ReqNodeList=node302' \
  'gres/gpu:a100:2' \
  "OAT_ZERO_OPS_SNAPSHOT_ROOT=${ops_root}"; do
  if [[ "$feedback_dump" != *"$required"* ]]; then
    echo "Feedback held-job audit failed: missing $required" >&2
    exit 1
  fi
done
for required in \
  'JobState=PENDING' \
  'ReqNodeList=node204' \
  'gres/gpu:a5000:2' \
  "OAT_ZERO_OPS_SNAPSHOT_ROOT=${ops_root}"; do
  if [[ "$dual_dump" != *"$required"* ]]; then
    echo "Dual held-job audit failed: missing $required" >&2
    exit 1
  fi
done

scontrol release "$feedback_id" "$dual_id"
echo "[graph-3b-canaries] released feedback-s45=$feedback_id dual-s43=$dual_id"
