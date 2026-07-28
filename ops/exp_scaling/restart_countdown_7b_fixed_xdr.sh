#!/usr/bin/env bash
# Restart only Countdown-7B fixed-xDr seeds 43/44 under the original E4 stamp.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
PHASE="${1:-retry}"
case "$PHASE" in
  config|retry) ;;
  *) echo "Usage: $0 {config|retry}" >&2; exit 2 ;;
esac

if [[ "$PHASE" == retry && "${OAT_ZERO_RETRY_FAILED_APPROVED:-0}" != "1" ]]; then
  echo "Set OAT_ZERO_RETRY_FAILED_APPROVED=1 to submit the two recovery jobs." >&2
  exit 2
fi

manifest="$ROOT_DIR/var/artifacts/cde4_taucontrol_7b_full_2xa6000_cs_v1_comparative_jobs.tsv"
if [[ ! -f "$manifest" ]]; then
  echo "Countdown-7B fixed-xDr recovery requires the original manifest: $manifest" >&2
  exit 1
fi

recovery_nodes="${OAT_ZERO_RECOVERY_NODELIST:-node103,node104}"
recovery_partition="${OAT_ZERO_RECOVERY_PARTITION:-lowprio}"
recovery_account="${OAT_ZERO_RECOVERY_ACCOUNT:-mltheory}"
runtime_ops_root="${OAT_ZERO_RECOVERY_OPS_ROOT:-$ROOT_DIR/ops}"
for required in \
  repo_env.sh \
  run_experiment.sh \
  train.sh \
  resolve_eval_cadence.py \
  slurm/train_node302.slurm; do
  if [[ ! -f "$runtime_ops_root/$required" ]]; then
    echo "Missing frozen recovery ops file: $runtime_ops_root/$required" >&2
    exit 1
  fi
done

common_recovery_env=(
  "OAT_ZERO_TRAIN_NODELIST=${recovery_nodes}"
  "OAT_ZERO_TRAIN_GRES=gpu:a6000:2"
  "OAT_ZERO_TRAIN_PARTITION=${recovery_partition}"
  "OAT_ZERO_TRAIN_ACCOUNT=${recovery_account}"
  "OAT_ZERO_TRAIN_MEMORY=192G"
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
  "OAT_ZERO_AUTO_RESUME=1"
  "OAT_ZERO_WATCHDOG_STALE_SECONDS=2700"
  "OAT_ZERO_WATCHDOG_STARTUP_GRACE_SECONDS=3600"
  "OAT_ZERO_WATCHDOG_POLL_SECONDS=30"
  "OAT_ZERO_WATCHDOG_REQUEUE=1"
  "OAT_ZERO_WATCHDOG_MAX_RESTARTS=8"
  "OAT_ZERO_OPS_SNAPSHOT_ROOT=${runtime_ops_root}"
  "PYTORCH_CUDA_ALLOC_CONF="
)

if [[ "$PHASE" == config ]]; then
  env "${common_recovery_env[@]}" \
    OAT_ZERO_7B_ANALYTICAL_APPROVED=1 \
    OAT_ZERO_COMPARATIVE_CONFIG_ONLY=1 \
    OAT_ZERO_TRAIN_SEEDS=43,44 \
    OAT_ZERO_ONLY_ARMS=xdr_tau0p05 \
    bash "$SCRIPT_DIR/launch_e4_tau_control_7b_countdown.sh"
  echo "[countdown-7b-fixed] recovery configuration passed; no jobs submitted"
  exit 0
fi

lines_before="$(wc -l < "$manifest")"
env "${common_recovery_env[@]}" \
  OAT_ZERO_7B_ANALYTICAL_APPROVED=1 \
  OAT_ZERO_APPEND_MANIFEST=1 \
  OAT_ZERO_SBATCH_HOLD=1 \
  OAT_ZERO_TRAIN_SEEDS=43,44 \
  OAT_ZERO_ONLY_ARMS=xdr_tau0p05 \
  bash "$SCRIPT_DIR/launch_e4_tau_control_7b_countdown.sh"

mapfile -t job_ids < <(
  tail -n "+$((lines_before + 1))" "$manifest" |
    awk -F '\t' '$1 == "xdr_tau0p05" && ($2 == "43" || $2 == "44") && $3 ~ /^[0-9]+$/ {print $3}'
)
if (( ${#job_ids[@]} != 2 )); then
  echo "Countdown-7B fixed-xDr recovery incomplete (${#job_ids[@]}/2); jobs remain held" >&2
  exit 1
fi

scontrol release "${job_ids[@]}"
echo "[countdown-7b-fixed] released seeds 43/44: ${job_ids[*]}"
