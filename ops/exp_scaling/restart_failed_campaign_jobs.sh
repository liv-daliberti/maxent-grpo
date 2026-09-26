#!/usr/bin/env bash
# Targeted recovery for the terminal failures in the maintained E4 campaign.
#
# Reuse the analytical run stamps so the monitor and curve parser retain all
# landed evaluations from earlier attempts. New allocations freeze both the
# Python objective and shell entrypoints, and the watchdog requeues failures
# from those immutable snapshots.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "${OAT_ZERO_RETRY_FAILED_APPROVED:-0}" != "1" ]]; then
  echo "Set OAT_ZERO_RETRY_FAILED_APPROVED=1 to submit three recovery jobs." >&2
  exit 2
fi

# These failed allocations all ran on node205 or on lower-memory recovery
# nodes. Keep replacements on the 48 GB CS A6000 pool while avoiding node205.
recovery_nodes="${OAT_ZERO_RECOVERY_NODELIST:-node206,node207}"

common_recovery_env=(
  "OAT_ZERO_TRAIN_NODELIST=${recovery_nodes}"
  "OAT_ZERO_TRAIN_GRES=gpu:a6000:2"
  "OAT_ZERO_TRAIN_PARTITION=cs"
  "OAT_ZERO_TRAIN_ACCOUNT=allcs"
  "OAT_ZERO_TRAIN_MEMORY=192G"
  "OAT_ZERO_N_GPU=2"
  "OAT_ZERO_NUM_GPUS_PER_ACTOR=2"
  "OAT_ZERO_ROLLOUT_BATCH_SIZE=2"
  "OAT_ZERO_ROLLOUT_BATCH_SIZE_PER_DEVICE=1"
  "OAT_ZERO_TRAIN_BATCH_SIZE=32"
  # More accumulation trades speed for headroom at long sampled sequences.
  "OAT_ZERO_TRAIN_BATCH_SIZE_PER_DEVICE=4"
  "OAT_ZERO_VLLM_GPU_RATIO=0.20"
  "OAT_ZERO_ADAM_OFFLOAD=1"
  "OAT_ZERO_ACTIVATION_OFFLOADING=1"
  "OAT_ZERO_AUTO_RESUME=1"
  "OAT_ZERO_WATCHDOG_STALE_SECONDS=2700"
  "OAT_ZERO_WATCHDOG_STARTUP_GRACE_SECONDS=3600"
  "OAT_ZERO_WATCHDOG_POLL_SECONDS=30"
  "OAT_ZERO_WATCHDOG_REQUEUE=1"
  "OAT_ZERO_WATCHDOG_MAX_RESTARTS=8"
  # The observed 7B optimizer failure had 13.35 GiB reserved but unallocated.
  "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
)

echo "[failed-recovery] nodes=${recovery_nodes} per_device_batch=4 vllm_ratio=0.20"

# Countdown 7B fixed-xDr seeds 43 and 44. Seed 43 has landed three passes and
# seed 44 two; a fresh attempt is stitched after the prior landed frontier.
env "${common_recovery_env[@]}" \
  OAT_ZERO_7B_ANALYTICAL_APPROVED=1 \
  OAT_ZERO_APPEND_MANIFEST=1 \
  OAT_ZERO_TRAIN_SEEDS=43,44 \
  OAT_ZERO_ONLY_ARMS=xdr_tau0p05 \
  bash "$SCRIPT_DIR/launch_e4_tau_control_7b_countdown.sh"

# Graph-coloring 3B feedback-xDr seed 45. Invoke the base 3B launcher directly
# because the general two-A6000 extension deliberately resets seeds and batch
# sizes for a full campaign, while this recovery must submit exactly one cell.
env "${common_recovery_env[@]}" \
  RUN_STAMP_PREFIX=gce4_taucontrol_3b_feedback_2xa6000_cs_v1 \
  OAT_ZERO_APPEND_MANIFEST=1 \
  OAT_ZERO_TRAIN_SEEDS=45 \
  OAT_ZERO_ONLY_ARMS=xdr_tau_control \
  OAT_ZERO_INCLUDE_XDR_TAU_CONTROL_ARM=1 \
  OAT_ZERO_SAVE_CKPT=1 \
  OAT_ZERO_SAVE_STEPS=1024 \
  OAT_ZERO_SAVE_FROM=1024 \
  OAT_ZERO_MAX_SAVE_NUM=1 \
  bash "$SCRIPT_DIR/launch_e1_3b.sh"

if [[ "${OAT_ZERO_E4_CONFIG_ONLY:-0}" == "1" || "${OAT_ZERO_COMPARATIVE_CONFIG_ONLY:-0}" == "1" ]]; then
  echo "[failed-recovery] configuration validated; no jobs submitted"
else
  echo "[failed-recovery] submitted Countdown 7B fixed-xDr s43/s44 and GC 3B feedback-xDr s45"
fi
