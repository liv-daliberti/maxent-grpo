#!/usr/bin/env bash
# Add only the entropy-feedback xDr arm to the landed 0.5B/3B scaling curves.
# Baseline and fixed-tau xDr already exist, so resubmitting them would waste
# compute. Each target delegates to its original E1 launcher to keep every
# non-method setting matched.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="${1:-all}"

launch_one() {
  local target="$1"
  local stamp launcher memory nodelist partition account
  case "$target" in
    countdown-05b)
      stamp=cde4_taucontrol_05b_feedback_v1
      launcher=launch_e1_05b_cd_pilot.sh
      memory=32G
      nodelist=node204
      partition=lowprio
      account=allcs
      ;;
    countdown-3b)
      stamp=cde4_taucontrol_3b_feedback_v1
      launcher=launch_e1_3b_cd_easy3.sh
      memory=96G
      nodelist=node302
      partition=mltheory
      account=mltheory
      ;;
    graph-05b)
      stamp=gce4_taucontrol_05b_feedback_v1
      launcher=launch_e1_05b_pilot.sh
      memory=32G
      nodelist=node204
      partition=lowprio
      account=allcs
      ;;
    graph-3b)
      stamp=gce4_taucontrol_3b_feedback_v1
      launcher=launch_e1_3b.sh
      memory=96G
      nodelist=node302
      partition=mltheory
      account=mltheory
      ;;
    *)
      echo "Unknown target: $target" >&2
      echo "Use all, countdown-05b, countdown-3b, graph-05b, or graph-3b." >&2
      return 2
      ;;
  esac

  echo "[feedback-extension] target=${target} stamp=${stamp}"
  RUN_STAMP_PREFIX="$stamp" \
  OAT_ZERO_ONLY_ARMS=xdr_tau_control \
  OAT_ZERO_INCLUDE_XDR_TAU_CONTROL_ARM=1 \
  OAT_ZERO_SAVE_CKPT=0 \
  OAT_ZERO_MAX_SAVE_NUM=1 \
  OAT_ZERO_TRAIN_MEMORY="$memory" \
  OAT_ZERO_TRAIN_NODELIST="$nodelist" \
  OAT_ZERO_TRAIN_PARTITION="$partition" \
  OAT_ZERO_TRAIN_ACCOUNT="$account" \
    bash "$SCRIPT_DIR/$launcher"
}

if [[ "$TARGET" == "all" ]]; then
  for target in countdown-05b countdown-3b graph-05b graph-3b; do
    launch_one "$target"
  done
else
  launch_one "$TARGET"
fi
