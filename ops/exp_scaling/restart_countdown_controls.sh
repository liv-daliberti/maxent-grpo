#!/usr/bin/env bash
# Restart the six-arm-per-scale Countdown controls after a silent actor crash.
# Existing run stamps are intentional: OAT creates a new debug attempt under
# the same run directory, auto-resumes from its highest checkpoint, and the
# analysis can retain the original pre-checkpoint trajectory.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ "${OAT_ZERO_RESTART_APPROVED:-0}" != "1" ]]; then
  echo "Set OAT_ZERO_RESTART_APPROVED=1 to submit replacement jobs." >&2
  exit 1
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
manifest="$ROOT_DIR/var/artifacts/countdown_control_restarts_${timestamp}.tsv"
printf 'scale\tarm\tseed\tjob_id\trun_stamp\n' > "$manifest"

submit_one() {
  local scale="$1" arm="$2" seed="$3"
  local model stamp_prefix samples max_train epochs eval_prompt_interval save_num node gres time_limit
  local variant tau
  case "$scale" in
    05b)
      model=qwen2.5-0.5b-instruct
      stamp_prefix=cde1_05b
      samples=16
      max_train=384
      epochs=5
      eval_prompt_interval=96
      save_num=12
      node=node105
      gres=gpu:a5000:1
      time_limit=12:00:00
      ;;
    3b)
      model=qwen2.5-3b-instruct
      stamp_prefix=cde1_3b
      samples=32
      max_train=384
      epochs=5
      eval_prompt_interval=96
      save_num=8
      node=node302
      gres=gpu:a100:1
      time_limit=48:00:00
      ;;
    *)
      echo "Unknown scale: $scale" >&2
      return 1
      ;;
  esac

  case "$arm" in
    grpo)
      variant=grpo
      tau=inf
      ;;
    xdr_tau0p05)
      variant=xdr
      tau=0.05
      ;;
    *)
      echo "Unknown arm: $arm" >&2
      return 1
      ;;
  esac

  local run_stamp="${stamp_prefix}_${arm}_s${seed}"
  local local_root="/tmp/${USER}/maxent-grpo-oat-zero-${run_stamp}"
  local exports="ALL"
  exports+=",RUN_STAMP=${run_stamp}"
  exports+=",OAT_ZERO_SEED=${seed}"
  exports+=",OAT_ZERO_VARIANT=${variant}"
  exports+=",OAT_ZERO_MODEL=${model}"
  exports+=",OAT_ZERO_DATA_ROOT=${ROOT_DIR}/var/data/exact_countdown_easy3_probe"
  exports+=",OAT_ZERO_NUM_SAMPLES=${samples}"
  exports+=",OAT_ZERO_MAX_TRAIN=${max_train}"
  exports+=",OAT_ZERO_MAX_QUERIES=100000000"
  exports+=",OAT_ZERO_NUM_PROMPT_EPOCH=${epochs}"
  exports+=",OAT_ZERO_EVAL_PROMPT_INTERVAL=${eval_prompt_interval}"
  exports+=",OAT_ZERO_EVAL_MODE_COVERAGE_K=8"
  exports+=",OAT_ZERO_EVAL_MODE_COVERAGE_TEMPERATURE=1.0"
  exports+=",OAT_ZERO_SAVE_STEPS=1152"
  exports+=",OAT_ZERO_SAVE_FROM=1152"
  exports+=",OAT_ZERO_SAVE_CKPT=1"
  exports+=",OAT_ZERO_MAX_SAVE_NUM=${save_num}"
  exports+=",OAT_ZERO_USE_WB=0"
  exports+=",OAT_ZERO_PROMPT_TEMPLATE=qwen_boxed"
  exports+=",OAT_ZERO_PROMPT_MAX_LENGTH=256"
  exports+=",OAT_ZERO_GENERATE_MAX_LENGTH=192"
  exports+=",OAT_ZERO_EVAL_GENERATE_MAX_LENGTH=192"
  exports+=",OAT_ZERO_EVAL_BATCH_SIZE=64"
  exports+=",OAT_ZERO_LOCAL_ROOT=${local_root}"
  exports+=",OAT_ZERO_RND_SEED=0"
  exports+=",OAT_ZERO_LEARNING_RATE=0.0000002"
  exports+=",OAT_ZERO_XDR_MODE_ADAPTIVE=0"
  exports+=",OAT_ZERO_XDR_TAU_CONTROL_TARGET_RATIO=0.0"
  exports+=",OAT_ZERO_POLICY_ENTROPY_COEF=0.0"
  exports+=",OAT_ZERO_XDR_TAU=${tau}"
  exports+=",OAT_ZERO_SEED_ENTROPY_ALPHA=0.0"
  exports+=",OAT_ZERO_AUTO_RESUME=1"
  exports+=",OAT_ZERO_WATCHDOG_STALE_SECONDS=2700"
  exports+=",OAT_ZERO_WATCHDOG_STARTUP_GRACE_SECONDS=3600"
  exports+=",OAT_ZERO_WATCHDOG_POLL_SECONDS=60"
  exports+=",OAT_ZERO_WATCHDOG_REQUEUE=1"
  exports+=",OAT_ZERO_WATCHDOG_MAX_RESTARTS=4"

  local job_id
  job_id="$(
    sbatch --parsable --export="$exports" --nodelist="$node" --gres="$gres" \
      --mem=96G --time="$time_limit" "$ROOT_DIR/ops/slurm/train_node302.slurm"
  )"
  printf '%s\t%s\t%s\t%s\t%s\n' "$scale" "$arm" "$seed" "$job_id" "$run_stamp" \
    >> "$manifest"
  echo "[restart] scale=${scale} arm=${arm} seed=${seed} job=${job_id}"
}

for scale in 05b 3b; do
  for seed in 43 44 45; do
    submit_one "$scale" grpo "$seed"
    submit_one "$scale" xdr_tau0p05 "$seed"
  done
done

echo "[restart] manifest=${manifest}"

if [[ "${OAT_ZERO_CANCEL_STUCK_JOBS:-0}" == "1" ]]; then
  old_jobs=(
    30001013 30001014 30001015 30001016 30001017 30001018
    30001019 30001020 30001021 30001022 30001023 30001024
  )
  scancel "${old_jobs[@]}"
  echo "[restart] cancelled_old_jobs=${old_jobs[*]}"
fi
