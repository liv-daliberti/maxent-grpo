#!/usr/bin/env bash
# Launch E22-v2's matched free-form control and base-preserving Haarnoja dual.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT_DIR/ops/repo_env.sh"

phase="${1:-}"
case "$phase" in
  config|full) ;;
  *)
    echo "Usage: $0 {config|full}" >&2
    exit 1
    ;;
esac

PYTHON_BIN="${OAT_ZERO_PYTHON:-$ROOT_DIR/var/seed_paper_eval/paper310/bin/python}"
AGGRESSIVE_05B="${OAT_ZERO_E27_AGGRESSIVE_05B:-0}"
PROTOCOL="$ROOT_DIR/paper/preregistration/e22_modebench_freeform_token_maxent_v2.md"
CALIBRATION="$ROOT_DIR/paper/results/e22_freeform_dual_v1_calibration.json"
MODEL_ROOT="$ROOT_DIR/var/cache/huggingface/transformers/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
SOURCE_HASH=217547637154ed74c2356eafec6dc885914793f8bb530c2b6918ca9a2c245448
EXECUTION_HASH=05d43e4ae78d75d9e98c5b3d07d6ebd88cc545cc9039774ad0b6bf7cfebc05ce
SOURCE_ROOT="$ROOT_DIR/var/artifacts/source_snapshots/e21_math_conditional_token_${SOURCE_HASH}/src"
OPS_ROOT="$ROOT_DIR/var/artifacts/source_snapshots/e21_math_conditional_token_ops_${EXECUTION_HASH}/ops"
GRAPH_DATA_ROOT="$ROOT_DIR/var/data/exact_answer_mode_probe"
COUNTDOWN_DATA_ROOT="$ROOT_DIR/var/data/exact_countdown_easy3_probe"
GRAPH_PREFIX=gce22_freeform_conditional_dual_05b_v2
COUNTDOWN_PREFIX=cde22_freeform_conditional_dual_05b_v2
IDENTITY="$ROOT_DIR/var/artifacts/e22_modebench_freeform_conditional_dual_05b_v2_identity.json"
GRAPH_TARGET=1.2981748407085736
COUNTDOWN_TARGET=1.0776875459899504
ONLY_ARMS=grpo,maxent_dual
EXPECTED_JOBS_PER_TASK=6
IDENTITY_SCHEMA=e22_modebench_freeform_conditional_dual_v2
CAMPAIGN_LABEL=e22-v2
DUAL_MAX_ALPHA=0.00015
DUAL_ALPHA_LR=0.005

if [[ "$AGGRESSIVE_05B" == 1 ]]; then
  PROTOCOL="$ROOT_DIR/paper/preregistration/e27_modebench_freeform_dual_05b_aggressive.md"
  CALIBRATION="$ROOT_DIR/paper/results/e27_modebench_freeform_dual_05b_aggressive_calibration.json"
  GRAPH_PREFIX=gce27_freeform_conditional_dual_05b_v1
  COUNTDOWN_PREFIX=cde27_freeform_conditional_dual_05b_v1
  IDENTITY="$ROOT_DIR/var/artifacts/e27_modebench_freeform_conditional_dual_05b_v1_identity.json"
  GRAPH_TARGET=1.622718550885717
  COUNTDOWN_TARGET=1.347109432487438
  ONLY_ARMS=maxent_dual
  EXPECTED_JOBS_PER_TASK=3
  IDENTITY_SCHEMA=e27_modebench_freeform_conditional_dual_aggressive_v1
  CAMPAIGN_LABEL=e27
  DUAL_MAX_ALPHA=0.00060
  DUAL_ALPHA_LR=0.010
elif [[ "$AGGRESSIVE_05B" != 0 ]]; then
  echo "OAT_ZERO_E27_AGGRESSIVE_05B must be 0 or 1" >&2
  exit 1
fi

for required in \
  "$PYTHON_BIN" \
  "$PROTOCOL" \
  "$CALIBRATION" \
  "$MODEL_ROOT/config.json" \
  "$MODEL_ROOT/tokenizer.json" \
  "$MODEL_ROOT/model.safetensors" \
  "$SOURCE_ROOT/oat_drgrpo/__init__.py" \
  "$OPS_ROOT/submit_countdown_comparative.sh" \
  "$GRAPH_DATA_ROOT/train/dataset_dict.json" \
  "$GRAPH_DATA_ROOT/eval/dataset_dict.json" \
  "$COUNTDOWN_DATA_ROOT/train/dataset_dict.json" \
  "$COUNTDOWN_DATA_ROOT/eval/dataset_dict.json"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing frozen E22-v2 prerequisite: $required" >&2
    exit 1
  fi
done

if ! grep -q '^\*\*Status: FROZEN' "$PROTOCOL"; then
  echo "E22-v2 protocol is not frozen" >&2
  exit 1
fi

export OAT_ZERO_REPO_ROOT="$ROOT_DIR"
export OAT_ZERO_OPS_SNAPSHOT_ROOT="$OPS_ROOT"
export OAT_ZERO_CAMPAIGN_SOURCE_ROOT="$SOURCE_ROOT"
export OAT_ZERO_PRETRAIN="$MODEL_ROOT"
export OAT_ZERO_REQUIRE_EXISTING_DATA=1
export OAT_ZERO_COMPARATIVE_REBUILD=0
export OAT_ZERO_APPEND_MANIFEST=0
export OAT_ZERO_COMPARATIVE_MODEL=qwen2.5-0.5b-instruct
export OAT_ZERO_TRAIN_SEEDS=43,44,45
export OAT_ZERO_ONLY_ARMS="$ONLY_ARMS"
export OAT_ZERO_XDR_TAUS=""
export OAT_ZERO_INCLUDE_TOKEN_ENTROPY_ARM=0
export OAT_ZERO_INCLUDE_SEED_ARM=0
export OAT_ZERO_INCLUDE_XDR_ADAPT_ARM=0
export OAT_ZERO_INCLUDE_XDR_TAU_CONTROL_ARM=0
export OAT_ZERO_INCLUDE_XDR_SAC_DUAL_ARM=0
export OAT_ZERO_INCLUDE_MAXENT_ARM=0
export OAT_ZERO_INCLUDE_MAXENT_CONTROL_ARM=0
export OAT_ZERO_INCLUDE_MAXENT_DUAL_ARM=1
export OAT_ZERO_INCLUDE_MAXENT_LENGTH_DUAL_ARM=0

export OAT_ZERO_MAXENT_OBJECTIVE=conditional_token_mean
export OAT_ZERO_MAXENT_DUAL_BASE_ALPHA=0.000075
export OAT_ZERO_MAXENT_DUAL_RATIO=1.0
export OAT_ZERO_MAXENT_DUAL_WARMUP_STEPS=64
export OAT_ZERO_MAXENT_DUAL_MIN_ALPHA=0.000075
export OAT_ZERO_MAXENT_DUAL_MAX_ALPHA="$DUAL_MAX_ALPHA"
export OAT_ZERO_MAXENT_DUAL_ALPHA_LR="$DUAL_ALPHA_LR"
export OAT_ZERO_MAXENT_LENGTH_TARGET=0
export OAT_ZERO_POLICY_ENTROPY_COEF=0

export OAT_ZERO_NUM_SAMPLES=16
export OAT_ZERO_LEARNING_RATE=0.0000002
export OAT_ZERO_MAX_QUERIES=100000000
export OAT_ZERO_NUM_PROMPT_EPOCH=5
export OAT_ZERO_NUM_PPO_EPOCHS=1
export OAT_ZERO_MAX_NORM=1
export OAT_ZERO_BETA=0
export OAT_ZERO_IGNORE_NO_EOS=0
export OAT_ZERO_TRAIN_BATCH_SIZE=16
export OAT_ZERO_TRAIN_BATCH_SIZE_PER_DEVICE=1
export OAT_ZERO_ROLLOUT_BATCH_SIZE=1
export OAT_ZERO_ROLLOUT_BATCH_SIZE_PER_DEVICE=1
export OAT_ZERO_N_GPU=1
export OAT_ZERO_NUM_GPUS_PER_ACTOR=1

export OAT_ZERO_PROMPT_TEMPLATE=qwen_boxed
export OAT_ZERO_INPUT_KEY=problem
export OAT_ZERO_OUTPUT_KEY=answer
export OAT_ZERO_EVAL_INPUT_KEY=problem
export OAT_ZERO_EVAL_OUTPUT_KEY=answer
export OAT_ZERO_TEST_SPLIT=multi_answer
export OAT_ZERO_VERIFIER_VERSION=fast
export OAT_ZERO_PROMPT_MAX_LENGTH=256
export OAT_ZERO_GENERATE_MAX_LENGTH=192
export OAT_ZERO_EVAL_GENERATE_MAX_LENGTH=192
export OAT_ZERO_MAX_MODEL_LEN=512
export OAT_ZERO_TEMPERATURE=1
export OAT_ZERO_TOP_P=1
export OAT_ZERO_EVAL_TEMPERATURE=0
export OAT_ZERO_EVAL_MODE_COVERAGE_K=8
export OAT_ZERO_EVAL_MODE_COVERAGE_TEMPERATURE=1
export OAT_ZERO_EVAL_BATCH_SIZE=64
export OAT_ZERO_SYNC_PARAMS_EVERY=1

export OAT_ZERO_CANONICAL_ACTION_TASK=none
export OAT_ZERO_CANONICAL_GRAPH_ACTIONS=0
export OAT_ZERO_CANONICAL_GRAPH_LEARNER_SAMPLING=0
export OAT_ZERO_CANONICAL_GRAPH_FIXED_SHAPE_SAMPLING=0
export OAT_ZERO_ZERO_STAGE=2
export OAT_ZERO_VLLM_GPU_RATIO=0.25
export OAT_ZERO_ENABLE_FLASH_ATTN=0
export OAT_ZERO_ADAM_OFFLOAD=0
export OAT_ZERO_ACTIVATION_OFFLOADING=0
export OAT_ZERO_COLLOCATE=1
export OAT_ZERO_VLLM_SLEEP=1
export VLLM_USE_V1=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export OAT_ZERO_SAVE_CKPT=1
export OAT_ZERO_MAX_SAVE_NUM=2
export OAT_ZERO_AUTO_RESUME=1
export OAT_ZERO_WATCHDOG_REQUEUE=1
export OAT_ZERO_WATCHDOG_MAX_RESTARTS=6
export OAT_ZERO_TRAIN_NODELIST="${OAT_ZERO_E22_TRAIN_NODELIST:-node105}"
export OAT_ZERO_TRAIN_GRES="${OAT_ZERO_E22_TRAIN_GRES:-gpu:a5000:1}"
export OAT_ZERO_TRAIN_PARTITION="${OAT_ZERO_E22_TRAIN_PARTITION:-mltheory}"
export OAT_ZERO_TRAIN_ACCOUNT="${OAT_ZERO_E22_TRAIN_ACCOUNT:-mltheory}"
export OAT_ZERO_TRAIN_MEMORY="${OAT_ZERO_E22_TRAIN_MEMORY:-64G}"
export OAT_ZERO_TRAIN_TIME_LIMIT="${OAT_ZERO_E22_TRAIN_TIME_LIMIT:-24:00:00}"

write_identity() {
  local protocol_sha256 launcher_sha256 calibration_sha256
  protocol_sha256="$(sha256sum "$PROTOCOL" | cut -d' ' -f1)"
  launcher_sha256="$(sha256sum "$0" | cut -d' ' -f1)"
  calibration_sha256="$(sha256sum "$CALIBRATION" | cut -d' ' -f1)"
  "$PYTHON_BIN" - "$IDENTITY" "$protocol_sha256" "$launcher_sha256" \
    "$calibration_sha256" "$SOURCE_HASH" "$EXECUTION_HASH" \
    "$GRAPH_TARGET" "$COUNTDOWN_TARGET" "$IDENTITY_SCHEMA" "$ONLY_ARMS" \
    "$DUAL_MAX_ALPHA" "$DUAL_ALPHA_LR" <<'PY'
import json
import os
import pathlib
import sys
import tempfile

path = pathlib.Path(sys.argv[1])
payload = {
    "schema": sys.argv[9],
    "protocol_sha256": sys.argv[2],
    "launcher_sha256": sys.argv[3],
    "calibration_sha256": sys.argv[4],
    "source_hash": sys.argv[5],
    "execution_surface_hash": sys.argv[6],
    "objective": "conditional_token_mean",
    "controller": "base_preserving_haarnoja_log_alpha_adam",
    "arms": sys.argv[10].split(","),
    "tasks": {
        "graph_coloring": {"target_entropy": float(sys.argv[7])},
        "countdown": {"target_entropy": float(sys.argv[8])},
    },
    "alpha": {"base": 0.000075, "min": 0.000075, "max": float(sys.argv[11])},
    "alpha_lr": float(sys.argv[12]),
    "scale": "0.5B",
    "seeds": [43, 44, 45],
}
path.parent.mkdir(parents=True, exist_ok=True)
fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
with os.fdopen(fd, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
os.replace(temporary, path)
PY
}

submit_task() {
  local task="$1"
  if [[ "$task" == graph_coloring ]]; then
    export RUN_STAMP_PREFIX="$GRAPH_PREFIX"
    export OAT_ZERO_COMPARATIVE_TASK=graph_coloring
    export OAT_ZERO_COMPARATIVE_DATA_ROOT="$GRAPH_DATA_ROOT"
    export OAT_ZERO_MAXENT_DUAL_TARGET_ENTROPY="$GRAPH_TARGET"
    export OAT_ZERO_MAX_TRAIN=192
    export OAT_ZERO_EVAL_PROMPT_INTERVAL=48
    export OAT_ZERO_SAVE_STEPS=48
    export OAT_ZERO_SAVE_FROM=48
  else
    export RUN_STAMP_PREFIX="$COUNTDOWN_PREFIX"
    export OAT_ZERO_COMPARATIVE_TASK=countdown
    export OAT_ZERO_COMPARATIVE_DATA_PRESET=easy3
    export OAT_ZERO_COMPARATIVE_DATA_ROOT="$COUNTDOWN_DATA_ROOT"
    export OAT_ZERO_MAXENT_DUAL_TARGET_ENTROPY="$COUNTDOWN_TARGET"
    export OAT_ZERO_MAX_TRAIN=384
    export OAT_ZERO_EVAL_PROMPT_INTERVAL=96
    export OAT_ZERO_SAVE_STEPS=96
    export OAT_ZERO_SAVE_FROM=96
  fi
  "$OPS_ROOT/submit_countdown_comparative.sh"
}

if [[ "$phase" == config ]]; then
  export OAT_ZERO_COMPARATIVE_CONFIG_ONLY=1
  export OAT_ZERO_SBATCH_HOLD=0
  submit_task graph_coloring
  submit_task countdown
  echo "[$CAMPAIGN_LABEL] both free-form configurations passed; no jobs submitted"
  exit 0
fi

unset OAT_ZERO_COMPARATIVE_CONFIG_ONLY
export OAT_ZERO_SBATCH_HOLD=1
write_identity
export OAT_ZERO_PROTOCOL_IDENTITY="$IDENTITY"

for manifest in \
  "$ROOT_DIR/var/artifacts/${GRAPH_PREFIX}_comparative_jobs.tsv" \
  "$ROOT_DIR/var/artifacts/${COUNTDOWN_PREFIX}_comparative_jobs.tsv"; do
  if [[ -e "$manifest" ]]; then
    echo "Fresh $CAMPAIGN_LABEL prefix required; manifest already exists: $manifest" >&2
    exit 1
  fi
done

submit_task graph_coloring
submit_task countdown

job_ids=()
for spec in "$GRAPH_PREFIX|$GRAPH_TARGET" "$COUNTDOWN_PREFIX|$COUNTDOWN_TARGET"; do
  IFS='|' read -r prefix target <<< "$spec"
  manifest="$ROOT_DIR/var/artifacts/${prefix}_comparative_jobs.tsv"
  mapfile -t task_jobs < <(awk -F '\t' 'NR > 1 && $3 ~ /^[0-9]+$/ {print $3}' "$manifest")
  if [[ "${#task_jobs[@]}" -ne "$EXPECTED_JOBS_PER_TASK" ]]; then
    if [[ "${#job_ids[@]}" -gt 0 ]]; then
      scancel "${job_ids[@]}" || true
    fi
    if [[ "${#task_jobs[@]}" -gt 0 ]]; then
      scancel "${task_jobs[@]}" || true
    fi
    echo "$CAMPAIGN_LABEL ${prefix} cohort has ${#task_jobs[@]} jobs; expected $EXPECTED_JOBS_PER_TASK" >&2
    exit 1
  fi
  for job_id in "${task_jobs[@]}"; do
    job_record="$(scontrol show job "$job_id" -o)"
    for required in \
      'JobState=PENDING' 'Reason=JobHeldUser' \
      'OAT_ZERO_MAXENT_OBJECTIVE=conditional_token_mean' \
      "OAT_ZERO_MAXENT_DUAL_TARGET_ENTROPY=${target}" \
      'OAT_ZERO_MAXENT_DUAL_MIN_ALPHA=0.000075' \
      "OAT_ZERO_MAXENT_DUAL_MAX_ALPHA=${DUAL_MAX_ALPHA}" \
      "OAT_ZERO_MAXENT_DUAL_ALPHA_LR=${DUAL_ALPHA_LR}"; do
      if [[ "$job_record" != *"$required"* ]]; then
        scancel "${task_jobs[@]}" || true
        echo "$CAMPAIGN_LABEL held-job audit failed for $job_id: missing $required" >&2
        exit 1
      fi
    done
    if [[ "$AGGRESSIVE_05B" == 1 && "$job_record" != *'OAT_ZERO_VARIANT=maxent_dual'* ]]; then
      scancel "${task_jobs[@]}" || true
      echo "$CAMPAIGN_LABEL held-job audit failed for $job_id: not maxent_dual" >&2
      exit 1
    fi
  done
  job_ids+=("${task_jobs[@]}")
done

for job_id in "${job_ids[@]}"; do
  job_record="$(scontrol show job "$job_id" -o)"
  if [[ "$job_record" != *"Partition=${OAT_ZERO_TRAIN_PARTITION}"* ]] || \
     [[ "$job_record" != *"ReqNodeList=${OAT_ZERO_TRAIN_NODELIST}"* ]] || \
     [[ "$job_record" != *"gres/gpu:a5000=1"* ]]; then
    scancel "${job_ids[@]}" || true
    echo "$CAMPAIGN_LABEL held job $job_id does not attest the frozen placement" >&2
    exit 1
  fi
done

scontrol release "${job_ids[@]}"
echo "[$CAMPAIGN_LABEL] released ${#job_ids[@]} free-form jobs: ${job_ids[*]}"
