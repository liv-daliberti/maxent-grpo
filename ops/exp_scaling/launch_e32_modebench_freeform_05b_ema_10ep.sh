#!/usr/bin/env bash
# Launch the clean matched E32 0.5B free-form Dr.GRPO/EMA-MaxEnt cohort.
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
PROTOCOL="$ROOT_DIR/paper/preregistration/e32_modebench_freeform_05b_ema_10ep.md"
MODEL_ROOT="$ROOT_DIR/var/cache/huggingface/transformers/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
GRAPH_DATA_ROOT="$ROOT_DIR/var/data/exact_answer_mode_probe"
COUNTDOWN_DATA_ROOT="$ROOT_DIR/var/data/exact_countdown_easy3_probe"
GRAPH_PREFIX=gce32_freeform_05b_ema_10ep_v5
COUNTDOWN_PREFIX=cde32_freeform_05b_ema_10ep_v5
IDENTITY="$ROOT_DIR/var/artifacts/e32_freeform_05b_ema_10ep_v5_identity.json"
GRAPH_TARGET=1.622718550885717
COUNTDOWN_TARGET=1.347109432487438
EXPECTED_JOBS_PER_TASK=6

for required in \
  "$PYTHON_BIN" \
  "$PROTOCOL" \
  "$MODEL_ROOT/config.json" \
  "$MODEL_ROOT/tokenizer.json" \
  "$MODEL_ROOT/model.safetensors" \
  "$GRAPH_DATA_ROOT/train/dataset_dict.json" \
  "$GRAPH_DATA_ROOT/eval/dataset_dict.json" \
  "$COUNTDOWN_DATA_ROOT/train/dataset_dict.json" \
  "$COUNTDOWN_DATA_ROOT/eval/dataset_dict.json"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing frozen E32 prerequisite: $required" >&2
    exit 1
  fi
done
if ! grep -q '^\*\*Status: FROZEN' "$PROTOCOL"; then
  echo "E32 protocol is not frozen" >&2
  exit 1
fi

for manifest in \
  "$ROOT_DIR/var/artifacts/${GRAPH_PREFIX}_comparative_jobs.tsv" \
  "$ROOT_DIR/var/artifacts/${COUNTDOWN_PREFIX}_comparative_jobs.tsv"; do
  if [[ "$phase" == full && -e "$manifest" ]]; then
    echo "Fresh E32 prefix required; manifest already exists: $manifest" >&2
    exit 1
  fi
done

hash_tree() {
  local tree="$1"
  (
    cd "$tree"
    find . -type f -print0 \
      | sort -z \
      | xargs -0 sha256sum \
      | sha256sum \
      | cut -d' ' -f1
  )
}

freeze_execution() {
  local staging
  SOURCE_HASH="$(hash_tree "$ROOT_DIR/src")"
  SOURCE_ROOT="$ROOT_DIR/var/artifacts/source_snapshots/e32_freeform_05b_${SOURCE_HASH}/src"
  if [[ ! -f "$SOURCE_ROOT/oat_drgrpo/__init__.py" ]]; then
    mkdir -p "$(dirname "$SOURCE_ROOT")"
    staging="$(mktemp -d "$(dirname "$SOURCE_ROOT")/.source.XXXXXX")"
    mkdir -p "$staging/src"
    cp -a "$ROOT_DIR/src/." "$staging/src/"
    mv "$staging/src" "$SOURCE_ROOT"
    rmdir "$staging"
  fi
  if [[ "$(hash_tree "$SOURCE_ROOT")" != "$SOURCE_HASH" ]]; then
    echo "E32 source snapshot hash mismatch" >&2
    exit 1
  fi

  local ops_staging ops_input
  ops_input="$(mktemp -d "$ROOT_DIR/var/artifacts/source_snapshots/.e32-ops-input.XXXXXX")"
  mkdir -p "$ops_input/slurm"
  cp "$ROOT_DIR/ops/repo_env.sh" "$ops_input/repo_env.sh"
  cp "$ROOT_DIR/ops/run_experiment.sh" "$ops_input/run_experiment.sh"
  cp "$ROOT_DIR/ops/train.sh" "$ops_input/train.sh"
  cp "$ROOT_DIR/ops/resolve_eval_cadence.py" "$ops_input/resolve_eval_cadence.py"
  cp "$ROOT_DIR/ops/submit_countdown_comparative.sh" "$ops_input/submit_countdown_comparative.sh"
  cp "$ROOT_DIR/ops/slurm/train_node302.slurm" "$ops_input/slurm/train_node302.slurm"
  EXECUTION_HASH="$(hash_tree "$ops_input")"
  OPS_ROOT="$ROOT_DIR/var/artifacts/source_snapshots/e32_freeform_05b_ops_${EXECUTION_HASH}/ops"
  if [[ ! -f "$OPS_ROOT/submit_countdown_comparative.sh" ]]; then
    mkdir -p "$(dirname "$OPS_ROOT")"
    ops_staging="$(mktemp -d "$(dirname "$OPS_ROOT")/.ops.XXXXXX")"
    mv "$ops_input" "$ops_staging/ops"
    mv "$ops_staging/ops" "$OPS_ROOT"
    rmdir "$ops_staging"
  else
    find "$ops_input" -type f -delete
    rmdir "$ops_input/slurm" "$ops_input"
  fi
  if [[ "$(hash_tree "$OPS_ROOT")" != "$EXECUTION_HASH" ]]; then
    echo "E32 execution snapshot hash mismatch" >&2
    exit 1
  fi
}

if [[ "$phase" == full ]]; then
  freeze_execution
else
  SOURCE_HASH="$(hash_tree "$ROOT_DIR/src")"
  EXECUTION_HASH=config-only-current-tree
  SOURCE_ROOT="$ROOT_DIR/src"
  OPS_ROOT="$ROOT_DIR/ops"
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
export OAT_ZERO_ONLY_ARMS=grpo,maxent_dual
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
export OAT_ZERO_MAXENT_DUAL_MAX_ALPHA=0.00060
export OAT_ZERO_MAXENT_DUAL_ALPHA_LR=0.010
export OAT_ZERO_MAXENT_DUAL_EMA_DECAY=0.7
export OAT_ZERO_MAXENT_LENGTH_TARGET=0
export OAT_ZERO_POLICY_ENTROPY_COEF=0

export OAT_ZERO_NUM_SAMPLES=16
export OAT_ZERO_LEARNING_RATE=0.0000002
export OAT_ZERO_MAX_QUERIES=100000000
export OAT_ZERO_MAX_PROMPT_EPOCHS=10
export OAT_ZERO_NUM_PROMPT_EPOCH=10
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
export OAT_ZERO_EVAL_MODE_COVERAGE_DRAWS=4
export OAT_ZERO_EVAL_MODE_COVERAGE_SEED=1001
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

# One rolling optimizer-resumable checkpoint per pass; terminal export only.
export OAT_ZERO_SAVE_CKPT=1
export OAT_ZERO_MAX_SAVE_NUM=2
export OAT_ZERO_EXPORT_STEPS=0
export OAT_ZERO_MAX_EXPORT_NUM=1
export OAT_ZERO_MAX_RESUME_NUM=2
export OAT_ZERO_PRUNE_RESUME_ON_SUCCESS=0
export OAT_ZERO_AUTO_RESUME=1
export OAT_ZERO_WATCHDOG_REQUEUE=1
export OAT_ZERO_WATCHDOG_MAX_RESTARTS=6
export OAT_ZERO_TRAIN_NODELIST="${OAT_ZERO_E32_TRAIN_NODELIST:-node105,node202,node203,node204}"
export OAT_ZERO_TRAIN_GRES="${OAT_ZERO_E32_TRAIN_GRES:-gpu:a5000:1}"
export OAT_ZERO_TRAIN_CPUS_PER_TASK="${OAT_ZERO_E32_TRAIN_CPUS_PER_TASK:-4}"
export OAT_ZERO_TRAIN_PARTITION="${OAT_ZERO_E32_TRAIN_PARTITION:-all}"
export OAT_ZERO_TRAIN_ACCOUNT="${OAT_ZERO_E32_TRAIN_ACCOUNT:-mltheory}"
export OAT_ZERO_TRAIN_MEMORY="${OAT_ZERO_E32_TRAIN_MEMORY:-32G}"
export OAT_ZERO_TRAIN_TIME_LIMIT="${OAT_ZERO_E32_TRAIN_TIME_LIMIT:-24:00:00}"

write_identity() {
  local protocol_hash launcher_hash
  protocol_hash="$(sha256sum "$PROTOCOL" | cut -d' ' -f1)"
  launcher_hash="$(sha256sum "$0" | cut -d' ' -f1)"
  "$PYTHON_BIN" - "$IDENTITY" "$protocol_hash" "$launcher_hash" \
    "$SOURCE_HASH" "$EXECUTION_HASH" <<'PY'
import json
import os
import pathlib
import sys
import tempfile

path = pathlib.Path(sys.argv[1])
payload = {
    "schema": "e32_freeform_05b_ema_10ep_v5",
    "protocol_sha256": sys.argv[2],
    "launcher_sha256": sys.argv[3],
    "source_hash": sys.argv[4],
    "execution_surface_hash": sys.argv[5],
    "model": "Qwen2.5-0.5B-Instruct@7ae557604adf67be50417f59c2c2f167def9a775",
    "tasks": {
        "graph_coloring": {"prompt_pool": 192, "target_entropy": 1.622718550885717},
        "countdown": {"prompt_pool": 384, "target_entropy": 1.347109432487438},
    },
    "arms": ["grpo", "maxent_dual"],
    "seeds": [43, 44, 45],
    "prompt_epochs": 10,
    "evaluation": {"k": 8, "draws": 4, "seeds": [1001, 1002, 1003, 1004], "pass_at_1": "greedy"},
    "controller": {"rule": "log_alpha_adam_entropy_ema_v2", "ema_decay": 0.7, "alpha_lr": 0.010},
    "resume": {"actor_sync_before_eval_or_rollout": True, "checkpoints_per_prompt_epoch": 1, "keep": 2},
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
    export OAT_ZERO_SAVE_STEPS=192
    export OAT_ZERO_SAVE_FROM=192
    export OAT_ZERO_RESUME_STEPS=192
    export OAT_ZERO_RESUME_FROM=192
  else
    export RUN_STAMP_PREFIX="$COUNTDOWN_PREFIX"
    export OAT_ZERO_COMPARATIVE_TASK=countdown
    export OAT_ZERO_COMPARATIVE_DATA_PRESET=easy3
    export OAT_ZERO_COMPARATIVE_DATA_ROOT="$COUNTDOWN_DATA_ROOT"
    export OAT_ZERO_MAXENT_DUAL_TARGET_ENTROPY="$COUNTDOWN_TARGET"
    export OAT_ZERO_MAX_TRAIN=384
    export OAT_ZERO_EVAL_PROMPT_INTERVAL=96
    export OAT_ZERO_SAVE_STEPS=384
    export OAT_ZERO_SAVE_FROM=384
    export OAT_ZERO_RESUME_STEPS=384
    export OAT_ZERO_RESUME_FROM=384
  fi
  "$OPS_ROOT/submit_countdown_comparative.sh"
}

if [[ "$phase" == config ]]; then
  export OAT_ZERO_COMPARATIVE_CONFIG_ONLY=1
  export OAT_ZERO_SBATCH_HOLD=0
  submit_task graph_coloring
  submit_task countdown
  echo "[e32] both matched 0.5B configurations passed; no jobs submitted"
  exit 0
fi

unset OAT_ZERO_COMPARATIVE_CONFIG_ONLY
export OAT_ZERO_SBATCH_HOLD=1
write_identity
export OAT_ZERO_PROTOCOL_IDENTITY="$IDENTITY"
submit_task graph_coloring
submit_task countdown

job_ids=()
for prefix in "$GRAPH_PREFIX" "$COUNTDOWN_PREFIX"; do
  manifest="$ROOT_DIR/var/artifacts/${prefix}_comparative_jobs.tsv"
  mapfile -t task_jobs < <(awk -F '\t' 'NR > 1 && $3 ~ /^[0-9]+$/ {print $3}' "$manifest")
  if [[ "${#task_jobs[@]}" -ne "$EXPECTED_JOBS_PER_TASK" ]]; then
    mapfile -t submitted_jobs < <(
      awk -F '\t' 'FNR > 1 && $3 ~ /^[0-9]+$/ {print $3}' \
        "$ROOT_DIR/var/artifacts/${GRAPH_PREFIX}_comparative_jobs.tsv" \
        "$ROOT_DIR/var/artifacts/${COUNTDOWN_PREFIX}_comparative_jobs.tsv"
    )
    scancel "${submitted_jobs[@]}" 2>/dev/null || true
    echo "E32 ${prefix} has ${#task_jobs[@]} jobs; expected $EXPECTED_JOBS_PER_TASK" >&2
    exit 1
  fi
  job_ids+=("${task_jobs[@]}")
done

# The site submit filter initially routes mltheory-account jobs to the
# mltheory partition even when --partition=all was requested. Normalize the
# still-held jobs explicitly, then attest the effective scheduler record.
for job_id in "${job_ids[@]}"; do
  if ! scontrol update JobId="$job_id" Partition="$OAT_ZERO_TRAIN_PARTITION"; then
    scancel "${job_ids[@]}" 2>/dev/null || true
    echo "E32 could not normalize held job $job_id to ${OAT_ZERO_TRAIN_PARTITION}" >&2
    exit 1
  fi
done

for spec in "$GRAPH_PREFIX|$GRAPH_TARGET|192|48" "$COUNTDOWN_PREFIX|$COUNTDOWN_TARGET|384|96"; do
  IFS='|' read -r prefix target resume_steps eval_interval <<< "$spec"
  manifest="$ROOT_DIR/var/artifacts/${prefix}_comparative_jobs.tsv"
  mapfile -t task_jobs < <(awk -F '\t' 'NR > 1 && $3 ~ /^[0-9]+$/ {print $3}' "$manifest")
  for job_id in "${task_jobs[@]}"; do
    arm="$(awk -F '\t' -v job_id="$job_id" '$3 == job_id {print $1}' "$manifest")"
    job_record="$(scontrol show job "$job_id" -o)"
    for required in \
      'JobState=PENDING' 'Reason=JobHeldUser' \
      'OAT_ZERO_MAX_PROMPT_EPOCHS=10' \
      'OAT_ZERO_NUM_PROMPT_EPOCH=10' \
      'OAT_ZERO_EVAL_MODE_COVERAGE_K=8' \
      'OAT_ZERO_EVAL_MODE_COVERAGE_DRAWS=4' \
      'OAT_ZERO_EVAL_MODE_COVERAGE_SEED=1001' \
      "OAT_ZERO_EVAL_PROMPT_INTERVAL=${eval_interval}" \
      'OAT_ZERO_MAXENT_OBJECTIVE=conditional_token_mean' \
      "OAT_ZERO_RESUME_STEPS=${resume_steps}" \
      "OAT_ZERO_RESUME_FROM=${resume_steps}" \
      'OAT_ZERO_MAX_RESUME_NUM=2' \
      'OAT_ZERO_PRUNE_RESUME_ON_SUCCESS=0' \
      'OAT_ZERO_EXPORT_STEPS=0' \
      'OAT_ZERO_MAX_EXPORT_NUM=1' \
      "OAT_ZERO_SOURCE_ROOT=${SOURCE_ROOT}" \
      "OAT_ZERO_OPS_SNAPSHOT_ROOT=${OPS_ROOT}" \
      "OAT_ZERO_PROTOCOL_IDENTITY=${IDENTITY}"; do
      if [[ "$job_record" != *"$required"* ]]; then
        scancel "${job_ids[@]}" 2>/dev/null || true
        echo "E32 held-job audit failed for $job_id: missing $required" >&2
        exit 1
      fi
    done
    if [[ "$arm" == maxent_dual ]]; then
      for required in \
        'OAT_ZERO_VARIANT=maxent_dual' \
        "OAT_ZERO_MAXENT_DUAL_TARGET_ENTROPY=${target}" \
        'OAT_ZERO_MAXENT_DUAL_MIN_ALPHA=0.000075' \
        'OAT_ZERO_MAXENT_DUAL_MAX_ALPHA=0.00060' \
        'OAT_ZERO_MAXENT_DUAL_ALPHA_LR=0.010' \
        'OAT_ZERO_MAXENT_DUAL_EMA_DECAY=0.7'; do
        if [[ "$job_record" != *"$required"* ]]; then
          scancel "${job_ids[@]}" 2>/dev/null || true
          echo "E32 treatment audit failed for $job_id: missing $required" >&2
          exit 1
        fi
      done
    elif [[ "$arm" == grpo ]]; then
      for required in \
        'OAT_ZERO_VARIANT=grpo' \
        'OAT_ZERO_MAXENT_DUAL_TARGET_ENTROPY=0.0' \
        'OAT_ZERO_MAXENT_ALPHA=0.0'; do
        if [[ "$job_record" != *"$required"* ]]; then
          scancel "${job_ids[@]}" 2>/dev/null || true
          echo "E32 control audit failed for $job_id: missing $required" >&2
          exit 1
        fi
      done
    else
      scancel "${job_ids[@]}" 2>/dev/null || true
      echo "E32 manifest has unexpected arm for $job_id: $arm" >&2
      exit 1
    fi
  done
done

for job_id in "${job_ids[@]}"; do
  job_record="$(scontrol show job "$job_id" -o)"
  if [[ "$job_record" != *"Partition=${OAT_ZERO_TRAIN_PARTITION}"* ]] || \
     [[ "$job_record" != *"--nodelist=${OAT_ZERO_TRAIN_NODELIST}"* ]] || \
     [[ "$job_record" != *"gres/gpu:a5000=1"* ]] || \
     [[ "$job_record" != *"NumCPUs=${OAT_ZERO_TRAIN_CPUS_PER_TASK}"* ]] || \
     [[ "$job_record" != *"MinMemoryNode=${OAT_ZERO_TRAIN_MEMORY}"* ]]; then
    scancel "${job_ids[@]}" 2>/dev/null || true
    echo "E32 held job $job_id does not attest the frozen placement" >&2
    exit 1
  fi
done

scontrol release "${job_ids[@]}"
echo "[e32] released ${#job_ids[@]} clean matched 0.5B jobs: ${job_ids[*]}"
