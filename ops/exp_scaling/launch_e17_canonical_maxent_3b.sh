#!/usr/bin/env bash
# Launch the frozen E17 3B canonical-action MaxEnt grid.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT_DIR/ops/repo_env.sh"

PHASE="${1:-full}"
case "$PHASE" in
  config|runtime-smoke|full|fixed-config|fixed-retry|fixed-seed45-stage|control-config|control-retry|dual-config|dual-retry) ;;
  *) echo "Usage: $0 {config|runtime-smoke|full|fixed-config|fixed-retry|fixed-seed45-stage|control-config|control-retry|dual-config|dual-retry}" >&2; exit 1 ;;
esac

PROTOCOL="$ROOT_DIR/paper/preregistration/e17_canonical_maxent_3b.md"
SOURCE_HASH=0e9929f09bac51ddcd85a6d5a506bf0613279a6fd983e0147da2f39a7aa320ec
SNAPSHOT_PARENT="$ROOT_DIR/var/artifacts/source_snapshots/e16_canonical_full_${SOURCE_HASH}"
SOURCE_ROOT="$SNAPSHOT_PARENT/src"
OPS_ROOT="$ROOT_DIR/var/artifacts/source_snapshots/e17_canonical_3b_ops_a8414bcd1932787ee1cea7e8c0b23868282f45903eeb9d495b23e4649f397f0c/ops"
MODEL_ROOT="$TRANSFORMERS_CACHE/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1"
PYTHON_BIN="$ROOT_DIR/var/seed_paper_eval/paper310/bin/python"

for required in \
  "$PROTOCOL" \
  "$SOURCE_ROOT/oat_drgrpo/__init__.py" \
  "$OPS_ROOT/submit_countdown_comparative.sh" \
  "$MODEL_ROOT/config.json" \
  "$MODEL_ROOT/tokenizer.json"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing frozen E17 prerequisite: $required" >&2
    exit 1
  fi
done

observed_source_hash="$(
  PYTHONPATH="$SNAPSHOT_PARENT" "$PYTHON_BIN" -c \
    'import sys; from pathlib import Path; from ops.exp_scaling.check_e14_preflight import source_tree_hash; print(source_tree_hash(Path(sys.argv[1]), logical_repo_root=Path(sys.argv[2])))' \
    "$SOURCE_ROOT" "$ROOT_DIR"
)"
if [[ "$observed_source_hash" != "$SOURCE_HASH" ]]; then
  echo "Frozen E17 source mismatch: expected=$SOURCE_HASH observed=$observed_source_hash" >&2
  exit 1
fi

# The canonical digit tokenizer must be identical to E16's audited tokenizer.
E16_MODEL_ROOT="$TRANSFORMERS_CACHE/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
for tokenizer_file in tokenizer.json tokenizer_config.json vocab.json merges.txt; do
  if ! cmp -s "$E16_MODEL_ROOT/$tokenizer_file" "$MODEL_ROOT/$tokenizer_file"; then
    echo "E17 tokenizer drift: $tokenizer_file differs from E16" >&2
    exit 1
  fi
done

export VLLM_USE_V1=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OAT_ZERO_REPO_ROOT="$ROOT_DIR"
export OAT_ZERO_PYTHON="$PYTHON_BIN"
export OAT_ZERO_PYTHON_LIB_DIR="$ROOT_DIR/var/seed_paper_eval/paper310/lib"
export OAT_ZERO_PRETRAIN="$MODEL_ROOT"
export OAT_ZERO_CAMPAIGN_SOURCE_ROOT="$SOURCE_ROOT"
export OAT_ZERO_OPS_SNAPSHOT_ROOT="$OPS_ROOT"
export OAT_ZERO_PROTOCOL_IDENTITY="$PROTOCOL"

export OAT_ZERO_COMPARATIVE_MODEL=qwen2.5-3b-instruct
export OAT_ZERO_TRAIN_SEEDS=43,44,45
export OAT_ZERO_ONLY_ARMS=maxent,maxent_control,maxent_dual
export OAT_ZERO_XDR_TAUS=""
export OAT_ZERO_INCLUDE_TOKEN_ENTROPY_ARM=0
export OAT_ZERO_INCLUDE_SEED_ARM=0
export OAT_ZERO_INCLUDE_XDR_ADAPT_ARM=0
export OAT_ZERO_INCLUDE_XDR_TAU_CONTROL_ARM=0
export OAT_ZERO_INCLUDE_XDR_SAC_DUAL_ARM=0
export OAT_ZERO_INCLUDE_MAXENT_ARM=1
export OAT_ZERO_INCLUDE_MAXENT_CONTROL_ARM=1
export OAT_ZERO_INCLUDE_MAXENT_DUAL_ARM=1
export OAT_ZERO_INCLUDE_MAXENT_LENGTH_DUAL_ARM=0
export OAT_ZERO_COMPARATIVE_REBUILD=0
export OAT_ZERO_REQUIRE_EXISTING_DATA=1
export OAT_ZERO_APPEND_MANIFEST=0

export OAT_ZERO_NUM_SAMPLES=16
export OAT_ZERO_NUM_PROMPT_EPOCH=5
export OAT_ZERO_LEARNING_RATE=0.0000002
export OAT_ZERO_MAX_QUERIES=100000000
export OAT_ZERO_NUM_PPO_EPOCHS=1
export OAT_ZERO_MAX_NORM=1.0
export OAT_ZERO_BETA=0
export OAT_ZERO_IGNORE_NO_EOS=0
export OAT_ZERO_POLICY_ENTROPY_COEF=0

export OAT_ZERO_MAXENT_ALPHA=0.10
export OAT_ZERO_MAXENT_FIXED_ALPHA=0.10
export OAT_ZERO_MAXENT_CONTROL_BASE_ALPHA=0.075
export OAT_ZERO_MAXENT_DUAL_BASE_ALPHA=0.075
export OAT_ZERO_MAXENT_CONTROL_RATIO=1.0
export OAT_ZERO_MAXENT_CONTROL_WARMUP_STEPS=1
export OAT_ZERO_MAXENT_CONTROL_MAX_ALPHA=0.10
export OAT_ZERO_MAXENT_CONTROL_EMA_DECAY=0.9
export OAT_ZERO_MAXENT_CONTROL_GAIN=4.0
export OAT_ZERO_MAXENT_DUAL_RATIO=1.0
export OAT_ZERO_MAXENT_DUAL_WARMUP_STEPS=1
export OAT_ZERO_MAXENT_DUAL_MIN_ALPHA=0.05
export OAT_ZERO_MAXENT_DUAL_MAX_ALPHA=0.10
export OAT_ZERO_MAXENT_DUAL_ALPHA_LR=0.005
export OAT_ZERO_MAXENT_LENGTH_TARGET=0

export OAT_ZERO_TRAIN_BATCH_SIZE=16
export OAT_ZERO_TRAIN_BATCH_SIZE_PER_DEVICE=4
export OAT_ZERO_ROLLOUT_BATCH_SIZE=1
export OAT_ZERO_ROLLOUT_BATCH_SIZE_PER_DEVICE=1
export OAT_ZERO_N_GPU=1
export OAT_ZERO_NUM_GPUS_PER_ACTOR=1
export OAT_ZERO_ZERO_STAGE=2
export OAT_ZERO_VLLM_GPU_RATIO=0.25
export OAT_ZERO_ENABLE_FLASH_ATTN=0
export OAT_ZERO_ADAM_OFFLOAD=1
export OAT_ZERO_ACTIVATION_OFFLOADING=1
export OAT_ZERO_COLLOCATE=1
export OAT_ZERO_VLLM_SLEEP=1

export OAT_ZERO_CANONICAL_GRAPH_ACTION_COUNT=3
export OAT_ZERO_CANONICAL_GRAPH_LEARNER_SAMPLING=1
export OAT_ZERO_CANONICAL_GRAPH_FIXED_SHAPE_SAMPLING=1
export OAT_ZERO_GENERATE_MAX_LENGTH=192
export OAT_ZERO_EVAL_GENERATE_MAX_LENGTH=192
export OAT_ZERO_TEMPERATURE=1
export OAT_ZERO_TOP_P=1
export OAT_ZERO_EVAL_MODE_COVERAGE_K=8
export OAT_ZERO_EVAL_MODE_COVERAGE_TEMPERATURE=1
export OAT_ZERO_EVAL_TEMPERATURE=0
export OAT_ZERO_EVAL_BATCH_SIZE=32
export OAT_ZERO_PROMPT_MAX_LENGTH=256
export OAT_ZERO_TEST_SPLIT=multi_answer

export OAT_ZERO_SAVE_CKPT=1
export OAT_ZERO_MAX_SAVE_NUM=1
export OAT_ZERO_MAX_SAVE_MEM=2000
export OAT_ZERO_AUTO_RESUME=1
export OAT_ZERO_WATCHDOG_REQUEUE=0
export OAT_ZERO_WATCHDOG_STALE_SECONDS=0

export OAT_ZERO_TRAIN_NODELIST="${OAT_ZERO_E17_TRAIN_NODELIST:-node103,node104,node208}"
export OAT_ZERO_TRAIN_GRES="${OAT_ZERO_E17_TRAIN_GRES:-gpu:a6000:1}"
export OAT_ZERO_TRAIN_PARTITION="${OAT_ZERO_E17_TRAIN_PARTITION:-lowprio}"
export OAT_ZERO_TRAIN_ACCOUNT="${OAT_ZERO_E17_TRAIN_ACCOUNT:-mltheory}"
export OAT_ZERO_TRAIN_MEMORY="${OAT_ZERO_E17_TRAIN_MEMORY:-96G}"
export OAT_ZERO_TRAIN_TIME_LIMIT="${OAT_ZERO_E17_TRAIN_TIME_LIMIT:-168:00:00}"
export OAT_ZERO_SBATCH_HOLD=1

submit_task() {
  local task="$1" prefix="$2" data_root="$3" prompt_template="$4"
  local target="$5" budget="$6" eval_interval="$7"
  export RUN_STAMP_PREFIX="$prefix"
  export OAT_ZERO_COMPARATIVE_TASK="$task"
  export OAT_ZERO_COMPARATIVE_DATA_ROOT="$data_root"
  export OAT_ZERO_PROMPT_TEMPLATE="$prompt_template"
  export OAT_ZERO_CANONICAL_ACTION_TASK="$task"
  export OAT_ZERO_MAXENT_CONTROL_TARGET_ENTROPY="$target"
  export OAT_ZERO_MAXENT_DUAL_TARGET_ENTROPY="$target"
  export OAT_ZERO_MAX_TRAIN="$budget"
  export OAT_ZERO_EVAL_PROMPT_INTERVAL="$eval_interval"
  export OAT_ZERO_SYNC_PARAMS_EVERY="$eval_interval"
  export OAT_ZERO_SAVE_STEPS="$eval_interval"
  export OAT_ZERO_SAVE_FROM="$eval_interval"
  if [[ "$task" == graph_coloring ]]; then
    export OAT_ZERO_CANONICAL_GRAPH_ACTIONS=1
  else
    export OAT_ZERO_CANONICAL_GRAPH_ACTIONS=0
  fi
  bash "$OPS_ROOT/submit_countdown_comparative.sh"
}

if [[ "$PHASE" == config ]]; then
  export OAT_ZERO_COMPARATIVE_CONFIG_ONLY=1
  submit_task graph_coloring gce17_canonical_maxent_3b_v5 \
    "$ROOT_DIR/var/data/exact_answer_mode_probe" qwen_graph_digits \
    2.7974740052946436 15344 48
  submit_task countdown cde17_canonical_maxent_3b_v5 \
    "$ROOT_DIR/var/data/exact_countdown_easy3_probe" qwen_countdown_digits \
    3.9741470618167156 30704 96
  echo "[e17] both frozen 3B task configurations passed; no jobs submitted"
  exit 0
fi

if [[ "$PHASE" == runtime-smoke ]]; then
  export OAT_ZERO_TRAIN_SEEDS=9006
  export OAT_ZERO_ONLY_ARMS=maxent
  export OAT_ZERO_NUM_PROMPT_EPOCH=1
  export OAT_ZERO_SAVE_CKPT=0
  export OAT_ZERO_AUTO_RESUME=0
  export OAT_ZERO_SBATCH_HOLD=0
  submit_task graph_coloring gce17_canonical_maxent_3b_runtime_smoke_v2 \
    "$ROOT_DIR/var/data/exact_answer_mode_probe" qwen_graph_digits \
    2.7974740052946436 496 32
  exit 0
fi

if [[ "$PHASE" == fixed-config ]]; then
  export OAT_ZERO_COMPARATIVE_CONFIG_ONLY=1
  export OAT_ZERO_ONLY_ARMS=maxent
  submit_task countdown cde17_canonical_maxent_3b_v5 \
    "$ROOT_DIR/var/data/exact_countdown_easy3_probe" qwen_countdown_digits \
    3.9741470618167156 30704 96
  echo "[e17] Countdown fixed-MaxEnt recovery configuration passed; no jobs submitted"
  exit 0
fi

if [[ "$PHASE" == fixed-seed45-stage ]]; then
  unset OAT_ZERO_COMPARATIVE_CONFIG_ONLY
  export OAT_ZERO_TRAIN_SEEDS=45
  export OAT_ZERO_ONLY_ARMS=maxent
  export OAT_ZERO_APPEND_MANIFEST=1
  export OAT_ZERO_TRAIN_NODELIST=node302
  export OAT_ZERO_TRAIN_GRES=gpu:a100:1
  export OAT_ZERO_TRAIN_PARTITION=mltheory
  export OAT_ZERO_TRAIN_ACCOUNT=mltheory

  countdown_manifest="$ROOT_DIR/var/artifacts/cde17_canonical_maxent_3b_v5_comparative_jobs.tsv"
  if [[ ! -f "$countdown_manifest" ]]; then
    echo "E17 fixed-MaxEnt seed-45 recovery requires the original manifest: $countdown_manifest" >&2
    exit 1
  fi

  manifest_lines_before="$(wc -l < "$countdown_manifest")"
  mapfile -t older_seed45_ids < <(
    head -n "$manifest_lines_before" "$countdown_manifest" |
      awk -F '\t' '$1 == "maxent" && $2 == "45" && $3 ~ /^[0-9]+$/ {print $3}'
  )
  for old_job_id in "${older_seed45_ids[@]}"; do
    old_job_dump="$(scontrol show job -o "$old_job_id" 2>/dev/null || true)"
    if [[ "$old_job_dump" =~ JobState=(PENDING|RUNNING|SUSPENDED|COMPLETING|CONFIGURING|RESIZING|REQUEUED) ]]; then
      echo "E17 fixed-MaxEnt seed-45 recovery found an older live writer: $old_job_id" >&2
      exit 1
    fi
  done

  submit_task countdown cde17_canonical_maxent_3b_v5 \
    "$ROOT_DIR/var/data/exact_countdown_easy3_probe" qwen_countdown_digits \
    3.9741470618167156 30704 96
  mapfile -t job_ids < <(
    tail -n "+$((manifest_lines_before + 1))" "$countdown_manifest" |
      awk -F '\t' '$1 == "maxent" && $2 == "45" && $3 ~ /^[0-9]+$/ {print $3}'
  )
  if (( ${#job_ids[@]} != 1 )); then
    echo "E17 fixed-MaxEnt seed-45 staging incomplete (${#job_ids[@]}/1); any new jobs remain held" >&2
    exit 1
  fi

  job_id="${job_ids[0]}"
  job_dump="$(scontrol show job -o "$job_id")"
  for required in \
    'JobState=PENDING' \
    'Reason=JobHeldUser' \
    'Account=mltheory' \
    'Partition=mltheory' \
    'ReqNodeList=node302' \
    'TresPerNode=gres/gpu:a100:1' \
    'mem=96G' \
    'RUN_STAMP=cde17_canonical_maxent_3b_v5_maxent_s45' \
    'OAT_ZERO_SEED=45' \
    'OAT_ZERO_VARIANT=maxent' \
    'OAT_ZERO_MAXENT_ALPHA=0.10' \
    'OAT_ZERO_MAX_TRAIN=30704' \
    'OAT_ZERO_SAVE_STEPS=96'; do
    if [[ "$job_dump" != *"$required"* ]]; then
      echo "E17 fixed-MaxEnt seed-45 held-job audit failed for $job_id: missing $required" >&2
      exit 1
    fi
  done

  echo "[e17] staged one held Countdown fixed-MaxEnt seed-45 recovery job: $job_id"
  echo "[e17] audit passed; release explicitly after checkpoint and writer-uniqueness checks"
  exit 0
fi

if [[ "$PHASE" == fixed-retry ]]; then
  unset OAT_ZERO_COMPARATIVE_CONFIG_ONLY
  export OAT_ZERO_ONLY_ARMS=maxent
  export OAT_ZERO_APPEND_MANIFEST=1
  countdown_manifest="$ROOT_DIR/var/artifacts/cde17_canonical_maxent_3b_v5_comparative_jobs.tsv"
  if [[ ! -f "$countdown_manifest" ]]; then
    echo "E17 fixed-MaxEnt recovery requires the original manifest: $countdown_manifest" >&2
    exit 1
  fi
  manifest_lines_before="$(wc -l < "$countdown_manifest")"
  submit_task countdown cde17_canonical_maxent_3b_v5 \
    "$ROOT_DIR/var/data/exact_countdown_easy3_probe" qwen_countdown_digits \
    3.9741470618167156 30704 96
  mapfile -t job_ids < <(
    tail -n "+$((manifest_lines_before + 1))" "$countdown_manifest" |
      awk -F '\t' '$1 == "maxent" && $3 ~ /^[0-9]+$/ {print $3}'
  )
  if (( ${#job_ids[@]} != 3 )); then
    echo "E17 fixed-MaxEnt recovery incomplete (${#job_ids[@]}/3); new jobs remain held" >&2
    exit 1
  fi
  scontrol release "${job_ids[@]}"
  echo "[e17] released three Countdown fixed-MaxEnt recovery jobs: ${job_ids[*]}"
  exit 0
fi

if [[ "$PHASE" == control-config ]]; then
  export OAT_ZERO_COMPARATIVE_CONFIG_ONLY=1
  export OAT_ZERO_TRAIN_SEEDS=43,44
  export OAT_ZERO_ONLY_ARMS=maxent_control
  submit_task countdown cde17_canonical_maxent_3b_v5 \
    "$ROOT_DIR/var/data/exact_countdown_easy3_probe" qwen_countdown_digits \
    3.9741470618167156 30704 96
  echo "[e17] Countdown proportional-MaxEnt recovery configuration passed; no jobs submitted"
  exit 0
fi

if [[ "$PHASE" == control-retry ]]; then
  unset OAT_ZERO_COMPARATIVE_CONFIG_ONLY
  # Seeds 43/44 have only landed model snapshots, not rolling optimizer
  # checkpoints. Seed 45 retains a resumable checkpoint and is recovered by
  # retargeting its existing pending job, so do not duplicate that writer.
  export OAT_ZERO_TRAIN_SEEDS=43,44
  export OAT_ZERO_ONLY_ARMS=maxent_control
  export OAT_ZERO_APPEND_MANIFEST=1
  countdown_manifest="$ROOT_DIR/var/artifacts/cde17_canonical_maxent_3b_v5_comparative_jobs.tsv"
  if [[ ! -f "$countdown_manifest" ]]; then
    echo "E17 proportional-MaxEnt recovery requires the original manifest: $countdown_manifest" >&2
    exit 1
  fi
  manifest_lines_before="$(wc -l < "$countdown_manifest")"
  submit_task countdown cde17_canonical_maxent_3b_v5 \
    "$ROOT_DIR/var/data/exact_countdown_easy3_probe" qwen_countdown_digits \
    3.9741470618167156 30704 96
  mapfile -t job_ids < <(
    tail -n "+$((manifest_lines_before + 1))" "$countdown_manifest" |
      awk -F '\t' '$1 == "maxent_control" && ($2 == "43" || $2 == "44") && $3 ~ /^[0-9]+$/ {print $3}'
  )
  if (( ${#job_ids[@]} != 2 )); then
    echo "E17 proportional-MaxEnt recovery incomplete (${#job_ids[@]}/2); new jobs remain held" >&2
    exit 1
  fi
  for job_id in "${job_ids[@]}"; do
    job_dump="$(scontrol show job -o "$job_id")"
    for required in \
      'JobState=PENDING' \
      'ReqNodeList=node302' \
      'TresPerNode=gres/gpu:a100:1' \
      'mem=96G'; do
      if [[ "$job_dump" != *"$required"* ]]; then
        echo "E17 proportional-MaxEnt held-job audit failed for $job_id: missing $required" >&2
        exit 1
      fi
    done
  done
  scontrol release "${job_ids[@]}"
  echo "[e17] released two Countdown proportional-MaxEnt node302 recovery jobs: ${job_ids[*]}"
  exit 0
fi

if [[ "$PHASE" == dual-config ]]; then
  export OAT_ZERO_COMPARATIVE_CONFIG_ONLY=1
  export OAT_ZERO_ONLY_ARMS=maxent_dual
  submit_task countdown cde17_canonical_maxent_3b_v5 \
    "$ROOT_DIR/var/data/exact_countdown_easy3_probe" qwen_countdown_digits \
    3.9741470618167156 30704 96
  echo "[e17] Countdown dual-MaxEnt recovery configuration passed; no jobs submitted"
  exit 0
fi

if [[ "$PHASE" == dual-retry ]]; then
  unset OAT_ZERO_COMPARATIVE_CONFIG_ONLY
  export OAT_ZERO_ONLY_ARMS=maxent_dual
  export OAT_ZERO_APPEND_MANIFEST=1
  countdown_manifest="$ROOT_DIR/var/artifacts/cde17_canonical_maxent_3b_v5_comparative_jobs.tsv"
  if [[ ! -f "$countdown_manifest" ]]; then
    echo "E17 dual-MaxEnt recovery requires the original manifest: $countdown_manifest" >&2
    exit 1
  fi
  manifest_lines_before="$(wc -l < "$countdown_manifest")"
  submit_task countdown cde17_canonical_maxent_3b_v5 \
    "$ROOT_DIR/var/data/exact_countdown_easy3_probe" qwen_countdown_digits \
    3.9741470618167156 30704 96
  mapfile -t job_ids < <(
    tail -n "+$((manifest_lines_before + 1))" "$countdown_manifest" |
      awk -F '\t' '$1 == "maxent_dual" && $3 ~ /^[0-9]+$/ {print $3}'
  )
  if (( ${#job_ids[@]} != 3 )); then
    echo "E17 dual-MaxEnt recovery incomplete (${#job_ids[@]}/3); new jobs remain held" >&2
    exit 1
  fi

  # A same-stamp writer must be unique. Cancel any older still-allocated dual
  # attempt only after all three replacements exist safely in held state.
  mapfile -t old_live_ids < <(
    comm -12 \
      <(head -n "$manifest_lines_before" "$countdown_manifest" |
        awk -F '\t' '$1 == "maxent_dual" && $3 ~ /^[0-9]+$/ {print $3}' |
        sort -u) \
      <(squeue -h -u "$USER" -o '%i' | sort -u)
  )
  if (( ${#old_live_ids[@]} > 0 )); then
    scancel "${old_live_ids[@]}"
    for _ in {1..30}; do
      any_live=0
      for old_job_id in "${old_live_ids[@]}"; do
        if [[ -n "$(squeue -h -j "$old_job_id" -o '%T')" ]]; then
          any_live=1
        fi
      done
      (( any_live == 0 )) && break
      sleep 1
    done
    if (( any_live != 0 )); then
      echo "Older dual-MaxEnt jobs did not leave Slurm; replacements remain held" >&2
      exit 1
    fi
  fi

  scontrol release "${job_ids[@]}"
  echo "[e17] cancelled stale dual jobs: ${old_live_ids[*]:-none}"
  echo "[e17] released three Countdown dual-MaxEnt recovery jobs: ${job_ids[*]}"
  exit 0
fi

unset OAT_ZERO_COMPARATIVE_CONFIG_ONLY
submit_task graph_coloring gce17_canonical_maxent_3b_v5 \
  "$ROOT_DIR/var/data/exact_answer_mode_probe" qwen_graph_digits \
  2.7974740052946436 15344 48
submit_task countdown cde17_canonical_maxent_3b_v5 \
  "$ROOT_DIR/var/data/exact_countdown_easy3_probe" qwen_countdown_digits \
  3.9741470618167156 30704 96

graph_manifest="$ROOT_DIR/var/artifacts/gce17_canonical_maxent_3b_v5_comparative_jobs.tsv"
countdown_manifest="$ROOT_DIR/var/artifacts/cde17_canonical_maxent_3b_v5_comparative_jobs.tsv"
mapfile -t job_ids < <(tail -n +2 "$graph_manifest" "$countdown_manifest" | awk -F '\t' '$3 ~ /^[0-9]+$/ {print $3}')
if (( ${#job_ids[@]} != 18 )); then
  echo "E17 cohort incomplete (${#job_ids[@]}/18); jobs remain held" >&2
  exit 1
fi
scontrol release "${job_ids[@]}"
echo "[e17] released complete 18-job cohort: ${job_ids[*]}"
