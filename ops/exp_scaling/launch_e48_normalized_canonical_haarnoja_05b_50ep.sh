#!/usr/bin/env bash
# Launch E48's fresh 50-pass matched Dr.GRPO/Haarnoja ModeBench cohort.
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
PROTOCOL="$ROOT_DIR/paper/preregistration/e48_normalized_canonical_haarnoja_05b_50ep.md"
MODEL_ROOT="$ROOT_DIR/var/cache/huggingface/transformers/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
GRAPH_DATA_ROOT="$ROOT_DIR/var/data/exact_answer_mode_probe"
COUNTDOWN_DATA_ROOT="$ROOT_DIR/var/data/exact_countdown_easy3_probe"
GRAPH_PREFIX=gce48_normalized_canonical_haarnoja_05b_50ep_v1
COUNTDOWN_PREFIX=cde48_normalized_canonical_haarnoja_05b_50ep_v1
IDENTITY="$ROOT_DIR/var/artifacts/e48_normalized_canonical_haarnoja_05b_50ep_v1_identity.json"
EXPECTED_JOBS_PER_TASK=6
EXPECTED_JOBS=12
PROMPT_EPOCHS=50

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
    echo "Missing frozen E48 prerequisite: $required" >&2
    exit 1
  fi
done
if ! grep -q '^\*\*Status: FROZEN BEFORE LAUNCH' "$PROTOCOL"; then
  echo "E48 protocol is not frozen" >&2
  exit 1
fi
for contract in \
  'src/oat_drgrpo/online_canonical_controller.py:verified_bank_entropy_over_log_support_v1' \
  'src/oat_drgrpo/learner/run.py:online_canonical_alpha_controller_state' \
  'src/oat_drgrpo/learner/grpo.py:verified_discovery_cumulative_outcomes' \
  'src/oat_drgrpo/learner/grpo.py:verified_discovery_mean_support_per_prompt' \
  'src/oat_drgrpo/args.py:verified_discovery_tracking: bool = True' \
  'src/oat_drgrpo/args.py:online_canonical_dual_target_ratio' \
  'ops/run_experiment.sh:online_canonical_haarnoja)' \
  'ops/submit_countdown_comparative.sh:OAT_ZERO_VERIFIED_DISCOVERY_TRACKING'; do
  path="${contract%%:*}"
  needle="${contract#*:}"
  if ! grep -q "$needle" "$ROOT_DIR/$path"; then
    echo "Current source lacks E48 contract: $path :: $needle" >&2
    exit 1
  fi
done

for manifest in \
  "$ROOT_DIR/var/artifacts/${GRAPH_PREFIX}_comparative_jobs.tsv" \
  "$ROOT_DIR/var/artifacts/${COUNTDOWN_PREFIX}_comparative_jobs.tsv"; do
  if [[ "$phase" == full && -e "$manifest" ]]; then
    echo "Fresh E48 prefix required; manifest exists: $manifest" >&2
    exit 1
  fi
done
if [[ "$phase" == full && -e "$IDENTITY" ]]; then
  echo "Fresh E48 identity required; file exists: $IDENTITY" >&2
  exit 1
fi

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
  local staging ops_staging ops_input
  SOURCE_HASH="$(hash_tree "$ROOT_DIR/src")"
  SOURCE_ROOT="$ROOT_DIR/var/artifacts/source_snapshots/e48_normalized_canonical_haarnoja_05b_50ep_${SOURCE_HASH}/src"
  if [[ ! -f "$SOURCE_ROOT/oat_drgrpo/__init__.py" ]]; then
    mkdir -p "$(dirname "$SOURCE_ROOT")"
    staging="$(mktemp -d "$(dirname "$SOURCE_ROOT")/.source.XXXXXX")"
    mkdir -p "$staging/src"
    cp -a "$ROOT_DIR/src/." "$staging/src/"
    mv "$staging/src" "$SOURCE_ROOT"
    rmdir "$staging"
  fi
  if [[ "$(hash_tree "$SOURCE_ROOT")" != "$SOURCE_HASH" ]]; then
    echo "E48 source snapshot hash mismatch" >&2
    exit 1
  fi

  ops_input="$(mktemp -d "$ROOT_DIR/var/artifacts/source_snapshots/.e48-ops-input.XXXXXX")"
  mkdir -p "$ops_input/slurm"
  for file in repo_env.sh run_experiment.sh train.sh resolve_eval_cadence.py submit_countdown_comparative.sh; do
    cp "$ROOT_DIR/ops/$file" "$ops_input/$file"
  done
  cp "$ROOT_DIR/ops/slurm/train_node302.slurm" "$ops_input/slurm/train_node302.slurm"
  EXECUTION_HASH="$(hash_tree "$ops_input")"
  OPS_ROOT="$ROOT_DIR/var/artifacts/source_snapshots/e48_normalized_canonical_haarnoja_05b_50ep_ops_${EXECUTION_HASH}/ops"
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
    echo "E48 execution snapshot hash mismatch" >&2
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
export OAT_ZERO_ONLY_ARMS=grpo,online_canonical_haarnoja
export OAT_ZERO_XDR_TAUS=""

for flag in \
  TOKEN_ENTROPY SEED XDR_ADAPT XDR_TAU_CONTROL XDR_SAC_DUAL \
  MAXENT MAXENT_CONTROL MAXENT_DUAL MAXENT_LENGTH_DUAL DIAYN \
  OUTCOME_COLLISION OUTCOME_COLLISION_OUTSIDE_CENTERING \
  SEMANTIC_SHANNON SEMANTIC_SHANNON_ADVANTAGE \
  QUALITY_GATED_SEMANTIC_NOVELTY \
  SUCCESS_CONDITIONED_SIGNED_SEMANTIC_SHANNON \
  SIGNAL_FIRST_SEMANTIC_BALANCE ONLINE_CANONICAL_MAXENT; do
  export "OAT_ZERO_INCLUDE_${flag}_ARM=0"
done
export OAT_ZERO_INCLUDE_ONLINE_CANONICAL_HAARNOJA_ARM=1

export OAT_ZERO_VERIFIED_DISCOVERY_TRACKING=1
export OAT_ZERO_ONLINE_CANONICAL_BANK_ALPHA=0.10
export OAT_ZERO_ONLINE_CANONICAL_NOVELTY_BETA=0.50
export OAT_ZERO_ONLINE_CANONICAL_BANK_PSEUDOCOUNT=1.0
export OAT_ZERO_ONLINE_CANONICAL_BANK_SURPRISAL_CLIP=5.0
export OAT_ZERO_ONLINE_CANONICAL_KEY_MODE=modebench_outcome
export OAT_ZERO_ONLINE_CANONICAL_DUAL_TARGET_RATIO=0.80
export OAT_ZERO_ONLINE_CANONICAL_DUAL_MIN_ALPHA=0.10
export OAT_ZERO_ONLINE_CANONICAL_DUAL_MAX_ALPHA=0.50
export OAT_ZERO_ONLINE_CANONICAL_DUAL_ALPHA_LR=0.003
export OAT_ZERO_ONLINE_CANONICAL_DUAL_EMA_DECAY=0.90
export OAT_ZERO_OUTCOME_COLLISION_COEF=0
export OAT_ZERO_SEMANTIC_SHANNON_COEF=0
export OAT_ZERO_MAXENT_OBJECTIVE=sequence
export OAT_ZERO_MAXENT_ALPHA=0
export OAT_ZERO_POLICY_ENTROPY_COEF=0
export OAT_ZERO_SEED_ENTROPY_ALPHA=0
export OAT_ZERO_DIAYN_NUM_OPTIONS=0
export OAT_ZERO_DIAYN_MI_BETA=0

export OAT_ZERO_NUM_SAMPLES=16
export OAT_ZERO_LEARNING_RATE=0.0000002
export OAT_ZERO_MAX_QUERIES=100000000
export OAT_ZERO_MAX_PROMPT_EPOCHS="$PROMPT_EPOCHS"
export OAT_ZERO_NUM_PROMPT_EPOCH="$PROMPT_EPOCHS"
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
export OAT_ZERO_EVAL_MODE_COVERAGE_SEED=440100
export OAT_ZERO_EVAL_BATCH_SIZE=64
export OAT_ZERO_SYNC_PARAMS_EVERY=1

export OAT_ZERO_CANONICAL_ACTION_TASK=none
export OAT_ZERO_CANONICAL_GRAPH_ACTIONS=0
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
export OAT_ZERO_EXPORT_STEPS=0
export OAT_ZERO_MAX_EXPORT_NUM=1
export OAT_ZERO_MAX_RESUME_NUM=2
export OAT_ZERO_PRUNE_RESUME_ON_SUCCESS=0
export OAT_ZERO_AUTO_RESUME=1
export OAT_ZERO_WATCHDOG_REQUEUE=1
export OAT_ZERO_WATCHDOG_MAX_RESTARTS=6
export OAT_ZERO_TRAIN_NODELIST="${OAT_ZERO_E48_TRAIN_NODELIST:-node302}"
export OAT_ZERO_TRAIN_GRES="${OAT_ZERO_E48_TRAIN_GRES:-gpu:a100:1}"
export OAT_ZERO_TRAIN_CPUS_PER_TASK="${OAT_ZERO_E48_TRAIN_CPUS_PER_TASK:-8}"
export OAT_ZERO_TRAIN_PARTITION="${OAT_ZERO_E48_TRAIN_PARTITION:-all}"
export OAT_ZERO_TRAIN_ACCOUNT="${OAT_ZERO_E48_TRAIN_ACCOUNT:-mltheory}"
export OAT_ZERO_TRAIN_MEMORY="${OAT_ZERO_E48_TRAIN_MEMORY:-64G}"
export OAT_ZERO_TRAIN_TIME_LIMIT="${OAT_ZERO_E48_TRAIN_TIME_LIMIT:-7-00:00:00}"

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
    "schema": "e48_normalized_canonical_haarnoja_05b_50ep_v1",
    "protocol_sha256": sys.argv[2],
    "launcher_sha256": sys.argv[3],
    "source_hash": sys.argv[4],
    "execution_surface_hash": sys.argv[5],
    "model": "Qwen2.5-0.5B-Instruct@7ae557604adf67be50417f59c2c2f167def9a775",
    "tasks": {
        "graph_coloring": {"prompt_pool": 192, "validator": "executable"},
        "countdown": {"prompt_pool": 384, "validator": "executable_ast"},
    },
    "arms": ["grpo", "online_canonical_haarnoja"],
    "contemporaneous": True,
    "online_canonical_haarnoja": {
        "initial_alpha": 0.10,
        "alpha_bounds": [0.10, 0.50],
        "normalized_entropy_target": 0.80,
        "alpha_lr": 0.003,
        "ema_decay": 0.90,
        "adam_betas": [0.9, 0.999],
        "adam_epsilon": 1e-8,
        "novelty_beta": 0.50,
        "pseudocount": 1.0,
        "surprisal_clip": 5.0,
        "sensor": "exact_postupdate_cumulative_H_over_log_support_v1",
        "singleton_policy": "skip",
        "update_timing": "after_policy_optimizer_for_next_round",
        "bank_and_controller_checkpointed": True,
    },
    "grpo": {
        "objective": "ordinary_drgrpo",
        "verified_discovery_tracking": "passive_zero_influence",
        "bank_checkpointed": True,
    },
    "seeds": [43, 44, 45],
    "num_samples": 16,
    "prompt_epochs": 50,
    "evaluation_interval_passes": 0.25,
    "placement": {
        "node": "node302",
        "gpu_per_job": "a100:1",
        "cpus_per_job": 8,
        "memory": "64G",
        "time_limit": "7-00:00:00",
    },
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
  echo "[e48] both 50-pass matched configurations passed; no jobs submitted"
  exit 0
fi

unset OAT_ZERO_COMPARATIVE_CONFIG_ONLY
export OAT_ZERO_SBATCH_HOLD=1
COHORT_RELEASED=0
job_ids=()
cleanup_partial_cohort() {
  local status="$?"
  trap - EXIT
  if [[ "$COHORT_RELEASED" != "1" && "${#job_ids[@]}" -gt 0 ]]; then
    scancel "${job_ids[@]}" 2>/dev/null || true
    echo "[e48] cancelled incomplete held cohort: ${job_ids[*]}" >&2
  fi
  exit "$status"
}
trap cleanup_partial_cohort EXIT

write_identity
export OAT_ZERO_PROTOCOL_IDENTITY="$IDENTITY"
submit_task graph_coloring
submit_task countdown

for prefix in "$GRAPH_PREFIX" "$COUNTDOWN_PREFIX"; do
  manifest="$ROOT_DIR/var/artifacts/${prefix}_comparative_jobs.tsv"
  mapfile -t task_jobs < <(
    awk -F '\t' 'NR > 1 && $3 ~ /^[0-9]+$/ {print $3}' "$manifest"
  )
  if [[ "${#task_jobs[@]}" -ne "$EXPECTED_JOBS_PER_TASK" ]]; then
    echo "E48 ${prefix} has ${#task_jobs[@]} jobs; expected $EXPECTED_JOBS_PER_TASK" >&2
    exit 1
  fi
  job_ids+=("${task_jobs[@]}")
  "$PYTHON_BIN" - "$manifest" <<'PY'
import csv
import pathlib
import sys

with pathlib.Path(sys.argv[1]).open(encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle, delimiter="\t"))
expected = {
    (arm, str(seed))
    for arm in ("grpo", "online_canonical_haarnoja")
    for seed in (43, 44, 45)
}
observed = {(row["arm"], row["seed"]) for row in rows}
if len(rows) != 6 or observed != expected:
    raise SystemExit(f"E48 held manifest mismatch: {observed!r}")
PY
done
if [[ "${#job_ids[@]}" -ne "$EXPECTED_JOBS" ]]; then
  echo "E48 cohort has ${#job_ids[@]} jobs; expected $EXPECTED_JOBS" >&2
  exit 1
fi

for spec in "$GRAPH_PREFIX|192|48" "$COUNTDOWN_PREFIX|384|96"; do
  IFS='|' read -r prefix resume_steps eval_interval <<< "$spec"
  manifest="$ROOT_DIR/var/artifacts/${prefix}_comparative_jobs.tsv"
  while IFS=$'\t' read -r arm seed job_id run_stamp; do
    [[ "$job_id" =~ ^[0-9]+$ ]] || continue
    job_record="$(scontrol show job "$job_id" -o)"
    expected_variant="$arm"
    for required in \
      'JobState=PENDING' \
      'Reason=JobHeldUser' \
      "OAT_ZERO_VARIANT=${expected_variant}" \
      'OAT_ZERO_MAX_PROMPT_EPOCHS=50' \
      'OAT_ZERO_NUM_PROMPT_EPOCH=50' \
      'OAT_ZERO_VERIFIED_DISCOVERY_TRACKING=1' \
      'OAT_ZERO_NUM_SAMPLES=16' \
      'OAT_ZERO_PROMPT_TEMPLATE=qwen_boxed' \
      'OAT_ZERO_TEST_SPLIT=multi_answer' \
      'OAT_ZERO_VERIFIER_VERSION=fast' \
      'OAT_ZERO_EVAL_MODE_COVERAGE_K=8' \
      'OAT_ZERO_EVAL_MODE_COVERAGE_DRAWS=4' \
      'OAT_ZERO_EVAL_MODE_COVERAGE_SEED=440100' \
      "OAT_ZERO_EVAL_PROMPT_INTERVAL=${eval_interval}" \
      "OAT_ZERO_RESUME_STEPS=${resume_steps}" \
      "OAT_ZERO_SOURCE_ROOT=${SOURCE_ROOT}" \
      "OAT_ZERO_OPS_SNAPSHOT_ROOT=${OPS_ROOT}" \
      "OAT_ZERO_PROTOCOL_IDENTITY=${IDENTITY}"; do
      if [[ "$job_record" != *"$required"* ]]; then
        echo "E48 held-job audit failed for $job_id: missing $required" >&2
        exit 1
      fi
    done
    if [[ "$arm" == grpo ]]; then
      for required in \
        'OAT_ZERO_ONLINE_CANONICAL_BANK_ALPHA=0.0' \
        'OAT_ZERO_ONLINE_CANONICAL_NOVELTY_BETA=0.0' \
        'OAT_ZERO_ONLINE_CANONICAL_DUAL_TARGET_RATIO=0.0'; do
        if [[ "$job_record" != *"$required"* ]]; then
          echo "E48 Dr.GRPO audit failed for $job_id: missing $required" >&2
          exit 1
        fi
      done
    else
      for required in \
        'OAT_ZERO_ONLINE_CANONICAL_BANK_ALPHA=0.10' \
        'OAT_ZERO_ONLINE_CANONICAL_NOVELTY_BETA=0.50' \
        'OAT_ZERO_ONLINE_CANONICAL_DUAL_TARGET_RATIO=0.80' \
        'OAT_ZERO_ONLINE_CANONICAL_DUAL_MIN_ALPHA=0.10' \
        'OAT_ZERO_ONLINE_CANONICAL_DUAL_MAX_ALPHA=0.50' \
        'OAT_ZERO_ONLINE_CANONICAL_DUAL_ALPHA_LR=0.003' \
        'OAT_ZERO_ONLINE_CANONICAL_DUAL_EMA_DECAY=0.90'; do
        if [[ "$job_record" != *"$required"* ]]; then
          echo "E48 Haarnoja audit failed for $job_id: missing $required" >&2
          exit 1
        fi
      done
    fi
    if [[ "$job_record" != *"--nodelist=node302"* ]] || \
       [[ "$job_record" != *"gres/gpu:a100:1"* ]] || \
       [[ "$job_record" != *"NumCPUs=8"* ]] || \
       [[ "$job_record" != *"MinMemoryNode=64G"* ]] || \
       [[ "$job_record" != *"TimeLimit=7-00:00:00"* ]]; then
      echo "E48 held job $job_id does not attest the frozen placement/time limit" >&2
      exit 1
    fi
  done < "$manifest"
done

scontrol release "${job_ids[@]}"
COHORT_RELEASED=1
trap - EXIT
echo "[e48] released ${#job_ids[@]} matched 50-pass jobs: ${job_ids[*]}"
echo "[e48] identity=$IDENTITY"
