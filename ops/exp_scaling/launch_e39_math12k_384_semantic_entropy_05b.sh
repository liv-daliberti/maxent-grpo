#!/usr/bin/env bash
# Launch E39's fresh matched MATH12K-384 three-arm semantic-entropy cohort.
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
PROTOCOL="$ROOT_DIR/paper/preregistration/e39_math12k_384_semantic_entropy_05b.md"
MATERIALIZER="$ROOT_DIR/ops/math500/materialize_e39_math12k_384.py"
MODEL_ROOT="$ROOT_DIR/var/cache/huggingface/transformers/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
DATA_ROOT="$ROOT_DIR/var/data/math12k_384_math500"
DATA_IDENTITY="$DATA_ROOT/MATERIALIZATION_MANIFEST.json"
TRAIN_PARENT="$ROOT_DIR/var/seed_paper_eval/external/SEED-GRPO/datasets/train/math_12k"
EVAL_PARENT="$ROOT_DIR/var/data/oat_drgrpo_math_paper/eval"
PREFIX=mte39_math12k_384_semantic_entropy_05b_v1
MANIFEST="$ROOT_DIR/var/artifacts/${PREFIX}_comparative_jobs.tsv"
IDENTITY="$ROOT_DIR/var/artifacts/${PREFIX}_identity.json"
EXPECTED_JOBS=9

for required in \
  "$PYTHON_BIN" \
  "$PROTOCOL" \
  "$MATERIALIZER" \
  "$MODEL_ROOT/config.json" \
  "$MODEL_ROOT/tokenizer.json" \
  "$MODEL_ROOT/model.safetensors" \
  "$TRAIN_PARENT/dataset_dict.json" \
  "$EVAL_PARENT/dataset_dict.json"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing frozen E39 prerequisite: $required" >&2
    exit 1
  fi
done
if ! grep -q '^\*\*Status: FROZEN BEFORE LAUNCH' "$PROTOCOL"; then
  echo "E39 protocol is not frozen" >&2
  exit 1
fi
if [[ ! -f "$ROOT_DIR/src/oat_drgrpo/outcome_collision.py" ]] || \
   [[ ! -f "$ROOT_DIR/src/oat_drgrpo/semantic_shannon.py" ]] || \
   ! grep -q 'outcome_collision_coef' "$ROOT_DIR/src/oat_drgrpo/args.py" || \
   ! grep -q 'semantic_shannon_tracker_state' \
     "$ROOT_DIR/src/oat_drgrpo/learner/run.py" || \
   ! grep -q '_last_evaluated_global_step' \
     "$ROOT_DIR/src/oat_drgrpo/learner/run.py"; then
  echo "Current source lacks the frozen E39 semantic-outcome mechanisms" >&2
  exit 1
fi
if [[ "$phase" == full && -e "$MANIFEST" ]]; then
  echo "Fresh E39 prefix required; manifest already exists: $MANIFEST" >&2
  exit 1
fi

if [[ -f "$DATA_IDENTITY" ]]; then
  "$PYTHON_BIN" "$MATERIALIZER" \
    --output-root "$DATA_ROOT" \
    --audit-only >/dev/null
else
  "$PYTHON_BIN" "$MATERIALIZER" --output-root "$DATA_ROOT" >/dev/null
fi
for required in \
  "$DATA_ROOT/train/dataset_dict.json" \
  "$DATA_ROOT/eval/dataset_dict.json" \
  "$DATA_IDENTITY"; do
  if [[ ! -e "$required" ]]; then
    echo "E39 materialization is incomplete: $required" >&2
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
  SOURCE_ROOT="$ROOT_DIR/var/artifacts/source_snapshots/e39_math12k_384_semantic_entropy_05b_${SOURCE_HASH}/src"
  if [[ ! -f "$SOURCE_ROOT/oat_drgrpo/__init__.py" ]]; then
    mkdir -p "$(dirname "$SOURCE_ROOT")"
    staging="$(mktemp -d "$(dirname "$SOURCE_ROOT")/.source.XXXXXX")"
    mkdir -p "$staging/src"
    cp -a "$ROOT_DIR/src/." "$staging/src/"
    mv "$staging/src" "$SOURCE_ROOT"
    rmdir "$staging"
  fi
  if [[ "$(hash_tree "$SOURCE_ROOT")" != "$SOURCE_HASH" ]]; then
    echo "E39 source snapshot hash mismatch" >&2
    exit 1
  fi

  local ops_staging ops_input
  ops_input="$(mktemp -d "$ROOT_DIR/var/artifacts/source_snapshots/.e39-ops-input.XXXXXX")"
  mkdir -p "$ops_input/math500" "$ops_input/slurm"
  cp "$ROOT_DIR/ops/repo_env.sh" "$ops_input/repo_env.sh"
  cp "$ROOT_DIR/ops/run_experiment.sh" "$ops_input/run_experiment.sh"
  cp "$ROOT_DIR/ops/train.sh" "$ops_input/train.sh"
  cp "$ROOT_DIR/ops/resolve_eval_cadence.py" "$ops_input/resolve_eval_cadence.py"
  cp "$ROOT_DIR/ops/submit_countdown_comparative.sh" \
    "$ops_input/submit_countdown_comparative.sh"
  cp "$MATERIALIZER" "$ops_input/math500/materialize_e39_math12k_384.py"
  cp "$ROOT_DIR/ops/slurm/train_node302.slurm" \
    "$ops_input/slurm/train_node302.slurm"
  EXECUTION_HASH="$(hash_tree "$ops_input")"
  OPS_ROOT="$ROOT_DIR/var/artifacts/source_snapshots/e39_math12k_384_semantic_entropy_05b_ops_${EXECUTION_HASH}/ops"
  if [[ ! -f "$OPS_ROOT/submit_countdown_comparative.sh" ]]; then
    mkdir -p "$(dirname "$OPS_ROOT")"
    ops_staging="$(mktemp -d "$(dirname "$OPS_ROOT")/.ops.XXXXXX")"
    mv "$ops_input" "$ops_staging/ops"
    mv "$ops_staging/ops" "$OPS_ROOT"
    rmdir "$ops_staging"
  else
    find "$ops_input" -type f -delete
    rmdir "$ops_input/math500" "$ops_input/slurm" "$ops_input"
  fi
  if [[ "$(hash_tree "$OPS_ROOT")" != "$EXECUTION_HASH" ]]; then
    echo "E39 execution snapshot hash mismatch" >&2
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

export RUN_STAMP_PREFIX="$PREFIX"
export OAT_ZERO_REPO_ROOT="$ROOT_DIR"
export OAT_ZERO_OPS_SNAPSHOT_ROOT="$OPS_ROOT"
export OAT_ZERO_CAMPAIGN_SOURCE_ROOT="$SOURCE_ROOT"
export OAT_ZERO_PRETRAIN="$MODEL_ROOT"
export OAT_ZERO_PROTOCOL_IDENTITY="$IDENTITY"
export OAT_ZERO_REQUIRE_EXISTING_DATA=1
export OAT_ZERO_COMPARATIVE_REBUILD=0
export OAT_ZERO_APPEND_MANIFEST=0
export OAT_ZERO_COMPARATIVE_TASK=math
export OAT_ZERO_COMPARATIVE_MODEL=qwen2.5-0.5b-instruct
export OAT_ZERO_COMPARATIVE_DATA_ROOT="$DATA_ROOT"
export OAT_ZERO_TRAIN_SEEDS=43,44,45
export OAT_ZERO_ONLY_ARMS=grpo,outcome_collision,semantic_shannon
export OAT_ZERO_XDR_TAUS=""
export OAT_ZERO_INCLUDE_TOKEN_ENTROPY_ARM=0
export OAT_ZERO_INCLUDE_SEED_ARM=0
export OAT_ZERO_INCLUDE_XDR_ADAPT_ARM=0
export OAT_ZERO_INCLUDE_XDR_TAU_CONTROL_ARM=0
export OAT_ZERO_INCLUDE_XDR_SAC_DUAL_ARM=0
export OAT_ZERO_INCLUDE_MAXENT_ARM=0
export OAT_ZERO_INCLUDE_MAXENT_CONTROL_ARM=0
export OAT_ZERO_INCLUDE_MAXENT_DUAL_ARM=0
export OAT_ZERO_INCLUDE_MAXENT_LENGTH_DUAL_ARM=0
export OAT_ZERO_INCLUDE_DIAYN_ARM=0
export OAT_ZERO_INCLUDE_OUTCOME_COLLISION_ARM=1
export OAT_ZERO_INCLUDE_SEMANTIC_SHANNON_ARM=1

export OAT_ZERO_OUTCOME_COLLISION_COEF=0.10
export OAT_ZERO_SEMANTIC_SHANNON_COEF=0.10
export OAT_ZERO_SEMANTIC_SHANNON_SURPRISAL_CLIP=5.0
export OAT_ZERO_SEMANTIC_SHANNON_PSEUDOCOUNT=1.0
export OAT_ZERO_MAXENT_OBJECTIVE=sequence
export OAT_ZERO_MAXENT_ALPHA=0
export OAT_ZERO_POLICY_ENTROPY_COEF=0
export OAT_ZERO_SEED_ENTROPY_ALPHA=0
export OAT_ZERO_DIAYN_NUM_OPTIONS=0
export OAT_ZERO_DIAYN_MI_BETA=0
export OAT_ZERO_DIAYN_MI_LEAVE_ONE_OUT=0

export OAT_ZERO_NUM_SAMPLES=16
export OAT_ZERO_LEARNING_RATE=0.0000002
export OAT_ZERO_MAX_TRAIN=384
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

export OAT_ZERO_PROMPT_TEMPLATE=qwen_math
export OAT_ZERO_INPUT_KEY=problem
export OAT_ZERO_OUTPUT_KEY=answer
export OAT_ZERO_EVAL_INPUT_KEY=problem
export OAT_ZERO_EVAL_OUTPUT_KEY=answer
export OAT_ZERO_TEST_SPLIT=math
export OAT_ZERO_VERIFIER_VERSION=math_verify
export OAT_ZERO_PROMPT_MAX_LENGTH=1024
export OAT_ZERO_GENERATE_MAX_LENGTH=1024
export OAT_ZERO_EVAL_GENERATE_MAX_LENGTH=1024
export OAT_ZERO_MAX_MODEL_LEN=2048
export OAT_ZERO_TEMPERATURE=1
export OAT_ZERO_TOP_P=1
export OAT_ZERO_EVAL_TEMPERATURE=0
export OAT_ZERO_EVAL_MODE_COVERAGE_K=8
export OAT_ZERO_EVAL_MODE_COVERAGE_TEMPERATURE=1
export OAT_ZERO_EVAL_MODE_COVERAGE_DRAWS=1
export OAT_ZERO_EVAL_MODE_COVERAGE_SEED=390100
export OAT_ZERO_EVAL_BATCH_SIZE=64
export OAT_ZERO_SYNC_PARAMS_EVERY=1

# Full MATH-500 is evaluated every two passes. This explicit exception is
# frozen because quarter-pass evaluation would dominate the training budget.
export OAT_ZERO_ALLOW_SPARSE_EVAL=1
export OAT_ZERO_EVAL_PROMPT_INTERVAL=768
export OAT_ZERO_EVAL_STEPS=768

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

# One rolling optimizer-resumable checkpoint per pass; terminal model export.
export OAT_ZERO_SAVE_CKPT=1
export OAT_ZERO_SAVE_STEPS=384
export OAT_ZERO_SAVE_FROM=384
export OAT_ZERO_MAX_SAVE_NUM=2
export OAT_ZERO_RESUME_STEPS=384
export OAT_ZERO_RESUME_FROM=384
export OAT_ZERO_MAX_RESUME_NUM=2
export OAT_ZERO_EXPORT_STEPS=0
export OAT_ZERO_EXPORT_FROM=0
export OAT_ZERO_MAX_EXPORT_NUM=1
export OAT_ZERO_PRUNE_RESUME_ON_SUCCESS=0
export OAT_ZERO_AUTO_RESUME=1
export OAT_ZERO_WATCHDOG_REQUEUE=1
export OAT_ZERO_WATCHDOG_MAX_RESTARTS=6

export OAT_ZERO_TRAIN_NODELIST="${OAT_ZERO_E39_TRAIN_NODELIST:-node105,node202,node203,node204}"
export OAT_ZERO_TRAIN_GRES="${OAT_ZERO_E39_TRAIN_GRES:-gpu:a5000:1}"
export OAT_ZERO_TRAIN_CPUS_PER_TASK="${OAT_ZERO_E39_TRAIN_CPUS_PER_TASK:-8}"
export OAT_ZERO_TRAIN_PARTITION="${OAT_ZERO_E39_TRAIN_PARTITION:-all}"
export OAT_ZERO_TRAIN_ACCOUNT="${OAT_ZERO_E39_TRAIN_ACCOUNT:-mltheory}"
export OAT_ZERO_TRAIN_MEMORY="${OAT_ZERO_E39_TRAIN_MEMORY:-64G}"
export OAT_ZERO_TRAIN_TIME_LIMIT="${OAT_ZERO_E39_TRAIN_TIME_LIMIT:-24:00:00}"

write_identity() {
  local protocol_hash launcher_hash data_hash
  protocol_hash="$(sha256sum "$PROTOCOL" | cut -d' ' -f1)"
  launcher_hash="$(sha256sum "$0" | cut -d' ' -f1)"
  data_hash="$(sha256sum "$DATA_IDENTITY" | cut -d' ' -f1)"
  "$PYTHON_BIN" - "$IDENTITY" "$protocol_hash" "$launcher_hash" \
    "$SOURCE_HASH" "$EXECUTION_HASH" "$data_hash" <<'PY'
import json
import os
import pathlib
import sys
import tempfile

path = pathlib.Path(sys.argv[1])
payload = {
    "schema": "e39_math12k_384_semantic_entropy_05b_v1",
    "protocol_sha256": sys.argv[2],
    "launcher_sha256": sys.argv[3],
    "source_hash": sys.argv[4],
    "execution_surface_hash": sys.argv[5],
    "data": {
        "root": "var/data/math12k_384_math500",
        "materialization_manifest": "MATERIALIZATION_MANIFEST.json",
        "materialization_manifest_sha256": sys.argv[6],
        "train_source": "SEED-GRPO math_12k rows 0:384",
        "train_rows": 384,
        "eval_source": "held-out MATH-500",
        "eval_rows": 500,
        "eval_split": "math",
    },
    "model": (
        "Qwen2.5-0.5B-Instruct@"
        "7ae557604adf67be50417f59c2c2f167def9a775"
    ),
    "arms": ["grpo", "outcome_collision", "semantic_shannon"],
    "outcome_collision": {"coefficient": 0.10},
    "semantic_shannon": {
        "coefficient": 0.10,
        "surprisal_clip": 5.0,
        "pseudocount": 1.0,
        "group_leave_one_out": True,
        "history_update": "after_group_scoring",
        "invalid_outcome": "shared_INVALID",
    },
    "answer_key": {
        "kind": "normalized_final_answer_string",
        "symbolic_equivalence_complete": False,
        "interpretation": "final_answer_outcome_stress_test_not_strategy_entropy",
    },
    "seeds": [43, 44, 45],
    "num_samples": 16,
    "prompt_epochs": 10,
    "evaluation": {
        "prompt": "neutral_qwen_math",
        "benchmark": "full_MATH_500",
        "passes": [0, 2, 4, 6, 8, 10],
        "count": 6,
        "prompt_interval": 768,
        "optimizer_step_interval": 768,
        "sparse_cadence_exception": True,
        "duplicate_terminal_policy_evaluation": "skip",
        "k": 8,
        "draws": 1,
        "seeds": [390100],
        "pass_at_1": "greedy",
        "step_zero_pairing": "exact_three_arm",
    },
    "lengths": {"prompt": 1024, "response": 1024, "context": 2048},
    "resume": {
        "semantic_history_checkpointed": True,
        "actor_sync_before_eval_or_rollout": True,
        "checkpoints_per_prompt_epoch": 1,
        "keep": 2,
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

if [[ "$phase" == config ]]; then
  export OAT_ZERO_COMPARATIVE_CONFIG_ONLY=1
  export OAT_ZERO_SBATCH_HOLD=0
  "$OPS_ROOT/submit_countdown_comparative.sh"
  echo "[e39] matched MATH12K-384 three-arm configuration passed; no jobs submitted"
  exit 0
fi

unset OAT_ZERO_COMPARATIVE_CONFIG_ONLY
export OAT_ZERO_SBATCH_HOLD=1

COHORT_RELEASED=0
cleanup_partial_cohort() {
  local status="$?"
  trap - EXIT
  if [[ "$COHORT_RELEASED" != "1" ]]; then
    local -a cleanup_ids=()
    local failure_stamp artifact
    if [[ -f "$MANIFEST" ]]; then
      mapfile -t cleanup_ids < <(
        awk -F '\t' 'NR > 1 && $3 ~ /^[0-9]+$/ {print $3}' "$MANIFEST"
      )
    fi
    if [[ "${#cleanup_ids[@]}" -gt 0 ]]; then
      scancel "${cleanup_ids[@]}" 2>/dev/null || true
      echo "[e39] cancelled incomplete held cohort: ${cleanup_ids[*]}" >&2
    fi
    failure_stamp="$(date +%Y%m%d_%H%M%S)"
    for artifact in "$MANIFEST" "$IDENTITY"; do
      if [[ -e "$artifact" ]]; then
        mv "$artifact" "${artifact}.failed_${failure_stamp}"
      fi
    done
  fi
  exit "$status"
}
trap cleanup_partial_cohort EXIT

write_identity
"$OPS_ROOT/submit_countdown_comparative.sh"

mapfile -t job_ids < <(
  awk -F '\t' 'NR > 1 && $3 ~ /^[0-9]+$/ {print $3}' "$MANIFEST"
)
if [[ "${#job_ids[@]}" -ne "$EXPECTED_JOBS" ]]; then
  echo "E39 cohort has ${#job_ids[@]} jobs; expected $EXPECTED_JOBS" >&2
  exit 1
fi
"$PYTHON_BIN" - "$MANIFEST" <<'PY'
import csv
import pathlib
import sys

with pathlib.Path(sys.argv[1]).open(encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle, delimiter="\t"))
expected = {
    (arm, str(seed))
    for arm in ("grpo", "outcome_collision", "semantic_shannon")
    for seed in (43, 44, 45)
}
observed = {(row["arm"], row["seed"]) for row in rows}
if len(rows) != 9 or observed != expected:
    raise SystemExit(
        f"E39 held manifest mismatch: rows={len(rows)} observed={sorted(observed)}"
    )
PY

for job_id in "${job_ids[@]}"; do
  if ! scontrol update JobId="$job_id" Partition="$OAT_ZERO_TRAIN_PARTITION"; then
    echo "E39 could not normalize held job $job_id to ${OAT_ZERO_TRAIN_PARTITION}" >&2
    exit 1
  fi
done

for job_id in "${job_ids[@]}"; do
  arm="$(awk -F '\t' -v job_id="$job_id" '$3 == job_id {print $1}' "$MANIFEST")"
  seed="$(awk -F '\t' -v job_id="$job_id" '$3 == job_id {print $2}' "$MANIFEST")"
  run_stamp="${PREFIX}_${arm}_s${seed}"
  job_record="$(scontrol show job "$job_id" -o)"
  for required in \
    'JobState=PENDING' 'Reason=JobHeldUser' \
    "RUN_STAMP=${run_stamp}" \
    "OAT_ZERO_SEED=${seed}" \
    'OAT_ZERO_MODEL=qwen2.5-0.5b-instruct' \
    "OAT_ZERO_PRETRAIN=${MODEL_ROOT}" \
    "OAT_ZERO_DATA_ROOT=${DATA_ROOT}" \
    'OAT_ZERO_REQUIRE_EXISTING_DATA=1' \
    'OAT_ZERO_MAX_TRAIN=384' \
    'OAT_ZERO_MAX_QUERIES=100000000' \
    'OAT_ZERO_MAX_PROMPT_EPOCHS=10' \
    'OAT_ZERO_NUM_PROMPT_EPOCH=10' \
    'OAT_ZERO_NUM_SAMPLES=16' \
    'OAT_ZERO_TRAIN_BATCH_SIZE=16' \
    'OAT_ZERO_TRAIN_BATCH_SIZE_PER_DEVICE=1' \
    'OAT_ZERO_ROLLOUT_BATCH_SIZE=1' \
    'OAT_ZERO_ROLLOUT_BATCH_SIZE_PER_DEVICE=1' \
    'OAT_ZERO_N_GPU=1' \
    'OAT_ZERO_NUM_GPUS_PER_ACTOR=1' \
    'OAT_ZERO_LEARNING_RATE=0.0000002' \
    'OAT_ZERO_NUM_PPO_EPOCHS=1' \
    'OAT_ZERO_MAX_NORM=1' \
    'OAT_ZERO_BETA=0' \
    'OAT_ZERO_IGNORE_NO_EOS=0' \
    'OAT_ZERO_INPUT_KEY=problem' \
    'OAT_ZERO_OUTPUT_KEY=answer' \
    'OAT_ZERO_EVAL_INPUT_KEY=problem' \
    'OAT_ZERO_EVAL_OUTPUT_KEY=answer' \
    'OAT_ZERO_PROMPT_TEMPLATE=qwen_math' \
    'OAT_ZERO_TEST_SPLIT=math' \
    'OAT_ZERO_VERIFIER_VERSION=math_verify' \
    'OAT_ZERO_PROMPT_MAX_LENGTH=1024' \
    'OAT_ZERO_GENERATE_MAX_LENGTH=1024' \
    'OAT_ZERO_EVAL_GENERATE_MAX_LENGTH=1024' \
    'OAT_ZERO_MAX_MODEL_LEN=2048' \
    'OAT_ZERO_TEMPERATURE=1' \
    'OAT_ZERO_TOP_P=1' \
    'OAT_ZERO_EVAL_TEMPERATURE=0' \
    'OAT_ZERO_EVAL_BATCH_SIZE=64' \
    'OAT_ZERO_SYNC_PARAMS_EVERY=1' \
    'OAT_ZERO_DIAYN_NUM_OPTIONS=0' \
    'OAT_ZERO_DIAYN_MI_BETA=0.0' \
    'OAT_ZERO_DIAYN_MI_LEAVE_ONE_OUT=0' \
    'OAT_ZERO_EVAL_MODE_COVERAGE_K=8' \
    'OAT_ZERO_EVAL_MODE_COVERAGE_TEMPERATURE=1' \
    'OAT_ZERO_EVAL_MODE_COVERAGE_DRAWS=1' \
    'OAT_ZERO_EVAL_MODE_COVERAGE_SEED=390100' \
    'OAT_ZERO_ALLOW_SPARSE_EVAL=1' \
    'OAT_ZERO_EVAL_PROMPT_INTERVAL=768' \
    'OAT_ZERO_EVAL_STEPS=768' \
    'OAT_ZERO_USE_WB=0' \
    'OAT_ZERO_CANONICAL_ACTION_TASK=none' \
    'OAT_ZERO_CANONICAL_GRAPH_ACTIONS=0' \
    'OAT_ZERO_CANONICAL_GRAPH_LEARNER_SAMPLING=0' \
    'OAT_ZERO_CANONICAL_GRAPH_FIXED_SHAPE_SAMPLING=0' \
    'OAT_ZERO_REPLICATED_FREEFORM_SAMPLING=0' \
    'OAT_ZERO_LOCAL_ACTOR_WEIGHT_SYNC=0' \
    'OAT_ZERO_ZERO_STAGE=2' \
    'OAT_ZERO_VLLM_GPU_RATIO=0.25' \
    'OAT_ZERO_ENABLE_FLASH_ATTN=0' \
    'OAT_ZERO_ADAM_OFFLOAD=0' \
    'OAT_ZERO_ACTIVATION_OFFLOADING=0' \
    'OAT_ZERO_COLLOCATE=1' \
    'OAT_ZERO_VLLM_SLEEP=1' \
    'OAT_ZERO_VLLM_SLEEP_LEVEL=1' \
    'VLLM_USE_V1=0' \
    'HF_HUB_OFFLINE=1' \
    'TRANSFORMERS_OFFLINE=1' \
    'OAT_ZERO_RND_SEED=0' \
    'OAT_ZERO_XDR_MODE_ADAPTIVE=0' \
    'OAT_ZERO_XDR_TAU=inf' \
    'OAT_ZERO_XDR_TAU_CONTROL_TARGET_RATIO=0.0' \
    'OAT_ZERO_XDR_SAC_DUAL_TARGET_RATIO=0.0' \
    'OAT_ZERO_MAXENT_ALPHA=0.0' \
    'OAT_ZERO_MAXENT_OBJECTIVE=sequence' \
    'OAT_ZERO_MAXENT_CONTROL_TARGET_RATIO=0.0' \
    'OAT_ZERO_MAXENT_DUAL_TARGET_RATIO=0.0' \
    'OAT_ZERO_MAXENT_CONTROL_TARGET_ENTROPY=0.0' \
    'OAT_ZERO_MAXENT_DUAL_TARGET_ENTROPY=0.0' \
    'OAT_ZERO_XDR_TAU_CONTROL_RATIO=0.0' \
    'OAT_ZERO_XDR_SAC_DUAL_RATIO=0.0' \
    'OAT_ZERO_MAXENT_LENGTH_TARGET=0.0' \
    'OAT_ZERO_POLICY_ENTROPY_COEF=0.0' \
    'OAT_ZERO_SEED_ENTROPY_ALPHA=0.0' \
    'OAT_ZERO_SEMANTIC_SHANNON_SURPRISAL_CLIP=5.0' \
    'OAT_ZERO_SEMANTIC_SHANNON_PSEUDOCOUNT=1.0' \
    'OAT_ZERO_SAVE_CKPT=1' \
    'OAT_ZERO_SAVE_STEPS=384' \
    'OAT_ZERO_SAVE_FROM=384' \
    'OAT_ZERO_MAX_SAVE_NUM=2' \
    'OAT_ZERO_AUTO_RESUME=1' \
    'OAT_ZERO_WATCHDOG_REQUEUE=1' \
    'OAT_ZERO_WATCHDOG_MAX_RESTARTS=6' \
    'OAT_ZERO_RESUME_STEPS=384' \
    'OAT_ZERO_RESUME_FROM=384' \
    'OAT_ZERO_MAX_RESUME_NUM=2' \
    'OAT_ZERO_PRUNE_RESUME_ON_SUCCESS=0' \
    'OAT_ZERO_EXPORT_STEPS=0' \
    'OAT_ZERO_EXPORT_FROM=0' \
    'OAT_ZERO_MAX_EXPORT_NUM=1' \
    "OAT_ZERO_SOURCE_ROOT=${SOURCE_ROOT}" \
    "OAT_ZERO_OPS_SNAPSHOT_ROOT=${OPS_ROOT}" \
    "OAT_ZERO_PROTOCOL_IDENTITY=${IDENTITY}"; do
    if [[ "$job_record" != *"$required"* ]]; then
      echo "E39 held-job audit failed for $job_id: missing $required" >&2
      exit 1
    fi
  done

  case "$arm" in
    grpo)
      arm_requirements=(
        'OAT_ZERO_VARIANT=grpo'
        'OAT_ZERO_OUTCOME_COLLISION_COEF=0.0'
        'OAT_ZERO_SEMANTIC_SHANNON_COEF=0.0'
      )
      ;;
    outcome_collision)
      arm_requirements=(
        'OAT_ZERO_VARIANT=outcome_collision'
        'OAT_ZERO_OUTCOME_COLLISION_COEF=0.10'
        'OAT_ZERO_SEMANTIC_SHANNON_COEF=0.0'
      )
      ;;
    semantic_shannon)
      arm_requirements=(
        'OAT_ZERO_VARIANT=semantic_shannon'
        'OAT_ZERO_OUTCOME_COLLISION_COEF=0.0'
        'OAT_ZERO_SEMANTIC_SHANNON_COEF=0.10'
        'OAT_ZERO_SEMANTIC_SHANNON_SURPRISAL_CLIP=5.0'
        'OAT_ZERO_SEMANTIC_SHANNON_PSEUDOCOUNT=1.0'
      )
      ;;
    *)
      echo "E39 manifest has unexpected arm for $job_id: $arm" >&2
      exit 1
      ;;
  esac
  for required in "${arm_requirements[@]}"; do
    if [[ "$job_record" != *"$required"* ]]; then
      echo "E39 arm audit failed for $job_id: missing $required" >&2
      exit 1
    fi
  done

  if [[ "$job_record" != *"Partition=${OAT_ZERO_TRAIN_PARTITION}"* ]] || \
     [[ "$job_record" != *"Account=${OAT_ZERO_TRAIN_ACCOUNT}"* ]] || \
     [[ "$job_record" != *"--nodelist=${OAT_ZERO_TRAIN_NODELIST}"* ]] || \
     [[ "$job_record" != *"gres/gpu:a5000=1"* ]] || \
     [[ "$job_record" != *"NumNodes=1"* ]] || \
     [[ "$job_record" != *"NumTasks=1"* ]] || \
     [[ "$job_record" != *"NumCPUs=${OAT_ZERO_TRAIN_CPUS_PER_TASK}"* ]] || \
     [[ "$job_record" != *"MinMemoryNode=${OAT_ZERO_TRAIN_MEMORY}"* ]] || \
     [[ "$job_record" != *"--time=${OAT_ZERO_TRAIN_TIME_LIMIT}"* ]]; then
    echo "E39 held job $job_id does not attest the frozen placement" >&2
    exit 1
  fi
done

scontrol release "${job_ids[@]}"
COHORT_RELEASED=1
trap - EXIT
echo "[e39] released ${#job_ids[@]} matched MATH12K-384 jobs: ${job_ids[*]}"
echo "[e39] manifest=$MANIFEST identity=$IDENTITY data=$DATA_IDENTITY"
