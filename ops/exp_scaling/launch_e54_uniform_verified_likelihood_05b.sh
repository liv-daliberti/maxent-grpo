#!/usr/bin/env bash
# Launch E54's target-free common-mass smoke or three-domain sentinel.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT_DIR/ops/repo_env.sh"

phase="${1:-}"
case "$phase" in
  config|smoke|sentinel) ;;
  *)
    echo "Usage: $0 {config|smoke|sentinel}" >&2
    exit 1
    ;;
esac

PYTHON_BIN="${OAT_ZERO_PYTHON:-$ROOT_DIR/var/seed_paper_eval/paper310/bin/python}"
PROTOCOL="$ROOT_DIR/paper/preregistration/e54_uniform_verified_likelihood_05b.md"
AUDITOR="$ROOT_DIR/ops/exp_scaling/audit_e54_sentinel.py"
E53_IDENTITY="$ROOT_DIR/var/artifacts/e53_verified_replay_05b_sentinel_identity.json"
MODEL_ROOT="$ROOT_DIR/var/cache/huggingface/transformers/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
GRAPH_DATA_ROOT="$ROOT_DIR/var/data/exact_answer_mode_probe"
COUNTDOWN_DATA_ROOT="$ROOT_DIR/var/data/exact_countdown_easy3_probe"
PYTHON_DATA_ROOT="$ROOT_DIR/var/data/python_factor_modebench_v1"
GRAPH_PREFIX=gce54_verified_likelihood_05b_50ep_sentinel
COUNTDOWN_PREFIX=cde54_verified_likelihood_05b_50ep_sentinel_allcs
PYTHON_PREFIX=pye54_verified_likelihood_05b_50ep_sentinel_allcs
SMOKE_PREFIX=gce54_verified_likelihood_05b_smoke
IDENTITY="$ROOT_DIR/var/artifacts/e54_uniform_verified_likelihood_identity.json"
SMOKE_IDENTITY="$ROOT_DIR/var/artifacts/e54_uniform_verified_likelihood_smoke_identity.json"

for required in \
  "$PROTOCOL" "$AUDITOR" "$E53_IDENTITY" "$MODEL_ROOT/config.json" \
  "$GRAPH_DATA_ROOT/train/dataset_dict.json" \
  "$GRAPH_DATA_ROOT/eval/dataset_dict.json" \
  "$COUNTDOWN_DATA_ROOT/train/dataset_dict.json" \
  "$COUNTDOWN_DATA_ROOT/eval/dataset_dict.json" \
  "$PYTHON_DATA_ROOT/identity.json" \
  "$PYTHON_DATA_ROOT/train/dataset_dict.json" \
  "$PYTHON_DATA_ROOT/eval/dataset_dict.json"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing E54 prerequisite: $required" >&2
    exit 1
  fi
done
if [[ "$phase" == sentinel && -e "$IDENTITY" ]]; then
  echo "Fresh E54 identity required; file exists: $IDENTITY" >&2
  exit 1
fi
if [[ "$phase" == smoke && -e "$SMOKE_IDENTITY" ]]; then
  echo "Fresh E54 smoke identity required; file exists: $SMOKE_IDENTITY" >&2
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
  local staging ops_input ops_staging
  SOURCE_HASH="$(hash_tree "$ROOT_DIR/src")"
  SOURCE_ROOT="$ROOT_DIR/var/artifacts/source_snapshots/e54_verified_likelihood_${SOURCE_HASH}/src"
  if [[ ! -f "$SOURCE_ROOT/oat_drgrpo/__init__.py" ]]; then
    mkdir -p "$(dirname "$SOURCE_ROOT")"
    staging="$(mktemp -d "$(dirname "$SOURCE_ROOT")/.source.XXXXXX")"
    mkdir -p "$staging/src"
    cp -a "$ROOT_DIR/src/." "$staging/src/"
    mv "$staging/src" "$SOURCE_ROOT"
    rmdir "$staging"
  fi
  if [[ "$(hash_tree "$SOURCE_ROOT")" != "$SOURCE_HASH" ]]; then
    echo "E54 source snapshot hash mismatch" >&2
    exit 1
  fi

  ops_input="$(mktemp -d "$ROOT_DIR/var/artifacts/source_snapshots/.e54-ops-input.XXXXXX")"
  mkdir -p "$ops_input/slurm"
  for file in \
    repo_env.sh run_experiment.sh train.sh resolve_eval_cadence.py \
    submit_countdown_comparative.sh; do
    cp "$ROOT_DIR/ops/$file" "$ops_input/$file"
  done
  cp "$ROOT_DIR/ops/slurm/train_node302.slurm" \
    "$ops_input/slurm/train_node302.slurm"
  EXECUTION_HASH="$(hash_tree "$ops_input")"
  OPS_ROOT="$ROOT_DIR/var/artifacts/source_snapshots/e54_verified_likelihood_ops_${EXECUTION_HASH}/ops"
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
    echo "E54 execution snapshot hash mismatch" >&2
    exit 1
  fi
}

if [[ "$phase" == config ]]; then
  SOURCE_HASH="$(hash_tree "$ROOT_DIR/src")"
  EXECUTION_HASH=config-only-current-tree
  SOURCE_ROOT="$ROOT_DIR/src"
  OPS_ROOT="$ROOT_DIR/ops"
else
  freeze_execution
fi

export OAT_ZERO_REPO_ROOT="$ROOT_DIR"
export OAT_ZERO_OPS_SNAPSHOT_ROOT="$OPS_ROOT"
export OAT_ZERO_CAMPAIGN_SOURCE_ROOT="$SOURCE_ROOT"
export OAT_ZERO_PRETRAIN="$MODEL_ROOT"
export OAT_ZERO_REQUIRE_EXISTING_DATA=1
export OAT_ZERO_COMPARATIVE_REBUILD=0
export OAT_ZERO_APPEND_MANIFEST=0
export OAT_ZERO_COMPARATIVE_MODEL=qwen2.5-0.5b-instruct
export OAT_ZERO_TRAIN_SEEDS=9010
export OAT_ZERO_ONLY_ARMS=maxent_inverse_canonical_replay
export OAT_ZERO_XDR_TAUS=""

for flag in \
  TOKEN_ENTROPY SEED XDR_ADAPT XDR_TAU_CONTROL XDR_SAC_DUAL \
  MAXENT MAXENT_CONTROL MAXENT_DUAL MAXENT_LENGTH_DUAL DIAYN \
  OUTCOME_COLLISION OUTCOME_COLLISION_OUTSIDE_CENTERING \
  SEMANTIC_SHANNON SEMANTIC_SHANNON_ADVANTAGE \
  QUALITY_GATED_SEMANTIC_NOVELTY \
  SUCCESS_CONDITIONED_SIGNED_SEMANTIC_SHANNON \
  SIGNAL_FIRST_SEMANTIC_BALANCE ONLINE_CANONICAL_MAXENT \
  ONLINE_CANONICAL_HAARNOJA ONLINE_CANONICAL_POLICY_ENTROPY \
  MAXENT_INVERSE MAXENT_INVERSE_CANONICAL; do
  export "OAT_ZERO_INCLUDE_${flag}_ARM=0"
done
export OAT_ZERO_INCLUDE_MAXENT_INVERSE_CANONICAL_REPLAY_ARM=1

export OAT_ZERO_VERIFIED_DISCOVERY_TRACKING=1
export OAT_ZERO_MAXENT_OBJECTIVE=conditional_token_mean
export OAT_ZERO_MAXENT_ALPHA=0.000075
export OAT_ZERO_MAXENT_INVERSE_BASE_ALPHA=0.000075
export OAT_ZERO_MAXENT_INVERSE_ADAPTATION=1
export OAT_ZERO_MAXENT_INVERSE_WARMUP_STEPS=64
export OAT_ZERO_MAXENT_INVERSE_EMA_DECAY=0.90
export OAT_ZERO_MAXENT_CONTROL_TARGET_RATIO=0.0
export OAT_ZERO_MAXENT_DUAL_TARGET_RATIO=0.0
export OAT_ZERO_ONLINE_CANONICAL_BANK_ALPHA=0.0
export OAT_ZERO_ONLINE_CANONICAL_NOVELTY_BETA=0.50
export OAT_ZERO_ONLINE_CANONICAL_BANK_PSEUDOCOUNT=1.0
export OAT_ZERO_ONLINE_CANONICAL_BANK_SURPRISAL_CLIP=5.0
export OAT_ZERO_ONLINE_CANONICAL_KEY_MODE=modebench_outcome
export OAT_ZERO_ONLINE_CANONICAL_DUAL_TARGET_RATIO=0.0
export OAT_ZERO_ONLINE_CANONICAL_POLICY_ENTROPY_ADAPTATION=0
export OAT_ZERO_ONLINE_CANONICAL_REPLAY_ALPHA=0.10
export OAT_ZERO_ONLINE_CANONICAL_REPLAY_OBJECTIVE=verified_likelihood
export OAT_ZERO_ONLINE_CANONICAL_REPLAY_CAPACITY=16
export OAT_ZERO_ONLINE_CANONICAL_REPLAY_WARMUP_STEPS=64
export OAT_ZERO_ONLINE_CANONICAL_REPLAY_EMA_DECAY=0.90
export OAT_ZERO_OUTCOME_COLLISION_COEF=0
export OAT_ZERO_SEMANTIC_SHANNON_COEF=0
export OAT_ZERO_POLICY_ENTROPY_COEF=0
export OAT_ZERO_SEED_ENTROPY_ALPHA=0
export OAT_ZERO_DIAYN_NUM_OPTIONS=0
export OAT_ZERO_DIAYN_MI_BETA=0

export OAT_ZERO_NUM_SAMPLES=16
export OAT_ZERO_LEARNING_RATE=0.0000002
export OAT_ZERO_MAX_QUERIES=100000000
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
export OAT_ZERO_EVAL_MODE_COVERAGE_SEED=530100
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

write_identity() {
  local path="$1" schema="$2" protocol_hash launcher_hash auditor_hash e53_hash
  protocol_hash="$(sha256sum "$PROTOCOL" | cut -d' ' -f1)"
  launcher_hash="$(sha256sum "$0" | cut -d' ' -f1)"
  auditor_hash="$(sha256sum "$AUDITOR" | cut -d' ' -f1)"
  e53_hash="$(sha256sum "$E53_IDENTITY" | cut -d' ' -f1)"
  "$PYTHON_BIN" - "$path" "$schema" "$protocol_hash" "$launcher_hash" \
    "$auditor_hash" "$SOURCE_HASH" "$EXECUTION_HASH" "$e53_hash" <<'PY'
import json
import os
import pathlib
import sys
import tempfile

path = pathlib.Path(sys.argv[1])
payload = {
    "schema": sys.argv[2],
    "protocol_sha256": sys.argv[3],
    "launcher_sha256": sys.argv[4],
    "auditor_sha256": sys.argv[5],
    "source_hash": sys.argv[6],
    "execution_surface_hash": sys.argv[7],
    "e53_control_identity_sha256": sys.argv[8],
    "model": (
        "Qwen2.5-0.5B-Instruct@"
        "7ae557604adf67be50417f59c2c2f167def9a775"
    ),
    "domains": {
        "graph_coloring": {"prompt_pool": 192, "eval_pool": 96},
        "countdown": {"prompt_pool": 384, "eval_pool": 128},
        "python_factor": {"prompt_pool": 384, "eval_pool": 128},
    },
    "arm": "maxent_inverse_canonical_replay",
    "seed": 9010,
    "num_samples": 16,
    "replay": {
        "loss": "uniform_verified_exemplar_likelihood_v1",
        "score_gradient_sum": -1.0,
        "base_alpha": 0.10,
        "capacity": 16,
        "warmup_eligible_steps": 64,
        "ema_decay": 0.90,
        "projection": None,
        "gold_support_feedback": False,
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

submit_domain() {
  local domain="$1"
  case "$domain" in
    graph_coloring)
      export RUN_STAMP_PREFIX="$GRAPH_PREFIX"
      export OAT_ZERO_COMPARATIVE_TASK=graph_coloring
      export OAT_ZERO_COMPARATIVE_DATA_ROOT="$GRAPH_DATA_ROOT"
      export OAT_ZERO_MAX_TRAIN=192
      export OAT_ZERO_EVAL_PROMPT_INTERVAL=48
      export OAT_ZERO_SAVE_STEPS=192
      export OAT_ZERO_SAVE_FROM=192
      export OAT_ZERO_RESUME_STEPS=192
      export OAT_ZERO_RESUME_FROM=192
      export OAT_ZERO_TRAIN_NODELIST="${OAT_ZERO_E54_GRAPH_NODELIST:-node302}"
      export OAT_ZERO_TRAIN_GRES="${OAT_ZERO_E54_GRAPH_GRES:-gpu:a100:1}"
      export OAT_ZERO_TRAIN_PARTITION="${OAT_ZERO_E54_GRAPH_PARTITION:-all}"
      export OAT_ZERO_TRAIN_ACCOUNT="${OAT_ZERO_E54_GRAPH_ACCOUNT:-mltheory}"
      ;;
    countdown)
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
      export OAT_ZERO_TRAIN_NODELIST="${OAT_ZERO_E54_COUNTDOWN_NODELIST:-node020,node022,node023,node024,node026}"
      export OAT_ZERO_TRAIN_GRES="${OAT_ZERO_E54_COUNTDOWN_GRES:-gpu:rtx_3090:1}"
      export OAT_ZERO_TRAIN_PARTITION="${OAT_ZERO_E54_COUNTDOWN_PARTITION:-lowprio}"
      export OAT_ZERO_TRAIN_ACCOUNT="${OAT_ZERO_E54_COUNTDOWN_ACCOUNT:-allcs}"
      ;;
    python_factor)
      export RUN_STAMP_PREFIX="$PYTHON_PREFIX"
      export OAT_ZERO_COMPARATIVE_TASK=python_factor
      export OAT_ZERO_COMPARATIVE_DATA_ROOT="$PYTHON_DATA_ROOT"
      export OAT_ZERO_MAX_TRAIN=384
      export OAT_ZERO_EVAL_PROMPT_INTERVAL=96
      export OAT_ZERO_SAVE_STEPS=384
      export OAT_ZERO_SAVE_FROM=384
      export OAT_ZERO_RESUME_STEPS=384
      export OAT_ZERO_RESUME_FROM=384
      export OAT_ZERO_TRAIN_NODELIST="${OAT_ZERO_E54_PYTHON_NODELIST:-node020,node022,node023,node024,node026}"
      export OAT_ZERO_TRAIN_GRES="${OAT_ZERO_E54_PYTHON_GRES:-gpu:rtx_3090:1}"
      export OAT_ZERO_TRAIN_PARTITION="${OAT_ZERO_E54_PYTHON_PARTITION:-lowprio}"
      export OAT_ZERO_TRAIN_ACCOUNT="${OAT_ZERO_E54_PYTHON_ACCOUNT:-allcs}"
      ;;
    *)
      echo "Unknown E54 domain: $domain" >&2
      exit 1
      ;;
  esac
  export OAT_ZERO_TRAIN_CPUS_PER_TASK=8
  export OAT_ZERO_TRAIN_MEMORY=64G
  export OAT_ZERO_TRAIN_TIME_LIMIT=7-00:00:00
  "$OPS_ROOT/submit_countdown_comparative.sh"
}

if [[ "$phase" == config ]]; then
  export OAT_ZERO_MAX_PROMPT_EPOCHS=50
  export OAT_ZERO_NUM_PROMPT_EPOCH=50
  export OAT_ZERO_COMPARATIVE_CONFIG_ONLY=1
  export OAT_ZERO_SBATCH_HOLD=0
  submit_domain graph_coloring
  submit_domain countdown
  submit_domain python_factor
  echo "[e54] all three sentinel configurations passed; no jobs submitted"
  exit 0
fi

if [[ "$phase" == smoke ]]; then
  write_identity "$SMOKE_IDENTITY" \
    e54_uniform_verified_likelihood_05b_smoke_v1
  export OAT_ZERO_PROTOCOL_IDENTITY="$SMOKE_IDENTITY"
  export OAT_ZERO_MAX_PROMPT_EPOCHS=1
  export OAT_ZERO_NUM_PROMPT_EPOCH=1
  export OAT_ZERO_MAX_TRAIN=16
  export OAT_ZERO_EVAL_PROMPT_INTERVAL=1000000
  export OAT_ZERO_SAVE_STEPS=16
  export OAT_ZERO_SAVE_FROM=16
  export OAT_ZERO_RESUME_STEPS=16
  export OAT_ZERO_RESUME_FROM=16
  export RUN_STAMP_PREFIX="$SMOKE_PREFIX"
  export OAT_ZERO_COMPARATIVE_TASK=graph_coloring
  export OAT_ZERO_COMPARATIVE_DATA_ROOT="$GRAPH_DATA_ROOT"
  export OAT_ZERO_TRAIN_NODELIST="${OAT_ZERO_E54_SMOKE_NODELIST:-node302}"
  export OAT_ZERO_TRAIN_GRES="${OAT_ZERO_E54_SMOKE_GRES:-gpu:a100:1}"
  export OAT_ZERO_TRAIN_PARTITION="${OAT_ZERO_E54_SMOKE_PARTITION:-all}"
  export OAT_ZERO_TRAIN_ACCOUNT="${OAT_ZERO_E54_SMOKE_ACCOUNT:-mltheory}"
  export OAT_ZERO_TRAIN_CPUS_PER_TASK=8
  export OAT_ZERO_TRAIN_MEMORY=64G
  export OAT_ZERO_TRAIN_TIME_LIMIT=04:00:00
  export OAT_ZERO_SBATCH_HOLD=0
  "$OPS_ROOT/submit_countdown_comparative.sh"
  echo "[e54] submitted graph engineering smoke"
  exit 0
fi

write_identity "$IDENTITY" e54_uniform_verified_likelihood_05b_sentinel_v1
export OAT_ZERO_PROTOCOL_IDENTITY="$IDENTITY"
export OAT_ZERO_MAX_PROMPT_EPOCHS=50
export OAT_ZERO_NUM_PROMPT_EPOCH=50
unset OAT_ZERO_COMPARATIVE_CONFIG_ONLY
export OAT_ZERO_SBATCH_HOLD=1
COHORT_RELEASED=0
job_ids=()
cleanup_partial_cohort() {
  local status="$?"
  trap - EXIT
  if [[ "$COHORT_RELEASED" != "1" && "${#job_ids[@]}" -gt 0 ]]; then
    scancel "${job_ids[@]}" 2>/dev/null || true
    echo "[e54] cancelled incomplete held sentinel: ${job_ids[*]}" >&2
  fi
  exit "$status"
}
trap cleanup_partial_cohort EXIT

submit_domain graph_coloring
submit_domain countdown
submit_domain python_factor

for prefix in "$GRAPH_PREFIX" "$COUNTDOWN_PREFIX" "$PYTHON_PREFIX"; do
  manifest="$ROOT_DIR/var/artifacts/${prefix}_comparative_jobs.tsv"
  mapfile -t task_jobs < <(
    awk -F '\t' 'NR > 1 && $3 ~ /^[0-9]+$/ {print $3}' "$manifest"
  )
  if [[ "${#task_jobs[@]}" -ne 1 ]]; then
    echo "E54 ${prefix} has ${#task_jobs[@]} jobs; expected 1" >&2
    exit 1
  fi
  job_ids+=("${task_jobs[@]}")
done
if [[ "${#job_ids[@]}" -ne 3 ]]; then
  echo "E54 held cohort has ${#job_ids[@]} jobs; expected 3" >&2
  exit 1
fi
scontrol release "${job_ids[@]}"
COHORT_RELEASED=1
trap - EXIT
echo "[e54] released held sentinel cohort: ${job_ids[*]}"
