#!/usr/bin/env bash
# Configure or launch E21 free-form conditional-token MaxEnt on MATH.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT_DIR/ops/repo_env.sh"

PHASE="${1:-}"
case "$PHASE" in
  smoke-config|smoke|full-config|full) ;;
  *)
    echo "Usage: $0 {smoke-config|smoke|full-config|full}" >&2
    exit 1
    ;;
esac

PROTOCOL="$ROOT_DIR/paper/preregistration/e21_math_token_policy_maxent.md"
SOURCE_HASH=a781f46e91b857f81b01753fa125b8aff4ffefe58bce6775b0cf028f093f00c6
SOURCE_PARENT="$ROOT_DIR/var/artifacts/source_snapshots/e21_math_token_policy_${SOURCE_HASH}"
SOURCE_ROOT="$SOURCE_PARENT/src"
OPS_HASH=0432750b2a379bd54260ec54ebc112d7e6cfa09d9aa83a98904f988eceb358f3
OPS_ROOT="$ROOT_DIR/var/artifacts/source_snapshots/e21_math_token_policy_ops_${OPS_HASH}/ops"
PYTHON_BIN="$ROOT_DIR/var/seed_paper_eval/paper310/bin/python"
MODEL_ROOT="$TRANSFORMERS_CACHE/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
DATA_ROOT="$ROOT_DIR/var/data/oat_drgrpo_math_paper"
SMOKE_PREFIX=me21_math_token_policy_smoke_v1
FULL_PREFIX=me21_math_token_policy_05b_v1
APPROVAL="$ROOT_DIR/var/artifacts/${SMOKE_PREFIX}_approval.json"

for required in \
  "$PROTOCOL" \
  "$PYTHON_BIN" \
  "$SOURCE_ROOT/oat_drgrpo/__init__.py" \
  "$OPS_ROOT/submit_countdown_comparative.sh" \
  "$OPS_ROOT/exp_scaling/check_e21_math_token_policy_smoke.py" \
  "$MODEL_ROOT/config.json" \
  "$MODEL_ROOT/tokenizer.json" \
  "$DATA_ROOT/IMPORT_MANIFEST.json" \
  "$DATA_ROOT/train/dataset_dict.json" \
  "$DATA_ROOT/eval/dataset_dict.json"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing frozen E21 prerequisite: $required" >&2
    exit 1
  fi
done

observed_source_hash="$(
  PYTHONPATH="$SOURCE_PARENT" "$PYTHON_BIN" -c \
    'import sys; from pathlib import Path; from ops.exp_scaling.check_e14_preflight import source_tree_hash; print(source_tree_hash(Path(sys.argv[1]), logical_repo_root=Path(sys.argv[2])))' \
    "$SOURCE_ROOT" "$ROOT_DIR"
)"
if [[ "$observed_source_hash" != "$SOURCE_HASH" ]]; then
  echo "Frozen E21 source mismatch: expected=$SOURCE_HASH observed=$observed_source_hash" >&2
  exit 1
fi

observed_ops_hash="$(
  "$PYTHON_BIN" - "$OPS_ROOT" <<'PY'
import hashlib
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
digest = hashlib.sha256()
for path in sorted(p for p in root.rglob("*") if p.is_file()):
    relative = path.relative_to(root).as_posix()
    digest.update(relative.encode("utf-8") + b"\0")
    digest.update(hashlib.sha256(path.read_bytes()).digest())
print(digest.hexdigest())
PY
)"
if [[ "$observed_ops_hash" != "$OPS_HASH" ]]; then
  echo "Frozen E21 execution surface mismatch: expected=$OPS_HASH observed=$observed_ops_hash" >&2
  exit 1
fi

protocol_sha256="$(sha256sum "$PROTOCOL" | cut -d' ' -f1)"
echo "[e21] phase=$PHASE label=free-form-conditional-token-MaxEnt"
echo "[e21] source_hash=$SOURCE_HASH execution_surface_hash=$OPS_HASH"
echo "[e21] protocol_sha256=$protocol_sha256"
echo "[e21] data_import_manifest_sha256=$(sha256sum "$DATA_ROOT/IMPORT_MANIFEST.json" | cut -d' ' -f1)"

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

export OAT_ZERO_COMPARATIVE_TASK=math
export OAT_ZERO_COMPARATIVE_MODEL=qwen2.5-0.5b-instruct
export OAT_ZERO_COMPARATIVE_DATA_ROOT="$DATA_ROOT"
export OAT_ZERO_ONLY_ARMS=grpo,maxent,maxent_control,maxent_dual
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
export OAT_ZERO_LEARNING_RATE=0.0000002
export OAT_ZERO_MAX_QUERIES=100000000
export OAT_ZERO_NUM_PROMPT_EPOCH=1
export OAT_ZERO_NUM_PPO_EPOCHS=1
export OAT_ZERO_MAX_NORM=1.0
export OAT_ZERO_BETA=0
export OAT_ZERO_IGNORE_NO_EOS=0
export OAT_ZERO_POLICY_ENTROPY_COEF=0
export OAT_ZERO_MAXENT_OBJECTIVE=conditional_token_mean
export OAT_ZERO_MAXENT_ALPHA=0.0001
export OAT_ZERO_MAXENT_FIXED_ALPHA=0.0001
export OAT_ZERO_MAXENT_CONTROL_BASE_ALPHA=0.000075
export OAT_ZERO_MAXENT_CONTROL_RATIO=0.8
export OAT_ZERO_MAXENT_CONTROL_TARGET_ENTROPY=0
export OAT_ZERO_MAXENT_CONTROL_MAX_ALPHA=0.00015
export OAT_ZERO_MAXENT_CONTROL_EMA_DECAY=0.9
export OAT_ZERO_MAXENT_CONTROL_GAIN=2
export OAT_ZERO_MAXENT_DUAL_BASE_ALPHA=0.000075
export OAT_ZERO_MAXENT_DUAL_RATIO=0.8
export OAT_ZERO_MAXENT_DUAL_TARGET_ENTROPY=0
export OAT_ZERO_MAXENT_DUAL_MIN_ALPHA=0.00005
export OAT_ZERO_MAXENT_DUAL_MAX_ALPHA=0.00015
export OAT_ZERO_MAXENT_DUAL_ALPHA_LR=0.005
export OAT_ZERO_MAXENT_LENGTH_TARGET=0

export OAT_ZERO_TRAIN_BATCH_SIZE=16
export OAT_ZERO_TRAIN_BATCH_SIZE_PER_DEVICE=4
export OAT_ZERO_ROLLOUT_BATCH_SIZE=1
export OAT_ZERO_ROLLOUT_BATCH_SIZE_PER_DEVICE=1
export OAT_ZERO_N_GPU=1
export OAT_ZERO_NUM_GPUS_PER_ACTOR=1
export OAT_ZERO_ZERO_STAGE=2
export OAT_ZERO_VLLM_GPU_RATIO=0.30
export OAT_ZERO_ENABLE_FLASH_ATTN=0
export OAT_ZERO_ADAM_OFFLOAD=0
export OAT_ZERO_ACTIVATION_OFFLOADING=0
export OAT_ZERO_COLLOCATE=1
export OAT_ZERO_VLLM_SLEEP=1

export OAT_ZERO_PROMPT_TEMPLATE=qwen_math
export OAT_ZERO_INPUT_KEY=problem
export OAT_ZERO_OUTPUT_KEY=answer
export OAT_ZERO_EVAL_INPUT_KEY=problem
export OAT_ZERO_EVAL_OUTPUT_KEY=answer
export OAT_ZERO_TEST_SPLIT=math
export OAT_ZERO_VERIFIER_VERSION=math_verify
export OAT_ZERO_CANONICAL_ACTION_TASK=none
export OAT_ZERO_CANONICAL_GRAPH_ACTIONS=0
export OAT_ZERO_CANONICAL_GRAPH_LEARNER_SAMPLING=0
export OAT_ZERO_CANONICAL_GRAPH_FIXED_SHAPE_SAMPLING=0
export OAT_ZERO_PROMPT_MAX_LENGTH=1024
export OAT_ZERO_GENERATE_MAX_LENGTH=1024
export OAT_ZERO_EVAL_GENERATE_MAX_LENGTH=1024
export OAT_ZERO_MAX_MODEL_LEN=2048
export OAT_ZERO_TEMPERATURE=1
export OAT_ZERO_TOP_P=1
export OAT_ZERO_EVAL_TEMPERATURE=0
export OAT_ZERO_EVAL_MODE_COVERAGE_K=8
export OAT_ZERO_EVAL_MODE_COVERAGE_TEMPERATURE=1
export OAT_ZERO_EVAL_BATCH_SIZE=32
export OAT_ZERO_SYNC_PARAMS_EVERY=1

export OAT_ZERO_TRAIN_NODELIST="${OAT_ZERO_E21_TRAIN_NODELIST:-node023,node024}"
export OAT_ZERO_TRAIN_GRES="${OAT_ZERO_E21_TRAIN_GRES:-gpu:rtx_3090:1}"
export OAT_ZERO_TRAIN_PARTITION="${OAT_ZERO_E21_TRAIN_PARTITION:-lowprio}"
export OAT_ZERO_TRAIN_ACCOUNT="${OAT_ZERO_E21_TRAIN_ACCOUNT:-allcs}"
export OAT_ZERO_TRAIN_MEMORY="${OAT_ZERO_E21_TRAIN_MEMORY:-32G}"
export OAT_ZERO_SBATCH_HOLD=1

verify_approval() {
  "$PYTHON_BIN" - "$APPROVAL" "$PROTOCOL" "$SOURCE_HASH" "$OPS_HASH" "$SMOKE_PREFIX" <<'PY'
import hashlib
import json
import pathlib
import sys

approval_path, protocol_path = map(pathlib.Path, sys.argv[1:3])
source_hash, ops_hash, stamp = sys.argv[3:6]
if not approval_path.is_file():
    raise SystemExit(f"E21 full launch requires smoke approval: {approval_path}")
approval = json.loads(approval_path.read_text(encoding="utf-8"))
expected = {
    "schema": "e21_math_token_policy_smoke_approval_v1",
    "approved": True,
    "stamp": stamp,
    "source_hash": source_hash,
    "execution_surface_hash": ops_hash,
    "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
}
for key, value in expected.items():
    if approval.get(key) != value:
        raise SystemExit(
            f"E21 smoke approval mismatch for {key}: "
            f"expected={value!r} observed={approval.get(key)!r}"
        )
print(f"[e21] verified smoke approval {approval_path}")
PY
}

submit_stage() {
  local stage="$1"
  if [[ "$stage" == smoke ]]; then
    export RUN_STAMP_PREFIX="$SMOKE_PREFIX"
    export OAT_ZERO_TRAIN_SEEDS=9007
    export OAT_ZERO_MAX_TRAIN=64
    export OAT_ZERO_EVAL_PROMPT_INTERVAL=16
    export OAT_ZERO_MAXENT_CONTROL_WARMUP_STEPS=16
    export OAT_ZERO_MAXENT_DUAL_WARMUP_STEPS=16
    export OAT_ZERO_SAVE_CKPT=0
    export OAT_ZERO_SAVE_STEPS=16
    export OAT_ZERO_SAVE_FROM=16
    export OAT_ZERO_MAX_SAVE_NUM=1
    export OAT_ZERO_AUTO_RESUME=0
    export OAT_ZERO_WATCHDOG_REQUEUE=0
    export OAT_ZERO_TRAIN_TIME_LIMIT="${OAT_ZERO_E21_SMOKE_TIME_LIMIT:-04:00:00}"
  else
    export RUN_STAMP_PREFIX="$FULL_PREFIX"
    export OAT_ZERO_TRAIN_SEEDS=43,44,45
    export OAT_ZERO_MAX_TRAIN=8523
    export OAT_ZERO_EVAL_PROMPT_INTERVAL=2130
    export OAT_ZERO_MAXENT_CONTROL_WARMUP_STEPS=64
    export OAT_ZERO_MAXENT_DUAL_WARMUP_STEPS=64
    export OAT_ZERO_SAVE_CKPT=1
    export OAT_ZERO_SAVE_STEPS=2130
    export OAT_ZERO_SAVE_FROM=2130
    export OAT_ZERO_MAX_SAVE_NUM=5
    export OAT_ZERO_AUTO_RESUME=1
    export OAT_ZERO_WATCHDOG_REQUEUE=1
    export OAT_ZERO_WATCHDOG_MAX_RESTARTS=6
    export OAT_ZERO_TRAIN_TIME_LIMIT="${OAT_ZERO_E21_FULL_TIME_LIMIT:-168:00:00}"
  fi
  bash "$OPS_ROOT/submit_countdown_comparative.sh"
}

if [[ "$PHASE" == *-config ]]; then
  export OAT_ZERO_COMPARATIVE_CONFIG_ONLY=1
  if [[ "$PHASE" == smoke-config ]]; then
    submit_stage smoke
  else
    submit_stage full
  fi
  echo "[e21] configuration passed; no jobs submitted"
  exit 0
fi

unset OAT_ZERO_COMPARATIVE_CONFIG_ONLY
if [[ "$PHASE" == full ]]; then
  verify_approval
  stage=full
  expected_jobs=12
  prefix="$FULL_PREFIX"
else
  stage=smoke
  expected_jobs=4
  prefix="$SMOKE_PREFIX"
fi

submit_stage "$stage"
manifest="$ROOT_DIR/var/artifacts/${prefix}_comparative_jobs.tsv"
mapfile -t job_ids < <(awk -F '\t' 'NR > 1 && $3 ~ /^[0-9]+$/ {print $3}' "$manifest")
if (( ${#job_ids[@]} != expected_jobs )); then
  echo "E21 ${stage} cohort incomplete (${#job_ids[@]}/${expected_jobs}); jobs remain held" >&2
  exit 1
fi
scontrol release "${job_ids[@]}"
echo "[e21] released complete ${stage} cohort: ${job_ids[*]}"
