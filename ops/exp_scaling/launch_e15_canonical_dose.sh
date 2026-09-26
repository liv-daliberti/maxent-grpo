#!/usr/bin/env bash
# Launch only E15's prospectively frozen single-seed canonical dose calibration.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT_DIR/ops/repo_env.sh"
export VLLM_USE_V1=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# The 128-update dose runs are restart-invalid. A failed allocation is a new
# attempt with a fresh stamp, never an automatic resume beyond the query gate.
export OAT_ZERO_AUTO_RESUME=0
export OAT_ZERO_WATCHDOG_REQUEUE=0
PYTHON_BIN="${OAT_ZERO_PYTHON:-$ROOT_DIR/var/seed_paper_eval/paper310/bin/python}"
PHASE="${1:-}"
STAMP="${RUN_STAMP_PREFIX:-e15_${PHASE}_$(date +%Y%m%d_%H%M%S)}"

case "$PHASE" in
  m075|m075-config)
    PROTOCOL_ARM=M075
    MAXENT_ALPHA=0.075
    ;;
  m10|m10-config)
    PROTOCOL_ARM=M10
    MAXENT_ALPHA=0.10
    ;;
  *)
    echo "Usage: $0 {m075-config|m075|m10-config|m10}" >&2
    exit 1
    ;;
esac
if [[ "$PHASE" == *-config ]]; then
  CONFIG_ONLY=1
else
  CONFIG_ONLY=0
fi

TARGET_UPDATES=128
LEARNING_RATE=0.0000002
SAVE_CKPT=1
MANIFEST_ARM=maxent
MAX_QUERIES="$($PYTHON_BIN "$ROOT_DIR/ops/exp_scaling/e14_budget.py" \
  --updates "$TARGET_UPDATES" --trajectories-per-update 16)"
if [[ "$MAX_QUERIES" != "2032" ]]; then
  echo "E15 budget helper returned ${MAX_QUERIES}; frozen value is 2032." >&2
  exit 1
fi
MAX_TRAIN="$MAX_QUERIES"

DATA_ROOT="${OAT_ZERO_COMPARATIVE_DATA_ROOT:-$ROOT_DIR/var/data/exact_answer_mode_probe}"
if [[ ! -f "$DATA_ROOT/train/dataset_dict.json" ]] \
  || [[ ! -f "$DATA_ROOT/eval/dataset_dict.json" ]]; then
  echo "Missing frozen graph-coloring data at $DATA_ROOT" >&2
  exit 1
fi

# Use E14's logical source-tree hash. Unlike a shell hash over sha256sum
# output, this is invariant to whether the identical tree is in src/ or an
# immutable source snapshot.
source_hash="$(PYTHONPATH="$ROOT_DIR" "$PYTHON_BIN" -c \
  'import sys; from pathlib import Path; from ops.exp_scaling.check_e14_preflight import source_tree_hash; print(source_tree_hash(Path(sys.argv[1]), logical_repo_root=Path(sys.argv[2])))' \
  "$ROOT_DIR/src" "$ROOT_DIR")"

c0_approval_path="${OAT_ZERO_E15_C0_APPROVAL:-}"
e14_outcome_path="${OAT_ZERO_E15_E14_OUTCOME:-}"
if [[ -z "$c0_approval_path" || -z "$e14_outcome_path" ]]; then
  echo "E15 requires explicit OAT_ZERO_E15_C0_APPROVAL and OAT_ZERO_E15_E14_OUTCOME artifact paths." >&2
  exit 1
fi
authorization_summary="$($PYTHON_BIN \
  "$ROOT_DIR/ops/exp_scaling/verify_e15_authorization.py" \
  --c0-approval "$c0_approval_path" \
  --e14-outcome "$e14_outcome_path" \
  --expected-source-hash "$source_hash" \
  --repo-root "$ROOT_DIR")"
c0_approval_path="$(realpath "$c0_approval_path")"
e14_outcome_path="$(realpath "$e14_outcome_path")"
c0_approval_sha256="$(sha256sum "$c0_approval_path" | cut -d' ' -f1)"
e14_outcome_sha256="$(sha256sum "$e14_outcome_path" | cut -d' ' -f1)"

runtime_identity="$($PYTHON_BIN "$ROOT_DIR/ops/exp_scaling/verify_e14_runtime.py" \
  | tail -n 1)"
dataset_identity="$($PYTHON_BIN "$ROOT_DIR/ops/exp_scaling/verify_e14_dataset.py" \
  --data-root "$DATA_ROOT")"

echo "[e15] phase=$PHASE stamp=$STAMP"
echo "[e15] protocol_arm=$PROTOCOL_ARM manifest_arm=$MANIFEST_ARM maxent_alpha=$MAXENT_ALPHA"
echo "[e15] formulation=canonical_graph_actions support={1,2,3} horizon=3 max_entropy=3.295836866004329"
echo "[e15] authorization=$authorization_summary"
echo "[e15] dataset_identity=$dataset_identity"
echo "[e15] source_hash=$source_hash"
echo "[e15] runtime_identity=$runtime_identity"
echo "[e15] target_optimizer_updates=$TARGET_UPDATES trajectory_query_budget=$MAX_QUERIES group_size=16"

identity_path="$ROOT_DIR/var/artifacts/${STAMP}_e15_identity.tsv"
{
  printf 'key\tvalue\nphase\t%s\nstamp\t%s\nsource_hash\t%s\ndataset_identity\t%s\nruntime_identity\t%s\ntarget_optimizer_updates\t128\ntrajectory_query_budget\t2032\ngroup_size\t16\nprotocol\tE15\nprotocol_arm\t%s\narm\tmaxent\nmaxent_alpha\t%s\nc0_approval\t%s\nc0_approval_sha256\t%s\ne14_outcome\t%s\ne14_outcome_sha256\t%s\n' \
    "$PHASE" "$STAMP" "$source_hash" "$dataset_identity" \
    "$runtime_identity" "$PROTOCOL_ARM" "$MAXENT_ALPHA" \
    "$c0_approval_path" "$c0_approval_sha256" \
    "$e14_outcome_path" "$e14_outcome_sha256"
} > "$identity_path"
echo "[e15] identity_artifact=$identity_path"
echo "[e15] recovery=disabled auto_resume=0 watchdog_requeue=0"

export RUN_STAMP_PREFIX="$STAMP"
export OAT_ZERO_COMPARATIVE_TASK=graph_coloring
export OAT_ZERO_COMPARATIVE_MODEL=qwen2.5-0.5b-instruct
export OAT_ZERO_COMPARATIVE_DATA_ROOT="$DATA_ROOT"
export OAT_ZERO_TRAIN_SEEDS=9005
export OAT_ZERO_ONLY_ARMS=maxent
export OAT_ZERO_INCLUDE_TOKEN_ENTROPY_ARM=0
export OAT_ZERO_INCLUDE_SEED_ARM=0
export OAT_ZERO_INCLUDE_XDR_ADAPT_ARM=0
export OAT_ZERO_INCLUDE_XDR_TAU_CONTROL_ARM=0
export OAT_ZERO_INCLUDE_XDR_SAC_DUAL_ARM=0
export OAT_ZERO_INCLUDE_MAXENT_ARM=1
export OAT_ZERO_INCLUDE_MAXENT_CONTROL_ARM=0
export OAT_ZERO_INCLUDE_MAXENT_DUAL_ARM=0
export OAT_ZERO_INCLUDE_MAXENT_LENGTH_DUAL_ARM=0
export OAT_ZERO_COMPARATIVE_REBUILD=0
export OAT_ZERO_APPEND_MANIFEST=0
export OAT_ZERO_XDR_TAUS=""
export OAT_ZERO_NUM_SAMPLES=16
export OAT_ZERO_MAX_TRAIN="$MAX_TRAIN"
export OAT_ZERO_MAX_QUERIES="$MAX_QUERIES"
export OAT_ZERO_NUM_PROMPT_EPOCH=1
export OAT_ZERO_EVAL_PROMPT_INTERVAL=32
export OAT_ZERO_SAVE_STEPS=32
export OAT_ZERO_SAVE_FROM=32
export OAT_ZERO_SAVE_CKPT="$SAVE_CKPT"
export OAT_ZERO_MAX_SAVE_NUM=5
export OAT_ZERO_MAX_SAVE_MEM=2000
export OAT_ZERO_LEARNING_RATE="$LEARNING_RATE"
export OAT_ZERO_MAXENT_ALPHA="$MAXENT_ALPHA"
export OAT_ZERO_MAXENT_CONTROL_RATIO=0
export OAT_ZERO_MAXENT_CONTROL_TARGET_ENTROPY=0
export OAT_ZERO_MAXENT_DUAL_RATIO=0
export OAT_ZERO_MAXENT_DUAL_TARGET_ENTROPY=0
export OAT_ZERO_MAXENT_LENGTH_TARGET=0
export OAT_ZERO_POLICY_ENTROPY_COEF=0
export OAT_ZERO_NUM_PPO_EPOCHS=1
export OAT_ZERO_MAX_NORM=1.0
export OAT_ZERO_BETA=0
export OAT_ZERO_IGNORE_NO_EOS=0
export OAT_ZERO_TRAIN_BATCH_SIZE=16
export OAT_ZERO_TRAIN_BATCH_SIZE_PER_DEVICE=4
export OAT_ZERO_ROLLOUT_BATCH_SIZE=1
export OAT_ZERO_ROLLOUT_BATCH_SIZE_PER_DEVICE=1
export OAT_ZERO_N_GPU=1
export OAT_ZERO_NUM_GPUS_PER_ACTOR=1
export OAT_ZERO_PROMPT_TEMPLATE=qwen_graph_digits
export OAT_ZERO_TEST_SPLIT=multi_answer
export OAT_ZERO_CANONICAL_GRAPH_ACTIONS=1
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
export OAT_ZERO_EVAL_BATCH_SIZE=64
export OAT_ZERO_PROMPT_MAX_LENGTH=256
export OAT_ZERO_ZERO_STAGE=2
export OAT_ZERO_VLLM_GPU_RATIO=0.25
export OAT_ZERO_ENABLE_FLASH_ATTN=0
export OAT_ZERO_ADAM_OFFLOAD=0
export OAT_ZERO_ACTIVATION_OFFLOADING=0
export OAT_ZERO_COLLOCATE=1
export OAT_ZERO_VLLM_SLEEP=1
export OAT_ZERO_PYTHON="$ROOT_DIR/var/seed_paper_eval/paper310/bin/python"
export OAT_ZERO_PYTHON_LIB_DIR="$ROOT_DIR/var/seed_paper_eval/paper310/lib"
export OAT_ZERO_PRETRAIN="$TRANSFORMERS_CACHE/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
export OAT_ZERO_COMPARATIVE_CONFIG_ONLY="$CONFIG_ONLY"
export OAT_ZERO_TRAIN_TIME_LIMIT="${OAT_ZERO_TRAIN_TIME_LIMIT:-02:00:00}"
export OAT_ZERO_TRAIN_NODELIST="${OAT_ZERO_TRAIN_NODELIST:-node302}"
export OAT_ZERO_TRAIN_GRES="${OAT_ZERO_TRAIN_GRES:-gpu:a100:1}"
export OAT_ZERO_TRAIN_PARTITION="${OAT_ZERO_TRAIN_PARTITION:-mltheory}"
export OAT_ZERO_TRAIN_ACCOUNT="${OAT_ZERO_TRAIN_ACCOUNT:-mltheory}"

# --export=ALL is used by the shared submitter. Remove inherited paths and
# snapshots that could redirect a nominal E15 run away from the frozen inputs.
unset OAT_ZERO_CAMPAIGN_SOURCE_ROOT
unset OAT_ZERO_PROMPT_DATA
unset OAT_ZERO_EVAL_DATA
unset OAT_ZERO_E14_C0_APPROVAL
unset OAT_ZERO_E14_PREFLIGHT_APPROVAL
unset SAVE_PATH

exec bash "$ROOT_DIR/ops/submit_countdown_comparative.sh"
