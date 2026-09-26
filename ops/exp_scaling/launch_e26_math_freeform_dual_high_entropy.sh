#!/usr/bin/env bash
# Launch E26's treatment-only high-entropy base-preserving MATH dual.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT_DIR/ops/repo_env.sh"

phase="${1:-}"
case "$phase" in
  config|full) ;;
  *) echo "Usage: $0 {config|full}" >&2; exit 2 ;;
esac

PYTHON_BIN="${OAT_ZERO_PYTHON:-$ROOT_DIR/var/seed_paper_eval/paper310/bin/python}"
PROTOCOL="$ROOT_DIR/paper/preregistration/e26_math_freeform_dual_high_entropy.md"
CALIBRATION="$ROOT_DIR/paper/results/e26_math_freeform_dual_high_entropy_calibration.json"
DATA_ROOT="$ROOT_DIR/var/data/oat_drgrpo_math_paper"
MODEL_ROOT="$ROOT_DIR/var/cache/huggingface/transformers/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
SOURCE_HASH=217547637154ed74c2356eafec6dc885914793f8bb530c2b6918ca9a2c245448
EXECUTION_HASH=05d43e4ae78d75d9e98c5b3d07d6ebd88cc545cc9039774ad0b6bf7cfebc05ce
SOURCE_ROOT="$ROOT_DIR/var/artifacts/source_snapshots/e21_math_conditional_token_${SOURCE_HASH}/src"
OPS_ROOT="$ROOT_DIR/var/artifacts/source_snapshots/e21_math_conditional_token_ops_${EXECUTION_HASH}/ops"
PREFIX=mte26_math_freeform_conditional_dual_high_entropy_05b_v2
MANIFEST="$ROOT_DIR/var/artifacts/${PREFIX}_comparative_jobs.tsv"
IDENTITY="$ROOT_DIR/var/artifacts/${PREFIX}_identity.json"
TARGET=0.4567
ALPHA_BASE=0.000075
ALPHA_MAX=0.00060
ALPHA_LR=0.010

for required in \
  "$PYTHON_BIN" "$PROTOCOL" "$CALIBRATION" \
  "$MODEL_ROOT/config.json" "$MODEL_ROOT/tokenizer.json" \
  "$DATA_ROOT/train/dataset_dict.json" "$DATA_ROOT/eval/dataset_dict.json" \
  "$SOURCE_ROOT/oat_drgrpo/__init__.py" \
  "$OPS_ROOT/submit_countdown_comparative.sh" \
  "$OPS_ROOT/math500/import_oat_math.py"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing frozen E26 prerequisite: $required" >&2
    exit 1
  fi
done

if ! grep -q '^\*\*Status: FROZEN' "$PROTOCOL"; then
  echo "E26 protocol is not frozen" >&2
  exit 1
fi

source_hash="$(PYTHONPATH="$ROOT_DIR" "$PYTHON_BIN" -c '
import sys
from pathlib import Path
from ops.exp_scaling.check_e14_preflight import source_tree_hash
print(source_tree_hash(Path(sys.argv[1]), logical_repo_root=Path(sys.argv[2])))
' "$SOURCE_ROOT" "$ROOT_DIR")"
execution_hash="$("$PYTHON_BIN" - "$OPS_ROOT" <<'PY'
import hashlib
import pathlib
import sys
root = pathlib.Path(sys.argv[1])
files = (
    "repo_env.sh", "train.sh", "run_experiment.sh",
    "submit_countdown_comparative.sh", "resolve_eval_cadence.py",
    "math500/import_oat_math.py", "slurm/train_node302.slurm",
    "exp_scaling/check_e21_math_token_smoke.py",
)
h = hashlib.sha256()
for relative in files:
    payload = (root / relative).read_bytes()
    h.update(relative.encode() + b"\0" + hashlib.sha256(payload).digest())
print(h.hexdigest())
PY
)"
if [[ "$source_hash" != "$SOURCE_HASH" || "$execution_hash" != "$EXECUTION_HASH" ]]; then
  echo "E26 frozen source/execution hash mismatch" >&2
  exit 1
fi

"$PYTHON_BIN" "$OPS_ROOT/math500/import_oat_math.py" \
  --output-root "$DATA_ROOT" --audit-only >/dev/null

export RUN_STAMP_PREFIX="$PREFIX"
export OAT_ZERO_REPO_ROOT="$ROOT_DIR"
export OAT_ZERO_OPS_SNAPSHOT_ROOT="$OPS_ROOT"
export OAT_ZERO_CAMPAIGN_SOURCE_ROOT="$SOURCE_ROOT"
export OAT_ZERO_PROTOCOL_IDENTITY="$PROTOCOL"
export OAT_ZERO_PRETRAIN="$MODEL_ROOT"
export OAT_ZERO_COMPARATIVE_TASK=math
export OAT_ZERO_COMPARATIVE_MODEL=qwen2.5-0.5b-instruct
export OAT_ZERO_COMPARATIVE_DATA_ROOT="$DATA_ROOT"
export OAT_ZERO_REQUIRE_EXISTING_DATA=1
export OAT_ZERO_COMPARATIVE_REBUILD=0
export OAT_ZERO_APPEND_MANIFEST=0
export OAT_ZERO_TRAIN_SEEDS=43,44,45
export OAT_ZERO_ONLY_ARMS=maxent_dual
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
export OAT_ZERO_MAXENT_DUAL_BASE_ALPHA="$ALPHA_BASE"
export OAT_ZERO_MAXENT_DUAL_MIN_ALPHA="$ALPHA_BASE"
export OAT_ZERO_MAXENT_DUAL_MAX_ALPHA="$ALPHA_MAX"
export OAT_ZERO_MAXENT_DUAL_TARGET_ENTROPY="$TARGET"
export OAT_ZERO_MAXENT_DUAL_RATIO=1.0
export OAT_ZERO_MAXENT_DUAL_WARMUP_STEPS=64
export OAT_ZERO_MAXENT_DUAL_ALPHA_LR="$ALPHA_LR"
export OAT_ZERO_MAXENT_LENGTH_TARGET=0
export OAT_ZERO_POLICY_ENTROPY_COEF=0

export OAT_ZERO_NUM_SAMPLES=16
export OAT_ZERO_LEARNING_RATE=0.0000002
export OAT_ZERO_MAX_TRAIN=8523
export OAT_ZERO_MAX_QUERIES=100000000
export OAT_ZERO_NUM_PROMPT_EPOCH=1
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

export OAT_ZERO_EVAL_PROMPT_INTERVAL=2129
export OAT_ZERO_SAVE_STEPS=2129
export OAT_ZERO_SAVE_FROM=2129
export OAT_ZERO_SAVE_CKPT=1
export OAT_ZERO_MAX_SAVE_NUM=2
export OAT_ZERO_AUTO_RESUME=1
export OAT_ZERO_WATCHDOG_REQUEUE=1
export OAT_ZERO_WATCHDOG_MAX_RESTARTS=6
export OAT_ZERO_TRAIN_NODELIST="${OAT_ZERO_E26_TRAIN_NODELIST:-node105}"
export OAT_ZERO_TRAIN_GRES="${OAT_ZERO_E26_TRAIN_GRES:-gpu:a5000:1}"
export OAT_ZERO_TRAIN_PARTITION="${OAT_ZERO_E26_TRAIN_PARTITION:-mltheory}"
export OAT_ZERO_TRAIN_ACCOUNT="${OAT_ZERO_E26_TRAIN_ACCOUNT:-mltheory}"
export OAT_ZERO_TRAIN_MEMORY="${OAT_ZERO_E26_TRAIN_MEMORY:-64G}"
export OAT_ZERO_TRAIN_TIME_LIMIT="${OAT_ZERO_E26_TRAIN_TIME_LIMIT:-7-00:00:00}"

if [[ "$phase" == config ]]; then
  export OAT_ZERO_COMPARATIVE_CONFIG_ONLY=1
  export OAT_ZERO_SBATCH_HOLD=0
  exec "$OPS_ROOT/submit_countdown_comparative.sh"
fi

unset OAT_ZERO_COMPARATIVE_CONFIG_ONLY
export OAT_ZERO_SBATCH_HOLD=1
if [[ -e "$MANIFEST" ]]; then
  echo "Fresh E26 prefix required; manifest already exists: $MANIFEST" >&2
  exit 1
fi

"$OPS_ROOT/submit_countdown_comparative.sh"
mapfile -t job_ids < <(
  awk -F '\t' 'NR > 1 && $1 == "maxent_dual" && $2 ~ /^(43|44|45)$/ && $3 ~ /^[0-9]+$/ {print $3}' "$MANIFEST"
)
if [[ "${#job_ids[@]}" -ne 3 ]]; then
  echo "E26 cohort incomplete (${#job_ids[@]}/3); submitted jobs remain held" >&2
  exit 1
fi

for job_id in "${job_ids[@]}"; do
  job_record="$(scontrol show job -o "$job_id")"
  manifest_row="$(awk -F '\t' -v id="$job_id" '$3 == id {print $2 "|" $4}' "$MANIFEST")"
  IFS='|' read -r seed run_stamp <<< "$manifest_row"
  for required in \
    'JobState=PENDING' 'Reason=JobHeldUser' 'ReqNodeList=node105' \
    'gres/gpu:a5000=1' 'mem=64G' \
    "RUN_STAMP=${run_stamp}" "OAT_ZERO_SEED=${seed}" \
    'OAT_ZERO_VARIANT=maxent_dual' \
    'OAT_ZERO_MAXENT_OBJECTIVE=conditional_token_mean' \
    "OAT_ZERO_MAXENT_DUAL_TARGET_ENTROPY=${TARGET}" \
    "OAT_ZERO_MAXENT_DUAL_MIN_ALPHA=${ALPHA_BASE}" \
    "OAT_ZERO_MAXENT_DUAL_MAX_ALPHA=${ALPHA_MAX}" \
    "OAT_ZERO_MAXENT_DUAL_ALPHA_LR=${ALPHA_LR}" \
    'OAT_ZERO_MAXENT_DUAL_WARMUP_STEPS=64' \
    'OAT_ZERO_MAX_TRAIN=8523'; do
    if [[ "$job_record" != *"$required"* ]]; then
      echo "E26 held-job audit failed for $job_id: missing $required" >&2
      exit 1
    fi
  done
done

protocol_sha256="$(sha256sum "$PROTOCOL" | cut -d' ' -f1)"
calibration_sha256="$(sha256sum "$CALIBRATION" | cut -d' ' -f1)"
"$PYTHON_BIN" - "$IDENTITY" "$protocol_sha256" "$calibration_sha256" \
  "$SOURCE_HASH" "$EXECUTION_HASH" "$MANIFEST" "$TARGET" "$ALPHA_BASE" "$ALPHA_MAX" "$ALPHA_LR" <<'PY'
import hashlib, json, os, pathlib, sys, tempfile
path = pathlib.Path(sys.argv[1])
manifest = pathlib.Path(sys.argv[6])
payload = {
    "schema": "e26_math_freeform_dual_high_entropy_identity_v2",
    "protocol_sha256": sys.argv[2], "calibration_sha256": sys.argv[3],
    "source_hash": sys.argv[4], "execution_surface_hash": sys.argv[5],
    "manifest": str(manifest.resolve()),
    "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
    "objective": "conditional_token_mean",
    "controller": "base_preserving_haarnoja_log_alpha_adam",
    "arms": ["maxent_dual"], "seeds": [43, 44, 45],
    "target_entropy": float(sys.argv[7]),
    "alpha": {"base": float(sys.argv[8]), "min": float(sys.argv[8]), "max": float(sys.argv[9])},
    "alpha_lr": float(sys.argv[10]),
}
path.parent.mkdir(parents=True, exist_ok=True)
fd, temporary = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
with os.fdopen(fd, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
os.replace(temporary, path)
PY

scontrol release "${job_ids[@]}"
echo "[e26] released high-entropy MATH dual cohort: ${job_ids[*]}"
echo "[e26] manifest=$MANIFEST identity=$IDENTITY"
