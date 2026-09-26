#!/usr/bin/env bash
# Launch E21's length-neutral free-form MATH smoke or approved full cohort.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT_DIR/ops/repo_env.sh"

PYTHON_BIN="${OAT_ZERO_PYTHON:-$ROOT_DIR/var/seed_paper_eval/paper310/bin/python}"
PROTOCOL="$ROOT_DIR/paper/preregistration/e21_math_token_policy_maxent.md"
DATA_ROOT="$ROOT_DIR/var/data/oat_drgrpo_math_paper"
MODEL_ROOT="$ROOT_DIR/var/cache/huggingface/transformers/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
EXPECTED_SOURCE_HASH="217547637154ed74c2356eafec6dc885914793f8bb530c2b6918ca9a2c245448"
EXPECTED_EXECUTION_HASH="05d43e4ae78d75d9e98c5b3d07d6ebd88cc545cc9039774ad0b6bf7cfebc05ce"
SOURCE_SNAPSHOT="$ROOT_DIR/var/artifacts/source_snapshots/e21_math_conditional_token_${EXPECTED_SOURCE_HASH}/src"
OPS_SNAPSHOT="$ROOT_DIR/var/artifacts/source_snapshots/e21_math_conditional_token_ops_${EXPECTED_EXECUTION_HASH}/ops"
APPROVAL="$ROOT_DIR/var/artifacts/e21_math_conditional_token_smoke_v10_approval.json"
SMOKE_PREFIX="mte21_math_conditional_token_smoke_v10"
FULL_PREFIX="mte21_math_conditional_token_05b_v4"

phase="${1:-}"
case "$phase" in
  smoke-config|smoke|smoke-check|full-config|full) ;;
  *)
    echo "Usage: $0 {smoke-config|smoke|smoke-check|full-config|full}" >&2
    exit 1
    ;;
esac

for path in "$PYTHON_BIN" "$PROTOCOL" "$MODEL_ROOT/config.json" \
  "$MODEL_ROOT/tokenizer.json" "$MODEL_ROOT/model.safetensors" \
  "$DATA_ROOT/train/dataset_dict.json" "$DATA_ROOT/eval/dataset_dict.json" \
  "$SOURCE_SNAPSHOT/oat_drgrpo/__init__.py" \
  "$OPS_SNAPSHOT/submit_countdown_comparative.sh"; do
  if [[ ! -e "$path" ]]; then
    echo "Missing frozen E21 prerequisite: $path" >&2
    exit 1
  fi
done
if ! grep -q '^\*\*Status: FROZEN' "$PROTOCOL"; then
  echo "E21 protocol is not frozen" >&2
  exit 1
fi

source_hash="$(PYTHONPATH="$ROOT_DIR" "$PYTHON_BIN" -c '
import sys
from pathlib import Path
from ops.exp_scaling.check_e14_preflight import source_tree_hash
print(source_tree_hash(Path(sys.argv[1]), logical_repo_root=Path(sys.argv[2])))
' "$SOURCE_SNAPSHOT" "$ROOT_DIR")"
execution_hash="$($PYTHON_BIN - "$OPS_SNAPSHOT" <<'PY'
import hashlib
import pathlib
import sys
root = pathlib.Path(sys.argv[1])
files = (
    "repo_env.sh",
    "train.sh",
    "run_experiment.sh",
    "submit_countdown_comparative.sh",
    "resolve_eval_cadence.py",
    "math500/import_oat_math.py",
    "slurm/train_node302.slurm",
    "exp_scaling/check_e21_math_token_smoke.py",
)
h = hashlib.sha256()
for relative in files:
    payload = (root / relative).read_bytes()
    h.update(relative.encode() + b"\0" + hashlib.sha256(payload).digest())
print(h.hexdigest())
PY
)"
if [[ "$source_hash" != "$EXPECTED_SOURCE_HASH" ]]; then
  echo "Frozen E21 source hash mismatch: $source_hash" >&2
  exit 1
fi
if [[ "$execution_hash" != "$EXPECTED_EXECUTION_HASH" ]]; then
  echo "Frozen E21 execution hash mismatch: $execution_hash" >&2
  exit 1
fi

"$PYTHON_BIN" "$OPS_SNAPSHOT/math500/import_oat_math.py" \
  --output-root "$DATA_ROOT" --audit-only >/dev/null

protocol_sha256="$(sha256sum "$PROTOCOL" | cut -d' ' -f1)"
echo "[e21] phase=$phase objective=conditional_token_mean"
echo "[e21] eos_entropy_gradient=excluded state_visitation=detached response_aggregation=mean"
echo "[e21] source_hash=$source_hash"
echo "[e21] execution_surface_hash=$execution_hash"
echo "[e21] protocol_sha256=$protocol_sha256"
echo "[e21] data_root=$DATA_ROOT model_root=$MODEL_ROOT"

if [[ "$phase" == "smoke-check" ]]; then
  "$PYTHON_BIN" "$OPS_SNAPSHOT/exp_scaling/check_e21_math_token_smoke.py" \
    --prefix "$SMOKE_PREFIX" \
    --run-data-root "$ROOT_DIR/var/data" \
    --protocol "$PROTOCOL" \
    --source-hash "$source_hash" \
    --execution-surface-hash "$execution_hash" \
    --manifest "$ROOT_DIR/var/artifacts/${SMOKE_PREFIX}_comparative_jobs.tsv" \
    --out "$APPROVAL"
  echo "[e21] approval=$APPROVAL sha256=$(sha256sum "$APPROVAL" | cut -d' ' -f1)"
  exit 0
fi

stage=smoke
prefix="$SMOKE_PREFIX"
seeds=9008
max_train=64
eval_interval=16
save_ckpt=0
time_limit=04:00:00
auto_resume=0
watchdog_requeue=0
if [[ "$phase" == full || "$phase" == full-config ]]; then
  stage=full
  prefix="$FULL_PREFIX"
  seeds=43,44,45
  # max_train selects source rows before OAT's frozen prompt-length filter.
  # Select all 8,523 source rows; the audited filter admits exactly 8,515.
  max_train=8523
  eval_interval=2129
  save_ckpt=1
  time_limit=7-00:00:00
  auto_resume=1
  watchdog_requeue=1
  if [[ ! -f "$APPROVAL" ]]; then
    echo "E21 full cohort requires smoke approval: $APPROVAL" >&2
    exit 1
  fi
  "$PYTHON_BIN" - "$APPROVAL" "$source_hash" "$execution_hash" \
    "$protocol_sha256" <<'PY'
import json, pathlib, sys
payload = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
expected = {
    "approved": True,
    "source_hash": sys.argv[2],
    "execution_surface_hash": sys.argv[3],
    "protocol_sha256": sys.argv[4],
}
for key, value in expected.items():
    if payload.get(key) != value:
        raise SystemExit(f"E21 approval mismatch for {key}")
PY
fi

config_only=0
if [[ "$phase" == smoke-config || "$phase" == full-config ]]; then
  config_only=1
fi

export RUN_STAMP_PREFIX="$prefix"
export OAT_ZERO_REPO_ROOT="$ROOT_DIR"
export OAT_ZERO_OPS_SNAPSHOT_ROOT="$OPS_SNAPSHOT"
export OAT_ZERO_CAMPAIGN_SOURCE_ROOT="$SOURCE_SNAPSHOT"
export OAT_ZERO_PRETRAIN="$MODEL_ROOT"
export OAT_ZERO_COMPARATIVE_TASK=math
export OAT_ZERO_COMPARATIVE_MODEL=qwen2.5-0.5b-instruct
export OAT_ZERO_COMPARATIVE_DATA_ROOT="$DATA_ROOT"
export OAT_ZERO_REQUIRE_EXISTING_DATA=1
export OAT_ZERO_COMPARATIVE_REBUILD=0
export OAT_ZERO_COMPARATIVE_CONFIG_ONLY="$config_only"
export OAT_ZERO_TRAIN_SEEDS="$seeds"
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
export OAT_ZERO_MAXENT_OBJECTIVE=conditional_token_mean
export OAT_ZERO_MAXENT_FIXED_ALPHA=0.00010
export OAT_ZERO_MAXENT_CONTROL_BASE_ALPHA=0.000075
export OAT_ZERO_MAXENT_CONTROL_MAX_ALPHA=0.00015
export OAT_ZERO_MAXENT_CONTROL_RATIO=0.8
export OAT_ZERO_MAXENT_CONTROL_TARGET_ENTROPY=0
export OAT_ZERO_MAXENT_CONTROL_WARMUP_STEPS="$([[ "$stage" == smoke ]] && echo 16 || echo 64)"
export OAT_ZERO_MAXENT_CONTROL_EMA_DECAY=0.9
export OAT_ZERO_MAXENT_CONTROL_GAIN=2
export OAT_ZERO_MAXENT_DUAL_BASE_ALPHA=0.000075
export OAT_ZERO_MAXENT_DUAL_MIN_ALPHA=0.00005
export OAT_ZERO_MAXENT_DUAL_MAX_ALPHA=0.00015
export OAT_ZERO_MAXENT_DUAL_RATIO=0.8
export OAT_ZERO_MAXENT_DUAL_TARGET_ENTROPY=0
export OAT_ZERO_MAXENT_DUAL_WARMUP_STEPS="$([[ "$stage" == smoke ]] && echo 16 || echo 64)"
export OAT_ZERO_MAXENT_DUAL_ALPHA_LR=0.005
export OAT_ZERO_MAXENT_LENGTH_TARGET=0
export OAT_ZERO_POLICY_ENTROPY_COEF=0
export OAT_ZERO_NUM_SAMPLES=16
export OAT_ZERO_MAX_TRAIN="$max_train"
# max_queries counts generated candidates, not prompt placements. Keep it
# nonbinding; max_train plus one prompt epoch supplies the exact 64/8,515
# placement endpoints (64 in smoke; 8,515 admitted prompts in Stage A).
export OAT_ZERO_MAX_QUERIES=100000000
export OAT_ZERO_NUM_PROMPT_EPOCH=1
export OAT_ZERO_EVAL_PROMPT_INTERVAL="$eval_interval"
export OAT_ZERO_SAVE_STEPS="$eval_interval"
export OAT_ZERO_SAVE_FROM="$eval_interval"
export OAT_ZERO_SAVE_CKPT="$save_ckpt"
export OAT_ZERO_MAX_SAVE_NUM="$([[ "$stage" == smoke ]] && echo 5 || echo 2)"
export OAT_ZERO_AUTO_RESUME="$auto_resume"
export OAT_ZERO_WATCHDOG_REQUEUE="$watchdog_requeue"
export OAT_ZERO_WATCHDOG_MAX_RESTARTS=6
export OAT_ZERO_LEARNING_RATE=0.0000002
export OAT_ZERO_NUM_PPO_EPOCHS=1
export OAT_ZERO_BETA=0
export OAT_ZERO_MAX_NORM=1
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
export OAT_ZERO_EVAL_MODE_COVERAGE_K="$([[ "$stage" == full ]] && echo 8 || echo 0)"
export OAT_ZERO_EVAL_BATCH_SIZE=64
export OAT_ZERO_SYNC_PARAMS_EVERY=1
export OAT_ZERO_ZERO_STAGE=2
export OAT_ZERO_VLLM_GPU_RATIO=0.25
export OAT_ZERO_ENABLE_FLASH_ATTN=0
export OAT_ZERO_ADAM_OFFLOAD=0
export OAT_ZERO_ACTIVATION_OFFLOADING=0
export OAT_ZERO_COLLOCATE=1
export OAT_ZERO_CANONICAL_GRAPH_ACTIONS=0
export OAT_ZERO_CANONICAL_ACTION_TASK=none
export VLLM_USE_V1=0
export OAT_ZERO_VLLM_SLEEP=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OAT_ZERO_TRAIN_MEMORY=64G
export OAT_ZERO_TRAIN_TIME_LIMIT="$time_limit"
if [[ "$stage" == smoke ]]; then
  export OAT_ZERO_TRAIN_NODELIST=node205,node206,node207
  export OAT_ZERO_TRAIN_GRES=gpu:a6000:1
  export OAT_ZERO_TRAIN_PARTITION=cs
  export OAT_ZERO_TRAIN_ACCOUNT=allcs
else
  export OAT_ZERO_TRAIN_NODELIST=node105
  export OAT_ZERO_TRAIN_GRES=gpu:a5000:1
  export OAT_ZERO_TRAIN_PARTITION=mltheory
  export OAT_ZERO_TRAIN_ACCOUNT=mltheory
fi
export OAT_ZERO_SBATCH_HOLD="$([[ "$config_only" == 0 ]] && echo 1 || echo 0)"

if [[ "$config_only" == 1 ]]; then
  exec "$OPS_SNAPSHOT/submit_countdown_comparative.sh"
fi

manifest="$ROOT_DIR/var/artifacts/${prefix}_comparative_jobs.tsv"
if [[ -e "$manifest" ]]; then
  echo "Fresh E21 prefix required; manifest already exists: $manifest" >&2
  exit 1
fi

"$OPS_SNAPSHOT/submit_countdown_comparative.sh"
expected_jobs=4
if [[ "$stage" == full ]]; then
  expected_jobs=12
fi
mapfile -t job_ids < <(tail -n +2 "$manifest" | cut -f3)
if [[ "${#job_ids[@]}" -ne "$expected_jobs" ]]; then
  if [[ "${#job_ids[@]}" -gt 0 ]]; then
    scancel "${job_ids[@]}" || true
  fi
  echo "E21 held cohort has ${#job_ids[@]} jobs; expected $expected_jobs" >&2
  exit 1
fi
for job_id in "${job_ids[@]}"; do
  if [[ ! "$job_id" =~ ^[0-9]+$ ]]; then
    scancel "${job_ids[@]}" || true
    echo "Invalid E21 job id in manifest: $job_id" >&2
    exit 1
  fi
  job_record="$(scontrol show job "$job_id" -o)"
  expected_gpu_tres=gres/gpu:a6000=1
  if [[ "$stage" == full ]]; then
    expected_gpu_tres=gres/gpu:a5000=1
  fi
  if [[ "$job_record" != *"Partition=${OAT_ZERO_TRAIN_PARTITION}"* ]] || \
     [[ "$job_record" != *"ReqNodeList=${OAT_ZERO_TRAIN_NODELIST}"* ]] || \
     [[ "$job_record" != *"${expected_gpu_tres}"* ]]; then
    scancel "${job_ids[@]}" || true
    echo "E21 held job $job_id does not attest the frozen placement" >&2
    exit 1
  fi
done

identity="$ROOT_DIR/var/artifacts/${prefix}_e21_identity.json"
"$PYTHON_BIN" - "$identity" "$stage" "$prefix" "$source_hash" \
  "$execution_hash" "$protocol_sha256" "$manifest" "$APPROVAL" <<'PY'
import hashlib, json, os, pathlib, sys, tempfile
path = pathlib.Path(sys.argv[1])
manifest = pathlib.Path(sys.argv[7])
approval = pathlib.Path(sys.argv[8])
payload = {
    "schema": "e21_math_conditional_token_campaign_identity_v1",
    "stage": sys.argv[2],
    "prefix": sys.argv[3],
    "objective": "conditional_token_mean",
    "source_hash": sys.argv[4],
    "execution_surface_hash": sys.argv[5],
    "protocol_sha256": sys.argv[6],
    "manifest": str(manifest.resolve()),
    "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
    "smoke_approval": None if not approval.exists() else str(approval.resolve()),
    "smoke_approval_sha256": None if not approval.exists() else hashlib.sha256(approval.read_bytes()).hexdigest(),
}
path.parent.mkdir(parents=True, exist_ok=True)
fd, temporary = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
with os.fdopen(fd, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
os.replace(temporary, path)
PY

job_csv="$(IFS=,; echo "${job_ids[*]}")"
scontrol release "$job_csv"
echo "[e21] released stage=$stage jobs=$job_csv manifest=$manifest identity=$identity"
