#!/usr/bin/env bash
# Launch the clean, non-preemptible replacement for E32 Countdown.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT_DIR/ops/repo_env.sh"

phase="${1:-}"
case "$phase" in
  config|full) ;;
  *) echo "Usage: $0 {config|full}" >&2; exit 2 ;;
esac

PYTHON_BIN="${OAT_ZERO_PYTHON:-$ROOT_DIR/var/seed_paper_eval/paper310/bin/python}"
PROTOCOL="$ROOT_DIR/paper/preregistration/e32_countdown_preemption_replacement_20260722.md"
MODEL_ROOT="$ROOT_DIR/var/cache/huggingface/transformers/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
DATA_ROOT="$ROOT_DIR/var/data/exact_countdown_easy3_probe"
SOURCE_ROOT="$ROOT_DIR/var/artifacts/source_snapshots/e32_freeform_05b_69a22e21276aa04bea617e4539afdd57a2056ef97b5b08fd925c710c4f561c0a/src"
PREFIX=cde32_freeform_05b_ema_10ep_v4_preemptsafe
IDENTITY="$ROOT_DIR/var/artifacts/${PREFIX}_identity.json"
TARGET=1.347109432487438
MANIFEST="$ROOT_DIR/var/artifacts/${PREFIX}_comparative_jobs.tsv"

for required in "$PYTHON_BIN" "$PROTOCOL" "$MODEL_ROOT/config.json" \
  "$MODEL_ROOT/tokenizer.json" "$MODEL_ROOT/model.safetensors" \
  "$SOURCE_ROOT/oat_drgrpo/__init__.py" "$DATA_ROOT/train/dataset_dict.json" \
  "$DATA_ROOT/eval/dataset_dict.json"; do
  [[ -e "$required" ]] || { echo "Missing replacement prerequisite: $required" >&2; exit 1; }
done
grep -q '^\*\*Status: FROZEN' "$PROTOCOL" || { echo "Protocol is not frozen" >&2; exit 1; }
if [[ "$phase" == full && -e "$MANIFEST" ]]; then
  echo "Fresh replacement prefix required; manifest exists: $MANIFEST" >&2
  exit 1
fi

hash_tree() {
  (cd "$1" && find . -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1)
}

freeze_ops() {
  local input staging
  input="$(mktemp -d "$ROOT_DIR/var/artifacts/source_snapshots/.e32-cd-ops-input.XXXXXX")"
  mkdir -p "$input/slurm"
  cp "$ROOT_DIR/ops/repo_env.sh" "$input/repo_env.sh"
  cp "$ROOT_DIR/ops/run_experiment.sh" "$input/run_experiment.sh"
  cp "$ROOT_DIR/ops/train.sh" "$input/train.sh"
  cp "$ROOT_DIR/ops/resolve_eval_cadence.py" "$input/resolve_eval_cadence.py"
  cp "$ROOT_DIR/ops/submit_countdown_comparative.sh" "$input/submit_countdown_comparative.sh"
  cp "$ROOT_DIR/ops/slurm/train_node302.slurm" "$input/slurm/train_node302.slurm"
  EXECUTION_HASH="$(hash_tree "$input")"
  OPS_ROOT="$ROOT_DIR/var/artifacts/source_snapshots/e32_countdown_preemptsafe_ops_${EXECUTION_HASH}/ops"
  if [[ ! -f "$OPS_ROOT/submit_countdown_comparative.sh" ]]; then
    mkdir -p "$(dirname "$OPS_ROOT")"
    staging="$(mktemp -d "$(dirname "$OPS_ROOT")/.ops.XXXXXX")"
    mv "$input" "$staging/ops"
    mv "$staging/ops" "$OPS_ROOT"
    rmdir "$staging"
  else
    find "$input" -type f -delete
    rmdir "$input/slurm" "$input"
  fi
  [[ "$(hash_tree "$OPS_ROOT")" == "$EXECUTION_HASH" ]] || { echo "Ops snapshot hash mismatch" >&2; exit 1; }
}

if [[ "$phase" == full ]]; then
  freeze_ops
else
  OPS_ROOT="$ROOT_DIR/ops"
  EXECUTION_HASH=config-only-current-tree
fi

export OAT_ZERO_REPO_ROOT="$ROOT_DIR"
export OAT_ZERO_OPS_SNAPSHOT_ROOT="$OPS_ROOT"
export OAT_ZERO_CAMPAIGN_SOURCE_ROOT="$SOURCE_ROOT"
export OAT_ZERO_PRETRAIN="$MODEL_ROOT"
export OAT_ZERO_REQUIRE_EXISTING_DATA=1
export OAT_ZERO_COMPARATIVE_REBUILD=0
export OAT_ZERO_APPEND_MANIFEST=0
export OAT_ZERO_COMPARATIVE_MODEL=qwen2.5-0.5b-instruct
export OAT_ZERO_COMPARATIVE_TASK=countdown
export OAT_ZERO_COMPARATIVE_DATA_PRESET=easy3
export OAT_ZERO_COMPARATIVE_DATA_ROOT="$DATA_ROOT"
export RUN_STAMP_PREFIX="$PREFIX"
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
export OAT_ZERO_MAXENT_DUAL_TARGET_ENTROPY="$TARGET"
export OAT_ZERO_MAXENT_DUAL_WARMUP_STEPS=64
export OAT_ZERO_MAXENT_DUAL_MIN_ALPHA=0.000075
export OAT_ZERO_MAXENT_DUAL_MAX_ALPHA=0.00060
export OAT_ZERO_MAXENT_DUAL_ALPHA_LR=0.010
export OAT_ZERO_MAXENT_DUAL_EMA_DECAY=0.7
export OAT_ZERO_MAXENT_LENGTH_TARGET=0
export OAT_ZERO_POLICY_ENTROPY_COEF=0

export OAT_ZERO_NUM_SAMPLES=16
export OAT_ZERO_LEARNING_RATE=0.0000002
export OAT_ZERO_MAX_TRAIN=384
export OAT_ZERO_MAX_QUERIES=100000000
export OAT_ZERO_NUM_PROMPT_EPOCH=10
export OAT_ZERO_MAX_PROMPT_EPOCHS=10
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
export OAT_ZERO_EVAL_PROMPT_INTERVAL=96
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
export OAT_ZERO_SAVE_STEPS=96
export OAT_ZERO_SAVE_FROM=96
export OAT_ZERO_MAX_SAVE_NUM=2
export OAT_ZERO_RESUME_STEPS=96
export OAT_ZERO_RESUME_FROM=96
export OAT_ZERO_MAX_RESUME_NUM=2
export OAT_ZERO_PRUNE_RESUME_ON_SUCCESS=0
export OAT_ZERO_EXPORT_STEPS=0
export OAT_ZERO_MAX_EXPORT_NUM=1
export OAT_ZERO_AUTO_RESUME=1
export OAT_ZERO_WATCHDOG_REQUEUE=1
export OAT_ZERO_WATCHDOG_MAX_RESTARTS=6
export OAT_ZERO_TRAIN_NODELIST=node203,node204
export OAT_ZERO_TRAIN_GRES=gpu:a5000:1
export OAT_ZERO_TRAIN_CPUS_PER_TASK=4
export OAT_ZERO_TRAIN_MEMORY=32G
export OAT_ZERO_TRAIN_TIME_LIMIT=24:00:00
export OAT_ZERO_TRAIN_PARTITION=cs
export OAT_ZERO_TRAIN_ACCOUNT=allcs

if [[ "$phase" == config ]]; then
  export OAT_ZERO_COMPARATIVE_CONFIG_ONLY=1
  export OAT_ZERO_SBATCH_HOLD=0
  "$OPS_ROOT/submit_countdown_comparative.sh"
  echo "[e32-cd-replacement] configuration passed"
  exit 0
fi

"$PYTHON_BIN" - "$IDENTITY" "$PROTOCOL" "$SOURCE_ROOT" "$EXECUTION_HASH" <<'PY'
import hashlib, json, os, pathlib, sys, tempfile
path = pathlib.Path(sys.argv[1])
protocol = pathlib.Path(sys.argv[2])
payload = {
    "schema": "e32_countdown_preemption_replacement_v1",
    "protocol_sha256": hashlib.sha256(protocol.read_bytes()).hexdigest(),
    "source_root": sys.argv[3],
    "execution_surface_hash": sys.argv[4],
    "model": "Qwen2.5-0.5B-Instruct@7ae557604adf67be50417f59c2c2f167def9a775",
    "task": "countdown", "arms": ["grpo", "maxent_dual"], "seeds": [43, 44, 45],
    "prompt_epochs": 10, "checkpoint_interval_prompts": 96,
    "evaluation": {"k": 8, "draws": 4, "seeds": [1001, 1002, 1003, 1004]},
    "controller": {"ema_decay": 0.7, "alpha_lr": 0.010, "target_entropy": 1.347109432487438},
    "placement": {"partition": "cs", "account": "allcs", "nodes": ["node203", "node204"], "preemptible": False},
    "supersedes": "cde32_freeform_05b_ema_10ep_v2",
}
path.parent.mkdir(parents=True, exist_ok=True)
fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
with os.fdopen(fd, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True); handle.write("\n")
os.replace(temporary, path)
PY

export OAT_ZERO_PROTOCOL_IDENTITY="$IDENTITY"
export OAT_ZERO_SBATCH_HOLD=1
"$OPS_ROOT/submit_countdown_comparative.sh"
mapfile -t job_ids < <(awk -F '\t' 'NR > 1 && $3 ~ /^[0-9]+$/ {print $3}' "$MANIFEST")
if [[ "${#job_ids[@]}" -ne 6 ]]; then
  scancel "${job_ids[@]}" 2>/dev/null || true
  echo "Expected six held jobs, found ${#job_ids[@]}" >&2
  exit 1
fi
for job_id in "${job_ids[@]}"; do
  arm="$(awk -F '\t' -v id="$job_id" '$3 == id {print $1}' "$MANIFEST")"
  record="$(scontrol show job -o "$job_id")"
  for required in 'JobState=PENDING' 'Reason=JobHeldUser' 'Partition=cs' 'Account=allcs' \
    'gres/gpu:a5000=1' 'NumCPUs=4' 'MinMemoryNode=32G' \
    'OAT_ZERO_NUM_PROMPT_EPOCH=10' 'OAT_ZERO_MAX_PROMPT_EPOCHS=10' \
    'OAT_ZERO_EVAL_MODE_COVERAGE_K=8' 'OAT_ZERO_EVAL_MODE_COVERAGE_DRAWS=4' \
    'OAT_ZERO_EVAL_MODE_COVERAGE_SEED=1001' 'OAT_ZERO_EVAL_PROMPT_INTERVAL=96' \
    'OAT_ZERO_SAVE_STEPS=96' 'OAT_ZERO_SAVE_FROM=96' 'OAT_ZERO_RESUME_STEPS=96' \
    'OAT_ZERO_RESUME_FROM=96' 'OAT_ZERO_MAX_RESUME_NUM=2' \
    'OAT_ZERO_PRUNE_RESUME_ON_SUCCESS=0' "OAT_ZERO_SOURCE_ROOT=${SOURCE_ROOT}" \
    "OAT_ZERO_OPS_SNAPSHOT_ROOT=${OPS_ROOT}" "OAT_ZERO_PROTOCOL_IDENTITY=${IDENTITY}"; do
    [[ "$record" == *"$required"* ]] || { scancel "${job_ids[@]}"; echo "Audit failed for $job_id: $required" >&2; exit 1; }
  done
  if [[ "$record" != *'ReqNodeList=node[203-204]'* && "$record" != *'ReqNodeList=node203,node204'* ]]; then
    scancel "${job_ids[@]}"
    echo "Audit failed for $job_id: unexpected requested node list" >&2
    exit 1
  fi
  if [[ "$arm" == maxent_dual ]]; then
    [[ "$record" == *'OAT_ZERO_MAXENT_DUAL_EMA_DECAY=0.7'* ]] || { scancel "${job_ids[@]}"; exit 1; }
    [[ "$record" == *"OAT_ZERO_MAXENT_DUAL_TARGET_ENTROPY=${TARGET}"* ]] || { scancel "${job_ids[@]}"; exit 1; }
  elif [[ "$arm" != grpo ]]; then
    scancel "${job_ids[@]}"; echo "Unexpected arm: $arm" >&2; exit 1
  fi
done
scontrol release "${job_ids[@]}"
echo "[e32-cd-replacement] released six clean jobs: ${job_ids[*]}"
