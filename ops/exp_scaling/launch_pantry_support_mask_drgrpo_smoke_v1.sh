#!/usr/bin/env bash
# Configure or submit PantryPlan's one-seed six-bit Dr.GRPO plumbing smoke.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT_DIR/ops/repo_env.sh"
phase="${1:-}"
case "$phase" in
  config|run) ;;
  *) echo "Usage: $0 {config|run}" >&2; exit 1 ;;
esac

PYTHON_BIN="$ROOT_DIR/var/seed_paper_eval/paper310/bin/python"
MODEL_ROOT="$ROOT_DIR/var/cache/huggingface/transformers/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
DATA_ROOT="$ROOT_DIR/var/data/pantry_plan_modebench_v2"
PROTOCOL="$ROOT_DIR/paper/preregistration/pantry_support_mask_drgrpo_smoke_v1_20260729.md"
VIABILITY="$ROOT_DIR/var/artifacts/pantry_support_mask_dev_v1.json"
VIABILITY_IDENTITY="$ROOT_DIR/var/artifacts/pantry_support_mask_dev_v1_identity.json"
VIABILITY_AUDIT="$ROOT_DIR/var/artifacts/pantry_support_mask_dev_v1_audit.json"
PREFIX=ppsmoke_support_mask_drgrpo_v1
IDENTITY="$ROOT_DIR/var/artifacts/pantry_support_mask_drgrpo_smoke_v1_identity.json"
SUBMISSION="$ROOT_DIR/var/artifacts/pantry_support_mask_drgrpo_smoke_v1_submission.json"
MANIFEST="$ROOT_DIR/var/artifacts/${PREFIX}_comparative_jobs.tsv"

for required in "$PYTHON_BIN" "$MODEL_ROOT/config.json" "$DATA_ROOT/identity.json" \
  "$DATA_ROOT/train/dataset_dict.json" "$DATA_ROOT/eval/dataset_dict.json" \
  "$PROTOCOL" "$VIABILITY" "$VIABILITY_IDENTITY" "$VIABILITY_AUDIT"; do
  [[ -e "$required" ]] || { echo "Missing Pantry smoke prerequisite: $required" >&2; exit 1; }
done
"$PYTHON_BIN" - "$VIABILITY" "$VIABILITY_IDENTITY" "$VIABILITY_AUDIT" <<'PYCHECK'
import json, sys
receipt, identity, audit = [json.load(open(p, encoding="utf-8")) for p in sys.argv[1:]]
if receipt.get("status") != "pass" or receipt.get("decision") != "development_interface_signal_pass":
    raise SystemExit("Pantry smoke requires the passing six-bit viability receipt")
if identity.get("job_id") != receipt.get("job_id"):
    raise SystemExit("Pantry viability identity and receipt disagree")
if audit.get("status") != "pass" or not all(audit.get("checks", {}).values()):
    raise SystemExit("Pantry smoke requires a clean independent viability audit")
PYCHECK

hash_tree() {
  local tree="$1"
  (cd "$tree"; find . -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1)
}

configure() {
  export RUN_STAMP_PREFIX="$PREFIX"
  export OAT_ZERO_REPO_ROOT="$ROOT_DIR"
  export OAT_ZERO_PRETRAIN="$MODEL_ROOT"
  export OAT_ZERO_COMPARATIVE_TASK=pantry_plan
  export OAT_ZERO_COMPARATIVE_DATA_ROOT="$DATA_ROOT"
  export OAT_ZERO_COMPARATIVE_MODEL=qwen2.5-0.5b-instruct
  export OAT_ZERO_REQUIRE_EXISTING_DATA=1
  export OAT_ZERO_COMPARATIVE_REBUILD=0
  export OAT_ZERO_APPEND_MANIFEST=0
  export OAT_ZERO_TRAIN_SEEDS=76201
  export OAT_ZERO_ONLY_ARMS=grpo
  export OAT_ZERO_DRGRPO_VARIANT=grpo
  export OAT_ZERO_XDR_TAUS=""
  for flag in TOKEN_ENTROPY SEED XDR_ADAPT XDR_TAU_CONTROL XDR_SAC_DUAL \
    MAXENT MAXENT_CONTROL MAXENT_DUAL MAXENT_INVERSE \
    MAXENT_INVERSE_CANONICAL MAXENT_INVERSE_CANONICAL_REPLAY \
    OPEN_SET_SPLIT_CANONICAL VERIFIED_FIRST_SPLIT_CANONICAL \
    VERIFIED_FIRST_GLOBAL_REPLAY_CANONICAL \
    VERIFIED_FIRST_BOOTSTRAP_LOCAL_CANONICAL VERIFIED_COUNTERFACTUAL_CANONICAL \
    VERIFIED_ENTROPY_GATED_SINGLETON_ESCAPE_CANONICAL VERIFIED_ROUTE_SUCCESSOR \
    MAXENT_LENGTH_DUAL DIAYN OUTCOME_COLLISION \
    OUTCOME_COLLISION_OUTSIDE_CENTERING SEMANTIC_SHANNON \
    SEMANTIC_SHANNON_ADVANTAGE QUALITY_GATED_SEMANTIC_NOVELTY \
    SUCCESS_CONDITIONED_SIGNED_SEMANTIC_SHANNON SIGNAL_FIRST_SEMANTIC_BALANCE \
    ONLINE_CANONICAL_MAXENT ONLINE_CANONICAL_HAARNOJA \
    ONLINE_CANONICAL_POLICY_ENTROPY; do
    export "OAT_ZERO_INCLUDE_${flag}_ARM=0"
  done
  export OAT_ZERO_VERIFIED_DISCOVERY_TRACKING=1
  export OAT_ZERO_NUM_SAMPLES=16
  export OAT_ZERO_LEARNING_RATE=0.0000002
  export OAT_ZERO_MAX_TRAIN=32
  export OAT_ZERO_MAX_QUERIES=32
  export OAT_ZERO_MAX_PROMPT_EPOCHS=1
  export OAT_ZERO_NUM_PROMPT_EPOCH=1
  export OAT_ZERO_NUM_PPO_EPOCHS=1
  export OAT_ZERO_E16_TARGET_OPTIMIZER_UPDATES=32
  export OAT_ZERO_EVAL_PROMPT_INTERVAL=8
  export OAT_ZERO_ALLOW_SPARSE_EVAL=0
  export OAT_ZERO_SAVE_STEPS=32
  export OAT_ZERO_SAVE_FROM=32
  export OAT_ZERO_SAVE_CKPT=1
  export OAT_ZERO_MAX_SAVE_NUM=1
  export OAT_ZERO_MAX_SAVE_MEM=2000
  export OAT_ZERO_TRAIN_BATCH_SIZE=16
  export OAT_ZERO_TRAIN_BATCH_SIZE_PER_DEVICE=4
  export OAT_ZERO_ROLLOUT_BATCH_SIZE=1
  export OAT_ZERO_ROLLOUT_BATCH_SIZE_PER_DEVICE=1
  export OAT_ZERO_N_GPU=1
  export OAT_ZERO_NUM_GPUS_PER_ACTOR=1
  export OAT_ZERO_PROMPT_TEMPLATE=qwen_pantry_support_mask
  export OAT_ZERO_CANONICAL_GRAPH_ACTIONS=0
  export OAT_ZERO_CANONICAL_ACTION_TASK=pantry_support_mask
  export OAT_ZERO_CANONICAL_GRAPH_ACTION_COUNT=6
  export OAT_ZERO_CANONICAL_GRAPH_LEARNER_SAMPLING=1
  export OAT_ZERO_CANONICAL_GRAPH_FIXED_SHAPE_SAMPLING=1
  export OAT_ZERO_INPUT_KEY=problem
  export OAT_ZERO_OUTPUT_KEY=answer
  export OAT_ZERO_EVAL_INPUT_KEY=problem
  export OAT_ZERO_EVAL_OUTPUT_KEY=answer
  export OAT_ZERO_TEST_SPLIT=multi_answer
  export OAT_ZERO_VERIFIER_VERSION=fast
  export OAT_ZERO_PROMPT_MAX_LENGTH=640
  export OAT_ZERO_GENERATE_MAX_LENGTH=8
  export OAT_ZERO_EVAL_GENERATE_MAX_LENGTH=8
  export OAT_ZERO_MAX_MODEL_LEN=704
  export OAT_ZERO_TEMPERATURE=1
  export OAT_ZERO_TOP_P=1
  export OAT_ZERO_EVAL_TEMPERATURE=0
  export OAT_ZERO_EVAL_MODE_COVERAGE_K=8
  export OAT_ZERO_EVAL_MODE_COVERAGE_TEMPERATURE=1
  export OAT_ZERO_EVAL_MODE_COVERAGE_DRAWS=1
  export OAT_ZERO_EVAL_MODE_COVERAGE_SEED=76299
  export OAT_ZERO_EVAL_BATCH_SIZE=64
  export OAT_ZERO_MAX_NORM=1
  export OAT_ZERO_BETA=0
  export OAT_ZERO_IGNORE_NO_EOS=0
  export OAT_ZERO_SYNC_PARAMS_EVERY=1
  export OAT_ZERO_ZERO_STAGE=2
  export OAT_ZERO_VLLM_GPU_RATIO=0.25
  export OAT_ZERO_ENABLE_FLASH_ATTN=0
  export OAT_ZERO_ADAM_OFFLOAD=0
  export OAT_ZERO_ACTIVATION_OFFLOADING=0
  export OAT_ZERO_COLLOCATE=1
  export OAT_ZERO_VLLM_SLEEP=1
  export OAT_ZERO_VLLM_SLEEP_LEVEL=1
  export VLLM_USE_V1=0
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  export OAT_ZERO_AUTO_RESUME=0
  export OAT_ZERO_WATCHDOG_REQUEUE=0
  export OAT_ZERO_WATCHDOG_MAX_RESTARTS=0
  export OAT_ZERO_RESUME_STEPS=-1
  export OAT_ZERO_RESUME_FROM=0
  export OAT_ZERO_MAX_RESUME_NUM=1
  export OAT_ZERO_PRUNE_RESUME_ON_SUCCESS=1
  export OAT_ZERO_TRAIN_CPUS_PER_TASK=8
  export OAT_ZERO_TRAIN_MEMORY=64G
  export OAT_ZERO_TRAIN_TIME_LIMIT=04:00:00
  export OAT_ZERO_TRAIN_NODELIST=node302
  export OAT_ZERO_TRAIN_GRES=gpu:a100:1
  export OAT_ZERO_TRAIN_PARTITION=mltheory
  export OAT_ZERO_TRAIN_ACCOUNT=mltheory
}

configure
if [[ "$phase" == config ]]; then
  export OAT_ZERO_COMPARATIVE_CONFIG_ONLY=1
  export OAT_ZERO_SBATCH_HOLD=0
  "$ROOT_DIR/ops/submit_countdown_comparative.sh"
  echo "[pantry-drgrpo-smoke] configuration passed; no job submitted"
  exit 0
fi

for fresh in "$IDENTITY" "$SUBMISSION" "$MANIFEST"; do
  [[ ! -e "$fresh" ]] || { echo "Fresh Pantry smoke artifact required: $fresh" >&2; exit 1; }
done
SOURCE_HASH="$(hash_tree "$ROOT_DIR/src")"
SOURCE_PARENT="$ROOT_DIR/var/artifacts/source_snapshots/pantry_support_mask_drgrpo_smoke_v1_${SOURCE_HASH}"
SOURCE_ROOT="$SOURCE_PARENT/src"
if [[ ! -f "$SOURCE_ROOT/oat_drgrpo/__init__.py" ]]; then
  mkdir -p "$SOURCE_PARENT"
  staging="$(mktemp -d "$SOURCE_PARENT/.source.XXXXXX")"
  mkdir -p "$staging/src"
  cp -a "$ROOT_DIR/src/." "$staging/src/"
  mv "$staging/src" "$SOURCE_ROOT"
  rmdir "$staging"
fi
[[ "$(hash_tree "$SOURCE_ROOT")" == "$SOURCE_HASH" ]] || { echo "Pantry smoke source snapshot mismatch" >&2; exit 1; }

OPS_INPUT="$(mktemp -d "$ROOT_DIR/var/artifacts/source_snapshots/.pantry-smoke-ops.XXXXXX")"
mkdir -p "$OPS_INPUT/slurm"
for file in repo_env.sh run_experiment.sh train.sh resolve_eval_cadence.py submit_countdown_comparative.sh; do
  cp "$ROOT_DIR/ops/$file" "$OPS_INPUT/$file"
done
cp "$ROOT_DIR/ops/slurm/train_node302.slurm" "$OPS_INPUT/slurm/train_node302.slurm"
EXECUTION_HASH="$(hash_tree "$OPS_INPUT")"
OPS_ROOT="$ROOT_DIR/var/artifacts/source_snapshots/pantry_support_mask_drgrpo_smoke_v1_ops_${EXECUTION_HASH}"
if [[ ! -f "$OPS_ROOT/submit_countdown_comparative.sh" ]]; then
  mv "$OPS_INPUT" "$OPS_ROOT"
else
  find "$OPS_INPUT" -type f -delete
  rmdir "$OPS_INPUT/slurm" "$OPS_INPUT"
fi
[[ "$(hash_tree "$OPS_ROOT")" == "$EXECUTION_HASH" ]] || { echo "Pantry smoke ops snapshot mismatch" >&2; exit 1; }

DATA_HASH="$(hash_tree "$DATA_ROOT")"
"$PYTHON_BIN" - "$IDENTITY" "$PROTOCOL" "$0" "$SOURCE_HASH" "$EXECUTION_HASH" \
  "$DATA_HASH" "$MODEL_ROOT/config.json" "$VIABILITY" "$VIABILITY_IDENTITY" \
  "$VIABILITY_AUDIT" "$SOURCE_ROOT" "$OPS_ROOT" <<'PYID'
import hashlib, json, os, pathlib, sys, tempfile
def digest(path): return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
path=pathlib.Path(sys.argv[1])
payload={
  "schema":"pantry-support-mask-drgrpo-smoke-identity-v1",
  "protocol_sha256":digest(sys.argv[2]),
  "launcher_sha256":digest(sys.argv[3]),
  "source_hash":sys.argv[4],
  "execution_hash":sys.argv[5],
  "data_tree_sha256":sys.argv[6],
  "model_config_sha256":digest(sys.argv[7]),
  "viability_receipt_sha256":digest(sys.argv[8]),
  "viability_identity_sha256":digest(sys.argv[9]),
  "viability_audit_sha256":digest(sys.argv[10]),
  "source_root":str(pathlib.Path(sys.argv[11]).resolve()),
  "ops_root":str(pathlib.Path(sys.argv[12]).resolve()),
  "development_only":True,
  "arm":"grpo",
  "seed":76201,
  "optimizer_updates":32,
  "rollouts_per_prompt":16,
  "canonical_action_task":"pantry_support_mask",
  "horizon":6,
  "sequence_count":64,
  "maxent_actuators_enabled":False,
  "recovery_enabled":False,
}
path.parent.mkdir(parents=True,exist_ok=True)
fd,tmp=tempfile.mkstemp(prefix=f".{path.name}.",dir=path.parent)
with os.fdopen(fd,"w",encoding="utf-8") as h:
  json.dump(payload,h,indent=2,sort_keys=True); h.write("\n")
os.replace(tmp,path)
PYID

export OAT_ZERO_CAMPAIGN_SOURCE_ROOT="$SOURCE_ROOT"
export OAT_ZERO_OPS_SNAPSHOT_ROOT="$OPS_ROOT"
export OAT_ZERO_PROTOCOL_IDENTITY="$IDENTITY"
export OAT_ZERO_COMPARATIVE_CONFIG_ONLY=0
export OAT_ZERO_SBATCH_HOLD=1
"$OPS_ROOT/submit_countdown_comparative.sh"
mapfile -t jobs < <(awk -F $'\t' 'NR>1 {print $3}' "$MANIFEST")
if [[ "${#jobs[@]}" -ne 1 || ! "${jobs[0]}" =~ ^[0-9]+$ ]]; then
  echo "Pantry smoke must expand to exactly one job" >&2
  exit 1
fi
job_id="${jobs[0]}"
cleanup() { local status="$?"; trap - EXIT; scancel "$job_id" 2>/dev/null || true; exit "$status"; }
trap cleanup EXIT
job_record="$(scontrol show job "$job_id" -o)"
for required in 'JobState=PENDING' 'Reason=JobHeldUser' \
  'OAT_ZERO_VARIANT=grpo' 'OAT_ZERO_SEED=76201' \
  'OAT_ZERO_MAX_PROMPT_EPOCHS=1' 'OAT_ZERO_NUM_PROMPT_EPOCH=1' \
  'OAT_ZERO_NUM_SAMPLES=16' 'OAT_ZERO_MAX_TRAIN=32' \
  'OAT_ZERO_CANONICAL_ACTION_TASK=pantry_support_mask' \
  'OAT_ZERO_CANONICAL_GRAPH_ACTION_COUNT=6' \
  'OAT_ZERO_CANONICAL_GRAPH_LEARNER_SAMPLING=1' \
  'OAT_ZERO_CANONICAL_GRAPH_FIXED_SHAPE_SAMPLING=1' \
  'OAT_ZERO_MAXENT_ALPHA=0.0' 'OAT_ZERO_ONLINE_CANONICAL_BANK_ALPHA=0.0' \
  'OAT_ZERO_ONLINE_CANONICAL_REPLAY=0' \
  "OAT_ZERO_SOURCE_ROOT=${SOURCE_ROOT}" "OAT_ZERO_OPS_SNAPSHOT_ROOT=${OPS_ROOT}" \
  "OAT_ZERO_PROTOCOL_IDENTITY=${IDENTITY}" 'NumCPUs=8' 'MinMemoryNode=64G' 'TimeLimit=04:00:00'; do
  [[ "$job_record" == *"$required"* ]] || { echo "Pantry held-job audit missing: $required" >&2; exit 1; }
done
"$PYTHON_BIN" - "$SUBMISSION" "$IDENTITY" "$MANIFEST" "$job_id" <<'PYSUB'
import hashlib,json,os,pathlib,sys,tempfile
def digest(p): return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
path=pathlib.Path(sys.argv[1]); payload={
 "schema":"pantry-support-mask-drgrpo-smoke-submission-v1",
 "identity_sha256":digest(sys.argv[2]),"manifest_sha256":digest(sys.argv[3]),
 "job_id":int(sys.argv[4]),"held_job_audit":"pass","released":True}
fd,tmp=tempfile.mkstemp(prefix=f".{path.name}.",dir=path.parent)
with os.fdopen(fd,"w",encoding="utf-8") as h: json.dump(payload,h,indent=2,sort_keys=True); h.write("\n")
os.replace(tmp,path)
PYSUB
scontrol update "JobId=$job_id" Requeue=0
scontrol release "$job_id"
trap - EXIT
echo "[pantry-drgrpo-smoke] released job $job_id"
echo "[pantry-drgrpo-smoke] identity=$IDENTITY"
