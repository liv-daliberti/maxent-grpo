#!/usr/bin/env bash
# E73 response-budget amendments: relaunch one domain with a budget that is
# non-binding for Falcon, as the manuscript's budget already was for Qwen.
#
# Falcon3-1B is uniformly more verbose than Qwen2.5-0.5B, so response budgets
# chosen against Qwen's terseness truncate Falcon rollouts before they emit a
# parseable answer. Measured against the paired Qwen runs:
#
#   Amendment 1  python_factor  Falcon 136.4 tok, 44% at 192  (Qwen 9.8, 0%)
#   Amendment 2  mathir         Falcon  15.9 tok, 14% at  64  (Qwen 4.0, 0%)
#
# Graph coloring (6%) and Countdown (2%) are deliberately not amended: they are
# close enough to the non-binding regime that changing them would add deviation
# without removing a confound. PantryPlan is unaffected because its canonical
# fixed-shape sampler emits exactly six action tokens by construction.
#
# Each amendment reuses the cohort's existing frozen source and execution
# snapshots: only a decoding budget changes, never the code.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT_DIR/ops/repo_env.sh"

phase="${1:-}"
domain="${2:-}"
case "$phase" in
  config|full) ;;
  *)
    echo "Usage: $0 {config|full} {python_factor|mathir}" >&2
    exit 1
    ;;
esac
case "$domain" in
  python_factor|mathir) ;;
  *)
    echo "Usage: $0 {config|full} {python_factor|mathir}" >&2
    exit 1
    ;;
esac

PYTHON_BIN="${OAT_ZERO_PYTHON:-$ROOT_DIR/var/seed_paper_eval/paper310/bin/python}"
PROTOCOL="$ROOT_DIR/paper/preregistration/e73_falcon3_1b_cross_family_replication_20260731.md"
MODEL_ROOT="$ROOT_DIR/var/cache/huggingface/transformers/models--tiiuae--Falcon3-1B-Instruct/snapshots/28ba2251970a01dd1edc7ba7dad2eb71216ccfdf"
IDENTITY="$ROOT_DIR/var/artifacts/e73_falcon3_1b_cross_family_identity.json"

case "$domain" in
  python_factor)
    AMENDMENT_LABEL="Amendment 1"
    DOMAIN_DATA="$ROOT_DIR/var/data/python_factor_modebench_v1"
    IDENTITY_DOMAIN_KEY=python_factor
    COMPARATIVE_TASK=python_factor
    DOMAIN_PREFIX=pye73_falcon3_1b_12pass_r1
    AMENDMENT="$ROOT_DIR/var/artifacts/e73_falcon3_1b_python_budget_amendment.json"
    NEW_GENERATE_MAX_LENGTH=512
    NEW_MAX_MODEL_LEN=768
    SUPERSEDED_GENERATE_MAX_LENGTH=192
    COVERAGE_SEED=610300
    ;;
  mathir)
    # 128 response tokens still fit the existing 384-token window alongside the
    # longest 226-token MathIR prompt, so max_model_len is unchanged.
    AMENDMENT_LABEL="Amendment 2"
    DOMAIN_DATA="$ROOT_DIR/var/data/mathir_action_menu_v1"
    IDENTITY_DOMAIN_KEY=mathir
    COMPARATIVE_TASK=math
    DOMAIN_PREFIX=mie73_falcon3_1b_12pass_r1
    AMENDMENT="$ROOT_DIR/var/artifacts/e73_falcon3_1b_mathir_budget_amendment.json"
    NEW_GENERATE_MAX_LENGTH=128
    NEW_MAX_MODEL_LEN=384
    SUPERSEDED_GENERATE_MAX_LENGTH=64
    COVERAGE_SEED=610400
    ;;
esac

DOMAIN_MANIFEST="$ROOT_DIR/var/artifacts/${DOMAIN_PREFIX}_comparative_jobs.tsv"

for required in "$PROTOCOL" "$IDENTITY" "$MODEL_ROOT/config.json" \
  "$DOMAIN_DATA/train/dataset_dict.json" "$DOMAIN_DATA/eval/dataset_dict.json"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing E73 amendment prerequisite: $required" >&2
    exit 1
  fi
done

# The amendment must be recorded in the protocol before it is executed.
if ! grep -q "${AMENDMENT_LABEL} (2026-07-31" "$PROTOCOL"; then
  echo "E73 amendment requires the protocol to record ${AMENDMENT_LABEL}" >&2
  exit 1
fi

if [[ "$phase" == full && -e "$DOMAIN_MANIFEST" ]]; then
  echo "Fresh E73 amendment manifest required; already exists: $DOMAIN_MANIFEST" >&2
  exit 1
fi

# Inherit the cohort's frozen snapshots rather than refreezing: this amendment
# changes a decoding budget, so the executed code must be bit-identical to the
# code the other four domains are running.
read -r SOURCE_ROOT OPS_ROOT SOURCE_HASH EXECUTION_HASH SUPERSEDED_JOBS < <(
  "$PYTHON_BIN" - "$IDENTITY" "$ROOT_DIR" "$IDENTITY_DOMAIN_KEY" <<'PY'
import json
import pathlib
import sys

identity = json.loads(pathlib.Path(sys.argv[1]).read_text())
root = pathlib.Path(sys.argv[2])
source_hash = identity["source_hash"]
execution_hash = identity["execution_surface_hash"]
source_root = root / "var/artifacts/source_snapshots" / f"e73_falcon3_1b_{source_hash}/src"
ops_root = root / "var/artifacts/source_snapshots" / f"e73_falcon3_1b_ops_{execution_hash}/ops"
jobs = ",".join(str(job["job_id"]) for job in identity["jobs"][sys.argv[3]])
print(source_root, ops_root, source_hash, execution_hash, jobs)
PY
)

for required in "$SOURCE_ROOT/oat_drgrpo/__init__.py" \
  "$OPS_ROOT/submit_countdown_comparative.sh"; do
  if [[ ! -e "$required" ]]; then
    echo "E73 amendment cannot find the cohort's frozen snapshot: $required" >&2
    exit 1
  fi
done

IFS=',' read -r -a superseded <<< "$SUPERSEDED_JOBS"
if [[ "${#superseded[@]}" -ne 10 ]]; then
  echo "E73 amendment expected 10 superseded $domain jobs; got ${#superseded[@]}" >&2
  exit 1
fi

export OAT_ZERO_REPO_ROOT="$ROOT_DIR"
export OAT_ZERO_OPS_SNAPSHOT_ROOT="$OPS_ROOT"
export OAT_ZERO_CAMPAIGN_SOURCE_ROOT="$SOURCE_ROOT"
export OAT_ZERO_PRETRAIN="$MODEL_ROOT"
export OAT_ZERO_REQUIRE_EXISTING_DATA=1
export OAT_ZERO_COMPARATIVE_REBUILD=0
export OAT_ZERO_APPEND_MANIFEST=0
export OAT_ZERO_COMPARATIVE_MODEL=falcon3-1b-instruct
export OAT_ZERO_TRAIN_SEEDS=43,44,45,46,47
export OAT_ZERO_ONLY_ARMS=grpo,verified_first_global_replay_canonical
export OAT_ZERO_DRGRPO_VARIANT=grpo_compute_matched
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
  MAXENT_INVERSE MAXENT_INVERSE_CANONICAL \
  MAXENT_INVERSE_CANONICAL_REPLAY OPEN_SET_SPLIT_CANONICAL \
  VERIFIED_FIRST_SPLIT_CANONICAL VERIFIED_FIRST_BOOTSTRAP_LOCAL_CANONICAL; do
  export "OAT_ZERO_INCLUDE_${flag}_ARM=0"
done
export OAT_ZERO_INCLUDE_VERIFIED_FIRST_GLOBAL_REPLAY_CANONICAL_ARM=1

export OAT_ZERO_VERIFIED_DISCOVERY_TRACKING=1
export OAT_ZERO_MAXENT_ALPHA=0
export OAT_ZERO_MAXENT_INVERSE_BASE_ALPHA=0
export OAT_ZERO_SEMANTIC_SHANNON_COEF=0.10
export OAT_ZERO_SEMANTIC_SHANNON_SURPRISAL_CLIP=5.0
export OAT_ZERO_SEMANTIC_SHANNON_PSEUDOCOUNT=1.0
export OAT_ZERO_SEMANTIC_SHANNON_OPEN_SET_WARMUP_STEPS=64
export OAT_ZERO_SEMANTIC_SHANNON_OPEN_SET_EMA_DECAY=0.90
export OAT_ZERO_ONLINE_CANONICAL_BANK_ALPHA=0
export OAT_ZERO_ONLINE_CANONICAL_NOVELTY_BETA=0.50
export OAT_ZERO_ONLINE_CANONICAL_BANK_PSEUDOCOUNT=1.0
export OAT_ZERO_ONLINE_CANONICAL_BANK_SURPRISAL_CLIP=5.0
export OAT_ZERO_ONLINE_CANONICAL_KEY_MODE=modebench_outcome
export OAT_ZERO_ONLINE_CANONICAL_REPLAY_ALPHA=0.10
export OAT_ZERO_ONLINE_CANONICAL_REPLAY_CAPACITY=16
export OAT_ZERO_ONLINE_CANONICAL_REPLAY_GLOBAL_GROUPS_PER_STEP=1
export OAT_ZERO_ONLINE_CANONICAL_REPLAY_WARMUP_STEPS=64
export OAT_ZERO_ONLINE_CANONICAL_REPLAY_EMA_DECAY=0.90
export OAT_ZERO_ONLINE_CANONICAL_REPLAY_MASS_ALPHA=0.10
export OAT_ZERO_ONLINE_CANONICAL_REPLAY_MASS_WARMUP_STEPS=64
export OAT_ZERO_ONLINE_CANONICAL_REPLAY_MASS_EMA_DECAY=0.90

export OAT_ZERO_NUM_SAMPLES=16
export OAT_ZERO_LEARNING_RATE=0.0000002
export OAT_ZERO_MAX_QUERIES=100000000
export OAT_ZERO_MAX_PROMPT_EPOCHS=12
export OAT_ZERO_NUM_PROMPT_EPOCH=12
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
export OAT_ZERO_PROMPT_TEMPLATE=falcon_boxed
export OAT_ZERO_INPUT_KEY=problem
export OAT_ZERO_OUTPUT_KEY=answer
export OAT_ZERO_EVAL_INPUT_KEY=problem
export OAT_ZERO_EVAL_OUTPUT_KEY=answer
export OAT_ZERO_TEST_SPLIT=multi_answer
export OAT_ZERO_VERIFIER_VERSION=fast
export OAT_ZERO_PROMPT_MAX_LENGTH=256
export OAT_ZERO_TEMPERATURE=1
export OAT_ZERO_TOP_P=1
export OAT_ZERO_EVAL_TEMPERATURE=0
export OAT_ZERO_EVAL_MODE_COVERAGE_K=8
export OAT_ZERO_EVAL_MODE_COVERAGE_TEMPERATURE=1
export OAT_ZERO_EVAL_MODE_COVERAGE_DRAWS=4
export OAT_ZERO_EVAL_BATCH_SIZE=64
export OAT_ZERO_SYNC_PARAMS_EVERY=1
export OAT_ZERO_CANONICAL_ACTION_TASK=none
export OAT_ZERO_CANONICAL_GRAPH_ACTIONS=0
export OAT_ZERO_ZERO_STAGE=2
export OAT_ZERO_VLLM_GPU_RATIO=0.25
export OAT_ZERO_ENABLE_FLASH_ATTN=0
export OAT_ZERO_ADAM_OFFLOAD=1
export OAT_ZERO_ACTIVATION_OFFLOADING=0
export OAT_ZERO_COLLOCATE=1
export OAT_ZERO_VLLM_SLEEP=1
export VLLM_USE_V1=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OAT_ZERO_SAVE_CKPT=1
export OAT_ZERO_MAX_SAVE_NUM=2
export OAT_ZERO_RESUME_FROM=0
export OAT_ZERO_MAX_RESUME_NUM=2
export OAT_ZERO_PRUNE_RESUME_ON_SUCCESS=0
export OAT_ZERO_AUTO_RESUME=1
export OAT_ZERO_WATCHDOG_REQUEUE=1
export OAT_ZERO_WATCHDOG_MAX_RESTARTS=6
export OAT_ZERO_TRAIN_CPUS_PER_TASK=8
export OAT_ZERO_TRAIN_MEMORY=64G
export OAT_ZERO_TRAIN_TIME_LIMIT=3-00:00:00

export RUN_STAMP_PREFIX="$DOMAIN_PREFIX"
export OAT_ZERO_COMPARATIVE_TASK="$COMPARATIVE_TASK"
export OAT_ZERO_COMPARATIVE_DATA_ROOT="$DOMAIN_DATA"
export OAT_ZERO_MAX_TRAIN=384
export OAT_ZERO_EVAL_PROMPT_INTERVAL=96
export OAT_ZERO_SAVE_STEPS=384
export OAT_ZERO_SAVE_FROM=384
export OAT_ZERO_RESUME_STEPS=384
export OAT_ZERO_GENERATE_MAX_LENGTH="$NEW_GENERATE_MAX_LENGTH"
export OAT_ZERO_EVAL_GENERATE_MAX_LENGTH="$NEW_GENERATE_MAX_LENGTH"
export OAT_ZERO_MAX_MODEL_LEN="$NEW_MAX_MODEL_LEN"
export OAT_ZERO_EVAL_MODE_COVERAGE_SEED="$COVERAGE_SEED"
export OAT_ZERO_TRAIN_NODELIST=node202,node203,node204,node205,node206,node207
export OAT_ZERO_TRAIN_GRES=gpu:a5000:1
export OAT_ZERO_TRAIN_PARTITION=cs
export OAT_ZERO_TRAIN_ACCOUNT=allcs

if [[ "$phase" == config ]]; then
  export OAT_ZERO_COMPARATIVE_CONFIG_ONLY=1
  export OAT_ZERO_SBATCH_HOLD=0
  "$OPS_ROOT/submit_countdown_comparative.sh"
  echo "[e73-amend] $domain ${NEW_GENERATE_MAX_LENGTH}-token configuration passed"
  exit 0
fi

# Cancel the superseded runs before submitting, so the two budgets can never be
# live in the same domain at once.
echo "[e73-amend] $domain: cancelling superseded ${SUPERSEDED_GENERATE_MAX_LENGTH}-token jobs: ${superseded[*]}"
scancel "${superseded[@]}" 2>/dev/null || true

export OAT_ZERO_PROTOCOL_IDENTITY="$IDENTITY"
export OAT_ZERO_SBATCH_HOLD=1
released=0
job_ids=()
cleanup_held() {
  local status="$?"
  trap - EXIT
  if [[ "$released" != "1" && "${#job_ids[@]}" -gt 0 ]]; then
    scancel "${job_ids[@]}" 2>/dev/null || true
    echo "[e73-amend] cancelled incomplete held cohort: ${job_ids[*]}" >&2
  fi
  exit "$status"
}
trap cleanup_held EXIT

"$OPS_ROOT/submit_countdown_comparative.sh"

mapfile -t job_ids < <(
  awk -F $'\t' 'NR > 1 && $3 ~ /^[0-9]+$/ {print $3}' "$DOMAIN_MANIFEST"
)
if [[ "${#job_ids[@]}" -ne 10 ]]; then
  echo "E73 amendment produced ${#job_ids[@]} jobs; expected 10" >&2
  exit 1
fi

for job_id in "${job_ids[@]}"; do
  job_record="$(scontrol show job "$job_id" -o)"
  for required in \
    'JobState=PENDING' \
    'Reason=JobHeldUser' \
    'Partition=cs' \
    'Account=allcs' \
    "OAT_ZERO_GENERATE_MAX_LENGTH=${NEW_GENERATE_MAX_LENGTH}" \
    "OAT_ZERO_EVAL_GENERATE_MAX_LENGTH=${NEW_GENERATE_MAX_LENGTH}" \
    "OAT_ZERO_MAX_MODEL_LEN=${NEW_MAX_MODEL_LEN}" \
    'OAT_ZERO_PROMPT_TEMPLATE=falcon_boxed' \
    'OAT_ZERO_MAX_PROMPT_EPOCHS=12' \
    'OAT_ZERO_NUM_SAMPLES=16' \
    'OAT_ZERO_ADAM_OFFLOAD=1' \
    "OAT_ZERO_SOURCE_ROOT=${SOURCE_ROOT}" \
    "OAT_ZERO_OPS_SNAPSHOT_ROOT=${OPS_ROOT}"; do
    if [[ "$job_record" != *"$required"* ]]; then
      echo "E73 amendment audit failed for $job_id: missing $required" >&2
      exit 1
    fi
  done
  if [[ "$job_record" == *"OAT_ZERO_GENERATE_MAX_LENGTH=${SUPERSEDED_GENERATE_MAX_LENGTH}"* ]]; then
    echo "E73 amendment job $job_id still carries the superseded budget" >&2
    exit 1
  fi
done

"$PYTHON_BIN" - \
  "$AMENDMENT" "$PROTOCOL" "$IDENTITY" "$DOMAIN_MANIFEST" \
  "$SOURCE_HASH" "$EXECUTION_HASH" "$SUPERSEDED_JOBS" \
  "$NEW_GENERATE_MAX_LENGTH" "$NEW_MAX_MODEL_LEN" \
  "$SUPERSEDED_GENERATE_MAX_LENGTH" <<'PY'
import csv
import hashlib
import json
import os
import pathlib
import sys
import tempfile

def digest(raw):
    return hashlib.sha256(pathlib.Path(raw).read_bytes()).hexdigest()

path = pathlib.Path(sys.argv[1])
manifest = pathlib.Path(sys.argv[4])
rows = list(csv.DictReader(manifest.open(), delimiter="\t"))
if len(rows) != 10:
    raise SystemExit(f"{manifest} has {len(rows)} jobs; expected 10")

payload = {
    "schema": "e73_falcon3_1b_python_budget_amendment_v1",
    "amends": "e73_falcon3_1b_cross_family_v1",
    "domain": "python_factor",
    "reason": (
        "Falcon3-1B answered Python in ~136 tokens against Qwen2.5-0.5B's ~10, "
        "so the 192-token response budget truncated 44% of Falcon rollouts "
        "while never binding the Qwen cohort (20 of 4609 responses)."
    ),
    "measured_before_amendment": {
        "falcon_mean_response_tokens": 136.4,
        "falcon_fraction_at_cap": 0.44,
        "qwen_mean_response_tokens": 9.8,
        "qwen_fraction_at_cap": 0.0,
        "optimizer_step_when_measured": 30,
        "optimizer_steps_per_run": 4608,
        "arm_ordering_known": False,
    },
    "superseded_generate_max_length": int(sys.argv[10]),
    "generate_max_length": int(sys.argv[8]),
    "eval_generate_max_length": int(sys.argv[8]),
    "max_model_len": int(sys.argv[9]),
    "applies_to_both_arms": True,
    "domains_left_unchanged": [
        "graph_coloring",
        "countdown",
        "mathir",
        "pantry_plan",
    ],
    "protocol_sha256": digest(sys.argv[2]),
    "cohort_identity_sha256": digest(sys.argv[3]),
    "manifest_sha256": digest(manifest),
    "source_hash": sys.argv[5],
    "execution_surface_hash": sys.argv[6],
    "code_refrozen": False,
    "superseded_job_ids": [int(v) for v in sys.argv[7].split(",")],
    "jobs": [
        {
            "arm": row["arm"],
            "seed": int(row["seed"]),
            "job_id": int(row["job_id"]),
            "run_stamp": row["run_stamp"],
        }
        for row in rows
    ],
}
path.parent.mkdir(parents=True, exist_ok=True)
fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
with os.fdopen(fd, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
os.replace(temporary, path)
PY

scontrol release "${job_ids[@]}"
released=1
trap - EXIT
echo "[e73-amend] released 10 $domain jobs at ${NEW_GENERATE_MAX_LENGTH} tokens: ${job_ids[*]}"
echo "[e73-amend] amendment=$AMENDMENT"
