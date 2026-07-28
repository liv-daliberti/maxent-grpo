#!/usr/bin/env bash
# Freeze and atomically submit E65R1's 12-job prospective repair cohort.
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

PROTOCOL="$ROOT_DIR/paper/preregistration/e65_five_domain_terminal_confirmation_05b.md"
MODEL_ROOT="$ROOT_DIR/var/cache/huggingface/transformers/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
GRAPH_DATA="$ROOT_DIR/var/data/exact_answer_mode_probe"
COUNTDOWN_DATA="$ROOT_DIR/var/data/exact_countdown_easy3_probe"
PYTHON_DATA="$ROOT_DIR/var/data/python_factor_modebench_v1"
MATHIR_DATA="$ROOT_DIR/var/data/mathir_action_menu_v1"
IDENTITY="$ROOT_DIR/var/artifacts/e65r1_entropy_gated_singleton_confirmation_identity.json"
VARIANT=verified_entropy_gated_singleton_escape_canonical

GRAPH_PREFIX=gce65r1_entropy_gated_singleton_05b_12ep
COUNTDOWN_PREFIX=cde65r1_entropy_gated_singleton_05b_12ep
PYTHON_PREFIX=pye65r1_entropy_gated_singleton_05b_12ep
MATHIR_PREFIX=mie65r1_entropy_gated_singleton_05b_12ep
GRAPH_MANIFEST="$ROOT_DIR/var/artifacts/${GRAPH_PREFIX}_comparative_jobs.tsv"
COUNTDOWN_MANIFEST="$ROOT_DIR/var/artifacts/${COUNTDOWN_PREFIX}_comparative_jobs.tsv"
PYTHON_MANIFEST="$ROOT_DIR/var/artifacts/${PYTHON_PREFIX}_comparative_jobs.tsv"
MATHIR_MANIFEST="$ROOT_DIR/var/artifacts/${MATHIR_PREFIX}_comparative_jobs.tsv"

for required in \
  "$PROTOCOL" "$MODEL_ROOT/config.json" \
  "$GRAPH_DATA/train/dataset_dict.json" \
  "$COUNTDOWN_DATA/train/dataset_dict.json" \
  "$PYTHON_DATA/train/dataset_dict.json" \
  "$MATHIR_DATA/train/dataset_dict.json"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing E65 prerequisite: $required" >&2
    exit 1
  fi
done

if [[ "$phase" == full ]]; then
  for fresh in \
    "$IDENTITY" "$GRAPH_MANIFEST" "$COUNTDOWN_MANIFEST" \
    "$PYTHON_MANIFEST" "$MATHIR_MANIFEST"; do
    if [[ -e "$fresh" ]]; then
      echo "Fresh E65 artifact required; already exists: $fresh" >&2
      exit 1
    fi
  done
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
  SOURCE_ROOT="$ROOT_DIR/var/artifacts/source_snapshots/e65_entropy_gate_${SOURCE_HASH}/src"
  if [[ ! -f "$SOURCE_ROOT/oat_drgrpo/__init__.py" ]]; then
    mkdir -p "$(dirname "$SOURCE_ROOT")"
    staging="$(mktemp -d "$(dirname "$SOURCE_ROOT")/.source.XXXXXX")"
    mkdir -p "$staging/src"
    cp -a "$ROOT_DIR/src/." "$staging/src/"
    mv "$staging/src" "$SOURCE_ROOT"
    rmdir "$staging"
  fi
  if [[ "$(hash_tree "$SOURCE_ROOT")" != "$SOURCE_HASH" ]]; then
    echo "E65 source snapshot hash mismatch" >&2
    exit 1
  fi

  ops_input="$(mktemp -d "$ROOT_DIR/var/artifacts/source_snapshots/.e65-ops-input.XXXXXX")"
  mkdir -p "$ops_input/slurm"
  for file in \
    repo_env.sh run_experiment.sh train.sh resolve_eval_cadence.py \
    submit_countdown_comparative.sh; do
    cp "$ROOT_DIR/ops/$file" "$ops_input/$file"
  done
  cp "$ROOT_DIR/ops/slurm/train_node302.slurm" \
    "$ops_input/slurm/train_node302.slurm"
  EXECUTION_HASH="$(hash_tree "$ops_input")"
  OPS_ROOT="$ROOT_DIR/var/artifacts/source_snapshots/e65_entropy_gate_ops_${EXECUTION_HASH}/ops"
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
    echo "E65 execution snapshot hash mismatch" >&2
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
export OAT_ZERO_TRAIN_SEEDS=43,44,45
export OAT_ZERO_ONLY_ARMS="$VARIANT"
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
  VERIFIED_FIRST_SPLIT_CANONICAL VERIFIED_FIRST_GLOBAL_REPLAY_CANONICAL \
  VERIFIED_FIRST_BOOTSTRAP_LOCAL_CANONICAL VERIFIED_COUNTERFACTUAL_CANONICAL; do
  export "OAT_ZERO_INCLUDE_${flag}_ARM=0"
done
export OAT_ZERO_INCLUDE_VERIFIED_ENTROPY_GATED_SINGLETON_ESCAPE_CANONICAL_ARM=1

export OAT_ZERO_VERIFIED_DISCOVERY_TRACKING=1
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
export OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_MAX_ATTEMPTS=3
export OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_SAMPLING_TEMPERATURE=1.0

export OAT_ZERO_NUM_SAMPLES=16
export OAT_ZERO_LEARNING_RATE=0.0000002
export OAT_ZERO_MAX_QUERIES=100000000
export OAT_ZERO_MAX_PROMPT_EPOCHS=12
export OAT_ZERO_NUM_PROMPT_EPOCH=12
export OAT_ZERO_NUM_PPO_EPOCHS=1
export OAT_ZERO_MAX_NORM=1
export OAT_ZERO_BETA=0
export OAT_ZERO_TRAIN_BATCH_SIZE=16
export OAT_ZERO_TRAIN_BATCH_SIZE_PER_DEVICE=1
export OAT_ZERO_ROLLOUT_BATCH_SIZE=1
export OAT_ZERO_ROLLOUT_BATCH_SIZE_PER_DEVICE=1
export OAT_ZERO_N_GPU=1
export OAT_ZERO_NUM_GPUS_PER_ACTOR=1
export OAT_ZERO_REPLICATED_FREEFORM_SAMPLING=1
export OAT_ZERO_LOCAL_ACTOR_WEIGHT_SYNC=1
export OAT_ZERO_PROMPT_TEMPLATE=qwen_boxed
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
export OAT_ZERO_RESUME_FROM=0
export OAT_ZERO_MAX_RESUME_NUM=2
export OAT_ZERO_PRUNE_RESUME_ON_SUCCESS=0
export OAT_ZERO_AUTO_RESUME=1
export OAT_ZERO_WATCHDOG_REQUEUE=1
export OAT_ZERO_WATCHDOG_MAX_RESTARTS=6
export OAT_ZERO_TRAIN_CPUS_PER_TASK=8
export OAT_ZERO_TRAIN_MEMORY=64G
export OAT_ZERO_TRAIN_TIME_LIMIT=1-00:00:00

submit_domain() {
  local domain="$1"
  case "$domain" in
    graph_coloring)
      export RUN_STAMP_PREFIX="$GRAPH_PREFIX"
      export OAT_ZERO_COMPARATIVE_TASK=graph_coloring
      export OAT_ZERO_COMPARATIVE_DATA_ROOT="$GRAPH_DATA"
      export OAT_ZERO_MAX_TRAIN=192
      export OAT_ZERO_EVAL_PROMPT_INTERVAL=48
      export OAT_ZERO_SAVE_STEPS=192
      export OAT_ZERO_RESUME_STEPS=192
      export OAT_ZERO_GENERATE_MAX_LENGTH=192
      export OAT_ZERO_EVAL_GENERATE_MAX_LENGTH=192
      export OAT_ZERO_MAX_MODEL_LEN=512
      export OAT_ZERO_EVAL_MODE_COVERAGE_SEED=610100
      export OAT_ZERO_TRAIN_NODELIST=node105,node202,node203,node204
      export OAT_ZERO_TRAIN_GRES=gpu:a5000:1
      export OAT_ZERO_TRAIN_PARTITION=lowprio
      export OAT_ZERO_TRAIN_ACCOUNT=mltheory
      ;;
    countdown)
      export RUN_STAMP_PREFIX="$COUNTDOWN_PREFIX"
      export OAT_ZERO_COMPARATIVE_TASK=countdown
      export OAT_ZERO_COMPARATIVE_DATA_PRESET=easy3
      export OAT_ZERO_COMPARATIVE_DATA_ROOT="$COUNTDOWN_DATA"
      export OAT_ZERO_MAX_TRAIN=384
      export OAT_ZERO_EVAL_PROMPT_INTERVAL=96
      export OAT_ZERO_SAVE_STEPS=384
      export OAT_ZERO_RESUME_STEPS=384
      export OAT_ZERO_GENERATE_MAX_LENGTH=192
      export OAT_ZERO_EVAL_GENERATE_MAX_LENGTH=192
      export OAT_ZERO_MAX_MODEL_LEN=512
      export OAT_ZERO_EVAL_MODE_COVERAGE_SEED=610200
      export OAT_ZERO_TRAIN_NODELIST=node021,node022,node023,node024,node026
      export OAT_ZERO_TRAIN_GRES=gpu:rtx_3090:1
      export OAT_ZERO_TRAIN_PARTITION=lowprio
      export OAT_ZERO_TRAIN_ACCOUNT=allcs
      ;;
    python_factor)
      export RUN_STAMP_PREFIX="$PYTHON_PREFIX"
      export OAT_ZERO_COMPARATIVE_TASK=python_factor
      export OAT_ZERO_COMPARATIVE_DATA_ROOT="$PYTHON_DATA"
      export OAT_ZERO_MAX_TRAIN=384
      export OAT_ZERO_EVAL_PROMPT_INTERVAL=96
      export OAT_ZERO_SAVE_STEPS=384
      export OAT_ZERO_RESUME_STEPS=384
      export OAT_ZERO_GENERATE_MAX_LENGTH=192
      export OAT_ZERO_EVAL_GENERATE_MAX_LENGTH=192
      export OAT_ZERO_MAX_MODEL_LEN=512
      export OAT_ZERO_EVAL_MODE_COVERAGE_SEED=610300
      export OAT_ZERO_TRAIN_NODELIST=node021,node022,node023,node024,node026
      export OAT_ZERO_TRAIN_GRES=gpu:rtx_3090:1
      export OAT_ZERO_TRAIN_PARTITION=lowprio
      export OAT_ZERO_TRAIN_ACCOUNT=allcs
      ;;
    mathir)
      export RUN_STAMP_PREFIX="$MATHIR_PREFIX"
      export OAT_ZERO_COMPARATIVE_TASK=math
      export OAT_ZERO_COMPARATIVE_DATA_ROOT="$MATHIR_DATA"
      export OAT_ZERO_MAX_TRAIN=384
      export OAT_ZERO_EVAL_PROMPT_INTERVAL=96
      export OAT_ZERO_SAVE_STEPS=384
      export OAT_ZERO_RESUME_STEPS=384
      export OAT_ZERO_GENERATE_MAX_LENGTH=64
      export OAT_ZERO_EVAL_GENERATE_MAX_LENGTH=64
      export OAT_ZERO_MAX_MODEL_LEN=384
      export OAT_ZERO_EVAL_MODE_COVERAGE_SEED=610400
      export OAT_ZERO_TRAIN_NODELIST=node105,node202,node203,node204
      export OAT_ZERO_TRAIN_GRES=gpu:a5000:1
      export OAT_ZERO_TRAIN_PARTITION=lowprio
      export OAT_ZERO_TRAIN_ACCOUNT=mltheory
      ;;
  esac
  "$OPS_ROOT/submit_countdown_comparative.sh"
}

if [[ "$phase" == config ]]; then
  export OAT_ZERO_COMPARATIVE_CONFIG_ONLY=1
  export OAT_ZERO_SBATCH_HOLD=0
  for domain in graph_coloring countdown python_factor mathir; do
    submit_domain "$domain"
  done
  echo "[e65] all four configurations passed"
  exit 0
fi

export OAT_ZERO_PROTOCOL_IDENTITY="$IDENTITY"
export OAT_ZERO_SBATCH_HOLD=1
released=0
job_ids=()
cleanup_held() {
  local status="$?"
  trap - EXIT
  if [[ "$released" != "1" && "${#job_ids[@]}" -gt 0 ]]; then
    scancel "${job_ids[@]}" 2>/dev/null || true
  fi
  if [[ "$released" != "1" ]]; then
    for incomplete in \
      "$IDENTITY" "$GRAPH_MANIFEST" "$COUNTDOWN_MANIFEST" \
      "$PYTHON_MANIFEST" "$MATHIR_MANIFEST"; do
      if [[ -e "$incomplete" ]]; then
        rm -f "$incomplete"
      fi
    done
  fi
  exit "$status"
}
trap cleanup_held EXIT

for domain in graph_coloring countdown python_factor mathir; do
  submit_domain "$domain"
done

for manifest in \
  "$GRAPH_MANIFEST" "$COUNTDOWN_MANIFEST" \
  "$PYTHON_MANIFEST" "$MATHIR_MANIFEST"; do
  mapfile -t manifest_jobs < <(
    awk -F $'\t' 'NR > 1 && $3 ~ /^[0-9]+$/ {print $3}' "$manifest"
  )
  if [[ "${#manifest_jobs[@]}" -ne 3 ]]; then
    echo "E65 manifest $manifest has ${#manifest_jobs[@]} jobs; expected 3" >&2
    exit 1
  fi
  job_ids+=("${manifest_jobs[@]}")
done

SOURCE_TREE_HASH="$SOURCE_HASH"
EXECUTION_SURFACE_HASH="$EXECUTION_HASH"
export SOURCE_TREE_HASH EXECUTION_SURFACE_HASH
python - "$IDENTITY" "$PROTOCOL" "$0" \
  "$GRAPH_MANIFEST" "$COUNTDOWN_MANIFEST" "$PYTHON_MANIFEST" \
  "$MATHIR_MANIFEST" <<'PY'
import csv
import hashlib
import json
import os
import pathlib
import sys
import tempfile

def digest(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()

identity = pathlib.Path(sys.argv[1])
domains = ("graph_coloring", "countdown", "python_factor", "mathir")
jobs = {}
manifest_hashes = {}
for domain, raw in zip(domains, sys.argv[4:8]):
    path = pathlib.Path(raw)
    rows = list(csv.DictReader(path.open(), delimiter="\t"))
    jobs[domain] = [
        {
            "arm": row["arm"],
            "seed": int(row["seed"]),
            "job_id": int(row["job_id"]),
            "run_stamp": row["run_stamp"],
        }
        for row in rows
    ]
    manifest_hashes[domain] = digest(path)
payload = {
    "schema": "e65r1_entropy_gated_singleton_confirmation_v1",
    "protocol_sha256": digest(sys.argv[2]),
    "launcher_sha256": digest(sys.argv[3]),
    "source_hash": os.environ["SOURCE_TREE_HASH"],
    "execution_surface_hash": os.environ["EXECUTION_SURFACE_HASH"],
    "manifest_sha256": manifest_hashes,
    "jobs": jobs,
    "arm": "verified_entropy_gated_singleton_escape_canonical",
    "seeds": [43, 44, 45],
    "passes": 12,
    "paper_checkpoints": [0, 1, 2, 3, 4, 5, 6, 8, 10, 12],
    "information_firewall": {
        "gold_support": False,
        "desired_entropy": None,
        "desired_mode_count": None,
        "evaluation_feedback": False,
        "coefficient_projection": None,
    },
}
identity.parent.mkdir(parents=True, exist_ok=True)
fd, temporary = tempfile.mkstemp(prefix=f".{identity.name}.", dir=identity.parent)
with os.fdopen(fd, "w") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
os.replace(temporary, identity)
PY

if [[ "${#job_ids[@]}" -ne 12 ]]; then
  echo "E65 expected 12 jobs; got ${#job_ids[@]}" >&2
  exit 1
fi
for job_id in "${job_ids[@]}"; do
  record="$(scontrol show job "$job_id" -o)"
  for required in \
    'JobState=PENDING' 'Reason=JobHeldUser' \
    "OAT_ZERO_VARIANT=${VARIANT}" \
    'OAT_ZERO_MAX_PROMPT_EPOCHS=12' \
    'OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_PROPOSALS=1' \
    'OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_SINGLETON_ENTROPY_GATE=1' \
    'OAT_ZERO_REPLICATED_FREEFORM_SAMPLING=1' \
    'OAT_ZERO_LOCAL_ACTOR_WEIGHT_SYNC=1' \
    'OAT_ZERO_SEMANTIC_SHANNON_OPEN_SET_INVERSE_ADAPTATION=1' \
    'OAT_ZERO_ONLINE_CANONICAL_REPLAY_GLOBAL_BOOTSTRAP_STEPS=0' \
    "OAT_ZERO_SOURCE_ROOT=${SOURCE_ROOT}" \
    "OAT_ZERO_PROTOCOL_IDENTITY=${IDENTITY}"; do
    if [[ "$record" != *"$required"* ]]; then
      echo "E65 held-job audit failed for $job_id: missing $required" >&2
      exit 1
    fi
  done
done

scontrol release "${job_ids[@]}"
released=1
trap - EXIT
echo "[e65] released 12 jobs: ${job_ids[*]}"
echo "[e65] identity=$IDENTITY"
