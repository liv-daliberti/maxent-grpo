#!/usr/bin/env bash
# Configure or atomically submit E71's 20-job matched cohort: Graph coloring and
# PantryPlan re-run on the common 384-train / 128-eval design.
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
PROTOCOL="$ROOT_DIR/paper/preregistration/e71_scale384_graph_pantry_05b_20260730.md"
MODEL_ROOT="$ROOT_DIR/var/cache/huggingface/transformers/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
GRAPH_DATA="$ROOT_DIR/var/data/graph_coloring_modebench_v2"
PANTRY_DATA="$ROOT_DIR/var/data/pantry_plan_modebench_v2"
E58_SMOKE="$ROOT_DIR/var/artifacts/e58_global_replay_smoke_audit_latest.json"
PANTRY_QUALIFICATION="$ROOT_DIR/var/artifacts/pantry_support_mask_paired_integration_v3_audit.json"
IDENTITY="$ROOT_DIR/var/artifacts/e71_scale384_05b_identity.json"

GRAPH_PREFIX=gce71_scale384_05b_12pass
PANTRY_PREFIX=ppe71_scale384_05b_12pass
GRAPH_MANIFEST="$ROOT_DIR/var/artifacts/${GRAPH_PREFIX}_comparative_jobs.tsv"
PANTRY_MANIFEST="$ROOT_DIR/var/artifacts/${PANTRY_PREFIX}_comparative_jobs.tsv"

for required in \
  "$PROTOCOL" "$MODEL_ROOT/config.json" "$E58_SMOKE" "$PANTRY_QUALIFICATION" \
  "$GRAPH_DATA/identity.json" \
  "$GRAPH_DATA/train/dataset_dict.json" \
  "$GRAPH_DATA/eval/dataset_dict.json" \
  "$PANTRY_DATA/train/dataset_dict.json" \
  "$PANTRY_DATA/eval/dataset_dict.json"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing E71 prerequisite: $required" >&2
    exit 1
  fi
done

"$PYTHON_BIN" - "$E58_SMOKE" <<'PY'
import json
import pathlib
import sys

for raw in sys.argv[1:]:
    path = pathlib.Path(raw)
    payload = json.loads(path.read_text())
    if payload.get("status") != "pass" or payload.get("violations"):
        raise SystemExit(f"E71 requires a clean smoke certificate: {path}")
PY

"$PYTHON_BIN" - "$PANTRY_QUALIFICATION" <<'PY'
import json
import pathlib
import sys

payload = json.loads(pathlib.Path(sys.argv[1]).read_text())
if payload.get("status") != "pass":
    raise SystemExit("E71 requires a passing PantryPlan qualification audit")
if payload.get("decision") != "eligible_for_ten_stage_b_jobs":
    raise SystemExit("E71 requires the PantryPlan ten-cell eligibility decision")
PY

"$PYTHON_BIN" - "$GRAPH_DATA/identity.json" <<'PY'
import json
import pathlib
import sys

payload = json.loads(pathlib.Path(sys.argv[1]).read_text())
rows = payload.get("split_rows", {})
if rows.get("train") != 384 or rows.get("eval_multi_answer") != 128:
    raise SystemExit("E71 graph data is not the registered 384/128 instance")
if any(payload.get("split_overlap_count", {"x": 1}).values()):
    raise SystemExit("E71 graph data has overlapping splits")
PY

if [[ "$phase" == full ]]; then
  for fresh in \
    "$IDENTITY" "$GRAPH_MANIFEST" "$PANTRY_MANIFEST"; do
    if [[ -e "$fresh" ]]; then
      echo "Fresh E71 artifact required; already exists: $fresh" >&2
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
  SOURCE_ROOT="$ROOT_DIR/var/artifacts/source_snapshots/e71_scale384_${SOURCE_HASH}/src"
  if [[ ! -f "$SOURCE_ROOT/oat_drgrpo/__init__.py" ]]; then
    mkdir -p "$(dirname "$SOURCE_ROOT")"
    staging="$(mktemp -d "$(dirname "$SOURCE_ROOT")/.source.XXXXXX")"
    mkdir -p "$staging/src"
    cp -a "$ROOT_DIR/src/." "$staging/src/"
    mv "$staging/src" "$SOURCE_ROOT"
    rmdir "$staging"
  fi
  if [[ "$(hash_tree "$SOURCE_ROOT")" != "$SOURCE_HASH" ]]; then
    echo "E71 source snapshot hash mismatch" >&2
    exit 1
  fi

  ops_input="$(mktemp -d "$ROOT_DIR/var/artifacts/source_snapshots/.e71-scale384-ops-input.XXXXXX")"
  mkdir -p "$ops_input/slurm"
  for file in \
    repo_env.sh run_experiment.sh train.sh resolve_eval_cadence.py \
    submit_countdown_comparative.sh; do
    cp "$ROOT_DIR/ops/$file" "$ops_input/$file"
  done
  cp "$ROOT_DIR/ops/slurm/train_node302.slurm" \
    "$ops_input/slurm/train_node302.slurm"
  EXECUTION_HASH="$(hash_tree "$ops_input")"
  OPS_ROOT="$ROOT_DIR/var/artifacts/source_snapshots/e71_scale384_ops_${EXECUTION_HASH}/ops"
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
    echo "E71 execution snapshot hash mismatch" >&2
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

GRAPH_DATA_HASH="$(hash_tree "$GRAPH_DATA")"
PANTRY_DATA_HASH="$(hash_tree "$PANTRY_DATA")"

write_identity() {
  "$PYTHON_BIN" - \
    "$IDENTITY" "$PROTOCOL" "$0" "$SOURCE_HASH" "$EXECUTION_HASH" \
    "$GRAPH_DATA_HASH" "$PANTRY_DATA_HASH" \
    "$GRAPH_MANIFEST" "$PANTRY_MANIFEST" <<'PY'
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
domain_names = ("graph_coloring", "pantry_plan")
manifests = [pathlib.Path(raw) for raw in sys.argv[8:10]]
jobs = {}
manifest_hashes = {}
for domain, manifest in zip(domain_names, manifests):
    rows = list(csv.DictReader(manifest.open(), delimiter="\t"))
    if len(rows) != 10:
        raise SystemExit(f"{manifest} has {len(rows)} jobs; expected 10")
    jobs[domain] = [
        {
            "arm": row["arm"],
            "seed": int(row["seed"]),
            "job_id": int(row["job_id"]),
            "run_stamp": row["run_stamp"],
        }
        for row in rows
    ]
    manifest_hashes[domain] = digest(manifest)

payload = {
    "schema": "e71_scale384_05b_v1",
    "protocol_sha256": digest(sys.argv[2]),
    "launcher_sha256": digest(sys.argv[3]),
    "source_hash": sys.argv[4],
    "execution_surface_hash": sys.argv[5],
    "data_tree_sha256": {
        "graph_coloring": sys.argv[6],
        "pantry_plan": sys.argv[7],
    },
    "train_rows": 384,
    "eval_rows": 128,
    "optimizer_updates_per_run": 4608,
    "supersedes": {
        "graph_coloring": "gce70_clean_stage_a_05b_12pass",
        "pantry_plan": "ppe70_clean_stage_b_05b_12pass",
    },
    "manifest_sha256": manifest_hashes,
    "jobs": jobs,
    "attempt_selection": "exact_manifest_job_ids",
    "model": (
        "Qwen2.5-0.5B-Instruct@"
        "7ae557604adf67be50417f59c2c2f167def9a775"
    ),
    "arms": [
        {
            "manifest_label": "grpo",
            "runtime_variant": "grpo_compute_matched",
            "objective": "plain_drgrpo",
        },
        {
            "manifest_label": "verified_first_global_replay_canonical",
            "runtime_variant": "verified_first_global_replay_canonical",
            "objective": "online_verified_maxent",
        },
    ],
    "seeds": [43, 44, 45, 46, 47],
    "prompt_epochs_per_run": 12,
    "num_samples": 16,
    "evaluation": {"k": 8, "replicates": 4, "cadence": "quarter_pass"},
    "coefficient_projection": None,
    "information_firewall": {
        "gold_support_feedback": False,
        "evaluation_feedback": False,
        "desired_entropy": None,
        "desired_mode_count": None,
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

export OAT_ZERO_REPO_ROOT="$ROOT_DIR"
export OAT_ZERO_OPS_SNAPSHOT_ROOT="$OPS_ROOT"
export OAT_ZERO_CAMPAIGN_SOURCE_ROOT="$SOURCE_ROOT"
export OAT_ZERO_PRETRAIN="$MODEL_ROOT"
export OAT_ZERO_REQUIRE_EXISTING_DATA=1
export OAT_ZERO_COMPARATIVE_REBUILD=0
export OAT_ZERO_APPEND_MANIFEST=0
export OAT_ZERO_COMPARATIVE_MODEL=qwen2.5-0.5b-instruct
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
      export OAT_ZERO_MAX_TRAIN=384
      export OAT_ZERO_EVAL_PROMPT_INTERVAL=96
      export OAT_ZERO_SAVE_STEPS=384
      export OAT_ZERO_SAVE_FROM=384
      export OAT_ZERO_RESUME_STEPS=384
      export OAT_ZERO_GENERATE_MAX_LENGTH=192
      export OAT_ZERO_EVAL_GENERATE_MAX_LENGTH=192
      export OAT_ZERO_MAX_MODEL_LEN=512
      export OAT_ZERO_EVAL_MODE_COVERAGE_SEED=610100
      export OAT_ZERO_TRAIN_NODELIST=node105,node302
      export OAT_ZERO_TRAIN_GRES=gpu:1
      export OAT_ZERO_TRAIN_PARTITION=mltheory
      export OAT_ZERO_TRAIN_ACCOUNT=mltheory
      ;;
    pantry_plan)
      # PantryPlan's support-mask interface overrides the shared boxed-prompt
      # decoding surface; every other coefficient is inherited unchanged.
      export RUN_STAMP_PREFIX="$PANTRY_PREFIX"
      export OAT_ZERO_COMPARATIVE_TASK=pantry_plan
      export OAT_ZERO_COMPARATIVE_DATA_ROOT="$PANTRY_DATA"
      export OAT_ZERO_MAX_TRAIN=384
      export OAT_ZERO_EVAL_PROMPT_INTERVAL=96
      export OAT_ZERO_SAVE_STEPS=384
      export OAT_ZERO_SAVE_FROM=384
      export OAT_ZERO_RESUME_STEPS=384
      export OAT_ZERO_PROMPT_TEMPLATE=qwen_pantry_support_mask
      export OAT_ZERO_CANONICAL_ACTION_TASK=pantry_support_mask
      export OAT_ZERO_CANONICAL_GRAPH_ACTION_COUNT=6
      export OAT_ZERO_CANONICAL_GRAPH_LEARNER_SAMPLING=1
      export OAT_ZERO_CANONICAL_GRAPH_FIXED_SHAPE_SAMPLING=1
      export OAT_ZERO_PROMPT_MAX_LENGTH=640
      export OAT_ZERO_GENERATE_MAX_LENGTH=8
      export OAT_ZERO_EVAL_GENERATE_MAX_LENGTH=8
      export OAT_ZERO_MAX_MODEL_LEN=704
      export OAT_ZERO_TRAIN_BATCH_SIZE_PER_DEVICE=4
      export OAT_ZERO_MAX_SAVE_MEM=2000
      export OAT_ZERO_EVAL_MODE_COVERAGE_SEED=76299
      export OAT_ZERO_TRAIN_NODELIST=node105,node302
      export OAT_ZERO_TRAIN_GRES=gpu:1
      export OAT_ZERO_TRAIN_PARTITION=mltheory
      export OAT_ZERO_TRAIN_ACCOUNT=mltheory
      ;;
    *)
      echo "Unknown E71 domain: $domain" >&2
      exit 1
      ;;
  esac
  "$OPS_ROOT/submit_countdown_comparative.sh"
}

if [[ "$phase" == config ]]; then
  export OAT_ZERO_COMPARATIVE_CONFIG_ONLY=1
  export OAT_ZERO_SBATCH_HOLD=0
  submit_domain graph_coloring
  submit_domain pantry_plan
  echo "[e71] both five-seed 384/128 12-pass configurations passed"
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
    echo "[e71] cancelled incomplete held cohort: ${job_ids[*]}" >&2
  fi
  exit "$status"
}
trap cleanup_held EXIT

submit_domain graph_coloring
submit_domain pantry_plan

for manifest in "$GRAPH_MANIFEST" "$PANTRY_MANIFEST"; do
  mapfile -t manifest_jobs < <(
    awk -F $'\t' 'NR > 1 && $3 ~ /^[0-9]+$/ {print $3}' "$manifest"
  )
  if [[ "${#manifest_jobs[@]}" -ne 10 ]]; then
    echo "E71 manifest $manifest has ${#manifest_jobs[@]} jobs; expected 10" >&2
    exit 1
  fi
  job_ids+=("${manifest_jobs[@]}")
done
if [[ "${#job_ids[@]}" -ne 20 ]]; then
  echo "E71 cohort has ${#job_ids[@]} jobs; expected 20" >&2
  exit 1
fi

write_identity

for job_id in "${job_ids[@]}"; do
  job_record="$(scontrol show job "$job_id" -o)"
  for required in \
    'JobState=PENDING' \
    'Reason=JobHeldUser' \
    'OAT_ZERO_MAX_PROMPT_EPOCHS=12' \
    'OAT_ZERO_NUM_PROMPT_EPOCH=12' \
    'OAT_ZERO_NUM_SAMPLES=16' \
    'OAT_ZERO_EVAL_MODE_COVERAGE_K=8' \
    'OAT_ZERO_EVAL_MODE_COVERAGE_DRAWS=4' \
    "OAT_ZERO_SOURCE_ROOT=${SOURCE_ROOT}" \
    "OAT_ZERO_OPS_SNAPSHOT_ROOT=${OPS_ROOT}" \
    "OAT_ZERO_PROTOCOL_IDENTITY=${IDENTITY}" \
    'NumCPUs=8' \
    'MinMemoryNode=64G' \
    'TimeLimit=1-00:00:00'; do
    if [[ "$job_record" != *"$required"* ]]; then
      echo "E71 held-job audit failed for $job_id: missing $required" >&2
      exit 1
    fi
  done
  if [[ "$job_record" == *'OAT_ZERO_VARIANT=verified_first_global_replay_canonical'* ]]; then
    for required in \
      'OAT_ZERO_MAXENT_ALPHA=0.0' \
      'OAT_ZERO_MAXENT_INVERSE_ADAPTATION=0' \
      'OAT_ZERO_SEMANTIC_SHANNON_OPEN_SET_INVERSE_ADAPTATION=1' \
      'OAT_ZERO_ONLINE_CANONICAL_NOVELTY_BETA=0.5' \
      'OAT_ZERO_ONLINE_CANONICAL_REPLAY_GLOBAL_GROUPS_PER_STEP=1' \
      'OAT_ZERO_ONLINE_CANONICAL_REPLAY_GLOBAL_BOOTSTRAP_STEPS=0' \
      'OAT_ZERO_ONLINE_CANONICAL_REPLAY_OBJECTIVE=split_mass_balance_per_rollout' \
      'OAT_ZERO_ONLINE_CANONICAL_REPLAY_COMPUTE_ONLY=0'; do
      if [[ "$job_record" != *"$required"* ]]; then
        echo "E71 treatment audit failed for $job_id: missing $required" >&2
        exit 1
      fi
    done
  elif [[ "$job_record" == *'OAT_ZERO_VARIANT=grpo_compute_matched'* ]]; then
    for required in \
      'OAT_ZERO_ONLINE_CANONICAL_BANK_ALPHA=0' \
      'OAT_ZERO_ONLINE_CANONICAL_NOVELTY_BETA=0' \
      'OAT_ZERO_ONLINE_CANONICAL_REPLAY=1' \
      'OAT_ZERO_ONLINE_CANONICAL_REPLAY_OBJECTIVE=verified_likelihood_per_rollout' \
      'OAT_ZERO_ONLINE_CANONICAL_REPLAY_GLOBAL_GROUPS_PER_STEP=1' \
      'OAT_ZERO_ONLINE_CANONICAL_REPLAY_COMPUTE_ONLY=1'; do
      if [[ "$job_record" != *"$required"* ]]; then
        echo "E71 baseline audit failed for $job_id: missing $required" >&2
        exit 1
      fi
    done
  else
    echo "E71 job $job_id is neither frozen arm" >&2
    exit 1
  fi
done

scontrol release "${job_ids[@]}"
released=1
trap - EXIT
echo "[e71] released 20 matched jobs: ${job_ids[*]}"
echo "[e71] identity=$IDENTITY"
