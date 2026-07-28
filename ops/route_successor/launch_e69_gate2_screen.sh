#!/usr/bin/env bash
# Configure or atomically submit E69 Gate 2's 18 physical jobs.
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
PARENT_PROTOCOL="$ROOT_DIR/paper/preregistration/e69_verified_route_successor_protocol_20260728.md"
EXEC_PROTOCOL="$ROOT_DIR/paper/preregistration/e69_gate2_compute_matched_screen_20260728.md"
MATH_ABSTENTION="$ROOT_DIR/paper/preregistration/e69_math_route_gate1_equation_v3_failure_20260728.md"
MODEL_ROOT="$ROOT_DIR/var/cache/huggingface/transformers/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
GRAPH_DATA="$ROOT_DIR/var/data/exact_answer_mode_probe"
COUNTDOWN_DATA="$ROOT_DIR/var/data/exact_countdown_easy3_probe"
PYTHON_DATA="$ROOT_DIR/var/data/python_factor_modebench_v1"
MATHIR_DATA="$ROOT_DIR/var/data/mathir_action_menu_v1"
MATH_DATA="$ROOT_DIR/var/data/math12k_384_route_dev128_v1"
IDENTITY="$ROOT_DIR/var/artifacts/e69_gate2_compute_matched_screen_identity.json"

GRAPH_PREFIX=gce69_gate2_compute_matched_05b_6pass
COUNTDOWN_PREFIX=cde69_gate2_compute_matched_05b_6pass
PYTHON_PREFIX=pye69_gate2_compute_matched_05b_6pass
MATHIR_PREFIX=mie69_gate2_compute_matched_05b_6pass
MATH_PREFIX=mde69_gate2_compute_matched_05b_6pass
GRAPH_MANIFEST="$ROOT_DIR/var/artifacts/${GRAPH_PREFIX}_comparative_jobs.tsv"
COUNTDOWN_MANIFEST="$ROOT_DIR/var/artifacts/${COUNTDOWN_PREFIX}_comparative_jobs.tsv"
PYTHON_MANIFEST="$ROOT_DIR/var/artifacts/${PYTHON_PREFIX}_comparative_jobs.tsv"
MATHIR_MANIFEST="$ROOT_DIR/var/artifacts/${MATHIR_PREFIX}_comparative_jobs.tsv"
MATH_MANIFEST="$ROOT_DIR/var/artifacts/${MATH_PREFIX}_comparative_jobs.tsv"

for required in \
  "$PARENT_PROTOCOL" "$EXEC_PROTOCOL" "$MATH_ABSTENTION" \
  "$MODEL_ROOT/config.json" \
  "$GRAPH_DATA/train/dataset_dict.json" \
  "$GRAPH_DATA/eval/dataset_dict.json" \
  "$COUNTDOWN_DATA/train/dataset_dict.json" \
  "$COUNTDOWN_DATA/eval/dataset_dict.json" \
  "$PYTHON_DATA/train/dataset_dict.json" \
  "$PYTHON_DATA/eval/dataset_dict.json" \
  "$MATHIR_DATA/train/dataset_dict.json" \
  "$MATHIR_DATA/eval/dataset_dict.json" \
  "$MATH_DATA/train/dataset_dict.json" \
  "$MATH_DATA/eval/dataset_dict.json" \
  "$MATH_DATA/MATERIALIZATION_MANIFEST.json"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing E69 Gate 2 prerequisite: $required" >&2
    exit 1
  fi
done

if [[ "$phase" == full ]]; then
  for fresh in \
    "$IDENTITY" "$GRAPH_MANIFEST" "$COUNTDOWN_MANIFEST" \
    "$PYTHON_MANIFEST" "$MATHIR_MANIFEST" "$MATH_MANIFEST"; do
    if [[ -e "$fresh" ]]; then
      echo "Fresh E69 Gate 2 artifact required; already exists: $fresh" >&2
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
  SOURCE_ROOT="$ROOT_DIR/var/artifacts/source_snapshots/e69_gate2_${SOURCE_HASH}/src"
  if [[ ! -f "$SOURCE_ROOT/oat_drgrpo/__init__.py" ]]; then
    mkdir -p "$(dirname "$SOURCE_ROOT")"
    staging="$(mktemp -d "$(dirname "$SOURCE_ROOT")/.source.XXXXXX")"
    mkdir -p "$staging/src"
    cp -a "$ROOT_DIR/src/." "$staging/src/"
    mv "$staging/src" "$SOURCE_ROOT"
    rmdir "$staging"
  fi
  if [[ "$(hash_tree "$SOURCE_ROOT")" != "$SOURCE_HASH" ]]; then
    echo "E69 Gate 2 source snapshot hash mismatch" >&2
    exit 1
  fi

  ops_input="$(mktemp -d "$ROOT_DIR/var/artifacts/source_snapshots/.e69-g2-ops.XXXXXX")"
  mkdir -p "$ops_input/slurm"
  for file in \
    repo_env.sh run_experiment.sh train.sh resolve_eval_cadence.py \
    submit_countdown_comparative.sh; do
    cp "$ROOT_DIR/ops/$file" "$ops_input/$file"
  done
  cp "$ROOT_DIR/ops/slurm/train_node302.slurm" \
    "$ops_input/slurm/train_node302.slurm"
  EXECUTION_HASH="$(hash_tree "$ops_input")"
  OPS_ROOT="$ROOT_DIR/var/artifacts/source_snapshots/e69_gate2_ops_${EXECUTION_HASH}/ops"
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
    echo "E69 Gate 2 execution snapshot hash mismatch" >&2
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
COUNTDOWN_DATA_HASH="$(hash_tree "$COUNTDOWN_DATA")"
PYTHON_DATA_HASH="$(hash_tree "$PYTHON_DATA")"
MATHIR_DATA_HASH="$(hash_tree "$MATHIR_DATA")"
MATH_DATA_HASH="$(hash_tree "$MATH_DATA")"

write_identity() {
  export SOURCE_HASH EXECUTION_HASH
  "$PYTHON_BIN" - \
    "$IDENTITY" "$PARENT_PROTOCOL" "$EXEC_PROTOCOL" "$MATH_ABSTENTION" \
    "$0" "$GRAPH_DATA_HASH" "$COUNTDOWN_DATA_HASH" "$PYTHON_DATA_HASH" \
    "$MATHIR_DATA_HASH" "$MATH_DATA_HASH" \
    "$GRAPH_MANIFEST" "$COUNTDOWN_MANIFEST" "$PYTHON_MANIFEST" \
    "$MATHIR_MANIFEST" "$MATH_MANIFEST" <<'PY'
import csv
import hashlib
import json
import os
import pathlib
import sys
import tempfile

def digest(raw):
    return hashlib.sha256(pathlib.Path(raw).read_bytes()).hexdigest()

identity = pathlib.Path(sys.argv[1])
domains = ("graph_coloring", "countdown", "python_factor", "mathir", "math_dev")
manifests = [pathlib.Path(raw) for raw in sys.argv[11:16]]
expected_rows = (4, 4, 4, 4, 2)
jobs = {}
manifest_hashes = {}
for domain, manifest, expected in zip(domains, manifests, expected_rows):
    rows = list(csv.DictReader(manifest.open(), delimiter="\t"))
    if len(rows) != expected:
        raise SystemExit(f"{manifest} has {len(rows)} rows; expected {expected}")
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
    "schema": "e69_gate2_compute_matched_screen_v1",
    "parent_protocol_sha256": digest(sys.argv[2]),
    "execution_protocol_sha256": digest(sys.argv[3]),
    "math_abstention_sha256": digest(sys.argv[4]),
    "launcher_sha256": digest(sys.argv[5]),
    "source_hash": os.environ["SOURCE_HASH"],
    "execution_surface_hash": os.environ["EXECUTION_HASH"],
    "data_tree_sha256": dict(zip(domains, sys.argv[6:11])),
    "manifest_sha256": manifest_hashes,
    "jobs": jobs,
    "attempt_selection": "exact_manifest_job_ids",
    "model": (
        "Qwen2.5-0.5B-Instruct@"
        "7ae557604adf67be50417f59c2c2f167def9a775"
    ),
    "seed": 43,
    "passes": 6,
    "num_samples": 16,
    "physical_job_count": 18,
    "modebench_arms": [
        "grpo",
        "verified_first_global_replay_canonical",
        "verified_entropy_gated_singleton_escape_canonical",
        "verified_route_successor",
    ],
    "math_physical_arms": [
        "grpo",
        "verified_first_global_replay_canonical",
    ],
    "math_aliases": {
        "E66": "verified_first_global_replay_canonical",
        "E68": "verified_first_global_replay_canonical",
        "E69": "verified_first_global_replay_canonical",
    },
    "fixed_compute": {
        "neutral_groups_per_prompt": 1,
        "proposal_control_groups_per_prompt": 3,
        "rows_per_group": 16,
        "sampled_rows_per_prompt": 64,
        "replay_groups_per_update": 1,
        "replay_capacity": 16,
        "replay_score_passes": 2,
        "drgrpo_replay_gradient": 0,
    },
    "reporting_passes": [0, 1, 2, 3, 4, 5, 6],
    "checkpoint_selection": "terminal_pass_6_only",
    "math500_sealed": True,
}
identity.parent.mkdir(parents=True, exist_ok=True)
fd, temporary = tempfile.mkstemp(prefix=f".{identity.name}.", dir=identity.parent)
with os.fdopen(fd, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
os.replace(temporary, identity)
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
export OAT_ZERO_TRAIN_SEEDS=43
export OAT_ZERO_XDR_TAUS=""
export OAT_ZERO_DRGRPO_VARIANT=grpo_compute_matched

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
  VERIFIED_FIRST_SPLIT_CANONICAL VERIFIED_FIRST_BOOTSTRAP_LOCAL_CANONICAL \
  VERIFIED_COUNTERFACTUAL_CANONICAL VERIFIED_FIRST_GLOBAL_REPLAY_CANONICAL \
  VERIFIED_ENTROPY_GATED_SINGLETON_ESCAPE_CANONICAL \
  VERIFIED_ROUTE_SUCCESSOR; do
  export "OAT_ZERO_INCLUDE_${flag}_ARM=0"
done

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
export OAT_ZERO_ONLINE_CANONICAL_REPLAY_ALPHA=0.10
export OAT_ZERO_ONLINE_CANONICAL_REPLAY_CAPACITY=16
export OAT_ZERO_ONLINE_CANONICAL_REPLAY_GLOBAL_GROUPS_PER_STEP=1
export OAT_ZERO_ONLINE_CANONICAL_REPLAY_GLOBAL_BOOTSTRAP_STEPS=0
export OAT_ZERO_ONLINE_CANONICAL_REPLAY_WARMUP_STEPS=64
export OAT_ZERO_ONLINE_CANONICAL_REPLAY_EMA_DECAY=0.90
export OAT_ZERO_ONLINE_CANONICAL_REPLAY_MASS_ALPHA=0.10
export OAT_ZERO_ONLINE_CANONICAL_REPLAY_MASS_WARMUP_STEPS=64
export OAT_ZERO_ONLINE_CANONICAL_REPLAY_MASS_EMA_DECAY=0.90
export OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_FIXED_CONTROL_GROUPS=3
export OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_MAX_ATTEMPTS=3
export OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_SAMPLING_TEMPERATURE=1.0
export OAT_ZERO_VERIFIED_ROUTE_REPLAY_CAPACITY_PER_ROUTE=16
export OAT_ZERO_VERIFIED_ROUTE_RECURRING_MIN_NEUTRAL_PROMPTS=2
export OAT_ZERO_VERIFIED_ROUTE_PROPOSAL_MAX_MEAN_LOGPROB_DROP=2.0

export OAT_ZERO_NUM_SAMPLES=16
export OAT_ZERO_LEARNING_RATE=0.0000002
export OAT_ZERO_MAX_QUERIES=100000000
export OAT_ZERO_MAX_PROMPT_EPOCHS=6
export OAT_ZERO_NUM_PROMPT_EPOCH=6
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
export OAT_ZERO_INPUT_KEY=problem
export OAT_ZERO_OUTPUT_KEY=answer
export OAT_ZERO_EVAL_INPUT_KEY=problem
export OAT_ZERO_EVAL_OUTPUT_KEY=answer
export OAT_ZERO_TEMPERATURE=1
export OAT_ZERO_TOP_P=1
export OAT_ZERO_EVAL_TEMPERATURE=0
export OAT_ZERO_EVAL_MODE_COVERAGE_K=8
export OAT_ZERO_EVAL_MODE_COVERAGE_TEMPERATURE=1
export OAT_ZERO_EVAL_MODE_COVERAGE_DRAWS=1
export OAT_ZERO_EVAL_BATCH_SIZE=64
export OAT_ZERO_ALLOW_SPARSE_EVAL=1
export OAT_ZERO_SYNC_PARAMS_EVERY=1
export OAT_ZERO_CANONICAL_ACTION_TASK=none
export OAT_ZERO_CANONICAL_GRAPH_ACTIONS=0
export OAT_ZERO_REPLICATED_FREEFORM_SAMPLING=1
export OAT_ZERO_LOCAL_ACTOR_WEIGHT_SYNC=1
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
export OAT_ZERO_MAX_RESUME_NUM=2
export OAT_ZERO_PRUNE_RESUME_ON_SUCCESS=0
export OAT_ZERO_AUTO_RESUME=1
export OAT_ZERO_WATCHDOG_REQUEUE=1
export OAT_ZERO_WATCHDOG_MAX_RESTARTS=8
export OAT_ZERO_TRAIN_CPUS_PER_TASK=8
export OAT_ZERO_TRAIN_MEMORY=64G

submit_domain() {
  local domain="$1"
  export OAT_ZERO_SEMANTIC_SHANNON_COEF=0.10
  export OAT_ZERO_ONLINE_CANONICAL_NOVELTY_BETA=0.50
  export OAT_ZERO_ONLINE_CANONICAL_KEY_MODE=modebench_outcome
  export OAT_ZERO_ONLY_ARMS="grpo,verified_first_global_replay_canonical,verified_entropy_gated_singleton_escape_canonical,verified_route_successor"
  export OAT_ZERO_INCLUDE_VERIFIED_FIRST_GLOBAL_REPLAY_CANONICAL_ARM=1
  export OAT_ZERO_INCLUDE_VERIFIED_ENTROPY_GATED_SINGLETON_ESCAPE_CANONICAL_ARM=1
  export OAT_ZERO_INCLUDE_VERIFIED_ROUTE_SUCCESSOR_ARM=1
  export OAT_ZERO_PROMPT_TEMPLATE=qwen_boxed
  export OAT_ZERO_TEST_SPLIT=multi_answer
  export OAT_ZERO_VERIFIER_VERSION=fast
  export OAT_ZERO_PROMPT_MAX_LENGTH=256
  export OAT_ZERO_EVAL_MODE_COVERAGE_DRAWS=1
  export OAT_ZERO_TRAIN_TIME_LIMIT=3-00:00:00
  case "$domain" in
    graph_coloring)
      export RUN_STAMP_PREFIX="$GRAPH_PREFIX"
      export OAT_ZERO_COMPARATIVE_TASK=graph_coloring
      export OAT_ZERO_COMPARATIVE_DATA_ROOT="$GRAPH_DATA"
      export OAT_ZERO_MAX_TRAIN=192
      export OAT_ZERO_EVAL_PROMPT_INTERVAL=192
      export OAT_ZERO_SAVE_STEPS=192
      export OAT_ZERO_SAVE_FROM=192
      export OAT_ZERO_RESUME_STEPS=192
      export OAT_ZERO_RESUME_FROM=192
      export OAT_ZERO_GENERATE_MAX_LENGTH=192
      export OAT_ZERO_EVAL_GENERATE_MAX_LENGTH=192
      export OAT_ZERO_MAX_MODEL_LEN=512
      export OAT_ZERO_EVAL_MODE_COVERAGE_SEED=690201
      export OAT_ZERO_TRAIN_NODELIST="${OAT_ZERO_E69_GRAPH_NODELIST:-node103,node104,node208}"
      export OAT_ZERO_TRAIN_GRES="${OAT_ZERO_E69_GRAPH_GRES:-gpu:a6000:1}"
      export OAT_ZERO_TRAIN_PARTITION="${OAT_ZERO_E69_GRAPH_PARTITION:-pvl-lowprio}"
      export OAT_ZERO_TRAIN_ACCOUNT="${OAT_ZERO_E69_GRAPH_ACCOUNT:-mltheory}"
      ;;
    countdown)
      export RUN_STAMP_PREFIX="$COUNTDOWN_PREFIX"
      export OAT_ZERO_COMPARATIVE_TASK=countdown
      export OAT_ZERO_COMPARATIVE_DATA_PRESET=easy3
      export OAT_ZERO_COMPARATIVE_DATA_ROOT="$COUNTDOWN_DATA"
      export OAT_ZERO_MAX_TRAIN=384
      export OAT_ZERO_EVAL_PROMPT_INTERVAL=384
      export OAT_ZERO_SAVE_STEPS=384
      export OAT_ZERO_SAVE_FROM=384
      export OAT_ZERO_RESUME_STEPS=384
      export OAT_ZERO_RESUME_FROM=384
      export OAT_ZERO_GENERATE_MAX_LENGTH=192
      export OAT_ZERO_EVAL_GENERATE_MAX_LENGTH=192
      export OAT_ZERO_MAX_MODEL_LEN=512
      export OAT_ZERO_EVAL_MODE_COVERAGE_SEED=690202
      export OAT_ZERO_TRAIN_NODELIST="${OAT_ZERO_E69_3090_NODELIST:-node020,node021,node022,node024,node026}"
      export OAT_ZERO_TRAIN_GRES="${OAT_ZERO_E69_3090_GRES:-gpu:rtx_3090:1}"
      export OAT_ZERO_TRAIN_PARTITION="${OAT_ZERO_E69_3090_PARTITION:-pvl-lowprio}"
      export OAT_ZERO_TRAIN_ACCOUNT="${OAT_ZERO_E69_3090_ACCOUNT:-mltheory}"
      ;;
    python_factor)
      export RUN_STAMP_PREFIX="$PYTHON_PREFIX"
      export OAT_ZERO_COMPARATIVE_TASK=python_factor
      export OAT_ZERO_COMPARATIVE_DATA_ROOT="$PYTHON_DATA"
      export OAT_ZERO_MAX_TRAIN=384
      export OAT_ZERO_EVAL_PROMPT_INTERVAL=384
      export OAT_ZERO_SAVE_STEPS=384
      export OAT_ZERO_SAVE_FROM=384
      export OAT_ZERO_RESUME_STEPS=384
      export OAT_ZERO_RESUME_FROM=384
      export OAT_ZERO_GENERATE_MAX_LENGTH=192
      export OAT_ZERO_EVAL_GENERATE_MAX_LENGTH=192
      export OAT_ZERO_MAX_MODEL_LEN=512
      export OAT_ZERO_EVAL_MODE_COVERAGE_SEED=690203
      export OAT_ZERO_TRAIN_NODELIST="${OAT_ZERO_E69_3090_NODELIST:-node020,node021,node022,node024,node026}"
      export OAT_ZERO_TRAIN_GRES="${OAT_ZERO_E69_3090_GRES:-gpu:rtx_3090:1}"
      export OAT_ZERO_TRAIN_PARTITION="${OAT_ZERO_E69_3090_PARTITION:-pvl-lowprio}"
      export OAT_ZERO_TRAIN_ACCOUNT="${OAT_ZERO_E69_3090_ACCOUNT:-mltheory}"
      ;;
    mathir)
      export RUN_STAMP_PREFIX="$MATHIR_PREFIX"
      export OAT_ZERO_COMPARATIVE_TASK=math
      export OAT_ZERO_COMPARATIVE_DATA_ROOT="$MATHIR_DATA"
      export OAT_ZERO_MAX_TRAIN=384
      export OAT_ZERO_EVAL_PROMPT_INTERVAL=384
      export OAT_ZERO_SAVE_STEPS=384
      export OAT_ZERO_SAVE_FROM=384
      export OAT_ZERO_RESUME_STEPS=384
      export OAT_ZERO_RESUME_FROM=384
      export OAT_ZERO_GENERATE_MAX_LENGTH=64
      export OAT_ZERO_EVAL_GENERATE_MAX_LENGTH=64
      export OAT_ZERO_MAX_MODEL_LEN=384
      export OAT_ZERO_EVAL_MODE_COVERAGE_SEED=690204
      export OAT_ZERO_TRAIN_NODELIST="${OAT_ZERO_E69_A100_NODELIST:-node302}"
      export OAT_ZERO_TRAIN_GRES="${OAT_ZERO_E69_A100_GRES:-gpu:a100:1}"
      export OAT_ZERO_TRAIN_PARTITION="${OAT_ZERO_E69_A100_PARTITION:-lowprio}"
      export OAT_ZERO_TRAIN_ACCOUNT="${OAT_ZERO_E69_A100_ACCOUNT:-mltheory}"
      ;;
    math_dev)
      export RUN_STAMP_PREFIX="$MATH_PREFIX"
      export OAT_ZERO_COMPARATIVE_TASK=math
      export OAT_ZERO_COMPARATIVE_DATA_ROOT="$MATH_DATA"
      export OAT_ZERO_ONLY_ARMS="grpo,verified_first_global_replay_canonical"
      export OAT_ZERO_INCLUDE_VERIFIED_FIRST_GLOBAL_REPLAY_CANONICAL_ARM=1
      export OAT_ZERO_INCLUDE_VERIFIED_ENTROPY_GATED_SINGLETON_ESCAPE_CANONICAL_ARM=0
      export OAT_ZERO_INCLUDE_VERIFIED_ROUTE_SUCCESSOR_ARM=0
      export OAT_ZERO_SEMANTIC_SHANNON_COEF=0
      export OAT_ZERO_ONLINE_CANONICAL_NOVELTY_BETA=0
      export OAT_ZERO_ONLINE_CANONICAL_KEY_MODE=math_verified_answer
      export OAT_ZERO_PROMPT_TEMPLATE=qwen_math
      export OAT_ZERO_TEST_SPLIT=math
      export OAT_ZERO_VERIFIER_VERSION=math_verify
      export OAT_ZERO_MAX_TRAIN=384
      export OAT_ZERO_EVAL_PROMPT_INTERVAL=384
      export OAT_ZERO_SAVE_STEPS=384
      export OAT_ZERO_SAVE_FROM=384
      export OAT_ZERO_RESUME_STEPS=384
      export OAT_ZERO_RESUME_FROM=384
      export OAT_ZERO_PROMPT_MAX_LENGTH=1024
      export OAT_ZERO_GENERATE_MAX_LENGTH=1024
      export OAT_ZERO_EVAL_GENERATE_MAX_LENGTH=1024
      export OAT_ZERO_MAX_MODEL_LEN=2048
      export OAT_ZERO_EVAL_MODE_COVERAGE_SEED=690205
      export OAT_ZERO_TRAIN_TIME_LIMIT=7-00:00:00
      export OAT_ZERO_TRAIN_NODELIST="${OAT_ZERO_E69_A100_NODELIST:-node302}"
      export OAT_ZERO_TRAIN_GRES="${OAT_ZERO_E69_A100_GRES:-gpu:a100:1}"
      export OAT_ZERO_TRAIN_PARTITION="${OAT_ZERO_E69_A100_PARTITION:-lowprio}"
      export OAT_ZERO_TRAIN_ACCOUNT="${OAT_ZERO_E69_A100_ACCOUNT:-mltheory}"
      ;;
    *)
      echo "Unknown E69 Gate 2 domain: $domain" >&2
      exit 1
      ;;
  esac
  "$OPS_ROOT/submit_countdown_comparative.sh"
}

if [[ "$phase" == config ]]; then
  export OAT_ZERO_COMPARATIVE_CONFIG_ONLY=1
  export OAT_ZERO_SBATCH_HOLD=0
  for domain in graph_coloring countdown python_factor mathir math_dev; do
    submit_domain "$domain"
  done
  echo "[e69-gate2] all five domain configurations passed"
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
    echo "[e69-gate2] cancelled incomplete held cohort: ${job_ids[*]}" >&2
  fi
  exit "$status"
}
trap cleanup_held EXIT

for domain in graph_coloring countdown python_factor mathir math_dev; do
  submit_domain "$domain"
done

for manifest_spec in \
  "$GRAPH_MANIFEST:4" "$COUNTDOWN_MANIFEST:4" "$PYTHON_MANIFEST:4" \
  "$MATHIR_MANIFEST:4" "$MATH_MANIFEST:2"; do
  manifest="${manifest_spec%:*}"
  expected="${manifest_spec##*:}"
  mapfile -t manifest_jobs < <(
    awk -F $'\t' 'NR > 1 && $3 ~ /^[0-9]+$/ {print $3}' "$manifest"
  )
  if [[ "${#manifest_jobs[@]}" -ne "$expected" ]]; then
    echo "E69 Gate 2 manifest $manifest has ${#manifest_jobs[@]} jobs; expected $expected" >&2
    exit 1
  fi
  job_ids+=("${manifest_jobs[@]}")
done
if [[ "${#job_ids[@]}" -ne 18 ]]; then
  echo "E69 Gate 2 cohort has ${#job_ids[@]} jobs; expected 18" >&2
  exit 1
fi

write_identity

for job_id in "${job_ids[@]}"; do
  record="$(scontrol show job "$job_id" -o)"
  for required in \
    'JobState=PENDING' 'Reason=JobHeldUser' \
    'OAT_ZERO_MAX_PROMPT_EPOCHS=6' \
    'OAT_ZERO_NUM_PROMPT_EPOCH=6' \
    'OAT_ZERO_NUM_SAMPLES=16' \
    'OAT_ZERO_REPLICATED_FREEFORM_SAMPLING=1' \
    'OAT_ZERO_LOCAL_ACTOR_WEIGHT_SYNC=1' \
    'OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_FIXED_CONTROL_GROUPS=3' \
    'OAT_ZERO_ONLINE_CANONICAL_REPLAY_GLOBAL_GROUPS_PER_STEP=1' \
    "OAT_ZERO_SOURCE_ROOT=${SOURCE_ROOT}" \
    "OAT_ZERO_OPS_SNAPSHOT_ROOT=${OPS_ROOT}" \
    "OAT_ZERO_PROTOCOL_IDENTITY=${IDENTITY}"; do
    if [[ "$record" != *"$required"* ]]; then
      echo "E69 Gate 2 held-job audit failed for $job_id: missing $required" >&2
      exit 1
    fi
  done
  if [[ "$record" == *'OAT_ZERO_VARIANT=grpo_compute_matched'* ]]; then
    for required in \
      'OAT_ZERO_ONLINE_CANONICAL_REPLAY=1' \
      'OAT_ZERO_ONLINE_CANONICAL_REPLAY_COMPUTE_ONLY=1' \
      'OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_PROPOSALS=0'; do
      [[ "$record" == *"$required"* ]] || {
        echo "E69 Gate 2 Dr.GRPO audit failed for $job_id: missing $required" >&2
        exit 1
      }
    done
  elif [[ "$record" == *'OAT_ZERO_VARIANT=verified_route_successor'* ]]; then
    for required in \
      'OAT_ZERO_ONLINE_CANONICAL_KEY_MODE=verified_route' \
      'OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_PROPOSALS=1' \
      'OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_SEPARATE_OBJECTIVE_SUPPORT=1'; do
      [[ "$record" == *"$required"* ]] || {
        echo "E69 Gate 2 route audit failed for $job_id: missing $required" >&2
        exit 1
      }
    done
  fi
done

scontrol release "${job_ids[@]}"
released=1
trap - EXIT
echo "[e69-gate2] released 18 compute-matched jobs: ${job_ids[*]}"
echo "[e69-gate2] identity=$IDENTITY"
