#!/usr/bin/env bash
# Launch E41's treatment-only, separately centered semantic-Shannon cohort.
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
PROTOCOL="$ROOT_DIR/paper/preregistration/e41_semantic_shannon_advantage_05b.md"
MATERIALIZER="$ROOT_DIR/ops/math500/materialize_e39_math12k_384.py"
MODEL_ROOT="$ROOT_DIR/var/cache/huggingface/transformers/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
GRAPH_DATA_ROOT="$ROOT_DIR/var/data/exact_answer_mode_probe"
COUNTDOWN_DATA_ROOT="$ROOT_DIR/var/data/exact_countdown_easy3_probe"
MATH_DATA_ROOT="$ROOT_DIR/var/data/math12k_384_math500"
MATH_DATA_IDENTITY="$MATH_DATA_ROOT/MATERIALIZATION_MANIFEST.json"
MATH_TRAIN_PARENT="$ROOT_DIR/var/seed_paper_eval/external/SEED-GRPO/datasets/train/math_12k"
MATH_EVAL_PARENT="$ROOT_DIR/var/data/oat_drgrpo_math_paper/eval"

GRAPH_PREFIX=gce41_semantic_shannon_advantage_05b_v1
COUNTDOWN_PREFIX=cde41_semantic_shannon_advantage_05b_v1
MATH_PREFIX=mte41_math12k_384_semantic_shannon_advantage_05b_v1
IDENTITY="$ROOT_DIR/var/artifacts/e41_semantic_shannon_advantage_05b_v1_identity.json"

E37_IDENTITY="$ROOT_DIR/var/artifacts/e37_outcome_collision_05b_v1_identity.json"
E37_GRAPH_MANIFEST="$ROOT_DIR/var/artifacts/gce37_outcome_collision_05b_v1_comparative_jobs.tsv"
E37_COUNTDOWN_MANIFEST="$ROOT_DIR/var/artifacts/cde37_outcome_collision_05b_v1_comparative_jobs.tsv"
E38_IDENTITY="$ROOT_DIR/var/artifacts/e38_semantic_shannon_05b_v1_identity.json"
E38_GRAPH_MANIFEST="$ROOT_DIR/var/artifacts/gce38_semantic_shannon_05b_v1_comparative_jobs.tsv"
E38_COUNTDOWN_MANIFEST="$ROOT_DIR/var/artifacts/cde38_semantic_shannon_05b_v1_comparative_jobs.tsv"
E39_IDENTITY="$ROOT_DIR/var/artifacts/mte39_math12k_384_semantic_entropy_05b_v1_identity.json"
E39_MATH_MANIFEST="$ROOT_DIR/var/artifacts/mte39_math12k_384_semantic_entropy_05b_v1_comparative_jobs.tsv"

EXPECTED_JOBS_PER_TASK=3
EXPECTED_JOBS=9

for required in \
  "$PYTHON_BIN" \
  "$PROTOCOL" \
  "$MATERIALIZER" \
  "$E37_IDENTITY" \
  "$E37_GRAPH_MANIFEST" \
  "$E37_COUNTDOWN_MANIFEST" \
  "$E38_IDENTITY" \
  "$E38_GRAPH_MANIFEST" \
  "$E38_COUNTDOWN_MANIFEST" \
  "$E39_IDENTITY" \
  "$E39_MATH_MANIFEST" \
  "$MODEL_ROOT/config.json" \
  "$MODEL_ROOT/tokenizer.json" \
  "$MODEL_ROOT/model.safetensors" \
  "$GRAPH_DATA_ROOT/train/dataset_dict.json" \
  "$GRAPH_DATA_ROOT/eval/dataset_dict.json" \
  "$COUNTDOWN_DATA_ROOT/train/dataset_dict.json" \
  "$COUNTDOWN_DATA_ROOT/eval/dataset_dict.json" \
  "$MATH_TRAIN_PARENT/dataset_dict.json" \
  "$MATH_EVAL_PARENT/dataset_dict.json"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing frozen E41 prerequisite: $required" >&2
    exit 1
  fi
done
if ! grep -q '^\*\*Status: FROZEN BEFORE LAUNCH' "$PROTOCOL"; then
  echo "E41 protocol is not frozen" >&2
  exit 1
fi

if [[ -f "$MATH_DATA_IDENTITY" ]]; then
  "$PYTHON_BIN" "$MATERIALIZER" \
    --output-root "$MATH_DATA_ROOT" \
    --audit-only >/dev/null
else
  "$PYTHON_BIN" "$MATERIALIZER" --output-root "$MATH_DATA_ROOT" >/dev/null
fi
for required in \
  "$MATH_DATA_ROOT/train/dataset_dict.json" \
  "$MATH_DATA_ROOT/eval/dataset_dict.json" \
  "$MATH_DATA_IDENTITY"; do
  if [[ ! -e "$required" ]]; then
    echo "E41 MATH materialization is incomplete: $required" >&2
    exit 1
  fi
done

if ! grep -q 'semantic_shannon_separate_advantage' \
    "$ROOT_DIR/src/oat_drgrpo/args.py" || \
   ! grep -q 'semantic_shannon_separate_advantage' \
     "$ROOT_DIR/src/oat_drgrpo/learner/grpo.py" || \
   ! grep -q 'OAT_ZERO_SEMANTIC_SHANNON_SEPARATE_ADVANTAGE' \
     "$ROOT_DIR/ops/train.sh" || \
   ! grep -q 'submit_arm semantic_shannon_advantage semantic_shannon_advantage' \
     "$ROOT_DIR/ops/submit_countdown_comparative.sh" || \
   ! grep -q 'semantic_shannon_tracker_state' \
     "$ROOT_DIR/src/oat_drgrpo/learner/run.py" || \
   ! grep -q '_last_evaluated_global_step' \
     "$ROOT_DIR/src/oat_drgrpo/learner/run.py"; then
  echo "Current source/execution stack lacks the frozen E41 semantic-advantage mechanism" >&2
  exit 1
fi

"$PYTHON_BIN" - \
  "$E37_IDENTITY" "$E37_GRAPH_MANIFEST" "$E37_COUNTDOWN_MANIFEST" \
  "$E38_IDENTITY" "$E38_GRAPH_MANIFEST" "$E38_COUNTDOWN_MANIFEST" \
  "$E39_IDENTITY" "$E39_MATH_MANIFEST" "$MATH_DATA_IDENTITY" <<'PY'
import csv
import hashlib
import json
import pathlib
import sys


def read_json(path: str) -> dict:
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))


def read_rows(path: str) -> list[dict[str, str]]:
    with pathlib.Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def sha256_file(path: str) -> str:
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


e37 = read_json(sys.argv[1])
expected_e37_evaluation = {
    "draws": 4,
    "k": 8,
    "pass_at_1": "greedy",
    "prompt": "neutral_qwen_boxed",
    "seeds": [370100, 370101, 370102, 370103],
    "step_zero_pairing": "exact",
}
if (
    e37.get("schema") != "e37_outcome_collision_05b_v1"
    or e37.get("arms") != ["grpo", "outcome_collision"]
    or e37.get("seeds") != [43, 44, 45]
    or e37.get("num_samples") != 16
    or e37.get("prompt_epochs") != 10
    or e37.get("outcome_collision_coefficient") != 0.10
    or e37.get("evaluation") != expected_e37_evaluation
):
    raise SystemExit("E37 identity does not match the frozen E41 comparator")

expected_e37_rows = {
    (arm, str(seed))
    for arm in ("grpo", "outcome_collision")
    for seed in (43, 44, 45)
}
for manifest_name in sys.argv[2:4]:
    rows = read_rows(manifest_name)
    observed = {(row["arm"], row["seed"]) for row in rows}
    if len(rows) != 6 or observed != expected_e37_rows:
        raise SystemExit(
            f"E37 comparator manifest is not the frozen six-run cohort: {manifest_name}"
        )

e38 = read_json(sys.argv[4])
e38_e37 = e38.get("e37_comparators", {})
expected_e38_evaluation = {
    "draws": 4,
    "k": 8,
    "pass_at_1": "greedy",
    "prompt": "neutral_qwen_boxed",
    "seeds": [370100, 370101, 370102, 370103],
    "step_zero_pairing_to_e37": "exact",
}
if (
    e38.get("schema") != "e38_semantic_shannon_05b_v1"
    or e38.get("new_arms") != ["semantic_shannon"]
    or e38.get("seeds") != [43, 44, 45]
    or e38.get("num_samples") != 16
    or e38.get("prompt_epochs") != 10
    or e38.get("evaluation") != expected_e38_evaluation
    or e38.get("semantic_shannon", {}).get("coefficient") != 0.10
    or e38.get("semantic_shannon", {}).get("surprisal_clip") != 5.0
    or e38.get("semantic_shannon", {}).get("pseudocount") != 1.0
    or e38_e37.get("identity_sha256") != sha256_file(sys.argv[1])
    or e38_e37.get("graph_manifest_sha256") != sha256_file(sys.argv[2])
    or e38_e37.get("countdown_manifest_sha256") != sha256_file(sys.argv[3])
):
    raise SystemExit("E38 identity does not match the frozen E41 comparator")

expected_e38_rows = {
    ("semantic_shannon", str(seed))
    for seed in (43, 44, 45)
}
for manifest_name in sys.argv[5:7]:
    rows = read_rows(manifest_name)
    observed = {(row["arm"], row["seed"]) for row in rows}
    if len(rows) != 3 or observed != expected_e38_rows:
        raise SystemExit(
            f"E38 comparator manifest is not the frozen three-run treatment: {manifest_name}"
        )

e39 = read_json(sys.argv[7])
expected_e39_evaluation = {
    "benchmark": "full_MATH_500",
    "count": 6,
    "draws": 1,
    "duplicate_terminal_policy_evaluation": "skip",
    "k": 8,
    "optimizer_step_interval": 768,
    "pass_at_1": "greedy",
    "passes": [0, 2, 4, 6, 8, 10],
    "prompt": "neutral_qwen_math",
    "prompt_interval": 768,
    "seeds": [390100],
    "sparse_cadence_exception": True,
    "step_zero_pairing": "exact_three_arm",
}
if (
    e39.get("schema") != "e39_math12k_384_semantic_entropy_05b_v1"
    or e39.get("arms") != ["grpo", "outcome_collision", "semantic_shannon"]
    or e39.get("seeds") != [43, 44, 45]
    or e39.get("num_samples") != 16
    or e39.get("prompt_epochs") != 10
    or e39.get("evaluation") != expected_e39_evaluation
    or e39.get("data", {}).get("root") != "var/data/math12k_384_math500"
    or e39.get("data", {}).get("train_rows") != 384
    or e39.get("data", {}).get("eval_rows") != 500
):
    raise SystemExit("E39 identity does not match the frozen E41 MATH comparator")

expected_e39_rows = {
    (arm, str(seed))
    for arm in ("grpo", "outcome_collision", "semantic_shannon")
    for seed in (43, 44, 45)
}
e39_rows = read_rows(sys.argv[8])
observed_e39 = {(row["arm"], row["seed"]) for row in e39_rows}
if len(e39_rows) != 9 or observed_e39 != expected_e39_rows:
    raise SystemExit("E39 comparator manifest is not the frozen nine-run cohort")

data_bytes = pathlib.Path(sys.argv[9]).read_bytes()
data_hash = hashlib.sha256(data_bytes).hexdigest()
if data_hash != e39.get("data", {}).get("materialization_manifest_sha256"):
    raise SystemExit("E41 MATH materialization identity differs from frozen E39")
PY

for prefix in "$GRAPH_PREFIX" "$COUNTDOWN_PREFIX" "$MATH_PREFIX"; do
  manifest="$ROOT_DIR/var/artifacts/${prefix}_comparative_jobs.tsv"
  if [[ "$phase" == full && -e "$manifest" ]]; then
    echo "Fresh E41 prefix required; manifest already exists: $manifest" >&2
    exit 1
  fi
done
if [[ "$phase" == full && -e "$IDENTITY" ]]; then
  echo "Fresh E41 identity required; artifact already exists: $IDENTITY" >&2
  exit 1
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
  local staging
  SOURCE_HASH="$(hash_tree "$ROOT_DIR/src")"
  SOURCE_ROOT="$ROOT_DIR/var/artifacts/source_snapshots/e41_semantic_shannon_advantage_05b_${SOURCE_HASH}/src"
  if [[ ! -f "$SOURCE_ROOT/oat_drgrpo/__init__.py" ]]; then
    mkdir -p "$(dirname "$SOURCE_ROOT")"
    staging="$(mktemp -d "$(dirname "$SOURCE_ROOT")/.source.XXXXXX")"
    mkdir -p "$staging/src"
    cp -a "$ROOT_DIR/src/." "$staging/src/"
    mv "$staging/src" "$SOURCE_ROOT"
    rmdir "$staging"
  fi
  if [[ "$(hash_tree "$SOURCE_ROOT")" != "$SOURCE_HASH" ]]; then
    echo "E41 source snapshot hash mismatch" >&2
    exit 1
  fi

  local ops_staging ops_input
  ops_input="$(mktemp -d "$ROOT_DIR/var/artifacts/source_snapshots/.e41-ops-input.XXXXXX")"
  mkdir -p "$ops_input/math500" "$ops_input/slurm"
  cp "$ROOT_DIR/ops/repo_env.sh" "$ops_input/repo_env.sh"
  cp "$ROOT_DIR/ops/run_experiment.sh" "$ops_input/run_experiment.sh"
  cp "$ROOT_DIR/ops/train.sh" "$ops_input/train.sh"
  cp "$ROOT_DIR/ops/resolve_eval_cadence.py" "$ops_input/resolve_eval_cadence.py"
  cp "$ROOT_DIR/ops/submit_countdown_comparative.sh" \
    "$ops_input/submit_countdown_comparative.sh"
  cp "$MATERIALIZER" "$ops_input/math500/materialize_e39_math12k_384.py"
  cp "$ROOT_DIR/ops/slurm/train_node302.slurm" \
    "$ops_input/slurm/train_node302.slurm"
  EXECUTION_HASH="$(hash_tree "$ops_input")"
  OPS_ROOT="$ROOT_DIR/var/artifacts/source_snapshots/e41_semantic_shannon_advantage_05b_ops_${EXECUTION_HASH}/ops"
  if [[ ! -f "$OPS_ROOT/submit_countdown_comparative.sh" ]]; then
    mkdir -p "$(dirname "$OPS_ROOT")"
    ops_staging="$(mktemp -d "$(dirname "$OPS_ROOT")/.ops.XXXXXX")"
    mv "$ops_input" "$ops_staging/ops"
    mv "$ops_staging/ops" "$OPS_ROOT"
    rmdir "$ops_staging"
  else
    find "$ops_input" -type f -delete
    rmdir "$ops_input/math500" "$ops_input/slurm" "$ops_input"
  fi
  if [[ "$(hash_tree "$OPS_ROOT")" != "$EXECUTION_HASH" ]]; then
    echo "E41 execution snapshot hash mismatch" >&2
    exit 1
  fi
}

if [[ "$phase" == full ]]; then
  freeze_execution
else
  SOURCE_HASH="$(hash_tree "$ROOT_DIR/src")"
  EXECUTION_HASH=config-only-current-tree
  SOURCE_ROOT="$ROOT_DIR/src"
  OPS_ROOT="$ROOT_DIR/ops"
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
export OAT_ZERO_ONLY_ARMS=semantic_shannon_advantage
export OAT_ZERO_XDR_TAUS=""
export OAT_ZERO_INCLUDE_TOKEN_ENTROPY_ARM=0
export OAT_ZERO_INCLUDE_SEED_ARM=0
export OAT_ZERO_INCLUDE_XDR_ADAPT_ARM=0
export OAT_ZERO_INCLUDE_XDR_TAU_CONTROL_ARM=0
export OAT_ZERO_INCLUDE_XDR_SAC_DUAL_ARM=0
export OAT_ZERO_INCLUDE_MAXENT_ARM=0
export OAT_ZERO_INCLUDE_MAXENT_CONTROL_ARM=0
export OAT_ZERO_INCLUDE_MAXENT_DUAL_ARM=0
export OAT_ZERO_INCLUDE_MAXENT_LENGTH_DUAL_ARM=0
export OAT_ZERO_INCLUDE_DIAYN_ARM=0
export OAT_ZERO_INCLUDE_OUTCOME_COLLISION_ARM=0
export OAT_ZERO_INCLUDE_OUTCOME_COLLISION_OUTSIDE_CENTERING_ARM=0
export OAT_ZERO_INCLUDE_SEMANTIC_SHANNON_ARM=0
export OAT_ZERO_INCLUDE_SEMANTIC_SHANNON_ADVANTAGE_ARM=1

export OAT_ZERO_OUTCOME_COLLISION_COEF=0
export OAT_ZERO_OUTCOME_COLLISION_OUTSIDE_CENTERING=0
export OAT_ZERO_SEMANTIC_SHANNON_COEF=0.10
export OAT_ZERO_SEMANTIC_SHANNON_SURPRISAL_CLIP=5.0
export OAT_ZERO_SEMANTIC_SHANNON_PSEUDOCOUNT=1.0
export OAT_ZERO_SEMANTIC_SHANNON_SEPARATE_ADVANTAGE=1
export OAT_ZERO_MAXENT_OBJECTIVE=sequence
export OAT_ZERO_MAXENT_ALPHA=0
export OAT_ZERO_POLICY_ENTROPY_COEF=0
export OAT_ZERO_SEED_ENTROPY_ALPHA=0
export OAT_ZERO_DIAYN_NUM_OPTIONS=0
export OAT_ZERO_DIAYN_MI_BETA=0
export OAT_ZERO_DIAYN_MI_LEAVE_ONE_OUT=0

export OAT_ZERO_NUM_SAMPLES=16
export OAT_ZERO_LEARNING_RATE=0.0000002
export OAT_ZERO_MAX_QUERIES=100000000
export OAT_ZERO_MAX_PROMPT_EPOCHS=10
export OAT_ZERO_NUM_PROMPT_EPOCH=10
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
export OAT_ZERO_EVAL_BATCH_SIZE=64
export OAT_ZERO_SYNC_PARAMS_EVERY=1

export OAT_ZERO_CANONICAL_ACTION_TASK=none
export OAT_ZERO_CANONICAL_GRAPH_ACTIONS=0
export OAT_ZERO_CANONICAL_GRAPH_LEARNER_SAMPLING=0
export OAT_ZERO_CANONICAL_GRAPH_FIXED_SHAPE_SAMPLING=0
export OAT_ZERO_REPLICATED_FREEFORM_SAMPLING=0
export OAT_ZERO_LOCAL_ACTOR_WEIGHT_SYNC=0
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

export OAT_ZERO_SAVE_CKPT=1
export OAT_ZERO_MAX_SAVE_NUM=2
export OAT_ZERO_EXPORT_STEPS=0
export OAT_ZERO_EXPORT_FROM=0
export OAT_ZERO_MAX_EXPORT_NUM=1
export OAT_ZERO_MAX_RESUME_NUM=2
export OAT_ZERO_PRUNE_RESUME_ON_SUCCESS=0
export OAT_ZERO_AUTO_RESUME=1
export OAT_ZERO_WATCHDOG_REQUEUE=1
export OAT_ZERO_WATCHDOG_MAX_RESTARTS=6

MODEBENCH_NODELIST="${OAT_ZERO_E41_MODEBENCH_TRAIN_NODELIST:-node105,node202,node203,node204}"
MODEBENCH_GRES="${OAT_ZERO_E41_MODEBENCH_TRAIN_GRES:-gpu:a5000:1}"
MODEBENCH_CPUS="${OAT_ZERO_E41_MODEBENCH_TRAIN_CPUS_PER_TASK:-4}"
MODEBENCH_MEMORY="${OAT_ZERO_E41_MODEBENCH_TRAIN_MEMORY:-32G}"
MATH_NODELIST="${OAT_ZERO_E41_MATH_TRAIN_NODELIST:-node105,node202,node203,node204}"
MATH_GRES="${OAT_ZERO_E41_MATH_TRAIN_GRES:-gpu:a5000:1}"
MATH_CPUS="${OAT_ZERO_E41_MATH_TRAIN_CPUS_PER_TASK:-8}"
MATH_MEMORY="${OAT_ZERO_E41_MATH_TRAIN_MEMORY:-64G}"
TRAIN_PARTITION="${OAT_ZERO_E41_TRAIN_PARTITION:-all}"
TRAIN_ACCOUNT="${OAT_ZERO_E41_TRAIN_ACCOUNT:-mltheory}"
TRAIN_TIME_LIMIT="${OAT_ZERO_E41_TRAIN_TIME_LIMIT:-24:00:00}"

write_identity() {
  local protocol_hash launcher_hash e37_identity_hash e37_graph_manifest_hash
  local e37_countdown_manifest_hash e38_identity_hash e38_graph_manifest_hash
  local e38_countdown_manifest_hash e39_identity_hash e39_manifest_hash
  local math_data_hash
  protocol_hash="$(sha256sum "$PROTOCOL" | cut -d' ' -f1)"
  launcher_hash="$(sha256sum "$0" | cut -d' ' -f1)"
  e37_identity_hash="$(sha256sum "$E37_IDENTITY" | cut -d' ' -f1)"
  e37_graph_manifest_hash="$(sha256sum "$E37_GRAPH_MANIFEST" | cut -d' ' -f1)"
  e37_countdown_manifest_hash="$(sha256sum "$E37_COUNTDOWN_MANIFEST" | cut -d' ' -f1)"
  e38_identity_hash="$(sha256sum "$E38_IDENTITY" | cut -d' ' -f1)"
  e38_graph_manifest_hash="$(sha256sum "$E38_GRAPH_MANIFEST" | cut -d' ' -f1)"
  e38_countdown_manifest_hash="$(sha256sum "$E38_COUNTDOWN_MANIFEST" | cut -d' ' -f1)"
  e39_identity_hash="$(sha256sum "$E39_IDENTITY" | cut -d' ' -f1)"
  e39_manifest_hash="$(sha256sum "$E39_MATH_MANIFEST" | cut -d' ' -f1)"
  math_data_hash="$(sha256sum "$MATH_DATA_IDENTITY" | cut -d' ' -f1)"
  "$PYTHON_BIN" - "$IDENTITY" "$protocol_hash" "$launcher_hash" \
    "$SOURCE_HASH" "$EXECUTION_HASH" "$e37_identity_hash" \
    "$e37_graph_manifest_hash" "$e37_countdown_manifest_hash" \
    "$e38_identity_hash" "$e38_graph_manifest_hash" \
    "$e38_countdown_manifest_hash" \
    "$e39_identity_hash" "$e39_manifest_hash" "$math_data_hash" <<'PY'
import json
import os
import pathlib
import sys
import tempfile

path = pathlib.Path(sys.argv[1])
payload = {
    "schema": "e41_semantic_shannon_advantage_05b_v1",
    "protocol_sha256": sys.argv[2],
    "launcher_sha256": sys.argv[3],
    "source_hash": sys.argv[4],
    "execution_surface_hash": sys.argv[5],
    "comparators": {
        "modebench_e37": {
            "experiment": "E37",
            "identity": "e37_outcome_collision_05b_v1_identity.json",
            "identity_sha256": sys.argv[6],
            "graph_prefix": "gce37_outcome_collision_05b_v1",
            "graph_manifest_sha256": sys.argv[7],
            "countdown_prefix": "cde37_outcome_collision_05b_v1",
            "countdown_manifest_sha256": sys.argv[8],
            "arms": ["grpo", "outcome_collision"],
        },
        "modebench_e38": {
            "experiment": "E38",
            "identity": "e38_semantic_shannon_05b_v1_identity.json",
            "identity_sha256": sys.argv[9],
            "graph_prefix": "gce38_semantic_shannon_05b_v1",
            "graph_manifest_sha256": sys.argv[10],
            "countdown_prefix": "cde38_semantic_shannon_05b_v1",
            "countdown_manifest_sha256": sys.argv[11],
            "arms": ["semantic_shannon"],
        },
        "math": {
            "experiment": "E39",
            "identity": "mte39_math12k_384_semantic_entropy_05b_v1_identity.json",
            "identity_sha256": sys.argv[12],
            "prefix": "mte39_math12k_384_semantic_entropy_05b_v1",
            "manifest_sha256": sys.argv[13],
            "arms_used": ["grpo", "outcome_collision", "semantic_shannon"],
        },
    },
    "model": (
        "Qwen2.5-0.5B-Instruct@"
        "7ae557604adf67be50417f59c2c2f167def9a775"
    ),
    "tasks": {
        "graph_coloring": {
            "prompt_pool": 192,
            "prefix": "gce41_semantic_shannon_advantage_05b_v1",
        },
        "countdown": {
            "prompt_pool": 384,
            "prefix": "cde41_semantic_shannon_advantage_05b_v1",
        },
        "math12k_384": {
            "prompt_pool": 384,
            "eval_rows": 500,
            "prefix": "mte41_math12k_384_semantic_shannon_advantage_05b_v1",
            "materialization_manifest_sha256": sys.argv[14],
        },
    },
    "new_arms": ["semantic_shannon_advantage"],
    "semantic_shannon_advantage": {
        "coefficient": 0.10,
        "surprisal_clip": 5.0,
        "pseudocount": 1.0,
        "separate_advantage": True,
        "centering": "detached_row_predictive_distribution_expectation",
        "application": "centered_task_advantage_plus_semantic_advantage",
        "group_leave_one_out": True,
        "history_update": "after_group_scoring",
        "invalid_outcome": "shared_INVALID",
    },
    "seeds": [43, 44, 45],
    "num_samples": 16,
    "prompt_epochs": 10,
    "evaluation": {
        "modebench": {
            "prompt": "neutral_qwen_boxed",
            "k": 8,
            "draws": 4,
            "seeds": [370100, 370101, 370102, 370103],
            "pass_at_1": "greedy",
            "step_zero_pairing_to_e37_e38": "exact",
        },
        "math": {
            "prompt": "neutral_qwen_math",
            "benchmark": "full_MATH_500",
            "passes": [0, 2, 4, 6, 8, 10],
            "count": 6,
            "prompt_interval": 768,
            "optimizer_step_interval": 768,
            "sparse_cadence_exception": True,
            "duplicate_terminal_policy_evaluation": "skip",
            "k": 8,
            "draws": 1,
            "seeds": [390100],
            "pass_at_1": "greedy",
            "step_zero_pairing_to_e39": "exact",
        },
    },
    "resume": {
        "semantic_history_checkpointed": True,
        "actor_sync_before_eval_or_rollout": True,
        "checkpoints_per_prompt_epoch": 1,
        "keep": 2,
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

submit_task() {
  local task="$1"
  case "$task" in
    graph_coloring)
      export RUN_STAMP_PREFIX="$GRAPH_PREFIX"
      export OAT_ZERO_COMPARATIVE_TASK=graph_coloring
      export OAT_ZERO_COMPARATIVE_DATA_ROOT="$GRAPH_DATA_ROOT"
      export OAT_ZERO_MAX_TRAIN=192
      export OAT_ZERO_PROMPT_TEMPLATE=qwen_boxed
      export OAT_ZERO_TEST_SPLIT=multi_answer
      export OAT_ZERO_VERIFIER_VERSION=fast
      export OAT_ZERO_PROMPT_MAX_LENGTH=256
      export OAT_ZERO_GENERATE_MAX_LENGTH=192
      export OAT_ZERO_EVAL_GENERATE_MAX_LENGTH=192
      export OAT_ZERO_MAX_MODEL_LEN=512
      export OAT_ZERO_EVAL_MODE_COVERAGE_DRAWS=4
      export OAT_ZERO_EVAL_MODE_COVERAGE_SEED=370100
      export OAT_ZERO_ALLOW_SPARSE_EVAL=0
      export OAT_ZERO_EVAL_PROMPT_INTERVAL=48
      export OAT_ZERO_EVAL_STEPS=64
      export OAT_ZERO_SAVE_STEPS=192
      export OAT_ZERO_SAVE_FROM=192
      export OAT_ZERO_RESUME_STEPS=192
      export OAT_ZERO_RESUME_FROM=192
      export OAT_ZERO_TRAIN_NODELIST="$MODEBENCH_NODELIST"
      export OAT_ZERO_TRAIN_GRES="$MODEBENCH_GRES"
      export OAT_ZERO_TRAIN_CPUS_PER_TASK="$MODEBENCH_CPUS"
      export OAT_ZERO_TRAIN_MEMORY="$MODEBENCH_MEMORY"
      ;;
    countdown)
      export RUN_STAMP_PREFIX="$COUNTDOWN_PREFIX"
      export OAT_ZERO_COMPARATIVE_TASK=countdown
      export OAT_ZERO_COMPARATIVE_DATA_PRESET=easy3
      export OAT_ZERO_COMPARATIVE_DATA_ROOT="$COUNTDOWN_DATA_ROOT"
      export OAT_ZERO_MAX_TRAIN=384
      export OAT_ZERO_PROMPT_TEMPLATE=qwen_boxed
      export OAT_ZERO_TEST_SPLIT=multi_answer
      export OAT_ZERO_VERIFIER_VERSION=fast
      export OAT_ZERO_PROMPT_MAX_LENGTH=256
      export OAT_ZERO_GENERATE_MAX_LENGTH=192
      export OAT_ZERO_EVAL_GENERATE_MAX_LENGTH=192
      export OAT_ZERO_MAX_MODEL_LEN=512
      export OAT_ZERO_EVAL_MODE_COVERAGE_DRAWS=4
      export OAT_ZERO_EVAL_MODE_COVERAGE_SEED=370100
      export OAT_ZERO_ALLOW_SPARSE_EVAL=0
      export OAT_ZERO_EVAL_PROMPT_INTERVAL=96
      export OAT_ZERO_EVAL_STEPS=64
      export OAT_ZERO_SAVE_STEPS=384
      export OAT_ZERO_SAVE_FROM=384
      export OAT_ZERO_RESUME_STEPS=384
      export OAT_ZERO_RESUME_FROM=384
      export OAT_ZERO_TRAIN_NODELIST="$MODEBENCH_NODELIST"
      export OAT_ZERO_TRAIN_GRES="$MODEBENCH_GRES"
      export OAT_ZERO_TRAIN_CPUS_PER_TASK="$MODEBENCH_CPUS"
      export OAT_ZERO_TRAIN_MEMORY="$MODEBENCH_MEMORY"
      ;;
    math)
      export RUN_STAMP_PREFIX="$MATH_PREFIX"
      export OAT_ZERO_COMPARATIVE_TASK=math
      export OAT_ZERO_COMPARATIVE_DATA_ROOT="$MATH_DATA_ROOT"
      export OAT_ZERO_MAX_TRAIN=384
      export OAT_ZERO_PROMPT_TEMPLATE=qwen_math
      export OAT_ZERO_TEST_SPLIT=math
      export OAT_ZERO_VERIFIER_VERSION=math_verify
      export OAT_ZERO_PROMPT_MAX_LENGTH=1024
      export OAT_ZERO_GENERATE_MAX_LENGTH=1024
      export OAT_ZERO_EVAL_GENERATE_MAX_LENGTH=1024
      export OAT_ZERO_MAX_MODEL_LEN=2048
      export OAT_ZERO_EVAL_MODE_COVERAGE_DRAWS=1
      export OAT_ZERO_EVAL_MODE_COVERAGE_SEED=390100
      export OAT_ZERO_ALLOW_SPARSE_EVAL=1
      export OAT_ZERO_EVAL_PROMPT_INTERVAL=768
      export OAT_ZERO_EVAL_STEPS=768
      export OAT_ZERO_SAVE_STEPS=384
      export OAT_ZERO_SAVE_FROM=384
      export OAT_ZERO_RESUME_STEPS=384
      export OAT_ZERO_RESUME_FROM=384
      export OAT_ZERO_TRAIN_NODELIST="$MATH_NODELIST"
      export OAT_ZERO_TRAIN_GRES="$MATH_GRES"
      export OAT_ZERO_TRAIN_CPUS_PER_TASK="$MATH_CPUS"
      export OAT_ZERO_TRAIN_MEMORY="$MATH_MEMORY"
      ;;
    *)
      echo "Unknown E41 task: $task" >&2
      exit 1
      ;;
  esac
  export OAT_ZERO_TRAIN_PARTITION="$TRAIN_PARTITION"
  export OAT_ZERO_TRAIN_ACCOUNT="$TRAIN_ACCOUNT"
  export OAT_ZERO_TRAIN_TIME_LIMIT="$TRAIN_TIME_LIMIT"
  "$OPS_ROOT/submit_countdown_comparative.sh"
}

if [[ "$phase" == config ]]; then
  export OAT_ZERO_COMPARATIVE_CONFIG_ONLY=1
  export OAT_ZERO_SBATCH_HOLD=0
  submit_task graph_coloring
  submit_task countdown
  submit_task math
  echo "[e41] all three treatment-only configurations passed; no jobs submitted"
  exit 0
fi

unset OAT_ZERO_COMPARATIVE_CONFIG_ONLY
export OAT_ZERO_SBATCH_HOLD=1

COHORT_RELEASED=0
cleanup_partial_cohort() {
  local status="$?"
  trap - EXIT
  if [[ "$COHORT_RELEASED" != "1" ]]; then
    local -a cleanup_ids=()
    local -a manifest_ids=()
    local manifest failure_stamp artifact
    for manifest in \
      "$ROOT_DIR/var/artifacts/${GRAPH_PREFIX}_comparative_jobs.tsv" \
      "$ROOT_DIR/var/artifacts/${COUNTDOWN_PREFIX}_comparative_jobs.tsv" \
      "$ROOT_DIR/var/artifacts/${MATH_PREFIX}_comparative_jobs.tsv"; do
      if [[ -f "$manifest" ]]; then
        mapfile -t manifest_ids < <(
          awk -F '\t' 'NR > 1 && $3 ~ /^[0-9]+$/ {print $3}' "$manifest"
        )
        cleanup_ids+=("${manifest_ids[@]}")
      fi
    done
    if [[ "${#cleanup_ids[@]}" -gt 0 ]]; then
      scancel "${cleanup_ids[@]}" 2>/dev/null || true
      echo "[e41] cancelled incomplete held cohort: ${cleanup_ids[*]}" >&2
    fi
    failure_stamp="$(date +%Y%m%d_%H%M%S)"
    for artifact in \
      "$ROOT_DIR/var/artifacts/${GRAPH_PREFIX}_comparative_jobs.tsv" \
      "$ROOT_DIR/var/artifacts/${COUNTDOWN_PREFIX}_comparative_jobs.tsv" \
      "$ROOT_DIR/var/artifacts/${MATH_PREFIX}_comparative_jobs.tsv" \
      "$IDENTITY"; do
      if [[ -e "$artifact" ]]; then
        mv "$artifact" "${artifact}.failed_${failure_stamp}"
      fi
    done
  fi
  exit "$status"
}
trap cleanup_partial_cohort EXIT

write_identity
export OAT_ZERO_PROTOCOL_IDENTITY="$IDENTITY"
submit_task graph_coloring
submit_task countdown
submit_task math

job_ids=()
for prefix in "$GRAPH_PREFIX" "$COUNTDOWN_PREFIX" "$MATH_PREFIX"; do
  manifest="$ROOT_DIR/var/artifacts/${prefix}_comparative_jobs.tsv"
  mapfile -t task_jobs < <(
    awk -F '\t' 'NR > 1 && $3 ~ /^[0-9]+$/ {print $3}' "$manifest"
  )
  if [[ "${#task_jobs[@]}" -ne "$EXPECTED_JOBS_PER_TASK" ]]; then
    echo "E41 ${prefix} has ${#task_jobs[@]} jobs; expected $EXPECTED_JOBS_PER_TASK" >&2
    exit 1
  fi
  job_ids+=("${task_jobs[@]}")
done
if [[ "${#job_ids[@]}" -ne "$EXPECTED_JOBS" ]]; then
  echo "E41 cohort has ${#job_ids[@]} jobs; expected $EXPECTED_JOBS" >&2
  exit 1
fi

"$PYTHON_BIN" - \
  "$ROOT_DIR/var/artifacts/${GRAPH_PREFIX}_comparative_jobs.tsv" \
  "$ROOT_DIR/var/artifacts/${COUNTDOWN_PREFIX}_comparative_jobs.tsv" \
  "$ROOT_DIR/var/artifacts/${MATH_PREFIX}_comparative_jobs.tsv" <<'PY'
import csv
import pathlib
import sys

expected = {
    ("semantic_shannon_advantage", str(seed))
    for seed in (43, 44, 45)
}
for manifest_name in sys.argv[1:]:
    with pathlib.Path(manifest_name).open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    observed = {(row["arm"], row["seed"]) for row in rows}
    if len(rows) != 3 or observed != expected:
        raise SystemExit(
            f"E41 held manifest mismatch: {manifest_name} "
            f"rows={len(rows)} observed={sorted(observed)}"
        )
PY

for job_id in "${job_ids[@]}"; do
  if ! scontrol update JobId="$job_id" Partition="$TRAIN_PARTITION"; then
    echo "E41 could not normalize held job $job_id to $TRAIN_PARTITION" >&2
    exit 1
  fi
done

for spec in \
  "$GRAPH_PREFIX|$GRAPH_DATA_ROOT|192|48|64|192|qwen_boxed|multi_answer|fast|256|192|512|4|370100|0|$MODEBENCH_NODELIST|$MODEBENCH_CPUS|$MODEBENCH_MEMORY" \
  "$COUNTDOWN_PREFIX|$COUNTDOWN_DATA_ROOT|384|96|64|384|qwen_boxed|multi_answer|fast|256|192|512|4|370100|0|$MODEBENCH_NODELIST|$MODEBENCH_CPUS|$MODEBENCH_MEMORY" \
  "$MATH_PREFIX|$MATH_DATA_ROOT|384|768|768|384|qwen_math|math|math_verify|1024|1024|2048|1|390100|1|$MATH_NODELIST|$MATH_CPUS|$MATH_MEMORY"; do
  IFS='|' read -r prefix data_root max_train eval_interval eval_steps \
    resume_steps prompt_template test_split verifier prompt_length \
    response_length model_length eval_draws eval_seed sparse_eval \
    nodelist cpus memory <<< "$spec"
  manifest="$ROOT_DIR/var/artifacts/${prefix}_comparative_jobs.tsv"
  mapfile -t task_jobs < <(
    awk -F '\t' 'NR > 1 && $3 ~ /^[0-9]+$/ {print $3}' "$manifest"
  )
  for job_id in "${task_jobs[@]}"; do
    arm="$(awk -F '\t' -v job_id="$job_id" '$3 == job_id {print $1}' "$manifest")"
    seed="$(awk -F '\t' -v job_id="$job_id" '$3 == job_id {print $2}' "$manifest")"
    run_stamp="${prefix}_${arm}_s${seed}"
    job_record="$(scontrol show job "$job_id" -o)"
    for required in \
      'JobState=PENDING' 'Reason=JobHeldUser' \
      "RUN_STAMP=${run_stamp}" \
      "OAT_ZERO_SEED=${seed}" \
      'OAT_ZERO_VARIANT=semantic_shannon_advantage' \
      'OAT_ZERO_MODEL=qwen2.5-0.5b-instruct' \
      "OAT_ZERO_PRETRAIN=${MODEL_ROOT}" \
      "OAT_ZERO_DATA_ROOT=${data_root}" \
      'OAT_ZERO_REQUIRE_EXISTING_DATA=1' \
      "OAT_ZERO_MAX_TRAIN=${max_train}" \
      'OAT_ZERO_MAX_QUERIES=100000000' \
      'OAT_ZERO_MAX_PROMPT_EPOCHS=10' \
      'OAT_ZERO_NUM_PROMPT_EPOCH=10' \
      'OAT_ZERO_NUM_SAMPLES=16' \
      'OAT_ZERO_TRAIN_BATCH_SIZE=16' \
      'OAT_ZERO_TRAIN_BATCH_SIZE_PER_DEVICE=1' \
      'OAT_ZERO_ROLLOUT_BATCH_SIZE=1' \
      'OAT_ZERO_ROLLOUT_BATCH_SIZE_PER_DEVICE=1' \
      'OAT_ZERO_N_GPU=1' \
      'OAT_ZERO_NUM_GPUS_PER_ACTOR=1' \
      'OAT_ZERO_LEARNING_RATE=0.0000002' \
      'OAT_ZERO_NUM_PPO_EPOCHS=1' \
      'OAT_ZERO_MAX_NORM=1' \
      'OAT_ZERO_BETA=0' \
      'OAT_ZERO_IGNORE_NO_EOS=0' \
      'OAT_ZERO_INPUT_KEY=problem' \
      'OAT_ZERO_OUTPUT_KEY=answer' \
      'OAT_ZERO_EVAL_INPUT_KEY=problem' \
      'OAT_ZERO_EVAL_OUTPUT_KEY=answer' \
      "OAT_ZERO_PROMPT_TEMPLATE=${prompt_template}" \
      "OAT_ZERO_TEST_SPLIT=${test_split}" \
      "OAT_ZERO_VERIFIER_VERSION=${verifier}" \
      "OAT_ZERO_PROMPT_MAX_LENGTH=${prompt_length}" \
      "OAT_ZERO_GENERATE_MAX_LENGTH=${response_length}" \
      "OAT_ZERO_EVAL_GENERATE_MAX_LENGTH=${response_length}" \
      "OAT_ZERO_MAX_MODEL_LEN=${model_length}" \
      'OAT_ZERO_TEMPERATURE=1' \
      'OAT_ZERO_TOP_P=1' \
      'OAT_ZERO_EVAL_TEMPERATURE=0' \
      'OAT_ZERO_EVAL_BATCH_SIZE=64' \
      'OAT_ZERO_SYNC_PARAMS_EVERY=1' \
      'OAT_ZERO_EVAL_MODE_COVERAGE_K=8' \
      'OAT_ZERO_EVAL_MODE_COVERAGE_TEMPERATURE=1' \
      "OAT_ZERO_EVAL_MODE_COVERAGE_DRAWS=${eval_draws}" \
      "OAT_ZERO_EVAL_MODE_COVERAGE_SEED=${eval_seed}" \
      "OAT_ZERO_ALLOW_SPARSE_EVAL=${sparse_eval}" \
      "OAT_ZERO_EVAL_PROMPT_INTERVAL=${eval_interval}" \
      "OAT_ZERO_EVAL_STEPS=${eval_steps}" \
      'OAT_ZERO_OUTCOME_COLLISION_COEF=0.0' \
      'OAT_ZERO_OUTCOME_COLLISION_OUTSIDE_CENTERING=0' \
      'OAT_ZERO_SEMANTIC_SHANNON_COEF=0.10' \
      'OAT_ZERO_SEMANTIC_SHANNON_SURPRISAL_CLIP=5.0' \
      'OAT_ZERO_SEMANTIC_SHANNON_PSEUDOCOUNT=1.0' \
      'OAT_ZERO_SEMANTIC_SHANNON_SEPARATE_ADVANTAGE=1' \
      'OAT_ZERO_DIAYN_NUM_OPTIONS=0' \
      'OAT_ZERO_DIAYN_MI_BETA=0.0' \
      'OAT_ZERO_DIAYN_MI_LEAVE_ONE_OUT=0' \
      'OAT_ZERO_USE_WB=0' \
      'OAT_ZERO_CANONICAL_ACTION_TASK=none' \
      'OAT_ZERO_CANONICAL_GRAPH_ACTIONS=0' \
      'OAT_ZERO_CANONICAL_GRAPH_LEARNER_SAMPLING=0' \
      'OAT_ZERO_CANONICAL_GRAPH_FIXED_SHAPE_SAMPLING=0' \
      'OAT_ZERO_REPLICATED_FREEFORM_SAMPLING=0' \
      'OAT_ZERO_LOCAL_ACTOR_WEIGHT_SYNC=0' \
      'OAT_ZERO_ZERO_STAGE=2' \
      'OAT_ZERO_VLLM_GPU_RATIO=0.25' \
      'OAT_ZERO_ENABLE_FLASH_ATTN=0' \
      'OAT_ZERO_ADAM_OFFLOAD=0' \
      'OAT_ZERO_ACTIVATION_OFFLOADING=0' \
      'OAT_ZERO_COLLOCATE=1' \
      'OAT_ZERO_VLLM_SLEEP=1' \
      'OAT_ZERO_VLLM_SLEEP_LEVEL=1' \
      'VLLM_USE_V1=0' \
      'HF_HUB_OFFLINE=1' \
      'TRANSFORMERS_OFFLINE=1' \
      'OAT_ZERO_RND_SEED=0' \
      'OAT_ZERO_XDR_MODE_ADAPTIVE=0' \
      'OAT_ZERO_XDR_TAU=inf' \
      'OAT_ZERO_XDR_TAU_CONTROL_TARGET_RATIO=0.0' \
      'OAT_ZERO_XDR_SAC_DUAL_TARGET_RATIO=0.0' \
      'OAT_ZERO_MAXENT_ALPHA=0.0' \
      'OAT_ZERO_MAXENT_OBJECTIVE=sequence' \
      'OAT_ZERO_MAXENT_CONTROL_TARGET_RATIO=0.0' \
      'OAT_ZERO_MAXENT_DUAL_TARGET_RATIO=0.0' \
      'OAT_ZERO_MAXENT_CONTROL_TARGET_ENTROPY=0.0' \
      'OAT_ZERO_MAXENT_DUAL_TARGET_ENTROPY=0.0' \
      'OAT_ZERO_XDR_TAU_CONTROL_RATIO=0.0' \
      'OAT_ZERO_XDR_SAC_DUAL_RATIO=0.0' \
      'OAT_ZERO_MAXENT_LENGTH_TARGET=0.0' \
      'OAT_ZERO_POLICY_ENTROPY_COEF=0.0' \
      'OAT_ZERO_SEED_ENTROPY_ALPHA=0.0' \
      'OAT_ZERO_SAVE_CKPT=1' \
      "OAT_ZERO_SAVE_STEPS=${resume_steps}" \
      "OAT_ZERO_SAVE_FROM=${resume_steps}" \
      'OAT_ZERO_MAX_SAVE_NUM=2' \
      'OAT_ZERO_AUTO_RESUME=1' \
      'OAT_ZERO_WATCHDOG_REQUEUE=1' \
      'OAT_ZERO_WATCHDOG_MAX_RESTARTS=6' \
      "OAT_ZERO_RESUME_STEPS=${resume_steps}" \
      "OAT_ZERO_RESUME_FROM=${resume_steps}" \
      'OAT_ZERO_MAX_RESUME_NUM=2' \
      'OAT_ZERO_PRUNE_RESUME_ON_SUCCESS=0' \
      'OAT_ZERO_EXPORT_STEPS=0' \
      'OAT_ZERO_EXPORT_FROM=0' \
      'OAT_ZERO_MAX_EXPORT_NUM=1' \
      "OAT_ZERO_SOURCE_ROOT=${SOURCE_ROOT}" \
      "OAT_ZERO_OPS_SNAPSHOT_ROOT=${OPS_ROOT}" \
      "OAT_ZERO_PROTOCOL_IDENTITY=${IDENTITY}"; do
      if [[ "$job_record" != *"$required"* ]]; then
        echo "E41 held-job audit failed for $job_id: missing $required" >&2
        exit 1
      fi
    done
    if [[ "$arm" != semantic_shannon_advantage ]]; then
      echo "E41 manifest has unexpected arm for $job_id: $arm" >&2
      exit 1
    fi
    if [[ "$job_record" != *"Partition=${TRAIN_PARTITION}"* ]] || \
       [[ "$job_record" != *"Account=${TRAIN_ACCOUNT}"* ]] || \
       [[ "$job_record" != *"--nodelist=${nodelist}"* ]] || \
       [[ "$job_record" != *"gres/gpu:a5000=1"* ]] || \
       [[ "$job_record" != *"NumNodes=1"* ]] || \
       [[ "$job_record" != *"NumTasks=1"* ]] || \
       [[ "$job_record" != *"NumCPUs=${cpus}"* ]] || \
       [[ "$job_record" != *"MinMemoryNode=${memory}"* ]] || \
       [[ "$job_record" != *"--time=${TRAIN_TIME_LIMIT}"* ]]; then
      echo "E41 held job $job_id does not attest the frozen placement" >&2
      exit 1
    fi
  done
done

scontrol release "${job_ids[@]}"
COHORT_RELEASED=1
trap - EXIT
echo "[e41] released ${#job_ids[@]} separately centered semantic-Shannon jobs: ${job_ids[*]}"
echo "[e41] identity=$IDENTITY math_data=$MATH_DATA_IDENTITY"
