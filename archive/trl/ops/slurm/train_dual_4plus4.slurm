#!/bin/bash
# Single-node experiment launcher:
# - GRPO + entropy-MaxEnt concurrently on one 8-GPU node, or
# - single-stack GRPO / entropy-MaxEnt / listwise-MaxEnt / SEED-GRPO on 4 GPUs.
#
# The default recipe profile renders the explicit three-way experiment presets
# onto the requested base recipe so the Slurm path matches the documented
# GRPO / MaxEnt / listwise-MaxEnt comparison.

#SBATCH --job-name=open_r1_dual_4plus4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --partition=mltheory
#SBATCH --account=mltheory
#SBATCH --time=48:00:00
#SBATCH --output=var/artifacts/logs/%x-%j.out
#SBATCH --error=var/artifacts/logs/%x-%j.err
#SBATCH --export=ALL,ENV_MODE=venv,ENV_ACTIVATE=/n/fs/similarity/maxent-grpo/var/openr1/bin/activate

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$ROOT_DIR"

VAR_DIR="${VAR_DIR:-$ROOT_DIR/var}"
mkdir -p "$VAR_DIR/artifacts/logs" "$VAR_DIR/tmp"

CACHE_ROOT="${CACHE_ROOT:-$VAR_DIR/cache}"
mkdir -p \
  "$CACHE_ROOT/xdg" \
  "$CACHE_ROOT/hf" \
  "$CACHE_ROOT/tmp" \
  "$CACHE_ROOT/triton" \
  "$CACHE_ROOT/torchinductor"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$CACHE_ROOT/xdg}"
export HF_HOME="${HF_HOME:-$CACHE_ROOT/hf}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$XDG_CACHE_HOME/huggingface/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$XDG_CACHE_HOME/huggingface/datasets}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$CACHE_ROOT/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$CACHE_ROOT/torchinductor}"
export TMPDIR="${TMPDIR:-$CACHE_ROOT/tmp}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export VLLM_NO_USAGE_STATS="${VLLM_NO_USAGE_STATS:-1}"

WANDB_ROOT="${WANDB_ROOT:-$VAR_DIR/wandb}"
mkdir -p \
  "$WANDB_ROOT/runs" \
  "$WANDB_ROOT/data" \
  "$WANDB_ROOT/cache" \
  "$WANDB_ROOT/artifacts"
export WANDB_DIR="${WANDB_DIR:-$WANDB_ROOT/runs}"
export WANDB_DATA_DIR="${WANDB_DATA_DIR:-$WANDB_ROOT/data}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$WANDB_ROOT/cache}"
export WANDB_ARTIFACT_DIR="${WANDB_ARTIFACT_DIR:-$WANDB_ROOT/artifacts}"
export WANDB__SERVICE_WAIT="${WANDB__SERVICE_WAIT:-300}"
export WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-300}"
export WANDB_START_METHOD="${WANDB_START_METHOD:-thread}"

ENV_MODE="${ENV_MODE:-venv}"
CONDA_SH="${CONDA_SH:-/usr/local/anaconda3/2024.02/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-/n/fs/similarity/.conda/envs/openr1}"
ENV_ACTIVATE="${ENV_ACTIVATE:-$VAR_DIR/openr1/bin/activate}"
VIRTUAL_ENV="${VIRTUAL_ENV:-$VAR_DIR/openr1}"
PYTHON_OVERLAY_DIR="${PYTHON_OVERLAY_DIR:-$VAR_DIR/python_overlay}"
DATASET_SITE_PACKAGES="${DATASET_SITE_PACKAGES:-/n/fs/similarity/.conda/envs/openr1/lib/python3.11/site-packages}"
REPO_RUNTIME_SITE_PACKAGES="${REPO_RUNTIME_SITE_PACKAGES:-$VAR_DIR/e2e-venv/lib/python3.11/site-packages:$VAR_DIR/seed_paper_eval/paper_venv/lib/python3.11/site-packages}"
EXTRA_SITE_PACKAGES="${EXTRA_SITE_PACKAGES:-$REPO_RUNTIME_SITE_PACKAGES}"

prepare_python_overlay() {
  mkdir -p "$PYTHON_OVERLAY_DIR"
  if [[ -d "$DATASET_SITE_PACKAGES/datasets" ]]; then
    ln -sfn "$DATASET_SITE_PACKAGES/datasets" "$PYTHON_OVERLAY_DIR/datasets"
  fi
  if [[ -d "$DATASET_SITE_PACKAGES/pyarrow" ]]; then
    ln -sfn "$DATASET_SITE_PACKAGES/pyarrow" "$PYTHON_OVERLAY_DIR/pyarrow"
  fi
}

discover_runtime_site_packages() {
  MAXENT_RUNTIME_PREFIX="$VAR_DIR" PYTHONNOUSERSITE="" python - <<'PY'
import os
import sys

required_modules = ("trl", "vllm")
repo_prefix = os.path.abspath(os.environ.get("MAXENT_RUNTIME_PREFIX", ""))
seen = set()
paths = []
for raw in sys.path:
    path = os.path.abspath(raw or ".")
    if path in seen or not os.path.isdir(path):
        continue
    seen.add(path)
    if repo_prefix and not path.startswith(repo_prefix):
        continue
    if any(os.path.isdir(os.path.join(path, mod)) for mod in required_modules):
        paths.append(path)
print(":".join(paths))
PY
}

has_runtime_modules() {
  MAXENT_RUNTIME_EXTRA_PATHS="${1:-}" PYTHONNOUSERSITE=1 python - <<'PY'
import importlib.util
import os
import sys

extra_paths = [
    os.path.abspath(path)
    for path in os.environ.get("MAXENT_RUNTIME_EXTRA_PATHS", "").split(":")
    if path
]
for path in reversed(extra_paths):
    if os.path.isdir(path) and path not in sys.path:
        sys.path.insert(0, path)
required_modules = ("trl", "vllm")
missing = [name for name in required_modules if importlib.util.find_spec(name) is None]
raise SystemExit(1 if missing else 0)
PY
}

activate_env() {
  if [[ "$ENV_MODE" == "conda" ]]; then
    if [[ -f "$CONDA_SH" ]]; then
      # shellcheck disable=SC1090
      source "$CONDA_SH"
    fi
    conda activate "$CONDA_ENV"
  elif [[ -f "$ENV_ACTIVATE" ]]; then
    # shellcheck disable=SC1090
    source "$ENV_ACTIVATE"
  fi
  if [[ "$ENV_MODE" == "conda" ]]; then
    export VIRTUAL_ENV="${CONDA_PREFIX:-$CONDA_ENV}"
  else
    export VIRTUAL_ENV
    export PATH="$VIRTUAL_ENV/bin:${PATH:-}"
  fi
  local resolved_extra_paths="${EXTRA_SITE_PACKAGES:-}"
  local discovered_runtime_paths=""
  if ! has_runtime_modules "$resolved_extra_paths"; then
    discovered_runtime_paths="$(discover_runtime_site_packages || true)"
  fi
  export PYTHONNOUSERSITE=1
  prepare_python_overlay
  if [[ -n "$discovered_runtime_paths" ]]; then
    resolved_extra_paths="${discovered_runtime_paths}${resolved_extra_paths:+:${resolved_extra_paths}}"
  fi
  if [[ -d "$PYTHON_OVERLAY_DIR" ]]; then
    resolved_extra_paths="${PYTHON_OVERLAY_DIR}${resolved_extra_paths:+:${resolved_extra_paths}}"
  fi
  if [[ -n "$resolved_extra_paths" ]]; then
    if [[ -n "${PYTHONPATH:-}" ]]; then
      export PYTHONPATH="${resolved_extra_paths}:${PYTHONPATH}"
    else
      export PYTHONPATH="${resolved_extra_paths}"
    fi
  fi
}

to_abs_path() {
  local p="$1"
  if command -v realpath >/dev/null 2>&1; then
    realpath "$p"
  elif command -v readlink >/dev/null 2>&1; then
    readlink -f "$p" 2>/dev/null || printf '%s\n' "$p"
  else
    printf '%s\n' "$p"
  fi
}

yaml_scalar() {
  local key="$1"
  local file="$2"
  awk -F': *' -v k="$key" '$1 ~ "^[[:space:]]*"k"[[:space:]]*$" {print $2; exit}' "$file" | tr -d '"' | tr -d "'" || true
}

infer_max_model_len() {
  local cfg="$1"
  local p c
  p="$(yaml_scalar max_prompt_length "$cfg")"
  c="$(yaml_scalar max_completion_length "$cfg")"
  if [[ "$p" =~ ^[0-9]+$ && "$c" =~ ^[0-9]+$ ]]; then
    echo $((p + c))
  elif [[ "$p" =~ ^[0-9]+$ ]]; then
    echo "$p"
  else
    echo 4096
  fi
}

csv_count() {
  local raw="${1//[[:space:]]/}"
  if [[ -z "$raw" ]]; then
    echo 0
    return 0
  fi
  local n=0 part
  IFS=',' read -r -a parts <<< "$raw"
  for part in "${parts[@]}"; do
    [[ -z "$part" ]] && continue
    [[ "$part" =~ ^[0-9]+$ ]] || return 1
    n=$((n + 1))
  done
  echo "$n"
}

assert_csv_ids() {
  local name="$1"
  local raw="${2//[[:space:]]/}"
  local max_id="$3"
  local part
  declare -A seen=()
  IFS=',' read -r -a parts <<< "$raw"
  for part in "${parts[@]}"; do
    [[ "$part" =~ ^[0-9]+$ ]] || { echo "Invalid ${name} id: ${part}" >&2; return 1; }
    (( part >= 0 && part <= max_id )) || { echo "${name} id ${part} out of range [0,${max_id}]" >&2; return 1; }
    [[ -z "${seen[$part]:-}" ]] || { echo "Duplicate ${name} id: ${part}" >&2; return 1; }
    seen[$part]=1
  done
}

contains_id() {
  local csv="${1//[[:space:]]/}"
  local id="$2"
  [[ ",$csv," == *",$id,"* ]]
}

append_flag_value() {
  local args="$1"
  local flag="$2"
  local value="$3"
  if [[ " $args " == *" ${flag} "* || "$args" == *"${flag}="* ]]; then
    printf '%s\n' "$args"
  else
    if [[ -n "$args" ]]; then
      printf '%s %s %s\n' "$args" "$flag" "$value"
    else
      printf '%s %s\n' "$flag" "$value"
    fi
  fi
}

is_truthy_flag() {
  local raw
  raw="$(printf '%s' "${1:-}" | tr '[:upper:]' '[:lower:]')"
  [[ "$raw" == "1" || "$raw" == "true" || "$raw" == "yes" || "$raw" == "on" ]]
}

default_model_for_suffix() {
  local suffix="$1"
  case "$suffix" in
    math|math_fair|math_stable) printf '%s\n' "Qwen2.5-1.5B-Instruct" ;;
    code_mbpp) printf '%s\n' "Qwen2.5-0.5B-Instruct" ;;
    *) printf '%s\n' "Qwen2.5-1.5B-Instruct" ;;
  esac
}

default_output_dir_for_stack() {
  local model="$1"
  local stack="$2"
  local suffix="$3"
  local suffix_slug="${suffix//_/-}"
  local label=""
  case "$stack" in
    grpo) label="GRPO-BASELINE" ;;
    maxent) label="MaxEnt-Entropy" ;;
    listwise) label="MaxEnt-Listwise" ;;
    seed) label="SEED-GRPO" ;;
    *) echo "Unknown stack for output dir: $stack" >&2; return 1 ;;
  esac
  printf 'var/data/%s-Open-R1-%s-%s-v1\n' "$model" "$label" "$suffix_slug"
}

render_experiment_recipe() {
  local base_recipe="$1"
  local preset_path="$2"
  local output_dir="$3"
  local dst="$4"
  python - "$base_recipe" "$preset_path" "$output_dir" "$dst" <<'PY'
import os
import sys

base_recipe, preset_path, output_dir, dst = sys.argv[1:5]

try:
    from omegaconf import OmegaConf
except Exception:
    OmegaConf = None

try:
    import yaml
except Exception:
    yaml = None


def _load(path: str):
    if OmegaConf is not None:
        return OmegaConf.to_container(OmegaConf.load(path), resolve=False)
    if yaml is None:
        raise SystemExit("Missing omegaconf/yaml; cannot render experiment recipe")
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _save(payload, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if OmegaConf is not None:
        OmegaConf.save(config=OmegaConf.create(payload), f=path)
        return
    if yaml is None:
        raise SystemExit("Missing omegaconf/yaml; cannot save experiment recipe")
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


base_cfg = _load(base_recipe) or {}
preset_cfg = _load(preset_path) or {}
command_block = preset_cfg.get("maxent") or preset_cfg.get("baseline") or {}
for section in ("script", "training", "model"):
    overrides = command_block.get(section) or {}
    if isinstance(overrides, dict):
        base_cfg.update(overrides)
base_cfg["output_dir"] = output_dir
_save(base_cfg, dst)
PY
}

resolve_profile_preset() {
  local stack="$1"
  local suffix="$2"
  case "$stack" in
    grpo) printf 'configs/recipes/hydra/grpo_custom_%s.yaml\n' "$suffix" ;;
    maxent) printf 'configs/recipes/hydra/maxent_entropy_%s.yaml\n' "$suffix" ;;
    listwise) printf 'configs/recipes/hydra/maxent_listwise_%s.yaml\n' "$suffix" ;;
    seed) printf 'configs/recipes/hydra/seed_grpo_%s.yaml\n' "$suffix" ;;
    *) echo "Unknown experiment stack: $stack" >&2; return 1 ;;
  esac
}

resolve_accelerator_name() {
  local requested="$1"
  if [[ "$requested" == "zero3" ]]; then
    if ! PYTHONNOUSERSITE=1 python -c "import deepspeed" >/dev/null 2>&1; then
      printf '%s\n' "ddp"
      return 0
    fi
  fi
  printf '%s\n' "$requested"
}

if [[ "$*" == *"--help"* ]]; then
  cat <<'USAGE'
Usage: sbatch archive/trl/ops/slurm/train_dual_4plus4.slurm [options]

Options:
  --model MODEL               Recipe model directory (default depends on --config)
  --config SUFFIX             Config suffix for both GRPO and MaxEnt (default: math)
  --grpo-config SUFFIX        Optional GRPO config suffix override
  --maxent-config SUFFIX      Optional MaxEnt config suffix override
  --recipe-profile MODE       experiment | paired (default: experiment)
  --accelerator NAME          Accelerate config name (default: zero3)
  --grpo-args "ARGS"          Extra GRPO args
  --maxent-args "ARGS"        Extra secondary-stack args (used for maxent/listwise/seed)
  --run-only MODE             both | grpo | maxent | listwise | seed (default: both)
USAGE
  exit 0
fi

CONFIG_SUFFIX="${CONFIG_SUFFIX:-math}"
CONFIG_SUFFIX_GRPO="${CONFIG_SUFFIX_GRPO:-$CONFIG_SUFFIX}"
CONFIG_SUFFIX_MAXENT="${CONFIG_SUFFIX_MAXENT:-$CONFIG_SUFFIX}"
MODEL_EXPLICIT=0
MODEL="${MODEL:-}"
RECIPE_PROFILE="$(echo "${RECIPE_PROFILE:-experiment}" | tr '[:upper:]' '[:lower:]')"
ACCELERATOR="${ACCELERATOR:-zero3}"
GRPO_ARGS="${GRPO_ARGS:-}"
MAXENT_ARGS="${MAXENT_ARGS:-}"
RUN_ONLY="$(echo "${RUN_ONLY:-both}" | tr '[:upper:]' '[:lower:]')"

GRPO_VLLM_GPU="${GRPO_VLLM_GPU:-0}"
GRPO_TRAIN_GPUS="${GRPO_TRAIN_GPUS:-1,2,3}"
MAXENT_VLLM_GPU="${MAXENT_VLLM_GPU:-4}"
MAXENT_TRAIN_GPUS="${MAXENT_TRAIN_GPUS:-5,6,7}"

JOB_PORT_SEED="${SLURM_JOB_ID:-0}"
if [[ "$JOB_PORT_SEED" =~ ^[0-9]+$ ]]; then
  JOB_PORT_OFFSET=$(( (JOB_PORT_SEED % 1000) * 10 ))
else
  JOB_PORT_OFFSET=0
fi
GRPO_VLLM_PORT_DEFAULT=$((18000 + JOB_PORT_OFFSET))
MAXENT_VLLM_PORT_DEFAULT=$((18001 + JOB_PORT_OFFSET))
GRPO_GROUP_PORT_DEFAULT=$((28000 + JOB_PORT_OFFSET))
MAXENT_GROUP_PORT_DEFAULT=$((28001 + JOB_PORT_OFFSET))
GRPO_MASTER_PORT_DEFAULT=$((38000 + JOB_PORT_OFFSET))
MAXENT_MASTER_PORT_DEFAULT=$((38001 + JOB_PORT_OFFSET))

GRPO_VLLM_PORT="${GRPO_VLLM_PORT:-$GRPO_VLLM_PORT_DEFAULT}"
MAXENT_VLLM_PORT="${MAXENT_VLLM_PORT:-$MAXENT_VLLM_PORT_DEFAULT}"
GRPO_GROUP_PORT="${GRPO_GROUP_PORT:-$GRPO_GROUP_PORT_DEFAULT}"
MAXENT_GROUP_PORT="${MAXENT_GROUP_PORT:-$MAXENT_GROUP_PORT_DEFAULT}"
GRPO_MASTER_PORT="${GRPO_MASTER_PORT:-$GRPO_MASTER_PORT_DEFAULT}"
MAXENT_MASTER_PORT="${MAXENT_MASTER_PORT:-$MAXENT_MASTER_PORT_DEFAULT}"
VLLM_INIT_COMMUNICATOR_TIMEOUT="${VLLM_INIT_COMMUNICATOR_TIMEOUT:-300}"

# Prefer the async TRL vLLM communicator patch to avoid startup handshake stalls.
MAXENT_TRL_VLLM_CLIENT_PATCH="${MAXENT_TRL_VLLM_CLIENT_PATCH:-1}"
MAXENT_VLLM_ASYNC_INIT="${MAXENT_VLLM_ASYNC_INIT:-auto}"
MAXENT_VLLM_INIT_RETRIES="${MAXENT_VLLM_INIT_RETRIES:-3}"
MAXENT_VLLM_INIT_RETRY_BACKOFF_S="${MAXENT_VLLM_INIT_RETRY_BACKOFF_S:-2}"
MAXENT_VLLM_INIT_TIMEOUT_S="${MAXENT_VLLM_INIT_TIMEOUT_S:-300}"
MAXENT_VLLM_TOKEN_BOUNDARY_SMOKE="${MAXENT_VLLM_TOKEN_BOUNDARY_SMOKE:-1}"

VLLM_DTYPE="${VLLM_DTYPE:-bfloat16}"
VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-false}"
# vLLM cold starts (weight load + compile cache warmup) can exceed 8 minutes on first run.
# Default to a longer health window: 900 * 2s = 30 minutes.
VLLM_HEALTH_MAX_ATTEMPTS="${VLLM_HEALTH_MAX_ATTEMPTS:-900}"
VLLM_HEALTH_INTERVAL_S="${VLLM_HEALTH_INTERVAL_S:-2}"
ACC_TEE="${ACC_TEE:-3}"
TRAIN_NUM_PROCESSES="${TRAIN_NUM_PROCESSES:-3}"

if ! [[ "$TRAIN_NUM_PROCESSES" =~ ^[0-9]+$ ]] || (( TRAIN_NUM_PROCESSES < 1 )); then
  echo "TRAIN_NUM_PROCESSES must be a positive integer (got: $TRAIN_NUM_PROCESSES)" >&2
  exit 1
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; MODEL_EXPLICIT=1; shift 2 ;;
    --config) CONFIG_SUFFIX="$2"; CONFIG_SUFFIX_GRPO="$2"; CONFIG_SUFFIX_MAXENT="$2"; shift 2 ;;
    --grpo-config) CONFIG_SUFFIX_GRPO="$2"; shift 2 ;;
    --maxent-config) CONFIG_SUFFIX_MAXENT="$2"; shift 2 ;;
    --recipe-profile) RECIPE_PROFILE="$(echo "$2" | tr '[:upper:]' '[:lower:]')"; shift 2 ;;
    --accelerator) ACCELERATOR="$2"; shift 2 ;;
    --grpo-args) GRPO_ARGS="$2"; shift 2 ;;
    --maxent-args) MAXENT_ARGS="$2"; shift 2 ;;
    --run-only) RUN_ONLY="$(echo "$2" | tr '[:upper:]' '[:lower:]')"; shift 2 ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

if (( ! MODEL_EXPLICIT )); then
  MODEL="$(default_model_for_suffix "$CONFIG_SUFFIX")"
fi

if [[ "$RECIPE_PROFILE" != "experiment" && "$RECIPE_PROFILE" != "paired" ]]; then
  echo "RECIPE_PROFILE must be experiment|paired" >&2
  exit 1
fi

if [[ "$RUN_ONLY" != "both" && "$RUN_ONLY" != "grpo" && "$RUN_ONLY" != "maxent" && "$RUN_ONLY" != "listwise" && "$RUN_ONLY" != "seed" ]]; then
  echo "RUN_ONLY must be both|grpo|maxent|listwise|seed" >&2
  exit 1
fi

USE_GRPO_STACK=0
USE_SECONDARY_STACK=0
if [[ "$RUN_ONLY" == "both" || "$RUN_ONLY" == "grpo" ]]; then
  USE_GRPO_STACK=1
fi
if [[ "$RUN_ONLY" == "both" || "$RUN_ONLY" == "maxent" || "$RUN_ONLY" == "listwise" || "$RUN_ONLY" == "seed" ]]; then
  USE_SECONDARY_STACK=1
fi

SECONDARY_STACK_TAG="maxent"
SECONDARY_STACK_LABEL="MaxEnt"
if [[ "$RUN_ONLY" == "listwise" ]]; then
  SECONDARY_STACK_TAG="listwise"
  SECONDARY_STACK_LABEL="Listwise MaxEnt"
elif [[ "$RUN_ONLY" == "seed" ]]; then
  SECONDARY_STACK_TAG="seed"
  SECONDARY_STACK_LABEL="SEED-GRPO"
fi

# In single-stack mode, default to a 4-GPU local layout (vLLM:0, train:1,2,3).
# This allows grpo/maxent/listwise single-stack jobs to run in a 4-GPU allocation.
if [[ "$RUN_ONLY" == "maxent" || "$RUN_ONLY" == "listwise" || "$RUN_ONLY" == "seed" ]]; then
  if [[ "$MAXENT_VLLM_GPU" == "4" ]]; then MAXENT_VLLM_GPU="0"; fi
  if [[ "$MAXENT_TRAIN_GPUS" == "5,6,7" ]]; then MAXENT_TRAIN_GPUS="1,2,3"; fi
fi
if [[ "$RUN_ONLY" == "grpo" ]]; then
  if [[ "$GRPO_VLLM_GPU" == "4" ]]; then GRPO_VLLM_GPU="0"; fi
  if [[ "$GRPO_TRAIN_GPUS" == "5,6,7" ]]; then GRPO_TRAIN_GPUS="1,2,3"; fi
fi

RECIPE_RENDER_DIR="$VAR_DIR/artifacts/rendered_recipes/${SLURM_JOB_ID:-manual}"
mkdir -p "$RECIPE_RENDER_DIR"

GRPO_CFG=""
MAXENT_CFG=""
if [[ "$RECIPE_PROFILE" == "experiment" ]]; then
  if (( USE_GRPO_STACK )); then
    GRPO_BASE_CFG="configs/recipes/${MODEL}/grpo/config_${CONFIG_SUFFIX_GRPO}.yaml"
    GRPO_PRESET_CFG="$(resolve_profile_preset grpo "$CONFIG_SUFFIX_GRPO")"
    [[ -f "$GRPO_BASE_CFG" ]] || { echo "Missing base config: $GRPO_BASE_CFG" >&2; exit 1; }
    [[ -f "$GRPO_PRESET_CFG" ]] || { echo "Missing experiment preset: $GRPO_PRESET_CFG" >&2; exit 1; }
    GRPO_CFG="$RECIPE_RENDER_DIR/grpo_${CONFIG_SUFFIX_GRPO}.yaml"
    render_experiment_recipe \
      "$GRPO_BASE_CFG" \
      "$GRPO_PRESET_CFG" \
      "$(default_output_dir_for_stack "$MODEL" "grpo" "$CONFIG_SUFFIX_GRPO")" \
      "$GRPO_CFG"
  fi
  if (( USE_SECONDARY_STACK )); then
    MAXENT_BASE_CFG="configs/recipes/${MODEL}/grpo/config_${CONFIG_SUFFIX_MAXENT}.yaml"
    MAXENT_PRESET_CFG="$(resolve_profile_preset "$SECONDARY_STACK_TAG" "$CONFIG_SUFFIX_MAXENT")"
    [[ -f "$MAXENT_BASE_CFG" ]] || { echo "Missing base config: $MAXENT_BASE_CFG" >&2; exit 1; }
    [[ -f "$MAXENT_PRESET_CFG" ]] || { echo "Missing experiment preset: $MAXENT_PRESET_CFG" >&2; exit 1; }
    MAXENT_CFG="$RECIPE_RENDER_DIR/${SECONDARY_STACK_TAG}_${CONFIG_SUFFIX_MAXENT}.yaml"
    render_experiment_recipe \
      "$MAXENT_BASE_CFG" \
      "$MAXENT_PRESET_CFG" \
      "$(default_output_dir_for_stack "$MODEL" "$SECONDARY_STACK_TAG" "$CONFIG_SUFFIX_MAXENT")" \
      "$MAXENT_CFG"
  fi
else
  if (( USE_GRPO_STACK )); then
    GRPO_CFG="configs/recipes/${MODEL}/grpo/config_${CONFIG_SUFFIX_GRPO}.yaml"
    [[ -f "$GRPO_CFG" ]] || { echo "Missing config: $GRPO_CFG" >&2; exit 1; }
  fi
  if (( USE_SECONDARY_STACK )); then
    MAXENT_CFG="configs/recipes/${MODEL}/maxent-grpo/config_${CONFIG_SUFFIX_MAXENT}.yaml"
    [[ -f "$MAXENT_CFG" ]] || { echo "Missing config: $MAXENT_CFG" >&2; exit 1; }
  fi
fi

GRPO_CFG_ABS=""
MAXENT_CFG_ABS=""
if [[ -n "$GRPO_CFG" ]]; then
  GRPO_CFG_ABS="$(to_abs_path "$GRPO_CFG")"
fi
if [[ -n "$MAXENT_CFG" ]]; then
  MAXENT_CFG_ABS="$(to_abs_path "$MAXENT_CFG")"
fi

GPU_COUNT=8
if command -v nvidia-smi >/dev/null 2>&1; then
  detected="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
  if [[ "$detected" =~ ^[0-9]+$ ]] && (( detected > 0 )); then
    GPU_COUNT="$detected"
  fi
fi
MIN_REQUIRED_GPUS=8
if [[ "$RUN_ONLY" == "grpo" || "$RUN_ONLY" == "maxent" || "$RUN_ONLY" == "listwise" || "$RUN_ONLY" == "seed" ]]; then
  MIN_REQUIRED_GPUS=4
fi
if (( GPU_COUNT < MIN_REQUIRED_GPUS )); then
  echo "This launcher expects at least ${MIN_REQUIRED_GPUS} visible GPUs for RUN_ONLY=${RUN_ONLY}; detected ${GPU_COUNT}." >&2
  exit 1
fi
MAX_GPU_ID=$((GPU_COUNT - 1))

if (( USE_GRPO_STACK )); then
  assert_csv_ids "GRPO_TRAIN_GPUS" "$GRPO_TRAIN_GPUS" "$MAX_GPU_ID"
  [[ "$GRPO_VLLM_GPU" =~ ^[0-9]+$ ]] || { echo "Invalid GRPO_VLLM_GPU=$GRPO_VLLM_GPU" >&2; exit 1; }
  (( GRPO_VLLM_GPU <= MAX_GPU_ID )) || { echo "GRPO_VLLM_GPU out of range" >&2; exit 1; }
  (( $(csv_count "$GRPO_TRAIN_GPUS") == TRAIN_NUM_PROCESSES )) || {
    echo "GRPO_TRAIN_GPUS must contain exactly ${TRAIN_NUM_PROCESSES} GPUs" >&2
    exit 1
  }
  contains_id "$GRPO_TRAIN_GPUS" "$GRPO_VLLM_GPU" && { echo "GRPO vLLM GPU overlaps GRPO train GPUs" >&2; exit 1; }
fi

if (( USE_SECONDARY_STACK )); then
  assert_csv_ids "MAXENT_TRAIN_GPUS" "$MAXENT_TRAIN_GPUS" "$MAX_GPU_ID"
  [[ "$MAXENT_VLLM_GPU" =~ ^[0-9]+$ ]] || { echo "Invalid MAXENT_VLLM_GPU=$MAXENT_VLLM_GPU" >&2; exit 1; }
  (( MAXENT_VLLM_GPU <= MAX_GPU_ID )) || { echo "MAXENT_VLLM_GPU out of range" >&2; exit 1; }
  (( $(csv_count "$MAXENT_TRAIN_GPUS") == TRAIN_NUM_PROCESSES )) || {
    echo "MAXENT_TRAIN_GPUS must contain exactly ${TRAIN_NUM_PROCESSES} GPUs" >&2
    exit 1
  }
  contains_id "$MAXENT_TRAIN_GPUS" "$MAXENT_VLLM_GPU" && { echo "${SECONDARY_STACK_LABEL} vLLM GPU overlaps ${SECONDARY_STACK_LABEL} train GPUs" >&2; exit 1; }
fi

if (( USE_GRPO_STACK )) && (( USE_SECONDARY_STACK )) && [[ "$RUN_ONLY" == "both" ]]; then
  contains_id "$GRPO_TRAIN_GPUS" "$MAXENT_VLLM_GPU" && { echo "${SECONDARY_STACK_LABEL} vLLM GPU overlaps GRPO train GPUs" >&2; exit 1; }
  contains_id "$MAXENT_TRAIN_GPUS" "$GRPO_VLLM_GPU" && { echo "GRPO vLLM GPU overlaps ${SECONDARY_STACK_LABEL} train GPUs" >&2; exit 1; }
  for gid in ${GRPO_TRAIN_GPUS//,/ }; do
    contains_id "$MAXENT_TRAIN_GPUS" "$gid" && { echo "Train GPU overlap between GRPO and ${SECONDARY_STACK_LABEL}: $gid" >&2; exit 1; }
  done
fi

GRPO_MODEL="${MODEL}"
MAXENT_MODEL="${MODEL}"
GRPO_REV=""
MAXENT_REV=""
GRPO_MAX_LEN=""
MAXENT_MAX_LEN=""
if [[ -n "$GRPO_CFG" ]]; then
  GRPO_MODEL="$(yaml_scalar model_name_or_path "$GRPO_CFG")"
  GRPO_REV="$(yaml_scalar model_revision "$GRPO_CFG")"
  GRPO_MODEL="${GRPO_MODEL:-$MODEL}"
  GRPO_MAX_LEN="${GRPO_MAX_LEN:-$(infer_max_model_len "$GRPO_CFG")}"
fi
if [[ -n "$MAXENT_CFG" ]]; then
  MAXENT_MODEL="$(yaml_scalar model_name_or_path "$MAXENT_CFG")"
  MAXENT_REV="$(yaml_scalar model_revision "$MAXENT_CFG")"
  MAXENT_MODEL="${MAXENT_MODEL:-$MODEL}"
  MAXENT_MAX_LEN="${MAXENT_MAX_LEN:-$(infer_max_model_len "$MAXENT_CFG")}"
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_GROUP="${WANDB_RUN_GROUP:-dual_4plus4_${MODEL//\//-}_${STAMP}}"
GRPO_RUN_NAME="${GRPO_RUN_NAME:-${RUN_GROUP}-grpo}"
if [[ "$SECONDARY_STACK_TAG" == "listwise" ]]; then
  MAXENT_RUN_NAME="${LISTWISE_RUN_NAME:-${MAXENT_RUN_NAME:-${RUN_GROUP}-listwise}}"
elif [[ "$SECONDARY_STACK_TAG" == "seed" ]]; then
  MAXENT_RUN_NAME="${SEED_RUN_NAME:-${MAXENT_RUN_NAME:-${RUN_GROUP}-seed}}"
else
  MAXENT_RUN_NAME="${MAXENT_RUN_NAME:-${RUN_GROUP}-maxent}"
fi

GRPO_ARGS="$(append_flag_value "$GRPO_ARGS" "--run_name" "$GRPO_RUN_NAME")"
GRPO_ARGS="$(append_flag_value "$GRPO_ARGS" "--wandb_run_group" "$RUN_GROUP")"
MAXENT_ARGS="$(append_flag_value "$MAXENT_ARGS" "--run_name" "$MAXENT_RUN_NAME")"
MAXENT_ARGS="$(append_flag_value "$MAXENT_ARGS" "--wandb_run_group" "$RUN_GROUP")"

STEP0_PAPER_EVAL_GPU="${MAXENT_STEP0_PAPER_EVAL_GPU:-0}"
STEP0_PAPER_EVAL_ENABLED="${MAXENT_STEP0_PAPER_EVAL_ENABLED:-auto}"
STEP0_PAPER_EVAL_WORKSPACE_DIR="${MAXENT_STEP0_PAPER_EVAL_WORKSPACE_DIR:-$VAR_DIR/seed_paper_eval}"
STEP0_PAPER_EVAL_RESULTS_DIR="${MAXENT_STEP0_PAPER_EVAL_RESULTS_DIR:-$VAR_DIR/artifacts/seed_paper_eval/${SLURM_JOB_ID:-manual}}"
STEP0_PAPER_EVAL_TEMPLATE="${MAXENT_STEP0_PAPER_EVAL_TEMPLATE:-}"
STEP0_PAPER_EVAL_TASKS="${MAXENT_STEP0_PAPER_EVAL_TASKS:-}"
STEP0_PAPER_EVAL_MAX_TEST="${MAXENT_STEP0_PAPER_EVAL_MAX_TEST:-}"
STEP0_PAPER_EVAL_PASS_AT_8_ENABLED="${MAXENT_STEP0_PAPER_EVAL_PASS_AT_8_ENABLED:-auto}"
STEP0_PAPER_EVAL_PASS_AT_8_SAMPLES="${MAXENT_STEP0_PAPER_EVAL_PASS_AT_8_SAMPLES:-8}"
STEP0_PAPER_EVAL_PASS_AT_8_TEMPERATURE="${MAXENT_STEP0_PAPER_EVAL_PASS_AT_8_TEMPERATURE:-1.0}"
STEP0_PAPER_EVAL_PASS_AT_8_TOP_P="${MAXENT_STEP0_PAPER_EVAL_PASS_AT_8_TOP_P:-1.0}"
STEP0_PAPER_EVAL_EXPECTED_PROFILE="${MAXENT_STEP0_PAPER_EVAL_EXPECTED_PROFILE:-}"
STEP0_PAPER_EVAL_ENFORCE_EXPECTED="${MAXENT_STEP0_PAPER_EVAL_ENFORCE_EXPECTED:-1}"
STEP0_PAPER_EVAL_RUNTIME_SWAP_SPACE="${MAXENT_STEP0_PAPER_EVAL_RUNTIME_SWAP_SPACE:-0}"
STEP0_PAPER_EVAL_WANDB_ENABLED="${MAXENT_STEP0_PAPER_EVAL_WANDB_ENABLED:-0}"
STEP0_PAPER_EVAL_WANDB_PROJECT="${MAXENT_STEP0_PAPER_EVAL_WANDB_PROJECT:-${WANDB_PROJECT:-huggingface}}"
STEP0_PAPER_EVAL_WANDB_ENTITY="${MAXENT_STEP0_PAPER_EVAL_WANDB_ENTITY:-${WANDB_ENTITY:-ogd3-princeton-university}}"
STEP0_PAPER_EVAL_WANDB_GROUP="${MAXENT_STEP0_PAPER_EVAL_WANDB_GROUP:-$RUN_GROUP}"
STEP0_PAPER_EVAL_WANDB_RUN_NAME="${MAXENT_STEP0_PAPER_EVAL_WANDB_RUN_NAME:-${SLURM_JOB_NAME:-step0}-step0-paper-eval}"
STEP0_PAPER_EVAL_WANDB_JOB_TYPE="${MAXENT_STEP0_PAPER_EVAL_WANDB_JOB_TYPE:-step0_paper_eval}"

VLLM_GRPO_LOG="$VAR_DIR/artifacts/logs/vllm-grpo-${SLURM_JOB_ID}.out"
VLLM_MAXENT_LOG="$VAR_DIR/artifacts/logs/vllm-${SECONDARY_STACK_TAG}-${SLURM_JOB_ID}.out"
TRAIN_GRPO_LOG="$VAR_DIR/artifacts/logs/train-grpo-${SLURM_JOB_ID}.log"
TRAIN_MAXENT_LOG="$VAR_DIR/artifacts/logs/train-${SECONDARY_STACK_TAG}-${SLURM_JOB_ID}.log"
STEP0_PAPER_EVAL_LOG="$VAR_DIR/artifacts/logs/step0-paper-eval-${SLURM_JOB_ID}.log"
: > "$VLLM_GRPO_LOG"
: > "$VLLM_MAXENT_LOG"
: > "$TRAIN_GRPO_LOG"
: > "$TRAIN_MAXENT_LOG"
: > "$STEP0_PAPER_EVAL_LOG"

activate_env
RESOLVED_ACCELERATOR="$(resolve_accelerator_name "$ACCELERATOR")"
ACCEL_FILE="configs/recipes/accelerate_configs/${RESOLVED_ACCELERATOR}.yaml"
[[ -f "$ACCEL_FILE" ]] || { echo "Missing accelerate config: $ACCEL_FILE" >&2; exit 1; }
ACCEL_FILE_ABS="$(to_abs_path "$ACCEL_FILE")"

if (( USE_GRPO_STACK )); then
  echo "[layout] GRPO:   vLLM GPU=${GRPO_VLLM_GPU}, train GPUs=${GRPO_TRAIN_GPUS}, vLLM port=${GRPO_VLLM_PORT}"
fi
if (( USE_SECONDARY_STACK )); then
  echo "[layout] ${SECONDARY_STACK_LABEL}: vLLM GPU=${MAXENT_VLLM_GPU}, train GPUs=${MAXENT_TRAIN_GPUS}, vLLM port=${MAXENT_VLLM_PORT}"
fi
echo "[layout] recipe_profile=${RECIPE_PROFILE}"
echo "[layout] accelerator_requested=${ACCELERATOR} accelerator_resolved=${RESOLVED_ACCELERATOR}"
echo "[layout] run_group=${RUN_GROUP}"
echo "[layout] wandb_dir=${WANDB_DIR}"
echo "[layout] wandb_data_dir=${WANDB_DATA_DIR}"

GRPO_VLLM_PID=""
MAXENT_VLLM_PID=""
GRPO_TRAIN_PID=""
MAXENT_TRAIN_PID=""
LAST_BG_PID=""

cleanup() {
  for pid in "$GRPO_TRAIN_PID" "$MAXENT_TRAIN_PID" "$GRPO_VLLM_PID" "$MAXENT_VLLM_PID"; do
    if [[ -n "$pid" ]]; then
      kill "$pid" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT

start_vllm() {
  local label="$1"
  local gpu_id="$2"
  local model_id="$3"
  local revision="$4"
  local port="$5"
  local max_len="$6"
  local log_file="$7"
  (
    activate_env
    export CUDA_VISIBLE_DEVICES="$gpu_id"
    export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
    export VLLM_INIT_COMMUNICATOR_TIMEOUT
    cmd=(
      python "$ROOT_DIR/tools/vllm_serve_compat.py"
      --model "$model_id"
      --host 127.0.0.1
      --port "$port"
      --tensor_parallel_size 1
      --data_parallel_size 1
      --max_model_len "$max_len"
      --dtype "$VLLM_DTYPE"
      --enforce_eager "$VLLM_ENFORCE_EAGER"
    )
    if [[ -n "$revision" ]]; then
      cmd+=(--revision "$revision")
    fi
    echo "[$label] ${cmd[*]}"
    exec "${cmd[@]}"
  ) >> "$log_file" 2>&1 &
  LAST_BG_PID="$!"
}

wait_vllm_health() {
  local label="$1"
  local port="$2"
  local pid="$3"
  local log_file="$4"
  for attempt in $(seq 1 "$VLLM_HEALTH_MAX_ATTEMPTS"); do
    if curl -fsS "http://127.0.0.1:${port}/health/" >/dev/null 2>&1; then
      echo "[$label] vLLM healthy on port ${port}"
      return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "[$label] vLLM process exited during startup." >&2
      tail -n 80 "$log_file" >&2 || true
      return 1
    fi
    sleep "$VLLM_HEALTH_INTERVAL_S"
  done
  echo "[$label] vLLM health timeout on port ${port}." >&2
  tail -n 80 "$log_file" >&2 || true
  return 1
}

run_vllm_token_boundary_smoke() {
  local label="$1"
  local port="$2"
  local model_id="$3"
  local log_file="$4"
  if ! is_truthy_flag "${MAXENT_VLLM_TOKEN_BOUNDARY_SMOKE:-0}"; then
    return 0
  fi
  (
    activate_env
    export PYTHONPATH="$ROOT_DIR/src:${PYTHONPATH:-}"
    cmd=(
      python "$ROOT_DIR/tools/vllm_token_boundary_smoke.py"
      --model-name "$model_id"
      --url "http://127.0.0.1:${port}/generate"
    )
    echo "[$label] ${cmd[*]}"
    exec "${cmd[@]}"
  ) >> "$log_file" 2>&1
  (
    activate_env
    export PYTHONPATH="$ROOT_DIR/src:${PYTHONPATH:-}"
    cmd=(
      python "$ROOT_DIR/tools/vllm_trl_client_boundary_smoke.py"
      --model-name "$model_id"
      --url "http://127.0.0.1:${port}/generate"
    )
    echo "[$label-trl-client] ${cmd[*]}"
    exec "${cmd[@]}"
  ) >> "$log_file" 2>&1
}

launch_train() {
  local label="$1"
  local entrypoint="$2"
  local recipe="$3"
  local train_gpus="$4"
  local main_port="$5"
  local group_port="$6"
  local vllm_port="$7"
  local extra_args="$8"
  local log_file="$9"
  local model_id="${10}"

  (
    activate_env
    export CUDA_VISIBLE_DEVICES="$train_gpus"
    export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
    export PYTHONPATH="$ROOT_DIR/src:${PYTHONPATH:-}"
    export MAXENT_VLLM_MODE=server
    export VLLM_MODE=server
    export MAXENT_VLLM_URL="http://127.0.0.1:${vllm_port}/generate"
    export VLLM_URL="$MAXENT_VLLM_URL"
    export MAXENT_VLLM_SERVER_MODEL_NAME="$model_id"
    export VLLM_GROUP_PORT="$group_port"
    export PORT_FOR_COMMUNICATION="$group_port"
    export GRPO_RECIPE="$recipe"
    export GRPO_RECIPE_USED="$recipe"
    export SEED_PAPER_EVAL_MODEL_NAME="$model_id"
    export MAXENT_STEP0_PAPER_EVAL_RESULTS_DIR="$STEP0_PAPER_EVAL_RESULTS_DIR"
    export MAXENT_TRL_VLLM_CLIENT_PATCH
    export MAXENT_VLLM_ASYNC_INIT
    export MAXENT_VLLM_INIT_RETRIES
    export MAXENT_VLLM_INIT_RETRY_BACKOFF_S
    export MAXENT_VLLM_INIT_TIMEOUT_S
    export VLLM_INIT_COMMUNICATOR_TIMEOUT

    launch=(
      python -m accelerate.commands.launch
      --config_file "$ACCEL_FILE_ABS"
      --num_machines 1
      --num_processes "$TRAIN_NUM_PROCESSES"
      --main_process_ip 127.0.0.1
      --main_process_port "$main_port"
      --machine_rank 0
      --tee "$ACC_TEE"
      "$entrypoint"
      --use_vllm true
      --vllm_mode server
      --vllm_server_host 127.0.0.1
      --vllm_server_port "$vllm_port"
    )
    cmd=("${launch[@]}")
    if [[ -n "$extra_args" ]]; then
      eval "cmd+=( ${extra_args} )"
    fi

    echo "[$label] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
    echo "[$label] ${cmd[*]}"
    exec "${cmd[@]}"
  ) >> "$log_file" 2>&1 &
  LAST_BG_PID="$!"
}

select_step0_paper_eval_model() {
  if (( USE_GRPO_STACK )) && [[ -n "$GRPO_MODEL" ]]; then
    printf '%s\n' "$GRPO_MODEL"
    return 0
  fi
  if (( USE_SECONDARY_STACK )) && [[ -n "$MAXENT_MODEL" ]]; then
    printf '%s\n' "$MAXENT_MODEL"
    return 0
  fi
  printf '%s\n' "$MODEL"
}

is_qwen_math_1_5b_model() {
  local model_id="$1"
  [[ "$model_id" == "Qwen/Qwen2.5-Math-1.5B" || "$model_id" == "Qwen2.5-Math-1.5B" ]]
}

should_run_step0_paper_eval() {
  local model_id="$1"
  local requested
  requested="$(printf '%s' "$STEP0_PAPER_EVAL_ENABLED" | tr '[:upper:]' '[:lower:]')"
  if [[ "$requested" == "0" || "$requested" == "false" || "$requested" == "no" || "$requested" == "off" ]]; then
    return 1
  fi
  if [[ "$requested" == "1" || "$requested" == "true" || "$requested" == "yes" || "$requested" == "on" ]]; then
    return 0
  fi
  if ! is_qwen_math_1_5b_model "$model_id"; then
    return 1
  fi
  case "${CONFIG_SUFFIX_GRPO,,},${CONFIG_SUFFIX_MAXENT,,}" in
    *math*)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

should_run_step0_paper_eval_pass_at_8() {
  local requested
  requested="$(printf '%s' "$STEP0_PAPER_EVAL_PASS_AT_8_ENABLED" | tr '[:upper:]' '[:lower:]')"
  if [[ "$requested" == "0" || "$requested" == "false" || "$requested" == "no" || "$requested" == "off" ]]; then
    return 1
  fi
  if [[ "$requested" == "1" || "$requested" == "true" || "$requested" == "yes" || "$requested" == "on" ]]; then
    return 0
  fi
  case "${CONFIG_SUFFIX_GRPO,,},${CONFIG_SUFFIX_MAXENT,,}" in
    *math_fair*)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

run_step0_paper_eval() {
  local model_id="$1"
  local log_file="$2"
  local cmd=(
    python "$ROOT_DIR/tools/seed_paper_eval.py"
    --model-name "$model_id"
    --workspace-dir "$STEP0_PAPER_EVAL_WORKSPACE_DIR"
    --results-dir "$STEP0_PAPER_EVAL_RESULTS_DIR"
  )
  if [[ -n "$STEP0_PAPER_EVAL_TEMPLATE" ]]; then
    cmd+=(--template "$STEP0_PAPER_EVAL_TEMPLATE")
  fi
  if [[ -n "$STEP0_PAPER_EVAL_TASKS" ]]; then
    cmd+=(--tasks "$STEP0_PAPER_EVAL_TASKS")
  fi
  if [[ -n "$STEP0_PAPER_EVAL_MAX_TEST" ]]; then
    cmd+=(--max-test "$STEP0_PAPER_EVAL_MAX_TEST")
  fi
  if should_run_step0_paper_eval_pass_at_8; then
    cmd+=(--pass-at-8)
    cmd+=(--pass-at-8-samples "$STEP0_PAPER_EVAL_PASS_AT_8_SAMPLES")
    cmd+=(--pass-at-8-temperature "$STEP0_PAPER_EVAL_PASS_AT_8_TEMPERATURE")
    cmd+=(--pass-at-8-top-p "$STEP0_PAPER_EVAL_PASS_AT_8_TOP_P")
  fi
  if [[ -n "$STEP0_PAPER_EVAL_EXPECTED_PROFILE" ]]; then
    cmd+=(
      --expected-profile "$STEP0_PAPER_EVAL_EXPECTED_PROFILE"
    )
    if is_truthy_flag "$STEP0_PAPER_EVAL_ENFORCE_EXPECTED"; then
      cmd+=(--enforce-expected)
    fi
  fi
  if [[ -n "$STEP0_PAPER_EVAL_RUNTIME_SWAP_SPACE" ]]; then
    cmd+=(--runtime-swap-space "$STEP0_PAPER_EVAL_RUNTIME_SWAP_SPACE")
  fi
  if is_truthy_flag "$STEP0_PAPER_EVAL_WANDB_ENABLED"; then
    cmd+=(--wandb)
  fi
  if [[ -n "$STEP0_PAPER_EVAL_WANDB_PROJECT" ]]; then
    cmd+=(--wandb-project "$STEP0_PAPER_EVAL_WANDB_PROJECT")
  fi
  if [[ -n "$STEP0_PAPER_EVAL_WANDB_ENTITY" ]]; then
    cmd+=(--wandb-entity "$STEP0_PAPER_EVAL_WANDB_ENTITY")
  fi
  if [[ -n "$STEP0_PAPER_EVAL_WANDB_GROUP" ]]; then
    cmd+=(--wandb-group "$STEP0_PAPER_EVAL_WANDB_GROUP")
  fi
  if [[ -n "$STEP0_PAPER_EVAL_WANDB_RUN_NAME" ]]; then
    cmd+=(--wandb-run-name "$STEP0_PAPER_EVAL_WANDB_RUN_NAME")
  fi
  if [[ -n "$STEP0_PAPER_EVAL_WANDB_JOB_TYPE" ]]; then
    cmd+=(--wandb-job-type "$STEP0_PAPER_EVAL_WANDB_JOB_TYPE")
  fi
  (
    activate_env
    mkdir -p "$STEP0_PAPER_EVAL_RESULTS_DIR"
    export CUDA_VISIBLE_DEVICES="$STEP0_PAPER_EVAL_GPU"
    export PYTHONPATH="$ROOT_DIR/src:${PYTHONPATH:-}"
    echo "[step0-paper-eval] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
    echo "[step0-paper-eval] ${cmd[*]}"
    exec "${cmd[@]}"
  ) >> "$log_file" 2>&1
}

STEP0_PAPER_EVAL_MODEL="$(select_step0_paper_eval_model)"
if [[ -z "$STEP0_PAPER_EVAL_TEMPLATE" ]]; then
  case "${CONFIG_SUFFIX_GRPO,,},${CONFIG_SUFFIX_MAXENT,,}" in
    *math_fair*)
      STEP0_PAPER_EVAL_TEMPLATE="no"
      ;;
  esac
fi
if [[ -z "$STEP0_PAPER_EVAL_EXPECTED_PROFILE" && -z "$STEP0_PAPER_EVAL_TEMPLATE" ]] && is_qwen_math_1_5b_model "$STEP0_PAPER_EVAL_MODEL"; then
  STEP0_PAPER_EVAL_EXPECTED_PROFILE="table2_qwen2_5_math_1_5b"
fi

if should_run_step0_paper_eval "$STEP0_PAPER_EVAL_MODEL"; then
  echo "[step0-paper-eval] enabled model=${STEP0_PAPER_EVAL_MODEL}"
  echo "[step0-paper-eval] workspace=${STEP0_PAPER_EVAL_WORKSPACE_DIR}"
  echo "[step0-paper-eval] results=${STEP0_PAPER_EVAL_RESULTS_DIR}"
  echo "[step0-paper-eval] log=${STEP0_PAPER_EVAL_LOG}"
  if [[ -n "$STEP0_PAPER_EVAL_TEMPLATE" ]]; then
    echo "[step0-paper-eval] template=${STEP0_PAPER_EVAL_TEMPLATE}"
  fi
  if should_run_step0_paper_eval_pass_at_8; then
    echo "[step0-paper-eval] sampled_metrics=pass_at_8,mean_at_8 n=${STEP0_PAPER_EVAL_PASS_AT_8_SAMPLES} temp=${STEP0_PAPER_EVAL_PASS_AT_8_TEMPERATURE} top_p=${STEP0_PAPER_EVAL_PASS_AT_8_TOP_P}"
  fi
  if [[ -n "$STEP0_PAPER_EVAL_EXPECTED_PROFILE" ]]; then
    echo "[step0-paper-eval] expected_profile=${STEP0_PAPER_EVAL_EXPECTED_PROFILE} enforce=${STEP0_PAPER_EVAL_ENFORCE_EXPECTED}"
  fi
  run_step0_paper_eval "$STEP0_PAPER_EVAL_MODEL" "$STEP0_PAPER_EVAL_LOG"
  echo "[step0-paper-eval] completed"
else
  echo "[step0-paper-eval] skipped"
fi

if (( USE_GRPO_STACK )); then
  start_vllm "grpo-vllm" "$GRPO_VLLM_GPU" "$GRPO_MODEL" "$GRPO_REV" "$GRPO_VLLM_PORT" "$GRPO_MAX_LEN" "$VLLM_GRPO_LOG"
  GRPO_VLLM_PID="$LAST_BG_PID"
fi
if (( USE_SECONDARY_STACK )); then
  start_vllm "${SECONDARY_STACK_TAG}-vllm" "$MAXENT_VLLM_GPU" "$MAXENT_MODEL" "$MAXENT_REV" "$MAXENT_VLLM_PORT" "$MAXENT_MAX_LEN" "$VLLM_MAXENT_LOG"
  MAXENT_VLLM_PID="$LAST_BG_PID"
fi

if [[ -n "$GRPO_VLLM_PID" ]]; then
  wait_vllm_health "grpo-vllm" "$GRPO_VLLM_PORT" "$GRPO_VLLM_PID" "$VLLM_GRPO_LOG"
  run_vllm_token_boundary_smoke "grpo-vllm-smoke" "$GRPO_VLLM_PORT" "$GRPO_MODEL" "$VLLM_GRPO_LOG"
fi
if [[ -n "$MAXENT_VLLM_PID" ]]; then
  wait_vllm_health "${SECONDARY_STACK_TAG}-vllm" "$MAXENT_VLLM_PORT" "$MAXENT_VLLM_PID" "$VLLM_MAXENT_LOG"
  run_vllm_token_boundary_smoke "${SECONDARY_STACK_TAG}-vllm-smoke" "$MAXENT_VLLM_PORT" "$MAXENT_MODEL" "$VLLM_MAXENT_LOG"
fi

if (( USE_GRPO_STACK )); then
  launch_train "grpo-train" "src/maxent_grpo/grpo.py" "$GRPO_CFG_ABS" "$GRPO_TRAIN_GPUS" "$GRPO_MASTER_PORT" "$GRPO_GROUP_PORT" "$GRPO_VLLM_PORT" "$GRPO_ARGS" "$TRAIN_GRPO_LOG" "$GRPO_MODEL"
  GRPO_TRAIN_PID="$LAST_BG_PID"
fi
if (( USE_SECONDARY_STACK )); then
  launch_train "${SECONDARY_STACK_TAG}-train" "src/maxent_grpo/grpo.py" "$MAXENT_CFG_ABS" "$MAXENT_TRAIN_GPUS" "$MAXENT_MASTER_PORT" "$MAXENT_GROUP_PORT" "$MAXENT_VLLM_PORT" "$MAXENT_ARGS" "$TRAIN_MAXENT_LOG" "$MAXENT_MODEL"
  MAXENT_TRAIN_PID="$LAST_BG_PID"
fi

grpo_status=0
maxent_status=0
if [[ -n "$GRPO_TRAIN_PID" ]]; then
  wait "$GRPO_TRAIN_PID" || grpo_status=$?
fi
if [[ -n "$MAXENT_TRAIN_PID" ]]; then
  wait "$MAXENT_TRAIN_PID" || maxent_status=$?
fi

if [[ -n "$GRPO_VLLM_PID" ]]; then
  kill "$GRPO_VLLM_PID" 2>/dev/null || true
  wait "$GRPO_VLLM_PID" 2>/dev/null || true
fi
if [[ -n "$MAXENT_VLLM_PID" ]]; then
  kill "$MAXENT_VLLM_PID" 2>/dev/null || true
  wait "$MAXENT_VLLM_PID" 2>/dev/null || true
fi

echo "[summary] grpo_status=${grpo_status} maxent_status=${maxent_status}"
echo "[summary] logs:"
echo "  $STEP0_PAPER_EVAL_LOG"
echo "  $VLLM_GRPO_LOG"
echo "  $VLLM_MAXENT_LOG"
echo "  $TRAIN_GRPO_LOG"
echo "  $TRAIN_MAXENT_LOG"

if (( grpo_status != 0 || maxent_status != 0 )); then
  exit 1
fi
