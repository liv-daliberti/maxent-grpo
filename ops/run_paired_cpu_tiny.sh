#!/usr/bin/env bash
# Tiny CPU-only paired run: baseline GRPO vs MaxEnt-GRPO using matched overrides.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODEL="${MODEL:-Qwen2.5-0.5B-Instruct}"
GRPO_RECIPE="${GRPO_RECIPE:-configs/recipes/${MODEL}/grpo/config_math.yaml}"
MAXENT_RECIPE="${MAXENT_RECIPE:-configs/recipes/${MODEL}/maxent-grpo/config_math.yaml}"
RECIPE_TMP_DIR="${RECIPE_TMP_DIR:-var/recipes/cpu_tiny}"

RUN_ONLY="${RUN_ONLY:-both}" # grpo | maxent | both
MAX_STEPS="${MAX_STEPS:-10}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1500}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-700}"
NUM_GENERATIONS="${NUM_GENERATIONS:-8}"
TRAIN_BATCH="${TRAIN_BATCH:-4}"
EVAL_BATCH="${EVAL_BATCH:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-32}"
DATASET_NAME="${DATASET_NAME:-}"
DO_EVAL="${DO_EVAL:-0}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-32}"
MKL_NUM_THREADS="${MKL_NUM_THREADS:-32}"

OUTPUT_DIR_GRPO="${OUTPUT_DIR_GRPO:-var/data/cpu_tiny_grpo}"
OUTPUT_DIR_MAXENT="${OUTPUT_DIR_MAXENT:-var/data/cpu_tiny_maxent}"

MAXENT_CLIP_ADV_BASELINE="${MAXENT_CLIP_ADV_BASELINE:-}"
if [[ -z "${MAXENT_CLIP_ADV_BASELINE}" ]]; then
  MAXENT_CLIP_ADV_BASELINE="$(python - <<'PY'
import os
val = float(os.environ.get("NUM_GENERATIONS", "2"))
print(1.0 / val if val else 0.0)
PY
  )"
fi

LOG_DIR="${LOG_DIR:-var/artifacts/logs/cpu_tiny}"
mkdir -p "${LOG_DIR}"

BASE_ENV=(
  CUDA_VISIBLE_DEVICES=""
  ACCELERATE_USE_CPU=1
  WANDB_DISABLED=true
  OMP_NUM_THREADS="${OMP_NUM_THREADS}"
  MKL_NUM_THREADS="${MKL_NUM_THREADS}"
)

render_recipe() {
  local src="$1"
  local dst="$2"
  local mode="$3"
  local num_generations="$4"
  python - "$src" "$dst" "$mode" "$num_generations" <<'PY'
import os
import sys

src, dst, mode, num_generations_raw = sys.argv[1:5]
try:
    num_generations = int(float(num_generations_raw))
except (TypeError, ValueError):
    num_generations = 2
overrides = {
    "bf16": False,
    "fp16": False,
    "use_vllm": False,
    "vllm_request_logprobs": False,
    "vllm_return_logprobs": False,
    "torch_dtype": "float32",
    "num_generations": num_generations,
    "generation_batch_size": None,
    "steps_per_generation": None,
}

try:
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(src)
    if mode == "grpo":
        for key in list(cfg.keys()):
            if str(key).startswith("maxent_"):
                del cfg[key]
    for key, val in overrides.items():
        cfg[key] = val
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    OmegaConf.save(cfg, dst)
except Exception:
    try:
        import yaml
    except Exception as exc:
        raise SystemExit(f"Failed to render recipe; missing omegaconf/yaml: {exc}") from exc
    with open(src, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    if mode == "grpo":
        for key in list(cfg):
            if str(key).startswith("maxent_"):
                cfg.pop(key, None)
    cfg.update(overrides)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)
PY
}

GRPO_RECIPE_RENDERED="${RECIPE_TMP_DIR}/grpo_cpu_tiny.yaml"
MAXENT_RECIPE_RENDERED="${RECIPE_TMP_DIR}/maxent_cpu_tiny.yaml"
render_recipe "${GRPO_RECIPE}" "${GRPO_RECIPE_RENDERED}" "grpo" "${NUM_GENERATIONS}"
render_recipe "${MAXENT_RECIPE}" "${MAXENT_RECIPE_RENDERED}" "maxent" "${NUM_GENERATIONS}"

BASE_TRAINING_ARGS=(
  "training.use_vllm=false"
  "training.vllm_request_logprobs=false"
  "training.vllm_return_logprobs=false"
  "training.bf16=false"
  "training.fp16=false"
  "model.torch_dtype=float32"
  "training.optim=adamw_torch"
  "training.gradient_checkpointing=false"
  "training.gradient_accumulation_steps=${GRAD_ACCUM}"
  "training.per_device_train_batch_size=${TRAIN_BATCH}"
  "training.per_device_eval_batch_size=${EVAL_BATCH}"
  "training.num_generations=${NUM_GENERATIONS}"
  "training.max_prompt_length=${MAX_PROMPT_LENGTH}"
  "training.max_completion_length=${MAX_COMPLETION_LENGTH}"
  "training.max_steps=${MAX_STEPS}"
  "training.save_strategy=no"
  "training.eval_strategy=no"
  "training.do_eval=false"
  "training.push_to_hub=false"
  "training.report_to=[]"
  "training.dataloader_num_workers=0"
  "training.dataloader_pin_memory=false"
)

if [[ "${DO_EVAL}" == "1" ]]; then
  BASE_TRAINING_ARGS+=(
    "training.do_eval=true"
    "training.eval_strategy=steps"
    "training.eval_steps=1"
  )
fi

SCRIPT_DATASET_ARGS=()
if [[ -n "${DATASET_NAME}" ]]; then
  SCRIPT_DATASET_ARGS=("script.dataset_name=${DATASET_NAME}")
fi

run_grpo() {
  echo "[run] baseline GRPO via shared trainer path (CPU tiny)"
  BASELINE_LOG="${LOG_DIR}/baseline_$(date +%Y%m%d_%H%M%S).log"
  env "${BASE_ENV[@]}" maxent-grpo command=train-maxent \
    maxent.recipe="${GRPO_RECIPE_RENDERED}" \
    "+maxent.training.output_dir=${OUTPUT_DIR_GRPO}" \
    "+maxent.training.train_grpo_objective=true" \
    "${SCRIPT_DATASET_ARGS[@]/#/+maxent.}" \
    "${BASE_TRAINING_ARGS[@]/#/+maxent.}" | tee "${BASELINE_LOG}"
}

run_maxent() {
  echo "[run] MaxEnt-GRPO (CPU tiny)"
  MAXENT_LOG="${LOG_DIR}/maxent_$(date +%Y%m%d_%H%M%S).log"
  env "${BASE_ENV[@]}" maxent-grpo command=train-maxent \
    maxent.recipe="${MAXENT_RECIPE_RENDERED}" \
    "+maxent.training.output_dir=${OUTPUT_DIR_MAXENT}" \
    "+maxent.training.maxent_clip_adv_baseline=${MAXENT_CLIP_ADV_BASELINE}" \
    "${SCRIPT_DATASET_ARGS[@]/#/+maxent.}" \
    "${BASE_TRAINING_ARGS[@]/#/+maxent.}" | tee "${MAXENT_LOG}"
}

summarize_runs() {
  python - "$OUTPUT_DIR_GRPO" "$OUTPUT_DIR_MAXENT" "$BASELINE_LOG" "$MAXENT_LOG" <<'PY'
import ast
import json
import os
import re
import sys

baseline_dir, maxent_dir, baseline_log, maxent_log = sys.argv[1:5]

def _load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        return None

def _baseline_summary():
    summary = {}
    train_results = _load_json(os.path.join(baseline_dir, "train_results.json"))
    if isinstance(train_results, dict):
        summary.update(train_results)
    trainer_state = _load_json(os.path.join(baseline_dir, "trainer_state.json"))
    if isinstance(trainer_state, dict):
        summary["global_step"] = trainer_state.get("global_step")
        log_history = trainer_state.get("log_history") or []
        for entry in reversed(log_history):
            if isinstance(entry, dict) and ("loss" in entry or "train_loss" in entry):
                summary.setdefault("loss", entry.get("loss"))
                summary.setdefault("train_loss", entry.get("train_loss"))
                if "global_step" in entry:
                    summary.setdefault("global_step", entry.get("global_step"))
                break
    return summary

def _parse_metric_line(line):
    parts = line.split(" | ")
    metrics = {}
    for part in parts:
        if "=" not in part:
            continue
        key, val = part.split("=", 1)
        key = key.strip()
        val = val.strip()
        try:
            metrics[key] = float(val)
        except ValueError:
            metrics[key] = val
    return metrics

def _maxent_summary():
    summary = {}
    if not os.path.exists(maxent_log):
        return summary
    last_metrics = None
    dict_pattern = re.compile(r"^\{.*\}$")
    with open(maxent_log, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if "Global metrics step" in line and " | " in line:
                metrics = _parse_metric_line(line.split("Global metrics step", 1)[1])
                if metrics:
                    last_metrics = metrics
                continue
            if dict_pattern.match(line) and "train/loss" in line:
                try:
                    parsed = ast.literal_eval(line)
                except (SyntaxError, ValueError):
                    continue
                if isinstance(parsed, dict):
                    last_metrics = parsed
    if isinstance(last_metrics, dict):
        summary.update(last_metrics)
    return summary

baseline = _baseline_summary()
maxent = _maxent_summary()

def _pick(mapping, *keys):
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None

baseline_loss = _pick(baseline, "train_loss", "loss")
baseline_step = _pick(baseline, "global_step")
maxent_loss = _pick(maxent, "train/loss", "train/loss/total", "train/loss/total_raw")
maxent_step = _pick(maxent, "train/global_step", "global_step")
baseline_kl = _pick(baseline, "kl", "eval_kl")
maxent_kl = _pick(maxent, "train/kl")

print("\n[summary] baseline")
print(json.dumps({"loss": baseline_loss, "kl": baseline_kl, "steps": baseline_step}, indent=2))
print("\n[summary] maxent")
print(json.dumps({"loss": maxent_loss, "kl": maxent_kl, "steps": maxent_step}, indent=2))

if baseline_loss is not None and maxent_loss is not None:
    try:
        delta = float(maxent_loss) - float(baseline_loss)
    except (TypeError, ValueError):
        delta = None
    print("\n[summary] comparison")
    print(json.dumps({"loss_delta": delta, "baseline_loss": baseline_loss, "maxent_loss": maxent_loss}, indent=2))
PY
}

case "${RUN_ONLY}" in
  grpo)
    run_grpo
    ;;
  maxent)
    run_maxent
    ;;
  both)
    run_grpo
    run_maxent
    summarize_runs
    ;;
  *)
    echo "[error] RUN_ONLY must be grpo, maxent, or both (got: ${RUN_ONLY})" >&2
    exit 1
    ;;
esac
