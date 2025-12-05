#!/usr/bin/env bash
# Submit one batch job per checkpoint: each job starts vLLM once and runs all tasks sequentially.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SLURM_SCRIPT="ops/slurm/evaluate_batch.slurm"
TASKS_FILE="${1:-${TASKS_FILE:-ops/slurm/tasks_math_suite.txt}}"
MODEL_ROOT="${MODEL_ROOT:-var/data/Qwen2.5-7B-Open-R1-GRPO-math-2k}"
REVISION="${REVISION:-main}"
TENSOR_PARALLEL="${TENSOR_PARALLEL:-false}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-false}"
SYSTEM_PROMPT_B64="${SYSTEM_PROMPT_B64:-}"
# When CHAIN_JOBS=true, submit jobs with --dependency=afterok so only one runs at a time.
CHAIN_JOBS="${CHAIN_JOBS:-false}"

if [[ ! -f "$TASKS_FILE" ]]; then
  echo "[error] Tasks file not found: $TASKS_FILE" >&2
  exit 1
fi

if [[ ! -d "$MODEL_ROOT" ]]; then
  echo "[error] MODEL_ROOT not found: $MODEL_ROOT" >&2
  exit 1
fi

LAST_JOB=""
while IFS= read -r -d '' ckpt; do
  DEP_ARGS=()
  if [[ "${CHAIN_JOBS,,}" == "true" && -n "$LAST_JOB" ]]; then
    DEP_ARGS+=(--dependency="afterok:$LAST_JOB")
  fi
  echo "[submit] tasks_file=$(basename "$TASKS_FILE") model=$ckpt dep=${DEP_ARGS[*]:-none}"
  SUBMIT_OUT=$(sbatch "${DEP_ARGS[@]}" "$SLURM_SCRIPT" "$TASKS_FILE" "$ckpt" "$REVISION" "$TENSOR_PARALLEL" "$TRUST_REMOTE_CODE" "$SYSTEM_PROMPT_B64")
  echo "$SUBMIT_OUT"
  if [[ "${CHAIN_JOBS,,}" == "true" ]]; then
    # sbatch prints: "Submitted batch job <id>"
    LAST_JOB=$(awk '{print $4}' <<<"$SUBMIT_OUT")
  fi
done < <(find "$MODEL_ROOT" -maxdepth 1 -mindepth 1 -type d -name 'checkpoint-*' -print0 | sort -zV)
