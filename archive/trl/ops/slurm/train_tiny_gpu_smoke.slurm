#!/bin/bash
# Tiny 1-GPU end-to-end smoke run for GRPO + MaxEnt on Slurm.

#SBATCH --job-name=maxent_grpo_tiny_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=mltheory
#SBATCH --account=mltheory
#SBATCH --time=00:30:00
#SBATCH --output=var/artifacts/logs/%x-%j.out
#SBATCH --error=var/artifacts/logs/%x-%j.err

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$ROOT_DIR"

mkdir -p var/artifacts/logs var/e2e/out/slurm_tiny

ENV_ACTIVATE="${ENV_ACTIVATE:-$ROOT_DIR/var/openr1/bin/activate}"
if [[ ! -f "$ENV_ACTIVATE" ]]; then
  echo "Missing virtualenv activation script: $ENV_ACTIVATE" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "$ENV_ACTIVATE"

export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
EXTERNAL_SITE_PACKAGES="${EXTERNAL_SITE_PACKAGES:-}"
VENV_SITE_PACKAGES="${VENV_SITE_PACKAGES:-$ROOT_DIR/var/openr1/lib/python3.11/site-packages}"
if [[ -n "$EXTERNAL_SITE_PACKAGES" && -d "$EXTERNAL_SITE_PACKAGES" ]]; then
  export PYTHONPATH="$ROOT_DIR/var/e2e/shims:$ROOT_DIR/src:$VENV_SITE_PACKAGES:$EXTERNAL_SITE_PACKAGES:${PYTHONPATH:-}"
else
  export PYTHONPATH="$ROOT_DIR/var/e2e/shims:$ROOT_DIR/src:$VENV_SITE_PACKAGES:${PYTHONPATH:-}"
fi

python - <<'PY'
from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path

import torch

from trl import ModelConfig

from maxent_grpo.config import load_grpo_recipe
from maxent_grpo.training.baseline import run_baseline_training

root = Path(os.getcwd())
job_id = os.environ.get("SLURM_JOB_ID", "nojobid")
max_steps = int(os.environ.get("TINY_MAX_STEPS", "1"))
run_root = root / "var" / "e2e" / "out" / "slurm_tiny" / job_id
run_root.mkdir(parents=True, exist_ok=True)

tiny_dataset = {
    "train": [
        {"problem": "2+2", "answer": "4"},
        {"problem": "3+5", "answer": "8"},
        {"problem": "10-7", "answer": "3"},
        {"problem": "9/3", "answer": "3"},
    ],
    "test": [
        {"problem": "6+1", "answer": "7"},
        {"problem": "8-2", "answer": "6"},
    ],
}

runs = [
    ("grpo", root / "var" / "e2e" / "recipes" / "grpo_local.yaml"),
    ("maxent", root / "var" / "e2e" / "recipes" / "maxent_local.yaml"),
]

summary: dict[str, object] = {
    "job_id": job_id,
    "requested_max_steps": max_steps,
    "cuda_available": torch.cuda.is_available(),
    "cuda_device_count": torch.cuda.device_count(),
    "runs": {},
}

for name, recipe in runs:
    out_dir = run_root / name
    out_dir.mkdir(parents=True, exist_ok=True)

    script_args, training_args, model_args = load_grpo_recipe(
        str(recipe), model_config_cls=ModelConfig
    )
    script_args.dataset = deepcopy(tiny_dataset)

    training_args.output_dir = str(out_dir)
    training_args.max_steps = max_steps
    training_args.num_train_epochs = 1
    training_args.logging_steps = 1
    training_args.save_strategy = "no"
    training_args.do_eval = True
    training_args.eval_strategy = "steps"
    training_args.eval_steps = 1
    training_args.eval_before_train = True
    training_args.use_vllm = False
    training_args.report_to = []
    training_args.push_to_hub = False
    training_args.bf16 = False
    training_args.fp16 = False

    run_baseline_training(script_args, training_args, model_args)

    train_results_path = out_dir / "train_results.json"
    trainer_state_path = out_dir / "trainer_state.json"
    train_results = json.loads(train_results_path.read_text())
    trainer_state = json.loads(trainer_state_path.read_text())
    eval_entries = [
        row
        for row in trainer_state.get("log_history", [])
        if any(str(k).startswith("eval") for k in row.keys())
    ]
    last_eval = eval_entries[-1] if eval_entries else {}
    summary["runs"][name] = {
        "output_dir": str(out_dir),
        "global_step": trainer_state.get("global_step"),
        "train_loss": train_results.get("train_loss"),
        "train_runtime": train_results.get("train_runtime"),
        "eval_loss": last_eval.get("eval/loss"),
        "eval_runtime": last_eval.get("eval/runtime"),
    }

summary_path = run_root / "summary.json"
summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
print(json.dumps(summary, indent=2, sort_keys=True))
print(f"summary_path={summary_path}")
PY
