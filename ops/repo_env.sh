#!/usr/bin/env bash

# Source this from any shell before ad hoc repo commands to keep caches and
# runtime state under ./var instead of falling back to ~/.cache or ~/.config.

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "source ops/repo_env.sh"
  exit 1
fi

_maxent_repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MAXENT_GRPO_ROOT="${MAXENT_GRPO_ROOT:-$_maxent_repo_root}"
export MAXENT_GRPO_VAR_ROOT="${MAXENT_GRPO_VAR_ROOT:-$MAXENT_GRPO_ROOT/var}"

export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$MAXENT_GRPO_VAR_ROOT/cache/xdg}"
export XDG_CONFIG_HOME="${XDG_CONFIG_HOME:-$MAXENT_GRPO_VAR_ROOT/config}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$MAXENT_GRPO_VAR_ROOT/cache/pip}"
export TMPDIR="${TMPDIR:-$MAXENT_GRPO_VAR_ROOT/tmp}"
export HF_HOME="${HF_HOME:-$MAXENT_GRPO_VAR_ROOT/cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HUGGINGFACE_HUB_CACHE}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_ASSETS_CACHE="${HF_ASSETS_CACHE:-$HF_HOME/assets}"
export TORCH_HOME="${TORCH_HOME:-$MAXENT_GRPO_VAR_ROOT/cache/torch}"
export WANDB_DIR="${WANDB_DIR:-$MAXENT_GRPO_VAR_ROOT/wandb/runs}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$MAXENT_GRPO_VAR_ROOT/wandb/cache}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-$MAXENT_GRPO_VAR_ROOT/wandb/config}"
export WANDB_DATA_DIR="${WANDB_DATA_DIR:-$MAXENT_GRPO_VAR_ROOT/wandb/data}"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-$MAXENT_GRPO_VAR_ROOT/pycache}"

if [[ ":${PYTHONPATH:-}:" != *":$MAXENT_GRPO_ROOT:"* ]]; then
  export PYTHONPATH="$MAXENT_GRPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
fi

mkdir -p \
  "$XDG_CACHE_HOME" \
  "$XDG_CONFIG_HOME" \
  "$PIP_CACHE_DIR" \
  "$TMPDIR" \
  "$HF_HOME" \
  "$HUGGINGFACE_HUB_CACHE" \
  "$HF_DATASETS_CACHE" \
  "$TRANSFORMERS_CACHE" \
  "$HF_ASSETS_CACHE" \
  "$TORCH_HOME" \
  "$WANDB_DIR" \
  "$WANDB_CACHE_DIR" \
  "$WANDB_CONFIG_DIR" \
  "$WANDB_DATA_DIR" \
  "$PYTHONPYCACHEPREFIX"

unset _maxent_repo_root
