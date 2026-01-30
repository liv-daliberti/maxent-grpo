#!/usr/bin/env bash
set -euo pipefail

# Bootstrap a fully local conda + pip environment under the repo.
# - Env prefix: ./var/openr1
# - Conda caches: ./var/conda/{envs,pkgs,cache}
# - Pip cache/config: ./var/cache/pip + ./var/pip/pip.conf
# - Tmp + XDG caches: ./var/tmp and ./var/cache/xdg

# Resolve repo root (supports running from anywhere inside the repo)
ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"
VAR_DIR="$ROOT_DIR/var"

echo "🔧 Bootstrapping local env in $ROOT_DIR"

# Localized dirs
unset CONDA_ENVS_PATH  # prefer CONDA_ENVS_DIRS to avoid alias conflict
export CONDARC="$ROOT_DIR/.condarc"
export CONDA_PKGS_DIRS="$VAR_DIR/conda/pkgs"
export CONDA_ENVS_DIRS="$VAR_DIR/conda/envs"
export XDG_CACHE_HOME="$VAR_DIR/cache/xdg"
export PIP_CACHE_DIR="$VAR_DIR/cache/pip"
export PIP_CONFIG_FILE="$VAR_DIR/pip/pip.conf"
export TMPDIR="$VAR_DIR/tmp"
export PYTHONNOUSERSITE=1

mkdir -p \
  "$VAR_DIR" \
  "$CONDA_PKGS_DIRS" "$CONDA_ENVS_DIRS" "$VAR_DIR/conda/cache" \
  "$PIP_CACHE_DIR" "$VAR_DIR/pip" "$XDG_CACHE_HOME" "$TMPDIR" \
  "$VAR_DIR/logs"

# Load conda
if [ -f /usr/local/anaconda3/2024.02/etc/profile.d/conda.sh ]; then
  # Common path on some clusters
  source /usr/local/anaconda3/2024.02/etc/profile.d/conda.sh
elif command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
else
  echo "❌ Conda not found on PATH" >&2
  exit 1
fi

ENV_PREFIX="$VAR_DIR/openr1"

echo "📦 Conda env prefix: $ENV_PREFIX"
echo "📁 CONDA_PKGS_DIRS:   $CONDA_PKGS_DIRS"
echo "📁 CONDA_ENVS_DIRS:   $CONDA_ENVS_DIRS"
echo "📁 PIP_CACHE_DIR:     $PIP_CACHE_DIR"
echo "📁 XDG_CACHE_HOME:    $XDG_CACHE_HOME"
echo "📁 TMPDIR:            $TMPDIR"

# Create env if missing
ENV_FILE_DIR="$ROOT_DIR/configs"
if [ ! -x "$ENV_PREFIX/bin/python" ]; then
  echo "➡️  Creating env from configs/environment.yml…"
  CONDARC="$CONDARC" CONDA_PKGS_DIRS="$CONDA_PKGS_DIRS" CONDA_ENVS_DIRS="$CONDA_ENVS_DIRS" \
  PIP_CACHE_DIR="$PIP_CACHE_DIR" XDG_CACHE_HOME="$XDG_CACHE_HOME" TMPDIR="$TMPDIR" \
    (cd "$ENV_FILE_DIR" && conda env create -p "$ENV_PREFIX" -f environment.yml)
else
  echo "ℹ️  Env already exists; updating from configs/environment.yml…"
  CONDARC="$CONDARC" CONDA_PKGS_DIRS="$CONDA_PKGS_DIRS" CONDA_ENVS_DIRS="$CONDA_ENVS_DIRS" \
  PIP_CACHE_DIR="$PIP_CACHE_DIR" XDG_CACHE_HOME="$XDG_CACHE_HOME" TMPDIR="$TMPDIR" \
    (cd "$ENV_FILE_DIR" && conda env update -p "$ENV_PREFIX" -f environment.yml --prune)
fi

conda activate "$ENV_PREFIX"

# Defensive: ensure pip stays inside this repo for caching/builds
python -m pip install --upgrade pip

echo "🔧 Patching TRL vllm_serve for logprob support"
python "$ROOT_DIR/ops/scripts/patch_trl_vllm_serve.py"

echo "✅ Done. Activate with: conda activate $ENV_PREFIX"
echo "python:  $(which python)"
echo "pip:     $(which pip)"
echo "pip cache: $PIP_CACHE_DIR"
