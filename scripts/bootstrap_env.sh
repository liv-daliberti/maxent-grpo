#!/usr/bin/env bash
set -euo pipefail

# Bootstrap a fully local conda + pip environment under the repo.
# - Env prefix: ./openr1
# - Conda caches: ./.conda_pkgs, ./.conda_envs, ./.conda_cache
# - Pip cache: ./.pip_cache with config at ./.pip/pip.conf
# - Tmp + XDG caches: ./.tmp and ./.cache

# Resolve repo root (supports running from anywhere inside the repo)
ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

echo "🔧 Bootstrapping local env in $ROOT_DIR"

# Localized dirs
unset CONDA_ENVS_PATH  # prefer CONDA_ENVS_DIRS to avoid alias conflict
export CONDARC="$ROOT_DIR/.condarc"
export CONDA_PKGS_DIRS="$ROOT_DIR/.conda_pkgs"
export CONDA_ENVS_DIRS="$ROOT_DIR/.conda_envs"
export XDG_CACHE_HOME="$ROOT_DIR/.cache"
export PIP_CACHE_DIR="$ROOT_DIR/.pip_cache"
export PIP_CONFIG_FILE="$ROOT_DIR/.pip/pip.conf"
export TMPDIR="$ROOT_DIR/.tmp"
export PYTHONNOUSERSITE=1

mkdir -p \
  "$CONDA_PKGS_DIRS" "$CONDA_ENVS_DIRS" "$ROOT_DIR/.conda_cache" \
  "$PIP_CACHE_DIR" "$ROOT_DIR/.pip" "$XDG_CACHE_HOME" "$TMPDIR" \
  "$ROOT_DIR/logs"

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

ENV_PREFIX="$ROOT_DIR/openr1"

echo "📦 Conda env prefix: $ENV_PREFIX"
echo "📁 CONDA_PKGS_DIRS:   $CONDA_PKGS_DIRS"
echo "📁 CONDA_ENVS_DIRS:   $CONDA_ENVS_DIRS"
echo "📁 PIP_CACHE_DIR:     $PIP_CACHE_DIR"
echo "📁 XDG_CACHE_HOME:    $XDG_CACHE_HOME"
echo "📁 TMPDIR:            $TMPDIR"

# Create env if missing
if [ ! -x "$ENV_PREFIX/bin/python" ]; then
  echo "➡️  Creating env from environment.yml…"
  CONDARC="$CONDARC" CONDA_PKGS_DIRS="$CONDA_PKGS_DIRS" CONDA_ENVS_DIRS="$CONDA_ENVS_DIRS" \
  PIP_CACHE_DIR="$PIP_CACHE_DIR" XDG_CACHE_HOME="$XDG_CACHE_HOME" TMPDIR="$TMPDIR" \
    conda env create -p "$ENV_PREFIX" -f "$ROOT_DIR/environment.yml"
else
  echo "ℹ️  Env already exists; updating from environment.yml…"
  CONDARC="$CONDARC" CONDA_PKGS_DIRS="$CONDA_PKGS_DIRS" CONDA_ENVS_DIRS="$CONDA_ENVS_DIRS" \
  PIP_CACHE_DIR="$PIP_CACHE_DIR" XDG_CACHE_HOME="$XDG_CACHE_HOME" TMPDIR="$TMPDIR" \
    conda env update -p "$ENV_PREFIX" -f "$ROOT_DIR/environment.yml" --prune
fi

conda activate "$ENV_PREFIX"

# Defensive: ensure pip stays inside this repo for caching/builds
python -m pip install --upgrade pip

echo "✅ Done. Activate with: conda activate $ENV_PREFIX"
echo "python:  $(which python)"
echo "pip:     $(which pip)"
echo "pip cache: $PIP_CACHE_DIR"
