#!/usr/bin/env bash
set -euo pipefail

# Create a local-only conda env using caches and dirs under the repo

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CONDARC="$ROOT_DIR/.condarc"
export CONDA_PKGS_DIRS="$ROOT_DIR/.conda_pkgs"
export CONDA_ENVS_DIRS="$ROOT_DIR/.conda_envs"
export PIP_CACHE_DIR="$ROOT_DIR/.pip_cache"
export PIP_CONFIG_FILE="$ROOT_DIR/.pip/pip.conf"
export TMPDIR="$ROOT_DIR/.tmp"

mkdir -p "$CONDA_PKGS_DIRS" "$CONDA_ENVS_DIRS" "$PIP_CACHE_DIR" "$TMPDIR" "$ROOT_DIR/.pip"

# Load conda (adjust if your system differs)
if [ -f /usr/local/anaconda3/2024.02/etc/profile.d/conda.sh ]; then
  source /usr/local/anaconda3/2024.02/etc/profile.d/conda.sh
elif command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
else
  echo "Conda not found on PATH" >&2
  exit 1
fi

# Install the env at a fixed local prefix
ENV_DIR="$ROOT_DIR/openr1"
conda env create -p "$ENV_DIR" -f "$ROOT_DIR/environment.yml"
echo "Env created at: $ENV_DIR"
echo "Activate with: conda activate $ENV_DIR"

