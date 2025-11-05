#!/usr/bin/env bash
set -euo pipefail

# Submit training.sh ensuring logs/ exists so Slurm writes into logs/
# Usage: scripts/sbatch_training.sh [additional sbatch args...]

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"
mkdir -p logs

# Provide defaults that ensure repo-local logging; allow user overrides after defaults
DEFAULT_OUT="--output=$ROOT_DIR/logs/slurm_%j.out"
DEFAULT_ERR="--error=$ROOT_DIR/logs/slurm_%j.err"
DEFAULT_CWD="--chdir=$ROOT_DIR"

exec sbatch $DEFAULT_OUT $DEFAULT_ERR $DEFAULT_CWD "$@" training.sh
