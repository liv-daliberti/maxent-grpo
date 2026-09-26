#!/bin/bash
# Execute the maintained single-GPU trainer without inheriting its historical
# node302/mltheory scheduler directives.  Command-line campaign constraints
# remain authoritative for jobs submitted to the shared `all` partition.

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=36G
#SBATCH --partition=all
#SBATCH --account=mltheory
#SBATCH --time=04:00:00

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-${OAT_ZERO_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}}"
RUNTIME_OPS_ROOT="${OAT_ZERO_OPS_SNAPSHOT_ROOT:-${ROOT_DIR}/ops}"

exec bash "${RUNTIME_OPS_ROOT}/slurm/train_node302.slurm"
