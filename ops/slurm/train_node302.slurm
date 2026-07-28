#!/bin/bash
# Run one Dr.GRPO/xDr.GRPO experiment arm on a single GPU.
#
# Override with, for example:
#   sbatch --export=ALL,OAT_ZERO_VARIANT=grpo ops/slurm/train_node302.slurm
#   sbatch --export=ALL,OAT_ZERO_VARIANT=xdr,OAT_ZERO_XDR_TAU=0.05 ops/slurm/train_node302.slurm

#SBATCH --job-name=xdr_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --partition=mltheory
#SBATCH --account=mltheory
#SBATCH --nodelist=node302
#SBATCH --time=24:00:00
#SBATCH --output=var/artifacts/logs/%x-%j.out
#SBATCH --error=var/artifacts/logs/%x-%j.err

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
README_PYTHON="${ROOT_DIR}/var/seed_paper_eval/paper310/bin/python"
HOST_TAG="${SLURMD_NODENAME:-$(hostname -s)}"

RUNTIME_OPS_ROOT="${OAT_ZERO_OPS_SNAPSHOT_ROOT:-${ROOT_DIR}/ops}"
export OAT_ZERO_REPO_ROOT="$ROOT_DIR"
source "${RUNTIME_OPS_ROOT}/repo_env.sh"
for required_script in run_experiment.sh train.sh resolve_eval_cadence.py; do
  if [[ ! -f "${RUNTIME_OPS_ROOT}/${required_script}" ]]; then
    echo "Missing immutable runtime ops file: ${RUNTIME_OPS_ROOT}/${required_script}" >&2
    exit 1
  fi
done

export OAT_ZERO_PYTHON="${OAT_ZERO_PYTHON:-${README_PYTHON}}"
# Requeues must execute the same Python objective that was first allocated.
# Prefer the submission-time campaign snapshot. For ad hoc submissions, make
# one persistent per-job snapshot on first allocation and reuse it on every
# Slurm restart instead of importing a subsequently edited working tree.
if [[ -z "${OAT_ZERO_SOURCE_ROOT:-}" ]]; then
  SOURCE_SNAPSHOT_PARENT="${ROOT_DIR}/var/artifacts/source_snapshots/jobs/${SLURM_JOB_ID:-manual}"
  SOURCE_SNAPSHOT_ROOT="${SOURCE_SNAPSHOT_PARENT}/src"
  if [[ ! -f "${SOURCE_SNAPSHOT_ROOT}/oat_drgrpo/__init__.py" ]]; then
    mkdir -p "$SOURCE_SNAPSHOT_PARENT"
    SOURCE_SNAPSHOT_TMP="$(mktemp -d "${SOURCE_SNAPSHOT_PARENT}/.source.XXXXXX")"
    mkdir -p "${SOURCE_SNAPSHOT_TMP}/src"
    cp -a "${ROOT_DIR}/src/." "${SOURCE_SNAPSHOT_TMP}/src/"
    if ! mv "${SOURCE_SNAPSHOT_TMP}/src" "$SOURCE_SNAPSHOT_ROOT" 2>/dev/null; then
      # A simultaneous restart may have won the atomic publish race.
      [[ -f "${SOURCE_SNAPSHOT_ROOT}/oat_drgrpo/__init__.py" ]] || exit 1
    fi
    rmdir "$SOURCE_SNAPSHOT_TMP" 2>/dev/null || true
  fi
  export OAT_ZERO_SOURCE_ROOT="$SOURCE_SNAPSHOT_ROOT"
fi
if [[ ! -f "${OAT_ZERO_SOURCE_ROOT}/oat_drgrpo/__init__.py" ]]; then
  echo "Missing immutable Python source snapshot: ${OAT_ZERO_SOURCE_ROOT}" >&2
  exit 1
fi
# repo_env.sh has already added the mutable checkout to PYTHONPATH. Put the
# immutable snapshot first; later repo_env.sh calls leave this ordering intact.
export PYTHONPATH="${OAT_ZERO_SOURCE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export OAT_ZERO_N_GPU="${OAT_ZERO_N_GPU:-1}"
export OAT_ZERO_ENABLE_FLASH_ATTN="${OAT_ZERO_ENABLE_FLASH_ATTN:-0}"
export OAT_ZERO_USE_WB="${OAT_ZERO_USE_WB:-0}"
export OAT_ZERO_LOCAL_ROOT="${OAT_ZERO_LOCAL_ROOT:-/tmp/${USER}/maxent-grpo-${HOST_TAG}}"
export OAT_ZERO_SHM_SIZE_MB="${OAT_ZERO_SHM_SIZE_MB:-8000}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${OAT_ZERO_LOCAL_ROOT}/triton_autotune}"
mkdir -p "$TRITON_CACHE_DIR"
export WANDB__SERVICE_WAIT="${WANDB__SERVICE_WAIT:-300}"
export WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-300}"
export WANDB_START_METHOD="${WANDB_START_METHOD:-thread}"

# Bash can read a script incrementally. Running directly from a shared working
# tree means an in-place edit during a multi-hour job can splice two versions
# together and turn a successful training run into a shutdown-time parse
# failure. Snapshot the two shell entry points once per allocation/restart.
RUNTIME_SCRIPT_DIR="${OAT_ZERO_LOCAL_ROOT}/job_scripts/${SLURM_JOB_ID:-manual}.${SLURM_RESTART_COUNT:-0}"
mkdir -p "$RUNTIME_SCRIPT_DIR"
cp "${RUNTIME_OPS_ROOT}/run_experiment.sh" "$RUNTIME_SCRIPT_DIR/run_experiment.sh"
cp "${RUNTIME_OPS_ROOT}/train.sh" "$RUNTIME_SCRIPT_DIR/train.sh"
chmod 755 "$RUNTIME_SCRIPT_DIR/run_experiment.sh" "$RUNTIME_SCRIPT_DIR/train.sh"
export OAT_ZERO_OPS_SNAPSHOT_ROOT="$RUNTIME_OPS_ROOT"
export OAT_ZERO_TRAIN_SCRIPT="$RUNTIME_SCRIPT_DIR/train.sh"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  export OAT_ZERO_WATCHDOG_LOG_PATH="$ROOT_DIR/var/artifacts/logs/xdr_train-${SLURM_JOB_ID}.out"
fi

echo "[slurm] host=$(hostname)"
echo "[slurm] job_id=${SLURM_JOB_ID:-}"
echo "[slurm] oat_zero_python=${OAT_ZERO_PYTHON}"
echo "[slurm] oat_zero_source_root=${OAT_ZERO_SOURCE_ROOT}"
echo "[slurm] oat_zero_ops_snapshot_root=${OAT_ZERO_OPS_SNAPSHOT_ROOT}"
echo "[slurm] model=${OAT_ZERO_MODEL:-qwen2.5-0.5b-instruct}"
echo "[slurm] variant=${OAT_ZERO_VARIANT:-xdr}"
echo "[slurm] oat_zero_n_gpu=${OAT_ZERO_N_GPU}"
echo "[slurm] oat_zero_use_wb=${OAT_ZERO_USE_WB}"
echo "[slurm] oat_zero_local_root=${OAT_ZERO_LOCAL_ROOT}"
echo "[slurm] vllm_use_v1=${VLLM_USE_V1:-unset}"
echo "[slurm] hf_home=${HF_HOME:-unset}"
echo "[slurm] wandb_dir=${WANDB_DIR:-unset}"

exec "$RUNTIME_SCRIPT_DIR/run_experiment.sh"
