#!/bin/bash
# E120 runtime fence. Scheduler options are supplied by the audited launcher.

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
record="$(scontrol show job -dd -o "${SLURM_JOB_ID:?}")"
if [[ "${record,,}" == *pvl* ]]; then
  echo "E120 refuses PVL scheduler state" >&2
  exit 120
fi
for value in   "${SLURM_JOB_PARTITION:-}"   "${SLURM_JOB_ACCOUNT:-}"   "${SLURM_JOB_RESERVATION:-}"   "${SLURMD_NODENAME:-}"   "${CUDA_VISIBLE_DEVICES:-}"; do
  if [[ "${value,,}" == *pvl* ]]; then
    echo "E120 refuses PVL runtime value: $value" >&2
    exit 120
  fi
done

TRAIN_WRAPPER="${OAT_ZERO_OPS_SNAPSHOT_ROOT:?}/slurm/train_node302.slurm"
if [[ "${E120_SMOKE:-0}" == "1" ]]; then
  bash "$TRAIN_WRAPPER"
  exec "${ROOT_DIR}/var/seed_paper_eval/paper310/bin/python"     "${OAT_ZERO_OPS_SNAPSHOT_ROOT}/exp_scaling/audit_e120_frequency_smoke.py"     "${SAVE_PATH:?}"
fi
exec bash "$TRAIN_WRAPPER"

