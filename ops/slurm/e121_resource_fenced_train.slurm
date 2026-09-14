#!/bin/bash
# E121 runtime fence. Scheduler options are supplied by the audited launcher.

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
record="$(scontrol show job -dd -o "${SLURM_JOB_ID:?}")"
if [[ "${record,,}" == *pvl* ]]; then
  echo "E121 refuses PVL scheduler state" >&2
  exit 121
fi
for value in \
  "${SLURM_JOB_PARTITION:-}" \
  "${SLURM_JOB_ACCOUNT:-}" \
  "${SLURM_JOB_RESERVATION:-}" \
  "${SLURMD_NODENAME:-}" \
  "${CUDA_VISIBLE_DEVICES:-}"; do
  if [[ "${value,,}" == *pvl* ]]; then
    echo "E121 refuses PVL runtime value: $value" >&2
    exit 121
  fi
done

exec bash "${OAT_ZERO_OPS_SNAPSHOT_ROOT:?}/slurm/train_node302.slurm"
