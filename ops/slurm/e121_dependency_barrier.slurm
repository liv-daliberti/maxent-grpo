#!/bin/bash
# E121 bounded fan-in dependency barrier; requests no GPU.

set -euo pipefail
record="$(scontrol show job -dd -o "${SLURM_JOB_ID:?}")"
if [[ "${record,,}" == *pvl* ]]; then
  echo "E121 barrier refuses PVL scheduler state" >&2
  exit 121
fi
for value in \
  "${SLURM_JOB_PARTITION:-}" \
  "${SLURM_JOB_ACCOUNT:-}" \
  "${SLURM_JOB_RESERVATION:-}" \
  "${SLURMD_NODENAME:-}"; do
  if [[ "${value,,}" == *pvl* ]]; then
    echo "E121 barrier refuses PVL runtime value: $value" >&2
    exit 121
  fi
done
printf 'E121 dependency barrier satisfied: %s\n' "${SLURM_JOB_ID}"
