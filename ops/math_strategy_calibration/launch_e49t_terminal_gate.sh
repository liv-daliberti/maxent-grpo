#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

MANIFEST="$ROOT_DIR/var/artifacts/e49t_natural_menu_math_toy_05b_3ep_v1_comparative_jobs.tsv"
if [[ ! -f "$MANIFEST" ]]; then
  echo "missing E49T comparative manifest: $MANIFEST" >&2
  exit 1
fi
mapfile -t job_ids < <(awk -F '\t' 'NR > 1 {print $3}' "$MANIFEST")
if [[ "${#job_ids[@]}" != 2 ]] \
  || [[ ! "${job_ids[0]}" =~ ^[0-9]+$ ]] \
  || [[ ! "${job_ids[1]}" =~ ^[0-9]+$ ]]; then
  echo "E49T manifest does not contain exactly two job IDs" >&2
  exit 2
fi

job_id="$(
  sbatch --parsable \
    --dependency="afterok:${job_ids[0]}:${job_ids[1]}" \
    "$ROOT_DIR/ops/slurm/e49t_terminal_gate_node302.slurm"
)"
printf '%s\n' "$job_id" > \
  "$ROOT_DIR/var/artifacts/e49t_terminal_gate_job.txt"
echo "submitted E49T terminal gate job $job_id"
