#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=/n/fs/similarity/maxent-grpo
cd "$ROOT_DIR"

OUTPUT_ROOT="$ROOT_DIR/var/artifacts/e69_math_route_gate1_base_v1"
if [[ -e "$OUTPUT_ROOT" ]]; then
  echo "E69 Gate 1 fixed archive already exists: $OUTPUT_ROOT" >&2
  exit 1
fi

identity="$(
  sha256sum \
    src/oat_drgrpo/math_route.py \
    src/oat_drgrpo/math_grader.py \
    src/oat_drgrpo/templates.py \
    ops/route_successor/sample_e69_math_route_gate1.py \
    paper/preregistration/e69_verified_route_successor_protocol_20260728.md \
    var/data/math12k_384_route_dev128_v1/MATERIALIZATION_MANIFEST.json \
    | sort -k2 \
    | sha256sum \
    | cut -d' ' -f1
)"

job_id="$(
  sbatch \
    --parsable \
    --export=ALL,OAT_E69_GATE1_INPUT_IDENTITY="$identity" \
    "$ROOT_DIR/ops/slurm/e69_math_route_gate1_node302.slurm"
)"
printf 'submitted E69 Gate 1 job %s input_identity=%s\n' \
  "$job_id" "$identity"
