#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
variant="${1:-}"
mode="${2:-run}"
case "$variant" in
  e50b)
    data="$ROOT_DIR/var/data/e50b_observed_route_math_toy"
    schema=e50b_observed_route_math_toy_v1
    protocol="$ROOT_DIR/paper/preregistration/e50b_observed_route_math_toy_20260726.md"
    prefix=e50b_observed_route_math_toy_05b_3ep_v1
    gate="$ROOT_DIR/var/artifacts/e50b_baseline_route_probe_v1/result.json"
    ;;
  e50d)
    data="$ROOT_DIR/var/data/e50d_teacher_route_math_toy"
    schema=e50d_teacher_route_math_toy_v1
    protocol="$ROOT_DIR/paper/preregistration/e50d_teacher_route_math_toy_20260726.md"
    prefix=e50d_teacher_route_math_toy_05b_3ep_v1
    gate="$ROOT_DIR/var/artifacts/e50d_baseline_route_probe_v1/result.json"
    ;;
  *)
    echo "usage: $0 {e50b|e50d} [config|run]" >&2
    exit 1
    ;;
esac
if [[ "$mode" != config && "$mode" != run ]]; then
  echo "usage: $0 {e50b|e50d} [config|run]" >&2
  exit 1
fi
if ! jq -e \
  '.schema == "e50_route_probe_result_v1"
   and .status == "pass"
   and .baseline_gate == true
   and .sample_count_per_prompt == 64
   and .dual_full_coverage_count >= 8
   and .all_prompts_have_accepted_route == true' \
  "$gate" >/dev/null; then
  echo "$variant requires a passing 64-sample base route-probe gate" >&2
  exit 1
fi

source_hash=3cf26e564b3cafbdc4d0a819febe6d0a61d443746750931c87250f662fcec0a6
ops_hash=782fbefe34db2a0053dcd79d76542b58b31ed70e260518302af71f5a7281c867
source_base="$ROOT_DIR/var/artifacts/source_snapshots/e49t_natural_menu_${source_hash}/src"
source_overlay="$ROOT_DIR/var/artifacts/source_snapshots/e50_math_corrected_overlay_v1/src"
expected_overlay_hash=ba032e9ca300c25385be9650582556f6c8f833ce1f4f5a7197c4259ce5e44a1e
observed_overlay_hash="$(
  "$ROOT_DIR/var/seed_paper_eval/paper310/bin/python" - "$source_overlay" <<'PY'
import hashlib
import pathlib
import sys
root = pathlib.Path(sys.argv[1])
digest = hashlib.sha256()
for path in sorted(
    item
    for item in root.rglob("*")
    if item.is_file()
    and "__pycache__" not in item.parts
    and item.suffix != ".pyc"
):
    digest.update(str(path.relative_to(root)).encode("utf-8"))
    digest.update(b"\0")
    digest.update(hashlib.sha256(path.read_bytes()).digest())
print(digest.hexdigest())
PY
)"
if [[ "$observed_overlay_hash" != "$expected_overlay_hash" ]]; then
  echo "E50 corrected source overlay identity mismatch" >&2
  exit 1
fi
export PYTHONDONTWRITEBYTECODE=1
export E49T_TOY_PROTOCOL="$protocol"
export E49T_TOY_DATA_ROOT="$data"
export E49T_TOY_PREFIX="$prefix"
export E49T_TOY_DATA_SCHEMA="$schema"
export E49T_TOY_TRAIN_MULTI=10
export E49T_TOY_EVAL_MULTI=1
export E49T_SOURCE_ROOT_OVERRIDE="$source_overlay"
export E49T_CANONICALIZER_SOURCE_OVERRIDE="$source_base/oat_drgrpo/math_strategy_canonicalizer.py"
export E49T_OPS_ROOT_OVERRIDE="$ROOT_DIR/var/artifacts/source_snapshots/e49t_natural_menu_ops_${ops_hash}/ops"
export E49T_EXTRA_GATE_RESULT="$gate"
export E49T_WRAPPER_HASH="$(sha256sum "${BASH_SOURCE[0]}" | cut -d' ' -f1)"

bash "$ROOT_DIR/ops/exp_scaling/launch_e49t_natural_menu_math_toy_05b.sh" \
  "$mode"
if [[ "$mode" == config ]]; then
  exit 0
fi

manifest="$ROOT_DIR/var/artifacts/${prefix}_comparative_jobs.tsv"
mapfile -t job_ids < <(awk -F '\t' 'NR > 1 {print $3}' "$manifest")
if [[ ${#job_ids[@]} != 2 ]]; then
  echo "$variant matched manifest does not contain two jobs" >&2
  exit 1
fi
terminal_job="$(sbatch --parsable \
  --dependency="afterany:${job_ids[0]}:${job_ids[1]}" \
  --export="ALL,E50_TERMINAL_VARIANT=$variant" \
  "$ROOT_DIR/ops/slurm/e50_terminal_probes_after_training.slurm")"
printf '%s\n' "$terminal_job" \
  > "$ROOT_DIR/var/artifacts/${variant}_terminal_probe_orchestrator_job.txt"
printf 'queued %s terminal route-probe orchestrator job %s\n' \
  "$variant" "$terminal_job"
