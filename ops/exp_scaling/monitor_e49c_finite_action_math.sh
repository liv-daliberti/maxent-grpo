#!/usr/bin/env bash
# Refresh the current E49C-only dashboard every 60 seconds.
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
stage="${1:-}"
case "$stage" in
  toy|full) ;;
  *)
    echo "Usage: $0 {toy|full}" >&2
    exit 2
    ;;
esac

cd "$ROOT_DIR"
export MPLCONFIGDIR="$ROOT_DIR/var/cache/matplotlib"
export XDG_CACHE_HOME="$ROOT_DIR/var/cache"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME/fontconfig"
LOG="$ROOT_DIR/var/artifacts/e49c_finite_action_math_${stage}_monitor.log"

while true; do
  {
    printf '[%s] refreshing E49C %s dashboard\n' \
      "$(date '+%F %T %Z')" "$stage"
    python "$ROOT_DIR/ops/exp_scaling/plot_e49c_finite_action_math.py" \
      --stage "$stage" \
      --out "$ROOT_DIR/paper/figures/e49c_finite_action_math_${stage}_live.png"
  } >>"$LOG" 2>&1
  sleep 60
done
