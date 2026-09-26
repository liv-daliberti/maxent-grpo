#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${E113R4_RUN_SCRIPT:-}" || ! -f "$E113R4_RUN_SCRIPT" ]]; then
  echo "E113R4_RUN_SCRIPT does not identify the frozen runner" >&2
  exit 2
fi

bash "$E113R4_RUN_SCRIPT"

