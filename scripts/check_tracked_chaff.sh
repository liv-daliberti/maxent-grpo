#!/usr/bin/env bash

set -euo pipefail

violations=()
while IFS= read -r path; do
  case "$path" in
    var/*|outputs/*|docs/_build/*|_build/*|.cache/*|*.bak|*/__pycache__/*|__pycache__/*)
      violations+=("$path")
      ;;
  esac
done < <(git ls-files)

if ((${#violations[@]} > 0)); then
  echo "Found forbidden tracked files:"
  for path in "${violations[@]}"; do
    echo "  - $path"
  done
  exit 1
fi

echo "Tracked-file chaff check passed."
