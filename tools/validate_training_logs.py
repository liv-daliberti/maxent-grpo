#!/usr/bin/env python
"""
Validate training logs for expected metrics and numeric values.

The validator scans ``logs/train_*.log`` (or a user-provided glob) for JSON
objects containing training metrics and asserts that:
- Required metric keys are present.
- All numeric values are finite (no NaN/inf).

It is intentionally tolerant:
- If no matching log files are found, it exits success (override with
  ``--require-logs``).
- Lines without JSON payloads are ignored.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set


DEFAULT_PATTERN = "logs/train_*.log"
DEFAULT_REQUIRED_KEYS = {"train/loss", "train/learning_rate", "train/global_step"}


def _extract_json(line: str) -> Dict[str, Any] | None:
    """Return the JSON object embedded in a log line, if any."""
    start = line.find("{")
    end = line.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    snippet = line[start : end + 1]
    try:
        return json.loads(snippet)
    except json.JSONDecodeError:
        return None


def _is_finite_number(value: Any) -> bool:
    """Return True when value is numeric and finite."""
    if isinstance(value, (int, float)):
        return math.isfinite(value)
    return False


def validate_logs(
    paths: Iterable[Path],
    required_keys: Sequence[str] = DEFAULT_REQUIRED_KEYS,
) -> List[str]:
    """Validate metrics emitted in the provided log files.

    :param paths: Iterable of log file paths to scan.
    :param required_keys: Keys that must be present in every metrics payload.
    :returns: List of error strings (empty when validation passes).
    """
    errors: List[str] = []
    required: Set[str] = set(required_keys)
    total_payloads = 0
    for path in paths:
        if not path.exists():
            continue
        for idx, line in enumerate(path.read_text().splitlines(), start=1):
            payload = _extract_json(line)
            if not isinstance(payload, dict):
                continue
            total_payloads += 1
            missing = required - set(payload.keys())
            if missing:
                errors.append(f"{path}:{idx} missing keys: {sorted(missing)}")
            for key, value in payload.items():
                if key.startswith("train/") and isinstance(value, (int, float)):
                    if not _is_finite_number(value):
                        errors.append(f"{path}:{idx} non-finite metric {key}={value}")
                elif key.startswith("train/") and not isinstance(
                    value, (int, float, bool)
                ):
                    errors.append(
                        f"{path}:{idx} non-numeric metric {key} type={type(value).__name__}"
                    )
    if total_payloads == 0:
        errors.append("No metrics payloads found in provided logs")
    return errors


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pattern",
        default=DEFAULT_PATTERN,
        help="Glob pattern for log files (default: %(default)s)",
    )
    parser.add_argument(
        "--required-key",
        action="append",
        default=list(DEFAULT_REQUIRED_KEYS),
        help="Additional required metric keys (can be passed multiple times).",
    )
    parser.add_argument(
        "--require-logs",
        action="store_true",
        help="Fail if no matching log files are found.",
    )
    args = parser.parse_args(argv)

    log_paths = sorted(Path(".").glob(args.pattern))
    if not log_paths and not args.require_logs:
        print(f"No log files matched pattern '{args.pattern}'; skipping validation.")
        return 0
    errors = validate_logs(log_paths, required_keys=args.required_key)
    if errors:
        print("Log validation failed:")
        for err in errors:
            print(f"- {err}")
        return 1
    print(f"Validated {len(log_paths)} log file(s); metrics look OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
