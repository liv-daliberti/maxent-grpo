"""
Validate metric logs emitted by training runs.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable, Sequence

DEFAULT_REQUIRED_KEYS: tuple[str, ...] = (
    "train/loss",
    "train/learning_rate",
    "train/global_step",
    "run/git_sha",
    "run/recipe_path",
)
DEFAULT_PATTERNS: tuple[str, ...] = ("var/artifacts/logs/train-*.log",)


def _extract_json_payload(line: str) -> dict[str, object] | None:
    start = line.find("{")
    if start < 0:
        return None
    try:
        payload = json.loads(line[start:])
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _is_nonfinite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not math.isfinite(float(value))


def validate_logs(
    log_paths: Iterable[Path],
    required_keys: Sequence[str] = DEFAULT_REQUIRED_KEYS,
) -> list[str]:
    """Validate JSON metric payloads in one or more log files."""

    errors: list[str] = []
    required = tuple(required_keys)
    for log_path in log_paths:
        if not log_path.exists():
            errors.append(f"{log_path}: file does not exist")
            continue
        found_payload = False
        for line_number, line in enumerate(
            log_path.read_text(errors="replace").splitlines(),
            start=1,
        ):
            payload = _extract_json_payload(line)
            if payload is None:
                continue
            found_payload = True
            missing = [key for key in required if key not in payload]
            if missing:
                errors.append(
                    f"{log_path}:{line_number}: missing keys: {', '.join(missing)}"
                )
            for key, value in payload.items():
                if key.startswith("train/") and _is_nonfinite_number(value):
                    errors.append(
                        f"{log_path}:{line_number}: non-finite metric for {key}"
                    )
        if not found_payload:
            errors.append(f"{log_path}: no JSON metric payloads found")
    return errors


def _discover_logs(patterns: Sequence[str]) -> list[Path]:
    found: dict[str, Path] = {}
    for pattern in patterns:
        for candidate in Path(".").glob(pattern):
            if candidate.is_file():
                found[str(candidate.resolve())] = candidate
    return sorted(found.values(), key=lambda path: str(path))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pattern",
        action="append",
        dest="patterns",
        default=[],
        help="Glob pattern(s) used to discover log files.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Exit successfully when no log files are found.",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    patterns = tuple(args.patterns) if args.patterns else DEFAULT_PATTERNS
    log_paths = _discover_logs(patterns)
    if not log_paths:
        if args.allow_missing:
            print("No training logs found; skipping validation.")
            return 0
        print("No training logs found.")
        return 1

    errors = validate_logs(log_paths, required_keys=DEFAULT_REQUIRED_KEYS)
    if errors:
        print("Training log validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print(f"Validated {len(log_paths)} training log file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
