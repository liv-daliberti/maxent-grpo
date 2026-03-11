"""Validate plain-text training logs for required metrics and finite values."""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

_METRIC_RE = re.compile(r"([A-Za-z0-9_./-]+)\s*[:=]\s*([^\s,]+)")


@dataclass(frozen=True)
class ValidationError:
    """Represents a single log-validation failure."""

    path: Path
    line: int
    key: str
    value: str
    message: str


def _is_nonfinite(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"nan", "+nan", "-nan", "inf", "+inf", "-inf", "infinity"}:
        return True
    try:
        parsed = float(value)
    except ValueError:
        return False
    return not math.isfinite(parsed)


def validate_file(
    path: Path,
    require_keys: Sequence[str],
) -> Tuple[List[ValidationError], Dict[str, int]]:
    """Validate one log file and return errors + required-key hit counts."""

    required = list(require_keys)
    hits = {key: 0 for key in required}
    errors: List[ValidationError] = []
    if not path.exists():
        errors.append(
            ValidationError(
                path=path,
                line=0,
                key="",
                value="",
                message="File does not exist",
            )
        )
        return errors, hits

    for line_num, line in enumerate(
        path.read_text(encoding="utf-8", errors="replace").splitlines(),
        start=1,
    ):
        for match in _METRIC_RE.finditer(line):
            key = match.group(1)
            value = match.group(2)
            if key in hits:
                hits[key] += 1
            if key.startswith("train/") and _is_nonfinite(value):
                errors.append(
                    ValidationError(
                        path=path,
                        line=line_num,
                        key=key,
                        value=value,
                        message="NaN/Inf metric value",
                    )
                )

    for key, count in hits.items():
        if count <= 0:
            errors.append(
                ValidationError(
                    path=path,
                    line=0,
                    key=key,
                    value="",
                    message="Missing required metric key",
                )
            )
    return errors, hits


def run_validation(
    log_paths: Iterable[Path],
    require_keys: Sequence[str],
) -> List[ValidationError]:
    """Validate multiple log files and return all errors."""

    all_errors: List[ValidationError] = []
    for log_path in log_paths:
        file_errors, _hits = validate_file(Path(log_path), require_keys=require_keys)
        all_errors.extend(file_errors)
    return all_errors


def _discover_logs(patterns: Sequence[str]) -> List[Path]:
    discovered: Dict[str, Path] = {}
    for pattern in patterns:
        for candidate in Path(".").glob(pattern):
            if candidate.is_file():
                discovered[str(candidate.resolve())] = candidate
    return sorted(discovered.values(), key=str)


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
        "--require-key",
        action="append",
        dest="required_keys",
        default=[],
        help="Metric key that must appear at least once.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Exit successfully when no log files are found.",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    patterns = tuple(args.patterns) if args.patterns else ("var/artifacts/logs/*.log",)
    required_keys = tuple(args.required_keys) if args.required_keys else ("train/loss",)
    log_paths = _discover_logs(patterns)
    if not log_paths:
        if args.allow_missing:
            print("No training logs found; skipping validation.")
            return 0
        print("No training logs found.")
        return 1
    errors = run_validation(log_paths, require_keys=required_keys)
    if errors:
        print("Training log validation failed:")
        for err in errors:
            line_part = f":{err.line}" if err.line > 0 else ""
            detail = f" ({err.value})" if err.value else ""
            print(f"- {err.path}{line_part} {err.key}: {err.message}{detail}")
        return 1
    print(f"Validated {len(log_paths)} training log file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
