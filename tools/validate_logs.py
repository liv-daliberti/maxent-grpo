#!/usr/bin/env python
"""
Validate training log files for well-formed metrics.

Default behaviour:
- Scan glob ``logs/train_*.log``.
- Flag any metric values that are NaN/Inf.
- Optionally ensure required metrics appear at least once.

Exit codes:
- 0: no files found (non-strict) or all files passed validation.
- 1: validation errors detected.
"""

from __future__ import annotations

import argparse
import glob
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

# Match key/value pairs like "train/loss=0.1" or "train/loss: 0.1"
METRIC_PATTERN = re.compile(
    r"(train/[A-Za-z0-9_\-/\.]+)\s*[:=]\s*([+\-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+\-]?\d+)?|nan|NaN|inf|Inf|INF|Infinity|-Infinity)"
)


@dataclass
class MetricError:
    path: Path
    line_no: int
    key: str
    value: str
    message: str


def _parse_metrics(line: str) -> Iterable[Tuple[str, float, str]]:
    """Yield (key, parsed_value, raw_value) tuples from a log line."""
    for key, raw in METRIC_PATTERN.findall(line):
        try:
            val = float(raw)
        except ValueError:
            # Skip unparsable values; let the caller decide on required keys.
            continue
        yield key, val, raw


def validate_file(
    path: Path, require_keys: Sequence[str]
) -> Tuple[List[MetricError], Dict[str, int]]:
    """Validate a single log file and return errors plus hit counts per key."""
    errors: List[MetricError] = []
    hits: Dict[str, int] = {}
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for idx, line in enumerate(f, 1):
            for key, val, raw in _parse_metrics(line):
                hits[key] = hits.get(key, 0) + 1
                if math.isnan(val) or math.isinf(val):
                    errors.append(
                        MetricError(
                            path=path,
                            line_no=idx,
                            key=key,
                            value=raw,
                            message="metric is NaN/Inf",
                        )
                    )
    if hits and require_keys:
        for key in require_keys:
            if key not in hits:
                errors.append(
                    MetricError(
                        path=path,
                        line_no=0,
                        key=key,
                        value="",
                        message="required metric not found in log",
                    )
                )
    return errors, hits


def run_validation(
    paths: Sequence[Path], require_keys: Sequence[str]
) -> List[MetricError]:
    all_errors: List[MetricError] = []
    for path in paths:
        errors, _hits = validate_file(path, require_keys=require_keys)
        all_errors.extend(errors)
    return all_errors


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate training log metrics.")
    parser.add_argument(
        "--glob",
        default="logs/train_*.log",
        help="Glob pattern of log files to validate (default: logs/train_*.log)",
    )
    parser.add_argument(
        "--require-key",
        action="append",
        default=["train/loss", "train/learning_rate"],
        help="Metric keys that must appear at least once (can be repeated).",
    )
    parser.add_argument(
        "--strict-missing",
        action="store_true",
        help="Fail when no log files match the glob.",
    )
    args = parser.parse_args(argv)

    paths = sorted(Path(p) for p in glob.glob(args.glob))
    if not paths:
        msg = f"No log files matched glob {args.glob}; skipping."
        print(msg, file=sys.stderr)
        return 1 if args.strict_missing else 0

    errors = run_validation(paths, require_keys=args.require_key)
    if errors:
        for err in errors:
            location = f"{err.path}"
            if err.line_no:
                location = f"{location}:{err.line_no}"
            print(
                f"[ERROR] {location} {err.key} ({err.value}) - {err.message}",
                file=sys.stderr,
            )
        return 1

    print(f"Validated {len(paths)} log file(s); no NaN/Inf metrics found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
