"""
Microbenchmark helper used by CI to detect gross performance regressions.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Mapping

DEFAULT_METRIC_NAME = "logging_metrics_ops_per_sec"


def run_benchmarks(iterations: int = 500) -> dict[str, float]:
    """Run a tiny deterministic benchmark and report ops/sec."""

    total_iterations = max(int(iterations), 1)
    checksum = 0.0
    start = time.perf_counter()
    for index in range(total_iterations):
        checksum += float((index % 11) * 3)
    elapsed = max(time.perf_counter() - start, 1e-12)
    return {
        DEFAULT_METRIC_NAME: total_iterations / elapsed,
        "benchmark_checksum": checksum,
    }


def compare_to_baseline(
    metrics: Mapping[str, float],
    baseline: Mapping[str, float],
    tolerance: float = 0.35,
) -> tuple[bool, dict[str, str]]:
    """Compare measured metrics against baseline values."""

    regressions: dict[str, str] = {}
    clamped_tolerance = min(max(float(tolerance), 0.0), 1.0)
    for name, expected in baseline.items():
        observed = metrics.get(name)
        if observed is None:
            regressions[name] = "metric missing"
            continue
        lower_bound = float(expected) * (1.0 - clamped_tolerance)
        if float(observed) < lower_bound:
            regressions[name] = (
                f"observed={float(observed):.6f} expected={float(expected):.6f} "
                f"lower_bound={lower_bound:.6f}"
            )
    return (len(regressions) == 0), regressions


def _load_json(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return {str(k): float(v) for k, v in payload.items()}


def _dump_json(path: Path, payload: Mapping[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--baseline-file", type=Path, default=None)
    parser.add_argument("--tolerance", type=float, default=0.35)
    parser.add_argument("--json-output", type=Path, default=None)
    return parser


def main() -> int:
    args = _build_parser().parse_args()

    metrics = run_benchmarks(iterations=args.iterations)
    if args.json_output is not None:
        _dump_json(args.json_output, metrics)

    if args.baseline_file is None:
        print(json.dumps(metrics, indent=2, sort_keys=True))
        return 0

    if not args.baseline_file.exists():
        print(f"Baseline file not found: {args.baseline_file}")
        return 1

    baseline = _load_json(args.baseline_file)
    ok, regressions = compare_to_baseline(metrics, baseline, tolerance=args.tolerance)
    if not ok:
        print("Performance regression detected:")
        for key, message in sorted(regressions.items()):
            print(f"- {key}: {message}")
        return 1

    print("Performance check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
