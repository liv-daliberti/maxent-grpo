"""
Copyright 2025 Liv d'Aliberti

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Unit tests for the perf microbenchmark helpers.
"""

from __future__ import annotations

import runpy
from pathlib import Path


def _load_perf_module():
    return runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "tools" / "perf_microbench.py")
    )


def test_compare_to_baseline_pass_and_fail(tmp_path):
    mod = _load_perf_module()
    metrics = {"logging_metrics_ops_per_sec": 100.0}
    baseline = {"logging_metrics_ops_per_sec": 120.0}
    ok, regressions = mod["compare_to_baseline"](metrics, baseline, tolerance=0.3)
    assert ok is True, "Should pass within tolerance"
    ok, regressions = mod["compare_to_baseline"](metrics, baseline, tolerance=0.1)
    assert ok is False and "logging_metrics_ops_per_sec" in regressions


def test_run_benchmarks_produces_positive_value():
    mod = _load_perf_module()
    metrics = mod["run_benchmarks"](iterations=10)
    assert "logging_metrics_ops_per_sec" in metrics
    assert metrics["logging_metrics_ops_per_sec"] > 0