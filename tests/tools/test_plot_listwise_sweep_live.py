from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import plot_listwise_sweep_live as mod


def test_live_sweep_dashboard_renders_with_partial_results(tmp_path: Path) -> None:
    rows = [
        {
            "tau": 0.35,
            "beta": 0.04,
            "job_id": "1",
            "job_name": "job-a",
            "run_name": "run-a",
            "output_dir": "/tmp/out-a",
            "summary_path": "/tmp/sum-a.json",
            "step": 25,
            "avg": 0.35,
            "pass_at_8_avg": 0.52,
            "mean_at_8_avg": 0.20,
            "avg_len_mean": 3200.0,
            "status": "ok",
            "scheduler_state": "RUNNING",
            "status_bucket": "running_with_summary",
            "status_label": "step 25 RUN",
        },
        {
            "tau": 0.35,
            "beta": 0.08,
            "job_id": "2",
            "job_name": "job-b",
            "run_name": "run-b",
            "output_dir": "/tmp/out-b",
            "summary_path": None,
            "step": None,
            "avg": None,
            "pass_at_8_avg": None,
            "mean_at_8_avg": None,
            "avg_len_mean": None,
            "status": "pending",
            "scheduler_state": "PENDING",
            "status_bucket": "pending",
            "status_label": "PENDING",
        },
        {
            "tau": 0.50,
            "beta": 0.04,
            "job_id": "3",
            "job_name": "job-c",
            "run_name": "run-c",
            "output_dir": "/tmp/out-c",
            "summary_path": "/tmp/sum-c.json",
            "step": 50,
            "avg": 0.38,
            "pass_at_8_avg": 0.55,
            "mean_at_8_avg": 0.23,
            "avg_len_mean": 2800.0,
            "status": "ok",
            "scheduler_state": None,
            "status_bucket": "done",
            "status_label": "step 50 DONE",
        },
        {
            "tau": 0.50,
            "beta": 0.08,
            "job_id": "4",
            "job_name": "job-d",
            "run_name": "run-d",
            "output_dir": "/tmp/out-d",
            "summary_path": None,
            "step": None,
            "avg": None,
            "pass_at_8_avg": None,
            "mean_at_8_avg": None,
            "avg_len_mean": None,
            "status": "pending",
            "scheduler_state": "RUNNING",
            "status_bucket": "running_no_summary",
            "status_label": "RUN",
        },
    ]
    output = tmp_path / "dashboard.svg"
    mod._plot_svg(rows, output, "Test Sweep")
    assert output.exists()
    assert output.stat().st_size > 0

    summary = mod._summary_payload(rows, "Test Sweep")
    assert summary["counts"] == {"done": 1, "running": 2, "pending": 1}
    assert summary["best"]["tau"] == 0.5
    assert summary["best"]["beta"] == 0.04
    assert summary["best"]["step"] == 50
    encoded = json.dumps(summary)
    assert "step 25 RUN" in encoded


def test_live_sweep_dashboard_aggregates_multiple_seeds_per_cell(tmp_path: Path) -> None:
    rows = [
        {
            "tau": 0.35,
            "beta": 0.04,
            "seed": 41,
            "job_id": "1",
            "job_name": "job-a",
            "run_name": "run-a",
            "output_dir": "/tmp/out-a",
            "summary_path": "/tmp/sum-a.json",
            "step": 50,
            "avg": 0.40,
            "pass_at_8_avg": 0.52,
            "mean_at_8_avg": 0.21,
            "avg_len_mean": 3200.0,
            "status": "ok",
            "scheduler_state": None,
            "status_bucket": "done",
            "status_label": "step 50 DONE",
        },
        {
            "tau": 0.35,
            "beta": 0.04,
            "seed": 42,
            "job_id": "2",
            "job_name": "job-b",
            "run_name": "run-b",
            "output_dir": "/tmp/out-b",
            "summary_path": "/tmp/sum-b.json",
            "step": 50,
            "avg": 0.44,
            "pass_at_8_avg": 0.56,
            "mean_at_8_avg": 0.25,
            "avg_len_mean": 3000.0,
            "status": "ok",
            "scheduler_state": None,
            "status_bucket": "done",
            "status_label": "step 50 DONE",
        },
        {
            "tau": 0.50,
            "beta": 0.08,
            "seed": 41,
            "job_id": "3",
            "job_name": "job-c",
            "run_name": "run-c",
            "output_dir": "/tmp/out-c",
            "summary_path": "/tmp/sum-c.json",
            "step": 50,
            "avg": 0.39,
            "pass_at_8_avg": 0.49,
            "mean_at_8_avg": 0.20,
            "avg_len_mean": 2800.0,
            "status": "ok",
            "scheduler_state": None,
            "status_bucket": "done",
            "status_label": "step 50 DONE",
        },
    ]
    aggregated = mod.report.aggregate_rows(rows)
    output = tmp_path / "dashboard.svg"
    mod._plot_svg(aggregated, output, "Test Sweep")
    assert output.exists()

    summary = mod._summary_payload(aggregated, "Test Sweep")
    assert summary["counts"] == {"done": 2, "running": 0, "pending": 0}
    assert summary["best"]["tau"] == 0.35
    assert summary["best"]["beta"] == 0.04
    assert summary["best"]["seed_count"] == 2
    assert summary["best"]["seed_completed_count"] == 2
    assert summary["best"]["avg"] == pytest.approx(0.42)
