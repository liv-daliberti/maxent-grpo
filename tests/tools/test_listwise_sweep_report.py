from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_MODULE_PATH = (
    Path(__file__).resolve().parents[2] / "tools" / "listwise_sweep_report.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "maxent_test_listwise_sweep_report",
    _MODULE_PATH,
)
assert _SPEC is not None and _SPEC.loader is not None
report = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = report
_SPEC.loader.exec_module(report)


def test_listwise_sweep_report_picks_best_available_summary(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text(
        "\n".join(
            [
                "tau\tbeta\tjob_id\tjob_name\trun_name\toutput_dir",
                "0.35\t0.04\t111\tjob-a\trun-a\t/tmp/out-a",
                "0.50\t0.08\t222\tjob-b\trun-b\t/tmp/out-b",
                "0.70\t0.12\t333\tjob-c\trun-c\t/tmp/out-c",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    results_root = tmp_path / "results"
    for run_name, step, avg, pass8, mean8 in (
        ("run-a", 100, 0.40, 0.52, 0.21),
        ("run-b", 100, 0.43, 0.49, 0.24),
    ):
        run_dir = results_root / run_name / f"step-{step:06d}"
        run_dir.mkdir(parents=True)
        (run_dir / "seed_paper_eval_20260322T000000Z.summary.json").write_text(
            json.dumps(
                {
                    "avg": avg,
                    "pass_at_8_avg": pass8,
                    "mean_at_8_avg": mean8,
                    "avg_lens": {"aime": 800.0, "math": 900.0},
                }
            ),
            encoding="utf-8",
        )

    entries = report.load_manifest(manifest)
    rows = report.aggregate_rows([report.build_row(entry, results_root) for entry in entries])
    rows.sort(key=report.rank_key, reverse=True)

    assert rows[0]["run_name"] == "run-b"
    assert rows[0]["tau"] == 0.5
    assert rows[0]["beta"] == 0.08
    assert rows[0]["pass_at_1_avg"] == pytest.approx(0.43)
    assert rows[-1]["run_name"] == "run-c"
    assert rows[-1]["status"] == "pending"


def test_listwise_sweep_report_aggregates_seed_repeats(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text(
        "\n".join(
            [
                "tau\tbeta\tseed\tjob_id\tjob_name\trun_name\toutput_dir",
                "0.35\t0.04\t41\t111\tjob-a\trun-a\t/tmp/out-a",
                "0.35\t0.04\t42\t222\tjob-b\trun-b\t/tmp/out-b",
                "0.50\t0.08\t41\t333\tjob-c\trun-c\t/tmp/out-c",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    results_root = tmp_path / "results"
    for run_name, step, avg, pass8, mean8 in (
        ("run-a", 50, 0.40, 0.52, 0.21),
        ("run-b", 50, 0.44, 0.56, 0.25),
        ("run-c", 50, 0.39, 0.49, 0.20),
    ):
        run_dir = results_root / run_name / f"step-{step:06d}"
        run_dir.mkdir(parents=True)
        (run_dir / "seed_paper_eval_20260322T000000Z.summary.json").write_text(
            json.dumps(
                {
                    "avg": avg,
                    "pass_at_8_avg": pass8,
                    "mean_at_8_avg": mean8,
                    "avg_lens": {"aime": 800.0, "math": 900.0},
                }
            ),
            encoding="utf-8",
        )

    rows = report.aggregate_rows(
        [report.build_row(entry, results_root) for entry in report.load_manifest(manifest)]
    )
    rows.sort(key=report.rank_key, reverse=True)

    assert len(rows) == 2
    assert rows[0]["tau"] == 0.35
    assert rows[0]["beta"] == 0.04
    assert rows[0]["seed_count"] == 2
    assert rows[0]["seed_completed_count"] == 2
    assert rows[0]["seeds"] == [41, 42]
    assert rows[0]["avg"] == pytest.approx(0.42)
    assert rows[0]["pass_at_1_avg"] == pytest.approx(0.42)
