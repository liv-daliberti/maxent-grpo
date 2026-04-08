from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import render_seed_eval_multiseed_curves as mod


def _write_summary(
    path: Path,
    *,
    sampling_seed: int,
    avg: float,
    pass_at_8_avg: float,
    mean_at_8_avg: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "sampling_seed": sampling_seed,
                "avg": avg,
                "pass_at_8_avg": pass_at_8_avg,
                "mean_at_8_avg": mean_at_8_avg,
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_render_seed_eval_multiseed_curves_aggregates_and_plots(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    for seed in (0, 1):
        for template, offset in (
            ("no", 0.00),
            ("qwen_math", 0.05),
            ("r1", 0.10),
        ):
            _write_summary(
                results_root
                / f"seed_{seed:03d}"
                / "step0"
                / template
                / "seed_paper_eval_sharded.summary.json",
                sampling_seed=seed,
                avg=0.40 + offset + (0.10 * seed),
                pass_at_8_avg=0.55 + offset + (0.05 * seed),
                mean_at_8_avg=0.30 + offset + (0.04 * seed),
            )
            _write_summary(
                results_root
                / f"seed_{seed:03d}"
                / "step16"
                / template
                / "seed_paper_eval_sharded.summary.json",
                sampling_seed=seed,
                avg=0.45 + offset + (0.10 * seed),
                pass_at_8_avg=0.60 + offset + (0.05 * seed),
                mean_at_8_avg=0.35 + offset + (0.04 * seed),
            )

    records = mod.load_summary_records(results_root)
    payload = mod.aggregate_records(records, results_root=results_root)

    assert payload["seeds"] == [0, 1]
    assert payload["templates"] == ["no", "qwen_math", "r1"]
    no_pass1_points = payload["metrics"]["pass_at_1"]["templates"]["no"]
    assert [point["step"] for point in no_pass1_points] == [0, 16]
    assert no_pass1_points[0]["mean"] == pytest.approx(0.45)
    assert no_pass1_points[0]["seed_count"] == 2
    assert no_pass1_points[0]["ci_low"] < no_pass1_points[0]["mean"]
    assert no_pass1_points[0]["ci_high"] > no_pass1_points[0]["mean"]

    svg_path = tmp_path / "curves.svg"
    grid_svg_path = tmp_path / "curves_template_grid.svg"
    summary_path = tmp_path / "curves.summary.json"
    tsv_path = tmp_path / "curves.tsv"
    mod.write_summary_json(payload, summary_path)
    mod.write_rows_tsv(payload["rows"], tsv_path)
    mod.plot_curves(payload, svg_path, title="Synthetic multiseed curves")
    mod.plot_template_metric_grid(
        payload,
        grid_svg_path,
        title="Synthetic multiseed curves",
    )

    assert svg_path.exists()
    assert grid_svg_path.exists()
    assert summary_path.exists()
    assert tsv_path.exists()
    assert "pass_at_8" in tsv_path.read_text(encoding="utf-8")
