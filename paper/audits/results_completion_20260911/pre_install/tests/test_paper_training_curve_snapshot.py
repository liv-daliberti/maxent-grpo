"""Scientific integrity checks for the frozen primary-method training histories."""
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ops"))
from exp_scaling import build_paper_training_curve_snapshot as curves


def draw(step, index, value=0.5, prompts=128):
    return {"step": step, "draw_index": index, "evaluation_kind": "fixed_seed_sampled_k_neutral",
            "sample_count": 8, "temperature": 1.0, "benchmark": "test", "seed": 100 + index,
            "prompts": [{}] * prompts,
            "metrics": {"any_correct_at_k": value, "distinct_correct_modes_at_k": value + 0.25}}


def task(tmp_path, rows, **run_fields):
    directory = tmp_path / "run"
    path = directory / "debug_job123" / "eval_mode_coverage_draws.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return {"level": "level1", "scale": "qwen05b", "domain": "graph_coloring", "method": "maxrl",
            "seed": 43, "ledger": "fixture", "run": {"run_dir": str(directory), "job_id": 123,
            "domain": "graph_coloring", "seed": 43, **run_fields}}


def test_four_draws_have_exact_metric_means_and_line_provenance(tmp_path):
    rows = [draw(0, i, i / 4) for i in range(4)]
    cell = curves.collect_cell(task(tmp_path, rows + [rows[0]]))
    assert cell["complete_steps"] == [0]
    checkpoint = cell["complete_checkpoints"]["0"]
    assert checkpoint["mean_metrics"] == {"any_correct_at_k": 0.375, "distinct_correct_modes_at_k": 0.625}
    assert [x["line"] for x in checkpoint["draws"][0]["origins"]] == [1, 5]
    assert cell["source_files"][0]["read_bytes"] == cell["source_files"][0]["size_before"]


@pytest.mark.parametrize("fault", ["missing_draw", "wrong_prompt_count", "wrong_k", "nonfinite", "conflict", "off_grid"])
def test_bad_checkpoint_is_a_gap_not_a_selected_retry(tmp_path, fault):
    rows = [draw(192, i) for i in range(4)]
    if fault == "missing_draw":
        rows.pop()
    elif fault == "wrong_prompt_count":
        rows[1]["prompts"].pop()
    elif fault == "wrong_k":
        rows[1]["sample_count"] = 4
    elif fault == "nonfinite":
        rows[1]["metrics"]["any_correct_at_k"] = float("nan")
    elif fault == "conflict":
        rows.append(draw(192, 0, 0.75))
    else:
        for row in rows:
            row["step"] = 193
    cell = curves.collect_cell(task(tmp_path, rows))
    assert not cell["complete_steps"]
    assert 192 in cell["missing_registered_steps"]


def test_documented_superseded_attempts_are_not_pooled(tmp_path, monkeypatch):
    from exp_scaling import plot_paper_aligned_domain_strips as source_reader
    monkeypatch.setattr(source_reader, "ROOT", tmp_path)
    data = task(tmp_path, [draw(192, i) for i in range(4)], replaced_job_ids=[122])
    old = Path(data["run"]["run_dir"]) / "debug_job122" / "eval_mode_coverage_draws.jsonl"
    old.parent.mkdir()
    old.write_text("".join(json.dumps(draw(0, i)) + "\n" for i in range(4)))
    cell = curves.collect_cell(data)
    assert cell["complete_steps"] == [192]
    assert cell["source_binding"]["excluded_sources"][0]["outcome_value_selected"] is False



def test_census_authorized_continuations_preserve_both_halves(tmp_path):
    data = task(tmp_path, [draw(192, i) for i in range(4)])
    old = Path(data["run"]["run_dir"]) / "debug_job122" / "eval_mode_coverage_draws.jsonl"
    old.parent.mkdir()
    old.write_text("".join(json.dumps(draw(0, i)) + "\n" for i in range(4)))
    data["authorized_sources"] = [{"path": str(path), "sha256": curves.digest(path.read_bytes())}
                                  for path in Path(data["run"]["run_dir"]).glob("debug_job*/*.jsonl")]
    cell = curves.collect_cell(data)
    assert cell["complete_steps"] == [0, 192]
    assert len(cell["source_files"]) == 2
    assert not cell["source_binding"]["excluded_sources"]


def test_undocumented_retry_fails_closed(tmp_path):
    data = task(tmp_path, [draw(0, i) for i in range(4)])
    rogue = Path(data["run"]["run_dir"]) / "debug_job999" / "eval_mode_coverage_draws.jsonl"
    rogue.parent.mkdir()
    rogue.write_text(json.dumps(draw(0, 0)) + "\n")
    with pytest.raises(RuntimeError, match="unregistered evaluation source"):
        curves.collect_cell(data)


def test_fixed_seed_cohort_never_shrinks_to_fill_a_gap():
    prefix = ("level1", "qwen05b", "graph_coloring")
    checkpoint = {"mean_metrics": {"any_correct_at_k": 0.5, "distinct_correct_modes_at_k": 1.25}}
    index = {(*prefix, "maxrl", seed): {"complete_checkpoints": records} for seed, records in
             [(43, {"0": checkpoint, "192": checkpoint}), (44, {"0": checkpoint})]}
    result = curves.series(index, prefix, "maxrl", [43, 44], "paired")
    assert result["points"][0]["mean"] == {"pass8": 0.5, "distinct8": 1.25}
    assert result["points"][1]["mean"] == {"pass8": None, "distinct8": None}
    assert result["points"][1]["missing_seeds"] == [44]
    assert result["points"][1]["per_seed"] == {"43": {"pass8": 0.5, "distinct8": 1.25}}


def test_frozen_main_paper_cohorts_and_terminal_metrics_reconstruct():
    snapshot = json.loads(curves.OUTPUT.read_text())
    figure = json.loads(curves.FIGURE.read_text())
    result = curves.validate_snapshot(snapshot, figure)
    assert result["registered_cells"] == 400
    assert result["panels"] == 20
    assert sum(len(p["paired_cohorts"]["drgrpo"]) for p in snapshot["panels"] if p["level"] == "level1") == 74
    assert sum(len(p["paired_cohorts"]["maxrl"]) for p in snapshot["panels"] if p["level"] == "level1") == 67
    pantry = next(p for p in snapshot["panels"] if (p["level"], p["domain"]) == ("level2", "pantry_plan"))
    assert pantry["paired_cohorts"] == {"drgrpo": [43, 46], "maxrl": []}
    assert pantry["supplementary_methods"]["maxrl"]["cohort_n"] > 0
    assert pantry["supplementary_methods"]["replay_maxrl"]["cohort_n"] > 0
    snapshot["panels"][0]["methods"]["drgrpo"]["points"][-1]["mean"]["pass8"] += 0.01
    with pytest.raises(RuntimeError, match="means/gaps"):
        curves.validate_snapshot(snapshot, figure)
