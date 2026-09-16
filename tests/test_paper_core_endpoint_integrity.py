"""Scientific endpoint readers must never choose among conflicting retries."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest


BUILDER = Path(__file__).resolve().parents[1] / "ops/exp_scaling/build_paper_core_terminal_endpoints.py"


@pytest.fixture
def core():
    spec = importlib.util.spec_from_file_location("core_endpoint_integrity_test", BUILDER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def draws(step=3072):
    return [
        {
            "evaluation_kind": "fixed_seed_sampled_k_neutral",
            "step": step,
            "draw_index": index,
            "metrics": {
                "any_correct_at_k": .4 + index * .1,
                "mean_at_k": .2 + index * .05,
                "distinct_correct_modes_at_k": .8 + index * .2,
            },
        }
        for index in range(4)
    ]


def write_rows(run_dir, rows, job=1):
    path = run_dir / f"debug_job{job}" / "eval_mode_coverage_draws.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def approve(core, monkeypatch, root, run_dir, source):
    amendment = root / "existing_integrity_amendment.md"
    amendment.write_text("Exclude this source from all efficacy values; select no retry.\n")
    monkeypatch.setattr(core, "ROOT", root)
    monkeypatch.setattr(core, "APPROVED_RUN_EXCLUSIONS", {
        str(run_dir.relative_to(root)): {
            "job_id": 1,
            "source_log": str(source.relative_to(root)),
            "source_log_sha256": core.sha256(source),
            "amendment": str(amendment.relative_to(root)),
            "reason": "conflicting duplicate sampled rows in the registered job",
            "scope": "all efficacy endpoints and training steps for this run",
        }
    })
    return amendment


def test_complete_endpoint_records_source_and_draw_provenance(core, tmp_path):
    run = tmp_path / "normal"
    source = write_rows(run, draws(0) + draws())
    audit = []
    result = core.sampled_endpoint(run, step=3072, audit=audit)
    assert result == pytest.approx({"pass8": .55, "mean8": .275, "distinct8": 1.1})
    assert audit[0]["status"] == "admitted"
    assert audit[0]["sources"] == [{"path": str(source), "sha256": core.sha256(source)}]
    assert [row["line"] for row in audit[0]["unique_draw_records"]] == [5, 6, 7, 8]
    assert audit[0]["identical_metric_repeats"] == []


def test_identical_retries_do_not_change_endpoint_or_count_as_extra_draws(core, tmp_path):
    run = tmp_path / "identical"
    original = draws()
    write_rows(run, original + [copy.deepcopy(original[0])] * 5)
    write_rows(run, copy.deepcopy(original), job=2)
    audit = []
    result = core.sampled_endpoint(run, step=3072, audit=audit)
    assert result == pytest.approx({"pass8": .55, "mean8": .275, "distinct8": 1.1})
    assert len(audit[0]["unique_draw_records"]) == 4
    assert len(audit[0]["identical_metric_repeats"]) == 9


@pytest.mark.parametrize("field", ["any_correct_at_k", "mean_at_k", "distinct_correct_modes_at_k"])
def test_unknown_conflict_aborts_instead_of_selecting_first_last_or_average(core, tmp_path, field):
    run = tmp_path / "unapproved"
    original = draws()
    conflicting = copy.deepcopy(original)
    conflicting[2]["metrics"][field] += .01
    write_rows(run, original)
    write_rows(run, conflicting, job=2)
    with pytest.raises(RuntimeError, match=f"conflicting duplicate.*draw=2.*{field}"):
        core.sampled_endpoint(run, step=3072)


def test_extra_metric_conflict_is_not_hidden_by_same_headline_endpoints(core, tmp_path):
    run = tmp_path / "extra_metric"
    original = draws()
    repeated = copy.deepcopy(original[0])
    repeated["metrics"]["other_scientific_measurement"] = 1
    write_rows(run, original + [repeated])
    with pytest.raises(RuntimeError, match="other_scientific_measurement"):
        core.sampled_endpoint(run, step=3072)


@pytest.mark.parametrize("step", [0, 3072])
def test_approved_conflicting_run_is_excluded_at_every_step_without_selected_values(core, monkeypatch, tmp_path, step):
    run = tmp_path / "excluded"
    original = draws()
    conflicting = copy.deepcopy(original)
    conflicting[0]["metrics"]["any_correct_at_k"] += .01
    source = write_rows(run, draws(0) + original + conflicting)
    amendment = approve(core, monkeypatch, tmp_path, run, source)
    audit = []
    assert core.sampled_endpoint(run, step=step, audit=audit) is None
    exclusion = audit[0]
    assert exclusion["status"] == "excluded"
    assert exclusion["step"] == step
    assert exclusion["outcome_value_selected"] is False
    assert exclusion["source_log_sha256"] == core.sha256(source)
    assert exclusion["amendment_sha256"] == core.sha256(amendment)
    assert not ({"metrics", "pass8", "mean8", "distinct8"} & exclusion.keys())


def test_approved_source_hash_change_is_not_treated_as_an_implicit_repair(core, monkeypatch, tmp_path):
    run = tmp_path / "excluded"
    source = write_rows(run, draws())
    approve(core, monkeypatch, tmp_path, run, source)
    source.write_text(source.read_text() + "\n")
    with pytest.raises(RuntimeError, match="source hash drifted"):
        core.sampled_endpoint(run, step=3072)


@pytest.mark.parametrize("count", [0, 3])
def test_missing_or_incomplete_endpoint_is_unavailable(core, tmp_path, count):
    run = tmp_path / "incomplete"
    if count:
        write_rows(run, draws()[:count])
    audit = []
    assert core.sampled_endpoint(run, step=3072, audit=audit) is None
    assert audit[0]["status"] == "incomplete"
    assert audit[0]["observed_draws"] == list(range(count))


def test_build_preserves_valid_control_and_emits_structured_exclusion(core, monkeypatch, tmp_path):
    good_control = tmp_path / "control"
    good_replay = tmp_path / "replay58"
    bad_replay = tmp_path / "replay59"
    write_rows(good_control, draws())
    write_rows(good_replay, draws())
    source = write_rows(bad_replay, draws())
    approve(core, monkeypatch, tmp_path, bad_replay, source)
    ledger = tmp_path / "ledger.json"
    ledger.write_text(json.dumps({
        "released": True, "target_steps": 3072, "train_rows": 384, "passes": 8,
        "runs": [
            {"domain": "countdown", "arm": "control", "seed": 59, "job_id": 2, "run_dir": str(good_control)},
            {"domain": "countdown", "arm": "replay", "seed": 58, "job_id": 3, "run_dir": str(good_replay)},
            {"domain": "countdown", "arm": "replay", "seed": 59, "job_id": 1, "run_dir": str(bad_replay)},
        ],
    }))
    monkeypatch.setattr(core, "SOURCES", {"Falcon3-1B": ("falcon1b", ledger)})
    result = core.build()
    methods = result["models"]["Falcon3-1B"]["domains"]["countdown"]["methods"]
    assert set(methods["control"]["per_seed"]) == {"59"}
    assert set(methods["replay"]["per_seed"]) == {"58"}
    assert len(result["exclusions"]) == 1
    assert result["exclusions"][0]["seed"] == 59
    assert result["exclusions"][0]["arm"] == "replay"
    assert result["exclusions"][0]["outcome_value_selected"] is False
    assert len(result["endpoint_audit"]) == 3
