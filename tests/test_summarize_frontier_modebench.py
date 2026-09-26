"""Finite-sample inference report estimands and completeness contracts."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SOURCE = Path(__file__).resolve().parents[1] / "ops/summarize_frontier_modebench.py"
spec = importlib.util.spec_from_file_location("frontier_summary_tests", SOURCE)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def row(index=0, domain="graph_coloring", level=1, modes=4):
    return {"level": level, "domain": domain, "row_index": index,
            "metadata": {"answer_mode_count": modes}}


def draws(r, keys, status="completed"):
    return [{**{k: r[k] for k in ("level", "domain", "row_index")},
             "sample_index": i, "verified": key is not None, "canonical_key": key,
             "text": "test", "response_status": status, "usage": {"input_tokens": 2,
             "output_tokens": 3, "output_tokens_details": {"reasoning_tokens": 1}},
             "model": "deployment-version", "latency_seconds": 1.5}
            for i, key in enumerate(keys)]


def write_run(path, rows, responses, errors=()):
    path.mkdir(exist_ok=True)
    for filename, records in (("rows.jsonl", rows), ("samples.jsonl", responses), ("errors.jsonl", errors)):
        (path / filename).write_text("".join(json.dumps(r) + "\n" for r in records))
    (path / "manifest.json").write_text(json.dumps({"model": "gpt-5.6-sol", "reasoning_effort": "medium"}))


def test_conditional_collision_uses_pair_counts_and_support_calibration(tmp_path):
    a, b = row(0, modes=4), row(1, modes=2)
    write_run(tmp_path, [a, b], draws(a, ["a", "a", "b", "b", None, None, None, None]) +
              draws(b, ["c", "c", None, None, None, None, None, None]))
    summary = m.summarize(tmp_path, 300, 12)
    cell = summary["cells"]["level1/graph_coloring"]
    estimates = {name: stat["estimate"] for name, stat in cell["metrics"].items()}
    assert estimates["pass1"] == 3 / 8
    assert estimates["pass8"] == 1
    assert estimates["distinct8"] == 1.5
    assert estimates["correct_pair_collision"] == pytest.approx(3 / 7)
    assert estimates["one_mode_rate"] == .5
    assert estimates["effective_modes_correct"] == 1.5
    assert estimates["correct_repeat_fraction"] == .5
    expected_distinct = (4 * (1 - (3 / 4) ** 4) + 2 * (1 - (1 / 2) ** 2)) / 2
    assert estimates["uniform_expected_distinct8"] == pytest.approx(expected_distinct)
    assert estimates["distinct8_gap_to_uniform"] == pytest.approx(expected_distinct - 1.5)
    assert estimates["uniform_correct_pair_collision"] == pytest.approx(2 / 7)
    assert estimates["correct_pair_collision_excess_uniform"] == pytest.approx(1 / 7)
    assert cell["counts"]["correct_pairs"] == 7
    assert cell["counts"]["colliding_correct_pairs"] == 3
    assert summary["usage_totals"]["output_tokens_details.reasoning_tokens"] == 16
    assert summary["status"] == "complete"


def test_zero_correct_is_not_a_one_mode_prompt_or_an_entropy_observation():
    r = row()
    stat = m.prompt_statistics(r, draws(r, [None] * 8))
    result, _ = m.bootstrap_cell([stat], 40, np.random.default_rng(1))
    assert stat["distinct8"] == 0
    assert stat["correct_pair_collision"] is None
    assert stat["one_mode"] is None
    assert stat["effective_modes_correct"] is None
    assert result["metrics"]["one_mode_rate"]["estimate"] is None
    assert result["metrics"]["effective_modes_correct"]["ci95"] is None
    assert result["metrics"]["distinct8"]["ci95"] == [0, 0]


def test_canonical_json_keys_are_order_independent_and_only_validated_keys_count():
    r = row()
    samples = draws(r, [{"a": [1, 2], "b": 3}, {"b": 3, "a": [1, 2]}, None, None, None, None, None, None])
    samples[2]["canonical_key"] = "invalid_untrusted_key"
    stat = m.prompt_statistics(r, samples)
    assert stat["correct_draws"] == 2
    assert stat["distinct8"] == 1
    assert stat["correct_pair_collision"] == 1
    assert stat["effective_modes_correct"] == 1


def test_missing_api_calls_exclude_incomplete_groups_not_count_as_incorrect(tmp_path):
    a, b = row(0), row(1)
    # An incomplete *returned response* is a scored observation; an API failure is not.
    responses = draws(a, ["a"] * 7 + [None], status="incomplete") + draws(b, [None] * 7)
    write_run(tmp_path, [a, b], responses, [{"error_type": "RateLimitError"}])
    summary = m.summarize(tmp_path, 30)
    assert summary["status"] == "incomplete"
    assert summary["complete_prompts"] == 1
    assert summary["missing_responses"] == 1
    assert summary["excluded_responses_in_incomplete_groups"] == 7
    assert summary["api_error_attempts"] == 1
    assert summary["response_status_counts"]["incomplete"] == 8
    assert summary["cells"]["level1/graph_coloring"]["metrics"]["pass1"]["estimate"] == 7 / 8
    assert summary["incomplete_prompts"][0]["missing_sample_indices"] == [7]


def test_macro_weights_domains_equally_and_bootstrap_preserves_whole_prompts(tmp_path):
    rows = [row(0), row(1), row(0, domain="mathir")]
    samples = draws(rows[0], ["a"] * 8) + draws(rows[1], ["a"] * 8) + draws(rows[2], [None] * 8)
    write_run(tmp_path, rows, samples)
    summary = m.summarize(tmp_path, 50)
    level = summary["levels"]["1"]["metrics"]
    assert level["pass1"]["estimate"] == .5  # Not the pooled 2/3.
    assert level["pass1"]["ci95"] == [.5, .5]
    # Conditional metrics do not silently omit an undefined domain.
    assert level["correct_pair_collision"]["estimate"] is None
    assert level["one_mode_rate"]["estimate"] is None


@pytest.mark.parametrize("mutation", ["duplicate", "unexpected", "out_of_range", "null_verified_key"])
def test_rejects_receipt_corruption(tmp_path, mutation):
    r = row()
    samples = draws(r, ["a"] * 8)
    if mutation == "duplicate":
        samples.append(samples[0])
    elif mutation == "unexpected":
        samples[0]["row_index"] = 99
    elif mutation == "out_of_range":
        samples[0]["sample_index"] = 8
    else:
        samples[0]["canonical_key"] = None
    write_run(tmp_path, [r], samples)
    with pytest.raises(ValueError):
        m.summarize(tmp_path, 10)


def test_support_count_one_and_unknown_support_are_safe():
    r = row(modes=1)
    stat = m.prompt_statistics(r, draws(r, ["a"] * 8))
    assert stat["uniform_expected_distinct_given_correct"] == 1
    assert stat["uniform_one_mode_probability_given_correct"] == 1
    r["metadata"]["support_is_open"] = True
    stat = m.prompt_statistics(r, draws(r, ["a"] * 8))
    result, _ = m.bootstrap_cell([stat], 10, np.random.default_rng(0))
    assert result["metrics"]["uniform_expected_distinct8"]["estimate"] is None
    assert result["metrics"]["distinct8"]["estimate"] == 1


def test_report_artifacts_and_seed_reproducibility(tmp_path):
    rows = [row(i, domain=d, level=level, modes=8) for level in (1, 2, 3)
            for d in m.DOMAIN_ORDER for i in (0, 1)]
    samples = [s for r in rows for s in draws(r, [str(i % (r["level"] + 1)) for i in range(8)])]
    write_run(tmp_path, rows, samples)
    assert m.main(["--input-dir", str(tmp_path), "--bootstrap-replicates", "20"]) == 0
    result = json.loads((tmp_path / "summary.json").read_text())
    again = m.summarize(tmp_path, 20)
    assert result["cells"] == again["cells"]
    assert (tmp_path / "modebench_frontier.png").stat().st_size > 10000
    assert (tmp_path / "modebench_frontier.pdf").stat().st_size > 10000
    report = (tmp_path / "report.md").read_text()
    assert "cannot establish a training-induced collapse" in report
    assert "six-bit support mask" in report
    assert "zero support" in report
    assert "small-divisor conditional chain" in report
    assert "symbolic-program compliance" in report
    assert "interpretation_notes.md" in report
    assert "[100.0, 100.0]" in report


def test_secondary_normalization_is_cached_monotone_and_preserves_primary(tmp_path, monkeypatch):
    from copy import deepcopy
    from types import SimpleNamespace
    import sys
    r = row()
    responses = draws(r, ["a"] * 6 + [None, None])
    write_run(tmp_path, [r], responses)
    calls = []
    def normalize_and_grade(row, text, strict_grade):
        calls.append(strict_grade["sample_index"])
        return {"verified": True, "canonical_key": "b", "normalized_text": "fixed formatting",
                "transformations": ["formatting"]}
    fake = SimpleNamespace(__file__=str(SOURCE), normalize_and_grade=normalize_and_grade)
    monkeypatch.setitem(sys.modules, "frontier_modebench_normalization", fake)
    result = m.summarize(tmp_path, 30)
    primary = deepcopy(result["cells"])
    m.add_normalized_analysis(result, tmp_path, 30, 20260911)
    assert result["cells"] == primary
    secondary = result["normalized_secondary"]
    assert secondary["additional_verified_responses"] == 2
    assert secondary["normalized_verified_responses"] == 8
    assert secondary["cells"]["level1/graph_coloring"]["metrics"]["distinct8"]["estimate"] == 2
    assert calls == [6, 7]
    m.add_normalized_analysis(result, tmp_path, 30, 20260911)
    assert calls == [6, 7]
    assert len(m.read_jsonl(tmp_path / "normalized_samples.jsonl")) == 8
    assert "Formatting normalization (post hoc secondary analysis)" in m.render_report(result)


def test_countdown_library_size_is_not_certified_total_verifier_support():
    r = row(domain="countdown", modes=2)
    stat = m.prompt_statistics(r, draws(r, ["mode" + str(i) for i in range(8)]))
    assert stat["distinct8"] == 8
    assert stat["certified_support_count"] is None
    result, _ = m.bootstrap_cell([stat], 10, np.random.default_rng(0))
    assert result["metrics"]["uniform_expected_distinct8"]["estimate"] is None
    assert result["metrics"]["correct_pair_collision"]["estimate"] == 0


def test_audited_primary_corrections_are_separate_and_unchanged_cache_is_reused(tmp_path, monkeypatch):
    from copy import deepcopy
    from types import SimpleNamespace
    import sys
    r = row()
    original = draws(r, ["a"] * 6 + [None, None])
    write_run(tmp_path, [r], original)
    calls = []
    def normalize_and_grade(row, text, strict_grade):
        calls.append(strict_grade["sample_index"])
        return {"verified": True, "canonical_key": "b", "normalized_text": "fixed formatting",
                "transformations": ["formatting"]}
    monkeypatch.setitem(sys.modules, "frontier_modebench_normalization", SimpleNamespace(__file__=str(SOURCE), normalize_and_grade=normalize_and_grade))
    first = m.summarize(tmp_path, 20)
    m.add_normalized_analysis(first, tmp_path, 20, 20260911)
    assert calls == [6, 7]
    audited = deepcopy(original)
    for sample in audited:
        sample["primary_regrade_corrected"] = sample["sample_index"] == 6
        sample["raw_strict_verified"] = sample["verified"]
    audited[6]["verified"] = True
    audited[6]["canonical_key"] = "b"
    path = tmp_path / "audited_primary_samples.jsonl"
    path.write_text("".join(json.dumps(sample) + "\n" for sample in audited))
    result = m.summarize(tmp_path, 20, primary_samples_path=path)
    m.add_normalized_analysis(result, tmp_path, 20, 20260911)
    assert result["cells"]["level1/graph_coloring"]["metrics"]["pass1"]["estimate"] == 7 / 8
    assert result["primary_grading_audit"]["corrected_records"] == 1
    assert result["normalized_secondary"]["additional_verified_responses"] == 1
    assert calls == [6, 7]  # Auditing metadata alone does not invalidate a grade cache.
    assert len(m.read_jsonl(tmp_path / "normalized_samples.jsonl")) == 9
    assert "operational corrections are separate" in m.render_report(result)


def test_audited_primary_cannot_modify_model_response_text(tmp_path):
    r = row()
    original = draws(r, ["a"] * 8)
    write_run(tmp_path, [r], original)
    original[0]["text"] = "changed model answer"
    path = tmp_path / "audited_primary_samples.jsonl"
    path.write_text("".join(json.dumps(sample) + "\n" for sample in original))
    with pytest.raises(ValueError, match="non-grading field text"):
        m.summarize(tmp_path, 20, primary_samples_path=path)


def test_audited_primary_rejects_stale_partial_sidecar(tmp_path):
    r = row()
    original = draws(r, ["a"] * 8)
    write_run(tmp_path, [r], original)
    path = tmp_path / "audited_primary_samples.jsonl"
    path.write_text("".join(json.dumps(sample) + "\n" for sample in original[:-1]))
    with pytest.raises(ValueError, match="stale or incomplete"):
        m.summarize(tmp_path, 20, primary_samples_path=path)
