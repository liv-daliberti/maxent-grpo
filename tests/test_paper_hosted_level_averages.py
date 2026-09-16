"""Hosted level summaries use empirical prompt success on complete cohorts."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "hosted_level_averages", ROOT / "ops/build_paper_hosted_level_averages.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


@pytest.fixture(scope="module")
def frozen_record():
    return m.build_record()


def test_exact_five_domain_macro_counts_and_no_source_mutation(frozen_record):
    display = frozen_record["display"]
    before = deepcopy(display)
    rows = m.build_level_averages(display)
    expected = {
        "gpt-5.6-sol": ((627, 1154), (630, 959), (635, 930)),
        "claude-opus-5": ((629, 918), (638, 919), (639, 871)),
        "gpt-5.4": ((628, 1352), (630, 1098), (637, 1077)),
        "grok-4.3": ((637, 1263), (640, 1004), (640, 994)),
        "FW-Kimi-K3": ((630, 1481), (640, 1312), (640, 1216)),
        "claude-opus-4-8": ((632, 1051), (640, 1069), (639, 967)),
        "DeepSeek-V4-Pro": ((630, 1289), (634, 1180), (638, 1036)),
    }
    assert {row["model"] for row in rows} == set(expected)
    for row in rows:
        for level, (success, distinct) in enumerate(expected[row["model"]], start=1):
            result = row["levels"][str(level)]
            assert result["counts"]["domains"] == 5
            assert result["counts"]["prompts"] == 640
            assert result["counts"]["responses"] == 5120
            assert result["counts"]["prompts_with_correct"] == success
            assert result["counts"]["distinct_correct_modes"] == distinct
            assert result["metrics"] == {"pass8": success / 640, "distinct8": distinct / 640}
    assert display == before


@pytest.mark.parametrize("success,correct,distinct", [(0, 0, 0), (1, 8, 1)])
def test_empirical_success_keeps_zero_prompts_and_is_not_accuracy_formula(
        frozen_record, success, correct, distinct):
    display = deepcopy(frozen_record["display"])
    cell = display["cells"][0]
    previous_success = cell["provenance"]["source_counts"]["prompts_with_correct"]
    previous_distinct = cell["counts"]["distinct_correct_modes"]
    cell["provenance"]["source_counts"]["prompts_with_correct"] = success
    cell["counts"]["correct_responses"] = correct
    cell["counts"]["distinct_correct_modes"] = distinct
    cell["metrics"]["accuracy"] = correct / 1024
    cell["metrics"]["distinct8"] = distinct / 128
    result = m.build_level_averages(display)[0]["levels"]["1"]
    original = frozen_record["rows"][0]["levels"]["1"]["counts"]
    assert result["counts"]["prompts"] == 640
    assert result["metrics"]["pass8"] == (
        original["prompts_with_correct"] - previous_success + success) / 640
    assert result["metrics"]["distinct8"] == (
        original["distinct_correct_modes"] - previous_distinct + distinct) / 640
    if success:
        assert success / 128 != pytest.approx(1 - (1 - correct / 1024) ** 8)


@pytest.mark.parametrize("tamper", ["partial_cell", "missing_cell", "duplicate_cell", "success_too_high"])
def test_rejects_incomplete_cohorts_and_invalid_success_counts(frozen_record, tamper):
    display = deepcopy(frozen_record["display"])
    if tamper == "partial_cell":
        display["cells"][0]["counts"]["responses"] -= 1
    elif tamper == "missing_cell":
        display["cells"].pop()
    elif tamper == "duplicate_cell":
        display["cells"][-1] = deepcopy(display["cells"][0])
    else:
        display["cells"][0]["provenance"]["source_counts"]["prompts_with_correct"] = 129
    with pytest.raises(ValueError):
        m.build_level_averages(display)


def test_rejects_source_pass8_drift(tmp_path):
    source = json.loads(m.DEFAULT_SOURCE.read_text())
    cell = source["models"][0]["normalized_secondary"]["cells"]["level1/countdown"]
    cell["metrics"]["pass8"]["estimate"] = 0.5
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(source))
    with pytest.raises(ValueError, match="empirical prompt-success counts"):
        m.build_record(changed)


def test_opus_python_uses_complete_alternative_success_counts(frozen_record):
    display = deepcopy(frozen_record["display"])
    cell = next(cell for cell in display["cells"]
                if cell["model"] == "claude-opus-5" and cell["domain"] == "python_factors"
                and cell["level"] == 3)
    assert cell["provenance"]["condition"] == "plain_direct_expression_no_system"
    assert cell["counts"]["responses"] == 1024
    assert cell["counts"]["correct_responses"] == 1021
    assert cell["provenance"]["source_counts"]["normalized"]["prompts_with_correct_answer"] == 128
    assert cell["provenance"]["source_counts"]["normalized"]["distinct8"] == 181
    cell["provenance"]["source_counts"]["normalized"]["prompts_with_correct_answer"] = 0
    with pytest.raises(ValueError, match="Prompt-success counts"):
        m.build_level_averages(display)


def test_retained_artifacts_reproduce(frozen_record):
    assert json.loads(m.DEFAULT_OUTPUT.with_suffix(".json").read_text()) == frozen_record
    tex = m.render_table(frozen_record)
    assert m.DEFAULT_OUTPUT.with_suffix(".tex").read_text() == tex
    assert tex.count(r"\label{tab:hosted-level-averages-medium}") == 1
    assert all(row["label"] in tex for row in frozen_record["rows"])
