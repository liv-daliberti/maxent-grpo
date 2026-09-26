"""The main hosted display must preserve denominators and disclose its cohort choice."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("hosted_breadth", ROOT / "ops/plot_paper_hosted_breadth.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


@pytest.fixture(scope="module")
def frozen_record():
    return json.loads(m.DEFAULT_SOURCE.read_text())


def test_all_deployments_domains_levels_and_complete_denominators(frozen_record):
    before = json.dumps(frozen_record, sort_keys=True)
    display = m.build_display_data(frozen_record)
    assert len(display["cells"]) == frozen_record["completed_model_count"] * 15
    assert display["sampling"]["displayed_responses"] == frozen_record["completed_response_count"]
    assert {(cell["model"], cell["domain"], cell["level"]) for cell in display["cells"]} == {
        (model, domain, level) for model in m.admitted_model_order(frozen_record) for domain in m.DOMAIN_ORDER for level in m.LEVELS
    }
    assert all(cell["counts"]["prompts"] == 128 and cell["counts"]["responses"] == 1024
               for cell in display["cells"])
    assert json.dumps(frozen_record, sort_keys=True) == before


def test_only_opus5_python_uses_complete_separate_prompt_cohort(frozen_record):
    display = m.build_display_data(frozen_record)
    alternate = [cell for cell in display["cells"]
                 if cell["provenance"]["condition"] == "plain_direct_expression_no_system"]
    assert {(cell["model"], cell["domain"], cell["level"]) for cell in alternate} == {
        ("claude-opus-5", "python_factors", level) for level in (1, 2, 3)
    }
    assert sum(cell["counts"]["responses"] for cell in alternate) == 3072
    assert sum(cell["counts"]["correct_responses"] for cell in alternate) == 3069
    level3 = next(cell for cell in alternate if cell["level"] == 3)
    assert level3["metrics"]["accuracy"] == 1021 / 1024
    assert level3["metrics"]["accuracy"] < 1  # The three unavailable draws stay in the denominator.
    assert level3["metrics"]["distinct8"] == 181 / 128
    assert level3["provenance"]["source_counts"]["native_refusals"] == 3
    assert all(cell["provenance"]["condition_source"]["selected_responses"] == 3072 for cell in alternate)


def test_every_original_cell_uses_normalized_metrics_not_strict(frozen_record):
    display = m.build_display_data(frozen_record)
    models = {model["model"]: model for model in frozen_record["models"]}
    differences_from_strict = 0
    for cell in display["cells"]:
        assert cell["grading"] == "frozen_formatting_normalized"
        if cell["provenance"]["condition"] != "original_benchmark_prompt":
            continue
        model = models[cell["model"]]
        key = f"level{cell['level']}/{cell['domain']}"
        source = model["normalized_secondary"]["cells"][key]
        assert cell["metrics"]["accuracy"] == source["metrics"]["pass1"]["estimate"]
        assert cell["metrics"]["distinct8"] == source["metrics"]["distinct8"]["estimate"]
        differences_from_strict += cell["metrics"]["accuracy"] != model["cells"][key]["metrics"]["pass1"]["estimate"]
    assert differences_from_strict > 0
    sol_python3 = next(cell for cell in display["cells"]
                       if cell["model"] == "gpt-5.6-sol" and cell["domain"] == "python_factors" and cell["level"] == 3)
    assert sol_python3["metrics"]["accuracy"] == 579 / 1024


def test_each_source_metric_pointer_resolves_to_the_displayed_value(frozen_record):
    for cell in m.build_display_data(frozen_record)["cells"]:
        for metric, pointer in cell["provenance"]["source_metric_fields"].items():
            value = frozen_record
            for part in pointer.lstrip("/").split("/"):
                key = part.replace("~1", "/").replace("~0", "~")
                value = value[int(key)] if isinstance(value, list) else value[key]
            assert value == cell["metrics"][metric]


@pytest.mark.parametrize("field,value", [("responses", 1021), ("prompts", 127)])
def test_rejects_outcome_filtered_alternate_denominator(frozen_record, field, value):
    changed = deepcopy(frozen_record)
    changed["python_prompt_sensitivity"]["levels"]["3"]["plain"]["counts"][field] = value
    with pytest.raises(ValueError, match="retain all 128 prompts"):
        m.build_display_data(changed)


def test_rejects_partial_alternate_cohort(frozen_record):
    changed = deepcopy(frozen_record)
    changed["python_prompt_sensitivity"]["conditions"]["plain"]["selected_responses"] = 3069
    with pytest.raises(ValueError, match="complete 3,072-response cohort"):
        m.build_display_data(changed)


def test_rejects_different_task_rows_for_substitution(frozen_record):
    changed = deepcopy(frozen_record)
    changed["python_prompt_sensitivity"]["conditions"]["plain"]["selected_rows_sha256"] = "changed"
    with pytest.raises(ValueError, match="same task rows"):
        m.build_display_data(changed)


def test_rejects_mixed_normalized_graders(frozen_record):
    changed = deepcopy(frozen_record)
    changed["models"][0]["normalized_secondary"]["normalization_source_sha256"] = "different"
    with pytest.raises(ValueError, match="same frozen normalized grading"):
        m.build_display_data(changed)


def test_rejects_missing_domain_level_cell(frozen_record):
    changed = deepcopy(frozen_record)
    del changed["models"][0]["normalized_secondary"]["cells"]["level1/countdown"]
    with pytest.raises(ValueError, match="all five domains at all three levels"):
        m.build_display_data(changed)


def test_rejects_count_metric_drift(frozen_record):
    changed = deepcopy(frozen_record)
    changed["models"][0]["normalized_secondary"]["cells"]["level3/python_factors"]["metrics"]["pass1"]["estimate"] = 1
    with pytest.raises(ValueError, match="metric does not match complete-cohort counts"):
        m.build_display_data(changed)


def test_rejects_alternate_total_drift(frozen_record):
    changed = deepcopy(frozen_record)
    changed["python_prompt_sensitivity"]["totals"]["plain"]["counts"]["normalized"]["correct_responses"] = 3072
    with pytest.raises(ValueError, match="do not match its complete totals"):
        m.build_display_data(changed)


def test_figure_displays_all_points_without_refusal_labels(frozen_record):
    plt = pytest.importorskip("matplotlib.pyplot")
    fig = m.build_figure(m.build_display_data(frozen_record))
    try:
        assert len(fig.axes) == 10
        assert sum(len(line.get_xdata()) for ax in fig.axes for line in ax.lines) == frozen_record["completed_model_count"] * 30
        assert all(min(ax.get_ylim()) <= y <= max(ax.get_ylim())
                   for ax in fig.axes for line in ax.lines for y in line.get_ydata())
        assert sum(len(ax.artists) for ax in fig.axes) == frozen_record["completed_model_count"] * 2
        assert all(text.get_fontsize() >= 8 for ax in fig.axes
                   for text in [*ax.get_xticklabels(), *ax.get_yticklabels(), ax.title])
        assert not any("refus" in text.get_text().lower()
                       for text in fig.findobj(match=lambda item: hasattr(item, "get_text")))
        assert tuple(fig.get_size_inches()) == (6.4, 3.2)
    finally:
        plt.close(fig)


def test_additional_admitted_model_is_included_without_population_filter(frozen_record):
    changed = deepcopy(frozen_record)
    extra = deepcopy(changed["models"][0])
    extra["model"] = "new-complete-deployment"
    extra["label"] = "New complete deployment"
    changed["models"].insert(0, extra)
    changed["model_icons"][extra["model"]] = deepcopy(changed["model_icons"]["gpt-5.6-sol"])
    changed["completed_model_count"] += 1
    changed["completed_response_count"] += 15360
    display = m.build_display_data(changed)
    assert display["models"][-1]["model"] == "new-complete-deployment"
    extra_cells = [cell for cell in display["cells"] if cell["model"] == extra["model"]]
    assert len(extra_cells) == 15
    assert sum(cell["counts"]["responses"] for cell in extra_cells) == 15360
    assert display["sampling"]["displayed_responses"] == changed["completed_response_count"]
    assert all(cell["provenance"]["condition"] == "original_benchmark_prompt" for cell in extra_cells)


def test_rejects_duplicate_admitted_model(frozen_record):
    changed = deepcopy(frozen_record)
    changed["models"].append(deepcopy(changed["models"][0]))
    changed["completed_model_count"] += 1
    changed["completed_response_count"] += 15360
    with pytest.raises(ValueError, match="duplicate model identities"):
        m.build_display_data(changed)


@pytest.mark.parametrize("field", ["completed_model_count", "completed_response_count"])
def test_rejects_admitted_population_accounting_mismatch(frozen_record, field):
    changed = deepcopy(frozen_record)
    changed[field] += 1
    with pytest.raises(ValueError, match=field):
        m.build_display_data(changed)
