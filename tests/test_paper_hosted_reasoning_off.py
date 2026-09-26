"""Publication admits complete, separate control conditions and empirical counts."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "paper_reasoning_off", ROOT / "ops/build_paper_hosted_reasoning_off.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


@pytest.fixture(scope="module")
def source():
    return json.loads(m.DEFAULT_SOURCE.read_text())


@pytest.fixture(scope="module")
def labels():
    catalog = json.loads(m.ICON_SOURCE.read_text())
    return {model["model"]: model["label"] for model in catalog["models"]}


def test_complete_metrics_derive_from_prompt_counts_and_keep_conditions(source, labels):
    rows = m.build_rows(source, labels)
    expected_overall = {
        "gpt-5.6-sol": ((471, 777), (361, 473)),
        "gpt-5.4": ((473, 890), (358, 565)),
        "grok-4.3": ((480, 815), (415, 719)),
        "FW-Kimi-K3": ((475, 989), (449, 780)),
        "claude-opus-4-8": ((477, 776), (466, 734)),
    }
    assert {row["model"] for row in rows} == set(expected_overall)
    for row in rows:
        for condition, (success, modes) in zip(("on", "off"), expected_overall[row["model"]]):
            overall = row["overall"][condition]
            assert overall["counts"]["prompts"] == 480
            assert overall["counts"]["responses"] == 3840
            assert overall["metrics"] == {"pass8": success / 480, "distinct8": modes / 480}
            for level in m.LEVELS:
                item = row["levels"][str(level)][condition]
                cells = [row["analyses"]["normalized"][condition]["cells"][f"level{level}/{d}"]
                         for d in m.DOMAINS]
                assert item["counts"]["prompts"] == 160
                assert item["counts"]["responses"] == 1280
                assert all(c["counts"]["prompts"] == 32 and c["counts"]["responses"] == 256 for c in cells)
                assert item["metrics"]["pass8"] == sum(c["counts"]["prompts_with_correct"] for c in cells) / 160
                assert item["metrics"]["distinct8"] == sum(c["counts"]["distinct_correct_modes"] for c in cells) / 160
            # Empirical pass@8 is not the independence approximation from accuracy.
            accuracy = overall["counts"]["correct_responses"] / 3840
            if condition == "off":
                assert overall["metrics"]["pass8"] != pytest.approx(1 - (1 - accuracy) ** 8)


@pytest.mark.parametrize("tamper", ["not_admitted", "mixed_protocol", "changed_prompt", "selected_outputs",
                                   "wrong_draws", "wrong_off_control", "wrong_budget", "wrong_baseline",
                                   "missing_prompt", "nonfirst_prompt", "missing_outcome", "false_success",
                                   "wrong_metric", "missing_cell", "wrong_difference"])
def test_admission_rejects_partial_or_mixed_conditions(source, tamper):
    model = deepcopy(source["models"][0])
    cohort = model["analyses"]["normalized"]["off"]
    if tamper == "not_admitted":
        model["status"] = "pending_collection"
    elif tamper == "mixed_protocol":
        model["protocol"]["schema"] = "medium"
    elif tamper == "changed_prompt":
        model["protocol"]["all_prompt_bytes_unchanged"] = False
    elif tamper == "selected_outputs":
        model["protocol"]["selection_uses_outputs"] = True
    elif tamper == "wrong_draws":
        model["protocol"]["samples_per_prompt"] = 7
    elif tamper == "wrong_off_control":
        model["settings"]["off_request_controls"]["reasoning"] = {"effort": "medium"}
    elif tamper == "wrong_budget":
        model["settings"]["off_request_controls"]["max_output_tokens"] = 4096
    elif tamper == "wrong_baseline":
        model["settings"]["on"]["reasoning_effort"] = "none"
    elif tamper == "missing_prompt":
        cohort["prompts"].pop()
    elif tamper == "nonfirst_prompt":
        cohort["prompts"][-1]["row_index"] = 32
    elif tamper == "missing_outcome":
        cohort["prompts"][0]["correct_responses"] = None
    elif tamper == "false_success":
        cohort["prompts"][0]["pass8"] = 0
    elif tamper == "wrong_metric":
        cohort["levels"]["1"]["metrics"]["pass8"] = 0.99
    elif tamper == "missing_cell":
        del cohort["cells"]["level1/countdown"]
    else:
        model["analyses"]["normalized"]["off_minus_on"]["1"]["pass8"] = 0.0
    with pytest.raises(ValueError):
        m.validate_model(model)


@pytest.mark.parametrize("tamper", ["duplicate_model", "missing_disposition", "score_incomplete"])
def test_pending_models_cannot_be_silently_dropped_or_scored(source, labels, tamper):
    altered = deepcopy(source)
    if tamper == "duplicate_model":
        altered["models"][1] = deepcopy(altered["models"][0])
    elif tamper == "missing_disposition":
        altered["pending_models"].pop()
    else:
        altered["pending_models"][0]["scores_admitted"] = True
    with pytest.raises(ValueError):
        m.build_rows(altered, labels)


def test_publication_rejects_changed_model_admission_receipt(source, tmp_path):
    altered = deepcopy(source)
    altered["models"][0]["returned_sampling"]["off"]["temperature"] = {"999": 3840}
    path = tmp_path / "changed_summary.json"
    path.write_text(json.dumps(altered))
    with pytest.raises(ValueError, match="differs from the model admission receipt"):
        m.build_record(path)
