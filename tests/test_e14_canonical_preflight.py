from __future__ import annotations

import json

import pytest

from ops.exp_scaling.verify_e14_dataset import validate_e14_rows


def _row(*, hidden_count: int = 3, prompt_suffix: str = "inside \\boxed{}."):
    partial = [None] * hidden_count + [1] * (5 - hidden_count)
    return {
        "problem": (
            "A partial coloring is ???11, where ? means uncolored. "
            "There are exactly 3 question marks. Return exactly one final answer "
            "and no explanation: exactly 3 digits, each one 1, 2, or 3, for the "
            f"missing positions from left to right, {prompt_suffix}"
        ),
        "answer": json.dumps(
            {
                "verifier": "graph_coloring",
                "n": 5,
                "edges": [],
                "partial_colors": partial,
            }
        ),
        "answer_mode_count": 27,
        "answer_mode_split": "eval_multi_answer",
        "modebench_task": "graph_coloring",
    }


def test_e14_dataset_contract_accepts_exactly_three_hidden_actions():
    assert len(validate_e14_rows([_row()], split_tag="multi_answer")) == 1


def test_e14_dataset_contract_rejects_hidden_count_or_prompt_drift():
    with pytest.raises(ValueError, match="three hidden"):
        validate_e14_rows([_row(hidden_count=2)], split_tag="multi_answer")

    with pytest.raises(ValueError, match="prompt contract"):
        validate_e14_rows(
            [_row(prompt_suffix="then explain your answer.")],
            split_tag="multi_answer",
        )


def test_e14_dataset_contract_rejects_grader_count_mismatch_or_no_valid_action():
    mismatched = _row()
    mismatched["answer_mode_count"] = 26
    with pytest.raises(ValueError, match="answer_mode_count mismatch"):
        validate_e14_rows([mismatched], split_tag="multi_answer")

    impossible = _row()
    reference = json.loads(impossible["answer"])
    reference["edges"] = [[4, 5]]
    impossible["answer"] = json.dumps(reference)
    impossible["answer_mode_count"] = 2
    with pytest.raises(ValueError, match="no grader-valid action"):
        validate_e14_rows([impossible], split_tag="multi_answer")
