from types import SimpleNamespace

import pytest

from diagnose_real_domains_vllm_20260921 import extract_score, validate_config


def row():
    return {"row_id": "r", "task_id": "dev", "prompt_token_ids": [10, 11],
            "response_token_ids": [20, 151645], "canonical_key": "mode", "origins": ["bank"]}


def output():
    return SimpleNamespace(prompt_token_ids=[10, 11, 20, 151645], prompt_logprobs=[
        None, {11: SimpleNamespace(logprob=-9)},
        {99: SimpleNamespace(logprob=-.1), 20: SimpleNamespace(logprob=-2)},
        {151645: SimpleNamespace(logprob=-3)},
    ])


def test_scores_response_offsets_and_eos_not_top1():
    score = extract_score(row(), output(), "remax_32")
    assert score["token_logprobs"] == [-2, -3]
    assert score["sum_logprob"] == -5
    assert score["mean_logprob"] == -2.5
    assert score["full_token_echo_verified"]


@pytest.mark.parametrize("change", ["echo", "length", "absent", "nan", "positive"])
def test_reject_inexact_or_invalid_scores(change):
    value = output()
    if change == "echo":
        value.prompt_token_ids[0] = 12
    elif change == "length":
        value.prompt_logprobs.pop()
    elif change == "absent":
        del value.prompt_logprobs[2][20]
    else:
        value.prompt_logprobs[2][20].logprob = float("nan") if change == "nan" else .5
    with pytest.raises(ValueError):
        extract_score(row(), value, "base")


def test_fixed_rows_valid_and_duplicate_refused():
    config = {"model": "model", "rows": [row()], "checkpoints": [{"name": "base", "adapter": None}]}
    validate_config(config)
    config["rows"].append(row())
    with pytest.raises(ValueError, match="duplicate"):
        validate_config(config)


@pytest.mark.parametrize("tokens", [[], [True], [-1], [1.2]])
def test_refuse_missing_or_invalid_original_tokens(tokens):
    value = row()
    value["response_token_ids"] = tokens
    with pytest.raises(ValueError):
        validate_config({"model": "model", "rows": [value], "checkpoints": [{"name": "base", "adapter": None}]})
