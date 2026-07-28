import json
import math
from types import SimpleNamespace

import pytest

from oat_drgrpo.args import ZeroMathArgs
from oat_drgrpo.learner.run import (
    ZeroMathRunMixin,
    _summarize_mode_coverage_draws,
)


def _draw(coverage, any_correct, mean, distinct):
    return {
        "mode_coverage_at_k": coverage,
        "any_correct_at_k": any_correct,
        "mean_at_k": mean,
        "distinct_correct_modes_at_k": distinct,
    }


def test_four_fixed_draws_are_the_default_contract():
    fields = ZeroMathArgs.__dataclass_fields__

    assert fields["eval_mode_coverage_draws"].default == 4
    assert fields["eval_mode_coverage_seed"].default == 1001


def test_repeated_draw_summary_keeps_raw_values_and_uncertainty():
    draws = [
        _draw(0.10, 0.4, 0.20, 0.5),
        _draw(0.20, 0.5, 0.25, 0.8),
        _draw(0.30, 0.6, 0.30, 1.1),
        _draw(0.40, 0.7, 0.35, 1.4),
    ]

    summary = _summarize_mode_coverage_draws(draws, "multi_answer", 8)
    key = "eval/multi_answer/sampled_mode_coverage_at_8"

    assert summary[key] == pytest.approx(0.25)
    assert [summary[f"{key}_draw_{index}"] for index in range(4)] == pytest.approx(
        [0.10, 0.20, 0.30, 0.40]
    )
    expected_std = math.sqrt(0.05 / 3.0)
    assert summary[f"{key}_draw_std"] == pytest.approx(expected_std)
    assert summary[f"{key}_draw_se"] == pytest.approx(expected_std / 2.0)
    assert summary[f"{key}_draw_min"] == 0.10
    assert summary[f"{key}_draw_max"] == 0.40
    assert summary[f"{key}_draw_count"] == 4.0


def test_prompt_level_draw_sidecar_is_durable_jsonl(tmp_path):
    harness = SimpleNamespace(save_path=str(tmp_path))
    record = {
        "step": 96,
        "draw_index": 2,
        "seed": 1003,
        "prompts": [{"prompt_index": 0, "rewards": [1.0, 0.0]}],
    }

    ZeroMathRunMixin._append_mode_coverage_draw_jsonl(harness, record)

    path = tmp_path / "eval_mode_coverage_draws.jsonl"
    assert [json.loads(line) for line in path.read_text().splitlines()] == [record]


def test_sampled_evaluation_retains_greedy_and_four_fixed_draw_traces():
    calls = []
    records = []

    def run_draw(
        _dataset, *, k, temperature, seed, condition_on_answer_options=False
    ):
        calls.append((k, temperature, seed, condition_on_answer_options))
        value = 0.5 if k == 1 else seed / 10_000
        return _draw(value, value, value, value), [{"responses": ["answer"]}]

    harness = SimpleNamespace(
        strategy=SimpleNamespace(is_rank_0=lambda: True),
        eval_dataset_dict={"multi_answer": []},
        _run_sampled_mode_coverage_draw=run_draw,
        _append_mode_coverage_draw_jsonl=records.append,
    )

    metrics = ZeroMathRunMixin._run_sampled_mode_coverage(
        harness,
        [],
        "multi_answer",
        96,
        k=8,
        temperature=1.0,
        draw_count=4,
        seed_base=1001,
    )

    assert calls == [
        (1, 0.0, 0, False),
        (8, 1.0, 1001, False),
        (8, 1.0, 1002, False),
        (8, 1.0, 1003, False),
        (8, 1.0, 1004, False),
    ]
    assert records[0]["evaluation_kind"] == "deterministic_greedy_trace_neutral"
    assert records[0]["sample_count"] == 1
    assert [record["evaluation_kind"] for record in records[1:]] == [
        "fixed_seed_sampled_k_neutral"
    ] * 4
    assert [record["seed"] for record in records[1:]] == [1001, 1002, 1003, 1004]
    assert metrics[
        "eval/multi_answer/sampled_mode_coverage_at_8_draw_count"
    ] == 4.0


def test_diayn_evaluation_separates_neutral_quality_from_latent_binding():
    calls = []
    records = []

    def run_draw(
        _dataset, *, k, temperature, seed, condition_on_answer_options=False
    ):
        calls.append((k, temperature, seed, condition_on_answer_options))
        option_ids = (
            [0, 0, 1, 1, 2, 2, 3, 3]
            if condition_on_answer_options
            else [None] * k
        )
        answers = ["a", "a", "b", "b", "c", "c", "d", "d"][:k]
        rewards = [1.0] * k
        outcome = {
            "prompt_index": 0,
            "answer_keys": answers,
            "option_ids": option_ids,
            "rewards": rewards,
        }
        return _draw(0.5, 1.0, 1.0, 4.0), [outcome]

    harness = SimpleNamespace(
        args=SimpleNamespace(
            diayn_num_options=4,
            diayn_mi_smoothing=1.0,
        ),
        strategy=SimpleNamespace(is_rank_0=lambda: True),
        eval_dataset_dict={"multi_answer": []},
        _run_sampled_mode_coverage_draw=run_draw,
        _append_mode_coverage_draw_jsonl=records.append,
    )

    metrics = ZeroMathRunMixin._run_sampled_mode_coverage(
        harness,
        [],
        "multi_answer",
        0,
        k=8,
        temperature=1.0,
        draw_count=4,
        seed_base=360100,
    )

    assert [call[3] for call in calls] == [False] * 5 + [True] * 4
    assert [record["evaluation_kind"] for record in records] == [
        "deterministic_greedy_trace_neutral",
        *(["fixed_seed_sampled_k_neutral"] * 4),
        *(["fixed_seed_sampled_k_latent_binding"] * 4),
        "crossfit_option_binding",
    ]
    assert "eval/multi_answer/sampled_mean_at_8" in metrics
    assert "eval/multi_answer/sampled_latent_mean_at_8" in metrics
    assert (
        "eval/multi_answer/sampled_option_answer_mi_lower_bound_nats_at_8"
        in metrics
    )
