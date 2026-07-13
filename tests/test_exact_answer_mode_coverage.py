from ops.eval_exact_answer_mode_coverage import compute_coverage_metrics
from ops.eval_exact_answer_mode_pareto import (
    _matched_correctness_rows,
    compute_prefix_coverage_metrics,
)


def test_coverage_counts_distinct_correct_exact_answer_modes():
    metrics = compute_coverage_metrics(
        rewards=[1.0, 1.0, 1.0, 0.0, 1.0],
        answer_keys=["a", "a", "b", "z", "c"],
        answer_mode_count=4,
    )

    assert metrics["any_correct_at_k"] == 1.0
    assert metrics["distinct_correct_modes_at_k"] == 3.0
    assert metrics["mode_coverage_at_k"] == 0.75
    assert metrics["all_modes_covered_at_k"] == 0.0


def test_coverage_requires_correct_attempts_for_mode_credit():
    metrics = compute_coverage_metrics(
        rewards=[0.0, 1.0, 0.0, 1.0],
        answer_keys=["a", "b", "c", None],
        answer_mode_count=2,
    )

    assert metrics["distinct_correct_modes_at_k"] == 1.0
    assert metrics["mode_coverage_at_k"] == 0.5
    assert metrics["correct_answer_key_extracted_frac"] == 0.5


def test_prefix_coverage_reuses_one_sample_order_for_multiple_ks():
    metrics_by_k = compute_prefix_coverage_metrics(
        rewards=[0.0, 1.0, 1.0, 1.0],
        answer_keys=["bad", "a", "a", "b"],
        answer_mode_count=4,
        sample_counts=[1, 4, 2],
    )

    assert metrics_by_k[1]["any_correct_at_k"] == 0.0
    assert metrics_by_k[2]["any_correct_at_k"] == 1.0
    assert metrics_by_k[2]["distinct_correct_modes_at_k"] == 1.0
    assert metrics_by_k[4]["distinct_correct_modes_at_k"] == 2.0
    assert metrics_by_k[4]["mode_coverage_at_k"] == 0.5


def test_matched_correctness_uses_best_grpo_coverage_at_equal_or_better_mean():
    rows = [
        {
            "variant": "grpo",
            "split": "multi_answer",
            "temperature": 0.6,
            "sample_count": 8,
            "metrics": {"mean_at_k": 0.50, "mode_coverage_at_k": 0.10},
        },
        {
            "variant": "grpo",
            "split": "multi_answer",
            "temperature": 1.2,
            "sample_count": 32,
            "metrics": {"mean_at_k": 0.30, "mode_coverage_at_k": 0.35},
        },
        {
            "variant": "answer_maxent",
            "split": "multi_answer",
            "temperature": 0.8,
            "sample_count": 16,
            "metrics": {"mean_at_k": 0.40, "mode_coverage_at_k": 0.45},
        },
    ]

    matched = _matched_correctness_rows(rows)

    assert len(matched) == 1
    assert matched[0]["baseline_temperature"] == 0.6
    assert matched[0]["coverage_advantage_at_matched_or_better_correctness"] == 0.35
