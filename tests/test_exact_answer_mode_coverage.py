from ops.eval_exact_answer_mode_coverage import _pairwise_deltas, compute_coverage_metrics
import math


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


def test_open_support_reports_nonseed_discoveries_without_fake_coverage():
    metrics = compute_coverage_metrics(
        rewards=[1.0, 1.0, 1.0, 0.0],
        answer_keys=["seed", "new-a", "new-a", "new-b"],
        answer_mode_count=0,
        public_seed_key="seed",
    )

    assert metrics["distinct_correct_modes_at_k"] == 2.0
    assert metrics["distinct_nonseed_correct_modes_at_k"] == 1.0
    assert metrics["any_nonseed_correct_at_k"] == 1.0
    assert math.isnan(metrics["mode_coverage_at_k"])
    assert math.isnan(metrics["all_modes_covered_at_k"])


def test_pairwise_deltas_compare_named_landed_arms():
    summaries = [
        {
            "alias": "grpo_s43",
            "splits": {"multi": {"metrics": {"mode_coverage_at_k": 0.25}}},
        },
        {
            "alias": "xdr_tau0p05_s43",
            "splits": {"multi": {"metrics": {"mode_coverage_at_k": 0.40}}},
        },
    ]

    rows = _pairwise_deltas(summaries)

    assert len(rows) == 1
    assert abs(rows[0]["mode_coverage_at_k_delta"] - 0.15) < 1e-12
