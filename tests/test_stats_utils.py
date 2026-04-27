from __future__ import annotations

import torch

from oat_drgrpo.stats_utils import (
    LISTWISE_OPTIONAL_LOGGING_METRIC_DEFAULTS,
    all_gather_variable_length_1d_tensor,
    concat_vector_stats,
    compute_correctness_group_rate_infos,
    entropy_split_means,
    fill_missing_scalar_info_defaults,
    finalize_listwise_info_stats,
    finalize_row_sharded_info_stats,
    mean_scalar_stats_by_key,
    resolve_stat_device,
    stack_scalar_stats,
    summarize_listwise_core_scalar_stats,
    summarize_listwise_weight_entropy_stats,
    summarize_semantic_prompt_diagnostics,
    tensor_correlation_or_zero,
)


def test_compute_correctness_group_rate_infos_splits_group_types():
    correctness_grouped = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    valid_row_mask_grouped = torch.tensor(
        [
            [True, True, True],
            [True, True, True],
            [True, True, True],
            [True, False, False],
        ],
        dtype=torch.bool,
    )

    infos = compute_correctness_group_rate_infos(
        correctness_grouped=correctness_grouped,
        valid_row_mask_grouped=valid_row_mask_grouped,
    )

    torch.testing.assert_close(
        infos["listwise_all_wrong_group_frac"],
        torch.tensor(0.25),
    )
    torch.testing.assert_close(
        infos["listwise_mixed_correctness_group_frac"],
        torch.tensor(0.25),
    )
    torch.testing.assert_close(
        infos["listwise_all_correct_group_frac"],
        torch.tensor(0.5),
    )
    torch.testing.assert_close(
        infos["listwise_any_correct_group_frac"],
        torch.tensor(0.75),
    )


def test_stack_scalar_stats_accepts_tensors_and_python_numbers():
    values = [torch.tensor(1.5), 2.0]

    stacked = stack_scalar_stats(values)

    torch.testing.assert_close(stacked, torch.tensor([1.5, 2.0]))


def test_concat_vector_stats_flattens_tensors_and_python_numbers():
    values = [torch.tensor([1.0, 2.0]), 3.0]

    concatenated = concat_vector_stats(values)

    torch.testing.assert_close(concatenated, torch.tensor([1.0, 2.0, 3.0]))


def test_mean_scalar_stats_by_key_aggregates_present_keys_only():
    summaries = mean_scalar_stats_by_key(
        {
            "present": [torch.tensor(1.0), 3.0],
            "empty": [],
        },
        ("present", "empty", "missing"),
    )

    assert set(summaries) == {"present"}
    torch.testing.assert_close(summaries["present"], torch.tensor(2.0))


def test_resolve_stat_device_prefers_tensor_stats_then_fallback():
    stats = {"metric": [torch.tensor(1.0)]}

    assert resolve_stat_device(stats) == torch.device("cpu")
    assert resolve_stat_device({}, fallback=torch.empty(1)) == torch.device("cpu")
    assert resolve_stat_device({}, fallback=torch.device("cpu")) == torch.device("cpu")
    assert resolve_stat_device({}) == torch.device("cpu")


def test_fill_missing_scalar_info_defaults_preserves_existing_values():
    infos = {"present": torch.tensor(2.0)}

    fill_missing_scalar_info_defaults(
        infos,
        {"present": 0.0, "missing": 1.5},
        device=torch.device("cpu"),
    )

    torch.testing.assert_close(infos["present"], torch.tensor(2.0))
    torch.testing.assert_close(infos["missing"], torch.tensor(1.5))
    assert infos["missing"].dtype == torch.float32


def test_summarize_listwise_weight_entropy_stats_adds_aliases_and_ranges():
    summaries = summarize_listwise_weight_entropy_stats(
        {
            "listwise_weight_entropy": [torch.tensor(1.0), 3.0],
            "listwise_weight_entropy_min": [torch.tensor(0.5), 0.25],
            "listwise_weight_entropy_max": [torch.tensor(2.0), 4.0],
            "listwise_weight_entropy_all": [torch.tensor(2.0), 4.0],
            "listwise_weight_entropy_all_min": [torch.tensor(1.0), 0.5],
            "listwise_weight_entropy_all_max": [torch.tensor(3.0), 5.0],
        }
    )

    torch.testing.assert_close(summaries["listwise_weight_entropy"], torch.tensor(2.0))
    torch.testing.assert_close(summaries["weight_entropy"], torch.tensor(2.0))
    torch.testing.assert_close(
        summaries["listwise_weight_entropy_active"],
        torch.tensor(2.0),
    )
    torch.testing.assert_close(
        summaries["listwise_weight_entropy_min"],
        torch.tensor(0.25),
    )
    torch.testing.assert_close(
        summaries["weight_entropy_min"],
        torch.tensor(0.25),
    )
    torch.testing.assert_close(
        summaries["listwise_weight_entropy_max"],
        torch.tensor(4.0),
    )
    torch.testing.assert_close(
        summaries["weight_entropy_max"],
        torch.tensor(4.0),
    )
    torch.testing.assert_close(
        summaries["listwise_weight_entropy_all"],
        torch.tensor(3.0),
    )
    torch.testing.assert_close(
        summaries["listwise_weight_entropy_all_min"],
        torch.tensor(0.5),
    )
    torch.testing.assert_close(
        summaries["listwise_weight_entropy_all_max"],
        torch.tensor(5.0),
    )


def test_summarize_listwise_core_scalar_stats_adds_core_metrics_and_aliases():
    summaries = summarize_listwise_core_scalar_stats(
        {
            "logprobs_diff_max": [torch.tensor(0.5), 1.5],
            "logprobs_diff_min": [torch.tensor(-0.5), -1.5],
            "zero_pg_loss_count": [torch.tensor(0.0), 2.0],
            "clip_ratio_low": [torch.tensor(0.1), 0.3],
            "clip_ratio_high": [torch.tensor(0.2), 0.4],
            "clip_ratio_region": [torch.tensor(0.3), 0.5],
            "pg_clipfrac": [torch.tensor(0.25), 0.75],
        },
        advantages=torch.tensor([1.0, 2.0, 4.0]),
        grouped_reward_values=torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0],
            ]
        ),
    )

    torch.testing.assert_close(summaries["logprobs_diff_max"], torch.tensor(1.5))
    torch.testing.assert_close(summaries["logprobs_diff_min"], torch.tensor(-1.5))
    torch.testing.assert_close(summaries["zero_pg_loss_count"], torch.tensor(1.0))
    torch.testing.assert_close(summaries["clip_ratio_low"], torch.tensor(0.2))
    torch.testing.assert_close(summaries["clip_ratio_high"], torch.tensor(0.3))
    torch.testing.assert_close(summaries["clip_ratio_region"], torch.tensor(0.4))
    torch.testing.assert_close(summaries["pg_clipfrac"], torch.tensor(0.5))
    torch.testing.assert_close(summaries["adv_mean"], torch.tensor(7.0 / 3.0))
    torch.testing.assert_close(summaries["adv_min"], torch.tensor(1.0))
    torch.testing.assert_close(summaries["adv_max"], torch.tensor(4.0))
    torch.testing.assert_close(summaries["all_zero_rewards_count"], torch.tensor(1.0))
    torch.testing.assert_close(
        summaries["listwise_all_zero_rewards_count"],
        torch.tensor(1.0),
    )
    torch.testing.assert_close(summaries["all_one_rewards_count"], torch.tensor(1.0))
    torch.testing.assert_close(
        summaries["listwise_all_one_rewards_count"],
        torch.tensor(1.0),
    )


def test_finalize_listwise_info_stats_combines_summaries_and_defaults():
    finalized = finalize_listwise_info_stats(
        {"existing": torch.tensor(9.0)},
        {
            "policy_grad_norm": [torch.tensor(1.0), 3.0],
            "logprobs_diff_max": [torch.tensor(0.5)],
            "logprobs_diff_min": [torch.tensor(-0.5)],
            "zero_pg_loss_count": [torch.tensor(0.0)],
            "listwise_semantic_cluster_entropy": [torch.tensor(0.7)],
        },
        {
            "listwise_semantic_entropy_prompt": [torch.tensor([0.1, 0.9])],
            "listwise_semantic_exploration_gain_any_correct_prompt": [
                torch.tensor([1.0, 3.0])
            ],
            "listwise_semantic_exploration_gain_drgrpo_prompt": [
                torch.tensor([2.0, 4.0])
            ],
        },
        advantages=torch.tensor([1.0, 3.0]),
        grouped_reward_values=torch.tensor([[0.0, 0.0], [1.0, 1.0]]),
    )

    assert list(finalized) == sorted(finalized)
    torch.testing.assert_close(finalized["existing"], torch.tensor(9.0))
    torch.testing.assert_close(finalized["policy_grad_norm"], torch.tensor(3.0))
    torch.testing.assert_close(
        finalized["listwise_semantic_cluster_entropy"],
        torch.tensor(0.7),
    )
    torch.testing.assert_close(
        finalized["listwise_semantic_cluster_entropy_norm"],
        torch.tensor(0.7),
    )
    torch.testing.assert_close(
        finalized["listwise_semantic_exploration_gain_any_correct"],
        torch.tensor(2.0),
    )
    torch.testing.assert_close(
        finalized["listwise_semantic_exploration_gain_drgrpo"],
        torch.tensor(3.0),
    )
    torch.testing.assert_close(
        finalized["listwise_projection_ce_loss_effective"],
        torch.tensor(0.0),
    )


def test_finalize_row_sharded_info_stats_applies_reductions_and_defaults():
    finalized = finalize_row_sharded_info_stats(
        {"existing": torch.tensor(9.0)},
        {
            "policy_grad_norm": [torch.tensor(1.0), 3.0],
            "logprobs_diff_max": [torch.tensor(0.5), 1.5],
            "logprobs_diff_min": [torch.tensor(-0.5), -1.5],
            "mean_metric": [torch.tensor(2.0), 4.0],
        },
        device=torch.device("cpu"),
    )

    assert list(finalized) == sorted(finalized)
    torch.testing.assert_close(finalized["existing"], torch.tensor(9.0))
    torch.testing.assert_close(finalized["policy_grad_norm"], torch.tensor(3.0))
    torch.testing.assert_close(finalized["logprobs_diff_max"], torch.tensor(1.5))
    torch.testing.assert_close(finalized["logprobs_diff_min"], torch.tensor(-1.5))
    torch.testing.assert_close(finalized["mean_metric"], torch.tensor(3.0))
    torch.testing.assert_close(finalized["get_grad_norm_time"], torch.tensor(0.0))
    torch.testing.assert_close(
        finalized["listwise_semantic_exploration_gain_any_correct"],
        torch.tensor(0.0),
    )
    torch.testing.assert_close(
        finalized["listwise_semantic_exploration_gain_drgrpo"],
        torch.tensor(0.0),
    )


def test_all_gather_variable_length_returns_flat_tensor_without_distributed():
    values = torch.tensor([[1.0, 2.0]])

    gathered = all_gather_variable_length_1d_tensor(values)

    torch.testing.assert_close(gathered, torch.tensor([1.0, 2.0]))


def test_listwise_optional_logging_defaults_cover_public_drx_metrics():
    expected_keys = {
        "listwise_grad_cosine",
        "listwise_projection_ce_loss_effective",
        "listwise_semantic_cluster_entropy",
        "listwise_semantic_exploration_gain_any_correct",
        "listwise_tau_adaptation_metric_value",
    }

    assert expected_keys <= LISTWISE_OPTIONAL_LOGGING_METRIC_DEFAULTS.keys()
    assert set(LISTWISE_OPTIONAL_LOGGING_METRIC_DEFAULTS.values()) == {0.0}


def test_tensor_correlation_or_zero_matches_centered_correlation():
    lhs = torch.tensor([1.0, 2.0, 3.0])
    rhs = torch.tensor([2.0, 4.0, 6.0])

    corr = tensor_correlation_or_zero(lhs, rhs)

    torch.testing.assert_close(corr, torch.tensor(1.0))
    torch.testing.assert_close(
        tensor_correlation_or_zero(lhs, torch.ones_like(lhs)),
        torch.tensor(0.0),
    )


def test_entropy_split_means_splits_values_by_entropy_rank():
    entropy = torch.tensor([0.4, 0.1, 0.8, 0.2])
    values = torch.tensor([4.0, 1.0, 8.0, 2.0])

    low, high = entropy_split_means(entropy, values)

    torch.testing.assert_close(low, torch.tensor(1.5))
    torch.testing.assert_close(high, torch.tensor(6.0))


def test_summarize_semantic_prompt_diagnostics_returns_expected_metrics():
    summaries, gain_any_mean, gain_drgrpo_mean = summarize_semantic_prompt_diagnostics(
        {
            "listwise_semantic_entropy_prompt": [torch.tensor([0.4, 0.1, 0.8, 0.2])],
            "listwise_semantic_exploration_gain_any_correct_prompt": [
                torch.tensor([4.0, 1.0, 8.0, 2.0])
            ],
            "listwise_semantic_exploration_gain_drgrpo_prompt": [
                torch.tensor([2.0, 1.0, 4.0, 3.0])
            ],
        }
    )

    torch.testing.assert_close(
        summaries["listwise_semantic_exploration_prompt_count"],
        torch.tensor(4.0),
    )
    torch.testing.assert_close(
        summaries["listwise_semantic_exploration_gain_any_correct_low_entropy"],
        torch.tensor(1.5),
    )
    torch.testing.assert_close(
        summaries["listwise_semantic_exploration_gain_any_correct_high_entropy"],
        torch.tensor(6.0),
    )
    torch.testing.assert_close(gain_any_mean, torch.tensor(3.75))
    torch.testing.assert_close(gain_drgrpo_mean, torch.tensor(2.5))


def test_summarize_semantic_prompt_diagnostics_empty_returns_no_overrides():
    summaries, gain_any_mean, gain_drgrpo_mean = summarize_semantic_prompt_diagnostics(
        {
            "listwise_semantic_entropy_prompt": [],
            "listwise_semantic_exploration_gain_any_correct_prompt": [],
            "listwise_semantic_exploration_gain_drgrpo_prompt": [],
        }
    )

    assert summaries == {}
    assert gain_any_mean is None
    assert gain_drgrpo_mean is None
