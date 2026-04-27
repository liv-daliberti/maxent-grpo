from oat_drgrpo.logging_utils import (
    WANDB_DEBUG_METRIC_ENV,
    add_public_drx_training_metrics,
    filter_wandb_logs_for_public_comparison,
)


def test_add_public_drx_training_metrics_summarizes_performance_and_signal():
    logs = {
        "train/learn_batch_time": 80.0,
        "train/total_time": 100.0,
        "actor/total_time": 20.0,
        "actor/num_data": 160.0,
        "train/listwise_zero_signal_skip": 0.75,
        "train/listwise_active_group_frac": 0.125,
        "train/listwise_active_group_count_global": 2.0,
        "train/listwise_drgrpo_token_active_row_frac": 0.5,
        "train/listwise_drgrpo_token_active_row_count_global": 8.0,
        "train/listwise_neutral_group_frac": 0.875,
        "train/listwise_raw_neutral_group_frac": 0.625,
        "train/listwise_all_wrong_group_frac": 0.3,
        "train/listwise_mixed_correctness_group_frac": 0.4,
        "train/listwise_all_correct_group_frac": 0.2,
        "train/listwise_any_correct_group_frac": 0.6,
        "train/listwise_semantic_answer_key_extracted_frac": 1.0,
        "train/listwise_semantic_trace_extracted_frac": 0.75,
        "train/listwise_semantic_signature_extracted_frac": 0.5,
        "train/listwise_semantic_cluster_valid_frac": 0.25,
        "train/listwise_semantic_cluster_count": 3.0,
        "train/listwise_semantic_cluster_entropy": 0.4,
        "train/listwise_semantic_exploration_gain_drgrpo": 0.06,
        "train/listwise_exact_semantic_gate_mean": 0.25,
        "train/listwise_adv_abs_mean": 0.03,
        "train/listwise_exact_semantic_piece_mean": 0.004,
        "train/listwise_exact_correctness_adv_abs_mean": 0.11,
        "train/listwise_exact_semantic_adv_abs_mean": 0.01,
        "train/listwise_exact_semantic_to_correctness_adv_ratio": 0.09,
        "train/listwise_exact_semantic_adv_fraction": 0.083,
        "train/drgrpo_adv_abs_mean": 0.12,
        "train/listwise_post_scale_ratio": 0.25,
        "train/listwise_helpfulness_proxy": 0.5,
    }

    enriched = add_public_drx_training_metrics(logs)

    assert enriched["drx/perf/learn_seconds"] == 80.0
    assert enriched["drx/perf/train_samples_per_second"] == 2.0
    assert enriched["drx/perf/learn_to_actor_time_ratio"] == 4.0
    assert enriched["drx/perf/learn_time_frac"] == 0.8
    assert enriched["drx/signal/zero_signal_skip_rate"] == 0.75
    assert enriched["drx/signal/useful_update_rate"] == 0.25
    assert enriched["drx/signal/token_active_row_count"] == 8.0
    assert enriched["drx/signal/all_wrong_group_frac"] == 0.3
    assert enriched["drx/signal/mixed_correctness_group_frac"] == 0.4
    assert enriched["drx/signal/all_correct_group_frac"] == 0.2
    assert enriched["drx/signal/any_correct_group_frac"] == 0.6
    assert enriched["drx/semantic/extraction_health"] == 0.625
    assert enriched["drx/semantic/exploration_gain"] == 0.06
    assert enriched["drx/semantic/correctness_gate"] == 0.25
    assert enriched["drx/objective/semantic_piece_mean"] == 0.004
    assert enriched["drx/objective/correctness_adv_abs_mean"] == 0.11
    assert enriched["drx/objective/semantic_adv_abs_mean"] == 0.01
    assert enriched["drx/objective/semantic_to_correctness_adv_ratio"] == 0.09
    assert enriched["drx/objective/semantic_adv_fraction"] == 0.083
    assert enriched["drx/objective/post_scale_ratio"] == 0.25


def test_filter_keeps_public_drx_metrics_while_hiding_raw_debug(monkeypatch):
    monkeypatch.delenv(WANDB_DEBUG_METRIC_ENV, raising=False)
    logs = {
        "drx/perf/learn_seconds": 80.0,
        "drx/signal/useful_update_rate": 0.25,
        "train/listwise_semantic_answer_key_extracted_frac": 1.0,
        "train/listwise_semantic_competitive_mode_count": 2.0,
    }

    filtered = filter_wandb_logs_for_public_comparison(logs)

    assert filtered == {
        "drx/perf/learn_seconds": 80.0,
        "drx/signal/useful_update_rate": 0.25,
    }


def test_filter_debug_env_keeps_all_metrics(monkeypatch):
    monkeypatch.setenv(WANDB_DEBUG_METRIC_ENV, "1")

    logs = {"train/listwise_semantic_answer_key_extracted_frac": 1.0}

    assert filter_wandb_logs_for_public_comparison(logs) == logs
