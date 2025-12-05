"""Tests for training.metrics aggregation and expected keys."""

from types import SimpleNamespace

import pytest

from maxent_grpo.training import metrics
from maxent_grpo.training.types.logging import (
    LoggingConfigView,
    RewardComponentStats,
    RewardLoggingView,
    TokenUsageStats,
    TrainingMetricsPayload,
    TrainingScalarStats,
)
from maxent_grpo.training.types.rewards import (
    BatchDiagnostics,
    LengthStats,
    LossOutputs,
    LossScalarBundle,
)
from maxent_grpo.training.weighting.types import (
    WeightLoggingView,
    WeightingSettings,
    WeightNormalizationSettings,
    QDistributionSettings,
    TauSchedule,
    KlControllerSettings,
)
from maxent_grpo.training.types.runtime import ClipSettings, OptimizationSchedule


def _weighting_settings() -> WeightingSettings:
    return WeightingSettings(
        tau=0.2,
        beta=0.5,
        normalization=WeightNormalizationSettings(denom=0.7, len_norm_ref=False),
        q_distribution=QDistributionSettings(temperature=1.0, epsilon=1e-6),
        tau_schedule=TauSchedule(
            target_entropy=None,
            learning_rate=0.0,
            minimum_value=0.0,
            maximum_value=1.0,
            warmup_steps=0,
        ),
        kl_controller=KlControllerSettings(target=0.1, horizon=10, step_size=0.1),
        train_grpo_objective=False,
    )


def _payload() -> TrainingMetricsPayload:
    weight_view = WeightLoggingView(
        entropy=0.6,
        entropy_min=0.6,
        entropy_max=0.6,
        advantage_entropy_mean=0.0,
        advantage_entropy_std=0.0,
    )
    reward_view = RewardLoggingView(
        reward_mean=0.1,
        reward_std=0.2,
        frac_zero_std=0.0,
        advantage_mean=0.0,
        advantage_std=1.0,
        advantage_count=2,
        per_reward={"r0": RewardComponentStats(mean=0.1, std=0.01)},
        q_entropy_mean=0.0,
        q_entropy_std=0.0,
        q_entropy_min=0.0,
        q_entropy_max=0.0,
    )
    loss_scalars = LossScalarBundle(
        total_loss=1.0, policy_loss=0.8, clip_loss=None, kl_loss=0.05, weighted_kl_loss=0.025
    )
    loss_outputs = LossOutputs(
        loss=None,
        scalars=loss_scalars,
        log_ratio_train=None,
        denom_tok_tensor=None,
        seed_loss_scalar=None,
        info_seed_entropy_scalar=None,
    )
    diagnostics = BatchDiagnostics(
        kl_value=0.05,
        clip_ratio=0.1,
        clip_ratio_low_mean=0.0,
        clip_ratio_low_min=0.0,
        clip_ratio_high_mean=0.2,
        clip_ratio_high_max=0.3,
        clip_ratio_region_mean=0.05,
        kl_per_token_by_len_bucket={"0-32": 0.01},
        kl_token_count_by_len_bucket={"0-32": 10.0},
    )
    length_stats = LengthStats(
        min_length=1.0,
        mean_length=2.0,
        max_length=3.0,
        clipped_ratio=0.2,
        min_terminated=1.0,
        mean_terminated=2.0,
        max_terminated=3.0,
    )
    weighting = _weighting_settings()
    config_view = LoggingConfigView(
        weighting=weighting,
        clipping=ClipSettings(
            clip_range=0.2,
            use_clip_objective=True,
            clip_objective_coef=1.0,
            clip_adv_baseline=None,
        ),
        schedule=OptimizationSchedule(
            num_epochs=1,
            num_generations=1,
            grad_accum_steps=1,
            max_grad_norm=1.0,
            steps_per_epoch=1,
            total_training_steps=1,
            warmup_steps=0,
        ),
    )
    scalars = TrainingScalarStats(
        ref_logp_mean=0.0,
        tokens=TokenUsageStats(
            avg_completion_tokens=1.0,
            num_completion_tokens=1.0,
            num_input_tokens=1.0,
        ),
        current_lr=1e-4,
        grad_norm_scalar=None,
        epoch_progress=0.1,
        vllm_latency_ms=None,
    )
    return TrainingMetricsPayload(
        reward_stats=reward_view,
        weight_stats=weight_view,
        loss_outputs=loss_outputs,
        diagnostics=diagnostics,
        length_stats=length_stats,
        config=config_view,
        scalars=scalars,
        seed_metrics=None,
    )


def test_build_training_metrics_dict_contains_weight_and_clip_keys():
    payload = _payload()
    metrics_dict = metrics.build_training_metrics_dict(payload, global_step=1)
    assert metrics_dict["train/weight_entropy"] == pytest.approx(0.6)
    assert metrics_dict["train/weight_norm_denom"] == pytest.approx(0.7)
    assert "train/clip_ratio/high_mean" in metrics_dict
    assert "train/clip_ratio/low_min" in metrics_dict
    assert metrics_dict["train/kl"] == pytest.approx(0.05)


def test_build_training_metrics_dict_sets_grpo_vs_maxent_flags():
    payload = _payload()
    payload.config.weighting.train_grpo_objective = False
    metrics_dict = metrics.build_training_metrics_dict(payload, global_step=1)
    assert metrics_dict["train/grpo_objective"] == 0.0
    assert metrics_dict["train/maxent_objective"] == 1.0
    payload.config.weighting.train_grpo_objective = True
    metrics_dict = metrics.build_training_metrics_dict(payload, global_step=1)
    assert metrics_dict["train/grpo_objective"] == 1.0
    assert metrics_dict["train/maxent_objective"] == 0.0
