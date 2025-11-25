"""
Copyright 2025 Liv d'Aliberti

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Integration tests for training loss helpers (requires torch).
"""

from __future__ import annotations

import pytest

try:
    import torch  # type: ignore
except Exception as exc:  # pragma: no cover - environment guard
    pytest.skip(f"torch import failed: {exc}", allow_module_level=True)

if not hasattr(torch, "tensor"):
    pytest.skip("torch tensor operations unavailable", allow_module_level=True)


def _import_training_modules():
    from maxent_grpo.training.weighting.loss import (
        LossInputConfig,
        SequenceScores,
        build_loss_inputs,
        evaluate_losses,
        _ratio_diagnostics,
    )
    from maxent_grpo.training.types import ClipSettings, ReferenceLogprobs
    from maxent_grpo.training.weighting import (
        KlControllerSettings,
        QDistributionSettings,
        TauSchedule,
        WeightNormalizationSettings,
        WeightStats,
        WeightingSettings,
    )

    return (
        LossInputConfig,
        SequenceScores,
        build_loss_inputs,
        evaluate_losses,
        _ratio_diagnostics,
        ClipSettings,
        ReferenceLogprobs,
        KlControllerSettings,
        QDistributionSettings,
        TauSchedule,
        WeightNormalizationSettings,
        WeightStats,
        WeightingSettings,
    )


(
    LossInputConfig,
    SequenceScores,
    build_loss_inputs,
    evaluate_losses,
    _ratio_diagnostics,
    ClipSettings,
    ReferenceLogprobs,
    KlControllerSettings,
    QDistributionSettings,
    TauSchedule,
    WeightNormalizationSettings,
    WeightStats,
    WeightingSettings,
) = _import_training_modules()


def _weighting(beta: float, len_norm_ref: bool = True) -> WeightingSettings:
    return WeightingSettings(
        tau=0.1,
        beta=beta,
        normalization=WeightNormalizationSettings(denom=1.0, len_norm_ref=len_norm_ref),
        q_distribution=QDistributionSettings(temperature=1.0, epsilon=1e-6),
        tau_schedule=TauSchedule(
            target_entropy=None,
            learning_rate=0.01,
            minimum_value=0.0,
            maximum_value=2.0,
            warmup_steps=0,
        ),
        kl_controller=KlControllerSettings(target=0.1, horizon=10, step_size=0.5),
        train_grpo_objective=True,
    )


def _reference_stats() -> ReferenceLogprobs:
    return ReferenceLogprobs(
        ref_logp_sum=torch.tensor([-0.6, -0.2, -0.4]),
        ref_tok_counts=torch.tensor([2.0, 2.0, 1.0]),
        ref_logp_sum_raw=torch.tensor([-1.2, -0.4, -0.4]),
        ref_logp_mean=0.0,
        avg_completion_tokens=1.0,
    )


def test_evaluate_losses_with_clip_objective():
    grouped = [["a", "b"], ["c"]]
    weight_stats = WeightStats(
        weights_grouped=[[0.5, 0.5], [1.0]],
        flat_weights=[0.5, 0.5, 1.0],
        weight_entropy=0.0,
        weight_entropy_min=0.0,
        weight_entropy_max=0.0,
        advantage_entropy=[],
    )
    scores = SequenceScores(
        cur_logp_sum=torch.tensor([-1.0, -0.5, -0.2]),
        behavior_logp_sum=torch.tensor([-0.8, -0.6, -0.3]),
        log_ratio_train=torch.zeros(3),
        denom_tok_tensor=torch.tensor([2.0, 2.0, 1.0]),
    )
    clip_cfg = ClipSettings(
        clip_range=0.1,
        use_clip_objective=True,
        clip_objective_coef=0.5,
        clip_adv_baseline=None,
    )
    config = LossInputConfig(
        clip_cfg=clip_cfg,
        weighting_cfg=_weighting(beta=0.2),
        ref_stats=_reference_stats(),
    )
    group_data, ratio_ctx = build_loss_inputs(grouped, weight_stats, scores, config)
    loss_outputs, diagnostics = evaluate_losses(group_data, ratio_ctx)
    assert loss_outputs.scalars.clip_loss is not None
    assert diagnostics.kl_value is not None


def test_ratio_diagnostics_without_reference():
    clip_cfg = ClipSettings(
        clip_range=0.1,
        use_clip_objective=False,
        clip_objective_coef=0.0,
        clip_adv_baseline=None,
    )
    empty_stats = ReferenceLogprobs(
        ref_logp_sum=torch.tensor([], dtype=torch.float32),
        ref_tok_counts=torch.tensor([], dtype=torch.float32),
        ref_logp_sum_raw=torch.tensor([], dtype=torch.float32),
        ref_logp_mean=0.0,
        avg_completion_tokens=0.0,
    )
    ratio_ctx = type("Ctx", (), {})()
    ratio_ctx.ref_stats = empty_stats
    ratio_ctx.denom_tok_tensor = torch.tensor([], dtype=torch.float32)
    ratio_ctx.cur_logp_sum = torch.tensor([], dtype=torch.float32)
    ratio_ctx.behavior_logp_sum = torch.tensor([], dtype=torch.float32)
    ratio_ctx.log_ratio_train = torch.tensor([], dtype=torch.float32)
    ratio_ctx.clip_cfg = clip_cfg
    ratio_ctx.weighting_cfg = _weighting(beta=0.0)
    stats = _ratio_diagnostics(ratio_ctx)
    assert stats.kl_value is None
    assert stats.clip_ratio == 0.0
