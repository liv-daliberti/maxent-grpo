"""GPU/CPU alignment tests for weighting.loss helpers."""

from __future__ import annotations

import math

import pytest
import torch

from maxent_grpo.training.types import ClipSettings, ReferenceLogprobs
from maxent_grpo.training.weighting.loss import (
    RatioContext,
    WeightingSettings,
    _kl_terms,
    _ratio_stats_with_ref,
)
from maxent_grpo.training.weighting.types import (
    KlControllerSettings,
    QDistributionSettings,
    TauSchedule,
    WeightNormalizationSettings,
)


def _weighting(len_norm_ref: bool = True, beta: float = 0.5) -> WeightingSettings:
    return WeightingSettings(
        tau=0.1,
        beta=beta,
        normalization=WeightNormalizationSettings(denom=1.0, len_norm_ref=len_norm_ref),
        q_distribution=QDistributionSettings(temperature=1.0, epsilon=1e-6),
        tau_schedule=TauSchedule(
            target_entropy=None,
            learning_rate=0.0,
            minimum_value=0.0,
            maximum_value=1.0,
            warmup_steps=0,
        ),
        kl_controller=KlControllerSettings(target=0.0, horizon=0, step_size=0.0),
        train_grpo_objective=False,
    )


_TORCH_HAS_CUDA = bool(getattr(torch, "cuda", None))
_CUDA_AVAILABLE = _TORCH_HAS_CUDA and bool(torch.cuda.is_available())


@pytest.mark.skipif(not _CUDA_AVAILABLE, reason="CUDA device required")
def test_kl_terms_aligns_ref_and_cur_devices():
    """Ensure reference tensors migrate to the policy device before subtraction."""

    cur_device = torch.device("cuda")
    cur_dtype = torch.float16
    cur_logp_sum = torch.tensor([0.1, 0.2], device=cur_device, dtype=cur_dtype)
    denom_tok_tensor = torch.ones_like(cur_logp_sum)

    ratio_ctx = RatioContext(
        log_ratio_train=torch.zeros_like(cur_logp_sum),
        denom_tok_tensor=denom_tok_tensor,
        clip_cfg=ClipSettings(
            clip_range=0.2,
            use_clip_objective=False,
            clip_objective_coef=1.0,
            clip_adv_baseline=None,
        ),
        weighting_cfg=_weighting(),
        ref_stats=ReferenceLogprobs(
            ref_logp_sum=torch.tensor([-0.1, -0.2], device="cpu", dtype=torch.float32),
            ref_tok_counts=torch.tensor([2.0, 3.0], device="cpu"),
            ref_logp_sum_raw=torch.tensor([-0.1, -0.2], device="cpu"),
            ref_logp_mean=0.0,
            avg_completion_tokens=2.0,
        ),
        cur_logp_sum=cur_logp_sum,
        behavior_logp_sum=torch.zeros_like(cur_logp_sum),
    )

    kl_tensor, kl_scalar, weighted = _kl_terms(ratio_ctx)

    assert kl_tensor.device == cur_device
    assert kl_tensor.dtype == cur_dtype
    assert math.isfinite(float(kl_scalar))
    assert math.isfinite(float(weighted))


@pytest.mark.skipif(not _CUDA_AVAILABLE, reason="CUDA device required")
def test_ratio_stats_with_ref_aligns_devices():
    cur_device = torch.device("cuda")
    cur_dtype = torch.float16
    cur_logp_sum = torch.tensor([0.5, 0.25], device=cur_device, dtype=cur_dtype)
    denom_tok_tensor = torch.ones_like(cur_logp_sum)

    ratio_ctx = RatioContext(
        log_ratio_train=torch.zeros_like(cur_logp_sum),
        denom_tok_tensor=denom_tok_tensor,
        clip_cfg=ClipSettings(
            clip_range=0.2,
            use_clip_objective=False,
            clip_objective_coef=1.0,
            clip_adv_baseline=None,
        ),
        weighting_cfg=_weighting(),
        ref_stats=ReferenceLogprobs(
            ref_logp_sum=torch.tensor([0.5, 0.25], device="cpu", dtype=torch.float32),
            ref_tok_counts=torch.tensor([1.0, 1.0], device="cpu"),
            ref_logp_sum_raw=torch.tensor([0.5, 0.25], device="cpu"),
            ref_logp_mean=0.0,
            avg_completion_tokens=1.0,
        ),
        cur_logp_sum=cur_logp_sum,
        behavior_logp_sum=torch.zeros_like(cur_logp_sum),
    )

    stats = _ratio_stats_with_ref(ratio_ctx)
    kl_value = stats[0]
    assert isinstance(kl_value, float)
    assert math.isfinite(kl_value)
