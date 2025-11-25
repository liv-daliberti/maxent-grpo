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

Unit tests for training.weighting.loss helpers.
"""

from __future__ import annotations

import math
import sys
import numpy as np

import pytest
import torch

from maxent_grpo.training.weighting import loss as loss_mod
from maxent_grpo.training.weighting.loss import (
    GroupLossData,
    LossInputConfig,
    RatioContext,
    SequenceScores,
    WeightingSettings,
    build_loss_inputs,
    evaluate_losses,
    _apply_clip_objective,
    _build_loss_outputs,
    _clip_bounds,
    _clip_region_metrics,
    _clip_loss_for_slice,
    _compute_clip_ratios,
    _kl_terms,
    _policy_loss_from_groups,
    _ratio_diagnostics,
    _ratio_stats_without_ref,
    _tensor_mean_std,
)
from maxent_grpo.training.weighting.types import (
    KlControllerSettings,
    QDistributionSettings,
    TauSchedule,
    WeightNormalizationSettings,
    WeightStats,
)
from maxent_grpo.training.types import ClipSettings, ReferenceLogprobs


@pytest.fixture(autouse=True)
def _patch_torch_stub(monkeypatch):
    """Patch the lightweight torch stub with missing helpers used by loss code."""

    from tests.helpers.run_setup_stubs import TORCH_STUB

    def _view(self, *shape):
        return TORCH_STUB.tensor(self.arr.reshape(*shape))

    def _exp(val):
        arr = val.arr if hasattr(val, "arr") else val
        return TORCH_STUB.tensor(np.exp(arr))

    def _where(cond, x, y):
        cond_arr = cond.arr if hasattr(cond, "arr") else cond
        x_arr = x.arr if hasattr(x, "arr") else x
        y_arr = y.arr if hasattr(y, "arr") else y
        return TORCH_STUB.tensor(np.where(cond_arr, x_arr, y_arr))

    def _std(self, unbiased=True):
        ddof = 1 if unbiased else 0
        return TORCH_STUB.tensor(np.std(self.arr, ddof=ddof))

    def _minimum(a, b):
        a_arr = a.arr if hasattr(a, "arr") else a
        b_arr = b.arr if hasattr(b, "arr") else b
        return TORCH_STUB.tensor(np.minimum(a_arr, b_arr))

    def _maximum(a, b):
        a_arr = a.arr if hasattr(a, "arr") else a
        b_arr = b.arr if hasattr(b, "arr") else b
        return TORCH_STUB.tensor(np.maximum(a_arr, b_arr))

    def _clamp(tensor, min=None, max=None):
        return tensor.clamp(min=min, max=max)

    def _stack(tensors, dim=0):
        arrays = [t.arr if hasattr(t, "arr") else np.array(t) for t in tensors]
        return TORCH_STUB.tensor(np.stack(arrays, axis=dim))

    monkeypatch.setattr(TORCH_STUB.Tensor, "view", _view, raising=False)
    monkeypatch.setattr(TORCH_STUB.Tensor, "std", _std, raising=False)
    monkeypatch.setattr(
        TORCH_STUB.Tensor, "__float__", lambda self: float(self.item()), raising=False
    )
    monkeypatch.setattr(
        TORCH_STUB.Tensor,
        "__rmul__",
        lambda self, other: self.__mul__(other),
        raising=False,
    )
    monkeypatch.setattr(
        TORCH_STUB.Tensor,
        "__radd__",
        lambda self, other: self.__add__(other),
        raising=False,
    )
    monkeypatch.setattr(
        TORCH_STUB.Tensor,
        "__le__",
        lambda self, other: TORCH_STUB.tensor(
            self.arr <= (other.arr if hasattr(other, "arr") else other)
        ),
        raising=False,
    )
    monkeypatch.setattr(
        TORCH_STUB.Tensor, "ndim", property(lambda self: len(self.shape)), raising=False
    )
    monkeypatch.setattr(TORCH_STUB, "exp", _exp, raising=False)
    monkeypatch.setattr(
        TORCH_STUB.Tensor, "exp", lambda self: _exp(self), raising=False
    )
    monkeypatch.setattr(TORCH_STUB, "where", _where, raising=False)
    monkeypatch.setattr(TORCH_STUB, "minimum", _minimum, raising=False)
    monkeypatch.setattr(TORCH_STUB, "maximum", _maximum, raising=False)
    monkeypatch.setattr(TORCH_STUB, "clamp", _clamp, raising=False)
    monkeypatch.setattr(TORCH_STUB, "stack", _stack, raising=False)
    sys.modules["torch"] = TORCH_STUB
    monkeypatch.setattr(loss_mod, "torch", TORCH_STUB)
    monkeypatch.setattr(loss_mod, "Tensor", TORCH_STUB.Tensor)
    globals()["torch"] = TORCH_STUB
    yield


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


def test_build_loss_inputs_mismatch_raises():
    grouped = [["a", "b"]]
    weight_stats = WeightStats(
        weights_grouped=[[0.5, 0.5]],
        flat_weights=[0.5],  # mismatch on purpose
        weight_entropy=0.0,
        weight_entropy_min=0.0,
        weight_entropy_max=0.0,
        advantage_entropy=[],
    )
    scores = SequenceScores(
        cur_logp_sum=torch.tensor([0.0, 0.0]),
        behavior_logp_sum=torch.tensor([0.0, 0.0]),
        log_ratio_train=torch.tensor([0.0, 0.0]),
        denom_tok_tensor=torch.tensor([1.0, 1.0]),
    )
    cfg = LossInputConfig(
        clip_cfg=ClipSettings(
            clip_range=0.2,
            use_clip_objective=False,
            clip_objective_coef=1.0,
            clip_adv_baseline=None,
        ),
        weighting_cfg=_weighting(),
        ref_stats=ReferenceLogprobs(
            ref_logp_sum=torch.tensor([0.0, 0.0]),
            ref_tok_counts=torch.tensor([1.0, 1.0]),
            ref_logp_sum_raw=torch.tensor([0.0, 0.0]),
            ref_logp_mean=0.0,
            avg_completion_tokens=1.0,
        ),
    )
    with pytest.raises(ValueError):
        build_loss_inputs(grouped, weight_stats, scores, cfg)


def test_policy_loss_requires_completions():
    group_data = GroupLossData(
        group_sizes=[0],
        weight_tensor=torch.tensor([], dtype=torch.float32),
        logp_sums=torch.tensor([], dtype=torch.float32),
        token_counts=torch.tensor([], dtype=torch.float32),
    )
    with pytest.raises(ValueError):
        _policy_loss_from_groups(group_data)


def test_clip_loss_for_slice_computes_objective():
    weights = torch.tensor([0.6, 0.4])
    ratio = torch.tensor([1.2, 0.8])
    clipped = torch.tensor([1.1, 0.9])
    loss = _clip_loss_for_slice(weights, ratio, clipped, adv_base_val=0.5)
    assert loss.item() != 0.0


def test_apply_clip_objective_disabled_returns_none():
    ratio_ctx = RatioContext(
        log_ratio_train=torch.zeros(2),
        denom_tok_tensor=torch.ones(2),
        clip_cfg=ClipSettings(
            clip_range=0.0,
            use_clip_objective=False,
            clip_objective_coef=1.0,
            clip_adv_baseline=None,
        ),
        weighting_cfg=_weighting(),
        ref_stats=ReferenceLogprobs(
            ref_logp_sum=torch.zeros(2),
            ref_tok_counts=torch.ones(2),
            ref_logp_sum_raw=torch.zeros(2),
            ref_logp_mean=0.0,
            avg_completion_tokens=1.0,
        ),
        cur_logp_sum=torch.zeros(2),
        behavior_logp_sum=torch.zeros(2),
    )
    group_data = GroupLossData(
        group_sizes=[2],
        weight_tensor=torch.tensor([0.5, 0.5]),
        logp_sums=torch.zeros(2),
        token_counts=torch.ones(2),
    )
    policy = torch.tensor(1.0)
    updated, clip_scalar = _apply_clip_objective(ratio_ctx, group_data, policy)
    assert updated is policy
    assert clip_scalar is None


def test_apply_clip_objective_updates_loss():
    clip_cfg = ClipSettings(
        clip_range=0.1,
        use_clip_objective=True,
        clip_objective_coef=2.0,
        clip_adv_baseline=None,
    )
    ratio_ctx = RatioContext(
        log_ratio_train=torch.zeros(2),
        denom_tok_tensor=torch.ones(2),
        clip_cfg=clip_cfg,
        weighting_cfg=_weighting(),
        ref_stats=ReferenceLogprobs(
            ref_logp_sum=torch.zeros(2),
            ref_tok_counts=torch.ones(2),
            ref_logp_sum_raw=torch.zeros(2),
            ref_logp_mean=0.0,
            avg_completion_tokens=1.0,
        ),
        cur_logp_sum=torch.tensor([0.1, -0.1]),
        behavior_logp_sum=torch.zeros(2),
    )
    group_data = GroupLossData(
        group_sizes=[2],
        weight_tensor=torch.tensor([0.6, 0.4]),
        logp_sums=torch.tensor([0.1, -0.1]),
        token_counts=torch.ones(2),
    )
    policy = torch.tensor(0.5)
    updated, clip_scalar = _apply_clip_objective(ratio_ctx, group_data, policy)
    assert updated.item() != policy.item()
    assert clip_scalar is not None


def test_apply_clip_objective_skips_empty_groups():
    clip_cfg = ClipSettings(
        clip_range=0.1,
        use_clip_objective=True,
        clip_objective_coef=1.0,
        clip_adv_baseline=None,
    )
    ratio_ctx = RatioContext(
        log_ratio_train=torch.zeros(0),
        denom_tok_tensor=torch.zeros(0),
        clip_cfg=clip_cfg,
        weighting_cfg=_weighting(),
        ref_stats=ReferenceLogprobs(
            ref_logp_sum=torch.zeros(0),
            ref_tok_counts=torch.zeros(0),
            ref_logp_sum_raw=torch.zeros(0),
            ref_logp_mean=0.0,
            avg_completion_tokens=0.0,
        ),
        cur_logp_sum=torch.zeros(0),
        behavior_logp_sum=torch.zeros(0),
    )
    group_data = GroupLossData(
        group_sizes=[0],
        weight_tensor=torch.tensor([]),
        logp_sums=torch.tensor([]),
        token_counts=torch.tensor([]),
    )
    policy = torch.tensor(0.7)
    updated, clip_scalar = _apply_clip_objective(ratio_ctx, group_data, policy)
    assert updated is policy
    assert clip_scalar is None


def test_kl_terms_respects_len_norm_ref_toggle():
    weighting = _weighting(len_norm_ref=False, beta=0.3)
    ratio_ctx = RatioContext(
        log_ratio_train=torch.zeros(2),
        denom_tok_tensor=torch.tensor([2.0, 2.0]),
        clip_cfg=ClipSettings(
            clip_range=0.2,
            use_clip_objective=False,
            clip_objective_coef=1.0,
            clip_adv_baseline=None,
        ),
        weighting_cfg=weighting,
        ref_stats=ReferenceLogprobs(
            ref_logp_sum=torch.tensor([1.0, 1.0]),
            ref_tok_counts=torch.tensor([2.0, 2.0]),
            ref_logp_sum_raw=torch.tensor([2.0, 2.0]),
            ref_logp_mean=1.0,
            avg_completion_tokens=2.0,
        ),
        cur_logp_sum=torch.tensor([1.0, 1.0]),
        behavior_logp_sum=torch.zeros(2),
    )
    kl_tensor, kl_scalar, weighted = _kl_terms(ratio_ctx)
    assert kl_tensor.item() >= 0.0
    assert math.isclose(weighted, weighting.beta * kl_scalar)


def test_compute_clip_ratios_clamps_values():
    clip_cfg = ClipSettings(
        clip_range=0.5,
        use_clip_objective=True,
        clip_objective_coef=1.0,
        clip_adv_baseline=None,
    )
    ratio_ctx = RatioContext(
        log_ratio_train=torch.zeros(2),
        denom_tok_tensor=torch.ones(2),
        clip_cfg=clip_cfg,
        weighting_cfg=_weighting(),
        ref_stats=ReferenceLogprobs(
            ref_logp_sum=torch.zeros(2),
            ref_tok_counts=torch.ones(2),
            ref_logp_sum_raw=torch.zeros(2),
            ref_logp_mean=0.0,
            avg_completion_tokens=1.0,
        ),
        cur_logp_sum=torch.tensor([2.0, -2.0]),
        behavior_logp_sum=torch.zeros(2),
    )
    ratio, clipped = _compute_clip_ratios(ratio_ctx, clip_cfg)
    assert torch.all(clipped <= 1.5)
    assert ratio.shape == clipped.shape


def test_build_loss_outputs_packs_scalars():
    clip_cfg = ClipSettings(
        clip_range=0.2,
        use_clip_objective=False,
        clip_objective_coef=1.0,
        clip_adv_baseline=None,
    )
    ratio_ctx = RatioContext(
        log_ratio_train=torch.ones(1),
        denom_tok_tensor=torch.ones(1),
        clip_cfg=clip_cfg,
        weighting_cfg=_weighting(),
        ref_stats=ReferenceLogprobs(
            ref_logp_sum=torch.ones(1),
            ref_tok_counts=torch.ones(1),
            ref_logp_sum_raw=torch.ones(1),
            ref_logp_mean=1.0,
            avg_completion_tokens=1.0,
        ),
        cur_logp_sum=torch.ones(1),
        behavior_logp_sum=torch.zeros(1),
    )
    scalar_inputs = type(
        "_S",
        (),
        {
            "clip_loss_scalar": None,
            "kl_loss_scalar": 0.1,
            "weighted_kl_loss_scalar": 0.2,
        },
    )()
    outputs = _build_loss_outputs(
        ratio_ctx, torch.tensor(0.5), torch.tensor(0.1), scalar_inputs
    )
    assert outputs.scalars.total_loss > 0
    assert outputs.scalars.policy_loss > 0


def test_clip_bounds_and_region_metrics():
    clip_cfg = ClipSettings(
        clip_range=0.2,
        use_clip_objective=True,
        clip_objective_coef=1.0,
        clip_adv_baseline=None,
    )
    low, high = _clip_bounds(clip_cfg)
    assert low < 0 < high
    log_ratio = torch.tensor([-1.0, 0.0, 1.0])
    stats = _clip_region_metrics(log_ratio, clip_cfg)
    assert len(stats) == 6
    assert stats[0] >= 0.0


def test_tensor_mean_std_handles_empty_and_singleton():
    assert _tensor_mean_std(torch.tensor([])) == (0.0, 0.0)
    mean, std = _tensor_mean_std(torch.tensor([2.0]))
    assert mean == 2.0 and std == 0.0


def test_tensor_mean_std_handles_multi_value_tensor():
    mean, std = _tensor_mean_std(torch.tensor([1.0, 3.0]))
    assert mean == 2.0
    assert math.isclose(std, 1.0)


def test_ratio_diagnostics_without_ref_counts_zero():
    ratio_ctx = RatioContext(
        log_ratio_train=torch.zeros(0),
        denom_tok_tensor=torch.zeros(0),
        clip_cfg=ClipSettings(
            clip_range=0.1,
            use_clip_objective=False,
            clip_objective_coef=1.0,
            clip_adv_baseline=None,
        ),
        weighting_cfg=_weighting(beta=0.0),
        ref_stats=ReferenceLogprobs(
            ref_logp_sum=torch.tensor([]),
            ref_tok_counts=torch.tensor([]),
            ref_logp_sum_raw=torch.tensor([]),
            ref_logp_mean=0.0,
            avg_completion_tokens=0.0,
        ),
        cur_logp_sum=torch.tensor([]),
        behavior_logp_sum=torch.tensor([]),
    )
    diag = _ratio_diagnostics(ratio_ctx)
    assert diag.clip_ratio == 0.0
    assert diag.kl_value is None
    assert _ratio_stats_without_ref()[0] == 0.0


def test_ratio_stats_with_ref_uses_raw_when_len_norm_disabled():
    weighting = _weighting(len_norm_ref=False)
    ratio_ctx = RatioContext(
        log_ratio_train=torch.zeros(1),
        denom_tok_tensor=torch.tensor([2.0]),
        clip_cfg=ClipSettings(
            clip_range=0.1,
            use_clip_objective=False,
            clip_objective_coef=1.0,
            clip_adv_baseline=None,
        ),
        weighting_cfg=weighting,
        ref_stats=ReferenceLogprobs(
            ref_logp_sum=torch.tensor([0.0]),
            ref_tok_counts=torch.tensor([2.0]),
            ref_logp_sum_raw=torch.tensor([2.0]),
            ref_logp_mean=0.0,
            avg_completion_tokens=2.0,
        ),
        cur_logp_sum=torch.tensor([0.0]),
        behavior_logp_sum=torch.tensor([0.0]),
    )
    kl_value, *_ = loss_mod._ratio_stats_with_ref(ratio_ctx)
    expected = (
        math.exp(1.0) - 1.0 - 1.0
    )  # delta=1.0 when using raw ref_logp_sum_raw/denom
    assert math.isclose(kl_value, expected, rel_tol=1e-5)


def test_evaluate_losses_end_to_end():
    weights = WeightStats(
        weights_grouped=[[0.7, 0.3]],
        flat_weights=[0.7, 0.3],
        weight_entropy=0.0,
        weight_entropy_min=0.0,
        weight_entropy_max=0.0,
        advantage_entropy=[],
    )
    scores = SequenceScores(
        cur_logp_sum=torch.tensor([0.2, 0.1]),
        behavior_logp_sum=torch.tensor([0.1, 0.1]),
        log_ratio_train=torch.tensor([0.1, 0.0]),
        denom_tok_tensor=torch.tensor([2.0, 2.0]),
    )
    cfg = LossInputConfig(
        clip_cfg=ClipSettings(
            clip_range=0.0,
            use_clip_objective=False,
            clip_objective_coef=1.0,
            clip_adv_baseline=None,
        ),
        weighting_cfg=_weighting(),
        ref_stats=ReferenceLogprobs(
            ref_logp_sum=torch.tensor([0.1, 0.1]),
            ref_tok_counts=torch.tensor([2.0, 2.0]),
            ref_logp_sum_raw=torch.tensor([0.1, 0.1]),
            ref_logp_mean=0.1,
            avg_completion_tokens=2.0,
        ),
    )
    group_data, ratio_ctx = build_loss_inputs([["a", "b"]], weights, scores, cfg)
    outputs, diag = evaluate_losses(group_data, ratio_ctx)
    assert outputs.loss.ndim == 0
    assert diag.clip_ratio >= 0.0
