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

Tests for the MaxEnt weighting helpers.
"""

from __future__ import annotations

from importlib import import_module, reload
import math
from types import SimpleNamespace

import pytest

from maxent_grpo.config import GRPOConfig


@pytest.fixture
def weighting_mod():
    """Load training.weighting module."""

    module = reload(import_module("maxent_grpo.training.weighting"))
    logic = reload(import_module("maxent_grpo.training.weighting.logic"))
    return SimpleNamespace(module=module, logic=logic, types=module)


def _build_weighting(
    weighting_mod,
    *,
    beta,
    tau=0.0,
    kl_target,
    kl_horizon,
    kl_ctl_step_size,
    train_grpo_objective=False,
):
    types = weighting_mod.types
    normalization = types.WeightNormalizationSettings(
        denom=(1.0 if train_grpo_objective else (tau if tau > 0 else 1.0)),
        len_norm_ref=True,
    )
    return types.WeightingSettings(
        tau=tau,
        beta=beta,
        normalization=normalization,
        q_distribution=types.QDistributionSettings(temperature=1.0, epsilon=1e-6),
        tau_schedule=types.TauSchedule(
            target_entropy=None,
            learning_rate=0.0,
            minimum_value=0.0,
            maximum_value=0.0,
            warmup_steps=0,
        ),
        kl_controller=types.KlControllerSettings(
            target=kl_target,
            horizon=kl_horizon,
            step_size=kl_ctl_step_size,
        ),
        train_grpo_objective=train_grpo_objective,
    )


def test_maybe_update_beta_uses_controller_settings(weighting_mod):
    weighting = _build_weighting(
        weighting_mod,
        beta=3.0,
        kl_target=0.07,
        kl_horizon=50000,
        kl_ctl_step_size=0.15,
    )
    weighting_mod.module.maybe_update_beta(weighting, measured_kl=0.14)
    expected_scale = 1.0 + (0.15 / 50000.0)
    assert weighting.beta == pytest.approx(3.0 * expected_scale)
    assert weighting.denom == pytest.approx(1.0)


def test_grpo_objective_keeps_unit_denom(weighting_mod):
    weighting = _build_weighting(
        weighting_mod,
        beta=3.0,
        tau=0.0,
        kl_target=0.0,
        kl_horizon=0,
        kl_ctl_step_size=0.0,
        train_grpo_objective=True,
    )
    weighting_mod.module.maybe_update_beta(weighting, measured_kl=0.14)
    weighting.tau_target_entropy = 0.5
    weighting.tau_lr = 0.1
    weighting.tau_warmup_steps = 0
    weighting_mod.module.maybe_update_tau(
        weighting,
        weight_stats=SimpleNamespace(weight_entropy=0.4),
        global_step=10,
    )
    assert weighting.denom == pytest.approx(1.0)


def test_grpo_beta_controller_keeps_unit_denom(weighting_mod):
    weighting = _build_weighting(
        weighting_mod,
        beta=3.0,
        tau=0.4,
        kl_target=0.07,
        kl_horizon=50000,
        kl_ctl_step_size=0.15,
        train_grpo_objective=True,
    )

    weighting_mod.module.maybe_update_beta(weighting, measured_kl=0.14)

    assert weighting.beta > 3.0
    assert weighting.denom == pytest.approx(1.0)


def test_build_weighting_settings_uses_unit_denom_for_grpo(weighting_mod):
    cfg = GRPOConfig(
        objective="grpo",
        maxent_tau=0.8,
        beta=0.6,
    )

    weighting = weighting_mod.logic.build_weighting_settings(cfg)

    assert weighting.train_grpo_objective is True
    assert weighting.denom == pytest.approx(1.0)


def test_build_weighting_settings_uses_tau_denom_for_listwise(weighting_mod):
    cfg = GRPOConfig(
        objective="maxent_listwise",
        maxent_tau=0.8,
        beta=0.6,
    )

    weighting = weighting_mod.logic.build_weighting_settings(cfg)

    assert weighting.train_grpo_objective is False
    assert weighting.denom == pytest.approx(0.8)


def test_build_weighting_settings_requires_positive_tau_for_listwise(weighting_mod):
    cfg = SimpleNamespace(
        train_grpo_objective=False,
        maxent_objective_variant="listwise",
        maxent_tau=0.0,
        beta=0.6,
        init_kl_coeff=None,
        maxent_length_normalize_ref=True,
        maxent_q_temperature=1.0,
        maxent_q_epsilon=1e-6,
        maxent_target_weight_entropy=None,
        maxent_target_weight_entropy_start=None,
        maxent_target_weight_entropy_final=None,
        maxent_target_weight_entropy_horizon=0,
        maxent_tau_lr=0.0,
        maxent_tau_min=0.0,
        maxent_tau_max=0.0,
        maxent_tau_warmup_steps=-1,
        kl_target=0.0,
        kl_horizon=0,
        kl_ctl_step_size=0.0,
        scale_rewards=True,
        controller_meta_enabled=False,
        controller_meta_method="analytic",
        controller_meta_lr=0.0,
        controller_meta_tau_lr=0.0,
        controller_meta_beta_lr=0.0,
        controller_meta_beta_grad_clip=0.0,
        controller_meta_update_interval=1,
        controller_meta_objective="potential",
        controller_meta_analytic_steps=1,
        controller_meta_optimizer="sgd",
        controller_meta_truncation_steps=1,
        controller_meta_use_hessian=False,
        maxent_allow_empty_weight_fallback=False,
    )

    with pytest.raises(ValueError, match="maxent_tau > 0"):
        weighting_mod.logic.build_weighting_settings(cfg)


def test_maybe_update_tau_treats_non_positive_tau_max_as_unbounded(weighting_mod):
    weighting = _build_weighting(
        weighting_mod,
        beta=0.2,
        tau=0.5,
        kl_target=0.0,
        kl_horizon=0,
        kl_ctl_step_size=0.0,
        train_grpo_objective=False,
    )
    weighting.tau_target_entropy = 0.5
    weighting.tau_lr = 0.5
    weighting.tau_min = 0.0
    weighting.tau_max = 0.0
    weighting.tau_warmup_steps = 0

    weighting_mod.module.maybe_update_tau(
        weighting,
        weight_stats=SimpleNamespace(weight_entropy=0.3),
        global_step=10,
    )

    assert weighting.tau > 0.5
    assert weighting.denom == pytest.approx(weighting.tau)


def test_apply_meta_controller_update_accepts_first_order_alias(weighting_mod):
    weighting = _build_weighting(
        weighting_mod,
        beta=0.2,
        tau=0.5,
        kl_target=0.0,
        kl_horizon=0,
        kl_ctl_step_size=0.0,
        train_grpo_objective=False,
    )
    weighting.controller_meta.enabled = True
    weighting.controller_meta.method = "first_order"
    weighting.controller_meta.learning_rate = 0.1

    updated = weighting_mod.module.apply_meta_controller_update(
        weighting,
        tau_grad=1.0,
        beta_grad=2.0,
    )

    assert updated is True
    assert weighting.tau == pytest.approx(0.4)
    assert weighting.beta == pytest.approx(0.4)


def test_weight_vector_from_q_uses_tau_for_fixed_q_targets(weighting_mod):
    weighting = _build_weighting(
        weighting_mod,
        beta=0.4,
        tau=0.5,
        kl_target=0.0,
        kl_horizon=0,
        kl_ctl_step_size=0.0,
        train_grpo_objective=False,
    )

    probs = weighting_mod.module.weight_vector_from_q(
        [0.8, 0.2],
        [0.0, 1.0],
        [1.0, 1.0],
        weighting,
        include_reference_term=True,
        normalize_by_tokens=False,
    )

    expected_log_terms = [
        math.log(0.8) / 0.5,
        (math.log(0.2) + 0.4) / 0.5,
    ]
    max_term = max(expected_log_terms)
    expected = [math.exp(val - max_term) for val in expected_log_terms]
    total = sum(expected)
    expected = [val / total for val in expected]

    assert probs == pytest.approx(expected)


def test_weight_matrix_from_q_matches_rowwise_helper(weighting_mod):
    weighting = _build_weighting(
        weighting_mod,
        beta=0.3,
        tau=0.7,
        kl_target=0.0,
        kl_horizon=0,
        kl_ctl_step_size=0.0,
        train_grpo_objective=False,
    )
    q_values = weighting_mod.logic.torch.tensor(
        [[0.8, 0.2], [0.35, 0.65]],
        dtype=weighting_mod.logic.torch.float32,
    )
    logp_values = weighting_mod.logic.torch.tensor(
        [[0.0, 1.0], [0.5, -0.25]],
        dtype=weighting_mod.logic.torch.float32,
    )
    token_counts = weighting_mod.logic.torch.tensor(
        [[3.0, 5.0], [2.0, 4.0]],
        dtype=weighting_mod.logic.torch.float32,
    )

    matrix = weighting_mod.module.weight_matrix_from_q(
        q_values,
        logp_values,
        token_counts,
        weighting,
        include_reference_term=True,
        normalize_by_tokens=False,
    )
    rowwise = [
        weighting_mod.module.weight_vector_from_q(
            q_row.tolist(),
            logp_row.tolist(),
            count_row.tolist(),
            weighting,
            include_reference_term=True,
            normalize_by_tokens=False,
        )
        for q_row, logp_row, count_row in zip(q_values, logp_values, token_counts)
    ]

    assert weighting_mod.logic.torch.allclose(
        matrix,
        weighting_mod.logic.torch.tensor(
            rowwise,
            dtype=matrix.dtype,
            device=matrix.device,
        ),
        atol=1e-6,
        rtol=1e-6,
    )


def test_weight_vector_from_q_does_not_reapply_q_temperature(weighting_mod):
    cold = _build_weighting(
        weighting_mod,
        beta=0.0,
        tau=1.0,
        kl_target=0.0,
        kl_horizon=0,
        kl_ctl_step_size=0.0,
        train_grpo_objective=False,
    )
    hot = _build_weighting(
        weighting_mod,
        beta=0.0,
        tau=1.0,
        kl_target=0.0,
        kl_horizon=0,
        kl_ctl_step_size=0.0,
        train_grpo_objective=False,
    )
    cold.q_temperature = 0.25
    hot.q_temperature = 4.0

    cold_probs = weighting_mod.module.weight_vector_from_q(
        [0.8, 0.2],
        [0.0, 0.0],
        [1.0, 1.0],
        cold,
        include_reference_term=False,
        normalize_by_tokens=False,
    )
    hot_probs = weighting_mod.module.weight_vector_from_q(
        [0.8, 0.2],
        [0.0, 0.0],
        [1.0, 1.0],
        hot,
        include_reference_term=False,
        normalize_by_tokens=False,
    )

    assert cold_probs == pytest.approx(hot_probs)


def test_weight_vector_from_q_entropy_increases_with_tau(weighting_mod):
    sharp = _build_weighting(
        weighting_mod,
        beta=0.0,
        tau=0.25,
        kl_target=0.0,
        kl_horizon=0,
        kl_ctl_step_size=0.0,
        train_grpo_objective=False,
    )
    smooth = _build_weighting(
        weighting_mod,
        beta=0.0,
        tau=2.0,
        kl_target=0.0,
        kl_horizon=0,
        kl_ctl_step_size=0.0,
        train_grpo_objective=False,
    )

    sharp_probs = weighting_mod.module.weight_vector_from_q(
        [0.8, 0.2],
        [0.0, 0.0],
        [1.0, 1.0],
        sharp,
        include_reference_term=False,
        normalize_by_tokens=False,
    )
    smooth_probs = weighting_mod.module.weight_vector_from_q(
        [0.8, 0.2],
        [0.0, 0.0],
        [1.0, 1.0],
        smooth,
        include_reference_term=False,
        normalize_by_tokens=False,
    )

    sharp_entropy = -sum(val * math.log(val) for val in sharp_probs)
    smooth_entropy = -sum(val * math.log(val) for val in smooth_probs)

    assert smooth_entropy > sharp_entropy


def test_weight_vector_from_q_beta_moves_weights_toward_reference(weighting_mod):
    no_ref = _build_weighting(
        weighting_mod,
        beta=0.0,
        tau=1.0,
        kl_target=0.0,
        kl_horizon=0,
        kl_ctl_step_size=0.0,
        train_grpo_objective=False,
    )
    with_ref = _build_weighting(
        weighting_mod,
        beta=1.0,
        tau=1.0,
        kl_target=0.0,
        kl_horizon=0,
        kl_ctl_step_size=0.0,
        train_grpo_objective=False,
    )

    no_ref_probs = weighting_mod.module.weight_vector_from_q(
        [0.5, 0.5],
        [0.0, 2.0],
        [1.0, 1.0],
        no_ref,
        include_reference_term=True,
        normalize_by_tokens=False,
    )
    ref_probs = weighting_mod.module.weight_vector_from_q(
        [0.5, 0.5],
        [0.0, 2.0],
        [1.0, 1.0],
        with_ref,
        include_reference_term=True,
        normalize_by_tokens=False,
    )

    assert no_ref_probs == pytest.approx([0.5, 0.5])
    assert ref_probs[1] > no_ref_probs[1]


def test_compute_weight_stats_no_longer_reweights_by_token_count(weighting_mod):
    reward_comp = SimpleNamespace(
        q_grouped=[[0.5, 0.5]],
        advantage=SimpleNamespace(grouped=[]),
    )
    ref_stats = SimpleNamespace(
        ref_logp_sum=[0.0, 0.0],
        ref_logp_sum_raw=[0.0, 0.0],
        ref_tok_counts=[1.0, 100.0],
    )
    weighting = _build_weighting(
        weighting_mod,
        beta=0.0,
        tau=1.0,
        kl_target=0.0,
        kl_horizon=0,
        kl_ctl_step_size=0.0,
        train_grpo_objective=False,
    )
    weighting.len_norm_ref = False

    stats = weighting_mod.module.compute_weight_stats(
        [["short", "long"]],
        reward_comp,
        ref_stats,
        weighting,
    )

    assert stats is not None
    assert stats.weights_grouped[0] == pytest.approx([0.5, 0.5])
    assert stats.flat_weights == pytest.approx([0.5, 0.5])
