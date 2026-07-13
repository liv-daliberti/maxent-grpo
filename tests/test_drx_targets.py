from __future__ import annotations

import pytest
import torch

from oat_drgrpo.drx_targets import (
    apply_sequence_aux_projection_gates,
    build_drgrpo_token_active_row_mask,
    build_token_primary_sequence_aux_projection,
)
from oat_drgrpo.listwise import (
    build_drx_target_bundle,
    compute_listwise_weights_from_utilities,
    compute_maxent_centered_advantages,
)


def _tensor(values) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32)


def test_drx_target_bundle_wires_semantic_remix_inputs():
    utility_grouped = _tensor([[0.90, 0.80, 1.20]])
    valid_row_mask_grouped = torch.ones_like(utility_grouped, dtype=torch.bool)
    cluster_ids_grouped = torch.tensor([[0, 0, 1]], dtype=torch.long)
    external_anchor_mass = _tensor([[0.49, 0.49, 0.02]])

    base_bundle = build_drx_target_bundle(
        utility_grouped=utility_grouped,
        ref_seq_logps_grouped=torch.zeros_like(utility_grouped),
        valid_row_mask_grouped=valid_row_mask_grouped,
        tau=0.2,
        candidate_kl_coef=0.0,
        cluster_ids_grouped=cluster_ids_grouped,
        competitive_mode_budget_grouped=_tensor([1.0]),
        competitive_mode_budget_max=1.0,
        semantic_remix_mode="anchor_rare",
    )
    anchor_bundle = build_drx_target_bundle(
        utility_grouped=utility_grouped,
        ref_seq_logps_grouped=torch.zeros_like(utility_grouped),
        valid_row_mask_grouped=valid_row_mask_grouped,
        tau=0.2,
        candidate_kl_coef=0.0,
        cluster_ids_grouped=cluster_ids_grouped,
        semantic_mass_weights_grouped=external_anchor_mass,
        competitive_mode_budget_grouped=_tensor([1.0]),
        competitive_mode_budget_max=1.0,
        semantic_remix_mode="anchor_rare",
    )

    assert base_bundle.semantic_diagnostics is not None
    assert anchor_bundle.semantic_diagnostics is not None
    assert anchor_bundle.w_star_grouped[0, 2] > base_bundle.w_star_grouped[0, 2]


def test_token_primary_sequence_aux_projects_active_maxent_targets():
    token_target_grouped = _tensor([[0.7, 0.2, 0.1], [0.0, 0.0, 0.0]])
    projection_target_grouped = _tensor([[0.0, 0.0, 0.0], [0.3, 0.3, 0.4]])
    valid_row_mask_grouped = torch.ones_like(token_target_grouped, dtype=torch.bool)

    target_grouped, group_scale = build_token_primary_sequence_aux_projection(
        token_target_grouped=token_target_grouped,
        projection_target_grouped=projection_target_grouped,
        informative_group_mask=torch.tensor([True, False], dtype=torch.bool),
        projection_group_scale=_tensor([0.0, 0.25]),
        valid_row_mask_grouped=valid_row_mask_grouped,
    )

    torch.testing.assert_close(target_grouped[0], token_target_grouped[0])
    torch.testing.assert_close(target_grouped[1], projection_target_grouped[1])
    torch.testing.assert_close(group_scale, _tensor([1.0, 0.25]))


def test_maxent_centered_advantages_reduce_to_centered_utility_at_high_tau():
    utility_grouped = _tensor([[0.2, 0.4, 0.6]])
    valid = torch.ones_like(utility_grouped, dtype=torch.bool)
    tau = 1000.0
    weights = compute_listwise_weights_from_utilities(
        utility_grouped=utility_grouped,
        ref_seq_logps_grouped=torch.zeros_like(utility_grouped),
        tau=tau,
        candidate_kl_coef=0.0,
        valid_row_mask_grouped=valid,
    )

    advantages = compute_maxent_centered_advantages(
        weights_grouped=weights,
        temperature=tau,
        valid_row_mask_grouped=valid,
    )

    expected = utility_grouped - utility_grouped.mean(dim=1, keepdim=True)
    torch.testing.assert_close(advantages, expected, atol=2e-4, rtol=0.0)


def test_maxent_centered_advantages_use_valid_row_uniform_prior():
    advantages = compute_maxent_centered_advantages(
        weights_grouped=_tensor([[0.7, 0.3, 0.0]]),
        temperature=0.25,
        valid_row_mask_grouped=torch.tensor([[True, True, False]], dtype=torch.bool),
    )

    torch.testing.assert_close(advantages, _tensor([[0.1, -0.1, 0.0]]))


def test_maxent_centered_token_active_mask_keeps_valid_zero_advantage_rows():
    active = build_drgrpo_token_active_row_mask(
        advantage_source="maxent_centered",
        informative_group_mask=torch.tensor([False, True], dtype=torch.bool),
        valid_row_mask_grouped=torch.tensor(
            [[True, True], [True, False]],
            dtype=torch.bool,
        ),
        utility_centered_advantages_grouped=_tensor([[0.0, 0.0], [0.2, 0.0]]),
    )

    assert active.tolist() == [[True, True], [True, False]]


def test_sequence_aux_projection_gates_keep_only_mixed_groups():
    target_grouped = _tensor(
        [
            [0.10, 0.90, 0.00],
            [0.50, 0.50, 0.00],
            [0.40, 0.60, 0.00],
        ]
    )
    behavior_probs_grouped = _tensor(
        [
            [0.50, 0.50, 0.00],
            [0.50, 0.50, 0.00],
            [0.50, 0.50, 0.00],
        ]
    )
    valid = torch.tensor(
        [
            [True, True, False],
            [True, True, False],
            [True, True, False],
        ],
        dtype=torch.bool,
    )
    correctness = _tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )

    gated_target, gated_scale, diagnostics = apply_sequence_aux_projection_gates(
        sequence_aux_target_grouped=target_grouped,
        sequence_aux_group_scale=_tensor([1.0, 1.0, 1.0]),
        behavior_probs_grouped=behavior_probs_grouped,
        valid_row_mask_grouped=valid,
        candidate_correctness_grouped=correctness,
        group_filter="mixed",
    )

    torch.testing.assert_close(gated_target[0], torch.zeros(3))
    torch.testing.assert_close(gated_target[1], torch.zeros(3))
    torch.testing.assert_close(gated_target[2], target_grouped[2])
    torch.testing.assert_close(gated_scale, _tensor([0.0, 0.0, 1.0]))
    assert diagnostics.rejected_group_filter_group_mask.tolist() == [
        True,
        True,
        False,
    ]
    assert diagnostics.kept_group_mask.tolist() == [False, False, True]


def test_sequence_aux_projection_gates_reject_short_or_format_dropping_targets():
    target_grouped = _tensor([[0.90, 0.10], [0.90, 0.10]])
    behavior_probs_grouped = _tensor([[0.10, 0.90], [0.10, 0.90]])
    valid = torch.ones_like(target_grouped, dtype=torch.bool)
    correctness = _tensor([[1.0, 0.0], [1.0, 0.0]])
    lengths = _tensor([[4.0, 20.0], [12.0, 12.0]])
    formatted = _tensor([[1.0, 1.0], [0.0, 1.0]])

    gated_target, gated_scale, diagnostics = apply_sequence_aux_projection_gates(
        sequence_aux_target_grouped=target_grouped,
        sequence_aux_group_scale=_tensor([1.0, 1.0]),
        behavior_probs_grouped=behavior_probs_grouped,
        valid_row_mask_grouped=valid,
        candidate_correctness_grouped=correctness,
        candidate_lengths_grouped=lengths,
        candidate_formatted_grouped=formatted,
        group_filter="mixed",
        max_expected_len_drop=2.0,
        max_expected_format_drop=0.0,
    )

    torch.testing.assert_close(gated_target, torch.zeros_like(target_grouped))
    torch.testing.assert_close(gated_scale, _tensor([0.0, 0.0]))
    assert diagnostics.rejected_len_guard_group_mask.tolist() == [True, False]
    assert diagnostics.rejected_format_guard_group_mask.tolist() == [False, True]
    assert diagnostics.target_expected_len_grouped[0].item() == pytest.approx(5.6)
    assert diagnostics.behavior_expected_len_grouped[0].item() == pytest.approx(18.4)


def test_sequence_aux_projection_gates_reject_long_or_correctness_dropping_targets():
    target_grouped = _tensor([[0.10, 0.90], [0.90, 0.10]])
    behavior_probs_grouped = _tensor([[0.90, 0.10], [0.10, 0.90]])
    valid = torch.ones_like(target_grouped, dtype=torch.bool)
    correctness = _tensor([[1.0, 0.0], [0.0, 1.0]])
    lengths = _tensor([[5.0, 20.0], [10.0, 10.0]])
    formatted = _tensor([[1.0, 1.0], [1.0, 1.0]])

    gated_target, gated_scale, diagnostics = apply_sequence_aux_projection_gates(
        sequence_aux_target_grouped=target_grouped,
        sequence_aux_group_scale=_tensor([1.0, 1.0]),
        behavior_probs_grouped=behavior_probs_grouped,
        valid_row_mask_grouped=valid,
        candidate_correctness_grouped=correctness,
        candidate_lengths_grouped=lengths,
        candidate_formatted_grouped=formatted,
        group_filter="mixed",
        max_expected_len_gain=2.0,
        max_expected_format_drop=0.0,
        min_expected_correctness_delta=0.0,
    )

    torch.testing.assert_close(gated_target, torch.zeros_like(target_grouped))
    torch.testing.assert_close(gated_scale, _tensor([0.0, 0.0]))
    assert diagnostics.rejected_len_gain_guard_group_mask.tolist() == [True, False]
    assert diagnostics.rejected_correctness_guard_group_mask.tolist() == [
        False,
        True,
    ]
    assert diagnostics.target_expected_len_grouped[0].item() == pytest.approx(18.5)
    assert diagnostics.behavior_expected_len_grouped[0].item() == pytest.approx(6.5)
    assert diagnostics.target_expected_correctness_grouped[1].item() == pytest.approx(
        0.1
    )
    assert diagnostics.behavior_expected_correctness_grouped[1].item() == pytest.approx(
        0.9
    )
