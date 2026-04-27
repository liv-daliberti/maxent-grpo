from __future__ import annotations

from types import SimpleNamespace

import torch

from oat_drgrpo.controllers import ListwiseControllerState
from oat_drgrpo.drx_targets import build_drgrpo_token_active_row_mask
from oat_drgrpo.learner.drx import ZeroMathDrxMixin
from oat_drgrpo.listwise import (
    build_drx_target_bundle,
    compute_group_centered_advantages,
    compute_listwise_weights_from_utilities,
    compute_quality_centered_semantic_drx_utilities,
    compute_drx_projection_sequence_coefficients,
    compute_token_level_clip_loss,
    masked_group_log_softmax,
)


def _tensor(values, *, requires_grad: bool = False) -> torch.Tensor:
    return torch.tensor(
        values,
        dtype=torch.float32,
        requires_grad=requires_grad,
    )


def test_record_listwise_controller_infos_logs_current_flags_and_optional_values():
    learner = ZeroMathDrxMixin()
    learner.args = SimpleNamespace(
        maxent_tau=0.25,
        beta=0.1,
        maxent_beta_controller_enabled=True,
        maxent_tau_learnable=False,
        maxent_tau_controller_enabled=True,
    )
    learner._maxent_controller_state = ListwiseControllerState(tau_metric_ema=0.7)
    infos = {}

    learner._record_listwise_controller_infos(
        infos,
        device=torch.device("cpu"),
        tau_loss_value=0.05,
        active_tau_target_metric=0.8,
    )

    torch.testing.assert_close(infos["tau"], torch.tensor(0.25))
    torch.testing.assert_close(infos["beta"], torch.tensor(0.1))
    torch.testing.assert_close(infos["weight_norm_denom"], torch.tensor(0.25))
    torch.testing.assert_close(infos["kl_controller_enabled"], torch.tensor(1.0))
    torch.testing.assert_close(infos["tau_learnable_enabled"], torch.tensor(0.0))
    torch.testing.assert_close(infos["tau_controller_enabled"], torch.tensor(1.0))
    torch.testing.assert_close(infos["tau_loss"], torch.tensor(0.05))
    torch.testing.assert_close(infos["listwise_tau_target_metric"], torch.tensor(0.8))
    torch.testing.assert_close(infos["listwise_tau_metric_ema"], torch.tensor(0.7))


def test_record_listwise_tau_adaptation_infos_logs_selection_and_signals():
    learner = ZeroMathDrxMixin()
    infos = {}

    learner._record_listwise_tau_adaptation_infos(
        infos,
        device=torch.device("cpu"),
        tau_metric_name="exploration_gain_any_correct",
        tau_metric_value=0.4,
        weight_entropy_controller=1.2,
        semantic_entropy_mu=0.6,
        exploration_gain_any_correct=0.4,
        exploration_gain_drgrpo=None,
    )

    torch.testing.assert_close(
        infos["listwise_tau_adaptation_metric_value"],
        torch.tensor(0.4),
    )
    torch.testing.assert_close(
        infos["listwise_weight_entropy_controller"],
        torch.tensor(1.2),
    )
    torch.testing.assert_close(
        infos["listwise_tau_adaptation_metric_is_semantic_entropy_mu"],
        torch.tensor(0.0),
    )
    torch.testing.assert_close(
        infos["listwise_tau_adaptation_metric_is_exploration_gain_any_correct"],
        torch.tensor(1.0),
    )
    torch.testing.assert_close(
        infos["listwise_tau_adaptation_metric_is_exploration_gain_drgrpo"],
        torch.tensor(0.0),
    )
    torch.testing.assert_close(
        infos["listwise_tau_signal_semantic_entropy_mu"],
        torch.tensor(0.6),
    )
    torch.testing.assert_close(
        infos["listwise_tau_signal_exploration_gain_any_correct"],
        torch.tensor(0.4),
    )
    assert "listwise_tau_signal_exploration_gain_drgrpo" not in infos


def test_record_listwise_runtime_infos_logs_chunk_reference_and_skip_state():
    learner = ZeroMathDrxMixin()
    learner.args = SimpleNamespace(
        maxent_backward_chunk_size=4,
        maxent_backward_token_budget=512,
        maxent_logprob_chunk_size=8,
        maxent_reference_logprobs_source="behavior",
    )
    infos = {}
    stats = {"listwise_zero_signal_skip": []}

    learner._record_listwise_runtime_infos(
        infos,
        device=torch.device("cpu"),
        skip_zero_signal_update=True,
        stats=stats,
    )

    torch.testing.assert_close(
        infos["listwise_logprob_chunk_size"],
        torch.tensor(8.0),
    )
    torch.testing.assert_close(
        infos["listwise_backward_chunk_size"],
        torch.tensor(4.0),
    )
    torch.testing.assert_close(
        infos["listwise_backward_token_budget"],
        torch.tensor(512.0),
    )
    torch.testing.assert_close(
        infos["listwise_reference_logprobs_from_model"],
        torch.tensor(0.0),
    )
    torch.testing.assert_close(
        infos["listwise_reference_logprobs_from_behavior"],
        torch.tensor(1.0),
    )
    torch.testing.assert_close(
        infos["listwise_zero_signal_skip"],
        torch.tensor(1.0),
    )
    torch.testing.assert_close(stats["listwise_zero_signal_skip"][0], torch.tensor(1.0))


def test_listwise_scalar_and_prompt_gather_helpers():
    learner = ZeroMathDrxMixin()
    learner.strategy = SimpleNamespace(world_size=2)
    learner._all_gather_same_shape_tensor = lambda value: torch.cat(
        [value, value + 10.0],
        dim=0,
    )

    scalar = learner._listwise_scalar(3.0, device=torch.device("cpu"))
    gathered = learner._all_gather_prompt_values(torch.tensor([1.0, 2.0]))

    torch.testing.assert_close(scalar, torch.tensor(3.0))
    torch.testing.assert_close(
        gathered,
        torch.tensor([[1.0, 2.0], [11.0, 12.0]]),
    )


def test_masked_prompt_mean_and_corr_helpers_handle_empty_and_degenerate_masks():
    learner = ZeroMathDrxMixin()
    values = torch.tensor([1.0, 3.0, 5.0])
    mask = torch.tensor([True, False, True])

    torch.testing.assert_close(
        learner._masked_prompt_mean(values, mask),
        torch.tensor(3.0),
    )
    torch.testing.assert_close(
        learner._masked_prompt_mean(values, torch.zeros_like(mask)),
        torch.tensor(0.0),
    )
    torch.testing.assert_close(
        learner._masked_prompt_corr(values, values * 2.0, mask),
        torch.tensor(1.0),
    )
    torch.testing.assert_close(
        learner._masked_prompt_corr(values, torch.ones_like(values), mask),
        torch.tensor(0.0),
    )


def _controller_test_learner() -> ZeroMathDrxMixin:
    learner = ZeroMathDrxMixin()
    learner.objective = "maxent_listwise"
    learner.global_step = 5
    learner._fixed_listwise_config = {}
    learner._fixed_listwise_tau = None
    learner._fixed_listwise_beta = None
    learner._maxent_tau_log = None
    learner._maxent_tau_optimizer = None
    learner._maxent_controller_state = ListwiseControllerState()
    learner.args = SimpleNamespace(
        beta=0.2,
        kl_ctl_step_size=0.2,
        kl_horizon=10,
        kl_target=0.1,
        maxent_beta_controller_enabled=False,
        maxent_tau=0.5,
        maxent_tau_adaptation_metric="semantic_entropy_mu",
        maxent_tau_controller_enabled=True,
        maxent_tau_controller_target_entropy=None,
        maxent_tau_learnable=False,
        maxent_tau_lr=1.0,
        maxent_tau_max=10.0,
        maxent_tau_min=0.01,
        maxent_tau_target_metric=0.5,
        maxent_tau_target_metric_final=None,
        maxent_tau_target_metric_horizon=0,
        maxent_tau_target_metric_peak=None,
        maxent_tau_target_metric_peak_step=0,
        maxent_tau_target_metric_start=None,
        maxent_tau_warmup_steps=0,
    )
    return learner


def test_apply_listwise_controller_updates_records_and_updates_tau():
    learner = _controller_test_learner()
    infos = {}

    learner._apply_listwise_controller_updates(
        infos,
        device=torch.device("cpu"),
        skip_zero_signal_update=False,
        weight_entropy_controller=0.9,
        semantic_entropy_mu=0.2,
        exploration_gain_any_correct=None,
        exploration_gain_drgrpo=None,
    )

    assert learner.args.maxent_tau > 0.5
    torch.testing.assert_close(
        infos["listwise_tau_adaptation_metric_value"],
        torch.tensor(0.2),
    )
    torch.testing.assert_close(
        infos["listwise_weight_entropy_controller"],
        torch.tensor(0.9),
    )
    torch.testing.assert_close(infos["tau"], torch.tensor(learner.args.maxent_tau))
    assert learner._maxent_controller_state.tau_metric_ema == 0.2


def test_apply_listwise_controller_updates_skip_still_records_controller_state():
    learner = _controller_test_learner()
    learner.args.maxent_beta_controller_enabled = True
    infos = {}

    learner._apply_listwise_controller_updates(
        infos,
        device=torch.device("cpu"),
        skip_zero_signal_update=True,
        weight_entropy_controller=None,
        semantic_entropy_mu=None,
        exploration_gain_any_correct=None,
        exploration_gain_drgrpo=None,
        measured_kl=1.0,
        update_beta_when_skipped=False,
    )

    torch.testing.assert_close(infos["tau"], torch.tensor(0.5))
    torch.testing.assert_close(infos["beta"], torch.tensor(0.2))
    assert "listwise_tau_adaptation_metric_value" not in infos


def test_group_centered_advantages_match_grpo_reward_baseline_with_valid_rows():
    reward_grouped = _tensor([[1.0, 0.0, 0.5], [0.2, 0.8, 100.0]])
    valid_row_mask_grouped = torch.tensor(
        [[True, True, True], [True, True, False]],
        dtype=torch.bool,
    )

    advantages = compute_group_centered_advantages(
        reward_grouped=reward_grouped,
        valid_row_mask_grouped=valid_row_mask_grouped,
    )

    torch.testing.assert_close(
        advantages,
        _tensor([[0.5, -0.5, 0.0], [-0.3, 0.3, 0.0]]),
        atol=1e-6,
        rtol=0.0,
    )


def test_drgrpo_token_clip_loss_matches_manual_clipping_behavior():
    new_logps = torch.log(_tensor([[1.30, 0.80], [0.70, 1.25]]))
    behavior_logps = torch.zeros_like(new_logps)
    response_masks = torch.ones_like(new_logps, dtype=torch.bool)
    row_advantages = _tensor([2.0, -1.5])

    per_row_loss, _, is_low_clipped, is_high_clipped = compute_token_level_clip_loss(
        new_logps=new_logps,
        behavior_logps=behavior_logps,
        response_masks=response_masks,
        row_advantages=row_advantages,
        clip_low=0.2,
        clip_high=0.2,
        constant_normalizer=2.0,
    )

    expected = _tensor([-2.0, 1.5375])
    torch.testing.assert_close(per_row_loss, expected, atol=1e-6, rtol=0.0)
    torch.testing.assert_close(
        is_low_clipped,
        torch.tensor([[False, False], [True, False]], dtype=torch.bool),
    )
    torch.testing.assert_close(
        is_high_clipped,
        torch.tensor([[True, False], [False, False]], dtype=torch.bool),
    )


def test_utility_centered_token_mask_keeps_zero_advantage_valid_rows():
    mask = build_drgrpo_token_active_row_mask(
        advantage_source="utility_centered",
        informative_group_mask=torch.tensor([True], dtype=torch.bool),
        valid_row_mask_grouped=torch.tensor([[True, True, False]], dtype=torch.bool),
        utility_centered_advantages_grouped=_tensor([[0.5, 0.0, 0.0]]),
    )

    torch.testing.assert_close(
        mask,
        torch.tensor([[True, True, False]], dtype=torch.bool),
    )


def test_build_drx_target_bundle_separates_informative_and_neutral_groups():
    utility_grouped = _tensor([[1.0, 0.0], [0.5, 0.5]])
    valid_row_mask_grouped = torch.tensor(
        [[True, True], [True, True]], dtype=torch.bool
    )
    bundle = build_drx_target_bundle(
        utility_grouped=utility_grouped,
        ref_seq_logps_grouped=torch.zeros_like(utility_grouped),
        valid_row_mask_grouped=valid_row_mask_grouped,
        tau=0.3,
        candidate_kl_coef=0.0,
        neutral_projection_coef=0.7,
    )

    assert bool(bundle.informative_group_mask[0].item())
    assert not bool(bundle.neutral_group_mask[0].item())
    assert not bool(bundle.informative_group_mask[1].item())
    assert bool(bundle.neutral_group_mask[1].item())

    torch.testing.assert_close(bundle.token_target_grouped[0], bundle.w_star_grouped[0])
    torch.testing.assert_close(
        bundle.token_target_grouped[1], torch.zeros_like(bundle.w_star_grouped[1])
    )
    torch.testing.assert_close(
        bundle.projection_target_grouped[0],
        torch.zeros_like(bundle.w_star_grouped[0]),
    )
    torch.testing.assert_close(
        bundle.projection_target_grouped[1],
        bundle.w_star_grouped[1],
    )
    torch.testing.assert_close(bundle.projection_group_scale, _tensor([0.0, 0.7]))


def test_quality_centered_semantic_drx_utility_balanced_mix_has_zero_gate():
    reward_grouped = _tensor([[1.0, 0.0, 0.5, 1.0]])
    cluster_ids_grouped = torch.tensor([[0, 1, 2, 2]], dtype=torch.long)
    candidate_correctness_grouped = _tensor([[1.0, 0.0, 0.0, 1.0]])
    valid_row_mask_grouped = torch.tensor([[True, True, True, True]], dtype=torch.bool)
    ref_seq_logps_grouped = _tensor([[0.2, -0.4, 0.1, -0.2]])

    utility_grouped, quality_grouped, semantic_diag = (
        compute_quality_centered_semantic_drx_utilities(
            reward_grouped=reward_grouped,
            cluster_ids_grouped=cluster_ids_grouped,
            semantic_entropy_lambda=0.1,
            candidate_correctness_grouped=candidate_correctness_grouped,
            valid_row_mask_grouped=valid_row_mask_grouped,
            semantic_correctness_target_frac=0.5,
            semantic_correctness_sharpness=4.0,
        )
    )

    expected_quality = _tensor([[1.0, 0.0, 0.5, 1.0]])
    torch.testing.assert_close(quality_grouped, expected_quality, atol=1e-6, rtol=0.0)

    cluster_probs = _tensor([0.25, 0.25, 0.5])
    entropy = -(cluster_probs * torch.log(cluster_probs)).sum()
    rare_bonus = max(0.0, float(-torch.log(cluster_probs[0]) - entropy))
    expected_bonus = _tensor([[rare_bonus, rare_bonus, 0.0, 0.0]])
    torch.testing.assert_close(
        semantic_diag.semantic_surprisal_grouped,
        expected_bonus,
        atol=1e-6,
        rtol=0.0,
    )
    torch.testing.assert_close(
        semantic_diag.semantic_gate_grouped,
        torch.zeros((1,), dtype=torch.float32),
        atol=1e-6,
        rtol=0.0,
    )

    expected_utility = expected_quality
    torch.testing.assert_close(
        utility_grouped,
        expected_utility,
        atol=1e-6,
        rtol=0.0,
    )

    weights_grouped = compute_listwise_weights_from_utilities(
        utility_grouped=utility_grouped,
        ref_seq_logps_grouped=ref_seq_logps_grouped,
        tau=0.5,
        candidate_kl_coef=0.3,
        valid_row_mask_grouped=valid_row_mask_grouped,
    )
    manual_logits = (expected_utility + (0.3 * ref_seq_logps_grouped)) / (0.5 + 0.3)
    manual_weights = torch.softmax(manual_logits, dim=1)
    torch.testing.assert_close(
        weights_grouped,
        manual_weights,
        atol=1e-6,
        rtol=0.0,
    )


def test_quality_centered_semantic_drx_utility_tanh_gate_tracks_correctness_rate():
    reward_grouped = _tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 0.0],
        ]
    )
    cluster_ids_grouped = torch.tensor(
        [
            [0, 1, 2, 2],
            [0, 1, 2, 2],
        ],
        dtype=torch.long,
    )
    valid_row_mask_grouped = torch.ones_like(reward_grouped, dtype=torch.bool)

    utility_grouped, quality_grouped, semantic_diag = (
        compute_quality_centered_semantic_drx_utilities(
            reward_grouped=reward_grouped,
            cluster_ids_grouped=cluster_ids_grouped,
            semantic_entropy_lambda=0.1,
            candidate_correctness_grouped=reward_grouped,
            valid_row_mask_grouped=valid_row_mask_grouped,
            semantic_correctness_target_frac=0.5,
            semantic_correctness_sharpness=4.0,
        )
    )

    cluster_probs = _tensor([0.25, 0.25, 0.5])
    entropy = -(cluster_probs * torch.log(cluster_probs)).sum()
    rare_bonus = max(0.0, float(-torch.log(cluster_probs[0]) - entropy))
    expected_surprisal = _tensor(
        [
            [rare_bonus, rare_bonus, 0.0, 0.0],
            [rare_bonus, rare_bonus, 0.0, 0.0],
        ]
    )
    expected_gates = -torch.tanh(4.0 * (_tensor([0.25, 0.75]) - 0.5))

    torch.testing.assert_close(quality_grouped, reward_grouped)
    torch.testing.assert_close(
        semantic_diag.semantic_surprisal_grouped,
        expected_surprisal,
        atol=1e-6,
        rtol=0.0,
    )
    torch.testing.assert_close(
        semantic_diag.semantic_gate_grouped,
        expected_gates,
        atol=1e-6,
        rtol=0.0,
    )
    torch.testing.assert_close(
        utility_grouped,
        reward_grouped + 0.1 * expected_gates[:, None] * expected_surprisal,
        atol=1e-6,
        rtol=0.0,
    )


def test_quality_centered_semantic_drx_utility_ignores_candidate_length():
    reward_grouped = _tensor([[0.0, 0.0, 0.0, 0.0]])
    cluster_ids_grouped = torch.tensor([[0, 1, 2, 2]], dtype=torch.long)
    candidate_correctness_grouped = _tensor([[0.0, 0.0, 0.0, 0.0]])
    valid_row_mask_grouped = torch.tensor([[True, True, True, True]], dtype=torch.bool)

    utility_grouped, _, semantic_diag = compute_quality_centered_semantic_drx_utilities(
        reward_grouped=reward_grouped,
        cluster_ids_grouped=cluster_ids_grouped,
        semantic_entropy_lambda=0.1,
        candidate_correctness_grouped=candidate_correctness_grouped,
        valid_row_mask_grouped=valid_row_mask_grouped,
        semantic_correctness_target_frac=0.5,
        semantic_correctness_sharpness=4.0,
    )

    cluster_probs = _tensor([0.25, 0.25, 0.5])
    entropy = -(cluster_probs * torch.log(cluster_probs)).sum()
    rare_bonus = max(0.0, float(-torch.log(cluster_probs[0]) - entropy))
    expected_bonus = _tensor([[rare_bonus, rare_bonus, 0.0, 0.0]])
    torch.testing.assert_close(
        semantic_diag.semantic_surprisal_grouped,
        expected_bonus,
        atol=1e-6,
        rtol=0.0,
    )

    expected_gate = -torch.tanh(_tensor([4.0 * (0.0 - 0.5)]))[0]
    expected_utility = _tensor(
        [[0.1 * expected_gate * rare_bonus, 0.1 * expected_gate * rare_bonus, 0.0, 0.0]]
    )
    expected_utility[0, 1] = 0.1 * rare_bonus
    expected_utility[0, 1] *= expected_gate * (50.0 / 50.0) ** 2
    torch.testing.assert_close(
        utility_grouped,
        expected_utility,
        atol=1e-6,
        rtol=0.0,
    )


def test_drx_projection_coefficients_match_autograd_and_row_sharded_surrogate():
    policy_seq_logps_grouped = _tensor([[0.4, -0.2, 0.1]], requires_grad=True)
    projection_target_grouped = _tensor([[0.2, 0.5, 0.3]])
    projection_group_scale = _tensor([0.75])
    valid_row_mask_grouped = torch.tensor([[True, True, True]], dtype=torch.bool)

    policy_log_probs_grouped = masked_group_log_softmax(
        policy_seq_logps_grouped,
        valid_row_mask_grouped,
    )
    per_group_ce = -(projection_target_grouped * policy_log_probs_grouped).sum(dim=1)
    grouped_loss = (projection_group_scale * per_group_ce).sum()
    grouped_loss.backward()
    grouped_grad = policy_seq_logps_grouped.grad.detach().clone()

    coeffs = compute_drx_projection_sequence_coefficients(
        policy_seq_logps_grouped=policy_seq_logps_grouped.detach(),
        projection_target_grouped=projection_target_grouped,
        projection_group_scale=projection_group_scale,
        valid_row_mask_grouped=valid_row_mask_grouped,
        normalizer_total_group_weight=1.0,
    )
    torch.testing.assert_close(coeffs, grouped_grad, atol=1e-6, rtol=0.0)

    row_sharded_policy = policy_seq_logps_grouped.detach().clone().requires_grad_(True)
    world_size = int(row_sharded_policy.size(1))
    local_objectives = [
        float(world_size) * row_sharded_policy[0, row_idx] * coeffs[0, row_idx].detach()
        for row_idx in range(world_size)
    ]
    row_sharded_loss = torch.stack(local_objectives).mean()
    row_sharded_loss.backward()

    torch.testing.assert_close(
        row_sharded_policy.grad,
        grouped_grad,
        atol=1e-6,
        rtol=0.0,
    )


def test_batch_two_row_sharded_projection_scaling_matches_grouped_projection_loss():
    policy_seq_logps_grouped = _tensor(
        [[0.4, -0.2, 0.1], [-0.3, 0.5, 0.2]],
        requires_grad=True,
    )
    projection_target_grouped = _tensor([[0.2, 0.5, 0.3], [0.1, 0.2, 0.7]])
    projection_group_scale = _tensor([0.75, 0.40])
    valid_row_mask_grouped = torch.tensor(
        [[True, True, True], [True, True, True]],
        dtype=torch.bool,
    )

    policy_log_probs_grouped = masked_group_log_softmax(
        policy_seq_logps_grouped,
        valid_row_mask_grouped,
    )
    per_group_ce = -(projection_target_grouped * policy_log_probs_grouped).sum(dim=1)
    grouped_loss = (projection_group_scale * per_group_ce).sum() / float(
        policy_seq_logps_grouped.size(0)
    )
    grouped_loss.backward()
    grouped_grad = policy_seq_logps_grouped.grad.detach().clone()

    coeffs = compute_drx_projection_sequence_coefficients(
        policy_seq_logps_grouped=policy_seq_logps_grouped.detach(),
        projection_target_grouped=projection_target_grouped,
        projection_group_scale=projection_group_scale,
        valid_row_mask_grouped=valid_row_mask_grouped,
        normalizer_total_group_weight=float(policy_seq_logps_grouped.size(0)),
    )
    torch.testing.assert_close(coeffs, grouped_grad, atol=1e-6, rtol=0.0)

    row_sharded_policy = policy_seq_logps_grouped.detach().clone().requires_grad_(True)
    world_size = int(row_sharded_policy.size(1))
    local_objectives = []
    for row_idx in range(world_size):
        local_objectives.append(
            float(world_size)
            * (row_sharded_policy[:, row_idx] * coeffs[:, row_idx].detach()).sum()
        )
    row_sharded_loss = torch.stack(local_objectives).mean()
    row_sharded_loss.backward()

    torch.testing.assert_close(
        row_sharded_policy.grad,
        grouped_grad,
        atol=1e-6,
        rtol=0.0,
    )


def test_row_sharded_drgrpo_token_scaling_matches_grouped_valid_row_mean():
    new_logps = (
        torch.log(
            _tensor(
                [
                    [1.10, 0.90],
                    [0.75, 1.25],
                    [1.05, 1.15],
                ]
            )
        )
        .detach()
        .requires_grad_(True)
    )
    behavior_logps = torch.zeros_like(new_logps)
    response_masks = torch.ones_like(new_logps, dtype=torch.bool)
    weighted_row_advantages = _tensor([1.4, 0.0, 0.3])
    active_row_mask = torch.tensor([True, True, True], dtype=torch.bool)

    grouped_row_loss, _, _, _ = compute_token_level_clip_loss(
        new_logps=new_logps,
        behavior_logps=behavior_logps,
        response_masks=response_masks,
        row_advantages=weighted_row_advantages,
        clip_low=0.2,
        clip_high=0.2,
        constant_normalizer=2.0,
    )
    grouped_loss = grouped_row_loss[active_row_mask].mean()
    grouped_loss.backward()
    grouped_grad = new_logps.grad.detach().clone()

    row_sharded_logps = new_logps.detach().clone().requires_grad_(True)
    row_sharded_row_loss, _, _, _ = compute_token_level_clip_loss(
        new_logps=row_sharded_logps,
        behavior_logps=behavior_logps,
        response_masks=response_masks,
        row_advantages=weighted_row_advantages,
        clip_low=0.2,
        clip_high=0.2,
        constant_normalizer=2.0,
    )
    world_size = int(row_sharded_logps.size(0))
    active_row_count = int(active_row_mask.to(torch.int64).sum().item())
    local_objectives = [
        (
            float(world_size) * row_sharded_row_loss[row_idx] / float(active_row_count)
            if bool(active_row_mask[row_idx].item())
            else row_sharded_row_loss[row_idx] * 0.0
        )
        for row_idx in range(world_size)
    ]
    row_sharded_loss = torch.stack(local_objectives).mean()
    row_sharded_loss.backward()

    torch.testing.assert_close(
        row_sharded_logps.grad,
        grouped_grad,
        atol=1e-6,
        rtol=0.0,
    )


def test_batch_two_row_sharded_drgrpo_token_scaling_matches_grouped_valid_row_mean():
    new_logps_grouped = (
        torch.log(
            _tensor(
                [
                    [[1.10, 0.90], [0.75, 1.25], [1.05, 1.15]],
                    [[0.95, 1.05], [1.20, 0.85], [0.80, 1.10]],
                ]
            )
        )
        .detach()
        .requires_grad_(True)
    )
    behavior_logps_grouped = torch.zeros_like(new_logps_grouped)
    response_masks_grouped = torch.ones_like(new_logps_grouped, dtype=torch.bool)
    weighted_row_advantages_grouped = _tensor([[1.4, 0.0, 0.3], [0.2, 1.1, 0.0]])
    active_row_mask_grouped = torch.tensor(
        [[True, True, True], [True, True, True]],
        dtype=torch.bool,
    )

    flat_new_logps = new_logps_grouped.reshape(-1, new_logps_grouped.size(-1))
    flat_behavior_logps = behavior_logps_grouped.reshape_as(flat_new_logps)
    flat_response_masks = response_masks_grouped.reshape_as(flat_new_logps)
    flat_weighted_row_advantages = weighted_row_advantages_grouped.reshape(-1)
    flat_active_row_mask = active_row_mask_grouped.reshape(-1)

    grouped_row_loss, _, _, _ = compute_token_level_clip_loss(
        new_logps=flat_new_logps,
        behavior_logps=flat_behavior_logps,
        response_masks=flat_response_masks,
        row_advantages=flat_weighted_row_advantages,
        clip_low=0.2,
        clip_high=0.2,
        constant_normalizer=2.0,
    )
    grouped_loss = grouped_row_loss[flat_active_row_mask].mean()
    grouped_loss.backward()
    grouped_grad = new_logps_grouped.grad.detach().clone()

    row_sharded_logps = new_logps_grouped.detach().clone().requires_grad_(True)
    row_sharded_row_loss, _, _, _ = compute_token_level_clip_loss(
        new_logps=row_sharded_logps.reshape(-1, row_sharded_logps.size(-1)),
        behavior_logps=flat_behavior_logps,
        response_masks=flat_response_masks,
        row_advantages=flat_weighted_row_advantages,
        clip_low=0.2,
        clip_high=0.2,
        constant_normalizer=2.0,
    )
    row_sharded_row_loss = row_sharded_row_loss.reshape_as(
        weighted_row_advantages_grouped
    )
    world_size = int(row_sharded_logps.size(1))
    active_row_count = int(active_row_mask_grouped.to(torch.int64).sum().item())
    local_objectives = []
    for row_idx in range(world_size):
        local_loss = row_sharded_row_loss[:, row_idx][
            active_row_mask_grouped[:, row_idx]
        ].sum()
        local_objectives.append(
            float(world_size) * local_loss / float(active_row_count)
        )
    row_sharded_loss = torch.stack(local_objectives).mean()
    row_sharded_loss.backward()

    torch.testing.assert_close(
        row_sharded_logps.grad,
        grouped_grad,
        atol=1e-6,
        rtol=0.0,
    )
