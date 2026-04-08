from __future__ import annotations

import ast
import math
from pathlib import Path

import pytest
import torch

from oat_zero_ext.listwise import (build_listwise_q_targets,
                                   build_padded_action_logprobs,
                                   cap_last_valid_token_pos_for_zero_advantage,
                                   clamp_listwise_tau,
                                   choose_prefix_chunk_size_for_token_budget,
                                   collect_weight_entropy,
                                   collect_weight_entropy_stats,
                                   compute_learnable_tau_loss,
                                   compute_listwise_clip_advantages,
                                   compute_listwise_sequence_coefficients,
                                   compute_token_level_clip_loss,
                                   compute_listwise_weights,
                                   gather_selected_logps_chunked,
                                   iter_budgeted_row_chunks,
                                   iter_fixed_row_chunks,
                                   iter_grouped_minibatch_indices,
                                   ListwiseControllerState,
                                   mask_and_normalize_listwise_q_targets,
                                   mask_invalid_logit_columns,
                                   masked_group_log_softmax,
                                   maybe_update_listwise_beta,
                                   maybe_update_listwise_tau,
                                    normalize_maxent_clip_mode,
                                    normalize_oat_objective)
from oat_zero_ext.listwise import (resolve_token_id_upper_bound,
                                   resolve_listwise_target_entropy,
                                   sanitize_scoring_token_ids)


def _zero_math_learner_method_source(name: str) -> str:
    source_path = (
        Path(__file__).resolve().parents[2]
        / "understand-r1-zero"
        / "train_zero_math.py"
    )
    source_text = source_path.read_text(encoding="utf-8")
    module = ast.parse(source_text)
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == "ZeroMathLearner":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == name:
                    method_source = ast.get_source_segment(source_text, item)
                    if method_source is not None:
                        return method_source
                    break
    raise AssertionError(f"Could not find ZeroMathLearner.{name} in {source_path}")


def test_normalize_oat_objective_accepts_aliases() -> None:
    assert normalize_oat_objective("grpo") == "grpo"
    assert normalize_oat_objective("listwise") == "maxent_listwise"
    assert normalize_oat_objective("maxent_listwise") == "maxent_listwise"
    with pytest.raises(ValueError, match="objective must be one of"):
        normalize_oat_objective("maxent_entropy")


def test_normalize_maxent_clip_mode_accepts_aliases() -> None:
    assert normalize_maxent_clip_mode("sequence") == "sequence"
    assert normalize_maxent_clip_mode("ppo_token") == "token"
    assert normalize_maxent_clip_mode("off") == "none"
    with pytest.raises(ValueError, match="maxent_clip_mode must be one of"):
        normalize_maxent_clip_mode("trajectory")


def test_build_listwise_q_targets_matches_grouped_softmax() -> None:
    rewards = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float32)
    q_targets = build_listwise_q_targets(
        rewards,
        group_size=2,
        temperature=1.0,
        epsilon=0.0,
    )
    expected = torch.stack(
        [
            torch.softmax(torch.tensor([0.0, 1.0]), dim=0),
            torch.softmax(torch.tensor([2.0, 3.0]), dim=0),
        ]
    ).to(torch.float32)
    assert torch.allclose(q_targets, expected, atol=1e-6, rtol=1e-6)


def test_mask_and_normalize_listwise_q_targets_renormalizes_valid_rows_only() -> None:
    q_grouped = torch.tensor([[0.2, 0.3, 0.5]], dtype=torch.float32)
    row_mask_grouped = torch.tensor([[True, False, True]], dtype=torch.bool)
    masked = mask_and_normalize_listwise_q_targets(
        q_grouped,
        row_mask_grouped=row_mask_grouped,
        context="test",
    )
    expected = torch.tensor([[2.0 / 7.0, 0.0, 5.0 / 7.0]], dtype=torch.float32)
    assert torch.allclose(masked, expected, atol=1e-6, rtol=1e-6)


def test_iter_grouped_minibatch_indices_preserves_whole_prompt_groups() -> None:
    batches = list(
        iter_grouped_minibatch_indices(
            total_rows=8,
            group_size=2,
            flat_batch_size=4,
            prompt_permutation=[2, 0, 3, 1],
        )
    )
    assert [batch.tolist() for batch in batches] == [[4, 5, 0, 1], [6, 7, 2, 3]]


def test_iter_grouped_minibatch_indices_rejects_partial_groups() -> None:
    with pytest.raises(ValueError, match="divisible by num_samples"):
        list(
            iter_grouped_minibatch_indices(
                total_rows=8,
                group_size=2,
                flat_batch_size=3,
            )
        )


def test_iter_fixed_row_chunks_keeps_constant_row_schedule() -> None:
    assert list(iter_fixed_row_chunks(8, chunk_size=2)) == [
        (0, 2),
        (2, 4),
        (4, 6),
        (6, 8),
    ]


def test_iter_budgeted_row_chunks_falls_back_to_single_rows_for_long_sequences() -> None:
    assert list(
        iter_budgeted_row_chunks(
            [350, 3400, 3300, 500],
            max_rows=2,
            token_budget=2048,
        )
    ) == [(0, 1), (1, 2), (2, 3), (3, 4)]


def test_compute_listwise_weights_matches_paper_closed_form() -> None:
    q_grouped = torch.tensor([[0.8, 0.2]], dtype=torch.float32)
    ref_seq_logps_grouped = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    weights = compute_listwise_weights(
        q_grouped=q_grouped,
        ref_seq_logps_grouped=ref_seq_logps_grouped,
        tau=0.5,
        beta=0.08,
    )
    expected = torch.softmax(
        (torch.log(q_grouped) + 0.08 * ref_seq_logps_grouped) / (0.5 + 0.08),
        dim=1,
    )
    assert torch.allclose(weights, expected, atol=1e-6, rtol=1e-6)


def test_compute_listwise_weights_ignores_ref_logps_when_beta_is_zero() -> None:
    q_grouped = torch.tensor([[0.8, 0.2]], dtype=torch.float32)
    ref_seq_logps_grouped = torch.tensor([[10.0, -10.0]], dtype=torch.float32)
    weights = compute_listwise_weights(
        q_grouped=q_grouped,
        ref_seq_logps_grouped=ref_seq_logps_grouped,
        tau=0.5,
        beta=0.0,
    )
    expected = torch.softmax(torch.log(q_grouped) / 0.5, dim=1)
    assert torch.allclose(weights, expected, atol=1e-6, rtol=1e-6)


def test_compute_listwise_weights_respects_zero_mass_support() -> None:
    q_grouped = torch.tensor([[0.75, 0.0, 0.25]], dtype=torch.float32)
    ref_seq_logps_grouped = torch.tensor([[5.0, 100.0, -5.0]], dtype=torch.float32)
    weights = compute_listwise_weights(
        q_grouped=q_grouped,
        ref_seq_logps_grouped=ref_seq_logps_grouped,
        tau=0.5,
        beta=0.3,
    )
    assert weights[0, 1].item() == pytest.approx(0.0, abs=1e-8)
    assert weights.sum(dim=1).tolist() == pytest.approx([1.0], rel=1e-6)


def test_collect_weight_entropy_matches_manual_entropy() -> None:
    weights = torch.tensor([[0.8, 0.2], [0.5, 0.5]], dtype=torch.float32)
    entropy_mean, entropy_min, entropy_max, adv_samples = collect_weight_entropy(weights)
    first = -(0.8 * math.log(0.8) + 0.2 * math.log(0.2))
    second = math.log(2.0)
    assert entropy_mean == pytest.approx((first + second) / 2.0, rel=1e-6)
    assert entropy_min == pytest.approx(min(first, second), rel=1e-6)
    assert entropy_max == pytest.approx(max(first, second), rel=1e-6)
    assert adv_samples == pytest.approx([0.3, -0.3, 0.0, 0.0], rel=1e-6)


def test_collect_weight_entropy_stats_matches_manual_entropy() -> None:
    weights = torch.tensor([[0.8, 0.2], [0.5, 0.5]], dtype=torch.float32)
    entropy_mean, entropy_min, entropy_max = collect_weight_entropy_stats(weights)
    first = -(0.8 * math.log(0.8) + 0.2 * math.log(0.2))
    second = math.log(2.0)
    assert float(entropy_mean) == pytest.approx((first + second) / 2.0, rel=1e-6)
    assert float(entropy_min) == pytest.approx(min(first, second), rel=1e-6)
    assert float(entropy_max) == pytest.approx(max(first, second), rel=1e-6)


def test_sanitize_scoring_token_ids_replaces_invalid_entries() -> None:
    class _Tokenizer:
        pad_token_id = 0
        eos_token_id = 1

    result = sanitize_scoring_token_ids(
        torch.tensor([[0, 5, -1, 2]], dtype=torch.long),
        upper_bound=4,
        tokenizer=_Tokenizer(),
    )
    assert result.invalid_count == 2
    assert result.replacement_id == 0
    assert result.token_ids.tolist() == [[0, 0, 0, 2]]


def test_mask_invalid_logit_columns_masks_tokenizer_inaccessible_tail() -> None:
    logits = torch.zeros((1, 2, 5), dtype=torch.float32)
    masked = mask_invalid_logit_columns(logits, valid_vocab_size=3)
    mask_value = torch.finfo(masked.dtype).min
    assert masked[..., :3].tolist() == logits[..., :3].tolist()
    assert torch.all(masked[..., 3:] == mask_value)


def test_resolve_token_id_upper_bound_uses_shared_model_tokenizer_limit() -> None:
    class _Embeddings:
        num_embeddings = 10

    class _Model:
        config = type("Cfg", (), {"vocab_size": 8})()
        vocab_size = 9

        def get_input_embeddings(self):
            return _Embeddings()

    class _Tokenizer:
        vocab_size = 12

        def __len__(self):
            return 12

    assert resolve_token_id_upper_bound(_Model(), _Tokenizer()) == 10


def test_maybe_update_listwise_tau_noops_without_target_entropy() -> None:
    state = ListwiseControllerState()
    assert (
        maybe_update_listwise_tau(
            0.5,
            measured_entropy=0.2,
            global_step=10,
            state=state,
            target_entropy=None,
            target_entropy_start=None,
            target_entropy_peak=None,
            target_entropy_peak_step=0,
            target_entropy_final=None,
            target_entropy_horizon=0,
            tau_lr=0.1,
            tau_min=0.0,
            tau_max=0.0,
            tau_warmup_steps=0,
        )
        == pytest.approx(0.5)
    )


def test_maybe_update_listwise_tau_updates_and_tracks_ema() -> None:
    state = ListwiseControllerState()
    updated_tau = maybe_update_listwise_tau(
        0.5,
        measured_entropy=0.2,
        global_step=10,
        state=state,
        target_entropy=0.6,
        target_entropy_start=None,
        target_entropy_peak=None,
        target_entropy_peak_step=0,
        target_entropy_final=None,
        target_entropy_horizon=0,
        tau_lr=0.5,
        tau_min=0.0,
        tau_max=0.0,
        tau_warmup_steps=0,
    )
    assert updated_tau > 0.5
    assert state.tau_entropy_ema == pytest.approx(0.2, rel=1e-6)


def test_resolve_listwise_target_entropy_supports_piecewise_sharp_loose_sharp() -> None:
    assert (
        resolve_listwise_target_entropy(
            target_entropy=None,
            target_entropy_start=1.7,
            target_entropy_peak=1.95,
            target_entropy_peak_step=10,
            target_entropy_final=1.75,
            target_entropy_horizon=40,
            global_step=0,
        )
        == pytest.approx(1.7)
    )
    assert (
        resolve_listwise_target_entropy(
            target_entropy=None,
            target_entropy_start=1.7,
            target_entropy_peak=1.95,
            target_entropy_peak_step=10,
            target_entropy_final=1.75,
            target_entropy_horizon=40,
            global_step=10,
        )
        == pytest.approx(1.95)
    )
    assert (
        resolve_listwise_target_entropy(
            target_entropy=None,
            target_entropy_start=1.7,
            target_entropy_peak=1.95,
            target_entropy_peak_step=10,
            target_entropy_final=1.75,
            target_entropy_horizon=40,
            global_step=25,
        )
        == pytest.approx(1.85)
    )
    assert (
        resolve_listwise_target_entropy(
            target_entropy=None,
            target_entropy_start=1.7,
            target_entropy_peak=1.95,
            target_entropy_peak_step=10,
            target_entropy_final=1.75,
            target_entropy_horizon=40,
            global_step=50,
        )
        == pytest.approx(1.75)
    )


def test_maybe_update_listwise_beta_matches_kl_controller_formula() -> None:
    updated_beta = maybe_update_listwise_beta(
        0.8,
        measured_kl=0.16,
        kl_target=0.08,
        kl_horizon=100,
        kl_ctl_step_size=0.5,
    )
    assert updated_beta == pytest.approx(0.8 * (1.0 + 0.5 / 100.0), rel=1e-6)


def test_clamp_listwise_tau_respects_bounds() -> None:
    assert clamp_listwise_tau(0.01, tau_min=0.2, tau_max=0.0) == pytest.approx(0.2)
    assert clamp_listwise_tau(3.0, tau_min=0.0, tau_max=0.7) == pytest.approx(0.7)


def test_compute_learnable_tau_loss_increases_tau_when_entropy_is_low() -> None:
    tau_log = torch.nn.Parameter(torch.tensor(math.log(0.5), dtype=torch.float32))
    optimizer = torch.optim.SGD([tau_log], lr=0.5)
    loss = compute_learnable_tau_loss(
        tau_log,
        measured_entropy=1.2,
        target_entropy=1.9,
    )
    assert loss is not None
    assert loss.requires_grad
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    assert math.exp(float(tau_log.detach().item())) > 0.5


def test_compute_learnable_tau_loss_decreases_tau_when_entropy_is_high() -> None:
    tau_log = torch.nn.Parameter(torch.tensor(math.log(0.5), dtype=torch.float32))
    optimizer = torch.optim.SGD([tau_log], lr=0.5)
    loss = compute_learnable_tau_loss(
        tau_log,
        measured_entropy=2.2,
        target_entropy=1.9,
    )
    assert loss is not None
    assert loss.requires_grad
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    assert math.exp(float(tau_log.detach().item())) < 0.5


def test_build_padded_action_logprobs_respects_response_mask_positions() -> None:
    response_masks = torch.tensor(
        [[False, True, True, False], [True, True, False, False]],
        dtype=torch.bool,
    )
    padded = build_padded_action_logprobs(
        [[-0.1, -0.2], [-0.3, -0.4]],
        response_masks,
    )
    expected = torch.tensor(
        [[0.0, -0.1, -0.2, 0.0], [-0.3, -0.4, 0.0, 0.0]],
        dtype=torch.float32,
    )
    assert torch.allclose(padded.cpu(), expected, atol=1e-6, rtol=1e-6)


def test_gather_selected_logps_chunked_matches_naive_log_softmax() -> None:
    torch.manual_seed(0)
    logits = torch.randn(2, 5, 7, dtype=torch.float32)
    labels = torch.tensor(
        [[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]],
        dtype=torch.long,
    )
    response_masks = torch.tensor(
        [[True, True, False, True], [False, True, True, True]],
        dtype=torch.bool,
    )

    actual = gather_selected_logps_chunked(
        logits,
        labels,
        response_masks,
        token_chunk_size=2,
    )

    shifted_labels = labels[:, 1:].clone()
    shifted_labels[~response_masks] = 0
    all_logp = logits[:, :-1, :].log_softmax(-1)
    expected = torch.gather(
        all_logp,
        dim=2,
        index=shifted_labels.unsqueeze(2),
    ).squeeze(2)
    expected = torch.where(response_masks, expected, torch.zeros_like(expected))

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_choose_prefix_chunk_size_for_token_budget_respects_budget() -> None:
    assert (
        choose_prefix_chunk_size_for_token_budget(
            [3515, 3515, 3515, 3515],
            max_rows=4,
            token_budget=8192,
        )
        == 2
    )


def test_choose_prefix_chunk_size_for_token_budget_allows_full_chunk_when_short() -> None:
    assert (
        choose_prefix_chunk_size_for_token_budget(
            [1200, 1300, 1100, 1250],
            max_rows=4,
            token_budget=8192,
        )
        == 4
    )


def test_cap_last_valid_token_pos_for_zero_advantage_keeps_short_prompt_prefix() -> None:
    assert cap_last_valid_token_pos_for_zero_advantage(
        prompt_len=128,
        last_valid_token_pos=1024,
        response_token_budget=8,
    ) == 136


def test_cap_last_valid_token_pos_for_zero_advantage_never_truncates_below_one_response() -> None:
    assert cap_last_valid_token_pos_for_zero_advantage(
        prompt_len=128,
        last_valid_token_pos=129,
        response_token_budget=0,
    ) == 129


def test_listwise_sequence_coefficients_match_direct_gradient_without_clip() -> None:
    policy_seq_logps = torch.tensor(
        [[0.4, -0.2, 0.1], [0.3, 0.7, -0.5]],
        dtype=torch.float32,
        requires_grad=True,
    )
    weights = torch.tensor(
        [[0.7, 0.2, 0.1], [0.2, 0.5, 0.3]],
        dtype=torch.float32,
    )
    active = torch.tensor([True, False], dtype=torch.bool)

    log_probs = torch.log_softmax(policy_seq_logps, dim=1)
    direct_loss = -(weights[0] * log_probs[0]).sum()
    direct_grad = torch.autograd.grad(direct_loss, policy_seq_logps)[0]

    coeffs = compute_listwise_sequence_coefficients(
        policy_seq_logps_grouped=policy_seq_logps.detach(),
        weights_grouped=weights,
        active_group_mask=active,
    )
    surrogate_grad = torch.autograd.grad(
        (coeffs.detach() * policy_seq_logps).sum(),
        policy_seq_logps,
    )[0]

    assert torch.allclose(surrogate_grad, direct_grad, atol=1e-6, rtol=1e-6)


def test_listwise_sequence_coefficients_match_direct_gradient_with_clip() -> None:
    policy_seq_logps = torch.tensor(
        [[0.2, -0.1], [0.4, 0.0]],
        dtype=torch.float32,
        requires_grad=True,
    )
    behavior_seq_logps = torch.tensor(
        [[0.0, -0.4], [0.1, 0.2]],
        dtype=torch.float32,
    )
    weights = torch.tensor(
        [[0.8, 0.2], [0.3, 0.7]],
        dtype=torch.float32,
    )
    active = torch.tensor([True, True], dtype=torch.bool)
    clip_low = 0.2
    clip_high = 0.2
    clip_coef = 1.0
    baseline_value = 0.5

    log_probs = torch.log_softmax(policy_seq_logps, dim=1)
    per_group_policy_loss = -(weights * log_probs).sum(dim=1)
    policy_loss = per_group_policy_loss.mean()
    clip_adv = weights - baseline_value
    seq_ratio = torch.exp((policy_seq_logps - behavior_seq_logps).clamp(-40.0, 40.0))
    seq_ratio_clipped = torch.clamp(seq_ratio, 1.0 - clip_low, 1.0 + clip_high)
    clip_obj = torch.min(seq_ratio * clip_adv, seq_ratio_clipped * clip_adv)
    direct_loss = policy_loss + (-(clip_obj.sum(dim=1)).mean() * clip_coef)
    direct_grad = torch.autograd.grad(direct_loss, policy_seq_logps)[0]

    coeffs = compute_listwise_sequence_coefficients(
        policy_seq_logps_grouped=policy_seq_logps.detach(),
        weights_grouped=weights,
        active_group_mask=active,
        behavior_seq_logps_grouped=behavior_seq_logps,
        clip_low=clip_low,
        clip_high=clip_high,
        clip_coef=clip_coef,
        baseline_value=baseline_value,
    )
    surrogate_grad = torch.autograd.grad(
        (coeffs.detach() * policy_seq_logps).sum(),
        policy_seq_logps,
    )[0]

    assert torch.allclose(surrogate_grad, direct_grad, atol=1e-6, rtol=1e-6)


def test_compute_listwise_clip_advantages_preserves_reward_mass() -> None:
    weights = torch.tensor(
        [[0.5, 0.5, 0.0, 0.0], [0.7, 0.2, 0.1, 0.0]],
        dtype=torch.float32,
    )
    valid = torch.ones_like(weights, dtype=torch.bool)
    reward_mass = torch.tensor([[2.0], [1.0]], dtype=torch.float32)

    clip_adv = compute_listwise_clip_advantages(
        weights_grouped=weights,
        valid_row_mask_grouped=valid,
        reward_mass_grouped=reward_mass,
    )

    expected = torch.tensor(
        [[0.5, 0.5, -0.5, -0.5], [0.45, -0.05, -0.15, -0.25]],
        dtype=torch.float32,
    )
    assert torch.allclose(clip_adv, expected, atol=1e-6, rtol=1e-6)


def test_listwise_sequence_coefficients_match_direct_gradient_with_masked_rows() -> None:
    policy_seq_logps = torch.tensor(
        [[0.3, -0.4, 0.1]],
        dtype=torch.float32,
        requires_grad=True,
    )
    valid_rows = torch.tensor([[True, False, True]], dtype=torch.bool)
    target_weights = torch.tensor(
        [[0.875, 0.0, 0.125]],
        dtype=torch.float32,
    )
    active = torch.tensor([True], dtype=torch.bool)

    valid_log_probs = torch.log_softmax(policy_seq_logps[:, [0, 2]], dim=1)
    direct_loss = -(
        torch.tensor([0.875, 0.125], dtype=torch.float32) * valid_log_probs.squeeze(0)
    ).sum()
    direct_grad = torch.autograd.grad(direct_loss, policy_seq_logps)[0]

    coeffs = compute_listwise_sequence_coefficients(
        policy_seq_logps_grouped=policy_seq_logps.detach(),
        weights_grouped=target_weights,
        active_group_mask=active,
        valid_row_mask_grouped=valid_rows,
    )
    surrogate_grad = torch.autograd.grad(
        (coeffs.detach() * policy_seq_logps).sum(),
        policy_seq_logps,
    )[0]

    assert torch.allclose(surrogate_grad, direct_grad, atol=1e-6, rtol=1e-6)
    assert surrogate_grad[0, 1].item() == pytest.approx(0.0, abs=1e-8)


def test_listwise_sequence_coefficients_respect_global_active_group_normalizer() -> None:
    policy_seq_logps = torch.tensor(
        [[0.4, -0.2, 0.1], [0.3, 0.7, -0.5]],
        dtype=torch.float32,
        requires_grad=True,
    )
    weights = torch.tensor(
        [[0.7, 0.2, 0.1], [0.2, 0.5, 0.3]],
        dtype=torch.float32,
    )
    active = torch.tensor([True, False], dtype=torch.bool)

    log_probs = torch.log_softmax(policy_seq_logps, dim=1)
    direct_loss = -((weights[0] * log_probs[0]).sum() / 2.0)
    direct_grad = torch.autograd.grad(direct_loss, policy_seq_logps)[0]

    coeffs = compute_listwise_sequence_coefficients(
        policy_seq_logps_grouped=policy_seq_logps.detach(),
        weights_grouped=weights,
        active_group_mask=active,
        normalizer_active_group_count=2,
    )
    surrogate_grad = torch.autograd.grad(
        (coeffs.detach() * policy_seq_logps).sum(),
        policy_seq_logps,
    )[0]

    assert torch.allclose(surrogate_grad, direct_grad, atol=1e-6, rtol=1e-6)


def test_masked_group_log_softmax_ignores_invalid_rows() -> None:
    values = torch.tensor([[0.3, -0.4, 0.1]], dtype=torch.float32)
    valid_rows = torch.tensor([[True, False, True]], dtype=torch.bool)
    actual = masked_group_log_softmax(values, valid_rows)
    expected = torch.tensor(
        [[0.3, 0.1]],
        dtype=torch.float32,
    ).log_softmax(dim=1)
    assert torch.allclose(actual[:, [0, 2]], expected, atol=1e-6, rtol=1e-6)
    assert actual[0, 1].item() == pytest.approx(0.0, abs=1e-8)


def test_compute_token_level_clip_loss_matches_direct_token_ppo_objective() -> None:
    new_logps = torch.tensor(
        [[0.2, -0.1, 0.0], [-0.2, 0.0, -0.2]],
        dtype=torch.float32,
    )
    behavior_logps = torch.tensor(
        [[0.0, -0.4, 0.0], [0.1, 0.2, -0.5]],
        dtype=torch.float32,
    )
    response_masks = torch.tensor(
        [[True, True, False], [True, True, True]],
        dtype=torch.bool,
    )
    row_advantages = torch.tensor([0.3, -0.2], dtype=torch.float32)
    clip_low = 0.2
    clip_high = 0.2

    per_row_loss, ratio, is_low_clipped, is_high_clipped = compute_token_level_clip_loss(
        new_logps=new_logps,
        behavior_logps=behavior_logps,
        response_masks=response_masks,
        row_advantages=row_advantages,
        clip_low=clip_low,
        clip_high=clip_high,
        constant_normalizer=None,
    )

    log_ratio = (new_logps - behavior_logps).clamp(-40.0, 40.0)
    expected_ratio = torch.exp(log_ratio)
    clipped_ratio = torch.clamp(expected_ratio, 1.0 - clip_low, 1.0 + clip_high)
    clip_obj = torch.min(
        expected_ratio * row_advantages.unsqueeze(1),
        clipped_ratio * row_advantages.unsqueeze(1),
    )
    per_token_loss = -clip_obj * response_masks.to(torch.float32)
    expected_per_row_loss = per_token_loss.sum(dim=1) / response_masks.to(
        torch.float32
    ).sum(dim=1).clamp(min=1.0)

    assert torch.allclose(ratio, expected_ratio, atol=1e-6, rtol=1e-6)
    assert torch.allclose(
        per_row_loss,
        expected_per_row_loss,
        atol=1e-6,
        rtol=1e-6,
    )
    assert is_low_clipped.tolist() == [
        [False, False, False],
        [True, False, False],
    ]
    assert is_high_clipped.tolist() == [
        [True, True, False],
        [False, False, False],
    ]


def test_compute_token_level_clip_loss_uses_constant_normalizer_for_drgrpo() -> None:
    new_logps = torch.tensor([[0.2, -0.1, 0.3]], dtype=torch.float32)
    behavior_logps = torch.tensor([[0.0, -0.4, 0.0]], dtype=torch.float32)
    response_masks = torch.tensor([[True, True, True]], dtype=torch.bool)
    row_advantages = torch.tensor([0.25], dtype=torch.float32)

    per_row_loss, _, _, _ = compute_token_level_clip_loss(
        new_logps=new_logps,
        behavior_logps=behavior_logps,
        response_masks=response_masks,
        row_advantages=row_advantages,
        clip_low=0.2,
        clip_high=0.2,
        constant_normalizer=8.0,
    )

    ratio = torch.exp((new_logps - behavior_logps).clamp(-40.0, 40.0))
    clipped_ratio = torch.clamp(ratio, 0.8, 1.2)
    expected = -torch.min(
        ratio * row_advantages.unsqueeze(1),
        clipped_ratio * row_advantages.unsqueeze(1),
    ).sum(dim=1) / 8.0
    assert torch.allclose(per_row_loss, expected, atol=1e-6, rtol=1e-6)


def test_listwise_sequence_backward_path_keeps_scoring_safeguards() -> None:
    method_source = _zero_math_learner_method_source(
        "_backward_listwise_sequence_coefficients"
    )
    assert "_sanitize_scoring_token_ids(" in method_source
    assert "_mask_invalid_scoring_logit_columns(" in method_source


def test_listwise_token_clip_backward_path_keeps_scoring_safeguards() -> None:
    method_source = _zero_math_learner_method_source(
        "_backward_listwise_token_clip_loss"
    )
    assert "_sanitize_scoring_token_ids(" in method_source
    assert "_mask_invalid_scoring_logit_columns(" in method_source


def test_listwise_token_clip_backward_path_keeps_zero_active_ranks_in_sync() -> None:
    method_source = _zero_math_learner_method_source(
        "_backward_listwise_token_clip_loss"
    )
    assert "or local_active_row_count <= 0" not in method_source
    assert "Do not short-circuit ranks with zero local active rows" in method_source


def test_listwise_learning_step_fuses_token_clip_into_shared_backward_pass() -> None:
    learner_source = Path(
        "understand-r1-zero/train_zero_math.py"
    ).read_text(encoding="utf-8")
    assert learner_source.count("_backward_listwise_token_clip_loss(") == 1
    sequence_method = _zero_math_learner_method_source(
        "_backward_listwise_sequence_coefficients"
    )
    assert "one shared distributed backward pattern" in sequence_method


def test_listwise_learning_step_keeps_partially_valid_groups_active() -> None:
    method_source = _zero_math_learner_method_source("_listwise_learning_step")
    assert "grouped_loss_masks.any(dim=1)" in method_source


def test_listwise_learning_step_uses_active_entropy_for_tau_control() -> None:
    method_source = _zero_math_learner_method_source("_listwise_learning_step")
    assert "collect_weight_entropy_stats(weights_grouped[active_group_mask])" in method_source
    assert "listwise_weight_entropy_all" in method_source
    assert "_distributed_weighted_mean_scalar(" in method_source


def test_listwise_learning_step_gates_simple_tau_controller_explicitly() -> None:
    method_source = _zero_math_learner_method_source("_listwise_learning_step")
    assert "elif bool(args.maxent_tau_controller_enabled):" in method_source
    assert "args.maxent_tau = maybe_update_listwise_tau(" in method_source


def test_listwise_learning_step_uses_global_active_group_normalizer() -> None:
    method_source = _zero_math_learner_method_source("_listwise_learning_step")
    assert "global_active_group_count" in method_source
    assert "normalizer_active_group_count=global_active_group_count" in method_source
    assert "active_row_count_normalizer=(" in method_source
    assert "token_clip_row_count_normalizer = drgrpo_active_row_count" in method_source
    assert "token_clip_row_count_normalizer = global_active_row_count" in method_source


def test_listwise_learning_step_supports_token_surrogate_primary_mode() -> None:
    method_source = _zero_math_learner_method_source("_listwise_learning_step")
    assert "token_surrogate_primary = bool(args.maxent_token_surrogate_primary)" in method_source
    assert "infos[\"listwise_ce_loss\"] = policy_loss.detach()" in method_source
    assert "loss = clip_coef * clip_loss" in method_source
    assert "seq_coeffs_grouped = torch.zeros_like(" in method_source


def test_listwise_learning_step_supports_drgrpo_token_primary_mode() -> None:
    method_source = _zero_math_learner_method_source("_listwise_learning_step")
    assert "drgrpo_token_primary = bool(args.maxent_drgrpo_token_primary)" in method_source
    assert 'infos["drgrpo_pg_loss"] = drgrpo_pg_loss.detach()' in method_source
    assert "drgrpo_row_adv_flat = mb_advantage.reshape(-1).to(" in method_source
    assert "token_clip_adv_for_backward = drgrpo_row_adv_flat" in method_source
    assert "token_clip_enabled = (" in method_source
    assert "_backward_listwise_sequence_coefficients(" in method_source
    assert "continuing with the Dr.GRPO token update" in method_source
    assert '"applied_drgrpo_only"' in method_source


def test_listwise_learning_step_can_preserve_reward_mass_in_clip_advantages() -> None:
    method_source = _zero_math_learner_method_source("_listwise_learning_step")
    assert "if bool(args.maxent_clip_preserve_reward_mass):" in method_source
    assert "reward_mass_grouped = torch.where(" in method_source
    assert "clip_adv = compute_listwise_clip_advantages(" in method_source


def test_sync_maxent_tau_from_state_hard_locks_fixed_tau_when_controllers_off() -> None:
    method_source = _zero_math_learner_method_source("_sync_maxent_tau_from_state")
    assert "if self._fixed_listwise_tau is not None:" in method_source
    assert "self.args.maxent_tau = float(current_tau)" in method_source


def test_enforce_fixed_listwise_hparams_restores_core_explorer_config() -> None:
    method_source = _zero_math_learner_method_source("_enforce_fixed_listwise_hparams")
    learner_source = Path(
        "understand-r1-zero/train_zero_math.py"
    ).read_text(encoding="utf-8")
    assert 'for name, value in self._fixed_listwise_config.items():' in method_source
    assert 'setattr(self.args, name, value)' in method_source
    assert 'self.args.beta = float(self._fixed_listwise_beta)' in method_source
    assert '"maxent_clip_range": (' in learner_source
    assert '"maxent_clip_adv_baseline": (' in learner_source
    assert '"maxent_logprob_chunk_size": int(args.maxent_logprob_chunk_size)' in learner_source
    assert '"maxent_backward_chunk_size": int(args.maxent_backward_chunk_size)' in learner_source
    assert '"maxent_backward_token_budget": int(' in learner_source
    assert '"maxent_drgrpo_token_primary": bool(' in learner_source
    assert '"maxent_sequence_aux_coef": float(args.maxent_sequence_aux_coef)' in learner_source


def test_effective_backward_token_budget_auto_throttles_long_fixed_chunks() -> None:
    method_source = _zero_math_learner_method_source("_effective_backward_token_budget")
    assert "safety_budget = 4096" in method_source
    assert "Listwise backward auto-enabled a synchronized token budget" in method_source
    assert "max_synchronized_tokens * safe_chunk_size > safety_budget" in method_source


def test_listwise_learning_step_reenforces_fixed_hparams_before_logging() -> None:
    method_source = _zero_math_learner_method_source("_listwise_learning_step")
    assert method_source.count("self._enforce_fixed_listwise_hparams()") >= 2
    assert 'infos["tau"] = torch.tensor(' in method_source
    assert 'infos["beta"] = torch.tensor(' in method_source
    assert '"listwise runtime config: tau=%s beta=%s q_temperature=%s "' in method_source
    assert 'infos["listwise_q_temperature"] = torch.tensor(' in method_source
    assert 'infos["listwise_q_epsilon"] = torch.tensor(' in method_source
    assert 'infos["listwise_length_normalize_ref"] = torch.tensor(' in method_source
    assert 'infos["listwise_length_normalize_policy"] = torch.tensor(' in method_source
    assert 'infos["listwise_skip_zero_variance_groups"] = torch.tensor(' in method_source
    assert 'infos["listwise_clip_range"] = torch.tensor(' in method_source
    assert 'infos["listwise_drgrpo_token_primary"] = torch.tensor(' in method_source
    assert 'infos["listwise_sequence_aux_coef"] = torch.tensor(' in method_source
    assert 'infos["listwise_logprob_chunk_size"] = torch.tensor(' in method_source
    assert 'infos["listwise_backward_chunk_size"] = torch.tensor(' in method_source
    assert 'infos["listwise_backward_token_budget"] = torch.tensor(' in method_source
    assert 'infos["listwise_reference_logprobs_from_model"] = torch.tensor(' in method_source
    assert 'infos["listwise_reference_logprobs_from_behavior"] = (' in method_source
    assert 'infos["listwise_branch_grad_diagnostics"] = torch.tensor(' in method_source
    assert 'infos["listwise_branch_grad_diagnostics_interval"] = torch.tensor(' in method_source
    assert 'infos["listwise_branch_grad_diagnostics_max_steps"] = torch.tensor(' in method_source


def test_listwise_learning_step_stabilizes_optional_logging_keys() -> None:
    method_source = _zero_math_learner_method_source("_listwise_learning_step")
    assert '"listwise_clip_reward_mass_mean": 0.0' in method_source
    assert '"listwise_grad_probe_update_index": 0.0' in method_source
    assert '"listwise_grad_ratio_scaled": 0.0' in method_source
    assert 'return {key: infos[key] for key in sorted(infos)}' in method_source


def test_zero_math_learn_sorts_train_info_before_distributed_logging() -> None:
    method_source = _zero_math_learner_method_source("learn")
    assert 'train_info = {key: train_info[key] for key in sorted(train_info)}' in method_source


def test_listwise_gradient_probe_helpers_exist() -> None:
    learner_source = Path("understand-r1-zero/train_zero_math.py").read_text(
        encoding="utf-8"
    )
    assert "def _should_run_listwise_branch_grad_diagnostics(" in learner_source
    assert "def _measure_listwise_sequence_gradient_squared_norm(" in learner_source
    assert "def _probe_listwise_branch_gradient_metrics(" in learner_source
    assert "listwise grad probe update %s:" in learner_source
    assert '"listwise_grad_ratio_scaled": grad_ratio_scaled.detach()' in learner_source
    assert '"listwise_grad_cosine": grad_cosine.detach()' in learner_source
    assert '"listwise_grad_probe_enabled": torch.tensor(' in learner_source


def test_zero_math_learner_defines_weighted_scalar_reduction() -> None:
    method_source = _zero_math_learner_method_source(
        "_distributed_weighted_mean_scalar"
    )
    assert "numerator" in method_source
    assert "denominator" in method_source
    assert "dist.all_reduce(numerator" in method_source


def test_oat_zero_explorer_launcher_defaults_ignore_no_eos_false() -> None:
    script_source = (
        Path(__file__).resolve().parents[2]
        / "ops"
        / "run_oat_zero_explorer_1p5b_upstream.sh"
    ).read_text(encoding="utf-8")
    assert 'export OAT_ZERO_IGNORE_NO_EOS="${OAT_ZERO_IGNORE_NO_EOS:-0}"' in script_source


def test_oat_zero_explorer_launcher_reverts_to_fixed_tau_by_default() -> None:
    script_source = (
        Path(__file__).resolve().parents[2]
        / "ops"
        / "run_oat_zero_explorer_1p5b_upstream.sh"
    ).read_text(encoding="utf-8")
    assert (
        'export OAT_ZERO_MAXENT_TAU_LEARNABLE="${OAT_ZERO_MAXENT_TAU_LEARNABLE:-0}"'
        in script_source
    )
    assert (
        'export OAT_ZERO_MAXENT_TAU_CONTROLLER_ENABLED="${OAT_ZERO_MAXENT_TAU_CONTROLLER_ENABLED:-0}"'
        in script_source
    )
    assert (
        'export OAT_ZERO_MAXENT_Q_TEMPERATURE="${OAT_ZERO_MAXENT_Q_TEMPERATURE:-2.0}"'
        in script_source
    )


def test_oat_zero_explorer_launcher_skips_zero_variance_groups_by_default() -> None:
    script_source = (
        Path(__file__).resolve().parents[2]
        / "ops"
        / "run_oat_zero_explorer_1p5b_upstream.sh"
    ).read_text(encoding="utf-8")
    assert (
        'export OAT_ZERO_MAXENT_LISTWISE_SKIP_ZERO_VARIANCE_GROUPS="${OAT_ZERO_MAXENT_LISTWISE_SKIP_ZERO_VARIANCE_GROUPS:-1}"'
        in script_source
    )


def test_oat_zero_explorer_launcher_length_normalizes_sequence_aux_by_default() -> None:
    script_source = (
        Path(__file__).resolve().parents[2]
        / "ops"
        / "run_oat_zero_explorer_1p5b_upstream.sh"
    ).read_text(encoding="utf-8")
    assert (
        'export OAT_ZERO_MAXENT_LENGTH_NORMALIZE_REF="${OAT_ZERO_MAXENT_LENGTH_NORMALIZE_REF:-1}"'
        in script_source
    )
    assert (
        'export OAT_ZERO_MAXENT_LENGTH_NORMALIZE_POLICY="${OAT_ZERO_MAXENT_LENGTH_NORMALIZE_POLICY:-1}"'
        in script_source
    )


def test_oat_zero_explorer_launcher_defaults_to_drgrpo_token_primary_combo() -> None:
    script_source = (
        Path(__file__).resolve().parents[2]
        / "ops"
        / "run_oat_zero_explorer_1p5b_upstream.sh"
    ).read_text(encoding="utf-8")
    assert (
        'export OAT_ZERO_MAXENT_CLIP_MODE="${OAT_ZERO_MAXENT_CLIP_MODE:-sequence}"'
        in script_source
    )
    assert (
        'export OAT_ZERO_MAXENT_TOKEN_SURROGATE_PRIMARY="${OAT_ZERO_MAXENT_TOKEN_SURROGATE_PRIMARY:-0}"'
        in script_source
    )
    assert (
        'export OAT_ZERO_MAXENT_DRGRPO_TOKEN_PRIMARY="${OAT_ZERO_MAXENT_DRGRPO_TOKEN_PRIMARY:-1}"'
        in script_source
    )
    assert (
        'export OAT_ZERO_MAXENT_SEQUENCE_AUX_COEF="${OAT_ZERO_MAXENT_SEQUENCE_AUX_COEF:-0.01}"'
        in script_source
    )
    assert (
        'export OAT_ZERO_MAXENT_BRANCH_GRAD_DIAGNOSTICS="${OAT_ZERO_MAXENT_BRANCH_GRAD_DIAGNOSTICS:-0}"'
        in script_source
    )
    assert (
        'export OAT_ZERO_MAXENT_BRANCH_GRAD_DIAGNOSTICS_INTERVAL="${OAT_ZERO_MAXENT_BRANCH_GRAD_DIAGNOSTICS_INTERVAL:-1}"'
        in script_source
    )
    assert (
        'export OAT_ZERO_MAXENT_BRANCH_GRAD_DIAGNOSTICS_MAX_STEPS="${OAT_ZERO_MAXENT_BRANCH_GRAD_DIAGNOSTICS_MAX_STEPS:-16}"'
        in script_source
    )


def test_oat_zero_exact_launcher_threads_ignore_no_eos_flag() -> None:
    script_source = (
        Path(__file__).resolve().parents[2]
        / "ops"
        / "run_oat_zero_exact_1p5b_upstream.sh"
    ).read_text(encoding="utf-8")
    assert 'IGNORE_NO_EOS="${OAT_ZERO_IGNORE_NO_EOS:-0}"' in script_source
    assert 'ignore_no_eos_flag=(--no-ignore-no-eos)' in script_source
    assert '"${ignore_no_eos_flag[@]}"' in script_source
