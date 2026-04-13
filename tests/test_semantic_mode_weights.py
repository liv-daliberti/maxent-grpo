import torch

from oat_drgrpo.listwise import (
    compute_listwise_weights_from_utilities,
    compute_semantic_cluster_weights_from_utilities,
)


def _tensor(values):
    return torch.tensor(values, dtype=torch.float32)


def _base_weights(
    utility_grouped: torch.Tensor,
    ref_seq_logps_grouped: torch.Tensor,
    *,
    tau: float = 0.2,
    candidate_kl_coef: float = 0.0,
    valid_row_mask_grouped: torch.Tensor | None = None,
) -> torch.Tensor:
    return compute_listwise_weights_from_utilities(
        utility_grouped=utility_grouped,
        ref_seq_logps_grouped=ref_seq_logps_grouped,
        tau=tau,
        candidate_kl_coef=candidate_kl_coef,
        valid_row_mask_grouped=valid_row_mask_grouped,
    )


def _run_semantic(
    utility_grouped: torch.Tensor,
    cluster_ids_grouped: torch.Tensor,
    *,
    ref_seq_logps_grouped: torch.Tensor | None = None,
    candidate_lengths_grouped: torch.Tensor | None = None,
    candidate_formatted_grouped: torch.Tensor | None = None,
    tau: float = 0.2,
    mode_tau: float = 0.05,
    mode_gap: float = 0.10,
    mode_top_k: int = 3,
    budget_grouped: torch.Tensor | None = None,
    budget_max: float = 0.10,
    intra_tau: float = 0.01,
    candidate_kl_coef: float = 0.0,
    prompt_select_min_alpha_frac: float = 0.0,
    positive_only: bool = False,
    max_expected_len_delta: float = float("inf"),
    max_expected_format_drop: float = 0.0,
    valid_row_mask_grouped: torch.Tensor | None = None,
) -> tuple[torch.Tensor, object]:
    if ref_seq_logps_grouped is None:
        ref_seq_logps_grouped = torch.zeros_like(utility_grouped)
    if candidate_lengths_grouped is None:
        candidate_lengths_grouped = torch.ones_like(utility_grouped)
    if candidate_formatted_grouped is None:
        candidate_formatted_grouped = torch.ones_like(utility_grouped)
    if valid_row_mask_grouped is None:
        valid_row_mask_grouped = torch.ones_like(utility_grouped, dtype=torch.bool)
    return compute_semantic_cluster_weights_from_utilities(
        utility_grouped=utility_grouped,
        ref_seq_logps_grouped=ref_seq_logps_grouped,
        cluster_ids_grouped=cluster_ids_grouped,
        candidate_lengths_grouped=candidate_lengths_grouped,
        candidate_formatted_grouped=candidate_formatted_grouped,
        tau=tau,
        mode_tau=mode_tau,
        mode_gap=mode_gap,
        mode_top_k=mode_top_k,
        budget_grouped=budget_grouped,
        budget_max=budget_max,
        intra_tau=intra_tau,
        candidate_kl_coef=candidate_kl_coef,
        prompt_select_min_alpha_frac=prompt_select_min_alpha_frac,
        positive_only=positive_only,
        max_expected_len_delta=max_expected_len_delta,
        max_expected_format_drop=max_expected_format_drop,
        valid_row_mask_grouped=valid_row_mask_grouped,
    )


def _cluster_mass(
    weights_grouped: torch.Tensor,
    cluster_ids_grouped: torch.Tensor,
    cluster_id: int,
) -> torch.Tensor:
    mask = cluster_ids_grouped[0] == cluster_id
    return weights_grouped[0, mask].sum()


def test_falls_back_to_baseline_when_fewer_than_two_eligible_modes():
    utility = _tensor([[1.0, 0.95, 0.20]])
    ref = torch.zeros_like(utility)
    cluster_ids = torch.tensor([[0, 1, 2]], dtype=torch.long)
    valid = torch.ones_like(utility, dtype=torch.bool)
    budget = _tensor([1.0])

    base = _base_weights(utility, ref, valid_row_mask_grouped=valid)
    weights, diag = _run_semantic(
        utility,
        cluster_ids,
        ref_seq_logps_grouped=ref,
        valid_row_mask_grouped=valid,
        mode_gap=0.01,
        budget_grouped=budget,
        budget_max=1.0,
    )

    torch.testing.assert_close(weights, base)
    assert float(diag.eligible_mode_count_grouped[0].item()) == 1.0
    assert not bool(diag.explore_applied_group_mask[0].item())
    assert float(diag.moved_mass_l1_grouped[0].item()) == 0.0


def test_falls_back_to_baseline_when_budget_max_is_zero():
    utility = _tensor([[1.0, 0.97, 0.96]])
    ref = torch.zeros_like(utility)
    cluster_ids = torch.tensor([[0, 1, 2]], dtype=torch.long)
    valid = torch.ones_like(utility, dtype=torch.bool)
    budget = _tensor([1.0])

    base = _base_weights(utility, ref, valid_row_mask_grouped=valid)
    weights, diag = _run_semantic(
        utility,
        cluster_ids,
        ref_seq_logps_grouped=ref,
        valid_row_mask_grouped=valid,
        mode_gap=0.10,
        budget_grouped=budget,
        budget_max=0.0,
    )

    torch.testing.assert_close(weights, base)
    assert float(diag.explore_budget_grouped[0].item()) == 0.0
    assert not bool(diag.explore_applied_group_mask[0].item())


def test_duplicate_count_does_not_change_total_mode_mass():
    utility_single = _tensor([[0.90, 0.70]])
    clusters_single = torch.tensor([[0, 1]], dtype=torch.long)
    utility_dupes = _tensor([[0.90, 0.85, 0.70]])
    clusters_dupes = torch.tensor([[0, 0, 1]], dtype=torch.long)
    budget = _tensor([1.0])

    weights_single, _ = _run_semantic(
        utility_single,
        clusters_single,
        budget_grouped=budget,
        budget_max=1.0,
        mode_gap=1.0,
    )
    weights_dupes, _ = _run_semantic(
        utility_dupes,
        clusters_dupes,
        budget_grouped=budget,
        budget_max=1.0,
        mode_gap=1.0,
    )

    torch.testing.assert_close(
        _cluster_mass(weights_single, clusters_single, 0),
        _cluster_mass(weights_dupes, clusters_dupes, 0),
        atol=1e-6,
        rtol=0.0,
    )


def test_unlabeled_rows_preserve_exact_baseline_mass():
    utility = _tensor([[1.0, 0.97, 0.96, 0.20]])
    ref = torch.zeros_like(utility)
    cluster_ids = torch.tensor([[0, 1, 2, -1]], dtype=torch.long)
    valid = torch.ones_like(utility, dtype=torch.bool)
    budget = _tensor([1.0])

    base = _base_weights(utility, ref, valid_row_mask_grouped=valid)
    weights, _ = _run_semantic(
        utility,
        cluster_ids,
        ref_seq_logps_grouped=ref,
        valid_row_mask_grouped=valid,
        budget_grouped=budget,
        budget_max=1.0,
        mode_gap=0.10,
    )

    torch.testing.assert_close(weights[0, 3], base[0, 3], atol=1e-7, rtol=0.0)


def test_only_labeled_semantic_mass_is_remixed():
    utility = _tensor([[1.0, 0.97, 0.96, 0.20]])
    ref = torch.zeros_like(utility)
    cluster_ids = torch.tensor([[0, 1, 2, -1]], dtype=torch.long)
    valid = torch.ones_like(utility, dtype=torch.bool)
    budget = _tensor([1.0])

    base = _base_weights(utility, ref, valid_row_mask_grouped=valid)
    weights, _ = _run_semantic(
        utility,
        cluster_ids,
        ref_seq_logps_grouped=ref,
        valid_row_mask_grouped=valid,
        budget_grouped=budget,
        budget_max=1.0,
        mode_gap=0.10,
    )

    semantic_mask = cluster_ids[0] >= 0
    torch.testing.assert_close(
        weights[0, semantic_mask].sum(),
        base[0, semantic_mask].sum(),
        atol=1e-6,
        rtol=0.0,
    )
    assert float(torch.abs(weights[0, semantic_mask] - base[0, semantic_mask]).sum().item()) > 0.0


def test_gap_excludes_low_utility_modes_from_semantic_exploration():
    utility = _tensor([[1.0, 0.95, 0.60]])
    cluster_ids = torch.tensor([[0, 1, 2]], dtype=torch.long)
    budget = _tensor([1.0])

    weights, _ = _run_semantic(
        utility,
        cluster_ids,
        budget_grouped=budget,
        budget_max=1.0,
        mode_gap=0.07,
    )

    assert float(weights[0, 2].item()) == 0.0


def test_top_k_cap_limits_eligible_modes():
    utility = _tensor([[1.0, 0.98, 0.97, 0.96]])
    cluster_ids = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    budget = _tensor([1.0])

    weights, diag = _run_semantic(
        utility,
        cluster_ids,
        budget_grouped=budget,
        budget_max=1.0,
        mode_gap=0.10,
        mode_top_k=2,
    )

    assert float(diag.eligible_mode_count_grouped[0].item()) == 2.0
    assert float(weights[0, 2].item()) == 0.0
    assert float(weights[0, 3].item()) == 0.0


def test_within_mode_exact_ties_split_uniformly():
    utility = _tensor([[1.0, 1.0, 0.97]])
    cluster_ids = torch.tensor([[0, 0, 1]], dtype=torch.long)
    budget = _tensor([1.0])

    weights, _ = _run_semantic(
        utility,
        cluster_ids,
        budget_grouped=budget,
        budget_max=1.0,
        mode_gap=1.0,
    )

    torch.testing.assert_close(weights[0, 0], weights[0, 1], atol=1e-7, rtol=0.0)
    assert float(_cluster_mass(weights, cluster_ids, 0).item()) > 0.0


def test_within_mode_near_ties_use_smooth_positive_split():
    utility = _tensor([[1.0, 0.999, 0.97]])
    cluster_ids = torch.tensor([[0, 0, 1]], dtype=torch.long)
    budget = _tensor([1.0])

    weights, _ = _run_semantic(
        utility,
        cluster_ids,
        budget_grouped=budget,
        budget_max=1.0,
        mode_gap=1.0,
        intra_tau=0.01,
    )

    assert float(weights[0, 0].item()) > float(weights[0, 1].item()) > 0.0


def test_final_weights_are_normalized_and_nonnegative():
    utility = _tensor([[1.0, 0.97, 0.96, 0.20]])
    cluster_ids = torch.tensor([[0, 1, 2, -1]], dtype=torch.long)
    budget = _tensor([0.07])

    weights, _ = _run_semantic(
        utility,
        cluster_ids,
        budget_grouped=budget,
        budget_max=0.10,
        mode_gap=0.10,
    )

    assert bool(torch.all(weights >= 0.0).item())
    torch.testing.assert_close(weights.sum(dim=1), torch.ones(1), atol=1e-6, rtol=0.0)


def test_default_knobs_move_a_small_but_nonzero_mass():
    utility = _tensor([[1.0, 0.97, 0.96, 0.20]])
    cluster_ids = torch.tensor([[0, 1, 2, -1]], dtype=torch.long)
    budget = _tensor([0.05])

    weights, diag = _run_semantic(
        utility,
        cluster_ids,
        tau=0.2,
        mode_tau=0.05,
        mode_gap=0.10,
        mode_top_k=3,
        budget_grouped=budget,
        budget_max=0.10,
        intra_tau=0.01,
    )

    moved = float(diag.moved_mass_l1_grouped[0].item())
    assert moved > 1e-5
    assert moved < 0.10
    torch.testing.assert_close(weights.sum(dim=1), torch.ones(1), atol=1e-6, rtol=0.0)


def test_falls_back_to_baseline_when_alpha_raw_is_below_selection_threshold():
    utility = _tensor([[1.0, 0.97, 0.96]])
    cluster_ids = torch.tensor([[0, 1, 2]], dtype=torch.long)
    budget = _tensor([0.04])

    base = _base_weights(utility, torch.zeros_like(utility))
    weights, diag = _run_semantic(
        utility,
        cluster_ids,
        budget_grouped=budget,
        budget_max=0.10,
        prompt_select_min_alpha_frac=0.5,
    )

    torch.testing.assert_close(weights, base)
    assert bool(diag.prompt_rejected_low_opp_group_mask[0].item())
    torch.testing.assert_close(diag.alpha_raw_grouped[0], torch.tensor(0.4))


def test_falls_back_to_baseline_when_best_mode_is_nonpositive():
    utility = _tensor([[-0.01, -0.02, -0.03]])
    cluster_ids = torch.tensor([[0, 1, 2]], dtype=torch.long)
    budget = _tensor([0.10])

    base = _base_weights(utility, torch.zeros_like(utility))
    weights, diag = _run_semantic(
        utility,
        cluster_ids,
        budget_grouped=budget,
        budget_max=0.10,
        prompt_select_min_alpha_frac=0.5,
        positive_only=True,
    )

    torch.testing.assert_close(weights, base)
    assert bool(diag.prompt_rejected_nonpositive_group_mask[0].item())


def test_falls_back_to_baseline_when_expected_length_guard_fails():
    utility = _tensor([[0.97, 0.98, 1.0]])
    cluster_ids = torch.tensor([[0, 1, 2]], dtype=torch.long)
    lengths = _tensor([[10.0, 40.0, 80.0]])
    budget = _tensor([0.10])

    base = _base_weights(utility, torch.zeros_like(utility))
    weights, diag = _run_semantic(
        utility,
        cluster_ids,
        candidate_lengths_grouped=lengths,
        budget_grouped=budget,
        budget_max=0.10,
        prompt_select_min_alpha_frac=0.5,
        max_expected_len_delta=5.0,
    )

    torch.testing.assert_close(weights, base)
    assert bool(diag.prompt_rejected_len_guard_group_mask[0].item())
    assert float(diag.expected_len_explore_target_grouped[0].item()) > (
        float(diag.expected_len_q_grouped[0].item()) + 5.0
    )


def test_falls_back_to_baseline_when_expected_format_guard_fails():
    utility = _tensor([[0.97, 0.98, 1.0]])
    cluster_ids = torch.tensor([[0, 1, 2]], dtype=torch.long)
    formatted = _tensor([[1.0, 1.0, 0.0]])
    budget = _tensor([0.10])

    base = _base_weights(utility, torch.zeros_like(utility))
    weights, diag = _run_semantic(
        utility,
        cluster_ids,
        candidate_formatted_grouped=formatted,
        budget_grouped=budget,
        budget_max=0.10,
        prompt_select_min_alpha_frac=0.5,
        max_expected_format_drop=0.0,
    )

    torch.testing.assert_close(weights, base)
    assert bool(diag.prompt_rejected_format_guard_group_mask[0].item())
    assert float(diag.expected_format_explore_target_grouped[0].item()) < float(
        diag.expected_format_q_grouped[0].item()
    )


def test_positive_only_filter_excludes_nonpositive_modes():
    utility = _tensor([[1.0, 0.96, -0.05]])
    cluster_ids = torch.tensor([[0, 1, 2]], dtype=torch.long)
    budget = _tensor([0.10])

    weights, diag = _run_semantic(
        utility,
        cluster_ids,
        budget_grouped=budget,
        budget_max=0.10,
        prompt_select_min_alpha_frac=0.5,
        positive_only=True,
    )

    assert float(diag.eligible_mode_count_grouped[0].item()) == 2.0
    assert float(weights[0, 2].item()) < float(
        _base_weights(utility, torch.zeros_like(utility))[0, 2].item()
    )
