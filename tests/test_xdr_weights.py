import math

import pytest
import torch

from oat_drgrpo.xdr import aggregation_group_diagnostics, compute_xdr_row_weights


def _weights(adv, counts, *, g=4, tau=0.25, t_max=256):
    return compute_xdr_row_weights(
        torch.tensor(adv, dtype=torch.float32),
        torch.tensor(counts, dtype=torch.float32),
        num_samples=g,
        tau=tau,
        t_max=t_max,
    )


def test_weights_sum_to_group_size_and_are_positive():
    adv = [0.75, -0.25, -0.25, -0.25, 0.5, 0.5, -0.5, -0.5]
    counts = [10, 20, 30, 40, 50, 60, 70, 80]
    w = _weights(adv, counts, g=4)
    assert w.shape == (8,)
    assert bool((w > 0).all())
    grouped = w.view(-1, 4).sum(dim=1)
    assert torch.allclose(grouped, torch.full_like(grouped, 4.0))


def test_zero_advantage_group_is_exactly_uniform():
    # Degenerate groups (all rewards equal) have zero advantages, hence zero
    # utilities and exactly uniform weights: the multiplier is exactly 1.0.
    w = _weights([0.0] * 8, [10, 20, 30, 40, 50, 60, 70, 80], g=8)
    assert torch.equal(w, torch.ones(8))


def test_higher_advantage_gets_higher_weight():
    w = _weights([0.75, -0.25, -0.25, -0.25], [32, 32, 32, 32], g=4)
    assert w[0] > w[1]
    assert torch.allclose(w[1:], w[1].expand(3))


def test_large_tau_approaches_uniform_and_small_tau_concentrates():
    adv = [0.75, -0.25, -0.25, -0.25]
    counts = [32, 32, 32, 32]
    w_large = _weights(adv, counts, g=4, tau=1e6)
    assert torch.allclose(w_large, torch.ones(4), atol=1e-4)
    w_small = _weights(adv, counts, g=4, tau=1e-4)
    assert w_small[0] > 3.999


def test_length_scales_utility_through_t_max():
    # Positive-advantage candidates with more active tokens have larger
    # utility A*T/T_max and therefore more weight; for negative advantages the
    # ordering flips.
    w_pos = _weights([0.5, 0.5, -0.5, -0.5], [200, 100, 100, 100], g=4)
    assert w_pos[0] > w_pos[1]
    w_neg = _weights([-0.5, -0.5, 0.5, 0.5], [200, 100, 100, 100], g=4)
    assert w_neg[0] < w_neg[1]


def test_weights_are_detached():
    adv = torch.tensor([0.5, -0.5, 0.25, -0.25], requires_grad=True)
    counts = torch.tensor([32.0, 32.0, 32.0, 32.0])
    w = compute_xdr_row_weights(
        adv, counts, num_samples=4, tau=0.25, t_max=256
    )
    assert not w.requires_grad


def test_column_vector_advantages_are_accepted():
    adv = torch.tensor([[0.5], [-0.5], [0.25], [-0.25]])
    counts = torch.tensor([32, 32, 32, 32])
    w = compute_xdr_row_weights(adv, counts, num_samples=4, tau=0.25, t_max=256)
    assert w.shape == (4,)


def test_ragged_batch_raises():
    with pytest.raises(ValueError):
        _weights([0.5, -0.5, 0.25], [32, 32, 32], g=4)


def test_invalid_tau_raises():
    with pytest.raises(ValueError):
        _weights([0.0] * 4, [32] * 4, g=4, tau=math.inf)
    with pytest.raises(ValueError):
        _weights([0.0] * 4, [32] * 4, g=4, tau=0.0)
    with pytest.raises(ValueError):
        _weights([0.0] * 4, [32] * 4, g=4, tau=-1.0)


def test_diagnostics_uniform_weights():
    rewards = torch.tensor([1.0, 0.0, 0.0, 0.0])
    diag = aggregation_group_diagnostics(torch.ones(4), rewards, num_samples=4)
    assert torch.allclose(diag["agg_eff_rollouts"], torch.tensor(4.0))
    assert torch.allclose(diag["agg_incorrect_mass"], torch.tensor(0.75))


def test_diagnostics_concentrated_weights():
    # All mass on the single correct candidate: one effective rollout and
    # zero incorrect mass (up to the clamp used inside the entropy).
    w = torch.tensor([4.0, 0.0, 0.0, 0.0])
    rewards = torch.tensor([1.0, 0.0, 0.0, 0.0])
    diag = aggregation_group_diagnostics(w, rewards, num_samples=4)
    assert float(diag["agg_eff_rollouts"]) == pytest.approx(1.0, abs=1e-4)
    assert float(diag["agg_incorrect_mass"]) == pytest.approx(0.0, abs=1e-6)


def test_diagnostics_masked_group_normalizes_over_valid_rows():
    # A prompt group with one loss-masked row (weight 0) is a distribution
    # over the three valid rows: uniform weights give 3 effective rollouts.
    w = torch.tensor([1.0, 1.0, 1.0, 0.0])
    rewards = torch.tensor([1.0, 0.0, 0.0, 0.0])
    diag = aggregation_group_diagnostics(w, rewards, num_samples=4)
    assert float(diag["agg_eff_rollouts"]) == pytest.approx(3.0, abs=1e-4)
    assert float(diag["agg_incorrect_mass"]) == pytest.approx(2.0 / 3.0, abs=1e-6)


def test_diagnostics_drop_all_masked_groups_from_means():
    # First group fully masked, second group uniform: means come from the
    # second group only.
    w = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    rewards = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0])
    diag = aggregation_group_diagnostics(w, rewards, num_samples=4)
    assert float(diag["agg_eff_rollouts"]) == pytest.approx(4.0, abs=1e-4)
    assert float(diag["agg_incorrect_mass"]) == pytest.approx(0.5, abs=1e-6)


def test_diagnostics_seed_style_prompt_scale_is_uniform_within_group():
    # SEED weights are s_x on every row of a prompt; the within-group
    # distribution is uniform regardless of s_x, so the diagnostics match the
    # Dr.GRPO baseline exactly.
    w = torch.tensor([0.5, 0.5, 0.5, 0.5])
    rewards = torch.tensor([1.0, 1.0, 0.0, 0.0])
    diag = aggregation_group_diagnostics(w, rewards, num_samples=4)
    assert float(diag["agg_eff_rollouts"]) == pytest.approx(4.0, abs=1e-4)
    assert float(diag["agg_incorrect_mass"]) == pytest.approx(0.5, abs=1e-6)


def test_weighted_mean_reduces_to_uniform_mean_for_unit_weights():
    # The learner's loss line multiplies (base_pg_loss * loss_masks * w).mean();
    # unit weights must reproduce the Dr.GRPO aggregation exactly.
    base = torch.tensor([0.3, -0.2, 0.1, 0.05])
    masks = torch.tensor([1.0, 1.0, 0.0, 1.0])
    w = torch.ones(4)
    assert torch.equal((base * masks * w).mean(), (base * masks).mean())


def test_masked_rows_get_zero_weight_and_valid_rows_renormalize():
    adv = [0.75, -0.25, -0.25, -0.25]
    counts = [32, 32, 32, 32]
    masks = torch.tensor([1.0, 1.0, 1.0, 0.0])
    w = compute_xdr_row_weights(
        torch.tensor(adv),
        torch.tensor(counts, dtype=torch.float32),
        num_samples=4,
        tau=0.25,
        t_max=256,
        loss_masks=masks,
    )
    assert w[3] == 0.0
    assert torch.allclose(w[:3].sum(), torch.tensor(3.0))


def test_all_masked_group_yields_zero_weights():
    masks = torch.zeros(4)
    w = compute_xdr_row_weights(
        torch.tensor([0.5, -0.5, 0.25, -0.25]),
        torch.tensor([32.0, 32.0, 32.0, 32.0]),
        num_samples=4,
        tau=0.25,
        t_max=256,
        loss_masks=masks,
    )
    assert torch.equal(w, torch.zeros(4))
