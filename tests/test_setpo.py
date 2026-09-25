"""Contract tests for SetPO set-level diversity advantage shaping."""

from __future__ import annotations

import math

import pytest
import torch

from oat_drgrpo.setpo import setpo_marginal_contributions, shape_setpo_advantages


def _brute_force_marginals(embeddings: torch.Tensor, group: int) -> torch.Tensor:
    """Independent O(G^3) restatement of the published definition."""

    normalized = torch.nn.functional.normalize(embeddings.double(), dim=1)
    out: list[float] = []
    for start in range(0, embeddings.size(0), group):
        block = normalized[start : start + group]
        kernel = (block @ block.T).clamp(0.0, 1.0)

        def diversity(indices: list[int]) -> float:
            size = len(indices)
            terms = [
                -math.log1p(float(sum(kernel[i, j] for j in indices) / size))
                for i in indices
            ]
            return sum(terms) / size

        full = diversity(list(range(group)))
        for removed in range(group):
            survivors = [i for i in range(group) if i != removed]
            out.append(full - diversity(survivors))
    return torch.tensor(out, dtype=torch.float64)


def test_marginals_match_the_published_definition():
    torch.manual_seed(0)
    group = 5
    embeddings = torch.randn(group * 3, 7)
    expected = _brute_force_marginals(embeddings, group)
    actual = setpo_marginal_contributions(embeddings, num_samples=group)
    assert torch.allclose(expected, actual.double(), atol=1e-6)


def test_rarer_trajectories_contribute_more():
    """The monotonicity the paper proves, asserted on the implementation."""

    dim = 8
    duplicated = torch.ones(4, dim)
    distinct = torch.zeros(1, dim)
    distinct[0, 0] = 1.0
    marginals = setpo_marginal_contributions(
        torch.cat([duplicated, distinct]), num_samples=5
    )
    assert marginals[4] > marginals[0]
    assert torch.allclose(marginals[:4], marginals[:1].expand(4), atol=1e-6)


def test_shaping_is_additive_and_scales_with_the_coefficient():
    torch.manual_seed(1)
    group = 4
    embeddings = torch.randn(group * 2, 6)
    base = torch.randn(group * 2, 1)
    marginals = setpo_marginal_contributions(embeddings, num_samples=group)
    shaped, _ = shape_setpo_advantages(
        base, embeddings, num_samples=group, coefficient=0.25
    )
    assert torch.allclose(
        shaped.reshape(-1), base.reshape(-1) + 0.25 * marginals, atol=1e-6
    )
    inert, _ = shape_setpo_advantages(
        base, embeddings, num_samples=group, coefficient=0.0
    )
    assert torch.allclose(inert, base, atol=1e-7)


def test_shaping_preserves_the_advantage_shape():
    embeddings = torch.randn(8, 5)
    base = torch.randn(8, 1)
    shaped, diagnostics = shape_setpo_advantages(
        base, embeddings, num_samples=4, coefficient=0.1
    )
    assert shaped.shape == base.shape
    assert diagnostics.groups == 2
    assert diagnostics.rows == 8


def test_uncentred_group_shift_is_reported():
    """The published rule does not centre, so the bias must be observable."""

    torch.manual_seed(2)
    embeddings = torch.randn(8, 5)
    _, diagnostics = shape_setpo_advantages(
        torch.zeros(8, 1), embeddings, num_samples=4, coefficient=1.0
    )
    assert diagnostics.group_mean_shift_abs_max > 0.0


def test_kernel_is_clamped_onto_a_similarity():
    """Opposed embeddings give kernel zero, never a negative mass."""

    dim = 4
    opposed = torch.zeros(4, dim)
    opposed[0, 0] = 1.0
    opposed[1, 0] = -1.0
    opposed[2, 1] = 1.0
    opposed[3, 1] = -1.0
    _, diagnostics = shape_setpo_advantages(
        torch.zeros(4, 1), opposed, num_samples=4, coefficient=1.0
    )
    assert diagnostics.kernel_min >= 0.0
    assert diagnostics.kernel_max <= 1.0 + 1e-6


def test_incomplete_groups_are_refused():
    with pytest.raises(ValueError, match="complete rollout groups"):
        setpo_marginal_contributions(torch.randn(7, 3), num_samples=4)


def test_non_finite_embeddings_are_refused():
    embeddings = torch.randn(8, 3)
    embeddings[2, 1] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        setpo_marginal_contributions(embeddings, num_samples=4)
