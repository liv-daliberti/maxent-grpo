"""
Regression tests for parity between MaxEnt and GRPO weighting paths.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(
    not callable(getattr(torch, "tensor", None)),
    reason="torch stub missing tensor support",
)
if not hasattr(torch, "log"):
    torch.log = lambda x: torch.tensor(np.log(getattr(x, "arr", x)))
if not hasattr(torch, "softmax"):
    torch.softmax = lambda x, dim=0: torch.tensor(
        np.exp(getattr(x, "arr", x)) / np.sum(np.exp(getattr(x, "arr", x)), axis=dim),
        dtype=getattr(x, "dtype", None),
    )
if hasattr(torch, "Tensor") and not hasattr(torch.Tensor, "__rmul__"):
    torch.Tensor.__rmul__ = lambda self, other: self.__mul__(other)
if hasattr(torch, "Tensor") and not hasattr(torch.Tensor, "log"):
    torch.Tensor.log = lambda self: torch.log(self)
if hasattr(torch, "Tensor") and not hasattr(torch.Tensor, "__neg__"):
    torch.Tensor.__neg__ = lambda self: self * -1

from maxent_grpo.training.types import ReferenceLogprobs  # noqa: E402
from maxent_grpo.training.weighting import (  # noqa: E402
    KlControllerSettings,
    QDistributionSettings,
    TauSchedule,
    WeightNormalizationSettings,
    WeightingSettings,
    compute_weight_stats,
)


def _weighting(train_grpo: bool) -> WeightingSettings:
    return WeightingSettings(
        tau=0.0,
        beta=1.0,
        normalization=WeightNormalizationSettings(denom=1.0, len_norm_ref=True),
        q_distribution=QDistributionSettings(temperature=1.0, epsilon=1e-6),
        tau_schedule=TauSchedule(
            target_entropy=None,
            learning_rate=0.0,
            minimum_value=0.0,
            maximum_value=2.0,
            warmup_steps=0,
        ),
        kl_controller=KlControllerSettings(target=0.0, horizon=0, step_size=0.0),
        train_grpo_objective=train_grpo,
    )


def _ref_stats() -> ReferenceLogprobs:
    ref_logp = torch.tensor([-1.0, -2.0], dtype=torch.float32)
    tok_counts = torch.tensor([1.0, 1.0], dtype=torch.float32)
    return ReferenceLogprobs(
        ref_logp_sum=ref_logp,
        ref_tok_counts=tok_counts,
        ref_logp_sum_raw=ref_logp.clone(),
        ref_logp_mean=float(ref_logp.mean().item()),
        avg_completion_tokens=1.0,
    )


def test_zero_tau_grpo_weights_ignore_reference():
    """When train_grpo_objective is set, weights should ignore the reference term."""

    grouped = [["a", "b"]]
    reward_comp = SimpleNamespace(
        q_grouped=[[0.5, 0.5]],
        total_utils=[0.0, 1.0],
        advantage=SimpleNamespace(grouped=[[-0.5, 0.5]], samples=[-0.5, 0.5]),
    )

    grpo_stats = compute_weight_stats(
        grouped, reward_comp, _ref_stats(), _weighting(True)
    )
    maxent_stats = compute_weight_stats(
        grouped, reward_comp, _ref_stats(), _weighting(False)
    )

    assert grpo_stats is not None and maxent_stats is not None
    assert grpo_stats.weights_grouped[0] == pytest.approx([-0.5, 0.5])
    assert maxent_stats.weights_grouped[0] != pytest.approx([-0.5, 0.5])
    assert maxent_stats.weights_grouped[0][0] > maxent_stats.weights_grouped[0][1]
