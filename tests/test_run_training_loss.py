"""Tests for the training loss helpers when torch is available."""

from __future__ import annotations

import importlib

import pytest

torch = pytest.importorskip(
    "torch",
    reason="torch is required for training loss tests",
)

if not hasattr(torch, "tensor"):
    pytest.skip("torch tensor operations unavailable", allow_module_level=True)

_loss_mod = importlib.import_module("maxent_helpers.run_training_loss")
RatioContext = _loss_mod.RatioContext
SequenceScores = _loss_mod.SequenceScores
_kl_terms = _loss_mod._kl_terms


def _make_ratio_ctx(beta: float = 0.5, len_norm_ref: bool = True) -> RatioContext:
    clip_cfg = type("ClipCfg", (), {"use_clip_objective": False, "clip_range": 0.2})()
    weighting_cfg = type(
        "WeightCfg",
        (),
        {"beta": beta, "len_norm_ref": len_norm_ref},
    )()
    ref_stats = type(
        "RefStats",
        (),
        {
            "ref_logp_sum": torch.tensor([0.0, 0.2]),
            "ref_logp_sum_raw": torch.tensor([0.0, 0.4]),
            "ref_tok_counts": torch.tensor([2.0, 2.0]),
        },
    )()
    denom = torch.tensor([2.0, 2.0])
    cur_logp = torch.tensor([-0.5, -0.3])
    return RatioContext(
        log_ratio_train=torch.zeros_like(cur_logp),
        denom_tok_tensor=denom,
        clip_cfg=clip_cfg,
        weighting_cfg=weighting_cfg,
        ref_stats=ref_stats,
        cur_logp_sum=cur_logp,
        behavior_logp_sum=torch.tensor([-0.4, -0.3]),
    )


def test_sequence_scores_round_trip():
    scores = SequenceScores(
        cur_logp_sum=torch.tensor([-1.0, -2.0]),
        behavior_logp_sum=torch.tensor([-0.9, -1.8]),
        log_ratio_train=torch.tensor([0.1, -0.2]),
        denom_tok_tensor=torch.tensor([3.0, 4.0]),
    )
    assert torch.allclose(scores.cur_logp_sum, torch.tensor([-1.0, -2.0]))
    assert scores.denom_tok_tensor.shape == (2,)


@pytest.mark.parametrize("len_norm_ref", [True, False])
def test_kl_terms_respects_reference_mode(len_norm_ref: bool):
    ctx = _make_ratio_ctx(beta=0.3, len_norm_ref=len_norm_ref)
    kl_tensor, kl_scalar, weighted = _kl_terms(ctx)
    assert pytest.approx(kl_scalar) == float(kl_tensor.detach().cpu().item())
    assert pytest.approx(weighted) == ctx.weighting_cfg.beta * kl_scalar
