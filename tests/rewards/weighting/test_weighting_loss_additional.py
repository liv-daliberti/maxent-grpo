import types

import numpy as np
import pytest

from maxent_grpo.training.weighting import loss
from maxent_grpo.training.weighting.types import (
    KlControllerSettings,
    QDistributionSettings,
    TauSchedule,
    WeightNormalizationSettings,
    WeightStats,
    WeightingSettings,
)
from maxent_grpo.training.types.runtime import ClipSettings
from maxent_grpo.training.types.rewards import ReferenceLogprobs


@pytest.fixture(autouse=True)
def _use_real_torch(monkeypatch):
    # Install a usable torch stub with the ops required by these unit tests.
    from maxent_grpo.training.runtime.torch_stub import _build_torch_stub

    torch_mod = _build_torch_stub()
    torch_mod.clamp = torch_mod.clamp  # already provided by stub

    def _normalize(x, dim=1):
        arr = getattr(x, "arr", np.array(x))
        denom = np.linalg.norm(arr, axis=dim, keepdims=True) + 1e-12
        return torch_mod.tensor(arr / denom)

    def _cross_entropy(logits, targets):
        logits_arr = getattr(logits, "arr", np.array(logits))
        targ_arr = getattr(targets, "arr", np.array(targets))
        targ_arr = np.asarray(targ_arr, dtype=int)
        # Simple negative-log-likelihood of the target index.
        idx = targ_arr.reshape(-1)[0] if targ_arr.size else 0
        if logits_arr.size == 0:
            return torch_mod.tensor(0.0)
        exp_logits = np.exp(logits_arr - np.max(logits_arr))
        probs = exp_logits / exp_logits.sum()
        nll = -np.log(probs.reshape(-1)[idx] + 1e-12)
        return torch_mod.tensor(nll)

    torch_mod.nn.functional.normalize = _normalize
    torch_mod.nn.functional.cross_entropy = _cross_entropy
    TensorCls = torch_mod.tensor([0]).__class__
    if not hasattr(TensorCls, "t"):
        TensorCls.t = lambda self: torch_mod.tensor(self.arr.T)
    if not hasattr(TensorCls, "bool"):
        TensorCls.bool = lambda self: torch_mod.tensor(self.arr.astype(bool))
    if not hasattr(TensorCls, "__invert__"):
        TensorCls.__invert__ = lambda self: torch_mod.tensor(~self.arr.astype(bool))
    if not hasattr(TensorCls, "eq"):
        TensorCls.eq = TensorCls.__eq__
    if not hasattr(TensorCls, "fill_diagonal_"):
        def _fill(self, val):
            arr = np.array(self.arr, copy=True)
            np.fill_diagonal(arr, val)
            self.arr = arr
            return self
        TensorCls.fill_diagonal_ = _fill
    monkeypatch.setattr(loss, "torch", torch_mod)


def _weighting_cfg(len_norm_ref: bool = True, beta: float = 0.1) -> WeightingSettings:
    return WeightingSettings(
        tau=0.5,
        beta=beta,
        normalization=WeightNormalizationSettings(denom=1.0, len_norm_ref=len_norm_ref),
        q_distribution=QDistributionSettings(temperature=1.0, epsilon=1e-6),
        tau_schedule=TauSchedule(
            target_entropy=None,
            learning_rate=0.1,
            minimum_value=0.0,
            maximum_value=1.0,
            warmup_steps=0,
        ),
        kl_controller=KlControllerSettings(target=0.1, horizon=1, step_size=0.1),
        train_grpo_objective=True,
    )


def test_bucketized_kl_per_token_falls_back_to_last_bucket(monkeypatch):
    # Force a bucket definition that will never match to exercise the loop `else` path.
    monkeypatch.setattr(loss, "_KL_LENGTH_BUCKETS", [(10, 20)])
    token_counts = loss.torch.tensor([1.0])  # tok_val coerced to 1.0 < lower bound
    kl_vals = loss.torch.tensor([2.0])
    means, counts = loss._bucketized_kl_per_token(token_counts, kl_vals)
    assert "10-20" in means and "10-20" in counts
    assert counts["10-20"] > 0.0


def test_build_loss_inputs_handles_padding_and_truncation():
    # Mismatched counts trigger target_count override and padding branch.
    grouped = []
    weight_stats = WeightStats(
        weights_grouped=[],
        flat_weights=[0.5],
        weight_entropy=0.0,
        weight_entropy_min=0.0,
        weight_entropy_max=0.0,
        advantage_entropy=[],
    )
    scores = loss.SequenceScores(
        cur_logp_sum=loss.torch.tensor([0.1, 0.2]),
        behavior_logp_sum=loss.torch.tensor([0.0, 0.0]),
        log_ratio_train=loss.torch.tensor([0.0, 0.0]),
        denom_tok_tensor=loss.torch.tensor([1.0, 1.0]),
    )
    config = loss.LossInputConfig(
        clip_cfg=ClipSettings(clip_range=0.0, use_clip_objective=False, clip_objective_coef=0.0, clip_adv_baseline=None),
        weighting_cfg=_weighting_cfg(),
        ref_stats=ReferenceLogprobs(
            ref_logp_sum=loss.torch.tensor([0.0, 0.0]),
            ref_tok_counts=loss.torch.tensor([1.0, 1.0]),
            ref_logp_sum_raw=loss.torch.tensor([0.0, 0.0]),
            ref_logp_mean=0.0,
            avg_completion_tokens=1.0,
        ),
    )
    group_data, _ = loss.build_loss_inputs(grouped, weight_stats, scores, config)
    assert group_data.weight_tensor.numel() == 2

    # Truncation path when too many weights are provided.
    weight_stats.flat_weights = [0.1, 0.2, 0.3]
    grouped = [["a"]]
    scores = loss.SequenceScores(
        cur_logp_sum=loss.torch.tensor([0.5]),
        behavior_logp_sum=loss.torch.tensor([0.0]),
        log_ratio_train=loss.torch.tensor([0.0]),
        denom_tok_tensor=loss.torch.tensor([1.0]),
    )
    group_data, _ = loss.build_loss_inputs(grouped, weight_stats, scores, config)
    assert group_data.weight_tensor.numel() == 1


def test_clip_region_metrics_handles_non_tensor_and_typeerror():
    class _Bad:
        def __init__(self):
            self.data = [-2.0, 3.0]

        def __lt__(self, other):
            raise TypeError("bad lt")

        def __gt__(self, other):
            raise TypeError("bad gt")

    clip_cfg = ClipSettings(clip_range=0.2, use_clip_objective=True, clip_objective_coef=0.0, clip_adv_baseline=None)
    result = loss._clip_region_metrics(_Bad(), clip_cfg)
    clip_ratio, low_mean, *_ = result
    assert clip_ratio >= 0.0
    assert low_mean >= 0.0


def test_ratio_diagnostics_coerces_stub_tensors(monkeypatch):
    # Provide stub-like inputs to exercise conversion branches.
    ratio_ctx = loss.RatioContext(
        log_ratio_train=loss.torch.tensor([0.1, -0.2]),
        denom_tok_tensor=loss.torch.tensor([2.0, 2.0]),
        clip_cfg=ClipSettings(clip_range=0.1, use_clip_objective=True, clip_objective_coef=1.0, clip_adv_baseline=None),
        weighting_cfg=_weighting_cfg(),
        ref_stats=ReferenceLogprobs(
            ref_logp_sum=types.SimpleNamespace(arr=[0.0, 0.1]),
            ref_tok_counts=types.SimpleNamespace(arr=[1.0, 2.0, 3.0]),
            ref_logp_sum_raw=types.SimpleNamespace(arr=[0.0]),
            ref_logp_mean=0.0,
            avg_completion_tokens=1.0,
        ),
        cur_logp_sum=types.SimpleNamespace(arr=[0.2, 0.0]),
        behavior_logp_sum=loss.torch.tensor([0.0, 0.0]),
    )
    diag = loss._ratio_diagnostics(ratio_ctx)
    assert diag.kl_value >= 0.0
    assert diag.clip_ratio >= 0.0


def test_contrastive_seed_loss_positive_path():
    hidden = loss.torch.randn(3, 4)
    seed_ids = loss.torch.tensor([0, 0, 1])
    inputs = loss.SeedInfoInputs(seed_ids=seed_ids, pooled_hidden=hidden)
    out = loss._contrastive_seed_loss(inputs)
    assert out is not None


def test_evaluate_losses_with_entropy_and_seed_loss():
    grouped = [["a", "b"]]
    weight_stats = WeightStats(
        weights_grouped=[[0.6, 0.4]],
        flat_weights=[0.6, 0.4],
        weight_entropy=0.0,
        weight_entropy_min=0.0,
        weight_entropy_max=0.0,
        advantage_entropy=[0.0, 0.0],
    )
    scores = loss.SequenceScores(
        cur_logp_sum=loss.torch.tensor([-1.0, -1.5]),
        behavior_logp_sum=loss.torch.tensor([-1.2, -1.0]),
        log_ratio_train=loss.torch.tensor([0.0, 0.0]),
        denom_tok_tensor=loss.torch.tensor([2.0, 2.0]),
    )
    config = loss.LossInputConfig(
        clip_cfg=ClipSettings(clip_range=0.1, use_clip_objective=True, clip_objective_coef=0.5, clip_adv_baseline=None),
        weighting_cfg=_weighting_cfg(beta=0.2),
        ref_stats=ReferenceLogprobs(
            ref_logp_sum=loss.torch.tensor([-1.1, -1.4]),
            ref_tok_counts=loss.torch.tensor([2.0, 2.0]),
            ref_logp_sum_raw=loss.torch.tensor([-1.1, -1.4]),
            ref_logp_mean=0.0,
            avg_completion_tokens=1.0,
        ),
    )
    group_data, ratio_ctx = loss.build_loss_inputs(grouped, weight_stats, scores, config)
    seed_inputs = loss.SeedInfoInputs(
        seed_ids=loss.torch.tensor([0, 1]),
        pooled_hidden=loss.torch.randn(2, 4),
        is_seed_aug=loss.torch.tensor([0, 1]),
        logits=loss.torch.randn(2, 2),
    )
    outputs, diagnostics = loss.evaluate_losses(
        group_data,
        ratio_ctx,
        seed_inputs=seed_inputs,
        info_seed_lambda=0.5,
        info_seed_temperature=0.2,
        info_seed_loss_type="ce",
        info_seed_alpha_entropy=0.3,
    )
    assert outputs.loss is not None
    assert outputs.seed_loss is not None
    assert diagnostics.clip_ratio >= 0.0
