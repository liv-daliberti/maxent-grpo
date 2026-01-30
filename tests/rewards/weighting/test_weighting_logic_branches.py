"""
Targeted coverage for weighting logic and reward types edge cases.
"""

from types import SimpleNamespace

import logging
import math

import pytest

from maxent_grpo.training.weighting import logic
from maxent_grpo.training.types.rewards import LossOutputs, LossScalarBundle


def test_weight_vector_from_q_falls_back_on_tensor_error(monkeypatch):
    """Ensure weight_vector_from_q returns uniform weights when torch tensor fails."""

    def _bad_tensor(*_a, **_k):
        raise TypeError("boom")

    monkeypatch.setattr(
        logic, "torch", SimpleNamespace(tensor=_bad_tensor, float32=object())
    )
    cfg = SimpleNamespace(tau=1.0, beta=1.0, denom=2.0)
    weights = logic.weight_vector_from_q([0.1, 0.2], [0.3, 0.4], None, cfg)
    assert weights == [0.5, 0.5]


def test_build_weighting_settings_respects_kl_penalty_beta():
    """Weighting builder should fall back to kl_penalty_beta when other aliases missing."""

    cfg = SimpleNamespace(
        maxent_tau=0.2,
        init_kl_coeff=None,
        init_kl_coef=None,
        kl_penalty_beta=0.07,
        beta=None,
        maxent_length_normalize_ref=True,
        maxent_q_temperature=1.0,
        maxent_q_epsilon=1e-6,
        maxent_target_weight_entropy=None,
        maxent_tau_lr=0.0,
        maxent_tau_min=0.0,
        maxent_tau_max=1.0,
        maxent_tau_warmup_steps=0,
        kl_target=0.0,
        kl_horizon=0,
        kl_ctl_step_size=0.0,
        controller_meta_enabled=False,
        controller_meta_method="analytic",
        controller_meta_lr=0.0,
        controller_meta_update_interval=1,
        controller_meta_objective="potential",
        controller_meta_analytic_steps=1,
        controller_meta_optimizer="sgd",
        controller_meta_truncation_steps=1,
        controller_meta_use_hessian=False,
        train_grpo_objective=False,
        maxent_allow_empty_weight_fallback=False,
    )
    weighting = logic.build_weighting_settings(cfg)
    assert weighting.beta == pytest.approx(0.07)


def test_maybe_update_tau_converts_non_numeric_tau_lr(monkeypatch):
    """String tau_lr should be coerced and used to update tau."""

    cfg = SimpleNamespace(
        tau_lr="0.5",
        tau_target_entropy=1.0,
        tau_warmup_steps=0,
        tau_entropy=0.0,
        tau=0.1,
        tau_min=0.0,
        tau_max=10.0,
        beta=0.2,
        train_grpo_objective=True,
        denom=1.0,
    )
    stats = SimpleNamespace(weight_entropy=0.0)
    logic.maybe_update_tau(cfg, stats, global_step=1, lr_scale=None)
    assert getattr(cfg, "_tau_lr_base") == 0.5
    assert cfg.denom == 1.0


def test_broadcast_controller_state_handles_payload_type_error(monkeypatch):
    """Non-numeric controller fields should short-circuit with False."""

    accel = SimpleNamespace(broadcast_object_list=lambda payload, src=0: payload)
    cfg = SimpleNamespace(beta="bad", tau=1.0, train_grpo_objective=False)
    assert logic.broadcast_controller_state(accel, cfg) is False


def test_broadcast_controller_state_caches_and_recovers(monkeypatch):
    """Cache from rank0 should be reused when later broadcasts fail."""

    calls = {"count": 0}

    def _flaky_bcast(payload, src=0):
        calls["count"] += 1
        raise RuntimeError("fail")

    # First call seeds cache on rank 0 even when broadcast raises
    cfg0 = SimpleNamespace(
        beta=0.3, tau=0.4, train_grpo_objective=True, process_index=0
    )
    accel0 = SimpleNamespace(broadcast_object_list=_flaky_bcast, process_index=0)
    assert logic.broadcast_controller_state(accel0, cfg0) is True
    assert cfg0.denom == 1.0
    # Second call pulls from cached payload on non-zero rank when broadcast returns None
    cfg1 = SimpleNamespace(
        beta=0.0, tau=0.0, train_grpo_objective=True, process_index=1
    )
    accel1 = SimpleNamespace(broadcast_object_list=lambda payload, src=0: None, process_index=1)
    assert logic.broadcast_controller_state(accel1, cfg1) is True
    assert cfg1.beta == cfg0.beta and cfg1.tau == cfg0.tau


def test_broadcast_controller_state_bad_payload_returns_false():
    """Malformed payload from broadcast should fail gracefully."""

    accel = SimpleNamespace(broadcast_object_list=lambda payload, src=0: [["onlybeta"]], process_index=0)
    cfg = SimpleNamespace(beta=0.1, tau=0.2, train_grpo_objective=False)
    assert logic.broadcast_controller_state(accel, cfg) is False


def test_collect_weight_entropy_inner_log_fallback(monkeypatch):
    """When tensor.log fails, fallback path using torch.log should execute."""

    class _DummyTensor:
        def __init__(self, data):
            self.data = list(data)

        def clamp(self, min=None):
            return self

        def log(self):
            raise RuntimeError("no log")

        def __neg__(self):
            return self

        def __mul__(self, _other):
            return self

        def sum(self):
            return SimpleNamespace(item=lambda: sum(self.data))

    torch_stub = SimpleNamespace(
        tensor=lambda data, dtype=None: _DummyTensor(data),
        log=lambda tensor: tensor,
        float32=object(),
    )
    monkeypatch.setattr(logic, "torch", torch_stub)
    mean_entropy, *_ = logic.collect_weight_entropy([[0.25, 0.75]])
    assert mean_entropy >= 0


def test_collect_weight_entropy_manual_fallback(monkeypatch):
    """When tensor creation fails, manual entropy computation should be used."""

    torch_stub = SimpleNamespace(
        tensor=lambda *_a, **_k: (_ for _ in ()).throw(TypeError("no tensor")),
        float32=object(),
    )
    monkeypatch.setattr(logic, "torch", torch_stub)
    mean_entropy, min_entropy, max_entropy, _ = logic.collect_weight_entropy([[0.5, 0.5]])
    assert mean_entropy == min_entropy == max_entropy
    assert mean_entropy > 0


# --- Length normalization and token scaling consistency ----------------------


def test_split_reference_logprobs_length_normalized(monkeypatch):
    """Length-normalized ref logp should divide by token counts."""

    ref_stats = SimpleNamespace(
        ref_logp_sum_raw=_FakeTensor([10.0, 8.0, 6.0]),
        ref_tok_counts=_FakeTensor([2.0, 4.0, 3.0]),
    )
    groups = [["a", "b"], ["c"]]
    result = logic.split_reference_logprobs(groups, ref_stats, len_norm_ref=True)
    assert result == [[5.0, 2.0], [2.0]]


def test_split_reference_logprobs_length_norm_handles_missing(caplog):
    """When len-norm is on and log-probs are missing, fall back to zeros and warn."""

    caplog.set_level(logging.WARNING)
    ref_stats = SimpleNamespace(
        ref_logp_sum=_FakeTensor([]),
        ref_logp_sum_raw=_FakeTensor([]),
        ref_tok_counts=_FakeTensor([3.0, 3.0, 3.0]),
    )
    groups = [["a", "b"], ["c"]]
    result = logic.split_reference_logprobs(groups, ref_stats, len_norm_ref=True)
    assert result == [[0.0, 0.0], [0.0]]
    assert "Reference log-prob/token mismatch" in caplog.text


def test_compute_weight_stats_disables_token_norm_when_len_norm_ref(monkeypatch):
    """When ref log-probs are length-normalized, token scaling is skipped."""

    called = {}

    def _fake_weight_vector(q_vals, logp_vals, tok_counts, weighting_cfg, *, include_reference_term, normalize_by_tokens):
        called["normalize_by_tokens"] = normalize_by_tokens
        return [0.6, 0.4]

    monkeypatch.setattr(logic, "weight_vector_from_q", _fake_weight_vector)
    monkeypatch.setattr(
        logic, "collect_weight_entropy", lambda weights: (1.0, 1.0, 1.0, [])
    )
    ref_stats = SimpleNamespace(
        ref_logp_sum=_FakeTensor([1.0, 2.0]),
        ref_logp_sum_raw=_FakeTensor([1.0, 2.0]),
        ref_tok_counts=_FakeTensor([1.0, 1.0]),
    )
    reward_comp = SimpleNamespace(q_grouped=[[0.5, 0.5]])
    cfg = SimpleNamespace(
        tau=0.1,
        beta=0.2,
        len_norm_ref=True,
        train_grpo_objective=False,
        denom=0.3,
    )
    stats = logic.compute_weight_stats([["a", "b"]], reward_comp, ref_stats, cfg)
    assert called["normalize_by_tokens"] is False
    assert stats.weights_grouped[0] == [0.6, 0.4]


def test_compute_weight_stats_includes_ref_for_maxent(monkeypatch):
    """MaxEnt mode should include reference term when building weights."""

    called = {}

    def _fake_weight_vector(q_vals, logp_vals, tok_counts, weighting_cfg, *, include_reference_term, normalize_by_tokens):
        called["include_reference_term"] = include_reference_term
        return [0.7, 0.3]

    monkeypatch.setattr(logic, "weight_vector_from_q", _fake_weight_vector)
    monkeypatch.setattr(
        logic, "collect_weight_entropy", lambda weights: (1.0, 1.0, 1.0, [])
    )
    ref_stats = SimpleNamespace(
        ref_logp_sum_raw=_FakeTensor([1.0, 2.0]),
        ref_tok_counts=_FakeTensor([1.0, 1.0]),
    )
    reward_comp = SimpleNamespace(q_grouped=[[0.4, 0.6]])
    cfg = SimpleNamespace(
        tau=0.1,
        beta=0.2,
        len_norm_ref=False,
        train_grpo_objective=False,
        denom=0.3,
    )
    stats = logic.compute_weight_stats([["a", "b"]], reward_comp, ref_stats, cfg)
    assert called["include_reference_term"] is True
    assert stats.weights_grouped[0] == [0.7, 0.3]


def test_compute_weight_stats_skips_ref_for_grpo(monkeypatch):
    """GRPO mode should use grouped advantages instead of q-weighting."""

    def _fake_weight_vector(*_args, **_kwargs):
        raise AssertionError("weight_vector_from_q should not be called in GRPO mode")

    monkeypatch.setattr(logic, "weight_vector_from_q", _fake_weight_vector)
    ref_stats = SimpleNamespace(
        ref_logp_sum_raw=_FakeTensor([1.0, 2.0]),
        ref_tok_counts=_FakeTensor([1.0, 1.0]),
    )
    reward_comp = SimpleNamespace(
        q_grouped=[[0.4, 0.6]],
        total_utils=[1.0, 2.0],
        advantage=SimpleNamespace(grouped=[[ -0.5, 0.5 ]], samples=[-0.5, 0.5]),
    )
    cfg = SimpleNamespace(
        tau=0.0,
        beta=0.0,
        len_norm_ref=False,
        train_grpo_objective=True,
        denom=1.0,
    )
    stats = logic.compute_weight_stats([["a", "b"]], reward_comp, ref_stats, cfg)
    assert stats.weights_grouped[0] == pytest.approx([-0.5, 0.5])


# --- Beta controller ----------------------------------------------------------


def test_maybe_update_beta_updates_and_sets_denom_for_grpo():
    """KL controller should scale beta and denom in GRPO mode."""

    cfg = SimpleNamespace(
        beta=0.2,
        tau=0.0,
        kl_target=0.1,
        kl_horizon=10,
        kl_ctl_step_size=0.2,
        train_grpo_objective=True,
        denom=1.0,
    )
    logic.maybe_update_beta(cfg, measured_kl=0.2)  # ratio=2 -> clipped error 0.2 -> scale=1.02
    assert cfg.beta == pytest.approx(0.204, rel=1e-6)
    assert cfg.denom == pytest.approx(cfg.beta)


def test_maybe_update_beta_updates_and_sets_denom_for_maxent():
    """KL controller should scale beta and recompute denom with tau for MaxEnt."""

    cfg = SimpleNamespace(
        beta=0.2,
        tau=0.3,
        kl_target=0.1,
        kl_horizon=10,
        kl_ctl_step_size=0.2,
        train_grpo_objective=False,
        denom=1.0,
    )
    logic.maybe_update_beta(cfg, measured_kl=0.2)
    expected_beta = 0.2 * (1.0 + 0.2 / 10.0)
    assert cfg.beta == pytest.approx(expected_beta)
    assert cfg.denom == pytest.approx(cfg.tau + expected_beta)


def test_maybe_update_tau_updates_entropy_ema_and_denom():
    """Tau controller should update tau, tau_log, entropy EMA, and denom."""

    cfg = SimpleNamespace(
        tau=0.1,
        beta=0.2,
        tau_target_entropy=1.0,
        tau_lr=0.1,
        tau_min=0.0,
        tau_max=10.0,
        tau_warmup_steps=0,
        train_grpo_objective=False,
        denom=0.3,
    )
    stats = SimpleNamespace(weight_entropy=0.5)
    logic.maybe_update_tau(cfg, stats, global_step=5, lr_scale=None)
    expected_tau_log = math.log(0.1) + 0.1 * (1.0 - 0.5)
    assert cfg._tau_log == pytest.approx(expected_tau_log)
    assert cfg.tau == pytest.approx(math.exp(expected_tau_log))
    assert cfg.denom == pytest.approx(cfg.tau + cfg.beta)
    assert cfg._tau_entropy_ema == pytest.approx(0.5)


# --- Token-count scaling guard for MaxEnt vs GRPO ----------------------------------
# --- Token-count scaling guard for MaxEnt vs GRPO ----------------------------------


class _FakeTensor(list):
    """Minimal tensor stub for weight_vector_from_q tests."""

    def __getitem__(self, key):
        result = super().__getitem__(key)
        return _FakeTensor(result) if isinstance(key, slice) else result

    def clamp(self, min=None):
        min_val = -math.inf if min is None else min
        return _FakeTensor([max(min_val, v) for v in self])

    def log(self):
        return _FakeTensor([math.log(v) for v in self])

    def __truediv__(self, other):
        if isinstance(other, _FakeTensor):
            return _FakeTensor([a / b for a, b in zip(self, other)])
        return _FakeTensor([v / other for v in self])

    def __add__(self, other):
        if isinstance(other, _FakeTensor):
            return _FakeTensor([a + b for a, b in zip(self, other)])
        return _FakeTensor([v + other for v in self])

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        if isinstance(other, _FakeTensor):
            return _FakeTensor([a * b for a, b in zip(self, other)])
        return _FakeTensor([v * other for v in self])

    def sum(self):
        return float(math.fsum(self))

    def tolist(self):
        return list(self)


def _softmax_stub(tensor, dim=0):
    _ = dim
    exp_vals = [math.exp(v) for v in tensor]
    denom = math.fsum(exp_vals) or 1.0
    return _FakeTensor([v / denom for v in exp_vals])


def _log_stub(tensor):
    return _FakeTensor([math.log(v) for v in tensor])


def test_weight_vector_respects_token_scaling_for_grpo(monkeypatch):
    """GRPO mode still applies length scaling after the softmax."""

    monkeypatch.setattr(
        logic,
        "torch",
        SimpleNamespace(
            tensor=_FakeTensor,
            float32=object(),
            softmax=_softmax_stub,
            log=_log_stub,
        ),
    )
    cfg = SimpleNamespace(tau=0.0, beta=0.0, denom=1.0, train_grpo_objective=True)
    weights = logic.weight_vector_from_q(
        [0.5, 0.5],
        [0.0, 0.0],
        token_counts=[1.0, 2.0],
        weighting_cfg=cfg,
    )
    assert weights == pytest.approx([1.0 / 3.0, 2.0 / 3.0])


def test_weight_vector_skips_token_scaling_for_maxent(monkeypatch):
    """MaxEnt mode still returns weights; token scaling is applied when enabled."""

    monkeypatch.setattr(
        logic,
        "torch",
        SimpleNamespace(
            tensor=_FakeTensor,
            float32=object(),
            softmax=_softmax_stub,
            log=_log_stub,
        ),
    )
    cfg = SimpleNamespace(tau=0.1, beta=0.2, denom=0.3, train_grpo_objective=False)
    weights = logic.weight_vector_from_q(
        [0.5, 0.5],
        [0.0, 0.0],
        token_counts=[1.0, 2.0],
        weighting_cfg=cfg,
    )
    assert weights == pytest.approx([1.0 / 3.0, 2.0 / 3.0])


def test_loss_outputs_info_seed_entropy_value():
    """Ensure convenience accessor forwards the scalar."""

    scalars = LossScalarBundle(
        total_loss=1.0,
        policy_loss=0.5,
        clip_loss=None,
        kl_loss=0.1,
        weighted_kl_loss=0.2,
    )
    loss_outputs = LossOutputs(
        loss=None,
        scalars=scalars,
        log_ratio_train=None,
        denom_tok_tensor=None,
        seed_loss_scalar=0.3,
        info_seed_entropy_scalar=0.4,
    )
    assert loss_outputs.info_seed_entropy_value == 0.4
