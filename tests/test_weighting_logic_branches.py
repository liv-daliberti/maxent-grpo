"""
Targeted coverage for weighting logic and reward types edge cases.
"""

from types import SimpleNamespace

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
