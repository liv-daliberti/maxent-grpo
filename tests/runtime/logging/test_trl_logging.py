from types import SimpleNamespace

import pytest

from maxent_grpo.telemetry.trl_logging import (
    ensure_weighting_logging,
    _WeightingLogCallback,
)


class _StubTrainer:
    def __init__(self):
        self.args = SimpleNamespace(
            maxent_tau=0.2,
            maxent_tau_lr=0.05,
            maxent_tau_min=0.1,
            maxent_tau_max=0.8,
            maxent_tau_warmup_steps=0,
            maxent_target_weight_entropy=0.3,
            maxent_q_temperature=1.2,
            maxent_q_epsilon=1e-4,
            maxent_length_normalize_ref=True,
            train_grpo_objective=True,
            kl_target=0.7,
            kl_horizon=10,
            kl_ctl_step_size=0.25,
        )
        self.state = SimpleNamespace(global_step=5)
        self.kl_ctl = SimpleNamespace(value=0.4)
        self.logged = None
        self._kl_increment = 0.05

    def log(self, logs):
        self.logged = logs
        # Mimic a KL controller update so delta metrics change between calls.
        self.kl_ctl.value += self._kl_increment
        return logs


def test_ensure_weighting_logging_adds_metrics_and_is_idempotent():
    Wrapped = ensure_weighting_logging(_StubTrainer)
    assert ensure_weighting_logging(Wrapped) is Wrapped
    trainer = Wrapped()

    trainer.log({"loss": 1.0})
    metrics = trainer.logged
    assert metrics["train/tau"] == pytest.approx(0.2)
    assert metrics["train/beta"] == pytest.approx(0.4)
    assert metrics["train/kl_coeff"] == pytest.approx(0.4)
    assert metrics["train/weight_norm_denom"] == pytest.approx(1.0)
    assert metrics["train/grpo_objective"] == 1.0
    assert metrics["train/maxent_objective"] == 0.0
    assert metrics["train/kl_controller_enabled"] == 1.0
    assert metrics["train/tau_target_enabled"] == 1.0
    assert metrics["train/tau_schedule_active"] == 1.0
    assert metrics["train/delta_beta"] == 0.0
    assert metrics["train/delta_tau"] == 0.0

    trainer.kl_ctl.value = 0.55
    trainer.args.maxent_tau = 0.25
    trainer.state.global_step = 6
    trainer.log({})
    metrics = trainer.logged
    assert metrics["train/beta"] == pytest.approx(0.55)
    assert metrics["train/tau"] == pytest.approx(0.25)
    assert metrics["train/delta_beta"] == pytest.approx(0.15)
    assert metrics["train/delta_tau"] == pytest.approx(0.05)


def test_maxent_objective_flagging_prefers_train_flag():
    class _StubTrainerMaxEnt(_StubTrainer):
        def __init__(self):
            super().__init__()
            self.args.train_grpo_objective = False
            self.args.maxent_objective = None  # ensure fallback flips the flags

    Wrapped = ensure_weighting_logging(_StubTrainerMaxEnt)
    trainer = Wrapped()
    trainer.log({})
    metrics = trainer.logged
    assert metrics["train/grpo_objective"] == 0.0
    assert metrics["train/maxent_objective"] == 1.0


def test_clipped_ratio_metrics_are_sanitized():
    class _StubTrainerClipped(_StubTrainer):
        def __init__(self):
            super().__init__()
            self.args.num_generations = 4
            # Simulate TRL internal metrics with negative counts.
            self._metrics = {"train": {"completions/clipped_ratio": [-8.0, -1.0]}}

        def log(self, logs):
            # Capture merged logs for inspection.
            self.logged = logs
            return logs

    Wrapped = ensure_weighting_logging(_StubTrainerClipped)
    trainer = Wrapped()
    trainer.log({})
    # Internal metrics should be normalized into [0, 1].
    normalized_vals = trainer._metrics["train"]["completions/clipped_ratio"]
    assert all(0.0 <= v <= 1.0 for v in normalized_vals)


def test_loss_raw_is_preserved_when_rounded_to_zero():
    class _FakeLoss:
        def __init__(self, val: float):
            self.val = val

        def mean(self):
            return self

        def item(self):
            return self.val

    class _LossyTrainer(_StubTrainer):
        def __init__(self):
            super().__init__()
            self._metrics = {"train": {}}
            self.args.num_generations = 1

        def compute_loss(self, *args, **kwargs):
            return _FakeLoss(1e-5)

        def log(self, logs):
            self.logged = logs
            return logs

    Wrapped = ensure_weighting_logging(_LossyTrainer)
    trainer = Wrapped()
    trainer.compute_loss()  # capture the raw loss
    trainer.log({"loss": 0.0})
    assert trainer.logged["train/loss/total_raw"] == pytest.approx(1e-5)
    # Loss should be backfilled when upstream rounded it away.
    assert trainer.logged["train/loss/total"] == pytest.approx(1e-5)


def test_callback_normalizes_logs_when_log_hook_is_bypassed():
    args = SimpleNamespace(num_generations=4, world_size=2)
    control = SimpleNamespace()
    logs = {"loss": 0.0, "kl": 0.1, "completions/clipped_ratio": -8.0}
    cb = _WeightingLogCallback()
    cb.on_log(args=args, state=None, control=control, logs=logs)
    assert "train/loss/total" in logs
    assert "train/kl" in logs
    assert logs["train/completions/clipped_frac"] == 0.0
    assert all(key.startswith("train/") for key in logs.keys())


def test_callback_attached_when_callback_handler_present():
    class _CallbackTrainer(_StubTrainer):
        def __init__(self):
            super().__init__()
            self.callback_handler = SimpleNamespace(callbacks=[])

        def add_callback(self, cb):
            self.callback_handler.callbacks.append(cb)

    Wrapped = ensure_weighting_logging(_CallbackTrainer)
    trainer = Wrapped()
    callbacks = getattr(trainer, "callback_handler").callbacks
    assert any(isinstance(cb, _WeightingLogCallback) for cb in callbacks)


def test_logging_tracks_delta_beta_changes():
    Wrapped = ensure_weighting_logging(_StubTrainer)
    trainer = Wrapped()
    trainer.log({"loss": 1.0})
    trainer.state.global_step = 6
    trainer.log({})
    metrics = trainer.logged
    assert metrics["train/delta_beta"] != 0.0


def test_weighting_logging_respects_tau_bounds_for_maxent():
    class _BoundedTrainer(_StubTrainer):
        def __init__(self):
            super().__init__()
            self.args.train_grpo_objective = False
            self.args.maxent_tau_min = 0.3
            self.args.maxent_tau_max = 0.3
            self.args.maxent_tau = 0.3

    Wrapped = ensure_weighting_logging(_BoundedTrainer)
    trainer = Wrapped()
    trainer.log({})
    metrics = trainer.logged
    assert metrics["train/maxent_objective"] == 1.0
    assert metrics["train/weighting/tau"] == pytest.approx(0.3)
    assert metrics["train/weighting/tau_min"] == pytest.approx(0.3)
    assert metrics["train/weighting/tau_max"] == pytest.approx(0.3)


def test_logging_respects_tau_bounds_for_maxent():
    class _BoundedTrainer(_StubTrainer):
        def __init__(self):
            super().__init__()
            self.args.train_grpo_objective = False
            self.args.maxent_tau = 0.5
            self.args.maxent_tau_min = 0.5
            self.args.maxent_tau_max = 0.5

    Wrapped = ensure_weighting_logging(_BoundedTrainer)
    trainer = Wrapped()
    trainer.log({})
    metrics = trainer.logged
    assert metrics["train/maxent_objective"] == 1.0
    assert metrics["train/weighting/tau"] == pytest.approx(0.5)
    assert metrics["train/tau"] == pytest.approx(0.5)
