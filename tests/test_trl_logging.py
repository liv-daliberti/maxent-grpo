from types import SimpleNamespace

import pytest

from maxent_grpo.telemetry.trl_logging import ensure_weighting_logging


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

    def log(self, logs):
        self.logged = logs
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
