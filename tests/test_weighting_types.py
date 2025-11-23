from __future__ import annotations

import importlib
import pytest


@pytest.fixture()
def training_stubs(monkeypatch):
    from tests.helpers.run_setup_stubs import install_training_stubs

    return install_training_stubs(monkeypatch)


def test_weighting_settings_passthrough_setters(training_stubs):
    mod = importlib.import_module("training.weighting.types")
    weighting = mod.WeightingSettings(
        tau=0.5,
        beta=0.2,
        normalization=mod.WeightNormalizationSettings(denom=2.0, len_norm_ref=True),
        q_distribution=mod.QDistributionSettings(temperature=1.0, epsilon=1e-6),
        tau_schedule=mod.TauSchedule(
            target_entropy=None,
            learning_rate=0.1,
            minimum_value=0.0,
            maximum_value=1.0,
            warmup_steps=5,
        ),
        kl_controller=mod.KlControllerSettings(target=0.3, horizon=10, step_size=0.01),
        train_grpo_objective=True,
    )

    weighting.q_epsilon = 0.123
    weighting.tau_target_entropy = 0.4

    assert weighting.q_epsilon == 0.123
    assert weighting.q_distribution.epsilon == 0.123
    assert weighting.tau_target_entropy == 0.4
    assert weighting.tau_schedule.target_entropy == 0.4
