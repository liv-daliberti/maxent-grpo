"""
Copyright 2025 Liv d'Aliberti

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import annotations

import importlib
import pytest
from types import SimpleNamespace


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


def test_weighting_settings_accessors_cover_all_fields(training_stubs):
    mod = importlib.import_module("training.weighting.types")
    weighting = mod.WeightingSettings(
        tau=0.1,
        beta=0.2,
        normalization=mod.WeightNormalizationSettings(denom=1.0, len_norm_ref=True),
        q_distribution=mod.QDistributionSettings(temperature=0.5, epsilon=0.05),
        tau_schedule=mod.TauSchedule(
            target_entropy=0.2,
            learning_rate=0.01,
            minimum_value=0.0,
            maximum_value=1.0,
            warmup_steps=3,
        ),
        kl_controller=mod.KlControllerSettings(target=0.1, horizon=5, step_size=0.2),
        train_grpo_objective=True,
    )

    assert weighting.len_norm_ref is True
    weighting.len_norm_ref = False
    weighting.q_temperature = 0.75
    weighting.tau_min = 0.25
    weighting.tau_max = 0.9
    weighting.kl_horizon = 12
    weighting.kl_ctl_step_size = 0.05

    assert weighting.len_norm_ref is False
    assert weighting.normalization.len_norm_ref is False
    assert weighting.q_temperature == 0.75
    assert weighting.q_distribution.temperature == 0.75
    assert weighting.tau_min == 0.25
    assert weighting.tau_schedule.minimum_value == 0.25
    assert weighting.tau_max == 0.9
    assert weighting.tau_schedule.maximum_value == 0.9
    assert weighting.kl_horizon == 12
    assert weighting.kl_controller.horizon == 12
    assert weighting.kl_ctl_step_size == 0.05
    assert weighting.kl_controller.step_size == 0.05


def test_weighting_getattr_lazily_imports_logic(monkeypatch):
    weighting_pkg = importlib.import_module("training.weighting")
    monkeypatch.delitem(weighting_pkg.__dict__, "compute_weight_stats", raising=False)
    calls = {"count": 0}
    dummy = SimpleNamespace(compute_weight_stats=lambda: "ok")

    def _fake_import(name, package=None):
        calls["count"] += 1
        assert name == ".logic"
        assert package == weighting_pkg.__name__
        return dummy

    monkeypatch.setattr(weighting_pkg, "import_module", _fake_import)
    fn = weighting_pkg.__getattr__("compute_weight_stats")
    assert callable(fn)
    assert fn() == "ok"
    assert weighting_pkg.compute_weight_stats is fn
    assert calls["count"] == 1


def test_weighting_getattr_unknown_raises(training_stubs):
    weighting_pkg = importlib.import_module("training.weighting")
    with pytest.raises(AttributeError):
        weighting_pkg.__getattr__("does_not_exist")


def test_weighting_dir_matches_exports(training_stubs):
    weighting_pkg = importlib.import_module("training.weighting")
    exported = set(weighting_pkg.__all__)
    assert set(weighting_pkg.__dir__()) == exported
    assert {"WeightingSettings", "compute_weight_stats"} <= exported
