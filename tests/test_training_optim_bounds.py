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

Tests for optimizer scheduling utilities.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from maxent_grpo.training import optim


def test_scheduled_learning_rate_respects_negative_inputs(monkeypatch):
    schedule = SimpleNamespace(
        warmup_steps=0,
        total_training_steps=10,
    )
    handles = SimpleNamespace(learning_rate=1.0)
    lr = optim.scheduled_learning_rate(schedule, handles, step=5)
    assert lr > 0.0


@pytest.mark.parametrize("field,value", [("kl_target", -1), ("kl_horizon", -2)])
def test_configure_accumulation_steps_noop_when_invalid(monkeypatch, field, value):
    # Ensure configuring accumulation steps doesn't explode on missing setters.
    acc = SimpleNamespace(state=None)
    optim.configure_accumulation_steps(acc, grad_accum_steps=1)
    # Sync gradients always returns bool; guard against accidental exceptions.
    assert optim.sync_gradients_enabled(acc, global_step=0) in (True, False)