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

Additional coverage for training.optim edge cases.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

# Ensure src/ is importable when tests are run directly.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from maxent_grpo.training import optim as opt


def test_clip_grad_norm_local_handles_zero_norm_returns_none():
    model = SimpleNamespace(parameters=lambda: [])
    accel = SimpleNamespace()
    assert opt.clip_grad_norm_local(model, accel, max_grad_norm=0.0) is None


def test_sync_gradients_enabled_logs_and_returns_flag(caplog):
    caplog.set_level("DEBUG")
    accel = SimpleNamespace(sync_gradients=False)
    assert opt.sync_gradients_enabled(accel, global_step=5) is False
    assert "sync_gradients=False" in caplog.text


def test_epoch_progress_with_steps_per_epoch():
    schedule = SimpleNamespace(steps_per_epoch=4)
    assert opt.epoch_progress(schedule, epoch=1, step_in_epoch=1) == pytest.approx(1.5)


def test_apply_learning_rate_skips_when_param_groups_missing():
    class _Opt:
        def __init__(self):
            self.param_groups = None

    handles = SimpleNamespace(optimizer=_Opt(), base_optimizer=_Opt())
    opt.apply_learning_rate(handles, learning_rate=0.1)  # should not raise


def test_configure_accumulation_steps_handles_missing_attributes():
    class _Accel:
        def __init__(self):
            self.gradient_state = None

    accel = _Accel()
    opt.configure_accumulation_steps(accel, grad_accum_steps=3)
    assert not hasattr(accel, "gradient_accumulation_steps") or accel.gradient_accumulation_steps == 3