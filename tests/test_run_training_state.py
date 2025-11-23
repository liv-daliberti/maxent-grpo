"""Tests for run_training_state helpers using lightweight stubs."""

from __future__ import annotations

import os
import sys
import types
from types import SimpleNamespace

import pytest

_TORCH_STUB = types.ModuleType("torch")
_TORCH_STUB.Tensor = type("Tensor", (object,), {})
_TORCH_STUB.device = type("device", (object,), {})
_TORCH_STUB.optim = SimpleNamespace(Optimizer=type("Optimizer", (object,), {}))
_TORCH_STUB.__spec__ = SimpleNamespace()
sys.modules["torch"] = _TORCH_STUB
torch_utils = types.ModuleType("torch.utils")
sys.modules["torch.utils"] = torch_utils
torch_utils_data = types.ModuleType("torch.utils.data")
torch_utils_data.DataLoader = type("DataLoader", (object,), {})
sys.modules["torch.utils.data"] = torch_utils_data
torch_nn = types.ModuleType("torch.nn")
torch_nn.functional = types.ModuleType("torch.nn.functional")
torch_nn.functional.log_softmax = lambda *args, **kwargs: None
sys.modules["torch.nn"] = torch_nn
sys.modules["torch.nn.functional"] = torch_nn.functional
accelerate_stub = types.ModuleType("accelerate")
accelerate_stub.Accelerator = type("Accelerator", (object,), {})
sys.modules["accelerate"] = accelerate_stub
transformers_stub = types.ModuleType("transformers")
transformers_stub.PreTrainedModel = type("PreTrainedModel", (object,), {})
transformers_stub.PreTrainedTokenizer = type("PreTrainedTokenizer", (object,), {})
sys.modules["transformers"] = transformers_stub

from maxent_helpers.run_training_state import (  # noqa: E402  import after stubs
    check_stop_condition,
    load_controller_state_chain,
    maybe_checkpoint,
    maybe_clear_stale_controller_state,
    maybe_load_accelerator_state,
    _checkpoint_log_once,
)
from maxent_helpers.run_training_weighting import CONTROLLER_STATE_FILENAME  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_checkpoint_logs():
    _checkpoint_log_once["config"] = False
    _checkpoint_log_once["strategy"] = False
    _checkpoint_log_once["steps"] = False
    yield
    _checkpoint_log_once["config"] = False
    _checkpoint_log_once["strategy"] = False
    _checkpoint_log_once["steps"] = False


class _Accel:
    def __init__(self, is_main_process: bool = True):
        self.is_main_process = is_main_process
        self.wait_calls = 0
        self.loaded_path = None

    def wait_for_everyone(self):
        self.wait_calls += 1

    def load_state(self, path: str):
        self.loaded_path = path


def test_maybe_clear_stale_controller_state_removes_file(tmp_path):
    state_path = tmp_path / "controller.bin"
    state_path.write_text("checkpoint")
    controller_cfg = SimpleNamespace(
        resume_from=None,
        overwrite_existing=True,
        state_path=str(state_path),
    )
    accel = _Accel()
    maybe_clear_stale_controller_state(accel, controller_cfg)
    assert not state_path.exists()
    assert accel.wait_calls == 1


def test_load_controller_state_chain_prefers_resume(monkeypatch, tmp_path):
    state_dir = tmp_path / "resume"
    state_dir.mkdir()
    resume_state = state_dir / CONTROLLER_STATE_FILENAME
    resume_state.write_text("state")
    controller_cfg = SimpleNamespace(
        resume_from=str(state_dir),
        overwrite_existing=False,
        state_path=str(tmp_path / "current.bin"),
    )
    accel = _Accel()
    weighting_cfg = SimpleNamespace(beta=0.0, tau=0.0)
    load_calls = []

    def _fake_load(path, _accel, _weight):
        load_calls.append(path)
        return path == str(resume_state)

    monkeypatch.setattr(
        "maxent_helpers.run_training_state._load_controller_file",
        _fake_load,
    )
    loaded = load_controller_state_chain(controller_cfg, accel, weighting_cfg)
    assert loaded is True
    assert load_calls[0].endswith(CONTROLLER_STATE_FILENAME)


def test_maybe_load_accelerator_state_invokes_load(tmp_path):
    state_dir = tmp_path / "accel"
    state_dir.mkdir()
    accel = _Accel()
    maybe_load_accelerator_state(str(state_dir), accel)
    assert accel.loaded_path == str(state_dir)
    assert accel.wait_calls == 1


def test_maybe_checkpoint_triggers_save(monkeypatch):
    saves = []

    class _LoggingCfg:
        save_strategy = "steps"
        save_steps = 2

        @staticmethod
        def save_checkpoint(name: str):
            saves.append(name)

    accel = _Accel()
    maybe_checkpoint(_LoggingCfg(), accel, global_step=1)
    assert not saves
    maybe_checkpoint(_LoggingCfg(), accel, global_step=2)
    assert saves == ["checkpoint-2"]


def test_check_stop_condition_sets_flag():
    schedule = SimpleNamespace(total_training_steps=10)
    state = SimpleNamespace(global_step=11, stop_training=False)
    check_stop_condition(schedule, state)
    assert state.stop_training is True
