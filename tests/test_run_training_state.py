"""Tests for :mod:`training.state` helpers using lightweight stubs."""

from __future__ import annotations

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
torch_utils_data.Sampler = type("Sampler", (object,), {})
sys.modules["torch.utils.data"] = torch_utils_data
torch_nn = types.ModuleType("torch.nn")
torch_nn.functional = types.ModuleType("torch.nn.functional")
torch_nn.functional.log_softmax = lambda *args, **kwargs: None
sys.modules["torch.nn"] = torch_nn
sys.modules["torch.nn.functional"] = torch_nn.functional
accelerate_stub = types.ModuleType("accelerate")
accelerate_stub.Accelerator = type("Accelerator", (object,), {})
sys.modules["accelerate"] = accelerate_stub
torch_optim = types.ModuleType("torch.optim")
torch_optim.Optimizer = type("Optimizer", (object,), {})
sys.modules["torch.optim"] = torch_optim
transformers_stub = types.ModuleType("transformers")
transformers_stub.PreTrainedModel = type("PreTrainedModel", (object,), {})
transformers_stub.PreTrainedTokenizer = type("PreTrainedTokenizer", (object,), {})
sys.modules["transformers"] = transformers_stub
trl_stub = types.ModuleType("trl")
trl_stub.ScriptArguments = type("ScriptArguments", (object,), {})
trl_stub.GRPOConfig = type("GRPOConfig", (object,), {})
sys.modules["trl"] = trl_stub

from training.state import (  # noqa: E402  import after stubs
    check_stop_condition,
    load_controller_state_chain,
    maybe_checkpoint,
    maybe_clear_stale_controller_state,
    maybe_load_accelerator_state,
    _checkpoint_log_once,
)
from training.weighting import CONTROLLER_STATE_FILENAME  # noqa: E402


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

    monkeypatch.setattr("training.state._load_controller_file", _fake_load)
    loaded = load_controller_state_chain(controller_cfg, accel, weighting_cfg)
    assert loaded is True
    assert load_calls[0].endswith(CONTROLLER_STATE_FILENAME)


def test_load_controller_file_logs_failure(monkeypatch, caplog):
    caplog.set_level("INFO")
    controller_cfg = SimpleNamespace(
        state_path="missing.bin",
        resume_from=None,
    )
    accel = _Accel()

    monkeypatch.setattr(
        "training.state.load_controller_state", lambda path, weighting: False
    )
    weighting_cfg = SimpleNamespace(beta=0.0, tau=0.0)
    loaded = load_controller_state_chain(controller_cfg, accel, weighting_cfg)
    assert loaded is False


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
        def __init__(self, strategy="steps", save_steps=2):
            self.save_strategy = strategy
            self.save_steps = save_steps

        @staticmethod
        def save_checkpoint(name: str):
            saves.append(name)

    accel = _Accel()
    maybe_checkpoint(_LoggingCfg(), accel, global_step=1)
    assert not saves
    maybe_checkpoint(_LoggingCfg(), accel, global_step=2)
    assert saves == ["checkpoint-2"]


def test_maybe_checkpoint_non_step_strategy_logs_once(monkeypatch, caplog):
    caplog.set_level("INFO")
    accel = _Accel()
    cfg = type(
        "_Cfg",
        (),
        {"save_strategy": "epoch", "save_steps": 0, "save_checkpoint": lambda *_: None},
    )()
    maybe_checkpoint(cfg, accel, global_step=1)
    maybe_checkpoint(cfg, accel, global_step=2)
    assert accel.wait_calls == 4  # before/after each call
    # Only one strategy log should be emitted
    msgs = [
        rec.message for rec in caplog.records if "Skipping checkpoint" in rec.message
    ]
    assert len(msgs) == 1


def test_maybe_checkpoint_zero_save_steps_logs_once(monkeypatch, caplog):
    caplog.set_level("INFO")
    accel = _Accel()
    cfg = type(
        "_Cfg",
        (),
        {"save_strategy": "steps", "save_steps": 0, "save_checkpoint": lambda *_: None},
    )()
    maybe_checkpoint(cfg, accel, global_step=1)
    maybe_checkpoint(cfg, accel, global_step=2)
    msgs = [rec.message for rec in caplog.records if "save_steps<=0" in rec.message]
    assert len(msgs) == 1


def test_check_stop_condition_sets_flag():
    schedule = SimpleNamespace(total_training_steps=10)
    state = SimpleNamespace(global_step=11, stop_training=False)
    check_stop_condition(schedule, state)
    assert state.stop_training is True
