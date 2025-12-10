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

import json
from contextlib import nullcontext
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

# Stub out training.optim to avoid importing the real module with heavy deps.
optim_stub = types.ModuleType("maxent_grpo.training.optim")
optim_stub.configure_accumulation_steps = (
    lambda *args, **kwargs: types.SimpleNamespace()
)
optim_stub.detect_deepspeed_state = lambda *_a, **_k: types.SimpleNamespace(
    use_deepspeed=False, zero_stage=0
)
optim_stub.epoch_progress = lambda *_a, **_k: 0.0
optim_stub.optimizer_step = lambda *_a, **_k: None
optim_stub.require_accumulation_context = lambda *_a, **_k: nullcontext()
optim_stub.scheduled_learning_rate = lambda *_a, **_k: 0.0
optim_stub.sync_gradients_enabled = lambda *_a, **_k: False
def _build_handles(_model=None, training_args=None):
    lr = float(getattr(training_args, "learning_rate", 0.0)) if training_args else 0.0
    optimizer = SimpleNamespace(
        step=lambda *_a, **_k: None,
        zero_grad=lambda **_k: None,
    )
    return SimpleNamespace(
        optimizer=optimizer,
        lr_scheduler=None,
        base_optimizer=optimizer,
        learning_rate=lr,
    )


optim_stub.build_optimization_handles = _build_handles
sys.modules["maxent_grpo.training.optim"] = optim_stub

trl_stub = types.ModuleType("trl")
trl_stub.ScriptArguments = type("ScriptArguments", (object,), {})
trl_stub.GRPOConfig = type("GRPOConfig", (object,), {})
sys.modules["trl"] = trl_stub

import maxent_grpo.training.metrics as metrics_mod  # noqa: E402
from maxent_grpo.training.state import (  # noqa: E402  import after stubs
    check_stop_condition,
    load_controller_state_chain,
    maybe_checkpoint,
    maybe_clear_stale_controller_state,
    maybe_load_accelerator_state,
    _checkpoint_log_once,
)
from maxent_grpo.training.types import LoggingHandles, LogStepArtifacts  # noqa: E402
from maxent_grpo.training.weighting import CONTROLLER_STATE_FILENAME  # noqa: E402
from maxent_grpo.training.weighting.logic import save_controller_state  # noqa: E402


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


def test_maybe_clear_stale_controller_state_returns_when_missing_file(tmp_path):
    controller_cfg = SimpleNamespace(
        resume_from=None,
        overwrite_existing=True,
        state_path=str(tmp_path / "nope.bin"),
    )
    accel = _Accel()
    maybe_clear_stale_controller_state(accel, controller_cfg)
    assert accel.wait_calls == 0  # early return before synchronization


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

    monkeypatch.setattr("maxent_grpo.training.state._load_controller_file", _fake_load)
    loaded = load_controller_state_chain(controller_cfg, accel, weighting_cfg)
    assert loaded is True


def test_load_controller_file_logs_failure(monkeypatch, caplog):
    caplog.set_level("INFO")
    controller_cfg = SimpleNamespace(
        state_path="missing.bin",
        resume_from=None,
    )
    accel = _Accel()

    monkeypatch.setattr(
        "maxent_grpo.training.state.load_controller_state",
        lambda path, weighting: False,
    )
    weighting_cfg = SimpleNamespace(beta=0.0, tau=0.0)
    loaded = load_controller_state_chain(controller_cfg, accel, weighting_cfg)
    assert loaded is False


def test__load_controller_file_logs_success(tmp_path, caplog):
    caplog.set_level("INFO")
    accel = _Accel()
    weighting_cfg = SimpleNamespace(
        beta=0.1,
        tau=0.2,
        train_grpo_objective=False,
        denom=1.0,
    )
    controller_path = tmp_path / "controller.bin"
    controller_path.write_text(
        json.dumps({"beta": 0.33, "tau": 0.44, "tau_log": 0.0}), encoding="utf-8"
    )
    from maxent_grpo.training import state as state_mod

    loaded = state_mod._load_controller_file(str(controller_path), accel, weighting_cfg)
    assert loaded is True
    assert "Loaded controller state from" in caplog.text
    assert weighting_cfg.beta == pytest.approx(0.33)
    assert weighting_cfg.tau == pytest.approx(0.44)


def test_load_controller_state_chain_marks_resume_even_when_missing(
    monkeypatch, tmp_path
):
    state_dir = tmp_path / "resume"
    state_dir.mkdir()
    controller_cfg = SimpleNamespace(
        resume_from=str(state_dir),
        overwrite_existing=False,
        state_path=str(tmp_path / "current.bin"),
    )
    accel = _Accel()
    weighting_cfg = SimpleNamespace(beta=0.0, tau=0.0)
    calls = []

    def _fake_load(path, *_args):
        calls.append(path)
        return False

    monkeypatch.setattr("maxent_grpo.training.state._load_controller_file", _fake_load)
    loaded = load_controller_state_chain(controller_cfg, accel, weighting_cfg)
    assert loaded is True  # resume path preferred even when load returns False


def test_controller_resume_influences_initial_metrics(monkeypatch, tmp_path):
    resume_dir = tmp_path / "resume"
    resume_dir.mkdir()
    resume_file = resume_dir / CONTROLLER_STATE_FILENAME
    saved_weighting = SimpleNamespace(
        tau=0.55,
        beta=0.09,
        denom=1.0,
        q_temperature=1.0,
        q_epsilon=1e-6,
        tau_lr=0.01,
        tau_min=0.05,
        tau_max=1.0,
        tau_warmup_steps=0,
        tau_target_entropy=0.6,
        kl_target=0.1,
        kl_horizon=10,
        kl_ctl_step_size=1.0,
        len_norm_ref=False,
        train_grpo_objective=False,
        _tau_entropy_ema=0.2,
    )
    save_controller_state(str(resume_file), saved_weighting)
    weighting_cfg = SimpleNamespace(
        tau=0.2,
        beta=0.04,
        denom=1.0,
        q_temperature=1.0,
        q_epsilon=1e-6,
        tau_lr=0.01,
        tau_min=0.05,
        tau_max=1.0,
        tau_warmup_steps=0,
        tau_target_entropy=0.6,
        kl_target=0.1,
        kl_horizon=10,
        kl_ctl_step_size=1.0,
        len_norm_ref=False,
        train_grpo_objective=False,
        _tau_entropy_ema=0.2,
    )
    controller_cfg = SimpleNamespace(
        resume_from=str(resume_dir),
        overwrite_existing=False,
        state_path=str(tmp_path / "controller/current" / CONTROLLER_STATE_FILENAME),
    )
    accel = _Accel()
    loaded = load_controller_state_chain(controller_cfg, accel, weighting_cfg)
    assert loaded is True
    assert weighting_cfg.tau == pytest.approx(saved_weighting.tau)
    assert weighting_cfg.beta == pytest.approx(saved_weighting.beta)

    class _Writer:
        def __init__(self):
            self.logged = []

        def log(self, metrics, step):
            self.logged.append((metrics, step))

        def flush(self):
            return None

    writer = _Writer()
    logging_handles = LoggingHandles(
        metric_writer=writer,
        save_checkpoint=lambda *_a, **_k: None,
        save_strategy="no",
        save_steps=0,
        wandb_run=None,
    )
    ctx = SimpleNamespace(
        runtime=SimpleNamespace(
            accelerator=_Accel(),
            device="cpu",
        ),
        logging=logging_handles,
        scoring=SimpleNamespace(
            weighting=weighting_cfg,
            clipping=SimpleNamespace(),
        ),
        optimization=SimpleNamespace(
            schedule=SimpleNamespace(steps_per_epoch=1),
        ),
        generation=SimpleNamespace(use_vllm=False, generation_stats={}),
    )
    reward_comp = SimpleNamespace(
        total_utils=[0.2, 0.4],
        advantage_samples=[0.0, 0.1],
        per_reward_values={"accuracy": [1.0, 0.0]},
        advantage=SimpleNamespace(grouped=[[0.0], [0.1]]),
        q_grouped=[[0.5], [0.6]],
        pairs=SimpleNamespace(prompts=["p1", "p2"], completions=["c1", "c2"]),
    )
    prepared = SimpleNamespace(
        reward_comp=reward_comp,
        weight_stats=SimpleNamespace(
            entropy=0.4,
            entropy_min=0.38,
            entropy_max=0.42,
            advantage_entropy_mean=0.0,
            advantage_entropy_std=0.0,
        ),
        length_stats=SimpleNamespace(
            min_length=2.0,
            mean_length=3.0,
            max_length=4.0,
            clipped_ratio=0.0,
            min_terminated=2.0,
            mean_terminated=2.5,
            max_terminated=3.0,
        ),
        ref_stats=SimpleNamespace(ref_logp_mean=-0.3, avg_completion_tokens=2.0),
        num_completion_tokens=2.0,
        seed_metrics={},
    )
    loss_outputs = SimpleNamespace(
        total_loss_scalar=0.5,
        kl_loss_scalar=0.03,
        policy_loss_scalar=0.47,
        weighted_kl_loss_scalar=0.03,
        clip_loss_scalar=None,
        scalars=SimpleNamespace(kl_loss=0.03),
    )
    diagnostics = metrics_mod.BatchDiagnostics(
        kl_value=0.03,
        clip_ratio=0.0,
        clip_ratio_low_mean=0.0,
        clip_ratio_low_min=0.0,
        clip_ratio_high_mean=0.0,
        clip_ratio_high_max=0.0,
        clip_ratio_region_mean=0.0,
        kl_per_token_by_len_bucket={},
        kl_token_count_by_len_bucket={},
    )
    log_artifacts = LogStepArtifacts(
        loss_outputs=loss_outputs,
        diagnostics=diagnostics,
        grad_norm_scalar=None,
        epoch_progress=0.0,
    )
    state = SimpleNamespace(
        global_step=1,
        metric_sums={},
        metric_counts={},
        num_input_tokens_seen=5.0,
    )

    metrics_mod.log_training_step(ctx, state, prepared, log_artifacts, current_lr=1e-4)
    assert writer.logged, "resume metrics should be logged"
    logged_metrics = writer.logged[-1][0]
    assert logged_metrics["train/tau"] == pytest.approx(saved_weighting.tau)
    assert logged_metrics["train/beta"] == pytest.approx(saved_weighting.beta)


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


def test_maybe_checkpoint_normalizes_prefixed_strategy():
    saves = []

    class _Cfg:
        def __init__(self):
            self.save_strategy = "SaveStrategy.steps"
            self.save_steps = 1

        def save_checkpoint(self, name: str):
            saves.append(name)

    accel = _Accel()
    maybe_checkpoint(_Cfg(), accel, global_step=1)
    assert saves == ["checkpoint-1"]
    # wait_for_everyone called before and after save
    assert accel.wait_calls == 2


def test_check_stop_condition_sets_flag():
    schedule = SimpleNamespace(total_training_steps=10)
    state = SimpleNamespace(global_step=11, stop_training=False)
    check_stop_condition(schedule, state)
    assert state.stop_training is True
