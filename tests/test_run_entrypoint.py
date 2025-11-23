"""Tests for run_maxent_grpo helpers."""

from __future__ import annotations

from importlib import import_module, reload
from types import SimpleNamespace

import pytest

from tests.test_run_setup_reference import _load_run_setup


@pytest.fixture
def run_mod(monkeypatch):
    """Import training.run with torch/accelerate stubs applied."""
    _load_run_setup(monkeypatch)
    return reload(import_module("training.run"))


def _ns(report_to):
    return SimpleNamespace(report_to=report_to)


def test_sync_report_to_from_script_args_backfills_none(run_mod):
    script_args = _ns(["wandb"])
    training_args = _ns(None)
    run_mod._sync_report_to_from_script_args(script_args, training_args)
    assert training_args.report_to == ["wandb"]


def test_sync_report_to_does_not_override_existing_training_value(run_mod):
    script_args = _ns(["wandb"])
    training_args = _ns(["tensorboard"])
    run_mod._sync_report_to_from_script_args(script_args, training_args)
    assert training_args.report_to == ["tensorboard"]


def test_sync_report_to_ignores_script_none(run_mod):
    script_args = _ns("none")
    training_args = _ns(None)
    run_mod._sync_report_to_from_script_args(script_args, training_args)
    assert training_args.report_to is None


def test_run_maxent_grpo_smoke(monkeypatch, run_mod, caplog, tmp_path):
    """run_maxent_grpo should build the context and invoke the training loop."""

    class _FakeAccel:
        def __init__(self):
            self.device = "cpu"
            self.is_main_process = True

        def wait_for_everyone(self):
            wait_calls.append("wait")

    wait_calls = []
    fake_setup = SimpleNamespace(
        accelerator=_FakeAccel(),
        checkpoint=SimpleNamespace(
            output_dir=str(tmp_path),
            save_strategy="steps",
            save_steps=1,
            save_total_limit=2,
        ),
        hyperparams=SimpleNamespace(num_epochs=2, grad_accum_steps=1),
        train_data=SimpleNamespace(
            steps_per_epoch=4,
            train_loader="loader",
            train_sampler="sampler",
        ),
        model_bundle=SimpleNamespace(
            model="model",
            tokenizer="tokenizer",
            get_ref_model=lambda: "ref_model",
        ),
    )
    components = SimpleNamespace(
        generation="gen_cfg",
        evaluation="eval_cfg",
        optimization=SimpleNamespace(
            schedule=SimpleNamespace(total_training_steps=8),
            handles=SimpleNamespace(lr_scheduler="lr_sched"),
        ),
        scoring="scoring_cfg",
        reward="reward_cfg",
        wandb_config={"project": "maxent"},
        lr_warmup_steps=1,
    )
    monkeypatch.setattr(run_mod, "bootstrap_runner", lambda *_: fake_setup)
    monkeypatch.setattr(run_mod, "build_training_components", lambda *_: components)
    monkeypatch.setattr(
        run_mod, "_copy_initial_model_snapshot", lambda *_: snapshot_calls.append(True)
    )
    monkeypatch.setattr(
        run_mod, "_maybe_wait_for_all", lambda accel: wait_calls.append(accel)
    )

    class _FakeRun:
        def __init__(self):
            self.logged = []
            self.name = "demo"
            self.project = "proj"

        def log(self, metrics, step=None):
            self.logged.append((metrics, step))

    fake_run = _FakeRun()
    monkeypatch.setattr(run_mod, "_maybe_init_wandb_run", lambda *_, **__: fake_run)

    checkpoint_instances = []

    class _FakeCheckpointManager:
        def __init__(self, handles, cfg, state):
            self.handles = handles
            self.cfg = cfg
            self.state = state
            self.saved = []
            self.finalized = False
            checkpoint_instances.append(self)

        def save(self, label):
            self.saved.append(label)

        def finalize(self):
            self.finalized = True

    class _FakeHandles:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    monkeypatch.setattr(run_mod, "CheckpointHandles", _FakeHandles)
    monkeypatch.setattr(run_mod, "CheckpointManager", _FakeCheckpointManager)

    training_calls = []
    monkeypatch.setattr(
        run_mod, "run_training_loop", lambda ctx: training_calls.append(ctx)
    )

    snapshot_calls = []
    caplog.set_level("INFO")
    script_args = SimpleNamespace(report_to=None)
    training_args = SimpleNamespace(
        report_to=None,
        logging_first_step=True,
        logging_strategy="steps",
        logging_steps=5,
        resume_from_checkpoint=None,
        overwrite_output_dir=False,
    )
    run_mod.run_maxent_grpo(script_args, training_args, SimpleNamespace())

    assert training_calls, "Training loop was not invoked"
    ctx = training_calls[0]
    assert ctx.runtime.model == "model"
    assert ctx.reward == "reward_cfg"
    assert checkpoint_instances and checkpoint_instances[0].finalized
    assert fake_run.logged and fake_run.logged[0][0] == {"train/learning_rate": 0.0}
    assert wait_calls, "Accelerator synchronization not triggered"
    assert "Plan: epochs=2" in caplog.text
