"""Unit tests for training top-level shims, CLI helpers, and eval helpers."""

from __future__ import annotations

import importlib
import os
import sys
from types import SimpleNamespace, ModuleType

import pytest


def test_training_init_run_maxent_grpo_raises(training_stubs):
    import training

    with pytest.raises(NotImplementedError):
        training.run_maxent_grpo()
    exported = set(dir(training))
    assert "run_maxent_grpo" in exported


def test_training_cli_reexports_parse_grpo_args(training_stubs):
    import training.cli as cli_pkg
    import training.cli.trl as cli_trl

    assert cli_pkg.parse_grpo_args is cli_trl.parse_grpo_args


def test_parse_grpo_args_uses_recipe_when_provided(monkeypatch, training_stubs):
    import training.cli.trl as cli_trl

    called = {}

    class _ModelConfig:
        pass

    trl_stub = ModuleType("trl")
    trl_stub.ModelConfig = _ModelConfig
    trl_stub.TrlParser = lambda *_: None  # should not be called when recipe provided
    monkeypatch.setitem(sys.modules, "trl", trl_stub)

    def _fake_load(recipe_path, model_config_cls):
        called["path"] = recipe_path
        called["cls"] = model_config_cls
        return ("s", "t", "m")

    monkeypatch.setattr(cli_trl, "load_grpo_recipe", _fake_load)
    assert cli_trl.parse_grpo_args("demo.yaml") == ("s", "t", "m")
    assert called["path"] == "demo.yaml"
    assert called["cls"] is _ModelConfig


def test_parse_grpo_args_defaults_to_trl_parser(monkeypatch, training_stubs):
    import training.cli.trl as cli_trl
    import sys

    class _ModelConfig:
        pass

    class _Parser:
        def __init__(self, types_tuple):
            self.types = types_tuple

        def parse_args_and_config(self):
            return ("s", "t", "m")

    trl_stub = ModuleType("trl")
    trl_stub.ModelConfig = _ModelConfig
    trl_stub.TrlParser = lambda classes: _Parser(classes)
    monkeypatch.setitem(sys.modules, "trl", trl_stub)
    monkeypatch.delenv("GRPO_RECIPE", raising=False)
    assert cli_trl.parse_grpo_args() == ("s", "t", "m")


def test_iter_eval_batches_and_compute_rewards(monkeypatch, training_stubs):
    from training.eval import _iter_eval_batches, _compute_eval_rewards
    from training.types import RewardSpec

    rows = [{"prompt": "p1", "answer": "a1"}, {"prompt": "p2", "answer": "a2"}]
    batches = list(_iter_eval_batches(rows, batch_size=1))
    assert batches == [(["p1"], ["a1"]), (["p2"], ["a2"])]

    def r1(completions, answers):
        return [1.0 for _ in completions]

    def r2(completions, answers):
        return [float(len(a)) for a in answers]

    rewards = _compute_eval_rewards(
        ["c1", "c2"], ["a1", "abc"], RewardSpec(reward_funcs=[r1, r2], reward_weights=[0.5, 1.0])
    )
    assert rewards == [2.5, 3.5]


def test_run_validation_step_logs_and_restores_model(monkeypatch, training_stubs, caplog):
    from training.eval import run_validation_step
    from training.types import RewardSpec, EvaluationSettings, ValidationContext

    class _Model:
        def __init__(self):
            self.training = True
            self.eval_called = False
            self.train_calls = []

        def eval(self):
            self.eval_called = True
            self.training = False

        def train(self, mode=True):
            self.training = mode
            self.train_calls.append(mode)

    def _generator(prompts, n, target_counts):
        return [[p + "_gen"] for p in prompts], None

    def _reward(completions, answers):
        return [1.0 for _ in completions]

    accel = SimpleNamespace(
        num_processes=1,
        process_index=0,
        is_main_process=True,
        gather_object=lambda obj: [obj],
        wait_for_everyone=lambda: caplog.records.append("waited"),  # type: ignore[arg-type]
    )
    model = _Model()
    eval_cfg = EvaluationSettings(enabled=True, rows=[{"prompt": "p", "answer": "a"}], batch_size=1, every_n_steps=None)
    ctx = ValidationContext(
        evaluation=eval_cfg,
        accelerator=accel,
        model=model,
        reward=RewardSpec(reward_funcs=[_reward], reward_weights=[1.0]),
        generator=_generator,
        logging=SimpleNamespace(
            log_metrics=lambda metrics, step: caplog.records.append(("log", metrics, step)),
            metric_writer=SimpleNamespace(log=lambda *_: None, flush=lambda: None),
        ),
    )

    run_validation_step(step=1, ctx=ctx)
    assert model.eval_called is True
    assert model.training is True  # restored to previous state
