"""
Unit tests for the MaxEnt training pipeline shim.
"""

from __future__ import annotations

from types import SimpleNamespace
import sys

import pytest

from maxent_grpo.config import GRPOConfig, GRPOScriptArguments
from maxent_grpo.pipelines.training import baseline
from maxent_grpo.pipelines.training import maxent


def _install_transformers_stub(monkeypatch):
    tf_stub = SimpleNamespace(
        set_seed=lambda *_a, **_k: None,
        trainer_utils=SimpleNamespace(get_last_checkpoint=lambda *_a, **_k: None),
        utils=SimpleNamespace(
            logging=SimpleNamespace(
                set_verbosity=lambda *_a, **_k: None,
                enable_default_handler=lambda *_a, **_k: None,
                enable_explicit_format=lambda *_a, **_k: None,
            )
        ),
        PreTrainedModel=type("PreTrainedModel", (), {}),
        PreTrainedTokenizer=type("PreTrainedTokenizer", (), {}),
    )
    monkeypatch.setitem(sys.modules, "transformers", tf_stub)
    monkeypatch.setitem(
        sys.modules, "transformers.trainer_utils", tf_stub.trainer_utils
    )
    monkeypatch.setitem(sys.modules, "transformers.utils", tf_stub.utils)


def _install_trl_stub(monkeypatch):
    created = []

    class _GRPOTrainer:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.args = kwargs.get("args")
            self.accelerator = SimpleNamespace(is_main_process=True)
            created.append(self)

        def train(self, resume_from_checkpoint=None):
            self.resume = resume_from_checkpoint
            return SimpleNamespace(metrics={"foo": 1})

        def save_model(self, *args, **kwargs):
            return None

        def log_metrics(self, split, metrics):
            self.logged = (split, metrics)

        def save_metrics(self, split, metrics):
            self.saved = (split, metrics)

        def save_state(self):
            self.state_saved = True

    trl_stub = SimpleNamespace(
        ModelConfig=type("ModelConfig", (), {}),
        GRPOTrainer=_GRPOTrainer,
        get_peft_config=lambda *_a, **_k: "peft",
        _created=created,
    )
    monkeypatch.setitem(sys.modules, "trl", trl_stub)
    return trl_stub


def test_to_prompt_handles_missing_chat_template(monkeypatch):
    _install_transformers_stub(monkeypatch)
    _install_trl_stub(monkeypatch)
    tokenizer = SimpleNamespace(apply_chat_template=None)
    out = maxent._to_prompt({"prompt": "hi", "answer": "42"}, tokenizer, "prompt", None)
    assert "hi" in out["prompt"]
    assert out["answer"] == "42"


def test_run_maxent_training_wires_baseline_call(monkeypatch):
    called = {}
    monkeypatch.setattr(
        maxent,
        "_run_baseline_training",
        lambda *args, **kwargs: called.setdefault("args", args),
    )
    monkeypatch.setattr(
        "maxent_grpo.pipelines.training.GRPOTrainerOverride", None, raising=False
    )
    script_args = GRPOScriptArguments(dataset_name="ds")
    training_args = GRPOConfig()
    training_args.train_grpo_objective = True
    model_args = SimpleNamespace()
    result = maxent.run_maxent_training(script_args, training_args, model_args)
    assert called["args"] == (script_args, training_args, model_args)
    assert result == called["args"]


@pytest.mark.parametrize(
    "train_grpo,info_seed_enabled",
    [
        (True, False),
        (True, True),
        (False, False),
        (False, True),
    ],
)
def test_run_maxent_training_pipeline_matrix(
    monkeypatch, train_grpo, info_seed_enabled
):
    baseline_calls = []
    monkeypatch.setattr(
        maxent,
        "_run_baseline_training",
        lambda *args, **kwargs: baseline_calls.append((args, kwargs)),
    )
    builder_calls = []
    ctx = SimpleNamespace()

    def _builder(*args, **kwargs):
        builder_calls.append((args, kwargs))
        return ctx

    loop_calls = []
    monkeypatch.setattr(maxent, "build_training_loop_context", _builder)
    monkeypatch.setattr(
        maxent,
        "run_training_loop",
        lambda context: loop_calls.append(context),
    )
    script_args = GRPOScriptArguments(dataset_name="ds")
    training_args = GRPOConfig()
    training_args.train_grpo_objective = train_grpo
    training_args.info_seed_enabled = info_seed_enabled
    training_args.get_process_log_level = lambda: 20
    model_args = SimpleNamespace()

    maxent.run_maxent_training(script_args, training_args, model_args)

    if train_grpo:
        assert baseline_calls and not builder_calls and not loop_calls
        assert baseline_calls[0][0] == (script_args, training_args, model_args)
    else:
        assert not baseline_calls
        assert builder_calls and loop_calls
        args, kwargs = builder_calls[0]
        assert args == (script_args, training_args, model_args)
        assert kwargs == {
            "deps_namespace": "maxent",
            "apply_info_seed_cfg": True,
            "force_grpo_objective": None,
        }
        assert loop_calls[0] is ctx


def test_run_maxent_training_with_meta_forces_custom_loop(monkeypatch):
    baseline_calls = []
    monkeypatch.setattr(
        maxent,
        "_run_baseline_training",
        lambda *args, **kwargs: baseline_calls.append((args, kwargs)),
    )
    builder_calls = []
    ctx = SimpleNamespace()

    def _builder(*args, **kwargs):
        builder_calls.append((args, kwargs))
        return ctx

    loop_calls = []
    monkeypatch.setattr(maxent, "build_training_loop_context", _builder)
    monkeypatch.setattr(maxent, "run_training_loop", lambda context: loop_calls.append(context))

    script_args = GRPOScriptArguments(dataset_name="ds")
    training_args = GRPOConfig()
    training_args.train_grpo_objective = True
    training_args.controller_meta_enabled = True
    training_args.get_process_log_level = lambda: 20
    model_args = SimpleNamespace()

    maxent.run_maxent_training(script_args, training_args, model_args)

    assert not baseline_calls
    assert builder_calls and loop_calls
    args, kwargs = builder_calls[0]
    assert args == (script_args, training_args, model_args)
    assert kwargs == {
        "deps_namespace": "maxent",
        "apply_info_seed_cfg": True,
        "force_grpo_objective": True,
    }
    assert loop_calls[0] is ctx
