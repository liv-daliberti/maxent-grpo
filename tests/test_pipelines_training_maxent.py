"""
Unit tests for the MaxEnt training pipeline shim.
"""

from __future__ import annotations

from types import SimpleNamespace
import sys

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
    model_args = SimpleNamespace()
    result = maxent.run_maxent_training(script_args, training_args, model_args)
    assert called["args"] == (script_args, training_args, model_args)
    assert result == called["args"]


def test_run_maxent_training_respects_flags(monkeypatch):
    called = {}
    monkeypatch.setattr(
        maxent,
        "_run_baseline_training",
        lambda *args, **kwargs: called.setdefault("args", args),
    )
    monkeypatch.setattr(
        "maxent_grpo.pipelines.training.GRPOTrainerOverride", None, raising=False
    )
    script_args = GRPOScriptArguments(dataset_name="ds", dataset_prompt_column="prompt")
    training_args = maxent.GRPOConfig()
    training_args.train_grpo_objective = True
    model_args = SimpleNamespace()
    maxent.run_maxent_training(script_args, training_args, model_args)
    assert called["args"] == (script_args, training_args, model_args)


def test_run_maxent_training_installs_override(monkeypatch):
    _install_transformers_stub(monkeypatch)
    trl_stub = _install_trl_stub(monkeypatch)

    raw_ds = {"train": [{"prompt": "p", "answer": "a"}]}
    monkeypatch.setattr(baseline, "get_dataset", lambda *_: raw_ds)
    monkeypatch.setattr(
        baseline,
        "get_tokenizer",
        lambda *_a, **_k: SimpleNamespace(
            eos_token_id=0,
            pad_token_id=0,
            padding_side="right",
            add_special_tokens=lambda *_a, **_k: None,
            __setattr__=object.__setattr__,
        ),
    )
    monkeypatch.setattr(
        baseline,
        "get_model",
        lambda *_a, **_k: SimpleNamespace(config=SimpleNamespace()),
    )
    monkeypatch.setattr(baseline, "get_reward_funcs", lambda *_a, **_k: [])
    monkeypatch.setattr(baseline, "ensure_vllm_group_port", lambda: None)
    monkeypatch.setattr(baseline.logging, "basicConfig", lambda **_k: None)
    monkeypatch.setattr(
        baseline.logging,
        "getLogger",
        lambda name=None: baseline.logging.Logger(name or "test"),
    )
    monkeypatch.setattr(baseline.os.path, "isdir", lambda path: False)
    monkeypatch.setattr(baseline.os, "makedirs", lambda *a, **k: None)

    script_args = GRPOScriptArguments(dataset_name="ds")
    training_args = GRPOConfig()
    training_args.train_grpo_objective = False
    training_args.do_eval = False
    training_args.seed = 0
    training_args.output_dir = "/tmp/out"
    training_args.get_process_log_level = lambda: 20
    model_args = trl_stub.ModelConfig()

    maxent.run_maxent_training(script_args, training_args, model_args)
    trainer = trl_stub._created[-1]
    assert trainer.__class__.__name__ == "MaxEntGRPOTrainer"
    assert getattr(trainer, "maxent_enabled", False) is True
