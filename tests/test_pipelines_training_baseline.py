"""Unit tests for the baseline training pipeline stub."""

from __future__ import annotations

from types import ModuleType, SimpleNamespace
import sys


from pipelines.training import baseline


def _install_transformers_stub(monkeypatch):
    tf_stub = ModuleType("transformers")
    tf_stub.set_seed = lambda *_a, **_k: None
    tf_stub.trainer_utils = ModuleType("transformers.trainer_utils")
    tf_stub.trainer_utils.get_last_checkpoint = lambda *_a, **_k: None
    tf_stub.utils = SimpleNamespace(
        logging=SimpleNamespace(
            set_verbosity=lambda *_a, **_k: None,
            enable_default_handler=lambda *_a, **_k: None,
            enable_explicit_format=lambda *_a, **_k: None,
        )
    )
    tf_stub.PreTrainedModel = type("PreTrainedModel", (), {})
    tf_stub.PreTrainedTokenizer = type("PreTrainedTokenizer", (), {})
    monkeypatch.setitem(sys.modules, "transformers", tf_stub)
    monkeypatch.setitem(
        sys.modules, "transformers.trainer_utils", tf_stub.trainer_utils
    )
    monkeypatch.setitem(
        sys.modules, "transformers.utils", ModuleType("transformers.utils")
    )


def _install_trl_stub(monkeypatch):
    trl_stub = ModuleType("trl")
    trl_stub.ModelConfig = type("ModelConfig", (), {})
    trl_stub.GRPOTrainer = lambda *args, **kwargs: SimpleNamespace(
        train=lambda *_a, **_k: None
    )  # noqa: E731
    trl_stub.get_peft_config = lambda *_a, **_k: "peft"
    monkeypatch.setitem(sys.modules, "trl", trl_stub)


def test_to_prompt_handles_missing_chat_template(monkeypatch):
    _install_transformers_stub(monkeypatch)
    _install_trl_stub(monkeypatch)
    tokenizer = SimpleNamespace(apply_chat_template=None)
    out = baseline._to_prompt(
        {"prompt": "hi", "answer": "42"}, tokenizer, "prompt", None
    )
    assert "hi" in out["prompt"]
    assert out["answer"] == "42"


def test_run_baseline_training_wires_components(monkeypatch):
    _install_transformers_stub(monkeypatch)
    _install_trl_stub(monkeypatch)

    class _DummySplit(list):
        column_names: list[str] = []

    class _DummyDS(dict):
        def __init__(self):
            super().__init__(train=_DummySplit(), validation=_DummySplit())

        def map(self, fn):
            return self

    monkeypatch.setattr(baseline, "get_dataset", lambda args: _DummyDS())
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
    train_called = {}

    class _Trainer:
        def __init__(self, **kwargs):
            train_called["kwargs"] = kwargs

        def train(self, resume_from_checkpoint=None):
            train_called["resume"] = resume_from_checkpoint
            return SimpleNamespace(metrics={"foo": 1})

        def save_model(self, *args, **kwargs):
            train_called["saved"] = True

        def log_metrics(self, split, metrics):
            train_called["log"] = (split, metrics)

        def save_metrics(self, split, metrics):
            train_called["saved_metrics"] = (split, metrics)

        def save_state(self):
            train_called["state_saved"] = True

    trl_stub = sys.modules["trl"]
    trl_stub.GRPOTrainer = _Trainer  # type: ignore[attr-defined]
    monkeypatch.setattr(baseline, "ensure_vllm_group_port", lambda: None)

    script_args = baseline.GRPOScriptArguments(dataset_name="ds")
    training_args = baseline.GRPOConfig()
    training_args.output_dir = "/tmp/out"
    training_args.do_eval = False
    training_args.seed = 0
    training_args.get_process_log_level = lambda: 20
    model_args = trl_stub.ModelConfig()

    baseline.run_baseline_training(script_args, training_args, model_args)
    assert train_called["kwargs"]["train_dataset"] == []
    assert train_called["saved"] is True
