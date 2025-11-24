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

Unit tests for the baseline training pipeline stub.
"""

from __future__ import annotations

from types import ModuleType, SimpleNamespace
import sys


from maxent_grpo.pipelines.training import baseline


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
    monkeypatch.setattr(baseline, "ensure_vllm_group_port", lambda: None)
    # Log to a dummy handler to avoid noisy output during test
    monkeypatch.setattr(baseline.logging, "basicConfig", lambda **_k: None)
    monkeypatch.setattr(baseline.logging, "getLogger", lambda name=None: baseline.logging.Logger(name or "test"))
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


def test_run_baseline_training_fallback_paths(monkeypatch):
    _install_transformers_stub(monkeypatch)
    _install_trl_stub(monkeypatch)
    baseline.GRPOTrainerOverride = None

    raw_ds = {
        "train": [{"prompt": "p", "answer": "a"} for _ in range(5)],
        "validation": [{"prompt": "v", "answer": "b"} for _ in range(20)],
    }

    class _Tokenizer:
        def __init__(self):
            self.pad_token_id = None
            self.eos_token_id = None
            self.pad_token = None
            self.resize_called = False

        def add_special_tokens(self, mapping):
            self.pad_token = mapping["pad_token"]
            self.pad_token_id = 1

        def __len__(self):
            return 10

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            return messages[-1]["content"] + "|chat"

        def __setattr__(self, name, value):
            if name == "padding_side":
                raise AttributeError("immutable")
            return object.__setattr__(self, name, value)

    class _Model:
        def __init__(self):
            self.config = SimpleNamespace(pad_token_id=None, save_pretrained=lambda *_: None)

        def resize_token_embeddings(self, size):
            self.config.pad_token_id = size

    class _Trainer:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.accelerator = SimpleNamespace(is_main_process=True)
            self.model = SimpleNamespace(config=SimpleNamespace(use_cache=False, save_pretrained=lambda *_: None))
            self.train_dataset = kwargs.get("train_dataset")
            self.eval_dataset = kwargs.get("eval_dataset")

        def train(self, resume_from_checkpoint=None):
            self.resume = resume_from_checkpoint
            return SimpleNamespace(metrics={"m": 1})

        def log_metrics(self, split, metrics):
            self.log_history = getattr(self, "log_history", [])
            self.log_history.append((split, metrics))

        def save_metrics(self, split, metrics):
            self.saved_metrics = getattr(self, "saved_metrics", [])
            self.saved_metrics.append((split, metrics))

        def save_state(self):
            self.state_saved = True

        def save_model(self, *args, **kwargs):
            if args:
                raise TypeError("bad args")
            self.model_saved = True

        def create_model_card(self, **kwargs):
            self.model_card = kwargs

        def evaluate(self):
            return {"eval": 1}

        def push_to_hub(self, **kwargs):
            self.pushed = kwargs

    trainer_instance = {"obj": None}

    def _trainer_factory(**kwargs):
        trainer_instance["obj"] = _Trainer(**kwargs)
        return trainer_instance["obj"]

    monkeypatch.setattr(baseline, "get_dataset", lambda args: raw_ds)
    monkeypatch.setattr(baseline, "get_tokenizer", lambda *_: _Tokenizer())
    monkeypatch.setattr(baseline, "get_model", lambda *_: _Model())
    monkeypatch.setattr(baseline, "get_reward_funcs", lambda *_: ["r"])
    monkeypatch.setattr(baseline, "ensure_vllm_group_port", lambda: None)
    baseline.GRPOTrainerOverride = _trainer_factory
    monkeypatch.setattr(baseline.logging, "basicConfig", lambda **_k: None)

    script_args = baseline.GRPOScriptArguments(dataset_name="ds")
    training_args = baseline.GRPOConfig()
    training_args.do_eval = True
    training_args.seed = 0
    training_args.output_dir = "/tmp/out"
    training_args.push_to_hub = True
    training_args.resume_from_checkpoint = "missing_ckpt"
    training_args.get_process_log_level = lambda: 20
    training_args.dataset_train_split = "train"
    training_args.dataset_test_split = None
    model_args = sys.modules["trl"].ModelConfig()

    monkeypatch.setattr(sys.modules["transformers.trainer_utils"], "get_last_checkpoint", lambda *_: None)
    monkeypatch.setattr(baseline.os.path, "isdir", lambda path: False)
    monkeypatch.setattr(baseline.os, "makedirs", lambda *a, **k: None)

    baseline.run_baseline_training(script_args, training_args, model_args)
    trainer = trainer_instance["obj"]
    assert training_args.resume_from_checkpoint is False
    assert trainer.resume is None
    assert trainer.log_history[0][0] == "train"
    assert trainer.saved_metrics[0][0] == "train"
    assert trainer.state_saved is True
    assert trainer.model_saved is True
    assert trainer.model.config.use_cache is True
    assert trainer.pushed["dataset_name"] == "ds"
    # Exercise fallback _Split helpers
    assert trainer.train_dataset.remove_columns("messages") is trainer.train_dataset
    assert trainer.train_dataset.shuffle(seed=training_args.seed) is trainer.train_dataset
    assert trainer.train_dataset.select(range(1)) is trainer.train_dataset


def test_resume_requests_and_messages_removed(monkeypatch, tmp_path):
    _install_transformers_stub(monkeypatch)
    _install_trl_stub(monkeypatch)
    baseline.GRPOTrainerOverride = None

    class _Split(list):
        column_names = ["messages"]

        def remove_columns(self, cols):
            self.removed = cols
            return self

    class _Dataset(dict):
        def map(self, fn):
            mapped = _Dataset()
            mapped["train"] = _Split([fn({"prompt": "p", "answer": "a"})])
            mapped["test"] = _Split([fn({"prompt": "t", "answer": "b"})])
            return mapped

    class _Trainer(SimpleNamespace):
        def train(self, resume_from_checkpoint=None):
            self.resume = resume_from_checkpoint
            return SimpleNamespace(metrics={})

        def save_model(self, *args, **kwargs):
            return None

    trainer_holder = {}

    def _trainer_factory(**kwargs):
        inst = _Trainer(**kwargs)
        trainer_holder["obj"] = inst
        return inst

    baseline.GRPOTrainerOverride = _trainer_factory
    monkeypatch.setattr(baseline, "get_dataset", lambda *_: _Dataset())
    monkeypatch.setattr(baseline, "get_tokenizer", lambda *_: SimpleNamespace(pad_token_id=0, eos_token_id=0))
    monkeypatch.setattr(baseline, "get_model", lambda *_: SimpleNamespace(config=SimpleNamespace(pad_token_id=0)))
    monkeypatch.setattr(baseline, "get_reward_funcs", lambda *_: [])
    monkeypatch.setattr(baseline, "ensure_vllm_group_port", lambda: None)
    monkeypatch.setattr(baseline.os.path, "isdir", lambda path: path == str(tmp_path))
    monkeypatch.setattr(baseline.logging, "basicConfig", lambda **_k: None)
    monkeypatch.setattr(sys.modules["transformers.trainer_utils"], "get_last_checkpoint", lambda path: None)
    monkeypatch.setattr(baseline.os, "makedirs", lambda *a, **k: None)

    script_args = baseline.GRPOScriptArguments(dataset_name="ds")
    training_args = baseline.GRPOConfig()
    training_args.do_eval = False
    training_args.seed = 0
    training_args.output_dir = str(tmp_path)
    training_args.resume_from_checkpoint = True
    training_args.get_process_log_level = lambda: 20
    model_args = sys.modules["trl"].ModelConfig()

    baseline.run_baseline_training(script_args, training_args, model_args)
    assert training_args.resume_from_checkpoint is False
    trainer = trainer_holder["obj"]
    assert isinstance(trainer.train_dataset, list)
    assert getattr(trainer.train_dataset, "removed", None) == "messages"


def test_output_dir_resume(monkeypatch, tmp_path):
    _install_transformers_stub(monkeypatch)
    _install_trl_stub(monkeypatch)

    class _Split(list):
        column_names: list[str] = []

    class _Dataset(dict):
        def map(self, fn):
            mapped = _Dataset()
            split = _Split([])
            mapped["train"] = split
            return mapped

    class _Trainer(SimpleNamespace):
        def train(self, resume_from_checkpoint=None):
            self.resume = resume_from_checkpoint
            return SimpleNamespace(metrics={})

        def save_model(self, *args, **kwargs):
            return None

    baseline.GRPOTrainerOverride = lambda **kwargs: _Trainer(**kwargs)
    monkeypatch.setattr(baseline, "get_dataset", lambda *_: _Dataset())
    monkeypatch.setattr(baseline, "get_tokenizer", lambda *_: SimpleNamespace(pad_token_id=0, eos_token_id=0))
    monkeypatch.setattr(baseline, "get_model", lambda *_: SimpleNamespace(config=SimpleNamespace(pad_token_id=0)))
    monkeypatch.setattr(baseline, "get_reward_funcs", lambda *_: [])
    monkeypatch.setattr(baseline, "ensure_vllm_group_port", lambda: None)
    monkeypatch.setattr(baseline.os.path, "isdir", lambda path: path == str(tmp_path))
    monkeypatch.setattr(sys.modules["transformers.trainer_utils"], "get_last_checkpoint", lambda path: "ckpt-path")
    monkeypatch.setattr(baseline.os, "makedirs", lambda *a, **k: None)
    monkeypatch.setattr(baseline.logging, "basicConfig", lambda **_k: None)

    script_args = baseline.GRPOScriptArguments(dataset_name="ds")
    training_args = baseline.GRPOConfig()
    training_args.do_eval = False
    training_args.seed = 0
    training_args.output_dir = str(tmp_path)
    training_args.resume_from_checkpoint = None
    training_args.get_process_log_level = lambda: 20
    model_args = sys.modules["trl"].ModelConfig()

    baseline.run_baseline_training(script_args, training_args, model_args)
    assert training_args.resume_from_checkpoint == "ckpt-path"


def test_eval_dataset_messages_removed(monkeypatch):
    _install_transformers_stub(monkeypatch)
    _install_trl_stub(monkeypatch)

    class _EvalDS:
        def __init__(self):
            self.removed = False
            self.column_names = ["messages"]

        def remove_columns(self, *_cols):
            self.removed = True
            return self

        def map(self, fn):
            # map should apply fn at least once to exercise mapping path
            fn({"prompt": "e", "answer": "a"})
            return self

    class _Trainer(SimpleNamespace):
        def train(self, resume_from_checkpoint=None):
            return SimpleNamespace(metrics={})

        def save_model(self, *args, **kwargs):
            return None

        def evaluate(self):
            return {"eval": 1}

    eval_ds = _EvalDS()
    baseline.GRPOTrainerOverride = lambda **kwargs: _Trainer(**kwargs)
    monkeypatch.setattr(baseline, "get_dataset", lambda *_: {"train": [{"prompt": "p", "answer": "a"}]})
    monkeypatch.setattr(baseline, "get_tokenizer", lambda *_: SimpleNamespace(pad_token_id=0, eos_token_id=0))
    monkeypatch.setattr(baseline, "get_model", lambda *_: SimpleNamespace(config=SimpleNamespace(pad_token_id=0)))
    monkeypatch.setattr(baseline, "get_reward_funcs", lambda *_: [])
    monkeypatch.setattr(baseline, "ensure_vllm_group_port", lambda: None)
    monkeypatch.setattr(
        baseline,
        "load_dataset_split",
        lambda *args, **kwargs: eval_ds,
    )
    monkeypatch.setattr(baseline.os, "makedirs", lambda *a, **k: None)

    script_args = baseline.GRPOScriptArguments(dataset_name="ds")
    script_args.eval_dataset_name = "eval_ds"
    script_args.eval_dataset_split = "validation"
    training_args = baseline.GRPOConfig()
    training_args.do_eval = True
    training_args.seed = 0
    training_args.output_dir = "/tmp/out"
    training_args.get_process_log_level = lambda: 20
    model_args = sys.modules["trl"].ModelConfig()

    baseline.run_baseline_training(script_args, training_args, model_args)
    assert eval_ds.removed is True


def test_pad_token_uses_eos_when_missing(monkeypatch):
    _install_transformers_stub(monkeypatch)
    _install_trl_stub(monkeypatch)

    class _Tokenizer:
        def __init__(self):
            self.pad_token_id = None
            self.eos_token_id = 5
            self.eos_token = "<eos>"
            self.pad_token = None

        def __setattr__(self, name, value):
            object.__setattr__(self, name, value)

    class _Model:
        def __init__(self):
            self.config = SimpleNamespace(pad_token_id=None)

        def resize_token_embeddings(self, *_args, **_kwargs):
            raise AssertionError("should not resize when eos_token_id is set")

    class _Trainer(SimpleNamespace):
        def train(self, resume_from_checkpoint=None):
            self.resume = resume_from_checkpoint
            return SimpleNamespace(metrics={})

        def save_model(self, *args, **kwargs):
            return None

    baseline.GRPOTrainerOverride = lambda **kwargs: _Trainer(**kwargs)
    monkeypatch.setattr(
        baseline, "get_dataset", lambda *_: {"train": [{"prompt": "p"}], "validation": []}
    )
    tok = _Tokenizer()
    monkeypatch.setattr(baseline, "get_tokenizer", lambda *_: tok)
    monkeypatch.setattr(baseline, "get_model", lambda *_: _Model())
    monkeypatch.setattr(baseline, "get_reward_funcs", lambda *_: [])
    monkeypatch.setattr(baseline, "ensure_vllm_group_port", lambda: None)
    monkeypatch.setattr(baseline.os, "makedirs", lambda *a, **k: None)

    script_args = baseline.GRPOScriptArguments(dataset_name="ds")
    training_args = baseline.GRPOConfig()
    training_args.do_eval = False
    training_args.seed = 0
    training_args.output_dir = "/tmp/out"
    training_args.get_process_log_level = lambda: 20
    model_args = sys.modules["trl"].ModelConfig()

    baseline.run_baseline_training(script_args, training_args, model_args)
    assert tok.pad_token == "<eos>"