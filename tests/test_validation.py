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

from types import ModuleType, SimpleNamespace
import sys
import logging

import grpo
from maxent_grpo.pipelines.training import baseline
import maxent_grpo.core.data as data_utils

try:
    import transformers.trainer_utils as trainer_utils  # type: ignore
except ModuleNotFoundError:
    # Provide a lightweight stub when transformers is not installed
    transformers_mod = sys.modules.setdefault(
        "transformers", ModuleType("transformers")
    )
    trainer_utils = ModuleType("transformers.trainer_utils")
    trainer_utils.get_last_checkpoint = lambda *_args, **_kwargs: None
    sys.modules["transformers.trainer_utils"] = trainer_utils
    transformers_mod.trainer_utils = trainer_utils


class FakeSplit:
    """Minimal stand-in for a HF dataset split."""

    def __init__(self, rows):
        self.rows = [dict(row) for row in rows]
        self._refresh_columns()

    def _refresh_columns(self):
        self.column_names = sorted(self.rows[0].keys()) if self.rows else []

    def map(self, fn):
        return FakeSplit([fn(dict(row)) for row in self.rows])

    def remove_columns(self, name):
        for row in self.rows:
            row.pop(name, None)
        self._refresh_columns()
        return self

    def shuffle(self, seed=0):
        _ = seed
        return self

    def select(self, rng):
        idxs = list(rng)
        return FakeSplit([self.rows[i] for i in idxs])

    def __len__(self):
        return len(self.rows)


class FakeDatasetDict(dict):
    """DatasetDict shim exposing .map like HF does."""

    def map(self, fn):
        return FakeDatasetDict({split: ds.map(fn) for split, ds in self.items()})


class DummyTokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def __init__(self):
        self.padding_side = "right"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        _ = (tokenize, add_generation_prompt)
        return " | ".join(m["content"] for m in messages)

    def add_special_tokens(self, tokens):
        if "pad_token" in tokens:
            self.pad_token_id = 1


class DummyModel:
    def __init__(self):
        self.config = SimpleNamespace(
            pad_token_id=None,
            use_cache=False,
            save_pretrained=lambda *args, **kwargs: None,
        )

    def resize_token_embeddings(self, *_args, **_kwargs):
        return None

    def save_pretrained(self, *_args, **_kwargs):
        return None

    def train(self):
        return None

    def eval(self):
        return None


class DummyTrainer:
    last_instance = None
    """Capture trainer construction while mimicking the TRL interface."""

    def __init__(
        self,
        *,
        model,
        reward_funcs,
        args,
        train_dataset,
        eval_dataset,
        peft_config,
        processing_class,
    ):
        _ = (reward_funcs, args, peft_config, processing_class)
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.accelerator = SimpleNamespace(is_main_process=True)
        self.logged = []
        DummyTrainer.last_instance = self

    def train(self, resume_from_checkpoint=None):
        _ = resume_from_checkpoint
        return SimpleNamespace(metrics={"loss": 0.0})

    def log_metrics(self, *args, **kwargs):
        self.logged.append(("log", args, kwargs))

    def save_metrics(self, *args, **kwargs):
        self.logged.append(("save", args, kwargs))

    def save_state(self):
        return None

    def save_model(self, *_args, **_kwargs):
        return None

    def create_model_card(self, *args, **_kwargs):
        self.logged.append(("card", args))

    def evaluate(self):
        return {"eval_loss": 0.0}

    def push_to_hub(self, *args, **kwargs):
        self.logged.append(("push", args, kwargs))


class DummyTrainingArgs(SimpleNamespace):
    def get_process_log_level(self):
        return logging.INFO


def test_load_dataset_split_invokes_hf(monkeypatch):
    calls = {}

    def fake_load_dataset(name, config=None, split=None):
        calls["args"] = (name, config, split)
        return ["ok"]

    monkeypatch.setattr(
        data_utils, "datasets", SimpleNamespace(load_dataset=fake_load_dataset)
    )

    out = data_utils.load_dataset_split("repo/ds", "cfg", "test")
    assert out == ["ok"]
    assert calls["args"] == ("repo/ds", "cfg", "test")


def test_grpo_main_prefers_dedicated_eval_dataset(monkeypatch):
    raw_ds = FakeDatasetDict(
        {
            "train": FakeSplit([{"problem": "train question", "answer": "42"}]),
        }
    )
    eval_calls = {}

    def fake_get_dataset(_args):
        return raw_ds

    def fake_load_eval(name, config, split):
        eval_calls["args"] = (name, config, split)
        return FakeSplit([{"problem": "eval question", "answer": "7"}])

    monkeypatch.setattr(baseline, "get_dataset", fake_get_dataset)
    monkeypatch.setattr(baseline, "load_dataset_split", fake_load_eval)
    monkeypatch.setattr(
        baseline, "get_tokenizer", lambda *args, **kwargs: DummyTokenizer()
    )
    monkeypatch.setattr(baseline, "get_model", lambda *args, **kwargs: DummyModel())
    monkeypatch.setattr(
        baseline,
        "get_reward_funcs",
        lambda *args, **kwargs: [lambda comps, answers: [1.0] * len(comps)],
    )
    monkeypatch.setattr(baseline, "ensure_vllm_group_port", lambda: None)
    monkeypatch.setattr(
        baseline.transformers, "set_seed", lambda *_args, **_kwargs: None
    )
    dummy_tf_logging = SimpleNamespace(
        set_verbosity=lambda *args, **kwargs: None,
        enable_default_handler=lambda *args, **kwargs: None,
        enable_explicit_format=lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        baseline.transformers,
        "utils",
        SimpleNamespace(logging=dummy_tf_logging),
    )
    monkeypatch.setattr(
        trainer_utils, "get_last_checkpoint", lambda *_args, **_kwargs: None
    )
    dummy_trl = SimpleNamespace(
        GRPOTrainer=DummyTrainer,
        get_peft_config=lambda *args, **kwargs: None,
    )
    monkeypatch.setitem(sys.modules, "trl", dummy_trl)
    # Ensure baseline uses the dummy trainer regardless of previous overrides.
    baseline.GRPOTrainerOverride = dummy_trl.GRPOTrainer

    script_args = SimpleNamespace(
        dataset_name="train/ds",
        dataset_prompt_column="problem",
        dataset_solution_column="answer",
        dataset_train_split="train",
        eval_dataset_name="hf/math500",
        eval_dataset_config=None,
        eval_dataset_split="test",
        eval_dataset_prompt_column="problem",
        eval_dataset_solution_column="answer",
    )
    training_args = DummyTrainingArgs(
        seed=0,
        system_prompt="SYS",
        do_eval=True,
        return_reward=True,
        resume_from_checkpoint=False,
        output_dir="tmp-out",
        push_to_hub=False,
        benchmarks=[],
    )
    model_args = SimpleNamespace()

    grpo.main(script_args, training_args, model_args)

    assert eval_calls["args"] == ("hf/math500", None, "test")

    # Dummy trainer should have seen the fully mapped eval dataset.
    captured_eval = DummyTrainer.last_instance.eval_dataset
    assert isinstance(captured_eval, FakeSplit)
    assert captured_eval.rows[0]["prompt"].startswith("SYS | eval question")
    assert captured_eval.rows[0]["answer"] == "7"
