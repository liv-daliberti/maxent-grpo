#!/usr/bin/env python
"""
Lightweight CLI smoke check for training and inference entrypoints.

This script stubs heavy dependencies (torch/transformers/trl) and exercises:
- GRPO training entrypoint (src/grpo.py -> pipelines.training.baseline)
- MaxEnt-GRPO entrypoint (src/maxent-grpo.py)
- math_500 inference runner (pipelines.inference.math500)

The goal is to catch wiring/packaging breakage without downloading models or
datasets. It should run quickly on CPU-only CI runners.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Dict, Iterable, List


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _install_trl_stub() -> None:
    """Provide a minimal trl stub so configs import cleanly without GPU deps."""
    trl = ModuleType("trl")

    class _ScriptArguments:
        def __init__(self, *args, **kwargs):
            for key, val in kwargs.items():
                setattr(self, key, val)

    class _GRPOConfig:
        def __init__(self, *args, **kwargs):
            for key, val in kwargs.items():
                setattr(self, key, val)

    class _ModelConfig:
        def __init__(self, *args, **kwargs):
            for key, val in kwargs.items():
                setattr(self, key, val)

    class _Parser:
        def __init__(self, classes):
            self.classes = classes

        def parse_args_and_config(self):
            return (
                _ScriptArguments(dataset_name="dummy"),
                _GRPOConfig(),
                _ModelConfig(),
            )

    class _Trainer:
        def __init__(self, **kwargs):
            self.model = kwargs.get("model")
            self.processing_class = kwargs.get("processing_class")
            self.accelerator = SimpleNamespace(is_main_process=True)

        def train(self, resume_from_checkpoint=None):
            return SimpleNamespace(metrics={"train_steps": 1, "loss": 0.0})

        def log_metrics(self, *_args, **_kwargs):
            return None

        def save_metrics(self, *_args, **_kwargs):
            return None

        def save_state(self):
            return None

        def save_model(self, *_args, **_kwargs):
            return None

        def evaluate(self):
            return {"eval_loss": 0.0}

        def create_model_card(self, **_kwargs):
            return None

        def push_to_hub(self, **_kwargs):
            return None

    trl.ScriptArguments = _ScriptArguments
    trl.GRPOConfig = _GRPOConfig
    trl.ModelConfig = _ModelConfig
    trl.TrlParser = lambda classes: _Parser(classes)
    trl.GRPOTrainer = _Trainer
    trl.get_peft_config = lambda *_a, **_k: {}
    sys.modules["trl"] = trl


def _install_transformers_stub() -> None:
    """Stub the handful of transformers symbols used by the baseline trainer."""
    tf_mod = ModuleType("transformers")
    tf_mod.__spec__ = SimpleNamespace()
    trainer_utils = ModuleType("transformers.trainer_utils")
    trainer_utils.get_last_checkpoint = lambda *_a, **_k: None
    tf_logging = SimpleNamespace(
        set_verbosity=lambda *_a, **_k: None,
        enable_default_handler=lambda *_a, **_k: None,
        enable_explicit_format=lambda *_a, **_k: None,
    )
    utils_mod = ModuleType("transformers.utils")
    utils_mod.logging = tf_logging
    tf_mod.trainer_utils = trainer_utils
    tf_mod.utils = utils_mod
    sys.modules["transformers"] = tf_mod
    sys.modules["transformers.trainer_utils"] = trainer_utils
    sys.modules["transformers.utils"] = utils_mod


def _install_training_run_stub() -> None:
    """Provide a stubbed training.run module so maxent CLI can import."""
    run_mod = ModuleType("training.run")

    def _run_maxent_grpo(script_args, training_args, model_args):
        # Simple marker to prove invocation occurred.
        run_mod.last_invocation = (script_args, training_args, model_args)
        return None

    run_mod.run_maxent_grpo = _run_maxent_grpo
    sys.modules["training.run"] = run_mod


class _DummyDataset:
    """Minimal dataset stand-in with HF Dataset-like surface."""

    def __init__(self, rows: Iterable[Dict[str, Any]]):
        self._rows = list(rows)

    def map(self, fn):
        return _DummyDataset([fn(row) for row in self._rows])

    def remove_columns(self, names):
        drop = set(names if isinstance(names, (list, tuple, set)) else [names])
        return _DummyDataset(
            [{k: v for k, v in row.items() if k not in drop} for row in self._rows]
        )

    def shuffle(self, seed=None):
        # Deterministic for speed; no-op is fine for smoke.
        return self

    def select(self, indices):
        return _DummyDataset([self._rows[i] for i in indices])

    def train_test_split(self, test_size=0.1, seed=0):
        n_test = max(1, int(len(self._rows) * float(test_size)))
        return {"train": self, "test": _DummyDataset(self._rows[:n_test])}

    @property
    def column_names(self):
        keys = set()
        for row in self._rows:
            keys.update(row.keys())
        return sorted(keys)

    def __len__(self):
        return len(self._rows)

    def __iter__(self):
        return iter(self._rows)


class _DummyDatasetDict(dict):
    def map(self, fn):
        return _DummyDatasetDict({k: v.map(fn) for k, v in self.items()})


def _build_dummy_data() -> _DummyDatasetDict:
    rows = [
        {"problem": "1+1", "answer": "2"},
        {"problem": "2+2", "answer": "4"},
    ]
    return _DummyDatasetDict(
        {
            "train": _DummyDataset(rows),
            "validation": _DummyDataset(rows[:1]),
        }
    )


def _run_grpo_smoke() -> None:
    """Call the GRPO entrypoint with stubbed data and trainer."""
    from grpo import main as grpo_main
    from maxent_grpo.config import GRPOConfig, GRPOScriptArguments
    import pipelines.training.baseline as baseline

    dummy_data = _build_dummy_data()

    baseline.get_dataset = lambda _args: dummy_data
    baseline.load_dataset_split = lambda *_a, **_k: dummy_data["validation"]
    baseline.get_reward_funcs = lambda *_a, **_k: []
    baseline.ensure_vllm_group_port = lambda: None

    class _Tok:
        chat_template = None
        eos_token_id = 0
        pad_token_id = 0

        def apply_chat_template(
            self, messages, tokenize=False, add_generation_prompt=True
        ):
            return f"USER: {messages[-1]['content']}\nASSISTANT:"

        def add_special_tokens(self, *_args, **_kwargs):
            return None

        def resize_token_embeddings(self, *_args, **_kwargs):
            return None

    class _Model:
        def __init__(self):
            self.config = SimpleNamespace(pad_token_id=None, use_cache=True)

        def resize_token_embeddings(self, *_a, **_k):
            return None

        def gradient_checkpointing_enable(self, *_a, **_k):
            return None

    baseline.get_tokenizer = lambda *_a, **_k: _Tok()
    baseline.get_model = lambda *_a, **_k: _Model()

    script_args = GRPOScriptArguments(dataset_name="dummy")
    training_args = GRPOConfig(
        output_dir=str(REPO_ROOT / ".tmp" / "smoke"),
        do_eval=True,
        seed=0,
        gradient_checkpointing=False,
    )
    training_args.get_process_log_level = lambda: 20
    model_args = SimpleNamespace(model_name_or_path="stub/model")

    grpo_main(script_args, training_args, model_args)


def _run_maxent_smoke() -> None:
    """Ensure the maxent CLI wires args into training.run_maxent_grpo."""
    import maxent_grpo

    script_args = SimpleNamespace()
    training_args = SimpleNamespace()
    model_args = SimpleNamespace()

    maxent_grpo.parse_grpo_args = lambda: (script_args, training_args, model_args)
    import training

    training.run_maxent_grpo = lambda *args: setattr(training, "last_maxent_args", args)
    maxent_grpo.run_maxent_grpo = training.run_maxent_grpo
    maxent_grpo.main()
    assert getattr(training, "last_maxent_args", None) == (
        script_args,
        training_args,
        model_args,
    )


def _run_inference_smoke() -> None:
    """Run math_500 inference with a stub runner."""
    import pipelines.inference.math500 as math500

    math500.load_math500_dataset = lambda cfg: [
        {cfg.prompt_column: "1+1", cfg.solution_column: "2"},
        {cfg.prompt_column: "2+2", cfg.solution_column: "4"},
    ]

    class _Runner:
        def __init__(self, spec):
            self.spec = spec

        def generate(self, problems: List[str]) -> List[str]:
            return ["2" if "1+1" in p else "4" for p in problems]

        def close(self):
            return None

    specs = [
        math500.InferenceModelSpec(
            model_name_or_path="stub/model", style="grpo", label="smoke"
        )
    ]
    results = math500.run_math500_inference(
        specs, runner_factory=lambda spec: _Runner(spec)
    )
    assert results and results[0].accuracy == 1.0


def main() -> None:
    _install_trl_stub()
    _install_transformers_stub()
    _install_training_run_stub()
    _run_grpo_smoke()
    _run_maxent_smoke()
    _run_inference_smoke()
    print("✅ Smoke CLI passed (grpo, maxent, inference)")


if __name__ == "__main__":
    main()
