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

Lightweight CLI smoke check for training and inference entrypoints.

This script stubs heavy dependencies (torch/transformers/trl) and exercises:
- GRPO training entrypoint (maxent_grpo.grpo -> pipelines.training.baseline)
- MaxEnt-GRPO entrypoint (maxent_grpo.maxent_grpo)
- math inference runner (maxent_grpo.pipelines.inference.inference)

The goal is to catch wiring/packaging breakage without downloading models or
datasets. It should run quickly on CPU-only CI runners.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Dict, Iterable, List


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
VAR_ROOT = REPO_ROOT / "var"


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
    tf_mod.PreTrainedModel = type("PreTrainedModel", (), {})
    tf_mod.PreTrainedTokenizer = type("PreTrainedTokenizer", (), {})
    sys.modules["transformers"] = tf_mod
    sys.modules["transformers.trainer_utils"] = trainer_utils
    sys.modules["transformers.utils"] = utils_mod


def _install_accelerate_stub() -> None:
    """Stub accelerate so training imports don't pull CUDA deps."""

    accel_mod = ModuleType("accelerate")

    class _AccelState(ModuleType):
        DistributedType = SimpleNamespace(DEEPSPEED="deepspeed")

    class _Accelerator:
        def __init__(self, *args, **kwargs):
            self.device = "cpu"
            self.is_main_process = True
            self.num_processes = 1
            self.process_index = 0
            self.gradient_accumulation_steps = 1
            self.sync_gradients = True

        def gather(self, obj):
            return obj

        def gather_object(self, obj):
            return [obj]

        def log(self, metrics, step=None):
            return None

        def wait_for_everyone(self):
            return None

        @contextmanager
        def accumulate(self, _model):
            yield

        def backward(self, _loss):
            return None

        def clip_grad_norm_(self, *_args, **_kwargs):
            return 0.0

        def unwrap_model(self, model):
            return model

        def save_state(self, _path):
            return None

        def load_state(self, _path):
            return None

    accel_mod.Accelerator = _Accelerator
    accel_state = _AccelState("accelerate.state")
    accel_mod.state = accel_state
    sys.modules["accelerate"] = accel_mod
    sys.modules["accelerate.state"] = accel_state


def _set_local_caches() -> None:
    """Route HF/pip/tmp caches into var/ for isolated smoke runs."""

    cache_root = VAR_ROOT / "cache"
    os.environ.setdefault("HF_HOME", str(cache_root / "huggingface"))
    os.environ.setdefault(
        "HF_DATASETS_CACHE", str(cache_root / "huggingface" / "datasets")
    )
    os.environ.setdefault(
        "TRANSFORMERS_CACHE", str(cache_root / "huggingface" / "transformers")
    )
    os.environ.setdefault("PIP_CACHE_DIR", str(cache_root / "pip"))
    os.environ.setdefault("WANDB_DIR", str(VAR_ROOT / "logs"))
    os.environ.setdefault("TMPDIR", str(VAR_ROOT / "tmp"))
    os.environ.setdefault("WANDB_MODE", os.environ.get("WANDB_MODE", "offline"))


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
    import maxent_grpo.pipelines.training.baseline as baseline

    called = {}
    baseline.run_baseline_training = lambda *args: called.setdefault("args", args)
    script_args = SimpleNamespace(dataset_name="dummy")
    training_args = SimpleNamespace(get_process_log_level=lambda: 20)
    model_args = SimpleNamespace(model_name_or_path="stub/model")
    baseline.run_baseline_training(script_args, training_args, model_args)
    assert called.get("args") == (script_args, training_args, model_args)


def _run_maxent_smoke() -> None:
    """Ensure the maxent CLI wires args into run_maxent_training."""
    import maxent_grpo.pipelines.training.maxent as maxent_training

    called = {}
    maxent_training.run_maxent_training = lambda *args: called.setdefault("args", args)
    script_args = SimpleNamespace()
    training_args = SimpleNamespace()
    model_args = SimpleNamespace()
    maxent_training.run_maxent_training(script_args, training_args, model_args)
    assert called.get("args") == (script_args, training_args, model_args)


def _run_generation_smoke() -> None:
    """Exercise the distilabel generation helper with stubs."""

    from maxent_grpo.pipelines.generation.distilabel import run_generation_job

    datasets_mod = ModuleType("datasets")
    datasets_mod.load_dataset = lambda *_a, **_k: [
        {"instruction": "Ping?", "answer": "Pong!"}
    ]
    sys.modules["datasets"] = datasets_mod

    class _StubDistiset(list):
        def push_to_hub(self, *args, **kwargs):
            return None

    def _builder(_cfg):
        class _Pipeline:
            def run(self, dataset, dataset_batch_size=None, use_cache=False):
                return _StubDistiset(dataset)

        return _Pipeline()

    args = SimpleNamespace(
        hf_dataset="stub/ds",
        hf_dataset_config=None,
        hf_dataset_split="train",
        model="stub/model",
        vllm_server_url="http://localhost:29525",
        prompt_template=None,
        prompt_column="instruction",
        temperature=0.1,
        top_p=0.9,
        max_new_tokens=8,
        num_generations=1,
        input_batch_size=1,
        client_replicas=1,
        timeout=5,
        retries=0,
        hf_output_dataset=None,
        private=False,
    )
    run_generation_job(args, builder=_builder)


def _run_inference_smoke() -> None:
    """Run math inference with a stub runner."""
    import maxent_grpo.pipelines.inference.inference as math_inf

    stub_data = [
        {"problem": "1+1", "answer": "2"},
        {"problem": "2+2", "answer": "4"},
    ]
    math_inf.load_math_dataset = lambda cfg: list(stub_data)

    class _Runner:
        def __init__(self, spec):
            self.spec = spec

        def generate(self, problems: List[str]) -> List[str]:
            return ["2" if "1+1" in p else "4" for p in problems]

        def close(self):
            return None

    specs = [
        math_inf.InferenceModelSpec(
            model_name_or_path="stub/model", style="grpo", label="smoke"
        )
    ]
    results = math_inf.run_math_inference(
        specs, runner_factory=lambda spec: _Runner(spec)
    )
    assert results and results[0].total == len(stub_data)


def main() -> None:
    _set_local_caches()
    _install_trl_stub()
    _install_transformers_stub()
    _install_accelerate_stub()
    _run_generation_smoke()
    _run_grpo_smoke()
    _run_maxent_smoke()
    _run_inference_smoke()
    print("✅ Smoke CLI passed (generation, grpo, maxent, inference)")


if __name__ == "__main__":
    main()
