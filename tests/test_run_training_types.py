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

Tests covering training type dataclass helpers.
"""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace


_TORCH_STUB = types.ModuleType("torch")
_TORCH_STUB.Tensor = type("Tensor", (object,), {})
_TORCH_STUB.device = type("device", (object,), {})
_TORCH_STUB.optim = types.SimpleNamespace(Optimizer=type("Optimizer", (object,), {}))
_TORCH_STUB.__spec__ = types.SimpleNamespace()
sys.modules["torch"] = _TORCH_STUB
torch_utils = types.ModuleType("torch.utils")
sys.modules["torch.utils"] = torch_utils
torch_utils_data = types.ModuleType("torch.utils.data")
torch_utils_data.DataLoader = type("DataLoader", (object,), {})
torch_utils_data.Sampler = type("Sampler", (object,), {})
sys.modules["torch.utils.data"] = torch_utils_data
accelerate_stub = types.ModuleType("accelerate")
accelerate_stub.Accelerator = type("Accelerator", (object,), {})
sys.modules["accelerate"] = accelerate_stub
transformers_stub = types.ModuleType("transformers")
transformers_stub.PreTrainedModel = type("PreTrainedModel", (object,), {})
transformers_stub.PreTrainedTokenizer = type("PreTrainedTokenizer", (object,), {})
sys.modules["transformers"] = transformers_stub
torch_optim = types.ModuleType("torch.optim")
torch_optim.Optimizer = type("Optimizer", (object,), {})
sys.modules["torch.optim"] = torch_optim
trl_stub = types.ModuleType("trl")
trl_stub.ScriptArguments = type("ScriptArguments", (object,), {})
trl_stub.GRPOConfig = type("GRPOConfig", (object,), {})
sys.modules["trl"] = trl_stub

from maxent_grpo.training.run_helpers import (  # noqa: E402
    GenerationPenaltyConfig,
    VLLMClientConfig,
)
from maxent_grpo.training.types import (  # noqa: E402
    ControllerPaths,
    LoggingHandles,
    OptimizerHandles,
    OptimizationSchedule,
    OptimizationSettings,
    RuntimeHandles,
    GenerationSettings,
)
from maxent_grpo.training.weighting import (  # noqa: E402
    KlControllerSettings,
    QDistributionSettings,
    TauSchedule,
    WeightNormalizationSettings,
    WeightingSettings,
)


def _base_vllm_cfg():
    return VLLMClientConfig(
        url="http://localhost",
        rounds_cfg=1,
        retry_sleep=0.0,
        backfill_local=False,
        request_logprobs=False,
    )


def test_generation_settings_property_passthrough():
    cfg = GenerationSettings(
        max_prompt_len=16,
        max_completion_len=8,
        gen_temperature=0.8,
        gen_top_p=0.9,
        use_vllm=False,
        vllm=_base_vllm_cfg(),
        penalty=GenerationPenaltyConfig(),
    )
    cfg.gen_top_k = 4
    cfg.gen_best_of = 2
    cfg.gen_stop_sequences = ["</s>"]
    assert cfg.penalty.gen_top_k == 4
    assert cfg.penalty.gen_best_of == 2
    assert cfg.penalty.gen_stop_sequences == ["</s>"]


def test_weighting_settings_updates_nested_structs():
    weighting = WeightingSettings(
        tau=0.1,
        beta=0.2,
        normalization=WeightNormalizationSettings(denom=1.0, len_norm_ref=True),
        q_distribution=QDistributionSettings(temperature=1.0, epsilon=1e-6),
        tau_schedule=TauSchedule(
            target_entropy=None,
            learning_rate=0.01,
            minimum_value=0.0,
            maximum_value=1.0,
            warmup_steps=5,
        ),
        kl_controller=KlControllerSettings(target=0.1, horizon=10, step_size=0.5),
        train_grpo_objective=False,
    )
    weighting.denom = 2.0
    weighting.q_temperature = 0.5
    weighting.tau_target_entropy = 0.3
    weighting.kl_target = 0.2
    assert weighting.normalization.denom == 2.0
    assert weighting.q_distribution.temperature == 0.5
    assert weighting.tau_schedule.target_entropy == 0.3
    assert weighting.kl_controller.target == 0.2


def test_runtime_handles_instantiation():
    handles = RuntimeHandles(
        accelerator=SimpleNamespace(),
        model=SimpleNamespace(),
        tokenizer=SimpleNamespace(),
        train_loader=SimpleNamespace(),
        train_sampler=None,
        device=SimpleNamespace(),
        get_ref_model=lambda: "ref",
    )
    assert handles.get_ref_model() == "ref"


def test_optimization_settings_wrapper():
    schedule = OptimizationSchedule(
        num_epochs=1,
        num_generations=2,
        grad_accum_steps=1,
        max_grad_norm=1.0,
        steps_per_epoch=10,
        total_training_steps=20,
        warmup_steps=0,
    )
    handles = OptimizerHandles(
        optimizer=SimpleNamespace(),
        lr_scheduler=None,
        base_optimizer=SimpleNamespace(),
        learning_rate=1e-4,
    )
    settings = OptimizationSettings(schedule=schedule, handles=handles)
    assert settings.schedule.total_training_steps == 20


def test_logging_handles_checkpoint_callback():
    saved = []

    class _Writer:
        def __init__(self):
            self.logged = {}

        def log(self, metrics, step):
            self.logged[step] = metrics

    handles = LoggingHandles(
        metric_writer=_Writer(),
        save_checkpoint=lambda name: saved.append(name),
        save_strategy="steps",
        save_steps=1,
        wandb_run=None,
    )
    handles.log_metrics({"loss": 1.0}, step=5)
    assert handles.metric_writer.logged[5]["loss"] == 1.0
    handles.save_checkpoint("test")
    assert saved == ["test"]


def test_controller_paths_fields():
    cfg = ControllerPaths(
        resume_from="foo",
        state_path="bar",
        overwrite_existing=True,
    )
    assert cfg.resume_from == "foo"