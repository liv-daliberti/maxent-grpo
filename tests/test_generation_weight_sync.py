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

Tests for vLLM weight synchronization in ``training.generation.helpers``.
"""

from __future__ import annotations

import sys
import types
import importlib
from types import SimpleNamespace


torch_stub = types.ModuleType("torch")
torch_stub.Tensor = type("Tensor", (), {})
torch_stub.cuda = SimpleNamespace(empty_cache=lambda: None)
torch_stub.distributed = SimpleNamespace(fsdp=None)
torch_nn = types.ModuleType("torch.nn")
torch_nn.functional = types.ModuleType("torch.nn.functional")
torch_nn.functional.log_softmax = lambda *args, **kwargs: None
torch_stub.nn = torch_nn
torch_stub.optim = SimpleNamespace(Optimizer=type("Optimizer", (), {}))
torch_stub.__path__ = []
torch_utils = types.ModuleType("torch.utils")
torch_utils.data = types.ModuleType("torch.utils.data")
torch_utils.data.DataLoader = type("DataLoader", (), {})
torch_utils.data.Dataset = type("Dataset", (), {})
torch_utils.data.Sampler = type("Sampler", (), {})
torch_stub.utils = torch_utils
sys.modules.setdefault("torch", torch_stub)
torch_optim = types.ModuleType("torch.optim")
torch_optim.Optimizer = torch_stub.optim.Optimizer
sys.modules.setdefault("torch.optim", torch_optim)
sys.modules.setdefault("torch.utils", torch_utils)
sys.modules.setdefault("torch.utils.data", torch_utils.data)
sys.modules.setdefault("torch.nn", torch_nn)
sys.modules.setdefault("torch.nn.functional", torch_nn.functional)


def _torch_cat(tensors, dim=0):
    if not tensors:
        return []
    first = tensors[0]
    data = []
    for tensor in tensors:
        data.extend(list(tensor))
    try:
        return type(first)(data)
    except Exception:
        return data


torch_stub.cat = _torch_cat
accelerate_stub = types.ModuleType("accelerate")
accelerate_stub.Accelerator = lambda **_kwargs: SimpleNamespace()
accelerate_state = types.ModuleType("accelerate.state")
accelerate_state.DistributedType = SimpleNamespace(DEEPSPEED="deepspeed")
accelerate_stub.state = accelerate_state
sys.modules.setdefault("accelerate", accelerate_stub)
transformers_stub = types.ModuleType("transformers")
transformers_stub.PreTrainedModel = type("PreTrainedModel", (), {})
transformers_stub.PreTrainedTokenizer = type("PreTrainedTokenizer", (), {})
sys.modules.setdefault("transformers", transformers_stub)


def _import_under_test():
    helpers_mod = importlib.reload(
        importlib.import_module("maxent_grpo.training.generation.helpers")
    )
    run_helpers_mod = importlib.reload(
        importlib.import_module("maxent_grpo.training.run_helpers")
    )
    return (
        helpers_mod.CompletionGenerator,
        helpers_mod.GenerationContext,
        run_helpers_mod.GenerationPenaltyConfig,
        run_helpers_mod.VLLMClientConfig,
    )


(
    CompletionGenerator,
    GenerationContext,
    GenerationPenaltyConfig,
    VLLMClientConfig,
) = _import_under_test()


class _DummyClient:
    def __init__(self):
        self.updated = []
        self.cache_reset = 0
        self.initted = False

    def init_communicator(self):
        self.initted = True

    def update_named_param(self, name, data):
        self.updated.append((name, data))

    def reset_prefix_cache(self):
        self.cache_reset += 1


class _DummyAccel:
    def __init__(self, is_main=True):
        self.is_main_process = is_main

    def wait_for_everyone(self):
        self.wait_called = True

    def unwrap_model(self, model):
        return model


def _generator_ctx(**overrides):
    vllm_cfg = VLLMClientConfig(
        url="http://localhost:8000/v1",
        rounds_cfg=1,
        retry_sleep=0.0,
        backfill_local=False,
        request_logprobs=False,
        sync_weights=False,
    )
    ctx = GenerationContext(
        accelerator=_DummyAccel(),
        model=SimpleNamespace(
            parameters=lambda: [],
            named_parameters=lambda: [],
            named_children=lambda: [],
        ),
        tokenizer=None,
        generation_stats={"current_step": 1},
        device="cpu",
        max_prompt_len=1,
        max_completion_len=1,
        gen_temperature=0.5,
        gen_top_p=0.9,
        use_vllm=True,
        vllm=vllm_cfg,
        penalty=GenerationPenaltyConfig(),
    )
    for key, value in overrides.items():
        if key == "vllm_sync_weights":
            ctx.vllm.sync_weights = bool(value)
        elif key == "vllm_url":
            ctx.vllm.url = value
        elif key == "vllm_rounds_cfg":
            ctx.vllm.rounds_cfg = int(value)
        else:
            setattr(ctx, key, value)
    return ctx


def test_ensure_vllm_client_skips_when_disabled(monkeypatch):
    ctx = _generator_ctx(vllm_sync_weights=False)
    gen = CompletionGenerator(ctx)
    assert gen._ensure_vllm_client() is False


def test_ensure_vllm_client_inits_client(monkeypatch):
    ctx = _generator_ctx(
        vllm_sync_weights=True, vllm_url="http://localhost:8000/generate"
    )
    client = _DummyClient()

    def fake_import():
        return lambda base_url: client

    monkeypatch.setattr(
        "maxent_grpo.training.generation.helpers._import_vllm_client_cls", fake_import
    )
    gen = CompletionGenerator(ctx)
    gen._import_vllm_client_cls = fake_import
    assert gen._ensure_vllm_client() is True
    assert gen._vllm_client is not None
    assert gen._vllm_sync_ready is True


def test_sync_standard_params_updates_client(monkeypatch):
    ctx = _generator_ctx(vllm_sync_weights=True)
    client = _DummyClient()
    monkeypatch.setattr(
        "maxent_grpo.training.generation.helpers._import_vllm_client_cls",
        lambda: lambda **_: client,
    )
    weights = {"layer.weight": SimpleNamespace(data="tensor")}

    class _Model:
        def parameters(self):
            return list(weights.values())

        def named_parameters(self):
            return list(weights.items())

        def named_children(self):
            return []

    ctx.model = _Model()
    gen = CompletionGenerator(ctx)
    gen._import_vllm_client_cls = lambda: lambda **_: client
    assert gen._ensure_vllm_client()
    gen._vllm_client = client  # ensure sync uses our stub client
    gen._vllm_sync_ready = True
    gen._sync_model_params_to_vllm(ctx.model, ctx.accelerator)
    assert client.updated == [("layer.weight", "tensor")]
    assert client.cache_reset == 1


def test_sync_model_params_calls_waiter(monkeypatch):
    ctx = _generator_ctx(vllm_sync_weights=True)
    client = _DummyClient()
    monkeypatch.setattr(
        "maxent_grpo.training.generation.helpers._import_vllm_client_cls",
        lambda: lambda **_: client,
    )
    ctx.accelerator = _DummyAccel()
    ctx.accelerator.unwrap_model = lambda model: model

    class _Model(SimpleNamespace):
        pass

    model = _Model(
        parameters=lambda: [SimpleNamespace(data="v")],
        named_parameters=lambda: [("p", SimpleNamespace(data="v"))],
        named_children=lambda: [],
    )
    ctx.model = model
    ctx.generation_stats = {"current_step": 1}
    gen = CompletionGenerator(ctx)
    gen._import_vllm_client_cls = lambda: lambda **_: client
    assert gen._ensure_vllm_client()
    gen._maybe_sync_vllm_weights()
    assert getattr(ctx.accelerator, "wait_called", False) is True
