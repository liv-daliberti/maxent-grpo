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

Tests for the vLLM helper utilities with lightweight stubs.
"""

from __future__ import annotations

from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
import sys
import types
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

_TORCH_STUB = types.ModuleType("torch")
_TORCH_STUB.Tensor = type("Tensor", (object,), {})
_TORCH_STUB.device = type("device", (object,), {})
_TORCH_STUB.optim = types.SimpleNamespace(Optimizer=type("Optimizer", (object,), {}))
_TORCH_STUB.__spec__ = types.SimpleNamespace()
_TORCH_STUB.__path__ = []
sys.modules["torch"] = _TORCH_STUB
torch_utils = types.ModuleType("torch.utils")
sys.modules["torch.utils"] = torch_utils
torch_utils_data = types.ModuleType("torch.utils.data")
torch_utils_data.DataLoader = type("DataLoader", (object,), {})
torch_utils_data.Sampler = type("Sampler", (object,), {})
sys.modules["torch.utils.data"] = torch_utils_data
torch_nn = types.ModuleType("torch.nn")
torch_nn.functional = types.ModuleType("torch.nn.functional")
torch_nn.functional.log_softmax = lambda *args, **kwargs: None
sys.modules["torch.nn"] = torch_nn
sys.modules["torch.nn.functional"] = torch_nn.functional
torch_optim = types.ModuleType("torch.optim")
torch_optim.Optimizer = type("Optimizer", (object,), {})
sys.modules["torch.optim"] = torch_optim
accelerate_stub = types.ModuleType("accelerate")
accelerate_stub.Accelerator = type("Accelerator", (object,), {})
sys.modules["accelerate"] = accelerate_stub
transformers_stub = types.ModuleType("transformers")
transformers_stub.PreTrainedModel = type("PreTrainedModel", (object,), {})
transformers_stub.PreTrainedTokenizer = type("PreTrainedTokenizer", (object,), {})
sys.modules["transformers"] = transformers_stub
trl_stub = types.ModuleType("trl")
trl_stub.ScriptArguments = type("ScriptArguments", (object,), {})
trl_stub.GRPOConfig = type("GRPOConfig", (object,), {})
trl_extras = types.ModuleType("trl.extras")
trl_vllm = types.ModuleType("trl.extras.vllm_client")


class _ClientStub:
    def __init__(self, base_url: str):
        self.base_url = base_url

    def init_communicator(self):
        pass

    def update_named_param(self, name, data):
        pass

    def reset_prefix_cache(self):
        pass


trl_vllm.VLLMClient = _ClientStub
sys.modules["trl"] = trl_stub
sys.modules["trl.extras"] = trl_extras
sys.modules["trl.extras.vllm_client"] = trl_vllm


def _vllm_classes():
    from maxent_grpo.generation.vllm import (
        VLLMGenerationHelper,
        _VLLMGenerationState,
    )

    return VLLMGenerationHelper, _VLLMGenerationState


def _helper_ctx():
    accelerator = SimpleNamespace(
        state=None,
        is_main_process=True,
        wait_for_everyone=lambda: None,
        unwrap_model=lambda m: m,
    )
    return SimpleNamespace(
        accelerator=accelerator,
        model=SimpleNamespace(named_parameters=lambda: []),
        generation_stats=defaultdict(int),
        vllm_url="http://localhost",
        vllm_sync_weights=False,
        vllm_backfill_local=False,
        vllm_rounds_cfg=1,
        vllm_retry_sleep=0.0,
    )


def _fallback_generate(prompts, num_samples, counts=None):
    grouped = []
    for prompt, need in zip(prompts, counts or [num_samples] * len(prompts)):
        grouped.append([f"{prompt}-fallback"] * max(need, 1))
    return grouped, None


def test_vllm_state_pending_and_trim():
    _, _VLLMGenerationState = _vllm_classes()
    state = _VLLMGenerationState(
        prompts=["p1", "p2"],
        target_counts=[2, 1],
        requested_n=2,
        round_limit=2,
        track_logprobs=True,
    )
    state.aggregated[0].extend(["a", "b", "c"])
    state.aggregated_meta[0].extend([None, None, None])
    pending = state.pending_indices()
    assert pending == [1]
    trimmed, meta = state.trim()
    assert trimmed[0] == ["a", "b"]
    assert meta[0] == [None, None]


def test_prepare_targets_dedup_enabled(monkeypatch):
    monkeypatch.setenv("MAXENT_VLLM_DEDUP", "1")
    VLLMGenerationHelper, _ = _vllm_classes()
    helper = VLLMGenerationHelper(_helper_ctx(), _fallback_generate)
    prompts = ["p1", "p1", "p2"]
    counts = [1, 2, 3]
    unique, target_counts, mapping = helper._prepare_vllm_targets(prompts, 2, counts)
    assert unique == ["p1", "p2"]
    assert target_counts == [1, 3]
    assert mapping == [0, 0, 1]
    expanded, _ = helper._expand_dedup_results(
        grouped=[["x"], ["y"]],
        meta=None,
        mapping=mapping,
    )
    assert expanded == [["x"], ["x"], ["y"]]


def test_vllm_helper_backfills_missing(monkeypatch):
    VLLMGenerationHelper, _VLLMGenerationState = _vllm_classes()
    ctx = _helper_ctx()
    ctx.vllm_backfill_local = True
    helper = VLLMGenerationHelper(ctx, _fallback_generate)

    def _always_fail(state, pending):
        return False

    monkeypatch.setattr(helper, "_execute_vllm_request", _always_fail)
    state = _VLLMGenerationState(
        prompts=["p1"],
        target_counts=[1],
        requested_n=1,
        round_limit=1,
        track_logprobs=False,
    )
    helper._run_vllm_rounds(state)
    assert state.aggregated[0] == ["p1-fallback"]
    assert ctx.generation_stats["vllm_backfilled_prompts"] == 1


def test_vllm_helper_maybe_sync_weights(monkeypatch):
    VLLMGenerationHelper, _ = _vllm_classes()
    ctx = _helper_ctx()
    ctx.vllm_sync_weights = True
    helper = VLLMGenerationHelper(ctx, _fallback_generate)
    pushed: list[str] = []

    class _Client:
        def update_named_param(self, name, data):
            pushed.append(name)

        def reset_prefix_cache(self):
            pushed.append("reset")

    helper._vllm_client = _Client()
    helper._vllm_sync_ready = True
    helper._ensure_vllm_client = lambda: True
    helper._gather_factory = lambda *_args, **_kwargs: nullcontext()
    ctx.model = SimpleNamespace(
        named_parameters=lambda: [("layer.weight", SimpleNamespace(data="w"))],
        parameters=lambda: [],
    )
    helper.maybe_sync_weights()
    assert "layer.weight" in pushed
    assert "reset" in pushed
    assert ctx.generation_stats["vllm_weight_syncs"] == 1
