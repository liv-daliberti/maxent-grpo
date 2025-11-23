"""Unit tests for CompletionGenerator logic not covered elsewhere."""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from typing import List


torch_stub = types.ModuleType("torch")
torch_stub.Tensor = list
torch_stub.device = type("device", (), {})
torch_stub.optim = types.SimpleNamespace(Optimizer=type("Optimizer", (), {}))
torch_stub.__spec__ = SimpleNamespace()
sys.modules["torch"] = torch_stub
torch_utils = types.ModuleType("torch.utils")
sys.modules["torch.utils"] = torch_utils
torch_utils_data = types.ModuleType("torch.utils.data")
torch_utils_data.DataLoader = type("DataLoader", (object,), {})
torch_utils_data.Sampler = type("Sampler", (object,), {})
sys.modules["torch.utils.data"] = torch_utils_data
torch_stub.utils = SimpleNamespace(data=torch_utils_data)
torch_optim = types.ModuleType("torch.optim")
torch_optim.Optimizer = type("Optimizer", (object,), {})
sys.modules["torch.optim"] = torch_optim
torch_nn = types.ModuleType("torch.nn")
torch_nn.functional = types.ModuleType("torch.nn.functional")
torch_nn.functional.log_softmax = lambda *args, **kwargs: None
sys.modules["torch.nn"] = torch_nn
sys.modules["torch.nn.functional"] = torch_nn.functional
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
sys.modules["trl"] = trl_stub

from training.generation import CompletionGenerator  # noqa: E402
from training.generation.helpers import _VLLMGenerationState  # noqa: E402
from training.run_helpers import (  # noqa: E402
    GenerationPenaltyConfig,
    GenerationSamplingConfig,
    VLLMClientConfig,
)


class _Tokenizer:
    def __call__(self, prompts, **_kwargs):
        return SimpleNamespace(
            to=lambda **_kwargs: SimpleNamespace(
                __getitem__=lambda self, key: SimpleNamespace(
                    sum=lambda dim: SimpleNamespace(
                        detach=lambda: SimpleNamespace(cpu=lambda: [len(prompts)])
                    )
                )
            )
        )


class _Model:
    def generate(self, **_kwargs):
        return [[1, 2, 3]]


class _Accel:
    def __init__(
        self, is_main: bool = True, num_processes: int = 1, process_index: int = 0
    ):
        self.is_main_process = is_main
        self.num_processes = num_processes
        self.process_index = process_index

    def unwrap_model(self, model):
        return model

    def wait_for_everyone(self):
        return None


def _ctx(*, use_vllm: bool = False, backfill_local: bool = False):
    vllm_cfg = VLLMClientConfig(
        url="http://localhost",
        rounds_cfg=1,
        retry_sleep=0.0,
        backfill_local=backfill_local,
        request_logprobs=False,
    )
    sampling_cfg = GenerationSamplingConfig(
        max_prompt_len=16,
        max_completion_len=4,
        gen_temperature=0.8,
        gen_top_p=0.9,
        use_vllm=use_vllm,
        vllm=vllm_cfg,
    )
    ctx = SimpleNamespace(
        accelerator=_Accel(),
        model=_Model(),
        tokenizer=_Tokenizer(),
        generation_stats={"vllm_backfilled_prompts": 0},
        penalty=GenerationPenaltyConfig(),
        max_prompt_len=sampling_cfg.max_prompt_len,
        max_completion_len=sampling_cfg.max_completion_len,
        gen_temperature=sampling_cfg.gen_temperature,
        gen_top_p=sampling_cfg.gen_top_p,
        gen_top_k=None,
        gen_best_of=None,
        gen_frequency_penalty=0.0,
        gen_presence_penalty=0.0,
        gen_stop_sequences=None,
        use_vllm=use_vllm,
        vllm_url=vllm_cfg.url,
        vllm=vllm_cfg,
        vllm_backfill_local=backfill_local,
        vllm_request_logprobs=False,
        vllm_sync_weights=False,
        device="cpu",
    )
    return ctx


def test_seed_generation_groups_handles_meta():
    groups, meta = CompletionGenerator._seed_generation_groups(
        prompt_count=2,
        grouped_comps=[["a"], []],
        grouped_meta=[[["meta"]], None],
    )
    assert groups[0] == ["a"]


def test_retry_incomplete_prompts_respects_limit(monkeypatch):
    helper = CompletionGenerator(_ctx())

    def _fake_generator(prompts, expected_generations, needed_counts=None):
        return [[["x"] for _ in prompts], None]

    aggregated = [[], []]
    meta = None
    grouped, meta = CompletionGenerator._retry_incomplete_prompts(
        helper,
        ["p1", "p2"],
        _fake_generator,
        expected_generations=1,
        aggregated_comps=aggregated,
        aggregated_meta=meta,
        max_retry_rounds=1,
    )
    assert grouped[0] == ["x"]


def test_generate_vllm_collective_invokes_remote(monkeypatch):
    ctx = _ctx(use_vllm=True)
    helper = CompletionGenerator(ctx)
    captured = {}

    def _fake_generate_with_vllm(self, prompts, num_samples, counts):
        captured["args"] = (list(prompts), num_samples, counts)
        return [["p0-out"], ["p1-out"]], None

    monkeypatch.setattr(
        CompletionGenerator,
        "_generate_with_vllm",
        _fake_generate_with_vllm,
    )
    grouped, meta = helper._generate_vllm_collective(
        ["p0", "p1"],
        num_samples=2,
        per_prompt_counts=[1, 2],
    )
    assert captured["args"][0] == ["p0", "p1"]
    assert grouped == [["p0-out"], ["p1-out"]]
    assert meta is None


def test_backfill_missing_uses_local_generation(monkeypatch):
    ctx = _ctx(use_vllm=True, backfill_local=True)
    helper = CompletionGenerator(ctx)
    state = _VLLMGenerationState(
        prompts=["p0", "p1"],
        target_counts=[2, 1],
        requested_n=2,
        round_limit=2,
        track_logprobs=False,
    )
    state.aggregated[0] = ["seed"]

    def _fake_local(prompts, num_samples, counts):
        groups: List[List[str]] = []
        for prompt, count in zip(prompts, counts):
            groups.append([f"{prompt}-fill"] * count)
        return groups, None

    monkeypatch.setattr(helper, "_generate_local", _fake_local)
    helper._backfill_missing(state, [0, 1])
    assert state.aggregated[0] == ["seed", "p0-fill"]
    assert state.aggregated[1] == ["p1-fill"]
    assert ctx.generation_stats["vllm_backfilled_prompts"] == 2
