"""Tests for training.rewards."""

import sys
from types import ModuleType, SimpleNamespace

import pytest

torch_module = sys.modules.setdefault("torch", ModuleType("torch"))
torch_module.__spec__ = getattr(torch_module, "__spec__", SimpleNamespace())
torch_module.__path__ = getattr(torch_module, "__path__", [])
if not hasattr(torch_module, "Tensor"):
    torch_module.Tensor = type("Tensor", (), {})
torch_utils_module = sys.modules.setdefault("torch.utils", ModuleType("torch.utils"))
torch_utils_module.__spec__ = getattr(torch_utils_module, "__spec__", SimpleNamespace())
torch_utils_module.__path__ = getattr(torch_utils_module, "__path__", [])
torch_data_module = sys.modules.setdefault(
    "torch.utils.data", ModuleType("torch.utils.data")
)
torch_data_module.__spec__ = getattr(torch_data_module, "__spec__", SimpleNamespace())
torch_data_module.__path__ = getattr(torch_data_module, "__path__", [])
if not hasattr(torch_data_module, "DataLoader"):

    class _DataLoader:  # minimal stub
        pass

    torch_data_module.DataLoader = _DataLoader
if not hasattr(torch_data_module, "Sampler"):

    class _Sampler:
        pass

    torch_data_module.Sampler = _Sampler
torch_utils_module.data = torch_data_module
torch_module.utils = torch_utils_module
torch_optim_module = sys.modules.setdefault("torch.optim", ModuleType("torch.optim"))
torch_optim_module.__spec__ = getattr(torch_optim_module, "__spec__", SimpleNamespace())
torch_optim_module.Optimizer = type("Optimizer", (), {})
torch_module.optim = torch_optim_module
torch_nn_module = sys.modules.setdefault("torch.nn", ModuleType("torch.nn"))
torch_nn_module.__spec__ = getattr(torch_nn_module, "__spec__", SimpleNamespace())
torch_nn_functional = sys.modules.setdefault(
    "torch.nn.functional",
    ModuleType("torch.nn.functional"),
)
torch_nn_functional.__spec__ = getattr(
    torch_nn_functional, "__spec__", SimpleNamespace()
)
if not hasattr(torch_nn_functional, "log_softmax"):

    def _log_softmax(*_args, **_kwargs):
        raise NotImplementedError

    torch_nn_functional.log_softmax = _log_softmax

accelerate_module = sys.modules.setdefault("accelerate", ModuleType("accelerate"))
accelerate_module.__spec__ = getattr(accelerate_module, "__spec__", SimpleNamespace())
if not hasattr(accelerate_module, "Accelerator"):

    class _Accel:
        def __init__(self, **_kwargs):
            self.is_main_process = True
            self.process_index = 0

    accelerate_module.Accelerator = _Accel

transformers_module = sys.modules.setdefault("transformers", ModuleType("transformers"))
transformers_module.__spec__ = getattr(
    transformers_module, "__spec__", SimpleNamespace()
)
if not hasattr(transformers_module, "PreTrainedModel"):

    class _PreTrainedModel:
        pass

    class _PreTrainedTokenizer:
        pass

    transformers_module.PreTrainedModel = _PreTrainedModel
    transformers_module.PreTrainedTokenizer = _PreTrainedTokenizer

import training.rewards as rr  # noqa: E402
from training.rewards import (  # noqa: E402
    RewardComputation,
    RewardSpec,
    prepare_generation_batch,
)
from generation import flatten_ref_metadata  # noqa: E402
from training.types import GenerationBatch  # noqa: E402
from patches.vllm import VLLMLogprobResult  # noqa: E402


def test_prepare_generation_batch_keeps_partial_and_drops_empty():
    batch = {
        "prompt": ["p1", "p2", "p3"],
        "answer": ["a1", "a2", "a3"],
    }

    calls = []

    def generator(prompts, expected_generations, per_prompt_counts=None):
        assert expected_generations == 2
        calls.append(
            (
                list(prompts),
                None if per_prompt_counts is None else list(per_prompt_counts),
            )
        )
        if len(calls) == 1:
            assert prompts == batch["prompt"]
            return (
                [["p1-c1", "p1-c2"], [], ["p3-c1"]],
                [[None, None], [], [[None]]],
            )
        # Subsequent retries only receive prompts missing completions.
        assert prompts == ["p2", "p3"]
        assert per_prompt_counts == [2, 1]
        return ([[] for _ in prompts], [[] for _ in prompts])

    stats = {
        "dropped_prompts": 0,
        "partial_prompts": 0,
    }
    gen_batch = prepare_generation_batch(
        batch, generator, stats, expected_generations=2
    )
    assert gen_batch.prompts == ["p1", "p3"]
    assert gen_batch.grouped_completions == [["p1-c1", "p1-c2"], ["p3-c1"]]
    assert stats["dropped_prompts"] == 1
    assert stats["partial_prompts"] == 1
    assert len(calls) > 1
    assert all(entry == (["p2", "p3"], [2, 1]) for entry in calls[1:])


def test_prepare_generation_batch_retries_until_prompt_fills():
    batch = {
        "prompt": ["p1", "p2"],
        "answer": ["a1", "a2"],
    }
    calls = []

    def generator(prompts, expected_generations, per_prompt_counts=None):
        calls.append(
            (
                list(prompts),
                None if per_prompt_counts is None else list(per_prompt_counts),
            )
        )
        if len(calls) == 1:
            return ([["p1-c1"], []], None)
        assert prompts == ["p2"]
        assert per_prompt_counts == [1]
        return ([["p2-c1"]], None)

    stats = {"dropped_prompts": 0, "partial_prompts": 0}
    gen_batch = prepare_generation_batch(
        batch, generator, stats, expected_generations=1
    )
    assert gen_batch.prompts == ["p1", "p2"]
    assert gen_batch.grouped_completions == [["p1-c1"], ["p2-c1"]]
    assert stats["dropped_prompts"] == 0
    assert len(calls) == 2
    assert calls[1] == (["p2"], [1])


def test_prepare_generation_batch_keeps_metadata_after_retry():
    batch = {"prompt": ["p1"], "answer": ["a1"]}
    rounds = {"count": 0}

    def generator(prompts, expected_generations, per_prompt_counts=None):
        rounds["count"] += 1
        assert per_prompt_counts is None or per_prompt_counts == [1]
        if rounds["count"] == 1:
            return ([[]], None)
        return ([["p1-c1"]], [["meta"]])

    stats = {"dropped_prompts": 0, "partial_prompts": 0}
    gen_batch = prepare_generation_batch(
        batch, generator, stats, expected_generations=1
    )
    assert gen_batch.grouped_completions == [["p1-c1"]]
    assert gen_batch.grouped_ref_meta == [["meta"]]


def test_prepare_generation_batch_trims_extra_completions():
    batch = {
        "prompt": ["p1", "p2"],
        "answer": ["a1", "a2"],
    }

    def generator(prompts, expected_generations, per_prompt_counts=None):
        assert expected_generations == 1
        assert per_prompt_counts is None
        return (
            [["p1-c1", "p1-c2"], ["p2-c1"]],
            [[{"meta": 1}, {"meta": 2}], [["m2"]]],
        )

    stats = {"dropped_prompts": 0, "partial_prompts": 0}
    gen_batch = prepare_generation_batch(
        batch, generator, stats, expected_generations=1
    )
    assert gen_batch.grouped_completions == [["p1-c1"], ["p2-c1"]]
    assert gen_batch.grouped_ref_meta == [[{"meta": 1}], [["m2"]]]
    assert stats["partial_prompts"] == 0


@pytest.fixture(name="reward_spec")
def _reward_spec():
    def _reward_fn(completions, answers):
        return [float(idx) for idx, _ in enumerate(completions, start=1)]

    return RewardSpec(
        reward_funcs=[_reward_fn],
        reward_weights=[1.0],
    )


def test_compute_reward_statistics(monkeypatch, reward_spec):
    gen_batch = GenerationBatch(
        prompts=["prompt"],
        answers=["answer"],
        grouped_completions=[["c1", "c2"]],
        grouped_ref_meta=[[["meta1"], ["meta2"]]],
    )
    # Patch helper functions to avoid heavy dependencies.
    monkeypatch.setattr(
        rr,
        "reward_moments",
        lambda total_utils, device: (sum(total_utils), 0.0),
    )
    monkeypatch.setattr(
        rr,
        "_group_softmax",
        lambda values, temperature, eps: [v / sum(values) for v in values],
    )

    device = SimpleNamespace(type="cpu")
    result: RewardComputation = rr.compute_reward_statistics(
        gen_batch,
        reward_spec,
        device,
        q_temperature=1.0,
        q_epsilon=0.0,
    )
    assert result.total_utils == [1.0, 2.0]
    assert result.advantage.grouped[0] == [-0.5, 0.5]
    assert result.q_distribution.grouped[0] == [1.0 / 3.0, 2.0 / 3.0]
    assert result.train_reward_mean == 3.0
    # Flattened metadata should align with completions order.
    assert result.ref_logprob_meta == [["meta1"], ["meta2"]]


def test_flatten_ref_metadata_emits_trl_payload():
    meta = VLLMLogprobResult(
        logprob_sum=-1.0,
        token_count=2,
        token_logprobs=[-0.5, -0.4],
        raw_output={"token_ids": [1, 2]},
    )
    flat = flatten_ref_metadata([["comp"]], [[meta]])
    assert flat == [
        {
            "logprob_sum": -1.0,
            "token_count": 2,
            "token_logprobs": [-0.5, -0.4],
            "raw_output": {"token_ids": [1, 2]},
        }
    ]
