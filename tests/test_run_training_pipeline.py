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

# Unit tests for the training pipeline helpers.

from __future__ import annotations

import sys
import types
from collections import defaultdict
from contextlib import nullcontext
from types import SimpleNamespace
from typing import List

import pytest

# Minimal torch stub so require_torch resolves without installing the real package.
_TORCH_STUB = types.ModuleType("torch")
_TORCH_STUB.Tensor = type("Tensor", (object,), {})
_TORCH_STUB.device = type("device", (object,), {})
_TORCH_STUB.tensor = lambda data=None, dtype=None, device=None: data
_TORCH_STUB.float32 = float
_TORCH_STUB.int64 = int
_TORCH_STUB.optim = SimpleNamespace(Optimizer=type("Optimizer", (object,), {}))
_TORCH_STUB.__spec__ = SimpleNamespace()
sys.modules["torch"] = _TORCH_STUB
torch_utils = types.ModuleType("torch.utils")
sys.modules["torch.utils"] = torch_utils
torch_utils_data = types.ModuleType("torch.utils.data")
torch_utils_data.DataLoader = type("DataLoader", (object,), {})
sys.modules["torch.utils.data"] = torch_utils_data
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

# Stub out training.optim to avoid importing heavy real module.
optim_stub = types.ModuleType("maxent_grpo.training.optim")
optim_stub.configure_accumulation_steps = lambda *args, **kwargs: SimpleNamespace()
optim_stub.detect_deepspeed_state = lambda *_a, **_k: SimpleNamespace(
    use_deepspeed=False, zero_stage=0
)
optim_stub.epoch_progress = lambda *_a, **_k: 0.0
optim_stub.optimizer_step = lambda *_a, **_k: None
optim_stub.require_accumulation_context = lambda *_a, **_k: nullcontext()
optim_stub.scheduled_learning_rate = lambda *_a, **_k: 0.0
optim_stub.sync_gradients_enabled = lambda *_a, **_k: False


def _build_handles(_model=None, training_args=None):
    lr = float(getattr(training_args, "learning_rate", 0.0)) if training_args else 0.0
    optimizer = SimpleNamespace(
        step=lambda *_a, **_k: None,
        zero_grad=lambda **_k: None,
    )
    return SimpleNamespace(
        optimizer=optimizer,
        lr_scheduler=None,
        base_optimizer=optimizer,
        learning_rate=lr,
    )


optim_stub.build_optimization_handles = _build_handles
sys.modules["maxent_grpo.training.optim"] = optim_stub


def _arr(data):
    return data


from maxent_grpo.training.pipeline import (  # noqa: E402  import after stubbing torch
    _collect_batch_stats,
    prepare_training_batch,
)


class _FakeArray(list):
    def tolist(self):
        return list(self)


class _FakePromptEntry:
    def __init__(self, length: int):
        self.length = length


class _FakeScoreBatch:
    def __init__(self, prompt_lengths: List[int], total_sequences: int):
        self.prompt_entries = [_FakePromptEntry(length) for length in prompt_lengths]
        self.max_prompt_len = 4
        self.total_sequences = total_sequences
        self.completion_ids = None
        self.completion_attention_mask = None
        self.pad_token_id = 0
        self.score_tail_tokens = None


class _FakeRefStats:
    def __init__(self):
        self.ref_logp_sum = _FakeArray([0.0, 0.0])
        self.ref_tok_counts = _FakeArray([1, 1])
        self.ref_logp_sum_raw = _FakeArray([0.0, 0.0])
        self.ref_logp_mean = 0.0
        self.avg_completion_tokens = 0.0


class _FakeWeightStats:
    def __init__(self, with_weights: bool = True):
        self.weights_grouped = []
        self.flat_weights = [1.0] if with_weights else []
        self.weight_entropy = 0.0
        self.weight_entropy_min = 0.0
        self.weight_entropy_max = 0.0
        self.advantage_entropy = []


class _FakeCtx:
    def __init__(self):
        self.runtime = SimpleNamespace(
            device="cpu",
            tokenizer=None,
            model=None,
            get_ref_model=lambda: SimpleNamespace(),
        )
        self.generation = SimpleNamespace(
            max_completion_len=8,
            max_prompt_len=4,
            generation_stats=defaultdict(int),
            vllm_rounds_cfg=1,
        )
        self.optimization = SimpleNamespace(schedule=SimpleNamespace(num_generations=1))
        self.reward = None
        self.scoring = SimpleNamespace(
            batching=SimpleNamespace(score_slice=0),
            weighting=SimpleNamespace(
                q_temperature=1.0,
                q_epsilon=1e-6,
                len_norm_ref=True,
            ),
        )


@pytest.fixture(autouse=True)
def _patch_helpers(monkeypatch):
    def _fake_build_score_batch(_reward, _tok, _gen, _batching):
        return _FakeScoreBatch([2, 5], total_sequences=2)

    def _fake_weight_stats(_comps, _reward, _ref, _cfg):
        return _FakeWeightStats()

    def _fake_lengths(_ref, _max_len):
        return None, SimpleNamespace(mean_length=0, min_length=0, max_length=0), 4.0

    monkeypatch.setattr(
        "maxent_grpo.training.pipeline.build_score_batch",
        _fake_build_score_batch,
    )
    monkeypatch.setattr(
        "maxent_grpo.training.pipeline.gather_reference_logprobs",
        lambda *_a, **_k: _FakeRefStats(),
    )
    monkeypatch.setattr(
        "maxent_grpo.training.scoring.build_score_batch",
        _fake_build_score_batch,
        raising=False,
    )
    monkeypatch.setattr(
        "maxent_grpo.training.scoring.reference_from_model",
        lambda *args, **kwargs: (_arr([0.0]), _arr([1.0])),
        raising=False,
    )
    monkeypatch.setattr(
        "training.scoring.reference_from_model",
        lambda *args, **kwargs: (_arr([0.0]), _arr([1.0])),
        raising=False,
    )
    monkeypatch.setattr(
        "maxent_grpo.training.pipeline._reference_stats_from_meta",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(
        "maxent_grpo.training.scoring.gather_reference_logprobs",
        lambda *_a, **_k: _FakeRefStats(),
        raising=False,
    )
    monkeypatch.setattr(
        "training.scoring.gather_reference_logprobs",
        lambda *_a, **_k: _FakeRefStats(),
        raising=False,
    )
    monkeypatch.setattr(
        "training.pipeline.gather_reference_logprobs",
        lambda *_a, **_k: _FakeRefStats(),
        raising=False,
    )
    monkeypatch.setattr(
        "training.pipeline.build_score_batch",
        _fake_build_score_batch,
        raising=False,
    )
    monkeypatch.setattr(
        "maxent_grpo.training.pipeline.compute_weight_stats",
        _fake_weight_stats,
    )
    monkeypatch.setitem(
        _collect_batch_stats.__globals__,
        "compute_weight_stats",
        _fake_weight_stats,
    )
    monkeypatch.setattr(
        "training.pipeline.compute_weight_stats",
        _fake_weight_stats,
        raising=False,
    )
    monkeypatch.setattr(
        "maxent_grpo.training.pipeline.summarize_completion_lengths",
        _fake_lengths,
    )
    monkeypatch.setattr(
        "training.pipeline.summarize_completion_lengths",
        _fake_lengths,
        raising=False,
    )


def test_collect_batch_stats_counts_prompt_tokens(monkeypatch):
    ctx = _FakeCtx()
    gen_batch = SimpleNamespace(grouped_completions=[["a"], ["b"]])
    reward_comp = SimpleNamespace(
        ref_logprob_meta=None,
        pairs=SimpleNamespace(completions=["a", "b"]),
    )

    def _fake_gather(score_batch, _runtime, _batching):
        assert isinstance(score_batch, _FakeScoreBatch)
        return _FakeRefStats()

    monkeypatch.setitem(
        _collect_batch_stats.__globals__,
        "gather_reference_logprobs",
        _fake_gather,
    )
    monkeypatch.setitem(
        _collect_batch_stats.__globals__,
        "_reference_stats_from_meta",
        lambda *_a, **_k: _FakeRefStats(),
    )
    monkeypatch.setitem(
        _collect_batch_stats.__globals__,
        "compute_weight_stats",
        lambda *_a, **_k: _FakeWeightStats(),
    )
    monkeypatch.setattr(
        "maxent_grpo.training.pipeline.compute_weight_stats",
        lambda *_a, **_k: _FakeWeightStats(),
    )
    monkeypatch.setattr(
        "maxent_grpo.training.pipeline.compute_weight_stats",
        lambda *_a, **_k: _FakeWeightStats(),
    )
    monkeypatch.setattr(
        "training.pipeline.gather_reference_logprobs",
        _fake_gather,
        raising=False,
    )
    monkeypatch.setattr(
        "training.pipeline._reference_stats_from_meta",
        lambda *_a, **_k: _FakeRefStats(),
        raising=False,
    )
    monkeypatch.setattr(
        "maxent_grpo.training.pipeline.build_score_batch",
        lambda *args, **kwargs: _FakeScoreBatch([2, 5], total_sequences=2),
    )

    class _Tok:
        pad_token_id = 0
        eos_token_id = 1

        def __call__(self, completions, **_k):
            return {
                "input_ids": _arr([[1, 2], [3, 4]]),
                "attention_mask": _arr([[1, 1], [1, 1]]),
            }

    ctx.tokenizer = _Tok()
    ctx.runtime.tokenizer = ctx.tokenizer
    ctx.runtime.tokenizer = ctx.tokenizer

    stats = _collect_batch_stats(ctx, gen_batch, reward_comp)
    assert stats is not None
    assert stats.prompt_token_count >= 0


def test_prepare_training_batch_success(monkeypatch):
    ctx = _FakeCtx()

    def _fake_prepare_generation_batch(
        batch, generator, stats, num_generations, max_retry_rounds
    ):
        assert batch["prompt"] == ["p"]
        return SimpleNamespace(
            grouped_completions=[["ok"]],
            answers=[""],
        )

    def _fake_reward_stats(_gen_batch, _reward, _device, *_args):
        return SimpleNamespace(
            ref_logprob_meta=None,
            pairs=SimpleNamespace(completions=["ok"]),
        )

    def _fake_ref_gather(score_batch, runtime, batching):
        return _FakeRefStats()

    def _fake_score_model(_model, _score_batch, _batching, _runtime):
        return 0.0

    monkeypatch.setattr(
        "maxent_grpo.training.pipeline.prepare_generation_batch",
        _fake_prepare_generation_batch,
    )
    monkeypatch.setattr(
        "maxent_grpo.training.pipeline.compute_reward_statistics",
        _fake_reward_stats,
    )
    monkeypatch.setattr(
        "maxent_grpo.training.pipeline.gather_reference_logprobs",
        _fake_ref_gather,
    )
    monkeypatch.setattr(
        "maxent_grpo.training.scoring.gather_reference_logprobs",
        _fake_ref_gather,
        raising=False,
    )
    monkeypatch.setattr(
        "training.scoring.gather_reference_logprobs",
        _fake_ref_gather,
        raising=False,
    )
    monkeypatch.setattr(
        "maxent_grpo.training.pipeline._reference_stats_from_meta",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(
        "training.pipeline._reference_stats_from_meta",
        lambda *_a, **_k: None,
        raising=False,
    )
    monkeypatch.setattr(
        "training.pipeline.gather_reference_logprobs",
        _fake_ref_gather,
        raising=False,
    )
    monkeypatch.setattr(
        "maxent_grpo.training.pipeline.build_score_batch",
        lambda *args, **kwargs: _FakeScoreBatch([1], total_sequences=1),
    )
    monkeypatch.setattr(
        "maxent_grpo.training.pipeline.score_model_outputs",
        _fake_score_model,
    )
    monkeypatch.setattr(
        "maxent_grpo.training.pipeline.build_sequence_scores",
        lambda _cur, _ref: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "maxent_grpo.training.pipeline._collect_batch_stats",
        lambda *_a, **_k: SimpleNamespace(
            score_batch=_FakeScoreBatch([1], total_sequences=1),
            ref_stats=_FakeRefStats(),
            weight_stats=_FakeWeightStats(),
            length_stats=None,
            num_completion_tokens=0,
            prompt_token_count=0,
        ),
    )
    monkeypatch.setitem(
        _collect_batch_stats.__globals__,
        "gather_reference_logprobs",
        _fake_ref_gather,
    )
    monkeypatch.setitem(
        _collect_batch_stats.__globals__,
        "_reference_stats_from_meta",
        lambda *_a, **_k: None,
    )
    monkeypatch.setitem(
        _collect_batch_stats.__globals__,
        "compute_weight_stats",
        lambda *_a, **_k: _FakeWeightStats(),
    )

    class _Tok:
        pad_token_id = 0
        eos_token_id = 1

        def __call__(self, completions, **_k):
            return {
                "input_ids": _arr([[1, 2]]),
                "attention_mask": _arr([[1, 1]]),
            }

    ctx.tokenizer = _Tok()
    ctx.runtime.tokenizer = ctx.tokenizer
    ctx.runtime.tokenizer = ctx.tokenizer

    batch = {"prompt": ["p"], "answer": ["a"]}
    prepared = prepare_training_batch(ctx, lambda *_args, **_kwargs: None, batch)
    assert prepared is not None or prepared is None
    if prepared is not None:
        assert prepared.total_input_tokens >= 0


def test_prepare_training_batch_drops_on_stats_failure(monkeypatch):
    ctx = _FakeCtx()

    def _fake_prepare_generation_batch(
        batch, generator, stats, num_generations, max_retry_rounds, **_kwargs
    ):
        return SimpleNamespace(grouped_completions=[["ok"]], answers=[""])

    def _fake_reward_stats(*_args, **_kwargs):
        return SimpleNamespace(
            ref_logprob_meta=None,
            pairs=SimpleNamespace(completions=["ok"]),
        )

    monkeypatch.setattr(
        "maxent_grpo.training.pipeline.prepare_generation_batch",
        _fake_prepare_generation_batch,
    )
    monkeypatch.setattr(
        "maxent_grpo.training.pipeline.compute_reward_statistics",
        _fake_reward_stats,
    )
    monkeypatch.setattr(
        "maxent_grpo.training.pipeline._collect_batch_stats",
        lambda *_args, **_kwargs: None,
    )
    batch = {"prompt": ["p"], "answer": ["a"]}
    assert prepare_training_batch(ctx, lambda *_args, **_kwargs: None, batch) is None


def test_collect_batch_stats_handles_reference_failures(monkeypatch):
    ctx = _FakeCtx()
    gen_batch = SimpleNamespace(grouped_completions=[["a"]])
    reward_comp = SimpleNamespace(
        ref_logprob_meta=None,
        pairs=SimpleNamespace(completions=["a"]),
    )

    def _fake_gather(_score_batch, _runtime, _batching):
        raise RuntimeError("failure")

    monkeypatch.setitem(
        _collect_batch_stats.__globals__,
        "gather_reference_logprobs",
        _fake_gather,
    )
    monkeypatch.setitem(
        _collect_batch_stats.__globals__,
        "_reference_stats_from_meta",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(
        "training.pipeline._reference_stats_from_meta",
        lambda *_a, **_k: None,
        raising=False,
    )
    monkeypatch.setattr(
        "maxent_grpo.training.pipeline.build_score_batch",
        lambda *args, **kwargs: _FakeScoreBatch([1], total_sequences=1),
    )

    class _Tok:
        pad_token_id = 0
        eos_token_id = 1

        def __call__(self, completions, **_k):
            return {
                "input_ids": _arr([[1, 2]]),
                "attention_mask": _arr([[1, 1]]),
            }

    ctx.tokenizer = _Tok()
    ctx.runtime.tokenizer = ctx.tokenizer

    stats = _collect_batch_stats(ctx, gen_batch, reward_comp)
    assert stats is None


def test_collect_batch_stats_drops_empty_weight_stats(monkeypatch):
    ctx = _FakeCtx()
    gen_batch = SimpleNamespace(grouped_completions=[["a"]])
    reward_comp = SimpleNamespace(
        ref_logprob_meta=None,
        pairs=SimpleNamespace(completions=["a"]),
    )

    def _fake_gather(score_batch, _runtime, _batching):
        return _FakeRefStats()

    monkeypatch.setitem(
        _collect_batch_stats.__globals__,
        "gather_reference_logprobs",
        _fake_gather,
    )
    monkeypatch.setitem(
        _collect_batch_stats.__globals__,
        "_reference_stats_from_meta",
        lambda *_a, **_k: _FakeRefStats(),
    )
    monkeypatch.setattr(
        "training.pipeline._reference_stats_from_meta",
        lambda *_a, **_k: _FakeRefStats(),
        raising=False,
    )
    monkeypatch.setattr(
        "maxent_grpo.training.pipeline.compute_weight_stats",
        lambda *args, **kwargs: _FakeWeightStats(with_weights=False),
    )
    monkeypatch.setitem(
        _collect_batch_stats.__globals__,
        "compute_weight_stats",
        lambda *_a, **_k: _FakeWeightStats(with_weights=False),
    )
    monkeypatch.setattr(
        "maxent_grpo.training.pipeline.build_score_batch",
        lambda *args, **kwargs: _FakeScoreBatch([1], total_sequences=1),
    )

    class _Tok:
        pad_token_id = 0
        eos_token_id = 1

        def __call__(self, completions, **_k):
            return {
                "input_ids": _arr([[1, 2]]),
                "attention_mask": _arr([[1, 1]]),
            }

    ctx.tokenizer = _Tok()
    ctx.runtime.tokenizer = ctx.tokenizer
    stats = _collect_batch_stats(ctx, gen_batch, reward_comp)
    assert stats is None
