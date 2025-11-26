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

Edge-case coverage for training.pipeline helpers.
"""

from __future__ import annotations

import sys
import types
from collections import defaultdict
from types import SimpleNamespace

import pytest

# Minimal torch/accelerate/transformers stubs so imports succeed.
_TORCH_STUB = types.ModuleType("torch")
_TORCH_STUB.Tensor = type("Tensor", (object,), {})
_TORCH_STUB.device = type("device", (object,), {})
_TORCH_STUB.optim = SimpleNamespace(Optimizer=type("Optimizer", (object,), {}))
_TORCH_STUB.__spec__ = SimpleNamespace()
sys.modules["torch"] = _TORCH_STUB
torch_utils = types.ModuleType("torch.utils")
sys.modules["torch.utils"] = torch_utils
torch_utils_data = types.ModuleType("torch.utils.data")
torch_utils_data.DataLoader = type("DataLoader", (object,), {})
sys.modules["torch.utils.data"] = torch_utils_data
transformers_stub = types.ModuleType("transformers")
transformers_stub.PreTrainedModel = type("PreTrainedModel", (object,), {})
transformers_stub.PreTrainedTokenizer = type("PreTrainedTokenizer", (object,), {})
sys.modules["transformers"] = transformers_stub

import maxent_grpo.training.pipeline as pipeline  # noqa: E402
from maxent_grpo.training.pipeline import (  # noqa: E402,E401
    PreparedBatch,
    _BatchStats,
    _TraceCounter,
    _SkipBatch,
    _collect_batch_stats,
    _reference_stats_from_meta,
    _require_artifact,
    prepare_training_batch,
)


class _FakePromptEntry:
    def __init__(self, length: int):
        self.length = length


class _FakeScoreBatch:
    def __init__(self, total_sequences: int, prompt_lengths=None):
        self.prompt_entries = [
            _FakePromptEntry(length) for length in (prompt_lengths or [])
        ] or None
        self.max_prompt_len = 4
        self.total_sequences = total_sequences
        self.completion_ids = None
        self.completion_attention_mask = None
        self.pad_token_id = 0


class _Ctx:
    def __init__(self):
        self.runtime = SimpleNamespace(device="cpu", tokenizer=None, model=None)
        self.generation = SimpleNamespace(
            max_completion_len=8,
            generation_stats=defaultdict(int),
            vllm_rounds_cfg=1,
        )
        self.optimization = SimpleNamespace(schedule=SimpleNamespace(num_generations=1))
        self.reward = None
        self.scoring = SimpleNamespace(
            batching=SimpleNamespace(),
            weighting=SimpleNamespace(q_temperature=1.0, q_epsilon=1e-6),
        )


def test_trace_counter_limits_and_reset():
    ctr = _TraceCounter(1)
    assert ctr.next_occurrence() == 1
    assert ctr.next_occurrence() is None
    ctr.reset()
    assert ctr.next_occurrence() == 1


def test_prepared_batch_properties():
    stats = _BatchStats(
        score_batch=None,
        ref_stats="ref",
        weight_stats="weights",
        length_stats="lens",
        num_completion_tokens=3.0,
        prompt_token_count=2.0,
    )
    prepared = PreparedBatch(
        grouped_completions=[["x"]],
        reward_comp="rewards",
        batch_stats=stats,
        total_input_tokens=5.0,
        scores="scores",
    )
    assert prepared.weight_stats == "weights"
    assert prepared.ref_stats == "ref"
    assert prepared.length_stats == "lens"
    assert prepared.num_completion_tokens == 3.0


def test_collect_batch_stats_uses_settings_fallback(monkeypatch):
    ctx = SimpleNamespace(
        runtime=SimpleNamespace(device="cpu", tokenizer=None),
        settings=SimpleNamespace(
            scoring=SimpleNamespace(
                batching=SimpleNamespace(),
                weighting=SimpleNamespace(q_temperature=1.0, q_epsilon=1e-6),
            ),
            generation=SimpleNamespace(max_completion_len=4),
        ),
    )
    gen_batch = SimpleNamespace(grouped_completions=[["a"]])
    reward_comp = SimpleNamespace(
        ref_logprob_meta=None, pairs=SimpleNamespace(completions=["a"])
    )

    monkeypatch.setattr(
        pipeline,
        "build_score_batch",
        lambda _reward, _tok, _gen, batching_cfg: (
            setattr(
                ctx,
                "scoring",
                SimpleNamespace(
                    batching=batching_cfg, weighting=ctx.settings.scoring.weighting
                ),
            )
            or setattr(ctx, "generation", ctx.settings.generation)
            or _FakeScoreBatch(total_sequences=1, prompt_lengths=[5])
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "gather_reference_logprobs",
        lambda *_args, **_kwargs: SimpleNamespace(
            ref_logp_sum=None,
            ref_tok_counts=None,
            ref_logp_sum_raw=None,
            ref_logp_mean=0.0,
            avg_completion_tokens=0.0,
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "compute_weight_stats",
        lambda *_args, **_kwargs: SimpleNamespace(
            flat_weights=[1.0], weights_grouped=[], weight_entropy=0.0
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "summarize_completion_lengths",
        lambda *_args, **_kwargs: (None, SimpleNamespace(), 0.0),
    )
    stats = _collect_batch_stats(ctx, gen_batch, reward_comp)
    assert stats is not None
    # prompt length 5 gets clamped by max_prompt_len=4
    assert stats.prompt_token_count == 4.0


def test_collect_batch_stats_uses_ref_meta_when_counts_mismatch(monkeypatch):
    ctx = _Ctx()
    gen_batch = SimpleNamespace(grouped_completions=[["a"], ["b"]])
    reward_comp = SimpleNamespace(
        ref_logprob_meta=["meta1", "meta2", "meta3"],
        pairs=SimpleNamespace(completions=["a", "b"]),
    )

    monkeypatch.setattr(
        pipeline,
        "build_score_batch",
        lambda *_args, **_kwargs: _FakeScoreBatch(total_sequences=3),
    )
    monkeypatch.setattr(
        pipeline,
        "reference_from_vllm_meta",
        lambda meta, total, device: ("ref", meta, total, device),
    )
    monkeypatch.setattr(
        pipeline,
        "compute_weight_stats",
        lambda *_args, **_kwargs: SimpleNamespace(
            flat_weights=[1.0, 1.0, 1.0], weights_grouped=[[1.0]], weight_entropy=0.0
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "summarize_completion_lengths",
        lambda *_args, **_kwargs: (None, SimpleNamespace(), 0.0),
    )
    stats = _collect_batch_stats(ctx, gen_batch, reward_comp)
    assert stats is not None
    assert hasattr(stats.ref_stats, "ref_logp_mean") or isinstance(
        stats.ref_stats, tuple
    )


def test_collect_batch_stats_returns_none_when_ref_missing(monkeypatch):
    ctx = _Ctx()
    gen_batch = SimpleNamespace(grouped_completions=[["a"]])
    reward_comp = SimpleNamespace(
        ref_logprob_meta=None, pairs=SimpleNamespace(completions=["a"])
    )

    monkeypatch.setattr(
        pipeline,
        "build_score_batch",
        lambda *_args, **_kwargs: _FakeScoreBatch(total_sequences=1),
    )
    monkeypatch.setattr(pipeline, "gather_reference_logprobs", lambda *_a, **_k: None)
    monkeypatch.setattr(
        pipeline,
        "compute_weight_stats",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError()),
    )

    assert _collect_batch_stats(ctx, gen_batch, reward_comp) is None


def test_collect_batch_stats_returns_none_when_score_batch_missing(monkeypatch):
    ctx = _Ctx()
    gen_batch = SimpleNamespace(grouped_completions=[["a"]])
    reward_comp = SimpleNamespace(
        ref_logprob_meta=None, pairs=SimpleNamespace(completions=["a"])
    )

    monkeypatch.setattr(pipeline, "build_score_batch", lambda *_a, **_k: None)
    assert _collect_batch_stats(ctx, gen_batch, reward_comp) is None


def test_reference_stats_from_meta_returns_none_for_missing_meta():
    assert _reference_stats_from_meta(None, 1, device="cpu") is None
    assert _reference_stats_from_meta([], 1, device="cpu") is None
    assert _reference_stats_from_meta(["x"], 0, device="cpu") is None


def test_collect_batch_stats_injects_prompt_length_cache(monkeypatch):
    ctx = SimpleNamespace(
        runtime=SimpleNamespace(device="cpu", tokenizer=None),
        settings=SimpleNamespace(
            scoring=SimpleNamespace(
                batching=SimpleNamespace(), weighting=SimpleNamespace()
            ),
            generation=SimpleNamespace(max_completion_len=4),
        ),
    )
    ctx.scoring = ctx.settings.scoring
    ctx.generation = ctx.settings.generation
    gen_batch = SimpleNamespace(grouped_completions=[["a"]])
    reward_comp = SimpleNamespace(
        ref_logprob_meta=None, pairs=SimpleNamespace(completions=["a"])
    )

    def _fake_build_score_batch(_reward, _tok, _gen, batching_cfg):
        # prompt_length_cache_get should be injected when missing
        entry = batching_cfg.prompt_length_cache_get("p")
        assert hasattr(entry, "input_ids") and hasattr(entry, "attention_mask")
        return _FakeScoreBatch(total_sequences=1, prompt_lengths=[3])

    monkeypatch.setattr(pipeline, "build_score_batch", _fake_build_score_batch)
    monkeypatch.setattr(
        pipeline,
        "gather_reference_logprobs",
        lambda *_args, **_kwargs: SimpleNamespace(
            ref_logp_sum=None,
            ref_tok_counts=None,
            ref_logp_sum_raw=None,
            ref_logp_mean=0.0,
            avg_completion_tokens=0.0,
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "compute_weight_stats",
        lambda *_args, **_kwargs: SimpleNamespace(
            flat_weights=[1.0], weights_grouped=[], weight_entropy=0.0
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "summarize_completion_lengths",
        lambda *_args, **_kwargs: (None, SimpleNamespace(), 0.0),
    )
    stats = _collect_batch_stats(ctx, gen_batch, reward_comp)
    assert stats is not None


def test_prepare_training_batch_uses_retry_fallback(monkeypatch):
    ctx = _Ctx()
    ctx.generation.vllm_rounds_cfg = 0
    ctx.optimization.schedule.num_generations = 2

    called = {}

    def _fake_prepare(
        batch, generator, stats, num_generations, max_retry_rounds, **_kwargs
    ):
        called["retry"] = max_retry_rounds
        return SimpleNamespace(grouped_completions=[["a"]], answers=[""])

    monkeypatch.setattr(pipeline, "prepare_generation_batch", _fake_prepare)
    monkeypatch.setattr(
        pipeline,
        "compute_reward_statistics",
        lambda *_args, **_kwargs: SimpleNamespace(
            ref_logprob_meta=None,
            pairs=SimpleNamespace(completions=["a"]),
            completion_metadata=[],
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "_collect_batch_stats",
        lambda *_args, **_kwargs: SimpleNamespace(
            score_batch=_FakeScoreBatch(total_sequences=1),
            ref_stats="ref",
            num_completion_tokens=1.0,
            prompt_token_count=0.0,
        ),
    )
    monkeypatch.setattr(pipeline, "score_model_outputs", lambda *_args, **_kwargs: 0.0)
    monkeypatch.setattr(
        pipeline, "build_sequence_scores", lambda *_args, **_kwargs: SimpleNamespace()
    )

    batch = {"prompt": ["p"], "answer": ["a"]}
    prepared = prepare_training_batch(ctx, lambda *_a, **_k: None, batch)
    assert prepared is not None
    assert called["retry"] == 2


def test_require_artifact_passes_and_raises():
    assert _require_artifact("ok") == "ok"
    with pytest.raises(_SkipBatch):
        _require_artifact(None)


def test_collect_batch_stats_rebuilds_ref_meta_on_sequence_mismatch(monkeypatch):
    ctx = _Ctx()
    gen_batch = SimpleNamespace(grouped_completions=[["x"]])
    reward_comp = SimpleNamespace(
        ref_logprob_meta=["m1", "m2"], pairs=SimpleNamespace(completions=[])
    )
    ref_calls: dict[str, tuple] = {}

    monkeypatch.setattr(
        pipeline,
        "build_score_batch",
        lambda *_args, **_kwargs: _FakeScoreBatch(
            total_sequences=2, prompt_lengths=[2, 5]
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "reference_from_vllm_meta",
        lambda meta, total, device: ref_calls.setdefault("args", (meta, total, device))
        or SimpleNamespace(
            ref_logp_sum=None,
            ref_tok_counts=None,
            ref_logp_sum_raw=None,
            ref_logp_mean=0.0,
            avg_completion_tokens=0.0,
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "gather_reference_logprobs",
        lambda *_args, **_kwargs: ref_calls.setdefault("gather", True),
    )
    monkeypatch.setattr(
        pipeline,
        "compute_weight_stats",
        lambda *_args, **_kwargs: SimpleNamespace(
            flat_weights=[1.0, 1.0], weights_grouped=[[1.0, 1.0]], weight_entropy=0.0
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "summarize_completion_lengths",
        lambda *_args, **_kwargs: (None, SimpleNamespace(), 3.0),
    )

    stats = _collect_batch_stats(ctx, gen_batch, reward_comp)
    assert stats is not None
    assert ref_calls["args"] == (["m1", "m2"], 2, "cpu")
    assert ref_calls.get("gather") is not True
    # prompt lengths are clamped by _FakeScoreBatch.max_prompt_len == 4
    assert stats.prompt_token_count == 6.0


def test_collect_batch_stats_logs_and_skips_on_reference_failure(monkeypatch, caplog):
    ctx = _Ctx()
    gen_batch = SimpleNamespace(grouped_completions=[["a"]])
    reward_comp = SimpleNamespace(
        ref_logprob_meta=None, pairs=SimpleNamespace(completions=["a"])
    )

    monkeypatch.setattr(
        pipeline,
        "build_score_batch",
        lambda *_args, **_kwargs: _FakeScoreBatch(total_sequences=1),
    )
    limiter = _TraceCounter(1)
    monkeypatch.setattr(pipeline, "_REF_LOGPROB_TRACE_LIMITER", limiter)
    monkeypatch.setattr(
        pipeline,
        "gather_reference_logprobs",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    monkeypatch.setattr(
        pipeline,
        "compute_weight_stats",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("unexpected weighting")
        ),
    )

    with caplog.at_level("WARNING"):
        assert _collect_batch_stats(ctx, gen_batch, reward_comp) is None
    assert limiter.next_occurrence() is None


def test_collect_batch_stats_returns_none_when_reference_missing_after_gather(
    monkeypatch,
):
    ctx = _Ctx()
    gen_batch = SimpleNamespace(grouped_completions=[["a"]])
    reward_comp = SimpleNamespace(
        ref_logprob_meta=None, pairs=SimpleNamespace(completions=["a"])
    )

    monkeypatch.setattr(
        pipeline,
        "build_score_batch",
        lambda *_args, **_kwargs: _FakeScoreBatch(total_sequences=1),
    )
    monkeypatch.setattr(
        pipeline, "gather_reference_logprobs", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        pipeline,
        "compute_weight_stats",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("should not compute weights")
        ),
    )

    assert _collect_batch_stats(ctx, gen_batch, reward_comp) is None


def test_collect_batch_stats_returns_none_when_weights_empty(monkeypatch):
    ctx = _Ctx()
    gen_batch = SimpleNamespace(grouped_completions=[["a"]])
    reward_comp = SimpleNamespace(
        ref_logprob_meta=None, pairs=SimpleNamespace(completions=["a"])
    )
    ref_obj = SimpleNamespace(
        ref_logp_sum=None,
        ref_tok_counts=None,
        ref_logp_sum_raw=None,
        ref_logp_mean=0.0,
        avg_completion_tokens=0.0,
    )

    monkeypatch.setattr(
        pipeline,
        "build_score_batch",
        lambda *_args, **_kwargs: _FakeScoreBatch(total_sequences=1),
    )
    monkeypatch.setattr(
        pipeline, "gather_reference_logprobs", lambda *_args, **_kwargs: ref_obj
    )
    monkeypatch.setattr(
        pipeline,
        "compute_weight_stats",
        lambda *_args, **_kwargs: SimpleNamespace(flat_weights=[]),
    )

    assert _collect_batch_stats(ctx, gen_batch, reward_comp) is None


def test_prepare_training_batch_full_flow(monkeypatch):
    ctx = _Ctx()
    gen_batch = SimpleNamespace(grouped_completions=[["c"]], answers=["c"])
    reward_comp = SimpleNamespace(
        ref_logprob_meta=None, pairs=SimpleNamespace(completions=["c"])
    )
    stats_obj = SimpleNamespace(
        score_batch=_FakeScoreBatch(total_sequences=1, prompt_lengths=[3]),
        ref_stats=SimpleNamespace(ref_logp_mean=0.0, avg_completion_tokens=1.0),
        weight_stats=SimpleNamespace(flat_weights=[1.0]),
        length_stats=SimpleNamespace(),
        num_completion_tokens=5.0,
        prompt_token_count=2.0,
    )

    monkeypatch.setattr(
        pipeline,
        "prepare_generation_batch",
        lambda *_args, **_kwargs: gen_batch,
    )
    monkeypatch.setattr(
        pipeline,
        "compute_reward_statistics",
        lambda *_args, **_kwargs: reward_comp,
    )
    reward_comp.completion_metadata = []
    monkeypatch.setattr(
        pipeline, "_collect_batch_stats", lambda *_args, **_kwargs: stats_obj
    )
    monkeypatch.setattr(pipeline, "score_model_outputs", lambda *_args, **_kwargs: 0.5)
    monkeypatch.setattr(
        pipeline,
        "build_sequence_scores",
        lambda cur_logp_sum, ref_stats, pooled_hidden=None: {
            "cur": cur_logp_sum,
            "ref": ref_stats,
            "hidden": pooled_hidden,
        },
    )

    prepared = prepare_training_batch(
        ctx, lambda *_a, **_k: None, {"prompt": ["p"], "answer": ["a"]}
    )
    assert isinstance(prepared, PreparedBatch)
    assert prepared.total_input_tokens == 7.0
    assert prepared.scores["cur"] == 0.5
