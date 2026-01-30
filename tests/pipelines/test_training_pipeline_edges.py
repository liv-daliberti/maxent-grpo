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
from maxent_grpo.generation.errors import (
    GenerationServiceError,
    ServiceErrorPayload,
)

# Placeholders populated by the module-scoped fixture below.
pipeline = None
PreparedBatch = None
_BatchStats = None
_TraceCounter = None
_SkipBatch = None
_collect_batch_stats = None
_reference_stats_from_meta = None
_require_artifact = None
prepare_training_batch = None


@pytest.fixture(scope="module", autouse=True)
def _pipeline_with_stubs():
    """Install lightweight stubs while importing pipeline, then restore originals."""

    orig_modules = {
        "torch": sys.modules.pop("torch", None),
        "torch.utils": sys.modules.pop("torch.utils", None),
        "torch.utils.data": sys.modules.pop("torch.utils.data", None),
        "transformers": sys.modules.pop("transformers", None),
    }

    torch_stub = types.ModuleType("torch")
    torch_stub.Tensor = type("Tensor", (object,), {})
    torch_stub.device = lambda *_a, **_k: SimpleNamespace(type="cpu")
    torch_stub.optim = SimpleNamespace(Optimizer=type("Optimizer", (object,), {}))
    torch_stub.nn = SimpleNamespace(
        Module=type("Module", (object,), {}),
        Linear=lambda *a, **k: SimpleNamespace(
            weight=None, __call__=lambda *_a, **_k: None
        ),
        Embedding=lambda *a, **k: SimpleNamespace(
            weight=None, __call__=lambda *_a, **_k: None
        ),
    )
    torch_stub.__spec__ = SimpleNamespace()
    sys.modules["torch"] = torch_stub

    torch_utils = types.ModuleType("torch.utils")
    sys.modules["torch.utils"] = torch_utils
    torch_utils_data = types.ModuleType("torch.utils.data")
    torch_utils_data.DataLoader = type("DataLoader", (object,), {})
    sys.modules["torch.utils.data"] = torch_utils_data

    transformers_stub = types.ModuleType("transformers")
    transformers_stub.PreTrainedModel = type("PreTrainedModel", (object,), {})
    transformers_stub.PreTrainedTokenizer = type("PreTrainedTokenizer", (object,), {})
    import maxent_grpo.training.pipeline as pipeline_mod  # noqa: E402

    globals().update(
        pipeline=pipeline_mod,
        PreparedBatch=pipeline_mod.PreparedBatch,
        _BatchStats=pipeline_mod._BatchStats,
        _TraceCounter=pipeline_mod._TraceCounter,
        _SkipBatch=pipeline_mod._SkipBatch,
        _collect_batch_stats=pipeline_mod._collect_batch_stats,
        _reference_stats_from_meta=pipeline_mod._reference_stats_from_meta,
        _require_artifact=pipeline_mod._require_artifact,
        prepare_training_batch=pipeline_mod.prepare_training_batch,
    )

    yield

    # Restore original modules after this test module completes.
    for name, mod in orig_modules.items():
        if mod is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = mod


class _BehaviorTensorStub:
    """Lightweight tensor stub storing flattened log-prob entries."""

    def __init__(self, values, device="cpu", dtype="float32"):
        self.arr = list(values)
        self.device = device
        self.dtype = dtype

    def view(self, *_args, **_kwargs):
        return self


class _LogpTensorStub:
    """Stub mimicking the current-policy log-prob tensor."""

    def __init__(self, value=0.0, device="cpu", dtype="float32"):
        self.value = value
        self.device = device
        self.dtype = dtype

    def new_tensor(self, values, **_kwargs):
        return _BehaviorTensorStub(values, device=self.device, dtype=self.dtype)


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
        self.score_tail_tokens = None


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
    monkeypatch.setattr(
        pipeline,
        "score_model_outputs",
        lambda *_args, **_kwargs: _LogpTensorStub(0.0),
    )
    monkeypatch.setattr(
        pipeline, "build_sequence_scores", lambda *_args, **_kwargs: SimpleNamespace()
    )

    batch = {"prompt": ["p"], "answer": ["a"]}
    prepared = prepare_training_batch(ctx, lambda *_a, **_k: None, batch)
    assert prepared is not None
    assert called["retry"] == 2


def test_require_artifact_passes_and_raises():
    assert _require_artifact("ok", stage="test") == "ok"
    with pytest.raises(_SkipBatch):
        _require_artifact(None, stage="test")


def test_prepare_training_batch_logs_generation_error(caplog):
    ctx = _Ctx()
    batch = {"prompt": ["p"], "answer": ["a"]}
    payload = ServiceErrorPayload(
        service="vllm",
        endpoint="http://host",
        model="model",
        prompt_count=1,
        payload_chars=1,
        payload_size_bytes=10,
        status_code=503,
        attempt=2,
        max_attempts=2,
        exception_type="RuntimeError",
        exception_message="boom",
        request_id="req",
    )

    def _generator(*_args, **_kwargs):
        raise GenerationServiceError("boom", payload)

    caplog.set_level("ERROR")
    with pytest.raises(GenerationServiceError):
        prepare_training_batch(ctx, _generator, batch)
    assert "Generation service failure" in caplog.text


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


def test_collect_batch_stats_skips_when_reference_missing_but_cached(monkeypatch):
    ctx = _Ctx()
    gen_batch = SimpleNamespace(grouped_completions=[["a"]])
    reward_comp = SimpleNamespace(
        ref_logprob_meta=None, pairs=SimpleNamespace(completions=["a"])
    )
    ctx._last_ref_stats = SimpleNamespace(
        ref_logp_sum=[-1.0],
        ref_logp_sum_raw=[-1.0],
        ref_tok_counts=[1.0],
        ref_logp_mean=-1.0,
        avg_completion_tokens=1.0,
    )
    ctx.scoring.allow_stale_reference_logprobs = False

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
            AssertionError("should not compute weights when skipping batch")
        ),
    )

    assert _collect_batch_stats(ctx, gen_batch, reward_comp) is None
    assert getattr(ctx.runtime, "_last_skip_stage", None) == "reference_logprobs"


def test_collect_batch_stats_allows_stale_reference_when_enabled(monkeypatch):
    ctx = _Ctx()
    gen_batch = SimpleNamespace(grouped_completions=[["a"]])
    reward_comp = SimpleNamespace(
        ref_logprob_meta=None, pairs=SimpleNamespace(completions=["a"])
    )
    ctx._last_ref_stats = SimpleNamespace(
        ref_logp_sum=[-1.0],
        ref_logp_sum_raw=[-1.0],
        ref_tok_counts=[1.0],
        ref_logp_mean=-1.0,
        avg_completion_tokens=1.0,
    )
    ctx.scoring.allow_stale_reference_logprobs = True

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
        lambda *_args, **_kwargs: SimpleNamespace(
            flat_weights=[1.0], weights_grouped=[[1.0]], weight_entropy=0.0
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "summarize_completion_lengths",
        lambda *_args, **_kwargs: (None, SimpleNamespace(clipped_ratio=0.0), 1.0),
    )

    stats = _collect_batch_stats(ctx, gen_batch, reward_comp)
    assert stats is not None
    assert stats.ref_stats is ctx._last_ref_stats


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


def test_collect_batch_stats_falls_back_to_uniform_weights(monkeypatch):
    ctx = _Ctx()
    ctx.scoring.weighting.allow_empty_weight_fallback = True
    gen_batch = SimpleNamespace(grouped_completions=[["a", "b"]])
    reward_comp = SimpleNamespace(
        ref_logprob_meta=None, pairs=SimpleNamespace(completions=["a", "b"])
    )
    ref_obj = SimpleNamespace(
        ref_logp_sum=None,
        ref_tok_counts=None,
        ref_logp_sum_raw=None,
        ref_logp_mean=0.0,
        avg_completion_tokens=0.0,
    )
    fallback_weights = SimpleNamespace(flat_weights=[0.5, 0.5])

    monkeypatch.setattr(
        pipeline,
        "build_score_batch",
        lambda *_args, **_kwargs: _FakeScoreBatch(total_sequences=2),
    )
    monkeypatch.setattr(
        pipeline,
        "gather_reference_logprobs",
        lambda *_args, **_kwargs: ref_obj,
    )
    monkeypatch.setattr(
        pipeline,
        "compute_weight_stats",
        lambda *_args, **_kwargs: SimpleNamespace(flat_weights=[]),
    )
    monkeypatch.setattr(
        pipeline,
        "build_uniform_weight_stats",
        lambda *_args, **_kwargs: fallback_weights,
    )
    monkeypatch.setattr(
        pipeline,
        "summarize_completion_lengths",
        lambda *_args, **_kwargs: (None, SimpleNamespace(), 0.0),
    )

    stats = _collect_batch_stats(ctx, gen_batch, reward_comp)
    assert stats.weight_stats is fallback_weights


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
    cur_stub = _LogpTensorStub(0.5)
    monkeypatch.setattr(
        pipeline,
        "score_model_outputs",
        lambda *_args, **_kwargs: cur_stub,
    )
    monkeypatch.setattr(
        pipeline,
        "build_sequence_scores",
        lambda cur_logp_sum, ref_stats, pooled_hidden=None, **_kw: {
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
    assert prepared.scores["cur"] is cur_stub


def test_prepare_training_batch_uses_behavior_meta(monkeypatch):
    ctx = _Ctx()
    gen_batch = SimpleNamespace(grouped_completions=[["c"]], answers=["c"])
    reward_comp = SimpleNamespace(
        ref_logprob_meta=[{"logprob_sum": -1.5}],
        pairs=SimpleNamespace(completions=["c"]),
        completion_metadata=[],
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
    monkeypatch.setattr(
        pipeline, "_collect_batch_stats", lambda *_args, **_kwargs: stats_obj
    )
    cur_stub = _LogpTensorStub(-0.2)
    monkeypatch.setattr(
        pipeline,
        "score_model_outputs",
        lambda *_args, **_kwargs: cur_stub,
    )
    captured = {}

    def _capture_scores(cur_logp_sum, ref_stats, pooled_hidden=None, **kwargs):
        captured["behavior"] = kwargs.get("behavior_logp_sum")
        return {
            "cur": cur_logp_sum,
            "ref": ref_stats,
            "hidden": pooled_hidden,
            "behavior": kwargs.get("behavior_logp_sum"),
        }

    monkeypatch.setattr(pipeline, "build_sequence_scores", _capture_scores)

    prepared = prepare_training_batch(
        ctx, lambda *_a, **_k: None, {"prompt": ["p"], "answer": ["a"]}
    )
    assert isinstance(prepared, PreparedBatch)
    behavior = captured.get("behavior")
    assert behavior is not None
    assert getattr(behavior, "arr", None) == [-1.5]


def test_prepare_training_batch_records_generation_stage(monkeypatch):
    ctx = _Ctx()

    monkeypatch.setattr(
        pipeline,
        "prepare_generation_batch",
        lambda *_a, **_k: None,
    )

    result = prepare_training_batch(
        ctx, lambda *_a, **_k: None, {"prompt": ["p"], "answer": ["a"]}
    )
    assert result is None
    assert getattr(ctx.runtime, "_last_skip_stage") == "generation"


def test_prepare_training_batch_records_policy_scoring_stage(monkeypatch):
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
        lambda *_a, **_k: gen_batch,
    )
    monkeypatch.setattr(
        pipeline,
        "compute_reward_statistics",
        lambda *_a, **_k: reward_comp,
    )
    monkeypatch.setattr(
        pipeline,
        "_collect_batch_stats",
        lambda *_a, **_k: stats_obj,
    )
    monkeypatch.setattr(
        pipeline,
        "score_model_outputs",
        lambda *_a, **_k: None,
    )

    result = prepare_training_batch(
        ctx, lambda *_a, **_k: None, {"prompt": ["p"], "answer": ["a"]}
    )
    assert result is None
    assert getattr(ctx.runtime, "_last_skip_stage") == "policy_scoring"


def test_prepare_training_batch_records_reward_stage(monkeypatch):
    ctx = _Ctx()
    gen_batch = SimpleNamespace(grouped_completions=[[]], answers=["a"])

    monkeypatch.setattr(
        pipeline,
        "prepare_generation_batch",
        lambda *_a, **_k: gen_batch,
    )
    monkeypatch.setattr(
        pipeline,
        "compute_reward_statistics",
        lambda *_a, **_k: None,
    )

    result = prepare_training_batch(
        ctx, lambda *_a, **_k: None, {"prompt": ["p"], "answer": ["a"]}
    )
    assert result is None
    assert getattr(ctx.runtime, "_last_skip_stage") == "reward_stats"


def test_prepare_training_batch_records_batch_stats_stage(monkeypatch):
    ctx = _Ctx()
    gen_batch = SimpleNamespace(grouped_completions=[["c"]], answers=["a"])
    reward_comp = SimpleNamespace(
        ref_logprob_meta=None, pairs=SimpleNamespace(completions=["c"])
    )

    monkeypatch.setattr(
        pipeline,
        "prepare_generation_batch",
        lambda *_a, **_k: gen_batch,
    )
    monkeypatch.setattr(
        pipeline,
        "compute_reward_statistics",
        lambda *_a, **_k: reward_comp,
    )
    monkeypatch.setattr(pipeline, "_collect_batch_stats", lambda *_a, **_k: None)

    result = prepare_training_batch(
        ctx, lambda *_a, **_k: None, {"prompt": ["p"], "answer": ["a"]}
    )
    assert result is None
    assert getattr(ctx.runtime, "_last_skip_stage") == "batch_stats"


def test_behavior_logp_tensor_from_meta_accepts_dict_entries():
    template = _LogpTensorStub()
    meta = [{"logprob_sum": -1.0}, {"logprob_sum": -2.5}]
    out = pipeline._behavior_logp_tensor_from_meta(meta, 2, template)
    assert isinstance(out, _BehaviorTensorStub)
    assert out.arr == [-1.0, -2.5]


def test_behavior_logp_tensor_from_meta_accepts_object_entries():
    class _MetaObj:
        def __init__(self, logprob_sum):
            self.logprob_sum = logprob_sum

    template = _LogpTensorStub()
    meta = [_MetaObj(-3.0), _MetaObj(-4.0)]
    out = pipeline._behavior_logp_tensor_from_meta(meta, 2, template)
    assert isinstance(out, _BehaviorTensorStub)
    assert out.arr == [-3.0, -4.0]


def test_behavior_logp_tensor_from_meta_returns_none_on_missing_values():
    template = _LogpTensorStub()
    meta = [{"token_count": 2}, {"logprob_sum": -1.0}]
    assert pipeline._behavior_logp_tensor_from_meta(meta, 2, template) is None
