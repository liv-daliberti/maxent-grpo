"""Unit coverage for vLLM request helpers."""

from __future__ import annotations

import builtins
import sys
import logging
from types import SimpleNamespace, ModuleType

import pytest

from maxent_grpo.generation.errors import GenerationServiceError, ServiceErrorPayload
from maxent_grpo.generation.vllm_requests import (
    VLLMRequestMixin,
    _DEFAULT_PROMPT_CHAR_LIMIT,
)
from maxent_grpo.generation.vllm_state import _VLLMGenerationState
from tests.helpers.vllm import make_vllm_context


class _Dummy(VLLMRequestMixin):
    def __init__(self, ctx):
        self.ctx = ctx
        self._safe_generate = None
        self._time = None
        self._fallback_generate = None


def _ctx(**overrides):
    return make_vllm_context(**overrides)


def test_prompt_char_limit_falls_back_on_import_error(monkeypatch):
    # Ensure import of maxent_grpo.generation.vllm fails to hit the ImportError branch.
    monkeypatch.delitem(sys.modules, "maxent_grpo.generation.vllm", raising=False)
    ctx = _ctx(prompt_char_limit=None, max_prompt_len=0)
    dummy = _Dummy(ctx)
    assert dummy._prompt_char_limit() == _DEFAULT_PROMPT_CHAR_LIMIT


def test_prompt_char_limit_falls_back_on_missing_attr(monkeypatch):
    # Provide a vllm module without PROMPT_CHAR_LIMIT to trigger AttributeError path.
    vllm_mod = ModuleType("maxent_grpo.generation.vllm")
    monkeypatch.setitem(sys.modules, "maxent_grpo.generation.vllm", vllm_mod)
    ctx = _ctx(prompt_char_limit=None, max_prompt_len=1)
    dummy = _Dummy(ctx)
    # approx_chars will be 4; fallback should choose max of default and approx.
    assert dummy._prompt_char_limit() == max(_DEFAULT_PROMPT_CHAR_LIMIT, 4)


def test_prompt_char_limit_prefers_imported_constant(monkeypatch):
    """When PROMPT_CHAR_LIMIT is importable, it should be returned."""

    monkeypatch.delitem(sys.modules, "maxent_grpo.generation.vllm", raising=False)
    vllm_mod = ModuleType("maxent_grpo.generation.vllm")
    vllm_mod.PROMPT_CHAR_LIMIT = 17
    monkeypatch.setitem(sys.modules, "maxent_grpo.generation.vllm", vllm_mod)
    pkg = sys.modules.get("maxent_grpo.generation")
    if pkg is not None:
        monkeypatch.setattr(pkg, "vllm", vllm_mod, raising=False)
    ctx = _ctx(prompt_char_limit=None, max_prompt_len=0)
    dummy = _Dummy(ctx)
    assert dummy._prompt_char_limit() == 17


def test_prompt_char_limit_handles_attribute_error(monkeypatch):
    """AttributeError during import should trigger the fallback constant."""

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "maxent_grpo.generation.vllm":
            raise AttributeError("missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    monkeypatch.delitem(sys.modules, "maxent_grpo.generation.vllm", raising=False)
    ctx = _ctx(prompt_char_limit=None, max_prompt_len=0)
    dummy = _Dummy(ctx)
    assert dummy._prompt_char_limit() == _DEFAULT_PROMPT_CHAR_LIMIT


def test_prompt_char_limit_uses_approx_when_default_disabled(monkeypatch):
    """If default limit is disabled and import fails, fall back to approx_chars."""

    real_import = __import__
    import_calls = []

    def fake_import(name, *args, **kwargs):
        if name.startswith("maxent_grpo.generation"):
            import_calls.append(name)
            raise ImportError("no vllm")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    monkeypatch.setattr(
        "maxent_grpo.generation.vllm_requests._DEFAULT_PROMPT_CHAR_LIMIT", -1
    )
    monkeypatch.delitem(sys.modules, "maxent_grpo.generation.vllm", raising=False)
    pkg = sys.modules.get("maxent_grpo.generation")
    if pkg is not None and hasattr(pkg, "vllm"):
        monkeypatch.delattr(pkg, "vllm", raising=False)
    ctx = _ctx(prompt_char_limit=None, max_prompt_len=2)
    dummy = _Dummy(ctx)
    result = dummy._prompt_char_limit()
    assert import_calls  # verify import path was exercised
    assert result == 8


def test_prepare_vllm_targets_with_dedup(monkeypatch):
    monkeypatch.setenv("MAXENT_VLLM_DEDUP", "1")
    ctx = _ctx()
    dummy = _Dummy(ctx)
    prompts, counts, mapping = dummy._prepare_vllm_targets(
        ["a", "b", "a"], num_samples=2, per_prompt_counts=[1, 2, 3]
    )
    assert prompts == ["a", "b"]
    assert counts == [1, 2]
    assert mapping == [0, 1, 0]


def test_prompt_char_limit_prefers_override():
    ctx = _ctx(prompt_char_limit=42, max_prompt_len=0)
    dummy = _Dummy(ctx)
    assert dummy._prompt_char_limit() == 42


def test_merge_vllm_results_records_overflow():
    ctx = _ctx(generation_stats={})
    dummy = _Dummy(ctx)
    state = _VLLMGenerationState(
        prompts=["p"],
        target_counts=[1],
        requested_n=1,
        round_limit=1,
        track_logprobs=False,
    )
    dummy._merge_vllm_results(
        state, grouped=[["x", "y"]], grouped_meta=None, pending_indices=[0]
    )
    assert ctx.generation_stats["vllm_excess_prompts"] == 1
    assert ctx.generation_stats["vllm_excess_completions"] == 1


def test_backfill_missing_skips_when_no_need(monkeypatch):
    ctx = _ctx(
        vllm_backfill_local=True,
        generation_stats={"vllm_backfilled_prompts": 0},
    )
    dummy = _Dummy(ctx)
    dummy._fallback_generate = lambda prompts, n, counts: ([["fill"]], None)
    state = _VLLMGenerationState(
        prompts=["p"],
        target_counts=[1],
        requested_n=1,
        round_limit=1,
        track_logprobs=False,
    )
    state.aggregated[0].append("existing")
    dummy._backfill_missing(state, [0])
    assert state.aggregated[0] == ["existing"]
    assert ctx.generation_stats["vllm_backfilled_prompts"] == 1


def test_record_vllm_failure_updates_stats(monkeypatch):
    ctx = _ctx(
        vllm_backfill_local=True,
        generation_stats={"vllm_failed_prompts": 0},
    )
    dummy = _Dummy(ctx)
    state = _VLLMGenerationState(
        prompts=["p1", "p2"],
        target_counts=[2, 1],
        requested_n=1,
        round_limit=2,
        track_logprobs=False,
    )
    state.aggregated[0].append("done")
    dummy._record_vllm_failure(state, [0, 1])
    assert ctx.generation_stats["vllm_failed_prompts"] == 2


def test_retry_attempt_logging_uses_wandb(monkeypatch):
    ctx = _ctx(generation_stats={"current_step": 7, "dataset_name": "ds"})
    dummy = _Dummy(ctx)
    records = {}

    class _Run:
        def log(self, metrics, step=None):
            records["metrics"] = metrics
            records["step"] = step

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(run=_Run()))
    dummy._record_retry_attempt_metric(502, attempt=2, pending_count=3, reason="http")
    assert records["metrics"]["generation/retry_attempts"] == 1
    assert records["metrics"]["generation/retry_status_code"] == 502
    assert records["metrics"]["generation/retry_dataset"] == "ds"
    assert records["step"] == 7


def test_retry_exhausted_logging_uses_wandb(monkeypatch):
    ctx = _ctx(generation_stats={"current_step": 0})
    dummy = _Dummy(ctx)
    payload = ServiceErrorPayload(
        service="vllm",
        endpoint="http://host",
        model="model",
        prompt_count=2,
        payload_chars=10,
        payload_size_bytes=20,
        status_code=503,
        attempt=3,
        max_attempts=3,
        exception_type="RuntimeError",
        exception_message="boom",
        request_id="req",
        extra={"prompt_hash": "abc", "dataset": "ds"},
    )
    records = {}

    class _Run:
        def log(self, metrics, step=None):
            records["metrics"] = metrics
            records["step"] = step

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(run=_Run()))
    dummy._log_retry_exhausted_metric(payload)
    assert records["metrics"]["generation/retry_exhausted"] == 1
    assert records["metrics"]["generation/retry_prompt_hash"] == "abc"
    assert records["metrics"]["generation/retry_dataset"] == "ds"


def test_retry_metrics_handle_wandb_failure(monkeypatch, caplog):
    ctx = SimpleNamespace(generation_stats={"current_step": 1})
    dummy = _Dummy(ctx)

    class _Run:
        def log(self, *_args, **_kwargs):
            raise RuntimeError("wandb-down")

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(run=_Run()))
    with caplog.at_level(logging.WARNING):
        dummy._record_retry_attempt_metric(500, attempt=1, pending_count=2)
    assert "Failed to log retry metrics to W&B" in caplog.text


def test_run_vllm_rounds_emits_metrics_on_exhaustion(monkeypatch, caplog):
    ctx = SimpleNamespace(
        vllm_retry_sleep=0.0,
        generation_stats={
            "dataset_name": "demo",
            "vllm_retry_rounds": 0,
            "vllm_backfilled_prompts": 0,
            "vllm_failed_prompts": 0,
        },
        vllm_backfill_local=False,
        vllm_url="http://host",
        vllm_request_id_prefix=None,
        accelerator=None,
        model=SimpleNamespace(name_or_path="model"),
    )
    dummy = _Dummy(ctx)
    attempt_log = []
    exhausted_calls = {"count": 0}

    def _record_attempt(status, attempt, pending, reason=None):
        attempt_log.append((status, attempt, pending, reason))

    def _log_exhausted(payload):
        exhausted_calls["count"] += 1

    dummy._record_retry_attempt_metric = _record_attempt
    dummy._log_retry_exhausted_metric = _log_exhausted
    dummy._log_structured_vllm_failure = lambda payload: None

    def _execute_fail(*_args, **_kwargs):
        raise RuntimeError("HTTP 503: bad")

    dummy._execute_vllm_request = _execute_fail
    state = _VLLMGenerationState(
        prompts=["p"],
        target_counts=[1],
        requested_n=1,
        round_limit=2,
        track_logprobs=False,
    )
    with caplog.at_level(logging.WARNING):
        with pytest.raises(GenerationServiceError):
            dummy._run_vllm_rounds(state)
    assert len(attempt_log) == 2
    assert attempt_log[0][0] == 503
    assert exhausted_calls["count"] == 1
    assert "policy=" in caplog.text


def test_build_failure_payload_includes_metadata():
    ctx = SimpleNamespace(
        vllm_backfill_local=False,
        vllm_url="http://host",
        vllm_request_id_prefix=None,
        accelerator=SimpleNamespace(process_index=0, num_processes=1),
        model=SimpleNamespace(name_or_path="org/model"),
        vllm_backoff=1.5,
        vllm_backoff_multiplier=2.5,
        vllm_retry_sleep=0.25,
        vllm_max_retries=4,
        generation_stats={"dataset_name": "hf/train", "model_id": "org/model"},
    )
    dummy = _Dummy(ctx)
    state = _VLLMGenerationState(
        prompts=["p"],
        target_counts=[1],
        requested_n=1,
        round_limit=1,
        track_logprobs=False,
    )
    payload = dummy._build_vllm_failure_payload(state, [0], 1, RuntimeError("err"))
    extra = payload.extra
    assert extra["dataset"] == "hf/train"
    assert extra["model_id"] == "org/model"
    assert extra["backoff_initial"] == pytest.approx(1.5)
    assert extra["backoff_multiplier"] == pytest.approx(2.5)
    assert extra["retry_sleep"] == pytest.approx(0.25)
    assert extra["max_retries"] == 4


def test_retry_metrics_include_model_id(monkeypatch):
    ctx = SimpleNamespace(generation_stats={"model_id": "org/model"})
    dummy = _Dummy(ctx)
    payload = ServiceErrorPayload(
        service="vllm",
        endpoint="http://host",
        model="org/model",
        prompt_count=1,
        payload_chars=10,
        payload_size_bytes=10,
        status_code=500,
        attempt=2,
        max_attempts=2,
        exception_type="RuntimeError",
        exception_message="boom",
        request_id="req",
        extra={"model_id": "org/model"},
    )
    records = {}

    class _Run:
        def log(self, metrics, **_kwargs):
            records["metrics"] = metrics

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(run=_Run()))
    dummy._log_retry_exhausted_metric(payload)
    assert records["metrics"]["generation/retry_model_id"] == "org/model"
