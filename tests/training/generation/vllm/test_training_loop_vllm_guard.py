"""Unit tests for vLLM logprob guard behavior in the training loop."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest

from maxent_grpo.training.loop import _maybe_guard_vllm_logprobs
from maxent_grpo.training.types import TrainingLoopContext


def _build_ctx(fail_after: int, *, fallback: bool = False) -> TrainingLoopContext:
    runtime = SimpleNamespace(_vllm_logprob_miss_steps=0)
    generation = SimpleNamespace(use_vllm=True, vllm_request_logprobs=True)
    scoring = SimpleNamespace(reference_logprobs_source="auto")
    training_args = SimpleNamespace(
        vllm_logprob_fail_after=fail_after,
        vllm_logprob_fallback=fallback,
    )
    settings = SimpleNamespace(scoring=SimpleNamespace(reference_logprobs_source="auto"))
    return cast(
        TrainingLoopContext,
        SimpleNamespace(
            runtime=runtime,
            generation=generation,
            scoring=scoring,
            training_args=training_args,
            settings=settings,
        ),
    )


def _build_prepared():
    score_batch = SimpleNamespace(total_sequences=2)
    batch_stats = SimpleNamespace(score_batch=score_batch)
    reward_comp = SimpleNamespace(ref_logprob_meta=[])
    return SimpleNamespace(reward_comp=reward_comp, batch_stats=batch_stats)


def test_vllm_logprob_guard_trips_on_missing():
    ctx = _build_ctx(fail_after=2, fallback=False)
    prepared = _build_prepared()
    _maybe_guard_vllm_logprobs(ctx, prepared, 1)
    with pytest.raises(RuntimeError):
        _maybe_guard_vllm_logprobs(ctx, prepared, 2)
