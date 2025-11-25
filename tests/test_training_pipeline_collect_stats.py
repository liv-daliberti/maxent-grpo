"""Targeted coverage for _collect_batch_stats edge paths."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import maxent_grpo.training.pipeline as pipeline


def test_collect_batch_stats_rebuilds_ref_stats_on_mismatch(monkeypatch):
    ctx = SimpleNamespace(
        runtime=SimpleNamespace(device="cpu", tokenizer="tok"),
        scoring=SimpleNamespace(
            batching=SimpleNamespace(prompt_length_cache_get=None),
            weighting=SimpleNamespace(),
        ),
        generation=SimpleNamespace(max_completion_len=4),
    )
    gen_batch = SimpleNamespace(grouped_completions=[["a"], ["b"]])
    reward_comp = SimpleNamespace(
        ref_logprob_meta=[{"meta": 1}],
        pairs=SimpleNamespace(completions=[1, 2]),
    )

    ref_calls = []
    monkeypatch.setattr(
        pipeline,
        "_reference_stats_from_meta",
        lambda meta, total, device: ref_calls.append((meta, total, device))
        or SimpleNamespace(meta=meta, total=total, device=device),
    )
    monkeypatch.setattr(
        pipeline,
        "build_score_batch",
        lambda reward_comp, tokenizer, gen_cfg, batching_cfg: SimpleNamespace(
            total_sequences=2, prompt_entries=[], max_prompt_len=1
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "compute_weight_stats",
        lambda grouped_completions, reward_comp, ref_stats, weighting_cfg: SimpleNamespace(
            flat_weights=[0.5]
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "summarize_completion_lengths",
        lambda ref_stats, max_completion_len: ("scores", "lengths", 3),
    )
    monkeypatch.setattr(
        pipeline,
        "gather_reference_logprobs",
        lambda *a, **k: pytest.fail("gather should not be called"),
    )

    result = pipeline._collect_batch_stats(ctx, gen_batch, reward_comp)

    assert result is not None
    assert len(ref_calls) >= 2
    assert ref_calls[0][1] == 2  # initial call uses score_batch count
    assert ref_calls[-1][1] == 2  # mismatch path rebuilds using same total
