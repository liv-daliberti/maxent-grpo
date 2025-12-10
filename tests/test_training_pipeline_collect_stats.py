"""Targeted coverage for _collect_batch_stats edge paths."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import maxent_grpo.training.pipeline as pipeline
from maxent_grpo.training import metrics as metrics_mod


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


def test_collect_batch_stats_gathers_reference_when_meta_missing(monkeypatch):
    ctx = SimpleNamespace(
        runtime=SimpleNamespace(device="cpu", tokenizer="tok"),
        scoring=SimpleNamespace(
            batching=SimpleNamespace(prompt_length_cache_get=None),
            weighting=SimpleNamespace(
                tau=0.3,
                beta=0.04,
                denom=1.0,
                q_temperature=1.0,
                q_epsilon=1e-6,
                tau_lr=0.01,
                tau_min=0.05,
                tau_max=1.0,
                tau_warmup_steps=0,
                tau_target_entropy=0.6,
                kl_target=0.1,
                kl_horizon=10,
                kl_ctl_step_size=1.0,
                len_norm_ref=False,
                train_grpo_objective=False,
            ),
        ),
        generation=SimpleNamespace(max_completion_len=4),
    )
    gen_batch = SimpleNamespace(grouped_completions=[["a", "b"], ["c", "d"]])
    reward_comp = SimpleNamespace(
        ref_logprob_meta=[],
        pairs=SimpleNamespace(
            completions=["c1", "c2", "c3", "c4"],
            prompts=["p1", "p2"],
        ),
        advantage=SimpleNamespace(grouped=[[0.1], [0.2]]),
        q_grouped=[[0.5], [0.4]],
        total_utils=[0.1, 0.2],
        advantage_samples=[0.05, -0.02],
        per_reward_values={"accuracy": [0.0, 1.0]},
    )
    score_batch = SimpleNamespace(
        total_sequences=4,
        prompt_entries=[SimpleNamespace(length=2), SimpleNamespace(length=2)],
        max_prompt_len=4,
    )
    fake_ref = SimpleNamespace(ref_logp_mean=-0.3, avg_completion_tokens=3.0)
    gather_calls = {"count": 0}
    weight_stats = SimpleNamespace(
        flat_weights=[0.4, 0.6, 0.5, 0.5],
        weights_grouped=[[0.4, 0.6], [0.5, 0.5]],
        weight_entropy=0.42,
        weight_entropy_min=0.4,
        weight_entropy_max=0.44,
        advantage_entropy=[0.1, 0.3],
    )
    length_stats = metrics_mod.LengthStats(
        min_length=2.0,
        mean_length=3.0,
        max_length=4.0,
        clipped_ratio=0.25,
        min_terminated=2.0,
        mean_terminated=2.5,
        max_terminated=3.0,
    )

    monkeypatch.setattr(
        pipeline,
        "build_score_batch",
        lambda *_args, **_kwargs: score_batch,
    )
    monkeypatch.setattr(
        pipeline,
        "compute_weight_stats",
        lambda *_args, **_kwargs: weight_stats,
    )
    monkeypatch.setattr(
        pipeline,
        "summarize_completion_lengths",
        lambda *_a, **_k: (None, length_stats, 4.0),
    )

    def _fake_gather(*_a, **_k):
        gather_calls["count"] += 1
        return fake_ref

    monkeypatch.setattr(pipeline, "gather_reference_logprobs", _fake_gather)

    batch_stats = pipeline._collect_batch_stats(ctx, gen_batch, reward_comp)
    assert batch_stats is not None
    assert gather_calls["count"] == 1
    assert batch_stats.ref_stats is fake_ref

    weight_entropy_mean = sum(weight_stats.advantage_entropy) / len(
        weight_stats.advantage_entropy
    )
    weight_view = metrics_mod.WeightLoggingView(
        entropy=weight_stats.weight_entropy,
        entropy_min=weight_stats.weight_entropy_min,
        entropy_max=weight_stats.weight_entropy_max,
        advantage_entropy_mean=weight_entropy_mean,
        advantage_entropy_std=0.1,
    )
    payload = metrics_mod.TrainingMetricsPayload(
        reward_stats=metrics_mod.RewardLoggingView(
            reward_mean=0.5,
            reward_std=0.25,
            frac_zero_std=0.0,
            advantage_mean=0.0,
            advantage_std=0.0,
            advantage_count=2,
            per_reward={"accuracy": SimpleNamespace(mean=0.5, std=0.5)},
            q_entropy_mean=0.0,
            q_entropy_std=0.0,
            q_entropy_min=0.0,
            q_entropy_max=0.0,
        ),
        weight_stats=weight_view,
        loss_outputs=SimpleNamespace(
            total_loss_scalar=0.8,
            kl_loss_scalar=0.05,
            policy_loss_scalar=0.75,
            weighted_kl_loss_scalar=0.05,
            clip_loss_scalar=None,
            scalars=SimpleNamespace(kl_loss=0.05),
        ),
        diagnostics=metrics_mod.BatchDiagnostics(
            kl_value=0.05,
            clip_ratio=0.0,
            clip_ratio_low_mean=0.0,
            clip_ratio_low_min=0.0,
            clip_ratio_high_mean=0.0,
            clip_ratio_high_max=0.0,
            clip_ratio_region_mean=0.0,
            kl_per_token_by_len_bucket={},
            kl_token_count_by_len_bucket={},
        ),
        length_stats=batch_stats.length_stats,
        config=metrics_mod.LoggingConfigView(
            weighting=ctx.scoring.weighting,
            clipping=SimpleNamespace(),
            schedule=SimpleNamespace(),
        ),
        scalars=metrics_mod.TrainingScalarStats(
            ref_logp_mean=batch_stats.ref_stats.ref_logp_mean,
            tokens=metrics_mod.TokenUsageStats(
                avg_completion_tokens=batch_stats.ref_stats.avg_completion_tokens,
                num_completion_tokens=batch_stats.num_completion_tokens,
                num_input_tokens=10.0,
            ),
            current_lr=1e-4,
            grad_norm_scalar=None,
            epoch_progress=0.1,
            vllm_latency_ms=None,
        ),
    )
    metrics = metrics_mod.build_training_metrics_dict(payload, global_step=3)
    assert metrics["train/weight_entropy"] == pytest.approx(
        weight_stats.weight_entropy
    )
    assert metrics["train/completions/mean_length_sampled"] == pytest.approx(
        length_stats.mean_length
    )
