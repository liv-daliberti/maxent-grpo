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

Unit tests for training.metrics helper functions.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict

import pytest

from maxent_grpo.training import metrics as metrics_mod


class _DummyAccel:
    def __init__(self):
        self.num_processes = 1

    def gather_object(self, payload):
        return [payload]


class _StubMetricWriter:
    def __init__(self):
        self.logged = []
        self.flushed = False

    def log(self, metrics, step):
        self.logged.append((metrics, step))

    def flush(self):
        self.flushed = True


def _weighting_stub() -> SimpleNamespace:
    return SimpleNamespace(
        tau=0.2,
        beta=0.1,
        denom=1.0,
        q_temperature=1.0,
        q_epsilon=1e-6,
        tau_lr=0.0,
        tau_min=0.0,
        tau_max=1.0,
        tau_warmup_steps=0,
        tau_target_entropy=None,
        kl_target=0.0,
        kl_horizon=0,
        kl_ctl_step_size=0.0,
        len_norm_ref=False,
        train_grpo_objective=True,
        _tau_entropy_ema=0.3,
    )


def _payload() -> metrics_mod.TrainingMetricsPayload:
    scalars = metrics_mod.TrainingScalarStats(
        ref_logp_mean=0.0,
        tokens=metrics_mod.TokenUsageStats(
            avg_completion_tokens=2.0,
            num_completion_tokens=4.0,
            num_input_tokens=6.0,
        ),
        current_lr=0.1,
        grad_norm_scalar=None,
        epoch_progress=0.5,
        vllm_latency_ms=None,
    )
    diagnostics = metrics_mod.BatchDiagnostics(
        kl_value=None,
        clip_ratio=0.0,
        clip_ratio_low_mean=0.0,
        clip_ratio_low_min=0.0,
        clip_ratio_high_mean=0.0,
        clip_ratio_high_max=0.0,
        clip_ratio_region_mean=0.0,
        kl_per_token_by_len_bucket={},
        kl_token_count_by_len_bucket={},
    )
    reward_stats = metrics_mod.RewardLoggingView(
        reward_mean=1.0,
        reward_std=0.5,
        frac_zero_std=0.0,
        advantage_mean=0.0,
        advantage_std=0.0,
        advantage_count=1,
        per_reward={},
        q_entropy_mean=0.0,
        q_entropy_std=0.0,
        q_entropy_min=0.0,
        q_entropy_max=0.0,
    )
    weight_stats = metrics_mod.WeightLoggingView(
        entropy=0.1,
        entropy_min=0.0,
        entropy_max=0.2,
        advantage_entropy_mean=0.0,
        advantage_entropy_std=0.0,
    )
    return metrics_mod.TrainingMetricsPayload(
        config=metrics_mod.LoggingConfigView(
            weighting=_weighting_stub(),
            clipping=SimpleNamespace(),
            schedule=SimpleNamespace(),
        ),
        scalars=scalars,
        diagnostics=diagnostics,
        reward_stats=reward_stats,
        weight_stats=weight_stats,
        loss_outputs=SimpleNamespace(
            total_loss_scalar=1.23,
            kl_loss_scalar=None,
            policy_loss_scalar=1.23,
            weighted_kl_loss_scalar=0.0,
            clip_loss_scalar=None,
            scalars=SimpleNamespace(kl_loss=0.0),
        ),
        length_stats=metrics_mod.LengthStats(
            min_length=1.0,
            mean_length=2.0,
            max_length=3.0,
            clipped_ratio=0.0,
            min_terminated=0.5,
            mean_terminated=1.0,
            max_terminated=1.5,
        ),
    )


def test_build_training_metrics_dict_includes_weighting():
    payload = _payload()
    result = metrics_mod.build_training_metrics_dict(payload, global_step=3)
    assert result["train/tau"] == pytest.approx(0.2)
    assert result["train/weighting/tau"] == pytest.approx(0.2)
    assert result["train/weighting/beta"] == pytest.approx(0.1)
    assert result["train/weight_entropy_ema"] == pytest.approx(0.3)
    assert "train/kl_per_completion_token" in result


def test_build_training_metrics_dict_covers_lengths_and_rewards():
    payload = _payload()
    # Introduce an out-of-range clipped ratio to ensure clamping occurs.
    payload.length_stats = payload.length_stats.__class__(**{
        "min_length": 1.0,
        "mean_length": 2.0,
        "max_length": 3.0,
        "clipped_ratio": -5.0,
        "min_terminated": 0.5,
        "mean_terminated": 1.0,
        "max_terminated": 1.5,
    })
    result = metrics_mod.build_training_metrics_dict(payload, global_step=10)
    # Completion metrics present and clipped_ratio clamped to [0, 1].
    assert result["train/completions/mean_length_sampled"] == pytest.approx(2.0)
    assert result["train/completions/clipped_frac"] == pytest.approx(0.0)
    # Reward metrics present.
    assert result["train/reward"] == pytest.approx(payload.reward_stats.reward_mean)
    assert result["train/reward_std"] == pytest.approx(payload.reward_stats.reward_std)
    assert result["train/frac_reward_zero_std"] == pytest.approx(
        payload.reward_stats.frac_zero_std
    )
    # Loss breakdown is emitted.
    assert result["train/loss/policy"] == pytest.approx(
        payload.loss_outputs.policy_loss_scalar
    )


def test_build_training_metrics_flags_grpo_vs_maxent():
    payload = _payload()
    payload.config.weighting.train_grpo_objective = True
    grpo_metrics = metrics_mod.build_training_metrics_dict(payload, global_step=1)
    assert grpo_metrics["train/grpo_objective"] == 1.0
    assert grpo_metrics["train/maxent_objective"] == 0.0
    payload.config.weighting.train_grpo_objective = False
    payload.config.weighting.tau = 0.45
    payload.config.weighting.beta = 0.33
    maxent_metrics = metrics_mod.build_training_metrics_dict(payload, global_step=1)
    assert maxent_metrics["train/grpo_objective"] == 0.0
    assert maxent_metrics["train/maxent_objective"] == 1.0
    # Len-norm flag should propagate.
    assert maxent_metrics["train/len_norm_ref"] == (
        1.0 if payload.config.weighting.len_norm_ref else 0.0
    )
    assert maxent_metrics["train/tau"] == pytest.approx(0.45)
    assert maxent_metrics["train/weighting/tau"] == pytest.approx(0.45)
    assert maxent_metrics["train/beta"] == pytest.approx(0.33)


def test_build_training_metrics_emits_kl_from_loss_when_missing_in_diag():
    payload = _payload()
    payload.diagnostics = payload.diagnostics.__class__(
        **{
            "kl_value": None,
            "clip_ratio": payload.diagnostics.clip_ratio,
            "clip_ratio_low_mean": payload.diagnostics.clip_ratio_low_mean,
            "clip_ratio_low_min": payload.diagnostics.clip_ratio_low_min,
            "clip_ratio_high_mean": payload.diagnostics.clip_ratio_high_mean,
            "clip_ratio_high_max": payload.diagnostics.clip_ratio_high_max,
            "clip_ratio_region_mean": payload.diagnostics.clip_ratio_region_mean,
            "kl_per_token_by_len_bucket": {},
            "kl_token_count_by_len_bucket": {},
        }
    )
    payload.loss_outputs.kl_loss_scalar = 0.42
    result = metrics_mod.build_training_metrics_dict(payload, global_step=5)
    assert result["train/kl"] == pytest.approx(0.42)


def test_build_training_metrics_emits_weight_entropy_key():
    payload = _payload()
    payload.weight_stats = metrics_mod.WeightLoggingView(
        entropy=0.33,
        entropy_min=0.1,
        entropy_max=0.5,
        advantage_entropy_mean=0.0,
        advantage_entropy_std=0.0,
    )
    result = metrics_mod.build_training_metrics_dict(payload, global_step=2)
    assert "train/weight_entropy" in result
    assert result["train/weight_entropy"] == pytest.approx(0.33)


def test_build_training_metrics_emits_weight_aliases_for_maxent():
    payload = _payload()
    payload.config.weighting.train_grpo_objective = False
    payload.config.weighting.tau = 0.62
    payload.config.weighting.beta = 0.17
    payload.loss_outputs.total_loss_scalar = 0.5
    metrics = metrics_mod.build_training_metrics_dict(payload, global_step=2)
    assert metrics["train/maxent_objective"] == 1.0
    assert metrics["train/tau"] == pytest.approx(0.62)
    assert metrics["train/weighting/tau"] == pytest.approx(0.62)
    assert metrics["train/beta"] == pytest.approx(0.17)
    assert metrics["train/weighting/beta"] == pytest.approx(0.17)


def test_build_training_metrics_falls_back_to_loss_kl_scalar():
    payload = _payload()
    payload.diagnostics = payload.diagnostics.__class__(**{
        **payload.diagnostics.__dict__,
        "kl_value": None,
    })
    payload.loss_outputs.kl_loss_scalar = 0.37
    metrics = metrics_mod.build_training_metrics_dict(payload, global_step=4)
    assert metrics["train/kl"] == pytest.approx(0.37)
    assert maxent_metrics["train/weighting/beta"] == pytest.approx(0.33)


def test_build_training_metrics_emits_controller_signals():
    payload = _payload()
    weighting = payload.config.weighting
    weighting.kl_target = 0.5
    weighting.kl_horizon = 10
    weighting.kl_ctl_step_size = 0.1
    weighting.tau_target_entropy = 0.3
    weighting.tau_warmup_steps = 0
    payload.diagnostics = payload.diagnostics.__class__(
        kl_value=0.8,
        clip_ratio=0.0,
        clip_ratio_low_mean=0.0,
        clip_ratio_low_min=0.0,
        clip_ratio_high_mean=0.0,
        clip_ratio_high_max=0.0,
        clip_ratio_region_mean=0.0,
        kl_per_token_by_len_bucket={},
        kl_token_count_by_len_bucket={},
    )
    # Make weight entropy non-zero to exercise tau error/loss.
    payload.weight_stats = payload.weight_stats.__class__(
        entropy=0.5,
        entropy_min=0.0,
        entropy_max=1.0,
        advantage_entropy_mean=0.0,
        advantage_entropy_std=0.0,
    )
    metrics = metrics_mod.build_training_metrics_dict(payload, global_step=2)
    assert metrics["train/kl_controller_enabled"] == 1.0
    assert metrics["train/kl_error_to_target"] == pytest.approx(0.8 - 0.5)
    assert metrics["train/kl_ratio_to_target"] == pytest.approx(0.8 / 0.5)
    assert metrics["train/tau_schedule_active"] == 1.0
    assert "train/tau_loss" in metrics and metrics["train/tau_loss"] > 0.0


def test_build_training_metrics_emits_clip_diagnostics_and_clamps_lengths():
    payload = _payload()
    payload.diagnostics = payload.diagnostics.__class__(
        kl_value=0.1,
        clip_ratio=0.75,
        clip_ratio_low_mean=0.1,
        clip_ratio_low_min=0.05,
        clip_ratio_high_mean=0.2,
        clip_ratio_high_max=0.25,
        clip_ratio_region_mean=0.3,
        kl_per_token_by_len_bucket={},
        kl_token_count_by_len_bucket={},
    )
    # Force an invalid clipped_ratio in length stats to exercise clamping.
    payload.length_stats = payload.length_stats.__class__(**{
        "min_length": 1.0,
        "mean_length": 2.0,
        "max_length": 3.0,
        "clipped_ratio": 5.0,  # out of range
        "min_terminated": 0.5,
        "mean_terminated": 1.0,
        "max_terminated": 1.5,
    })
    metrics = metrics_mod.build_training_metrics_dict(payload, global_step=4)
    assert metrics["train/clip_ratio"] == pytest.approx(0.75)
    assert metrics["train/clip_ratio/low_mean"] == pytest.approx(0.1)
    assert metrics["train/clip_ratio/low_min"] == pytest.approx(0.05)
    assert metrics["train/clip_ratio/high_mean"] == pytest.approx(0.2)
    assert metrics["train/clip_ratio/high_max"] == pytest.approx(0.25)
    assert metrics["train/clip_ratio/region_mean"] == pytest.approx(0.3)
    # Length clamping should map 5.0 to 1.0.
    assert metrics["train/completions/clipped_frac"] == pytest.approx(1.0)


def test_build_training_metrics_propagates_raw_kl_value():
    payload = _payload()
    payload.diagnostics = payload.diagnostics.__class__(
        kl_value=0.42,
        clip_ratio=0.0,
        clip_ratio_low_mean=0.0,
        clip_ratio_low_min=0.0,
        clip_ratio_high_mean=0.0,
        clip_ratio_high_max=0.0,
        clip_ratio_region_mean=0.0,
        kl_per_token_by_len_bucket={},
        kl_token_count_by_len_bucket={},
    )
    metrics = metrics_mod.build_training_metrics_dict(payload, global_step=11)
    assert metrics["train/kl"] == pytest.approx(0.42)


def test_build_training_metrics_preserves_kl_buckets():
    payload = _payload()
    payload.diagnostics = payload.diagnostics.__class__(
        kl_value=0.2,
        clip_ratio=0.0,
        clip_ratio_low_mean=0.0,
        clip_ratio_low_min=0.0,
        clip_ratio_high_mean=0.0,
        clip_ratio_high_max=0.0,
        clip_ratio_region_mean=0.0,
        kl_per_token_by_len_bucket={"0-32": 0.1, "33-64": 0.05},
        kl_token_count_by_len_bucket={"0-32": 100.0, "33-64": 50.0},
    )
    metrics = metrics_mod.build_training_metrics_dict(payload, global_step=7)
    assert metrics["train/kl_per_token_bucket/0-32"] == pytest.approx(0.1)
    assert metrics["train/kl_per_token_bucket_tokens/0-32"] == pytest.approx(100.0)
    assert metrics["train/kl_per_token_bucket/33-64"] == pytest.approx(0.05)
    assert metrics["train/kl_per_token_bucket_tokens/33-64"] == pytest.approx(50.0)


def test_build_training_metrics_logs_seed_and_entropy_losses():
    payload = _payload()
    # Inject seed/info entropy scalars into loss outputs and ensure they surface.
    payload.loss_outputs = payload.loss_outputs.__class__(
        total_loss_scalar=1.0,
        policy_loss_scalar=0.5,
        kl_loss_scalar=0.1,
        weighted_kl_loss_scalar=0.1,
        clip_loss_scalar=None,
        seed_loss_value=0.25,
        info_seed_entropy_scalar=0.125,
    )
    metrics = metrics_mod.build_training_metrics_dict(payload, global_step=9)
    assert metrics["train/loss/seed"] == pytest.approx(0.25)
    assert metrics["train/info_seed/entropy"] == pytest.approx(0.125)


def test_build_training_metrics_logs_controller_deltas():
    payload = _payload()
    weighting = payload.config.weighting
    weighting._prev_tau = 0.1
    weighting._prev_beta = 0.05
    weighting.tau = 0.3
    weighting.beta = 0.2
    metrics = metrics_mod.build_training_metrics_dict(payload, global_step=3)
    assert metrics["train/delta_tau"] == pytest.approx(0.2)
    assert metrics["train/delta_tau_abs"] == pytest.approx(0.2)
    assert metrics["train/delta_beta"] == pytest.approx(0.15)
    assert metrics["train/delta_beta_abs"] == pytest.approx(0.15)


def test_build_training_metrics_logs_clip_and_kl_per_token_zero():
    payload = _payload()
    # Set clip loss and zero KL to exercise per-token path.
    payload.loss_outputs = payload.loss_outputs.__class__(
        total_loss_scalar=1.0,
        policy_loss_scalar=0.5,
        kl_loss_scalar=0.0,
        weighted_kl_loss_scalar=0.0,
        clip_loss_scalar=0.75,
        seed_loss_value=None,
        info_seed_entropy_scalar=None,
    )
    metrics = metrics_mod.build_training_metrics_dict(payload, global_step=4)
    assert metrics["train/loss/clip"] == pytest.approx(0.75)
    # With nonzero completion tokens, kl_per_completion_token should be present and zero.
    assert metrics["train/kl_per_completion_token"] == pytest.approx(0.0)


def test_build_training_metrics_logs_vllm_latency_and_grad_norm():
    payload = _payload()
    # Inject vLLM latency and grad norm.
    payload.scalars = payload.scalars.__class__(
        ref_logp_mean=payload.scalars.ref_logp_mean,
        tokens=payload.scalars.tokens,
        current_lr=payload.scalars.current_lr,
        grad_norm_scalar=0.9,
        epoch_progress=payload.scalars.epoch_progress,
        vllm_latency_ms=123.4,
    )
    metrics = metrics_mod.build_training_metrics_dict(payload, global_step=5)
    assert metrics["train/grad_norm"] == pytest.approx(0.9)
    assert metrics["train/vllm_latency_ms"] == pytest.approx(123.4)


def test_log_training_metrics_calls_writer(monkeypatch):
    payload = _payload()
    logged: Dict[str, Any] = {}

    class _Writer:
        def log(self, metrics, step):
            logged["metrics"] = metrics
            logged["step"] = step

    handles = metrics_mod.LoggingHandles(
        metric_writer=_Writer(),
        save_checkpoint=lambda name: logged.setdefault("ckpt", name),
        save_strategy="steps",
        save_steps=2,
        wandb_run=None,
    )

    result = metrics_mod.log_training_metrics(handles, global_step=5, payload=payload)
    assert logged["metrics"] == result
    assert logged["step"] == 5


def test_log_local_step_accumulates(monkeypatch):
    state = SimpleNamespace(
        global_step=1, num_input_tokens_seen=10.0, metric_sums={}, metric_counts={}
    )
    payload = _payload()
    payload.scalars.tokens.num_input_tokens = 10.0
    reward_comp = SimpleNamespace(
        total_utils=[0.0],
        per_reward_values={},
        advantage_samples=[0.0],
        advantage=SimpleNamespace(grouped=[[0.0]]),
        q_grouped=[[1.0]],
    )
    weight_stats = metrics_mod.WeightStats(
        weights_grouped=[[1.0]],
        flat_weights=[1.0],
        weight_entropy=0.1,
        weight_entropy_min=0.1,
        weight_entropy_max=0.1,
        advantage_entropy=[0.0],
    )
    writer = _StubMetricWriter()
    logging_handles = metrics_mod.LoggingHandles(
        metric_writer=writer,
        save_checkpoint=lambda *_a, **_k: None,
        save_strategy="steps",
        save_steps=1,
        wandb_run=None,
    )
    metrics_mod.log_local_step(
        SimpleNamespace(
            runtime=SimpleNamespace(accelerator=SimpleNamespace(is_main_process=True)),
            logging=logging_handles,
            scoring=SimpleNamespace(
                weighting=_weighting_stub(), clipping=SimpleNamespace()
            ),
            generation=SimpleNamespace(use_vllm=False, generation_stats={}),
            optimization=SimpleNamespace(
                schedule=SimpleNamespace(total_training_steps=1)
            ),
        ),
        state,
        SimpleNamespace(
            reward_comp=reward_comp,
            weight_stats=weight_stats,
            ref_stats=SimpleNamespace(ref_logp_mean=0.0, avg_completion_tokens=1.0),
            length_stats=payload.length_stats,
            num_completion_tokens=payload.scalars.tokens.num_completion_tokens,
            seed_metrics=None,
        ),
        metrics_mod.LogStepArtifacts(
            loss_outputs=payload.loss_outputs,
            diagnostics=payload.diagnostics,
            grad_norm_scalar=None,
            epoch_progress=0.5,
        ),
        current_lr=payload.scalars.current_lr,
    )
    assert state.num_input_tokens_seen == 10.0
    assert "train/loss" in state.metric_sums


def test_log_training_step_aggregates(monkeypatch):
    accel = SimpleNamespace(is_main_process=True, num_processes=1)
    payload = _payload()
    reward_comp = SimpleNamespace(
        total_utils=[0.0],
        per_reward_values={},
        advantage_samples=[0.0],
        advantage=SimpleNamespace(grouped=[[0.0]]),
        q_grouped=[[1.0]],
    )
    weight_stats = metrics_mod.WeightStats(
        weights_grouped=[[1.0]],
        flat_weights=[1.0],
        weight_entropy=0.1,
        weight_entropy_min=0.1,
        weight_entropy_max=0.1,
        advantage_entropy=[0.0],
    )
    writer = _StubMetricWriter()
    logging_handles = metrics_mod.LoggingHandles(
        metric_writer=writer,
        save_checkpoint=lambda *_a, **_k: None,
        save_strategy="steps",
        save_steps=0,
        wandb_run=None,
    )
    state = SimpleNamespace(
        global_step=1, num_input_tokens_seen=10.0, metric_sums={}, metric_counts={}
    )
    log_step_artifacts = metrics_mod.LogStepArtifacts(
        loss_outputs=payload.loss_outputs,
        diagnostics=payload.diagnostics,
        grad_norm_scalar=None,
        epoch_progress=0.5,
    )
    ctx_full = SimpleNamespace(
        runtime=SimpleNamespace(accelerator=accel),
        logging=logging_handles,
        scoring=SimpleNamespace(
            weighting=_weighting_stub(), clipping=SimpleNamespace()
        ),
        generation=SimpleNamespace(use_vllm=False, generation_stats={}),
        optimization=SimpleNamespace(schedule=SimpleNamespace(total_training_steps=1)),
    )
    prepared = SimpleNamespace(
        reward_comp=reward_comp,
        weight_stats=weight_stats,
        ref_stats=SimpleNamespace(ref_logp_mean=0.0, avg_completion_tokens=1.0),
        length_stats=payload.length_stats,
        num_completion_tokens=payload.scalars.tokens.num_completion_tokens,
        seed_metrics=None,
    )
    metrics_mod.log_local_step(
        ctx_full,
        state,
        prepared,
        log_step_artifacts,
        current_lr=payload.scalars.current_lr,
    )
    metrics_mod.log_training_step(
        ctx_full,
        state,
        prepared,
        log_step_artifacts,
        current_lr=payload.scalars.current_lr,
    )
    assert writer.logged, "global metrics should be logged via writer"
    last_metrics, _step = writer.logged[-1]
    assert last_metrics["train/loss"] == pytest.approx(
        payload.loss_outputs.total_loss_scalar
    )
def test_build_training_metrics_falls_back_to_loss_kl_scalar():
    payload = _payload()
    payload.diagnostics = payload.diagnostics.__class__(
        kl_value=None,
        clip_ratio=0.0,
        clip_ratio_low_mean=0.0,
        clip_ratio_low_min=0.0,
        clip_ratio_high_mean=0.0,
        clip_ratio_high_max=0.0,
        clip_ratio_region_mean=0.0,
        kl_per_token_by_len_bucket={},
        kl_token_count_by_len_bucket={},
    )
    payload.loss_outputs = payload.loss_outputs.__class__(
        total_loss_scalar=1.0,
        kl_loss_scalar=0.25,
        policy_loss_scalar=0.5,
        weighted_kl_loss_scalar=0.0,
        clip_loss_scalar=None,
        scalars=SimpleNamespace(kl_loss=0.25),
    )
    payload.config.weighting.kl_target = 0.5
    payload.config.weighting.kl_horizon = 10
    payload.config.weighting.kl_ctl_step_size = 0.1
    metrics = metrics_mod.build_training_metrics_dict(payload, global_step=6)
    assert metrics["train/kl"] == pytest.approx(0.25)
    assert metrics["train/kl_error_to_target"] == pytest.approx(
        0.25 - payload.config.weighting.kl_target
    )
