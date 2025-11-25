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
    )
    reward_stats = metrics_mod.RewardLoggingView(
        reward_mean=1.0,
        reward_std=0.5,
        frac_zero_std=0.0,
        advantage_mean=0.0,
        advantage_std=0.0,
        advantage_count=1,
        per_reward={},
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
            weighting=SimpleNamespace(beta=0.1, tau=0.2, _tau_entropy_ema=0.3),
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
    assert result["train/weight_entropy_ema"] == pytest.approx(0.3)
    assert "train/kl_per_completion_token" in result


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
                weighting=SimpleNamespace(tau=0.2, beta=0.1), clipping=SimpleNamespace()
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
    ctx = SimpleNamespace(
        logging=logging_handles,
        weighting=SimpleNamespace(),
        scoring=SimpleNamespace(),
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
        logging=ctx.logging,
        scoring=SimpleNamespace(
            weighting=SimpleNamespace(beta=0.1, tau=0.2), clipping=SimpleNamespace()
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
