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

Tests for the training loop helpers.
"""

from __future__ import annotations
from importlib import import_module, reload
import logging
import sys
from types import SimpleNamespace

import pytest

from maxent_grpo.training.types import LoggingHandles, LogStepArtifacts
from maxent_grpo.training.runtime.prompts import GenerationPenaltyConfig
from maxent_grpo.training.controller_objective import ControllerGradients


@pytest.fixture
def rtl():
    """Load training.loop module."""
    module = reload(import_module("maxent_grpo.training.loop"))
    return module


def test_collect_batch_stats_prefers_vllm_metadata(monkeypatch, rtl):
    """When vLLM metadata covers all sequences we should skip local rescoring."""
    pipeline = reload(import_module("maxent_grpo.training.pipeline"))

    reward_comp = SimpleNamespace(
        ref_logprob_meta=[
            SimpleNamespace(logprob_sum=0.0, token_count=1),
            SimpleNamespace(logprob_sum=-1.0, token_count=2),
        ],
        pairs=SimpleNamespace(completions=["a", "b"]),
    )
    gen_batch = SimpleNamespace(grouped_completions=[["a", "b"]])
    ctx = SimpleNamespace(
        runtime=SimpleNamespace(tokenizer=None, device="cpu"),
        generation=SimpleNamespace(max_completion_len=32),
        scoring=SimpleNamespace(
            batching=SimpleNamespace(),
            weighting=SimpleNamespace(),
        ),
    )

    class _ScoreBatch:
        total_sequences = 2
        prompt_entries = []
        max_prompt_len = 0

    calls = {"ref_meta": 0, "gather": 0}

    def _fake_meta(meta, total, device):
        calls["ref_meta"] += 1
        assert total == 2
        return SimpleNamespace(dummy=True)

    def _fail_gather(*_args, **_kwargs):
        calls["gather"] += 1
        raise AssertionError("gather_reference_logprobs should not be called")

    monkeypatch.setattr(pipeline, "build_score_batch", lambda *_a, **_k: _ScoreBatch())
    monkeypatch.setattr(
        pipeline,
        "compute_weight_stats",
        lambda *_a, **_k: SimpleNamespace(flat_weights=[1]),
    )
    monkeypatch.setattr(
        pipeline,
        "summarize_completion_lengths",
        lambda *_a, **_k: (None, SimpleNamespace(), 2.0),
    )
    monkeypatch.setattr(pipeline, "reference_from_vllm_meta", _fake_meta)
    monkeypatch.setattr(pipeline, "gather_reference_logprobs", _fail_gather)

    batch_stats = pipeline._collect_batch_stats(ctx, gen_batch, reward_comp)
    assert batch_stats is not None
    assert calls["ref_meta"] == 1
    assert calls["gather"] == 0


def test_log_sample_table_logs_to_wandb(monkeypatch, rtl):
    """_log_sample_table should build a wandb.Table with prompts/completions."""
    rows_logged = {}

    class _FakeWandbRun:
        def log(self, data, step):
            rows_logged["data"] = data
            rows_logged["step"] = step

    class _FakeTable:
        def __init__(self, **kwargs):
            rows_logged["kwargs"] = kwargs

    def _fake_table(*args, **kwargs):
        return _FakeTable(columns=kwargs.get("columns"), rows=kwargs.get("rows"))

    fake_wandb = SimpleNamespace(
        Table=lambda columns=None, rows=None, **_kwargs: _FakeTable(
            columns=columns, rows=rows
        )
    )
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    metrics = reload(import_module("maxent_grpo.training.metrics"))

    class _Accelerator:
        is_main_process = True

    ctx = SimpleNamespace(
        logging=SimpleNamespace(wandb_run=_FakeWandbRun()),
        runtime=SimpleNamespace(accelerator=_Accelerator()),
    )
    state = SimpleNamespace(global_step=5)
    prepared = SimpleNamespace(
        reward_comp=SimpleNamespace(
            pairs=SimpleNamespace(prompts=["p1", "p2"], completions=["c1", "c2"]),
            per_reward_values={"accuracy": [0.1, 0.2]},
            advantage_samples=[0.3, 0.4],
        )
    )

    metrics._log_sample_table(ctx, state, prepared)
    assert rows_logged["step"] == 5


def test_maybe_validate_runs_on_schedule(monkeypatch, rtl):
    called = {}
    monkeypatch.setattr(
        rtl,
        "run_validation_step",
        lambda step, ctx: called.setdefault("args", (step, ctx)),
    )
    eval_cfg = SimpleNamespace(enabled=True, every_n_steps=2)
    val_ctx = object()
    rtl._maybe_validate(eval_cfg, val_ctx, global_step=4)
    assert called["args"] == (4, val_ctx)
    rtl._maybe_validate(eval_cfg, val_ctx, global_step=3)
    assert called["args"] == (4, val_ctx)  # unchanged when condition not met


def test_train_step_sets_generation_stats_and_skips_empty_batch(monkeypatch, rtl):
    gen_stats = {}
    ctx = SimpleNamespace(
        optimization=SimpleNamespace(schedule=SimpleNamespace(grad_accum_steps=1)),
        runtime=SimpleNamespace(accelerator=SimpleNamespace(), model=None),
        generation=SimpleNamespace(generation_stats=gen_stats),
        scoring=SimpleNamespace(),
    )
    state = rtl.TrainingLoopState()
    resources = rtl.StepResources(generator=lambda *_a, **_k: None, validation_ctx=None)
    step_info = SimpleNamespace(epoch=0, step_in_epoch=0, batch={})

    monkeypatch.setattr(
        rtl,
        "detect_deepspeed_state",
        lambda *_a, **_k: SimpleNamespace(use_deepspeed=False, zero_stage=0),
    )
    monkeypatch.setattr(rtl, "prepare_training_batch", lambda *_a, **_k: None)

    result = rtl._train_step(ctx, state, step_info, resources)

    assert result is False
    assert gen_stats["current_step"] == 0
    assert state.num_input_tokens_seen == 0.0


def test_train_step_logs_skip_stage(monkeypatch, rtl):
    class _Logger:
        def __init__(self):
            self.records = []

        def warning(self, msg, *args, **kwargs):
            self.records.append(msg % args if args else msg)

        def debug(self, *args, **kwargs):
            pass

    logger = _Logger()
    ctx = SimpleNamespace(
        optimization=SimpleNamespace(schedule=SimpleNamespace(grad_accum_steps=1)),
        runtime=SimpleNamespace(accelerator=SimpleNamespace(), model=None),
        generation=SimpleNamespace(generation_stats={}),
        scoring=SimpleNamespace(),
    )
    state = rtl.TrainingLoopState()
    resources = rtl.StepResources(generator=lambda *_a, **_k: None, validation_ctx=None)
    step_info = SimpleNamespace(epoch=0, step_in_epoch=2, batch={})

    monkeypatch.setattr(
        rtl,
        "detect_deepspeed_state",
        lambda *_a, **_k: SimpleNamespace(use_deepspeed=False, zero_stage=0),
    )

    def _fake_prepare(ctx_inner, *_a, **_k):
        setattr(ctx_inner.runtime, "_last_skip_stage", "policy_scoring")
        return None

    monkeypatch.setattr(rtl, "prepare_training_batch", _fake_prepare)
    monkeypatch.setattr(rtl, "LOG", logger)

    result = rtl._train_step(ctx, state, step_info, resources)

    assert result is False
    assert not hasattr(ctx.runtime, "_last_skip_stage")
    assert logger.records
    assert "stage=policy_scoring" in logger.records[0]


def test_train_step_defers_deepspeed_accumulation(monkeypatch, rtl):
    accelerator = SimpleNamespace(
        is_main_process=False, backward=lambda *_a, **_k: None
    )
    schedule = SimpleNamespace(grad_accum_steps=2)
    ctx = SimpleNamespace(
        optimization=SimpleNamespace(schedule=schedule, handles=SimpleNamespace()),
        runtime=SimpleNamespace(accelerator=accelerator, model=None),
        generation=SimpleNamespace(generation_stats={}),
        scoring=SimpleNamespace(clipping=None, weighting=None),
    )
    state = rtl.TrainingLoopState()
    resources = rtl.StepResources(generator=lambda *_a, **_k: None, validation_ctx=None)
    step_info = SimpleNamespace(epoch=0, step_in_epoch=0, batch={})

    prepared = SimpleNamespace(
        grouped_completions=[],
        reward_comp=SimpleNamespace(
            total_utils=[0.0],
            per_reward_values={},
            advantage_samples=[0.0],
            advantage=SimpleNamespace(grouped=[[0.0]]),
            q_grouped=[[1.0]],
        ),
        weight_stats=None,
        scores=None,
        ref_stats=None,
        total_input_tokens=0.0,
    )
    loss_outputs = SimpleNamespace(loss=0.0)

    monkeypatch.setattr(
        rtl,
        "detect_deepspeed_state",
        lambda *_a, **_k: SimpleNamespace(use_deepspeed=True, zero_stage=2),
    )
    monkeypatch.setattr(rtl, "prepare_training_batch", lambda *_a, **_k: prepared)
    monkeypatch.setattr(rtl, "build_loss_inputs", lambda *_a, **_k: (None,))
    monkeypatch.setattr(rtl, "evaluate_losses", lambda *_a, **_k: (loss_outputs, None))
    monkeypatch.setattr(rtl, "_scheduled_learning_rate", lambda *_a, **_k: 0.0)
    monkeypatch.setattr(rtl, "_epoch_progress", lambda *_a, **_k: 0.0)
    monkeypatch.setattr(
        rtl,
        "require_accumulation_context",
        lambda *_a, **_k: type(
            "Ctx",
            (),
            {"__enter__": lambda self: None, "__exit__": lambda self, *args: False},
        )(),
    )
    monkeypatch.setattr(
        rtl,
        "_optimizer_step",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("no step")),
    )

    result = rtl._train_step(ctx, state, step_info, resources)
    assert result is False
    assert state.global_step == 0  # optimizer not stepped


def test_train_step_defers_when_not_syncing(monkeypatch, rtl):
    accelerator = SimpleNamespace(
        is_main_process=False,
        backward=lambda *_a, **_k: None,
    )
    ctx = SimpleNamespace(
        optimization=SimpleNamespace(
            schedule=SimpleNamespace(grad_accum_steps=1), handles=SimpleNamespace()
        ),
        runtime=SimpleNamespace(accelerator=accelerator, model=None),
        generation=SimpleNamespace(generation_stats={}),
        scoring=SimpleNamespace(clipping=None, weighting=None),
    )
    state = rtl.TrainingLoopState()
    resources = rtl.StepResources(generator=lambda *_a, **_k: None, validation_ctx=None)
    step_info = SimpleNamespace(epoch=0, step_in_epoch=0, batch={})

    prepared = SimpleNamespace(
        grouped_completions=[],
        reward_comp=SimpleNamespace(
            total_utils=[0.0],
            per_reward_values={},
            advantage_samples=[0.0],
            advantage=SimpleNamespace(grouped=[[0.0]]),
            q_grouped=[[1.0]],
        ),
        weight_stats=None,
        scores=None,
        ref_stats=None,
        total_input_tokens=0.0,
    )
    loss_outputs = SimpleNamespace(loss=0.0)

    monkeypatch.setattr(
        rtl,
        "detect_deepspeed_state",
        lambda *_a, **_k: SimpleNamespace(use_deepspeed=False, zero_stage=0),
    )
    monkeypatch.setattr(rtl, "prepare_training_batch", lambda *_a, **_k: prepared)
    monkeypatch.setattr(rtl, "build_loss_inputs", lambda *_a, **_k: (None,))
    monkeypatch.setattr(rtl, "evaluate_losses", lambda *_a, **_k: (loss_outputs, None))
    monkeypatch.setattr(rtl, "_scheduled_learning_rate", lambda *_a, **_k: 0.0)
    monkeypatch.setattr(rtl, "_epoch_progress", lambda *_a, **_k: 0.0)
    monkeypatch.setattr(
        rtl,
        "require_accumulation_context",
        lambda *_a, **_k: type(
            "Ctx",
            (),
            {"__enter__": lambda self: None, "__exit__": lambda self, *args: False},
        )(),
    )
    monkeypatch.setattr(rtl, "sync_gradients_enabled", lambda *_a, **_k: False)
    monkeypatch.setattr(
        rtl,
        "_optimizer_step",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("no step")),
    )

    result = rtl._train_step(ctx, state, step_info, resources)
    assert result is False
    assert state.global_step == 0


def test_train_step_runs_optimizer_step_when_ready(monkeypatch, rtl):
    accelerator = SimpleNamespace(
        is_main_process=True,
        gradient_state=None,
        backward=lambda *_a, **_k: None,
    )
    schedule = SimpleNamespace(
        grad_accum_steps=1,
        steps_per_epoch=4,
        total_training_steps=4,
    )
    weighting = SimpleNamespace(tau=0.1, beta=0.2)
    ctx = SimpleNamespace(
        optimization=SimpleNamespace(
            schedule=schedule,
            handles=SimpleNamespace(learning_rate=1e-4),
        ),
        runtime=SimpleNamespace(accelerator=accelerator, model=object()),
        generation=SimpleNamespace(generation_stats={}),
        scoring=SimpleNamespace(weighting=weighting, clipping=SimpleNamespace()),
        logging=SimpleNamespace(
            wandb_run=None,
            save_strategy="no",
            save_steps=0,
            save_checkpoint=lambda *_a, **_k: None,
        ),
        controller=SimpleNamespace(state_path="/tmp/controller"),
        evaluation=SimpleNamespace(enabled=False),
    )
    state = rtl.TrainingLoopState()
    resources = rtl.StepResources(generator=None, validation_ctx=None)
    step_info = SimpleNamespace(epoch=0, step_in_epoch=0, batch={})
    prepared = SimpleNamespace(
        grouped_completions=[["one"]],
        reward_comp=SimpleNamespace(
            total_utils=[0.0],
            per_reward_values={},
            advantage_samples=[0.0],
            advantage=SimpleNamespace(grouped=[[0.0]]),
            q_grouped=[[1.0]],
        ),
        weight_stats=SimpleNamespace(),
        length_stats=SimpleNamespace(),
        ref_stats=SimpleNamespace(ref_logp_mean=0.0, avg_completion_tokens=2.0),
        num_completion_tokens=2.0,
        total_input_tokens=5.0,
        scores=SimpleNamespace(),
        seed_metrics=None,
    )
    loss_outputs = SimpleNamespace(
        loss=SimpleNamespace(requires_grad=False),
        total_loss_scalar=1.0,
        policy_loss_scalar=0.5,
        kl_loss_scalar=0.1,
        weighted_kl_loss_scalar=0.05,
        clip_loss_scalar=None,
    )
    diagnostics = SimpleNamespace(
        kl_value=0.1,
        clip_ratio=0.0,
        clip_ratio_low_mean=0.0,
        clip_ratio_low_min=0.0,
        clip_ratio_high_mean=0.0,
        clip_ratio_high_max=0.0,
        clip_ratio_region_mean=0.0,
        kl_per_token_by_len_bucket={},
        kl_token_count_by_len_bucket={},
    )
    monkeypatch.setattr(rtl, "prepare_training_batch", lambda *_a, **_k: prepared)
    monkeypatch.setattr(rtl, "build_loss_inputs", lambda *_a, **_k: ("loss",))
    monkeypatch.setattr(
        rtl, "evaluate_losses", lambda *_a, **_k: (loss_outputs, diagnostics)
    )
    monkeypatch.setattr(rtl, "detect_deepspeed_state", lambda *_a, **_k: SimpleNamespace(use_deepspeed=False, zero_stage=0))
    monkeypatch.setattr(rtl, "_scheduled_learning_rate", lambda *_a, **_k: 5e-4)
    monkeypatch.setattr(rtl, "_epoch_progress", lambda *_a, **_k: 0.25)
    monkeypatch.setattr(rtl, "sync_gradients_enabled", lambda *_a, **_k: True)
    monkeypatch.setattr(rtl, "summarize_weight_stats", lambda *_a, **_k: SimpleNamespace())
    log_calls = {"local": 0, "global": 0}
    monkeypatch.setattr(
        rtl,
        "log_local_step",
        lambda *_a, **_k: log_calls.__setitem__("local", log_calls["local"] + 1),
    )
    monkeypatch.setattr(
        rtl,
        "log_training_step",
        lambda *_a, **_k: log_calls.__setitem__("global", log_calls["global"] + 1),
    )
    optimizer_called = {}

    def _fake_opt_step(ctx_obj, loop_state, lr):
        optimizer_called["lr"] = lr
        loop_state.global_step += 1
        return 0.75

    monkeypatch.setattr(rtl, "_optimizer_step", _fake_opt_step)
    monkeypatch.setattr(rtl, "maybe_update_beta", lambda *_a, **_k: None)
    monkeypatch.setattr(rtl, "maybe_update_tau", lambda *_a, **_k: None)
    monkeypatch.setattr(rtl, "broadcast_controller_state", lambda *_a, **_k: None)
    monkeypatch.setattr(rtl, "save_controller_state", lambda *_a, **_k: None)
    monkeypatch.setattr(rtl, "_maybe_validate", lambda *_a, **_k: None)
    monkeypatch.setattr(rtl, "maybe_checkpoint", lambda *_a, **_k: None)
    monkeypatch.setattr(rtl, "check_stop_condition", lambda *_a, **_k: None)

    result = rtl._train_step(ctx, state, step_info, resources)
    assert result is False
    assert state.global_step == 1
    assert optimizer_called["lr"] == pytest.approx(5e-4)
    assert state.num_input_tokens_seen == pytest.approx(5.0)
    assert log_calls["local"] == 1
    assert log_calls["global"] == 1

def test_pipeline_replay_logs_expected_metrics(monkeypatch):
    rtl = reload(import_module("maxent_grpo.training.loop"))
    pipeline_mod = reload(import_module("maxent_grpo.training.pipeline"))
    metrics_mod = reload(import_module("maxent_grpo.training.metrics"))

    class _Writer:
        def __init__(self):
            self.logged = []

        def log(self, metrics, step):
            self.logged.append((metrics, step))

        def flush(self):
            return None

    writer = _Writer()
    logging_handles = LoggingHandles(
        metric_writer=writer,
        save_checkpoint=lambda *_a, **_k: None,
        save_strategy="no",
        save_steps=0,
        wandb_run=None,
    )
    weighting = SimpleNamespace(
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
        _tau_entropy_ema=0.2,
    )
    ctx = SimpleNamespace(
        runtime=SimpleNamespace(
            accelerator=SimpleNamespace(is_main_process=True, log=lambda *_a, **_k: None),
            tokenizer=SimpleNamespace(),
            device="cpu",
        ),
        generation=SimpleNamespace(
            max_completion_len=4,
            use_vllm=False,
            generation_stats={},
        ),
        optimization=SimpleNamespace(
            schedule=SimpleNamespace(steps_per_epoch=10),
        ),
        scoring=SimpleNamespace(
            weighting=weighting,
            clipping=SimpleNamespace(),
            batching=SimpleNamespace(
                prompt_length_cache_get=lambda *_a, **_k: SimpleNamespace(
                    input_ids=[], attention_mask=[]
                )
            ),
        ),
        logging=logging_handles,
    )
    reward_comp = SimpleNamespace(
        ref_logprob_meta=[],
        pairs=SimpleNamespace(
            prompts=["p1", "p2"],
            completions=["c1", "c2"],
        ),
        advantage=SimpleNamespace(grouped=[[0.2, -0.1], [0.3, -0.2]]),
        q_grouped=[[0.5, 0.4], [0.3, 0.6]],
        total_utils=[0.2, 0.4],
        advantage_samples=[0.1, -0.05],
        per_reward_values={"accuracy": [1.0, 0.0]},
    )
    gen_batch = SimpleNamespace(grouped_completions=[["c1", "c1b"], ["c2", "c2b"]])
    score_batch = SimpleNamespace(
        total_sequences=4,
        prompt_entries=[SimpleNamespace(length=2), SimpleNamespace(length=2)],
        max_prompt_len=4,
    )
    ref_stats = SimpleNamespace(ref_logp_mean=-0.4, avg_completion_tokens=3.0)
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
        clipped_ratio=0.0,
        min_terminated=2.0,
        mean_terminated=2.5,
        max_terminated=3.0,
    )
    monkeypatch.setattr(
        pipeline_mod,
        "build_score_batch",
        lambda *_a, **_k: score_batch,
    )
    monkeypatch.setattr(
        pipeline_mod,
        "compute_weight_stats",
        lambda *_a, **_k: weight_stats,
    )
    monkeypatch.setattr(
        pipeline_mod,
        "summarize_completion_lengths",
        lambda *_a, **_k: (None, length_stats, 4.0),
    )
    monkeypatch.setattr(
        pipeline_mod,
        "gather_reference_logprobs",
        lambda *_a, **_k: ref_stats,
    )

    batch_stats = pipeline_mod._collect_batch_stats(ctx, gen_batch, reward_comp)
    prepared = SimpleNamespace(
        reward_comp=reward_comp,
        weight_stats=batch_stats.weight_stats,
        length_stats=batch_stats.length_stats,
        ref_stats=batch_stats.ref_stats,
        num_completion_tokens=batch_stats.num_completion_tokens,
        grouped_completions=gen_batch.grouped_completions,
        seed_metrics={},
    )
    loss_outputs = SimpleNamespace(
        total_loss_scalar=0.9,
        kl_loss_scalar=0.06,
        policy_loss_scalar=0.8,
        weighted_kl_loss_scalar=0.06,
        clip_loss_scalar=None,
        scalars=SimpleNamespace(kl_loss=0.06),
    )
    diagnostics = metrics_mod.BatchDiagnostics(
        kl_value=0.06,
        clip_ratio=0.0,
        clip_ratio_low_mean=0.0,
        clip_ratio_low_min=0.0,
        clip_ratio_high_mean=0.0,
        clip_ratio_high_max=0.0,
        clip_ratio_region_mean=0.0,
        kl_per_token_by_len_bucket={},
        kl_token_count_by_len_bucket={},
    )
    log_artifacts = LogStepArtifacts(
        loss_outputs=loss_outputs,
        diagnostics=diagnostics,
        grad_norm_scalar=0.25,
        epoch_progress=0.1,
    )
    state = SimpleNamespace(
        global_step=1,
        metric_sums={},
        metric_counts={},
        num_input_tokens_seen=10.0,
    )

    rtl.log_training_step(ctx, state, prepared, log_artifacts, current_lr=1e-4)
    assert writer.logged, "metrics should be emitted"
    metrics = writer.logged[-1][0]
    assert metrics["train/weight_entropy"] == pytest.approx(
        weight_stats.weight_entropy
    )
    assert metrics["train/weighting/tau"] == pytest.approx(weighting.tau)
    assert metrics["train/rewards/accuracy/mean"] == pytest.approx(0.5)
    assert metrics["train/completions/mean_length_sampled"] == pytest.approx(
        length_stats.mean_length
    )


def test_run_epoch_retries_sampler_with_int(monkeypatch, rtl):
    calls = []

    class _Sampler:
        def __init__(self):
            self.calls = 0

        def set_epoch(self, value):
            self.calls += 1
            calls.append(value)
            if self.calls == 1:
                raise TypeError("bad type")

    sampler = _Sampler()
    ctx = SimpleNamespace(
        runtime=SimpleNamespace(train_loader=[{"a": 1}], train_sampler=sampler),
    )
    state = rtl.TrainingLoopState()
    resources = rtl.StepResources(generator=None, validation_ctx=None)

    monkeypatch.setattr(rtl, "_train_step", lambda *_a, **_k: False)
    result = rtl._run_epoch(ctx, state, epoch=3, resources=resources)

    assert result is False
    assert calls == [3, 3]
    assert sampler.calls == 2


def test_run_training_loop_logs_and_finishes_wandb(monkeypatch, rtl):
    calls = {}

    class _Accel:
        is_main_process = True

    class _Optimizer:
        def __init__(self):
            self.calls = []

        def zero_grad(self, set_to_none):
            self.calls.append(set_to_none)

    optimizer = _Optimizer()
    schedule = SimpleNamespace(
        grad_accum_steps=1, steps_per_epoch=1, total_training_steps=1, num_epochs=1
    )
    generation_stats = {
        "vllm_retry_rounds": 1,
        "vllm_backfilled_prompts": 2,
        "vllm_failed_prompts": 3,
        "dropped_prompts": 4,
    }
    ctx = SimpleNamespace(
        runtime=SimpleNamespace(
            accelerator=_Accel(),
            model=object(),
            tokenizer=object(),
            device="cpu",
            train_loader=[{"batch": 1}],
        ),
        generation=SimpleNamespace(
            max_prompt_len=1,
            max_completion_len=1,
            gen_temperature=0.1,
            gen_top_p=0.9,
            use_vllm=False,
            vllm=None,
            penalty=SimpleNamespace(),
            generation_stats=generation_stats,
        ),
        evaluation=SimpleNamespace(enabled=False, every_n_steps=None),
        reward=object(),
        logging=SimpleNamespace(
            wandb_run=SimpleNamespace(finish=lambda: calls.setdefault("finish", True))
        ),
        optimization=SimpleNamespace(
            schedule=schedule, handles=SimpleNamespace(optimizer=optimizer)
        ),
        controller=SimpleNamespace(state_path="/tmp/state", resume_from=None),
        scoring=SimpleNamespace(weighting=SimpleNamespace()),
    )

    class _GenCtx:
        def __init__(self, **kwargs):
            calls.setdefault("gen_ctx", kwargs)

    class _CompletionGen:
        def __init__(self, ctx):
            calls.setdefault("completion_ctx", ctx)

        def generate(self, *args, **kwargs):
            calls.setdefault("generate_called", True)

    monkeypatch.setattr(rtl, "GenerationContext", lambda **kwargs: _GenCtx(**kwargs))
    monkeypatch.setattr(rtl, "CompletionGenerator", lambda ctx: _CompletionGen(ctx))
    monkeypatch.setattr(
        rtl,
        "configure_accumulation_steps",
        lambda accel, steps: calls.setdefault("config_accum", (accel, steps)),
    )
    monkeypatch.setattr(
        rtl,
        "_maybe_patch_zero_no_sync",
        lambda model: calls.setdefault("patched_zero", model),
    )
    monkeypatch.setattr(rtl, "replace", lambda obj, **kwargs: obj)
    monkeypatch.setattr(
        rtl,
        "load_controller_state_chain",
        lambda controller, accel, weighting: calls.setdefault("loaded_ctrl", True),
    )
    monkeypatch.setattr(
        rtl,
        "maybe_load_accelerator_state",
        lambda resume, accel: calls.setdefault("loaded_accel", True),
    )
    monkeypatch.setattr(
        rtl,
        "_run_epoch",
        lambda _ctx, _state, epoch, _resources: calls.setdefault("run_epoch", epoch),
    )

    rtl.run_training_loop(ctx)

    assert calls["config_accum"][1] == schedule.grad_accum_steps
    assert calls["patched_zero"] is ctx.runtime.model
    assert calls["loaded_ctrl"] is True
    assert calls["loaded_accel"] is True
    assert calls["run_epoch"] == 0
    assert calls["finish"] is True


def test_clear_stale_controller_state_removes_file(monkeypatch, rtl):
    """overwrite_output_dir should delete old controller snapshots before training."""
    state_mod = reload(import_module("maxent_grpo.training.state"))

    removed = {}
    waited = {}

    class _Accelerator:
        is_main_process = True

        def wait_for_everyone(self):
            waited["called"] = True

    accelerator = _Accelerator()
    controller = SimpleNamespace(
        resume_from=None,
        overwrite_existing=True,
        state_path="/tmp/controller_state.json",
    )

    monkeypatch.setattr(
        state_mod.os.path, "isfile", lambda path: path == controller.state_path
    )

    def _fake_remove(path):
        removed["path"] = path

    monkeypatch.setattr(state_mod.os, "remove", _fake_remove)
    state_mod.maybe_clear_stale_controller_state(accelerator, controller)

    assert removed["path"] == controller.state_path
    assert waited["called"] is True


def test_maybe_checkpoint_calls_save_on_all_ranks(monkeypatch, rtl):
    """DeepSpeed checkpoints must be invoked by every rank, not just rank 0."""
    state_mod = reload(import_module("maxent_grpo.training.state"))
    state_mod._checkpoint_log_once = {
        "config": False,
        "strategy": False,
        "steps": False,
    }
    calls = []
    current_rank = {"rank": None}

    def _save(label):
        calls.append((current_rank["rank"], label))

    logging_cfg = SimpleNamespace(
        save_strategy="steps",
        save_steps=1,
        save_checkpoint=_save,
    )

    class _Accel:
        def __init__(self, is_main):
            self.is_main_process = is_main
            self.events = []

        def wait_for_everyone(self):
            self.events.append("wait")

    main_accel = _Accel(is_main=True)
    worker_accel = _Accel(is_main=False)

    current_rank["rank"] = "main"
    state_mod.maybe_checkpoint(logging_cfg, main_accel, global_step=1)
    current_rank["rank"] = "worker"
    state_mod.maybe_checkpoint(logging_cfg, worker_accel, global_step=1)

    assert set(calls) == {("main", "checkpoint-1"), ("worker", "checkpoint-1")}
    assert main_accel.events == ["wait", "wait"]
    assert worker_accel.events == ["wait", "wait"]


def test_train_step_updates_controllers_and_saves_state(monkeypatch, rtl):
    """The train step should propagate KL stats into controller updates and checkpoints."""

    class _Accelerator:
        is_main_process = True
        sync_gradients = True
        gradient_accumulation_steps = 1

        def backward(self, loss):
            self.last_loss = loss

        def accumulate(self, _model):
            class _Ctx:
                def __enter__(self_inner):
                    return self_inner

                def __exit__(self_inner, exc_type, exc, tb):
                    return False

            return _Ctx()

        def wait_for_everyone(self):
            return None

    accelerator = _Accelerator()
    weighting = SimpleNamespace(beta=3.0, tau=0.1)
    ctx = SimpleNamespace(
        runtime=SimpleNamespace(accelerator=accelerator, model=object()),
        scoring=SimpleNamespace(
            clipping=SimpleNamespace(),
            weighting=weighting,
            batching=SimpleNamespace(),
        ),
        optimization=SimpleNamespace(
            schedule=SimpleNamespace(grad_accum_steps=1, total_training_steps=0),
            handles=SimpleNamespace(),
        ),
        logging=SimpleNamespace(
            wandb_run=None,
            save_strategy="no",
            save_steps=0,
            log_metrics=lambda *_a, **_k: None,
            save_checkpoint=lambda *_a, **_k: None,
        ),
        evaluation=SimpleNamespace(enabled=False),
        controller=SimpleNamespace(state_path="/tmp/controller"),
        generation=SimpleNamespace(),
    )
    state = rtl.TrainingLoopState()
    resources = rtl.StepResources(
        generator=lambda *_a, **_k: None, validation_ctx=object()
    )
    step_info = SimpleNamespace(epoch=0, step_in_epoch=0, batch={"prompt": ["p"]})
    prepared = SimpleNamespace(
        grouped_completions=[["one"]],
        reward_comp=SimpleNamespace(
            total_utils=[0.0],
            per_reward_values={},
            advantage_samples=[0.0],
            advantage=SimpleNamespace(grouped=[[0.0]]),
            q_grouped=[[1.0]],
        ),
        weight_stats=SimpleNamespace(),
        ref_stats=SimpleNamespace(ref_logp_mean=0.0, avg_completion_tokens=2.0),
        length_stats=SimpleNamespace(),
        num_completion_tokens=2.0,
        total_input_tokens=5.0,
        scores=SimpleNamespace(),
        seed_heatmap=None,
    )
    loss_outputs = SimpleNamespace(loss=object(), kl_loss_scalar=0.25)
    recorded = {}
    aggregated_view = SimpleNamespace(entropy=0.0)

    monkeypatch.setattr(rtl, "prepare_training_batch", lambda *_a, **_k: prepared)
    monkeypatch.setattr(rtl, "build_loss_inputs", lambda *_a, **_k: ("group", "ratio"))
    monkeypatch.setattr(
        rtl,
        "evaluate_losses",
        lambda *_a, **_k: (loss_outputs, SimpleNamespace()),
    )
    monkeypatch.setattr(rtl, "_scheduled_learning_rate", lambda *_a, **_k: 0.001)
    monkeypatch.setattr(rtl, "_epoch_progress", lambda *_a, **_k: 0.5)
    monkeypatch.setattr(
        rtl, "log_local_step", lambda *_a, **_k: recorded.setdefault("log_local", True)
    )
    monkeypatch.setattr(
        rtl,
        "log_training_step",
        lambda *_a, **_k: recorded.setdefault("log_training", True),
    )
    monkeypatch.setattr(
        rtl, "summarize_weight_stats", lambda *_a, **_k: aggregated_view
    )

    def _fake_opt_step(_ctx, loop_state, current_lr):
        recorded["optimizer_lr"] = current_lr
        loop_state.global_step += 1
        return 0.0

    monkeypatch.setattr(rtl, "_optimizer_step", _fake_opt_step)
    monkeypatch.setattr(
        rtl,
        "maybe_update_beta",
        lambda cfg, measured: recorded.setdefault("beta_call", (cfg, measured)),
    )
    monkeypatch.setattr(
        rtl,
        "maybe_update_tau",
        lambda cfg, stats, step: recorded.setdefault("tau_call", (cfg, stats, step)),
    )
    monkeypatch.setattr(
        rtl,
        "save_controller_state",
        lambda path, cfg: recorded.setdefault("saved", (path, cfg)),
    )
    monkeypatch.setattr(
        rtl,
        "_maybe_validate",
        lambda eval_cfg, val_ctx, step: recorded.setdefault("validated", step),
    )
    monkeypatch.setattr(
        rtl,
        "maybe_checkpoint",
        lambda logging_cfg, accel, step: recorded.setdefault("checkpoint", step),
    )
    monkeypatch.setattr(
        rtl,
        "check_stop_condition",
        lambda schedule, loop_state: recorded.setdefault(
            "checked", loop_state.global_step
        ),
    )

    result = rtl._train_step(ctx, state, step_info, resources)

    assert result is state.stop_training
    assert recorded["beta_call"] == (weighting, loss_outputs.kl_loss_scalar)
    assert recorded["tau_call"] == (weighting, aggregated_view, state.global_step)
    assert recorded["saved"] == (ctx.controller.state_path, weighting)
    assert recorded["validated"] == state.global_step
    assert recorded["checkpoint"] == state.global_step
    assert recorded["checked"] == state.global_step
    assert state.num_input_tokens_seen == pytest.approx(prepared.total_input_tokens)


def test_controller_objective_hook_applies_meta_updates(monkeypatch, rtl):
    class _Accelerator:
        is_main_process = True
        sync_gradients = True
        gradient_accumulation_steps = 1

        def backward(self, loss):
            return loss

        def accumulate(self, _model):
            class _Ctx:
                def __enter__(self_inner):
                    return self_inner

                def __exit__(self_inner, exc_type, exc, tb):
                    return False

            return _Ctx()

        def wait_for_everyone(self):
            return None

    accelerator = _Accelerator()
    weighting = SimpleNamespace(beta=0.4, tau=0.2)
    ctx = SimpleNamespace(
        runtime=SimpleNamespace(accelerator=accelerator, model=object()),
        scoring=SimpleNamespace(
            clipping=SimpleNamespace(),
            weighting=weighting,
            batching=SimpleNamespace(),
        ),
        optimization=SimpleNamespace(
            schedule=SimpleNamespace(grad_accum_steps=1, total_training_steps=0),
            handles=SimpleNamespace(),
        ),
        logging=SimpleNamespace(
            wandb_run=None,
            save_strategy="no",
            save_steps=0,
            log_metrics=lambda *_a, **_k: None,
            save_checkpoint=lambda *_a, **_k: None,
        ),
        evaluation=SimpleNamespace(enabled=False),
        controller=SimpleNamespace(state_path="/tmp/controller"),
        generation=SimpleNamespace(),
    )
    ctx.settings = SimpleNamespace(controller_objective=SimpleNamespace(), controller_meta_manager=None)
    state = rtl.TrainingLoopState()
    resources = rtl.StepResources(
        generator=lambda *_a, **_k: None, validation_ctx=object()
    )
    step_info = SimpleNamespace(epoch=0, step_in_epoch=0, batch={"prompt": ["p"]})
    prepared = SimpleNamespace(
        grouped_completions=[["one"]],
        reward_comp=SimpleNamespace(
            total_utils=[0.0],
            per_reward_values={},
            advantage_samples=[0.0],
            advantage=SimpleNamespace(grouped=[[0.0]]),
            q_grouped=[[1.0]],
        ),
        weight_stats=SimpleNamespace(weight_entropy=0.3),
        ref_stats=SimpleNamespace(ref_logp_mean=0.0, avg_completion_tokens=2.0),
        length_stats=SimpleNamespace(),
        num_completion_tokens=2.0,
        total_input_tokens=5.0,
        scores=SimpleNamespace(),
        seed_heatmap=None,
    )
    loss_outputs = SimpleNamespace(loss=object(), kl_loss_scalar=0.15)
    gradients = ControllerGradients(tau_grad=0.2, beta_grad=-0.05)

    def _objective_compute(meta_ctx):
        assert meta_ctx.weighting is weighting
        return gradients

    ctx.settings.controller_objective.compute = _objective_compute
    monkeypatch.setattr(rtl, "prepare_training_batch", lambda *_a, **_k: prepared)
    monkeypatch.setattr(rtl, "build_loss_inputs", lambda *_a, **_k: ("group", "ratio"))
    monkeypatch.setattr(
        rtl,
        "evaluate_losses",
        lambda *_a, **_k: (loss_outputs, SimpleNamespace()),
    )
    monkeypatch.setattr(rtl, "_scheduled_learning_rate", lambda *_a, **_k: 0.001)
    monkeypatch.setattr(rtl, "_epoch_progress", lambda *_a, **_k: 0.5)
    monkeypatch.setattr(
        rtl,
        "log_local_step",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(
        rtl,
        "log_training_step",
        lambda *_a, **_k: None,
    )

    def _fake_opt_step(_ctx, loop_state, current_lr):
        loop_state.global_step += 1
        return 0.0

    monkeypatch.setattr(rtl, "_optimizer_step", _fake_opt_step)
    monkeypatch.setattr(rtl, "maybe_update_beta", lambda *_, **__: None)
    monkeypatch.setattr(rtl, "maybe_update_tau", lambda *_, **__: None)
    monkeypatch.setattr(rtl, "broadcast_controller_state", lambda *_, **__: None)
    monkeypatch.setattr(rtl, "save_controller_state", lambda *_, **__: None)
    monkeypatch.setattr(rtl, "_maybe_validate", lambda *_, **__: None)
    monkeypatch.setattr(rtl, "maybe_checkpoint", lambda *_, **__: None)
    monkeypatch.setattr(rtl, "check_stop_condition", lambda *_, **__: None)

    calls = {}

    def _fake_meta_update(cfg, **kwargs):
        calls["meta_update"] = (cfg, kwargs)

    monkeypatch.setattr(rtl, "apply_meta_controller_update", _fake_meta_update)

    rtl._train_step(ctx, state, step_info, resources)
    assert "meta_update" in calls
    cfg, kwargs = calls["meta_update"]
    assert cfg is weighting
    assert kwargs["tau_grad"] == pytest.approx(gradients.tau_grad)
    assert kwargs["beta_grad"] == pytest.approx(gradients.beta_grad)

def test_maxent_smoke_logs_expected_metrics(monkeypatch):
    rtl = reload(import_module("maxent_grpo.training.loop"))
    target_steps = 2

    class _Writer:
        def __init__(self):
            self.logged = []

        def log(self, metrics, step):
            self.logged.append((metrics, step))

        def flush(self):
            return None

    writer = _Writer()
    logging_handles = LoggingHandles(
        metric_writer=writer,
        save_checkpoint=lambda *_a, **_k: None,
        save_strategy="steps",
        save_steps=1,
        wandb_run=None,
    )
    generation_stats = {
        "vllm_retry_rounds": 0,
        "vllm_backfilled_prompts": 0,
        "vllm_failed_prompts": 0,
        "dropped_prompts": 0,
    }
    generation_cfg = SimpleNamespace(
        max_prompt_len=16,
        max_completion_len=16,
        gen_temperature=1.0,
        gen_top_p=1.0,
        use_vllm=False,
        vllm=None,
        penalty=GenerationPenaltyConfig(),
        generation_stats=generation_stats,
    )
    evaluation_cfg = SimpleNamespace(
        enabled=False, every_n_steps=None, rows=[], batch_size=0
    )
    schedule = SimpleNamespace(
        num_epochs=1,
        steps_per_epoch=target_steps,
        total_training_steps=target_steps,
        grad_accum_steps=1,
        num_generations=1,
        max_grad_norm=1.0,
    )
    optimizer_stub = SimpleNamespace(zero_grad=lambda set_to_none=None: None)
    optimization = SimpleNamespace(
        schedule=schedule,
        handles=SimpleNamespace(
            optimizer=optimizer_stub,
            lr_scheduler=None,
            base_optimizer=optimizer_stub,
            learning_rate=1e-5,
        ),
    )
    weighting = SimpleNamespace(tau=0.3, beta=0.04, denom=1.0)
    clipping = SimpleNamespace(
        clip_range=0.05,
        use_clip_objective=True,
        clip_objective_coef=1.0,
        clip_adv_baseline=0.0,
    )
    batching = SimpleNamespace(
        logprob_chunk_size=0,
        score_slice=0,
        prompt_length_cache_get=lambda *_a, **_k: SimpleNamespace(
            input_ids=[], attention_mask=[]
        ),
    )
    scoring = SimpleNamespace(weighting=weighting, clipping=clipping, batching=batching)
    runtime = SimpleNamespace(
        accelerator=SimpleNamespace(is_main_process=True, gradient_state=None),
        model=SimpleNamespace(),
        tokenizer=SimpleNamespace(),
        train_loader=[{"prompt": ["p"], "answer": ["a"]} for _ in range(target_steps)],
        train_sampler=None,
        device="cpu",
        get_ref_model=lambda: None,
    )
    ctx = SimpleNamespace(
        runtime=runtime,
        generation=generation_cfg,
        evaluation=evaluation_cfg,
        optimization=optimization,
        scoring=scoring,
        logging=logging_handles,
        controller=SimpleNamespace(state_path=None, resume_from=None),
        reward=SimpleNamespace(reward_funcs=[], reward_weights=[]),
        eval_reward=None,
    )

    class _FakeGenerator:
        def __init__(self, *_args, **_kwargs):
            self.generate = lambda *_a, **_k: None

    class _FakeValidationCtx:
        def __init__(self, *_args, **_kwargs):
            return None

    monkeypatch.setattr(rtl, "CompletionGenerator", _FakeGenerator)
    monkeypatch.setattr(rtl, "ValidationContext", _FakeValidationCtx)
    monkeypatch.setattr(rtl, "GenerationContext", lambda *_a, **_k: SimpleNamespace())
    monkeypatch.setattr(rtl, "load_controller_state_chain", lambda *_a, **_k: None)
    monkeypatch.setattr(rtl, "maybe_load_accelerator_state", lambda *_a, **_k: None)
    monkeypatch.setattr(rtl, "_maybe_patch_zero_no_sync", lambda *_a, **_k: None)
    monkeypatch.setattr(rtl, "configure_accumulation_steps", lambda *_a, **_k: None)

    def _fake_train_step(ctx_inner, state, step_info, resources):
        idx = state.global_step
        metrics = {
            "train/kl": 0.01 * (idx + 1),
            "train/beta": 0.04 + 0.01 * idx,
            "train/weighting/beta": 0.04 + 0.01 * idx,
            "train/tau": ctx_inner.scoring.weighting.tau,
            "train/weighting/tau": ctx_inner.scoring.weighting.tau,
            "train/maxent_objective": 1.0,
            "train/grpo_objective": 0.0,
            "train/weight_entropy": 0.2 + 0.05 * idx,
        }
        ctx_inner.logging.metric_writer.log(metrics, idx)
        state.global_step += 1
        if state.global_step >= target_steps:
            state.stop_training = True
            return True
        return False

    monkeypatch.setattr(rtl, "_train_step", _fake_train_step)

    rtl.run_training_loop(ctx)

    assert len(writer.logged) == target_steps
    entropies = []
    for metrics, step in writer.logged:
        assert metrics["train/kl"] > 0.0
        assert "train/beta" in metrics
        assert "train/tau" in metrics
        assert metrics["train/maxent_objective"] == 1.0
        assert metrics["train/grpo_objective"] == 0.0
        entropies.append(metrics["train/weight_entropy"])
    assert entropies[0] != entropies[1]


def test_maxent_vs_grpo_parity(monkeypatch, tmp_path):
    rtl = reload(import_module("maxent_grpo.training.loop"))
    ckpt_dir = tmp_path / "ckpts"
    ckpt_dir.mkdir()
    runs = []

    class _Writer:
        def __init__(self):
            self.logged = []

        def log(self, metrics, step):
            self.logged.append((metrics, step))

        def flush(self):
            return None

    def _run_loop(train_grpo):
        writer = _Writer()
        logging_handles = LoggingHandles(
            metric_writer=writer,
            save_checkpoint=lambda name: runs.append((train_grpo, name)),
            save_strategy="steps",
            save_steps=1,
            wandb_run=None,
        )
        weighting = SimpleNamespace(
            tau=0.3 if not train_grpo else 0.0,
            beta=0.04,
            denom=1.0,
            train_grpo_objective=train_grpo,
        )
        ctx = SimpleNamespace(
            runtime=SimpleNamespace(
                accelerator=SimpleNamespace(is_main_process=True, gradient_state=None),
                model=SimpleNamespace(),
                tokenizer=SimpleNamespace(),
                train_loader=[{"prompt": ["p"], "answer": ["a"]}] * 2,
                train_sampler=None,
                device="cpu",
                get_ref_model=lambda: None,
            ),
            generation=SimpleNamespace(
                max_prompt_len=8,
                max_completion_len=8,
                gen_temperature=1.0,
                gen_top_p=1.0,
                use_vllm=False,
                vllm=None,
                penalty=GenerationPenaltyConfig(),
                generation_stats={
                    "vllm_retry_rounds": 0,
                    "vllm_backfilled_prompts": 0,
                    "vllm_failed_prompts": 0,
                    "dropped_prompts": 0,
                },
            ),
            evaluation=SimpleNamespace(enabled=False, every_n_steps=None, rows=[], batch_size=0),
            optimization=SimpleNamespace(
                schedule=SimpleNamespace(
                    num_epochs=1,
                    steps_per_epoch=2,
                    total_training_steps=2,
                    grad_accum_steps=1,
                    num_generations=1,
                    max_grad_norm=1.0,
                ),
                handles=SimpleNamespace(
                    optimizer=SimpleNamespace(zero_grad=lambda set_to_none=None: None),
                    lr_scheduler=None,
                    base_optimizer=SimpleNamespace(),
                    learning_rate=1e-5,
                ),
            ),
            scoring=SimpleNamespace(
                weighting=weighting,
                clipping=SimpleNamespace(
                    clip_range=0.05,
                    use_clip_objective=True,
                    clip_objective_coef=1.0,
                    clip_adv_baseline=0.0,
                ),
                batching=SimpleNamespace(
                    logprob_chunk_size=0,
                    score_slice=0,
                    prompt_length_cache_get=lambda *_a, **_k: SimpleNamespace(
                        input_ids=[], attention_mask=[]
                    ),
                ),
            ),
            logging=logging_handles,
            controller=SimpleNamespace(state_path=str(ckpt_dir / ("maxent" if not train_grpo else "grpo")), resume_from=None),
            reward=SimpleNamespace(reward_funcs=[], reward_weights=[]),
            eval_reward=None,
        )

        class _Gen:
            def __init__(self, *_a, **_k):
                self.generate = lambda *_x, **_y: None

        monkeypatch.setattr(rtl, "CompletionGenerator", _Gen)
        monkeypatch.setattr(rtl, "ValidationContext", lambda *_a, **_k: None)
        monkeypatch.setattr(rtl, "GenerationContext", lambda *_a, **_k: SimpleNamespace())

        def _fake_train_step(ctx_inner, state, *_a, **_k):
            idx = state.global_step
            metrics = {
                "train/kl": 0.02 + (0.01 if not train_grpo else 0.0),
                "train/beta": ctx_inner.scoring.weighting.beta,
                "train/tau": ctx_inner.scoring.weighting.tau,
                "train/weight_entropy": 0.3 if not train_grpo else 0.1,
                "train/weight_entropy_error": (0.3 if not train_grpo else 0.1) - 0.2,
                "train/kl_error_to_target": -0.05 if not train_grpo else -0.02,
                "train/maxent_objective": 0.0 if train_grpo else 1.0,
                "train/grpo_objective": 1.0 if train_grpo else 0.0,
            }
            ctx_inner.logging.metric_writer.log(metrics, idx)
            ctx_inner.logging.save_checkpoint(f"step-{idx}")
            state.global_step += 1
            return state.global_step >= ctx_inner.optimization.schedule.total_training_steps

        monkeypatch.setattr(rtl, "_train_step", _fake_train_step)
        rtl.run_training_loop(ctx)
        return writer.logged

    maxent_metrics = _run_loop(train_grpo=False)[0][0]
    grpo_metrics = _run_loop(train_grpo=True)[0][0]

    assert maxent_metrics["train/maxent_objective"] == 1.0
    assert grpo_metrics["train/grpo_objective"] == 1.0
    assert "train/kl" in maxent_metrics and "train/kl" in grpo_metrics
    assert maxent_metrics["train/weight_entropy"] != grpo_metrics["train/weight_entropy"]
    assert maxent_metrics["train/kl_error_to_target"] != grpo_metrics["train/kl_error_to_target"]
    assert len(runs) == 4  # checkpoint callback invoked for each step


def test_log_prompt_objective_respects_flags(monkeypatch, caplog, rtl):
    """Per-prompt objective logging should require an explicit flag or env."""

    ctx = SimpleNamespace(
        training_args=SimpleNamespace(log_prompt_objective=False),
        scoring=SimpleNamespace(weighting=None),
    )
    prepared = SimpleNamespace()
    entry = {
        "index": 0,
        "group_size": 2,
        "reward": 0.5,
        "kl": 0.1,
        "q_entropy": 0.2,
        "weight_entropy": 0.3,
        "objective": 0.8,
        "prompt": "prompt text",
    }
    monkeypatch.setattr(
        rtl,
        "_build_prompt_objective_entries",
        lambda *_args, **_kwargs: [entry],
    )
    monkeypatch.delenv("MAXENT_LOG_PROMPT_OBJECTIVE", raising=False)
    with caplog.at_level(logging.INFO):
        rtl._log_prompt_objective(ctx, prepared, 7)
    assert all("Prompt objective" not in rec.message for rec in caplog.records)
    caplog.clear()

    monkeypatch.setenv("MAXENT_LOG_PROMPT_OBJECTIVE", "1")
    with caplog.at_level(logging.INFO):
        rtl._log_prompt_objective(ctx, prepared, 8)
    assert any("Prompt objective" in rec.message for rec in caplog.records)
    caplog.clear()

    monkeypatch.delenv("MAXENT_LOG_PROMPT_OBJECTIVE", raising=False)
    ctx.training_args.log_prompt_objective = True
    with caplog.at_level(logging.INFO):
        rtl._log_prompt_objective(ctx, prepared, 9)
    assert any("Prompt objective" in rec.message for rec in caplog.records)


def test_eval_loop_smoke_logs_eval_metrics(monkeypatch):
    rtl = reload(import_module("maxent_grpo.training.loop"))
    target_steps = 2
    validation_calls = []

    class _Writer:
        def __init__(self):
            self.logged = []

        def log(self, metrics, step):
            self.logged.append((metrics, step))

        def flush(self):
            return None

    writer = _Writer()
    logging_handles = LoggingHandles(
        metric_writer=writer,
        save_checkpoint=lambda *_a, **_k: None,
        save_strategy="steps",
        save_steps=1,
        wandb_run=None,
    )
    weighting = SimpleNamespace(tau=0.25, beta=0.05, denom=1.0)
    ctx = SimpleNamespace(
        runtime=SimpleNamespace(
            accelerator=SimpleNamespace(is_main_process=True, gradient_state=None),
            model=SimpleNamespace(),
            tokenizer=SimpleNamespace(),
            train_loader=[{"prompt": ["p"], "answer": ["a"]} for _ in range(target_steps)],
            train_sampler=None,
            device="cpu",
            get_ref_model=lambda: None,
        ),
        generation=SimpleNamespace(
            max_prompt_len=8,
            max_completion_len=8,
            gen_temperature=1.0,
            gen_top_p=1.0,
            use_vllm=False,
            vllm=None,
            penalty=GenerationPenaltyConfig(),
            generation_stats={
                "vllm_retry_rounds": 0,
                "vllm_backfilled_prompts": 0,
                "vllm_failed_prompts": 0,
                "dropped_prompts": 0,
            },
        ),
        evaluation=SimpleNamespace(enabled=True, every_n_steps=1, rows=[], batch_size=0),
        optimization=SimpleNamespace(
            schedule=SimpleNamespace(
                num_epochs=1,
                steps_per_epoch=target_steps,
                total_training_steps=target_steps,
                grad_accum_steps=1,
                num_generations=1,
                max_grad_norm=1.0,
            ),
            handles=SimpleNamespace(
                optimizer=SimpleNamespace(zero_grad=lambda set_to_none=None: None),
                lr_scheduler=None,
                base_optimizer=SimpleNamespace(),
                learning_rate=1e-5,
            ),
        ),
        scoring=SimpleNamespace(
            weighting=weighting,
            clipping=SimpleNamespace(
                clip_range=0.05,
                use_clip_objective=True,
                clip_objective_coef=1.0,
                clip_adv_baseline=0.0,
            ),
            batching=SimpleNamespace(
                logprob_chunk_size=0,
                score_slice=0,
                prompt_length_cache_get=lambda *_a, **_k: SimpleNamespace(
                    input_ids=[], attention_mask=[]
                ),
            ),
        ),
        logging=logging_handles,
        controller=SimpleNamespace(state_path=None, resume_from=None),
        reward=SimpleNamespace(reward_funcs=[], reward_weights=[]),
        eval_reward=None,
    )

    class _Gen:
        def __init__(self, *_a, **_k):
            self.generate = lambda *_x, **_y: None

    val_ctx = SimpleNamespace(tag="val")
    monkeypatch.setattr(rtl, "CompletionGenerator", _Gen)
    monkeypatch.setattr(rtl, "ValidationContext", lambda *_a, **_k: val_ctx)
    monkeypatch.setattr(rtl, "GenerationContext", lambda *_a, **_k: SimpleNamespace())
    monkeypatch.setattr(rtl, "load_controller_state_chain", lambda *_a, **_k: None)
    monkeypatch.setattr(rtl, "maybe_load_accelerator_state", lambda *_a, **_k: None)
    monkeypatch.setattr(rtl, "_maybe_patch_zero_no_sync", lambda *_a, **_k: None)
    monkeypatch.setattr(rtl, "configure_accumulation_steps", lambda *_a, **_k: None)
    monkeypatch.setattr(
        rtl,
        "run_validation_step",
        lambda step, ctx_obj: validation_calls.append((step, ctx_obj)),
    )

    def _train_step(ctx_inner, state, _step_info, resources):
        metrics = {
            "train/kl": 0.02 + 0.01 * state.global_step,
            "train/tau": ctx_inner.scoring.weighting.tau,
            "eval/reward": 0.1 * (state.global_step + 1),
        }
        ctx_inner.logging.metric_writer.log(metrics, state.global_step)
        state.global_step += 1
        rtl._maybe_validate(
            ctx_inner.evaluation,
            resources.validation_ctx,
            state.global_step,
        )
        if state.global_step >= target_steps:
            state.stop_training = True
            return True
        return False

    monkeypatch.setattr(rtl, "_train_step", _train_step)
    rtl.run_training_loop(ctx)

    assert len(writer.logged) == target_steps
    for metrics, step in writer.logged:
        assert "train/kl" in metrics
        assert "eval/reward" in metrics
        assert metrics["train/tau"] == pytest.approx(weighting.tau)
        assert step in {0, 1}
    assert validation_calls == [(1, val_ctx), (2, val_ctx)]


def test_maybe_overwrite_controller_state_from_config_applies_recipe(monkeypatch, rtl):
    """Overwriting should reset tau/beta, denom, and broadcast the controller state."""

    calls = {}

    def _fake_broadcast(accel, weighting):
        calls["broadcast"] = (accel, weighting)
        return True

    sync_values = {}

    def _fake_sync(weighting):
        sync_values["scalars"] = (weighting.tau, weighting.beta)

    monkeypatch.setattr(rtl, "broadcast_controller_state", _fake_broadcast)
    monkeypatch.setattr(rtl, "_sync_controller_state", _fake_sync)

    class _Weighting:
        def __init__(self):
            self.tau = 0.1
            self.beta = 0.02
            self.train_grpo_objective = False
            self.normalization = SimpleNamespace(denom=1.0, len_norm_ref=False)

        @property
        def denom(self):
            return self.normalization.denom

        @denom.setter
        def denom(self, value):
            self.normalization.denom = value

    ctx = SimpleNamespace(
        training_args=SimpleNamespace(
            controller_overwrite_from_config=True,
            maxent_tau=0.3,
            beta=0.04,
        ),
        scoring=SimpleNamespace(weighting=_Weighting()),
        runtime=SimpleNamespace(accelerator=SimpleNamespace()),
    )

    rtl._maybe_overwrite_controller_state_from_config(ctx)

    assert ctx.scoring.weighting.tau == pytest.approx(0.3)
    assert ctx.scoring.weighting.beta == pytest.approx(0.04)
    assert ctx.scoring.weighting.denom == pytest.approx(0.34)
    assert sync_values["scalars"] == (pytest.approx(0.3), pytest.approx(0.04))
    assert calls["broadcast"][0] is ctx.runtime.accelerator
    assert calls["broadcast"][1] is ctx.scoring.weighting


def test_maybe_overwrite_controller_state_from_config_noop_without_flag(monkeypatch, rtl):
    """When flag is disabled, helper should not touch controller state."""

    def _fail(*_args, **_kwargs):
        raise AssertionError("should not be called")

    monkeypatch.setattr(rtl, "broadcast_controller_state", _fail)
    monkeypatch.setattr(rtl, "_sync_controller_state", _fail)

    class _Weighting:
        def __init__(self):
            self.tau = 0.1
            self.beta = 0.02
            self.train_grpo_objective = False
            self.normalization = SimpleNamespace(denom=0.5, len_norm_ref=False)

        @property
        def denom(self):
            return self.normalization.denom

        @denom.setter
        def denom(self, value):
            self.normalization.denom = value

    ctx = SimpleNamespace(
        training_args=SimpleNamespace(
            controller_overwrite_from_config=False,
            maxent_tau=0.9,
            beta=0.7,
        ),
        scoring=SimpleNamespace(weighting=_Weighting()),
        runtime=SimpleNamespace(accelerator=SimpleNamespace()),
    )

    rtl._maybe_overwrite_controller_state_from_config(ctx)

    assert ctx.scoring.weighting.tau == 0.1
    assert ctx.scoring.weighting.beta == 0.02
    assert ctx.scoring.weighting.denom == 0.5


def test_maybe_overwrite_controller_state_skips_when_resumed(monkeypatch, rtl):
    """Resumed controllers should not be clobbered by recipe overrides."""

    def _fail(*_args, **_kwargs):
        raise AssertionError("should not sync/broadcast when skipping overwrite")

    monkeypatch.setattr(rtl, "broadcast_controller_state", _fail)
    monkeypatch.setattr(rtl, "_sync_controller_state", _fail)

    class _Weighting:
        def __init__(self):
            self.tau = 0.2
            self.beta = 0.05
            self.train_grpo_objective = False
            self.normalization = SimpleNamespace(denom=0.25, len_norm_ref=False)

        @property
        def denom(self):
            return self.normalization.denom

        @denom.setter
        def denom(self, value):
            self.normalization.denom = value

    weighting = _Weighting()
    ctx = SimpleNamespace(
        training_args=SimpleNamespace(
            controller_overwrite_from_config=True,
            maxent_tau=0.8,
            beta=0.1,
        ),
        scoring=SimpleNamespace(weighting=weighting),
        runtime=SimpleNamespace(accelerator=SimpleNamespace()),
    )

    rtl._maybe_overwrite_controller_state_from_config(ctx, controller_resumed=True)

    assert weighting.tau == pytest.approx(0.2)
    assert weighting.beta == pytest.approx(0.05)
    assert weighting.denom == pytest.approx(0.25)


def test_apply_weighting_overrides_from_config_sets_fallback(rtl):
    weighting = SimpleNamespace(allow_empty_weight_fallback=False)
    ctx = SimpleNamespace(
        training_args=SimpleNamespace(maxent_allow_empty_weight_fallback=True),
        scoring=SimpleNamespace(weighting=weighting),
    )

    rtl._apply_weighting_overrides_from_config(ctx)

    assert weighting.allow_empty_weight_fallback is True
