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

Unit tests for training.loop helpers with lightweight stubs.
"""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import maxent_grpo.training.loop as loop
from maxent_grpo.training.types import StepBatchInfo, TrainingLoopState, StepResources


def test_train_step_runs_deepspeed_optimizer(monkeypatch):
    """Ensure _train_step executes the DeepSpeed branch and calls optimizer_step."""
    # Minimal torch stub
    stub_torch = SimpleNamespace(Tensor=object)
    monkeypatch.setattr(loop, "torch", stub_torch)
    monkeypatch.setattr(loop, "Tensor", object)
    # Stub helpers
    monkeypatch.setattr(
        loop,
        "detect_deepspeed_state",
        lambda _accel: SimpleNamespace(use_deepspeed=True, zero_stage=2),
    )
    monkeypatch.setattr(
        loop, "require_accumulation_context", lambda *a, **k: nullcontext()
    )
    monkeypatch.setattr(loop, "_scheduled_learning_rate", lambda *a, **k: 0.001)
    monkeypatch.setattr(
        loop, "build_loss_inputs", lambda *a, **k: (None, None, None, None)
    )
    monkeypatch.setattr(
        loop,
        "evaluate_losses",
        lambda *_args, **_kwargs: (
            SimpleNamespace(loss=0.0, kl_loss_scalar=None),
            SimpleNamespace(),
        ),
    )
    optimizer_called = {}

    def _fake_opt(ctx, state, lr):
        optimizer_called["lr"] = lr
        state.global_step += 1
        return 3.14

    monkeypatch.setattr(loop, "_optimizer_step", _fake_opt)
    monkeypatch.setattr(loop, "log_local_step", lambda *a, **k: None)
    monkeypatch.setattr(loop, "log_training_step", lambda *a, **k: None)
    monkeypatch.setattr(loop, "maybe_update_beta", lambda *a, **k: None)
    monkeypatch.setattr(loop, "maybe_update_tau", lambda *a, **k: None)
    monkeypatch.setattr(loop, "save_controller_state", lambda *a, **k: None)
    monkeypatch.setattr(loop, "_maybe_validate", lambda *a, **k: None)
    monkeypatch.setattr(loop, "maybe_checkpoint", lambda *a, **k: None)
    monkeypatch.setattr(loop, "check_stop_condition", lambda *a, **k: None)

    class _Accel:
        is_main_process = True

        def backward(self, *_args, **_kwargs):
            pass

    ctx = SimpleNamespace(
        optimization=SimpleNamespace(
            schedule=SimpleNamespace(
                grad_accum_steps=1, total_training_steps=4, steps_per_epoch=1
            ),
            handles=SimpleNamespace(),
        ),
        runtime=SimpleNamespace(
            accelerator=_Accel(), model=SimpleNamespace(), tokenizer=None
        ),
        generation=SimpleNamespace(generation_stats={}),
        scoring=SimpleNamespace(
            clipping=SimpleNamespace(),
            weighting=SimpleNamespace(),
        ),
        controller=SimpleNamespace(state_path=None),
        evaluation=SimpleNamespace(enabled=False, every_n_steps=None),
        logging=SimpleNamespace(
            save_strategy="steps",
            save_steps=1,
            save_checkpoint=lambda *_a, **_k: None,
            log_metrics=lambda *_a, **_k: None,
        ),
    )
    prepared = SimpleNamespace(
        total_input_tokens=1,
        grouped_completions=[],
        weight_stats=SimpleNamespace(),
        scores=None,
        ref_stats=None,
        seed_heatmap=None,
    )
    monkeypatch.setattr(loop, "prepare_training_batch", lambda *a, **k: prepared)
    resources = StepResources(generator=lambda *_a, **_k: None, validation_ctx=None)
    state = TrainingLoopState()
    step_info = StepBatchInfo(epoch=0, step_in_epoch=0, batch=None)

    stop = loop._train_step(ctx, state, step_info, resources)
    assert stop is False
    assert optimizer_called["lr"] == 0.001
    assert state.global_step == 1


def test_run_epoch_stops_when_train_step_requests(monkeypatch):
    """_run_epoch should stop early when _train_step signals True."""
    calls = {"train_step": 0}
    monkeypatch.setattr(
        loop,
        "_train_step",
        lambda *a, **k: (calls.update(train_step=calls["train_step"] + 1) or True),
    )
    ctx = SimpleNamespace(
        runtime=SimpleNamespace(
            train_loader=[1, 2, 3],
            train_sampler=SimpleNamespace(set_epoch=lambda *_a, **_k: None),
        )
    )
    state = TrainingLoopState()
    resources = StepResources(generator=lambda *_a, **_k: None, validation_ctx=None)
    assert loop._run_epoch(ctx, state, epoch=5, resources=resources) is True
    assert calls["train_step"] == 1


def test_run_training_loop_breaks_after_epoch(monkeypatch, caplog):
    """run_training_loop should break on _run_epoch True and close wandb if present."""
    monkeypatch.setattr(loop, "_maybe_patch_zero_no_sync", lambda *_a, **_k: None)
    monkeypatch.setattr(loop, "_run_epoch", lambda *_a, **_k: True)
    monkeypatch.setattr(loop, "replace", lambda obj, **kwargs: obj)
    # Keep logging quiet
    caplog.set_level("INFO")

    class _Wandb:
        def __init__(self):
            self.closed = False

        def finish(self):
            self.closed = True

    wandb = _Wandb()
    runtime = SimpleNamespace(
        accelerator=SimpleNamespace(
            is_main_process=True, model=None, tokenizer=None, device="cpu"
        ),
        model=SimpleNamespace(),
        tokenizer=None,
        device="cpu",
        train_loader=[],
    )
    ctx = SimpleNamespace(
        runtime=runtime,
        generation=SimpleNamespace(
            generation_stats={
                "vllm_retry_rounds": 0,
                "vllm_backfilled_prompts": 0,
                "vllm_failed_prompts": 0,
                "dropped_prompts": 0,
            },
            max_prompt_len=1,
            max_completion_len=1,
            gen_temperature=1.0,
            gen_top_p=1.0,
            use_vllm=False,
            vllm=None,
            penalty=SimpleNamespace(),
        ),
        evaluation=SimpleNamespace(enabled=False, every_n_steps=None),
        reward=None,
        logging=SimpleNamespace(wandb_run=wandb),
        controller=SimpleNamespace(state_path=None, resume_from=None),
        optimization=SimpleNamespace(
            schedule=SimpleNamespace(
                num_epochs=2,
                grad_accum_steps=1,
                steps_per_epoch=1,
                total_training_steps=1,
            ),
            handles=SimpleNamespace(
                optimizer=SimpleNamespace(zero_grad=lambda set_to_none=True: None)
            ),
        ),
        scoring=SimpleNamespace(weighting=SimpleNamespace()),
    )
    loop.run_training_loop(ctx)
    assert wandb.closed is True
