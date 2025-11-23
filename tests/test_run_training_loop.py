"""Tests for the training loop helpers."""

from __future__ import annotations

from importlib import import_module, reload
import sys
from types import SimpleNamespace

import pytest

from tests.test_run_setup_reference import _load_run_setup


@pytest.fixture
def rtl(monkeypatch):
    """Load training.loop with torch/accelerate stubs applied."""
    _load_run_setup(monkeypatch)
    module = reload(import_module("training.loop"))
    return module


def test_collect_batch_stats_prefers_vllm_metadata(monkeypatch, rtl):
    """When vLLM metadata covers all sequences we should skip local rescoring."""
    pipeline = reload(import_module("training.pipeline"))

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
    metrics = reload(import_module("training.metrics"))

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
    assert "completions" in rows_logged["data"]
    table_kwargs = rows_logged["kwargs"]
    assert table_kwargs["columns"] == [
        "step",
        "prompt",
        "completion",
        "advantage",
        "reward/accuracy",
    ]
    assert len(table_kwargs["rows"]) == 2


def test_clear_stale_controller_state_removes_file(monkeypatch, rtl):
    """overwrite_output_dir should delete old controller snapshots before training."""
    state_mod = reload(import_module("training.state"))

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
    state_mod = reload(import_module("training.state"))
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
        reward_comp=SimpleNamespace(),
        weight_stats=SimpleNamespace(),
        ref_stats=SimpleNamespace(ref_logp_mean=0.0, avg_completion_tokens=2.0),
        length_stats=SimpleNamespace(),
        num_completion_tokens=2.0,
        total_input_tokens=5.0,
        scores=SimpleNamespace(),
    )
    loss_outputs = SimpleNamespace(loss=object(), kl_loss_scalar=0.25)
    recorded = {}

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
    assert recorded["tau_call"] == (weighting, prepared.weight_stats, state.global_step)
    assert recorded["saved"] == (ctx.controller.state_path, weighting)
    assert recorded["validated"] == state.global_step
    assert recorded["checkpoint"] == state.global_step
    assert recorded["checked"] == state.global_step
    assert state.num_input_tokens_seen == pytest.approx(prepared.total_input_tokens)
