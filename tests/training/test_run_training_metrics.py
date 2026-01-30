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

Unit tests for the training metrics helpers.
"""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from contextlib import nullcontext

import pytest

_TORCH_STUB = types.ModuleType("torch")
_TORCH_STUB.Tensor = type("Tensor", (object,), {})
_TORCH_STUB.device = type("device", (object,), {})
_TORCH_STUB.optim = types.SimpleNamespace(Optimizer=type("Optimizer", (object,), {}))
_TORCH_STUB.__spec__ = types.SimpleNamespace()
sys.modules["torch"] = _TORCH_STUB
torch_utils = types.ModuleType("torch.utils")
sys.modules["torch.utils"] = torch_utils
torch_utils_data = types.ModuleType("torch.utils.data")
torch_utils_data.DataLoader = type("DataLoader", (object,), {})
torch_utils_data.Sampler = type("Sampler", (object,), {})
sys.modules["torch.utils.data"] = torch_utils_data
accelerate_stub = types.ModuleType("accelerate")
accelerate_stub.Accelerator = type("Accelerator", (object,), {})
sys.modules["accelerate"] = accelerate_stub
torch_nn = types.ModuleType("torch.nn")
torch_nn.functional = types.ModuleType("torch.nn.functional")
torch_nn.functional.log_softmax = lambda *args, **kwargs: None
sys.modules["torch.nn"] = torch_nn
sys.modules["torch.nn.functional"] = torch_nn.functional
transformers_stub = types.ModuleType("transformers")
transformers_stub.PreTrainedModel = type("PreTrainedModel", (object,), {})
transformers_stub.PreTrainedTokenizer = type("PreTrainedTokenizer", (object,), {})
sys.modules["transformers"] = transformers_stub
trl_stub = types.ModuleType("trl")
trl_stub.ScriptArguments = type("ScriptArguments", (object,), {})
trl_stub.GRPOConfig = type("GRPOConfig", (object,), {})
sys.modules["trl"] = trl_stub
torch_optim = types.ModuleType("torch.optim")
torch_optim.Optimizer = type("Optimizer", (object,), {})
sys.modules["torch.optim"] = torch_optim

import maxent_grpo.training.metrics as metrics_mod  # noqa: E402
from maxent_grpo.training.metrics import (  # noqa: E402
    accumulate_metrics,
    flush_metric_averages,
    log_local_step,
    _summarize_reward_stats,
    _summarize_weight_stats,
)
from maxent_grpo.training.types import LoggingHandles  # noqa: E402


def _accel():
    return SimpleNamespace(num_processes=1, gather_object=None, is_main_process=True)


class _ScriptedGatherAccel:
    def __init__(self, script):
        self.num_processes = 2
        self.is_main_process = True
        self._script = list(script)

    def gather_object(self, payload):
        extra = self._script.pop(0) if self._script else []
        return [payload] + extra


def test_summarize_reward_stats_computes_means():
    reward_comp = SimpleNamespace(
        total_utils=[1.0, 2.0, 3.0],
        advantage_samples=[0.0, 0.1, -0.1],
        per_reward_values={"reward_0": [1.0, 2.0, 3.0]},
        advantage=SimpleNamespace(grouped=[[0.0, 0.0], [0.5, -0.5]]),
        q_grouped=[[0.6, 0.4], [0.25, 0.75]],
    )
    stats = _summarize_reward_stats(_accel(), reward_comp)
    assert pytest.approx(stats.reward_mean) == 2.0
    assert pytest.approx(stats.frac_zero_std) == 0.5
    assert "reward_0" in stats.per_reward


def test_summarize_weight_stats_aggregates_entropy():
    weight_stats = SimpleNamespace(
        weights_grouped=[[0.2, 0.8], [1.0]],
        flat_weights=[0.2, 0.8, 1.0],
        weight_entropy=0.5,
        weight_entropy_min=0.1,
        weight_entropy_max=0.9,
        advantage_entropy=[0.0, 0.5],
    )
    view = _summarize_weight_stats(_accel(), weight_stats)
    assert view.entropy_min == 0.1
    assert view.entropy_max == 0.9
    assert pytest.approx(view.advantage_entropy_mean) == 0.25


def test_accumulate_and_flush_metrics():
    state = SimpleNamespace(
        metric_sums={},
        metric_counts={},
    )
    accumulate_metrics(
        state,
        {
            "train/loss": 2.0,
            "train/global_step": 5,
            "train/epoch": 1.0,
            "train/non_numeric": "n/a",
        },
    )
    assert state.metric_sums["train/loss"] == 2.0
    averages = flush_metric_averages(state)
    assert averages["train/loss"] == 2.0
    assert state.metric_sums == {}
    assert state.metric_counts == {}


def test_log_local_step_uses_payload(monkeypatch):
    ctx = SimpleNamespace(
        runtime=SimpleNamespace(accelerator=_accel()),
        scoring=SimpleNamespace(weighting=None, clipping=None),
        optimization=SimpleNamespace(schedule=None),
        generation=SimpleNamespace(use_vllm=False, generation_stats={}),
        logging=SimpleNamespace(
            log_metrics=lambda metrics, step: logs.append((metrics, step)),
            step_logger=lambda *_args, **_kwargs: nullcontext(),
        ),
    )
    logs: list = []
    emitted: list = []
    state = SimpleNamespace(
        global_step=3,
        num_input_tokens_seen=10.0,
        metric_sums={},
        metric_counts={},
    )
    prepared = SimpleNamespace(
        reward_comp=None,
        weight_stats=None,
        length_stats=None,
        ref_stats=SimpleNamespace(avg_completion_tokens=1.0, ref_logp_mean=0.0),
        num_completion_tokens=2.0,
    )
    artifacts = SimpleNamespace(
        loss_outputs=SimpleNamespace(),
        diagnostics=SimpleNamespace(),
        grad_norm_scalar=None,
        epoch_progress=0.5,
    )

    monkeypatch.setattr(
        metrics_mod,
        "_build_metrics_payload",
        lambda *_args, **_kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        metrics_mod,
        "build_training_metrics_dict",
        lambda payload, step: {"train/loss": 1.0, "train/global_step": step},
    )

    def _fake_emit(_ctx, metrics, step, **_kwargs):
        emitted.append((metrics, step))
        return metrics

    monkeypatch.setattr(
        metrics_mod,
        "_emit_metrics",
        _fake_emit,
    )

    log_local_step(ctx, state, prepared, artifacts, current_lr=1e-4)
    assert emitted[0][0]["train/loss"] == 1.0
    assert state.metric_sums["train/loss"] == 1.0


def test_log_local_step_emits_debug_metrics(monkeypatch, caplog):
    ctx = SimpleNamespace(
        runtime=SimpleNamespace(accelerator=_accel()),
        scoring=SimpleNamespace(weighting=None, clipping=None),
        optimization=SimpleNamespace(schedule=None),
        generation=SimpleNamespace(use_vllm=False, generation_stats={}),
        logging=SimpleNamespace(
            log_metrics=lambda *_a, **_k: None,
            step_logger=lambda *_a, **_k: nullcontext(),
        ),
        training_args=SimpleNamespace(logging_strategy="no"),
    )
    state = SimpleNamespace(
        global_step=7,
        num_input_tokens_seen=4.0,
        metric_sums={},
        metric_counts={},
    )
    prepared = SimpleNamespace(
        reward_comp=None,
        weight_stats=None,
        length_stats=None,
        ref_stats=SimpleNamespace(avg_completion_tokens=1.0, ref_logp_mean=0.0),
        num_completion_tokens=1.0,
    )
    artifacts = SimpleNamespace(
        loss_outputs=SimpleNamespace(),
        diagnostics=SimpleNamespace(),
        grad_norm_scalar=None,
        epoch_progress=0.25,
    )
    monkeypatch.setattr(
        metrics_mod,
        "_build_metrics_payload",
        lambda *_a, **_k: SimpleNamespace(),
    )
    metric_dict = {
        "train/loss": 0.25,
        "train/reward": 0.5,
        "train/reward_std": 0.1,
        "train/q_entropy_mean": 0.0,
        "train/weight_entropy": 0.0,
        "train/kl": 0.01,
        "train/tau": 0.4,
        "train/beta": 0.03,
        "train/rewards/pure_accuracy/mean": 0.6,
        "train/global_step": state.global_step,
    }
    monkeypatch.setattr(
        metrics_mod,
        "build_training_metrics_dict",
        lambda *_a, **_k: dict(metric_dict),
    )
    with caplog.at_level("DEBUG", logger=metrics_mod.LOG.name):
        log_local_step(ctx, state, prepared, artifacts, current_lr=1e-5)
    debug_lines = [
        record.message for record in caplog.records if "debug metrics step" in record.message
    ]
    assert debug_lines, "expected debug metrics log"
    assert "tau=0.400000" in debug_lines[0]


def test_log_sample_table_emits_rows(monkeypatch):
    logged = []

    class _FakeRun:
        def log(self, data, step):
            logged.append((data, step))

    class _FakeWandb:
        def __init__(self):
            self.tables = []

        def Table(self, columns, rows):
            self.tables.append((columns, rows))
            return {"columns": columns, "rows": rows}

    fake_wandb = _FakeWandb()
    monkeypatch.setattr(
        metrics_mod,
        "_get_wandb",
        lambda: fake_wandb,
    )
    ctx = SimpleNamespace(
        logging=SimpleNamespace(wandb_run=_FakeRun()),
        runtime=SimpleNamespace(accelerator=SimpleNamespace(is_main_process=True)),
    )
    state = SimpleNamespace(global_step=42)
    prepared = SimpleNamespace(
        reward_comp=SimpleNamespace(
            pairs=SimpleNamespace(
                prompts=["p1", "p2"],
                completions=["c1", "c2"],
            ),
            per_reward_values={"reward": [0.5, 0.25]},
            advantage_samples=[0.1, -0.05],
        )
    )
    metrics_mod._log_sample_table(ctx, state, prepared)
    assert logged and logged[0][1] == 42
    assert fake_wandb.tables and fake_wandb.tables[0][0] == [
        "step",
        "prompt",
        "completion",
        "advantage",
        "reward/reward",
    ]


def _controller_payload():
    weighting = SimpleNamespace(
        tau=0.3,
        beta=0.15,
        denom=0.45,
        q_temperature=1.0,
        q_epsilon=1e-6,
        tau_lr=0.0,
        tau_min=0.0,
        tau_max=1.0,
        tau_warmup_steps=0,
        tau_target_entropy=0.6,
        kl_target=0.4,
        kl_horizon=10,
        kl_ctl_step_size=0.5,
        len_norm_ref=True,
        train_grpo_objective=False,
    )
    payload = metrics_mod.TrainingMetricsPayload(
        reward_stats=metrics_mod.RewardLoggingView(
            reward_mean=0.5,
            reward_std=0.1,
            frac_zero_std=0.0,
            advantage_mean=0.0,
            advantage_std=0.0,
            advantage_count=1,
            per_reward={},
            q_entropy_mean=0.0,
            q_entropy_std=0.0,
            q_entropy_min=0.0,
            q_entropy_max=0.0,
        ),
        weight_stats=metrics_mod.WeightLoggingView(
            entropy=0.2,
            entropy_min=0.1,
            entropy_max=0.3,
            advantage_entropy_mean=0.0,
            advantage_entropy_std=0.0,
        ),
        loss_outputs=SimpleNamespace(
            total_loss_scalar=1.0,
            kl_loss_scalar=0.45,
            policy_loss_scalar=0.8,
            weighted_kl_loss_scalar=0.45,
            clip_loss_scalar=None,
            scalars=SimpleNamespace(kl_loss=0.45),
        ),
        diagnostics=metrics_mod.BatchDiagnostics(
            kl_value=0.5,
            clip_ratio=0.0,
            clip_ratio_low_mean=0.0,
            clip_ratio_low_min=0.0,
            clip_ratio_high_mean=0.0,
            clip_ratio_high_max=0.0,
            clip_ratio_region_mean=0.0,
            kl_per_token_by_len_bucket={},
            kl_token_count_by_len_bucket={},
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
        config=metrics_mod.LoggingConfigView(
            weighting=weighting,
            clipping=SimpleNamespace(),
            schedule=SimpleNamespace(),
        ),
        scalars=metrics_mod.TrainingScalarStats(
            ref_logp_mean=0.0,
            tokens=metrics_mod.TokenUsageStats(
                avg_completion_tokens=2.0,
                num_completion_tokens=4.0,
                num_input_tokens=6.0,
            ),
            current_lr=1e-4,
            grad_norm_scalar=None,
            epoch_progress=0.1,
            vllm_latency_ms=None,
        ),
    )
    return weighting, payload


def test_build_training_metrics_dict_logs_controller_errors():
    _, payload = _controller_payload()
    metrics = metrics_mod.build_training_metrics_dict(payload, global_step=8)
    assert metrics["train/kl_error_to_target"] == pytest.approx(0.5 - 0.4)
    assert metrics["train/weight_entropy_error"] == pytest.approx(0.2 - 0.6)
    assert "train/tau_loss" in metrics and metrics["train/tau_loss"] > 0.0


def test_log_training_metrics_emits_controller_errors(monkeypatch):
    weighting, payload = _controller_payload()
    payload.config.weighting = weighting
    writer_logs = []

    class _Writer:
        def __init__(self):
            self.logged = []

        def log(self, metrics, step):
            self.logged.append((metrics, step))
            writer_logs.append(metrics)

        def flush(self):
            return None

    handles = LoggingHandles(
        metric_writer=_Writer(),
        save_checkpoint=lambda *_a, **_k: None,
        save_strategy="no",
        save_steps=0,
        wandb_run=None,
    )
    metrics = metrics_mod.log_training_metrics(handles, global_step=3, payload=payload)
    assert writer_logs, "metric writer should receive payload"
    logged_metrics = writer_logs[-1]
    assert "train/tau_loss" in logged_metrics
    assert "train/weight_entropy_error" in logged_metrics
    assert "train/kl_error_to_target" in logged_metrics
    assert metrics is logged_metrics


def test_gather_dict_of_lists_single_rank():
    accel = _accel()
    result = metrics_mod._gather_dict_of_lists_for_metrics(
        accel,
        {"reward": [1.0, 2.0]},
    )
    assert result == {"reward": [1.0, 2.0]}


def test_gather_dict_of_lists_multi_rank_merges_entries():
    accel = _ScriptedGatherAccel(
        [[{"reward": [3.0], "other": [5.0, 6.0]}]],
    )
    result = metrics_mod._gather_dict_of_lists_for_metrics(
        accel,
        {"reward": [1.0], "other": [2.0]},
    )
    assert result["reward"] == [1.0, 3.0]
    assert result["other"] == [2.0, 5.0, 6.0]


def test_fraction_zero_std_groups_single_rank():
    frac = metrics_mod._fraction_zero_std_groups(
        _accel(),
        [[0.0, 0.0], [0.1, -0.1]],
    )
    assert frac == pytest.approx(0.5)


def test_fraction_zero_std_groups_multi_rank():
    accel = _ScriptedGatherAccel([[[1.0]], [[2.0]]])
    frac = metrics_mod._fraction_zero_std_groups(
        accel,
        [[0.0], [0.5]],
    )
    assert frac == pytest.approx(0.5)


def test_mean_std_handles_empty_sequence():
    mean, std = metrics_mod._mean_std([])
    assert mean == 0.0 and std == 0.0


def test_gather_helpers_skip_when_gather_missing():
    accel = SimpleNamespace(num_processes=2, gather_object=None)
    assert metrics_mod._gather_list_for_metrics(accel, [1.2, 3]) == [1.2, 3.0]
    gathered = metrics_mod._gather_dict_of_lists_for_metrics(accel, {"reward": [1, 2]})
    assert gathered == {"reward": [1.0, 2.0]}


def test_fraction_zero_std_groups_skips_empty_groups():
    frac = metrics_mod._fraction_zero_std_groups(
        _accel(),
        [[], [0.0, 0.0], [0.1, -0.1]],
    )
    assert frac == pytest.approx(0.5)


def test_epoch_from_global_step_selects_schedule_fields():
    assert metrics_mod._epoch_from_global_step(
        SimpleNamespace(steps_per_epoch=5), 10
    ) == pytest.approx(2.0)
    assert metrics_mod._epoch_from_global_step(
        SimpleNamespace(steps_per_epoch=0, num_generations=4),
        3,
    ) == pytest.approx(0.75)
    assert metrics_mod._epoch_from_global_step(
        SimpleNamespace(
            steps_per_epoch=None,
            num_generations=0,
            total_training_steps=20,
            num_epochs=5,
        ),
        4,
    ) == pytest.approx(1.0)


def test_emit_metrics_handles_type_error_and_unorderable_keys(monkeypatch):
    log_calls = []
    accel_calls = []

    class _Accel:
        def __init__(self):
            self.is_main_process = True

        def log(self, metrics):
            accel_calls.append(metrics)

    ctx = SimpleNamespace(
        runtime=SimpleNamespace(accelerator=_Accel()),
        logging=SimpleNamespace(
            log_metrics=lambda metrics, step: log_calls.append((metrics, step))
        ),
    )
    metrics = {1: "a", "b": 2}
    result = metrics_mod._emit_metrics(
        ctx, metrics, global_step=7, log_to_wandb=True, tag="Test"
    )
    assert log_calls == [(metrics, 7)]
    assert accel_calls == [metrics]
    assert result is metrics


def test_log_local_step_noop_when_not_main(monkeypatch):
    payload_called = {"called": False}
    monkeypatch.setattr(
        metrics_mod,
        "_build_metrics_payload",
        lambda *_a, **_k: payload_called.__setitem__("called", True),
    )
    state = SimpleNamespace(
        global_step=0, num_input_tokens_seen=0, metric_sums={}, metric_counts={}
    )
    ctx = SimpleNamespace(
        runtime=SimpleNamespace(accelerator=SimpleNamespace(is_main_process=False))
    )
    metrics_mod.log_local_step(
        ctx, state, prepared=None, log_artifacts=None, current_lr=0.0
    )
    assert payload_called["called"] is False
    assert state.metric_sums == {}


def test_log_sample_table_uses_fallback_wandb(monkeypatch):
    monkeypatch.setattr(metrics_mod, "_get_wandb", lambda: None)
    monkeypatch.setitem(sys.modules, "wandb", None)
    run_logs = []
    ctx = SimpleNamespace(
        logging=SimpleNamespace(
            wandb_run=SimpleNamespace(log=lambda *_a, **_k: run_logs.append("logged"))
        ),
        runtime=SimpleNamespace(accelerator=SimpleNamespace(is_main_process=True)),
    )
    state = SimpleNamespace(global_step=0)
    prepared = SimpleNamespace(
        reward_comp=SimpleNamespace(pairs=SimpleNamespace(prompts=[], completions=[]))
    )
    metrics_mod._log_sample_table(ctx, state, prepared)
    assert run_logs == []


def test_log_sample_table_aborts_when_max_rows_zero(monkeypatch):
    monkeypatch.setattr(metrics_mod, "_WANDB_SAMPLE_ROWS", 0)
    monkeypatch.setattr(
        metrics_mod,
        "_get_wandb",
        lambda: SimpleNamespace(Table=lambda *args, **kwargs: {}),
    )
    run_logs = []
    ctx = SimpleNamespace(
        logging=SimpleNamespace(
            wandb_run=SimpleNamespace(log=lambda *_a, **_k: run_logs.append("logged"))
        ),
        runtime=SimpleNamespace(accelerator=SimpleNamespace(is_main_process=True)),
    )
    prepared = SimpleNamespace(
        reward_comp=SimpleNamespace(
            pairs=SimpleNamespace(prompts=["p"], completions=["c"]),
            per_reward_values={},
            advantage_samples=[],
        )
    )
    metrics_mod._log_sample_table(ctx, SimpleNamespace(global_step=1), prepared)
    assert run_logs == []


def test_log_sample_table_skips_empty_rows(monkeypatch):
    monkeypatch.setattr(
        metrics_mod,
        "_get_wandb",
        lambda: SimpleNamespace(Table=lambda *args, **kwargs: {}),
    )
    monkeypatch.setattr(
        metrics_mod, "_build_sample_table", lambda *_a, **_k: (["step"], [])
    )
    run_logs = []
    ctx = SimpleNamespace(
        logging=SimpleNamespace(
            wandb_run=SimpleNamespace(log=lambda *_a, **_k: run_logs.append("logged"))
        ),
        runtime=SimpleNamespace(accelerator=SimpleNamespace(is_main_process=True)),
    )
    prepared = SimpleNamespace(
        reward_comp=SimpleNamespace(
            pairs=SimpleNamespace(prompts=["p"], completions=["c"]),
            per_reward_values={},
            advantage_samples=[],
        )
    )
    metrics_mod._log_sample_table(ctx, SimpleNamespace(global_step=2), prepared)
    assert run_logs == []


def test_log_sample_table_swallows_wandb_errors(monkeypatch):
    class _FakeWandb:
        def __init__(self):
            self.tables = []

        def Table(self, columns=None, rows=None, **_kwargs):
            self.tables.append((columns, rows))
            return {"columns": columns, "rows": rows}

    fake_wandb = _FakeWandb()
    monkeypatch.setattr(metrics_mod, "_get_wandb", lambda: fake_wandb)

    class _Run:
        def __init__(self):
            self.calls = 0

        def log(self, *_args, **_kwargs):
            self.calls += 1
            raise metrics_mod.WandbError("boom")

    run = _Run()
    ctx = SimpleNamespace(
        logging=SimpleNamespace(wandb_run=run),
        runtime=SimpleNamespace(accelerator=SimpleNamespace(is_main_process=True)),
    )
    prepared = SimpleNamespace(
        reward_comp=SimpleNamespace(
            pairs=SimpleNamespace(prompts=["p1"], completions=["c1"]),
            per_reward_values={"reward": [0.1]},
            advantage_samples=[0.2],
        )
    )
    metrics_mod._log_sample_table(ctx, SimpleNamespace(global_step=3), prepared)
    assert run.calls == 1
    assert fake_wandb.tables


def test_log_sample_table_uses_fallback_table(monkeypatch):
    monkeypatch.setattr(metrics_mod, "_get_wandb", lambda: None)
    monkeypatch.delitem(sys.modules, "wandb", raising=False)
    logged = {}

    class _Run:
        def log(self, payload, step):
            logged["payload"] = payload
            logged["step"] = step

    ctx = SimpleNamespace(
        logging=SimpleNamespace(wandb_run=_Run()),
        runtime=SimpleNamespace(accelerator=SimpleNamespace(is_main_process=True)),
    )
    prepared = SimpleNamespace(
        reward_comp=SimpleNamespace(
            pairs=SimpleNamespace(prompts=["p"], completions=["c"]),
            per_reward_values={"r": [0.1]},
            advantage_samples=[0.2],
        )
    )
    monkeypatch.setattr(
        metrics_mod, "_build_sample_table", lambda *_a, **_k: (["step"], [["row"]])
    )
    metrics_mod._log_sample_table(ctx, SimpleNamespace(global_step=7), prepared)
    assert logged["payload"]["completions"]["rows"] == [["row"]]
    assert logged["step"] == 7


def test_log_training_step_builds_payload_when_no_averages(monkeypatch):
    payload_obj = SimpleNamespace(tag="payload")
    build_calls = []
    dict_calls = []
    emitted = []
    monkeypatch.setattr(
        metrics_mod,
        "_build_metrics_payload",
        lambda *args, **kwargs: build_calls.append(True) or payload_obj,
    )
    monkeypatch.setattr(
        metrics_mod,
        "build_training_metrics_dict",
        lambda payload, step: dict_calls.append((payload, step))
        or {"train/loss": 0.0, "train/global_step": float(step)},
    )
    monkeypatch.setattr(
        metrics_mod,
        "_emit_metrics",
        lambda ctx, metrics, step, **kwargs: emitted.append((metrics, step)) or metrics,
    )
    monkeypatch.setattr(
        metrics_mod,
        "_log_sample_table",
        lambda *args, **kwargs: emitted.append(("table", args[1].global_step)),
    )
    ctx = SimpleNamespace(
        runtime=SimpleNamespace(accelerator=SimpleNamespace(is_main_process=True)),
        logging=SimpleNamespace(step_logger=lambda *_a, **_k: nullcontext()),
        scoring=SimpleNamespace(
            weighting=SimpleNamespace(tau=0.1, beta=0.2), clipping=None
        ),
        optimization=SimpleNamespace(schedule=None),
    )
    state = SimpleNamespace(global_step=9, metric_sums={}, metric_counts={})
    prepared = SimpleNamespace()
    artifacts = SimpleNamespace(
        loss_outputs=SimpleNamespace(total_loss_scalar=0.0),
        diagnostics=None,
        grad_norm_scalar=None,
        epoch_progress=0.0,
    )
    metrics_mod.log_training_step(ctx, state, prepared, artifacts, current_lr=0.0)
    assert build_calls and dict_calls
    assert emitted[0][0]["train/loss"] == 0.0
    assert ("table", 9) in emitted
