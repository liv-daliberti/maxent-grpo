"""Unit tests for the training metrics helpers."""

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

import training.metrics as metrics_mod  # noqa: E402
from training.metrics import (  # noqa: E402
    accumulate_metrics,
    flush_metric_averages,
    log_local_step,
    _summarize_reward_stats,
    _summarize_weight_stats,
)


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
