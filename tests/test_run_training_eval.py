"""Tests for the validation helper."""

from __future__ import annotations

from importlib import import_module, reload
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import pytest

from tests.test_run_setup_reference import _load_run_setup


class _StubMetricWriter:
    def __init__(self):
        self.logged: List[Tuple[Dict[str, Any], int]] = []

    def log(self, metrics: Dict[str, Any], step: int) -> None:
        self.logged.append((metrics, step))

    def flush(self) -> None:
        return None


@pytest.fixture
def rte(monkeypatch):
    """Load training.eval with dependency stubs."""
    _load_run_setup(monkeypatch)
    eval_mod = reload(import_module("training.eval"))
    types_mod = reload(import_module("training.types"))
    return eval_mod, types_mod


def test_run_validation_step_uses_per_prompt_counts(rte):
    run_training_eval, rtt = rte
    calls = []

    def fake_generator(prompts, num_samples, per_prompt_counts=None):
        calls.append((prompts, num_samples, per_prompt_counts))
        return [[f"{prompt}-0"] for prompt in prompts], None

    reward_spec = rtt.RewardSpec(
        reward_funcs=[lambda comps, answers: [0.0 for _ in comps]],
        reward_weights=[1.0],
    )
    evaluation_settings = rtt.EvaluationSettings(
        enabled=True,
        rows=[{"prompt": "p1", "answer": "a1"}, {"prompt": "p2", "answer": "a2"}],
        batch_size=2,
        every_n_steps=None,
    )
    logging_handles = rtt.LoggingHandles(
        metric_writer=_StubMetricWriter(),
        save_checkpoint=lambda *_a, **_k: None,
        save_strategy="no",
        save_steps=0,
        wandb_run=None,
    )

    class _Model:
        def __init__(self) -> None:
            self.training = True

        def eval(self) -> None:
            self.training = False

        def train(self) -> None:
            self.training = True

    accelerator = type(
        "Accel",
        (),
        {
            "is_main_process": True,
            "num_processes": 1,
            "process_index": 0,
        },
    )()
    ctx = rtt.ValidationContext(
        evaluation=evaluation_settings,
        accelerator=accelerator,
        model=_Model(),
        reward=reward_spec,
        generator=fake_generator,
        logging=logging_handles,
    )
    run_training_eval.run_validation_step(1, ctx)
    assert calls, "generator was not called"
    prompts, num_samples, per_prompt_counts = calls[0]
    assert prompts == ["p1", "p2"]
    assert num_samples == 1
    assert per_prompt_counts == [1, 1]


def test_run_validation_step_synchronizes_all_ranks(rte):
    run_training_eval, rtt = rte

    def fake_generator(prompts, num_samples, per_prompt_counts=None):
        return [[f"{prompt}-0"] for prompt in prompts], None

    reward_spec = rtt.RewardSpec(
        reward_funcs=[lambda comps, answers: [0.0 for _ in comps]],
        reward_weights=[1.0],
    )
    evaluation_settings = rtt.EvaluationSettings(
        enabled=True,
        rows=[{"prompt": "p1", "answer": "a1"}],
        batch_size=1,
        every_n_steps=None,
    )
    logging_handles = rtt.LoggingHandles(
        metric_writer=_StubMetricWriter(),
        save_checkpoint=lambda *_a, **_k: None,
        save_strategy="no",
        save_steps=0,
        wandb_run=None,
    )
    waits = []

    class _Model:
        def __init__(self) -> None:
            self.training = True

        def eval(self) -> None:
            self.training = False

        def train(self) -> None:
            self.training = True

    class _Accel:
        def __init__(self, is_main: bool) -> None:
            self.is_main_process = is_main
            self.num_processes = 2
            self.process_index = 0 if is_main else 1

        def wait_for_everyone(self) -> None:
            waits.append(self.is_main_process)

    for is_main in (True, False):
        waits.clear()
        model = _Model()
        accelerator = _Accel(is_main)
        ctx = rtt.ValidationContext(
            evaluation=evaluation_settings,
            accelerator=accelerator,
            model=model,
            reward=reward_spec,
            generator=fake_generator,
            logging=logging_handles,
        )
        run_training_eval.run_validation_step(2, ctx)
        assert waits == [is_main, is_main], "all ranks must wait before and after eval"
        assert model.training is True


def test_run_validation_step_aggregates_rewards_across_ranks(rte):
    run_training_eval, rtt = rte
    calls = []

    def fake_generator(prompts, num_samples, per_prompt_counts=None):
        calls.append(list(prompts))
        return [[f"{prompt}-0"] for prompt in prompts], None

    reward_spec = rtt.RewardSpec(
        reward_funcs=[lambda comps, answers: [1.0 for _ in comps]],
        reward_weights=[1.0],
    )
    evaluation_settings = rtt.EvaluationSettings(
        enabled=True,
        rows=[{"prompt": "p1", "answer": "a1"}, {"prompt": "p2", "answer": "a2"}],
        batch_size=2,
        every_n_steps=None,
    )
    writer = _StubMetricWriter()

    logging_handles = rtt.LoggingHandles(
        metric_writer=writer,
        save_checkpoint=lambda *_a, **_k: None,
        save_strategy="no",
        save_steps=0,
        wandb_run=None,
    )

    class _Model:
        def __init__(self) -> None:
            self.training = True

        def eval(self) -> None:
            self.training = False

        def train(self) -> None:
            self.training = True

    class _Accel:
        def __init__(self) -> None:
            self.is_main_process = True
            self.num_processes = 2
            self.process_index = 0

        def wait_for_everyone(self) -> None:
            return None

        def gather_object(self, obj):
            # Simulate second rank contributing 3 samples with total reward 3.0.
            return [obj, (3.0, 3.0)]

    ctx = rtt.ValidationContext(
        evaluation=evaluation_settings,
        accelerator=_Accel(),
        model=_Model(),
        reward=reward_spec,
        generator=fake_generator,
        logging=logging_handles,
    )
    run_training_eval.run_validation_step(3, ctx)
    assert writer.logged
    metrics, step = writer.logged[-1]
    assert metrics["eval/mean_reward"] == 1.0
    assert step == 3
    assert calls == [["p1"]]


def test_log_eval_start_only_logs_on_main_rank(rte, caplog):
    run_training_eval, _ = rte
    caplog.set_level("INFO")
    shard_non_main = run_training_eval._EvalShardInfo(
        rows=[],
        total_rows=10,
        shard_total=5,
        world_size=2,
        log_every=1,
        is_main=False,
    )
    run_training_eval._log_eval_start(5, shard_non_main, batch_size=4)
    assert "eval step" not in caplog.text

    caplog.clear()
    shard_main = run_training_eval._EvalShardInfo(
        rows=[],
        total_rows=12,
        shard_total=6,
        world_size=2,
        log_every=1,
        is_main=True,
    )
    run_training_eval._log_eval_start(6, shard_main, batch_size=8)
    assert "eval step 6" in caplog.text


def test_gather_eval_stats_falls_back_to_local_sum(rte):
    run_training_eval, _ = rte
    accelerator = SimpleNamespace(gather_object=None)
    total_sum, total_count = run_training_eval._gather_eval_stats(
        accelerator, [1.0, 2.0]
    )
    assert total_sum == pytest.approx(3.0)
    assert total_count == pytest.approx(2.0)


def test_gather_eval_stats_uses_gather_when_available(rte):
    run_training_eval, _ = rte

    class _Accel:
        def __init__(self):
            self.calls = 0

        def gather_object(self, payload):
            self.calls += 1
            return [payload, (5.0, 3.0)]

    accel = _Accel()
    total_sum, total_count = run_training_eval._gather_eval_stats(accel, [2.0])
    assert accel.calls == 1
    assert total_sum == pytest.approx(7.0)
    assert total_count == pytest.approx(4.0)
