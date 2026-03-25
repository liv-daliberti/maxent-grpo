from __future__ import annotations

import json
from types import SimpleNamespace

from maxent_grpo.training import seed_paper_eval_callback as seed_eval_cb
from maxent_grpo.training.seed_paper_eval_callback import (
    SeedPaperEvalCallback,
    build_live_seed_paper_eval_command,
)


def test_build_live_seed_paper_eval_command_uses_repo_local_paths(
    monkeypatch,
) -> None:
    monkeypatch.setenv("MAXENT_VLLM_URL", "http://127.0.0.1:8001/generate")
    args = SimpleNamespace(
        run_name="quartet-test-run",
        model_name_or_path="Qwen/Qwen2.5-Math-1.5B",
        seed_paper_eval_template="no",
        seed_paper_eval_tasks=None,
        seed_paper_eval_max_test=999999,
        seed_paper_eval_vllm_batch_size=32,
        seed_paper_eval_workspace_dir=None,
        seed_paper_eval_results_dir=None,
        seed_paper_eval_python=None,
        seed_paper_eval_pass_at_8_enabled=True,
        seed_paper_eval_pass_at_8_samples=8,
        seed_paper_eval_pass_at_8_temperature=1.0,
        seed_paper_eval_pass_at_8_top_p=1.0,
    )
    command, results_dir = build_live_seed_paper_eval_command(args, step=32)
    assert command[0].endswith("/bin/python")
    assert command[1].endswith("/tools/seed_paper_eval.py")
    assert "--vllm-url" in command
    assert "http://127.0.0.1:8001/generate" in command
    assert "--template" in command
    assert "no" in command
    assert "--vllm-batch-size" in command
    assert "32" in command
    assert "--pass-at-8" in command
    assert "--pass-at-8-samples" in command
    assert "8" in command
    assert "--pass-at-8-temperature" in command
    assert "--pass-at-8-top-p" in command
    assert str(results_dir).endswith("/var/artifacts/seed_paper_eval/live/quartet-test-run/step-000032")


def test_build_live_seed_paper_eval_command_falls_back_to_env_model_name(
    monkeypatch,
) -> None:
    monkeypatch.setenv("MAXENT_VLLM_URL", "http://127.0.0.1:8001/generate")
    monkeypatch.setenv("SEED_PAPER_EVAL_MODEL_NAME", "Qwen/Qwen2.5-Math-1.5B")
    args = SimpleNamespace(
        run_name="seed-test-run",
        model_name_or_path=None,
        seed_paper_eval_template="no",
        seed_paper_eval_tasks=None,
        seed_paper_eval_max_test=None,
        seed_paper_eval_vllm_batch_size=None,
        seed_paper_eval_workspace_dir=None,
        seed_paper_eval_results_dir=None,
        seed_paper_eval_python=None,
    )
    command, _ = build_live_seed_paper_eval_command(args, step=0)
    assert "--model-name" in command
    assert command[command.index("--model-name") + 1] == "Qwen/Qwen2.5-Math-1.5B"


def test_build_live_seed_paper_eval_command_reads_model_name_from_recipe_env(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("MAXENT_VLLM_URL", "http://127.0.0.1:8001/generate")
    recipe_path = tmp_path / "recipe.yaml"
    recipe_path.write_text(
        "model_name_or_path: Qwen/Qwen2.5-Math-1.5B\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("GRPO_RECIPE_USED", str(recipe_path))
    args = SimpleNamespace(
        run_name="seed-test-run",
        model_name_or_path=None,
        seed_paper_eval_template="no",
        seed_paper_eval_tasks=None,
        seed_paper_eval_max_test=None,
        seed_paper_eval_vllm_batch_size=None,
        seed_paper_eval_workspace_dir=None,
        seed_paper_eval_results_dir=None,
        seed_paper_eval_python=None,
    )
    command, _ = build_live_seed_paper_eval_command(args, step=0)
    assert "--model-name" in command
    assert command[command.index("--model-name") + 1] == "Qwen/Qwen2.5-Math-1.5B"


def test_seed_paper_eval_callback_direct_schedule_runs_on_train_begin(
    monkeypatch,
    tmp_path,
) -> None:
    results_root = tmp_path / "results"
    summary_dir = results_root / "direct-run" / "step-000000"
    summary_dir.mkdir(parents=True)
    summary_path = summary_dir / "seed_paper_eval_20260319T000000Z.summary.json"
    summary_path.write_text(
        json.dumps({"scores": {"aime": 0.1, "avg": 0.1}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("MAXENT_VLLM_URL", "http://127.0.0.1:8001/generate")
    calls: list[list[str]] = []

    def _fake_run(command, **kwargs):
        calls.append(list(command))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(
        "maxent_grpo.training.seed_paper_eval_callback.subprocess.run",
        _fake_run,
    )
    args = SimpleNamespace(
        run_name="direct-run",
        model_name_or_path="Qwen/Qwen2.5-Math-1.5B",
        seed_paper_eval_enabled=True,
        seed_paper_eval_template="no",
        seed_paper_eval_tasks=None,
        seed_paper_eval_max_test=999999,
        seed_paper_eval_vllm_batch_size=32,
        seed_paper_eval_workspace_dir=str(tmp_path / "workspace"),
        seed_paper_eval_results_dir=str(results_root),
        seed_paper_eval_python="/usr/bin/python3",
        seed_paper_eval_timeout_s=60,
        seed_paper_eval_fail_on_error=True,
        do_eval=False,
        eval_strategy="no",
        eval_on_start=False,
        seed_paper_eval_on_start=True,
        eval_steps=32,
    )
    callback = SeedPaperEvalCallback(args)
    state = SimpleNamespace(is_world_process_zero=True, global_step=0)
    control = object()
    out = callback.on_train_begin(args=None, state=state, control=control)
    assert out is control
    assert len(calls) == 1
    assert "--results-dir" in calls[0]
    assert str(summary_dir) == calls[0][calls[0].index("--results-dir") + 1]


def test_seed_paper_eval_callback_direct_schedule_runs_on_step_end(
    monkeypatch,
    tmp_path,
) -> None:
    results_root = tmp_path / "results"
    summary_dir = results_root / "direct-run" / "step-000032"
    summary_dir.mkdir(parents=True)
    summary_path = summary_dir / "seed_paper_eval_20260319T000032Z.summary.json"
    summary_path.write_text(
        json.dumps({"scores": {"aime": 0.2, "avg": 0.2}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("MAXENT_VLLM_URL", "http://127.0.0.1:8001/generate")
    calls: list[list[str]] = []

    def _fake_run(command, **kwargs):
        calls.append(list(command))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(
        "maxent_grpo.training.seed_paper_eval_callback.subprocess.run",
        _fake_run,
    )
    args = SimpleNamespace(
        run_name="direct-run",
        model_name_or_path="Qwen/Qwen2.5-Math-1.5B",
        seed_paper_eval_enabled=True,
        seed_paper_eval_template="no",
        seed_paper_eval_tasks=None,
        seed_paper_eval_max_test=999999,
        seed_paper_eval_vllm_batch_size=32,
        seed_paper_eval_workspace_dir=str(tmp_path / "workspace"),
        seed_paper_eval_results_dir=str(results_root),
        seed_paper_eval_python="/usr/bin/python3",
        seed_paper_eval_timeout_s=60,
        seed_paper_eval_fail_on_error=True,
        do_eval=False,
        eval_strategy="no",
        eval_on_start=True,
        eval_steps=32,
    )
    callback = SeedPaperEvalCallback(args)
    state = SimpleNamespace(is_world_process_zero=True, global_step=32)
    control = object()
    out = callback.on_step_end(args=None, state=state, control=control)
    assert out is control
    assert len(calls) == 1
    assert "--results-dir" in calls[0]
    assert str(summary_dir) == calls[0][calls[0].index("--results-dir") + 1]


def test_seed_paper_eval_callback_direct_schedule_runs_on_train_end_for_final_step(
    monkeypatch,
    tmp_path,
) -> None:
    results_root = tmp_path / "results"
    summary_dir = results_root / "direct-run" / "step-000384"
    summary_dir.mkdir(parents=True)
    summary_path = summary_dir / "seed_paper_eval_20260319T000384Z.summary.json"
    summary_path.write_text(
        json.dumps({"scores": {"aime": 0.3, "avg": 0.3}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("MAXENT_VLLM_URL", "http://127.0.0.1:8001/generate")
    calls: list[list[str]] = []

    def _fake_run(command, **kwargs):
        calls.append(list(command))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(
        "maxent_grpo.training.seed_paper_eval_callback.subprocess.run",
        _fake_run,
    )
    args = SimpleNamespace(
        run_name="direct-run",
        model_name_or_path="Qwen/Qwen2.5-Math-1.5B",
        seed_paper_eval_enabled=True,
        seed_paper_eval_template="no",
        seed_paper_eval_tasks=None,
        seed_paper_eval_max_test=999999,
        seed_paper_eval_vllm_batch_size=32,
        seed_paper_eval_workspace_dir=str(tmp_path / "workspace"),
        seed_paper_eval_results_dir=str(results_root),
        seed_paper_eval_python="/usr/bin/python3",
        seed_paper_eval_timeout_s=60,
        seed_paper_eval_fail_on_error=True,
        do_eval=False,
        eval_strategy="no",
        eval_on_start=True,
        eval_steps=25,
    )
    callback = SeedPaperEvalCallback(args)
    state = SimpleNamespace(is_world_process_zero=True, global_step=384)
    control = object()
    out = callback.on_train_end(args=None, state=state, control=control)
    assert out is control
    assert len(calls) == 1
    assert "--results-dir" in calls[0]
    assert str(summary_dir) == calls[0][calls[0].index("--results-dir") + 1]


def test_seed_paper_eval_callback_syncs_existing_step0_summary_into_trainer_run(
    monkeypatch,
    tmp_path,
) -> None:
    results_root = tmp_path / "results"
    step0_dir = tmp_path / "step0"
    step0_dir.mkdir(parents=True)
    summary_path = step0_dir / "seed_paper_eval_20260319T000000Z.summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "results": {"aime": 0.1, "avg": 0.1},
                "avg": 0.1,
                "process_exit_code": 0,
            }
        ),
        encoding="utf-8",
    )

    class _FakeRun:
        def __init__(self) -> None:
            self.logged = []
            self.summary = {}
            self.defined_metrics = []

        def log(self, payload, step=None, commit=None):
            self.logged.append((dict(payload), step, commit))

        def define_metric(self, name, step_metric=None):
            self.defined_metrics.append((name, step_metric))

    fake_run = _FakeRun()
    calls: list[list[str]] = []

    def _fake_run_subprocess(command, **kwargs):
        calls.append(list(command))
        return SimpleNamespace(returncode=0)

    monkeypatch.setenv("MAXENT_VLLM_URL", "http://127.0.0.1:8001/generate")
    monkeypatch.setenv("MAXENT_STEP0_PAPER_EVAL_RESULTS_DIR", str(step0_dir))
    monkeypatch.setattr(
        "maxent_grpo.training.seed_paper_eval_callback._wandb_run",
        lambda: fake_run,
    )
    monkeypatch.setattr(
        "maxent_grpo.training.seed_paper_eval_callback.subprocess.run",
        _fake_run_subprocess,
    )
    args = SimpleNamespace(
        run_name="direct-run",
        model_name_or_path="Qwen/Qwen2.5-Math-1.5B",
        seed_paper_eval_enabled=True,
        seed_paper_eval_template="no",
        seed_paper_eval_tasks=None,
        seed_paper_eval_max_test=999999,
        seed_paper_eval_vllm_batch_size=32,
        seed_paper_eval_workspace_dir=str(tmp_path / "workspace"),
        seed_paper_eval_results_dir=str(results_root),
        seed_paper_eval_python="/usr/bin/python3",
        seed_paper_eval_timeout_s=60,
        seed_paper_eval_fail_on_error=True,
        do_eval=False,
        eval_strategy="no",
        eval_on_start=False,
        seed_paper_eval_on_start=True,
        eval_steps=32,
    )
    callback = SeedPaperEvalCallback(args)
    state = SimpleNamespace(is_world_process_zero=True, global_step=0)
    control = object()
    out = callback.on_train_begin(args=None, state=state, control=control)
    assert out is control
    assert calls == []
    assert fake_run.logged
    payload, step, commit = fake_run.logged[-1]
    assert step is None
    assert commit is True
    assert payload["step0_paper_eval/aime"] == 0.1
    assert payload["paper_eval/aime"] == 0.1
    assert payload["step0_paper_eval/training_step"] == 0
    assert payload["paper_eval/training_step"] == 0
    assert fake_run.summary["paper_eval/source"] == "step0"
    assert ("step0_paper_eval/training_step", None) in fake_run.defined_metrics
    assert ("step0_paper_eval/*", "step0_paper_eval/training_step") in fake_run.defined_metrics
    assert ("paper_eval/training_step", None) in fake_run.defined_metrics
    assert ("paper_eval/*", "paper_eval/training_step") in fake_run.defined_metrics


def test_live_seed_paper_eval_logging_uses_prefixed_training_step_axis(
    monkeypatch,
    tmp_path,
) -> None:
    class _FakeRun:
        def __init__(self) -> None:
            self.logged = []
            self.summary = {}
            self.defined_metrics = []

        def log(self, payload, step=None, commit=None):
            self.logged.append((dict(payload), step, commit))

        def define_metric(self, name, step_metric=None):
            self.defined_metrics.append((name, step_metric))

    fake_run = _FakeRun()
    monkeypatch.setattr(seed_eval_cb, "_wandb_run", lambda: fake_run)
    seed_eval_cb._log_summary_to_wandb(
        summary={"results": {"aime": 0.2}, "avg": 0.2},
        summary_path=tmp_path / "summary.json",
        step=32,
    )
    payload, step, commit = fake_run.logged[-1]
    assert step is None
    assert commit is True
    assert payload["paper_eval/training_step"] == 32
    assert payload["paper_eval/ok"] == 1.0
    assert payload["seed_paper_eval_live/training_step"] == 32
    assert payload["seed_paper_eval_live/ok"] == 1.0
    assert ("paper_eval/training_step", None) in fake_run.defined_metrics
    assert ("paper_eval/*", "paper_eval/training_step") in fake_run.defined_metrics
    assert ("seed_paper_eval_live/training_step", None) in fake_run.defined_metrics
    assert (
        "seed_paper_eval_live/*",
        "seed_paper_eval_live/training_step",
    ) in fake_run.defined_metrics


def test_live_seed_paper_eval_failure_logs_prefixed_training_step_without_wandb_step(
    monkeypatch,
    tmp_path,
) -> None:
    class _FakeRun:
        def __init__(self) -> None:
            self.logged = []
            self.summary = {}
            self.defined_metrics = []

        def log(self, payload, step=None, commit=None):
            self.logged.append((dict(payload), step, commit))

        def define_metric(self, name, step_metric=None):
            self.defined_metrics.append((name, step_metric))

    fake_run = _FakeRun()
    monkeypatch.setattr(seed_eval_cb, "_wandb_run", lambda: fake_run)
    monkeypatch.setenv("MAXENT_VLLM_URL", "http://127.0.0.1:8001/generate")

    def _fake_run_subprocess(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(seed_eval_cb.subprocess, "run", _fake_run_subprocess)
    args = SimpleNamespace(
        run_name="direct-run",
        model_name_or_path="Qwen/Qwen2.5-Math-1.5B",
        seed_paper_eval_enabled=True,
        seed_paper_eval_template="no",
        seed_paper_eval_tasks=None,
        seed_paper_eval_max_test=999999,
        seed_paper_eval_vllm_batch_size=32,
        seed_paper_eval_workspace_dir=str(tmp_path / "workspace"),
        seed_paper_eval_results_dir=str(tmp_path / "results"),
        seed_paper_eval_python="/usr/bin/python3",
        seed_paper_eval_timeout_s=60,
        seed_paper_eval_fail_on_error=False,
        do_eval=False,
        eval_strategy="no",
        eval_on_start=False,
        seed_paper_eval_on_start=False,
        eval_steps=25,
    )
    callback = SeedPaperEvalCallback(args)
    callback._run_live_eval(25)
    payload, step, commit = fake_run.logged[-1]
    assert step is None
    assert commit is True
    assert payload["seed_paper_eval_live/training_step"] == 25
    assert payload["seed_paper_eval_live/ok"] == 0.0
    assert payload["paper_eval/training_step"] == 25
    assert payload["paper_eval/ok"] == 0.0
    assert ("paper_eval/training_step", None) in fake_run.defined_metrics
    assert ("paper_eval/*", "paper_eval/training_step") in fake_run.defined_metrics
