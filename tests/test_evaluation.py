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
"""

from types import SimpleNamespace
import pytest
import core.evaluation as E


def test_register_lighteval_task_formats_list():
    cfg = {}
    E.register_lighteval_task(cfg, "lighteval", "toy", "task1,task2", num_fewshot=3)
    assert cfg["toy"] == "lighteval|task1|3|0,lighteval|task2|3|0"


def test_run_benchmark_jobs_invokes_subprocess(monkeypatch):
    # Keep benchmark set small and stub subprocess + GPU helpers
    cfg = {}
    E.register_lighteval_task(cfg, "lighteval", "tiny", "task", 0)
    E.LIGHTEVAL_TASKS.update(cfg)

    calls = {}
    monkeypatch.setattr(E, "get_gpu_count_for_vllm", lambda *a, **k: 2)
    monkeypatch.setattr(E, "get_param_count_from_repo_id", lambda *a, **k: 1_000_000)

    def fake_run(cmd, check):
        calls["cmd"] = cmd

    monkeypatch.setattr(E.subprocess, "run", fake_run)

    training_args = SimpleNamespace(
        benchmarks=["tiny"],
        hub_model_id="org/model",
        hub_model_revision="main",
        system_prompt=None,
    )
    model_args = SimpleNamespace(trust_remote_code=True)
    E.run_benchmark_jobs(training_args, model_args)
    assert "cmd" in calls


def test_build_slurm_gpu_flag_styles(monkeypatch):
    monkeypatch.setenv("SLURM_GPU_FLAG_STYLE", "gpus")
    assert E._build_slurm_gpu_flag(3) == ["--gpus=3"]
    monkeypatch.setenv("SLURM_GPU_FLAG_STYLE", "gres")
    assert E._build_slurm_gpu_flag(2) == ["--gres=gpu:2"]
    monkeypatch.delenv("SLURM_GPU_FLAG_STYLE", raising=False)
    assert E._build_slurm_gpu_flag(1) == []


def test_run_lighteval_job_tensor_parallel_and_prompt(monkeypatch):
    """Large models trigger tensor_parallel=True and encode prompts."""
    monkeypatch.setattr(E, "get_gpu_count_for_vllm", lambda *a, **k: 4)
    monkeypatch.setattr(E, "get_param_count_from_repo_id", lambda *_: 40_000_000_000)
    monkeypatch.setenv("SLURM_GPU_FLAG_STYLE", "gpus")
    captured = {}

    def fake_run(cmd, check):
        captured["cmd"] = cmd
        captured["check"] = check

    monkeypatch.setattr(E.subprocess, "run", fake_run)
    training_args = SimpleNamespace(
        hub_model_id="org/bigm",
        hub_model_revision="main",
        system_prompt="special prompt",
    )
    model_args = SimpleNamespace(trust_remote_code=False)
    E.run_lighteval_job("math_500", training_args, model_args)
    assert captured["check"] is True
    assert any("--gpus=" in part or "--gres=" in part for part in captured["cmd"])
    assert "special prompt" not in " ".join(captured["cmd"])  # encoded


def test_run_benchmark_jobs_all_and_unknown(monkeypatch):
    calls = []
    monkeypatch.setattr(E, "run_lighteval_job", lambda name, *_args: calls.append(name))
    training_args = SimpleNamespace(
        benchmarks=["all"],
        hub_model_id="org/model",
        hub_model_revision="main",
        system_prompt=None,
    )
    model_args = SimpleNamespace(trust_remote_code=True)
    E.run_benchmark_jobs(training_args, model_args)
    assert set(calls) == set(E.get_lighteval_tasks())

    training_args.benchmarks = ["nope"]
    with pytest.raises(ValueError):
        E.run_benchmark_jobs(training_args, model_args)


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
"""
