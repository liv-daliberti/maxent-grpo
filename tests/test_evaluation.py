from types import SimpleNamespace
import src.utils.evaluation as E


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

    training_args = SimpleNamespace(benchmarks=["tiny"], hub_model_id="org/model", hub_model_revision="main", system_prompt=None)
    model_args = SimpleNamespace(trust_remote_code=True)
    E.run_benchmark_jobs(training_args, model_args)
    assert "cmd" in calls

