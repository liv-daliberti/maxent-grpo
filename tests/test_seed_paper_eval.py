from __future__ import annotations

import json
from pathlib import Path
from types import ModuleType, SimpleNamespace

from maxent_grpo.seed_paper_eval import (
    SeedPaperEvalConfig,
    _run_vllm_server_eval,
    absolute_path,
    build_seed_paper_eval_payload,
    build_step0_wandb_payload,
    build_launch_command,
    build_official_eval_command,
    child_env,
    compare_against_expected_profile,
    default_python_executable,
    parse_official_eval_summary,
    parse_task_list,
    resolve_template,
)


def test_default_python_executable_prefers_repo_venv(tmp_path: Path, monkeypatch) -> None:
    paper_python = tmp_path / "var" / "seed_paper_eval" / "paper_venv" / "bin" / "python"
    paper_python.parent.mkdir(parents=True, exist_ok=True)
    paper_python.write_text("#!/bin/sh\n", encoding="utf-8")
    paper_python.chmod(0o755)
    venv_python = tmp_path / "var" / "e2e-venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("#!/bin/sh\n", encoding="utf-8")
    venv_python.chmod(0o755)
    assert default_python_executable(tmp_path) == paper_python


def test_absolute_path_preserves_venv_symlink(tmp_path: Path) -> None:
    target = tmp_path / "python-real"
    target.write_text("#!/bin/sh\n", encoding="utf-8")
    link = tmp_path / "python"
    link.symlink_to(target)
    assert absolute_path(link) == link.absolute()


def test_parse_task_list_uses_paper_defaults() -> None:
    assert parse_task_list(None) == (
        "aime",
        "amc",
        "math",
        "minerva",
        "olympiad_bench",
    )
    assert parse_task_list("amc,math") == ("amc", "math")


def test_resolve_template_uses_profile_specific_paper_default() -> None:
    assert resolve_template(None, None) == "qwen_math"
    assert resolve_template(None, "table2_qwen2_5_math_1_5b") == "no"
    assert (
        resolve_template("qwen_math", "table2_qwen2_5_math_1_5b")
        == "qwen_math"
    )


def test_build_official_eval_command_uses_paper_settings(tmp_path: Path) -> None:
    config = SeedPaperEvalConfig(
        python_executable=tmp_path / "python",
        workspace_dir=tmp_path / "workspace",
        seed_repo_dir=tmp_path / "seed",
        dataset_dir=tmp_path / "seed" / "datasets" / "evaluation_suite",
        results_dir=tmp_path / "results",
        requirements_file=tmp_path / "requirements.txt",
        seed_repo_url="https://example.com/seed.git",
        seed_repo_commit="deadbeef",
        model_name="Qwen/Qwen2.5-Math-1.5B",
        template="qwen_math",
        tasks=("aime", "amc", "math", "minerva", "olympiad_bench"),
        temperature=0.0,
        top_p=1.0,
        max_tokens=3000,
        max_model_len=4096,
        vllm_url=None,
        vllm_batch_size=32,
        vllm_use_rollout_token_guard=False,
        vllm_stop_sequences=None,
        n_samples=1,
        max_test=999999,
        prompt_start=None,
        prompt_end=None,
        save_outputs=False,
        auto_install=True,
        prepare_only=False,
        use_srun=False,
        srun_args=(),
        expected_profile=None,
        expected_tolerance=1e-6,
        enforce_expected=False,
        runtime_enforce_eager=False,
        runtime_disable_async_output_proc=False,
        runtime_gpu_memory_utilization=None,
        runtime_swap_space=None,
        wandb_enabled=False,
        wandb_project=None,
        wandb_entity=None,
        wandb_group=None,
        wandb_run_name=None,
        wandb_job_type=None,
    )
    command = build_official_eval_command(config)
    assert command[0] == str(tmp_path / "python")
    assert command[1] == "-u"
    assert command[2] == str(tmp_path / "seed" / "evaluate_model.py")
    assert f"--model_name={config.model_name}" in command
    assert "--template=qwen_math" in command
    assert f"--dataset_name={tmp_path / 'seed' / 'datasets' / 'evaluation_suite'}" in command
    assert "--temperature=0.0" in command
    assert "--top_p=1.0" in command
    assert "--max_tokens=3000" in command
    assert "--max_model_len=4096" in command
    assert "--n_samples=1" in command
    assert "--max_test=999999" in command
    assert not any(part.startswith("--vllm_batch_size=") for part in command)
    assert not any(part.startswith("--vllm-url=") for part in command)
    assert not any(part.startswith("--tasks=") for part in command)


def test_build_launch_command_wraps_srun_when_requested(tmp_path: Path) -> None:
    config = SeedPaperEvalConfig(
        python_executable=tmp_path / "python",
        workspace_dir=tmp_path / "workspace",
        seed_repo_dir=tmp_path / "seed",
        dataset_dir=tmp_path / "seed" / "datasets" / "evaluation_suite",
        results_dir=tmp_path / "results",
        requirements_file=tmp_path / "requirements.txt",
        seed_repo_url="https://example.com/seed.git",
        seed_repo_commit="deadbeef",
        model_name="Qwen/Qwen2.5-Math-1.5B",
        template="qwen_math",
        tasks=("aime",),
        temperature=0.0,
        top_p=1.0,
        max_tokens=3000,
        max_model_len=4096,
        vllm_url=None,
        vllm_batch_size=32,
        vllm_use_rollout_token_guard=False,
        vllm_stop_sequences=None,
        n_samples=1,
        max_test=1,
        prompt_start=None,
        prompt_end=None,
        save_outputs=False,
        auto_install=True,
        prepare_only=False,
        use_srun=True,
        srun_args=("--partition=all", "--gres=gpu:1"),
        expected_profile=None,
        expected_tolerance=1e-6,
        enforce_expected=False,
        runtime_enforce_eager=False,
        runtime_disable_async_output_proc=False,
        runtime_gpu_memory_utilization=None,
        runtime_swap_space=None,
        wandb_enabled=False,
        wandb_project=None,
        wandb_entity=None,
        wandb_group=None,
        wandb_run_name=None,
        wandb_job_type=None,
    )
    command = build_launch_command(config)
    assert command[:3] == ["srun", "--partition=all", "--gres=gpu:1"]
    assert command[3] == str(tmp_path / "python")


def test_build_official_eval_command_appends_runtime_workarounds(tmp_path: Path) -> None:
    config = SeedPaperEvalConfig(
        python_executable=tmp_path / "python",
        workspace_dir=tmp_path / "workspace",
        seed_repo_dir=tmp_path / "seed",
        dataset_dir=tmp_path / "seed" / "datasets" / "evaluation_suite",
        results_dir=tmp_path / "results",
        requirements_file=tmp_path / "requirements.txt",
        seed_repo_url="https://example.com/seed.git",
        seed_repo_commit="deadbeef",
        model_name="Qwen/Qwen2.5-Math-1.5B",
        template="qwen_math",
        tasks=("aime",),
        temperature=0.0,
        top_p=1.0,
        max_tokens=3000,
        max_model_len=4096,
        vllm_url=None,
        vllm_batch_size=32,
        vllm_use_rollout_token_guard=False,
        vllm_stop_sequences=None,
        n_samples=1,
        max_test=1,
        prompt_start=None,
        prompt_end=None,
        save_outputs=False,
        auto_install=True,
        prepare_only=False,
        use_srun=False,
        srun_args=(),
        expected_profile=None,
        expected_tolerance=1e-6,
        enforce_expected=False,
        runtime_enforce_eager=True,
        runtime_disable_async_output_proc=True,
        runtime_gpu_memory_utilization=0.25,
        runtime_swap_space=0.0,
        wandb_enabled=False,
        wandb_project=None,
        wandb_entity=None,
        wandb_group=None,
        wandb_run_name=None,
        wandb_job_type=None,
    )
    command = build_official_eval_command(config)
    assert "--enforce_eager=True" in command
    assert "--disable_async_output_proc=True" in command
    assert "--gpu_memory_utilization=0.25" in command
    assert "--swap_space=0.0" in command


def test_build_step0_wandb_payload_includes_results_and_expected_comparison() -> None:
    summary = {
        "results": {
            "aime": 5.0 / 30.0,
            "amc": 36.0 / 83.0,
        },
        "avg": 0.3,
        "pass_at_8": {
            "aime": 0.4,
            "amc": 0.5,
        },
        "pass_at_8_avg": 0.45,
        "mean_at_8": {
            "aime": 0.2,
            "amc": 0.3,
        },
        "mean_at_8_avg": 0.25,
        "process_exit_code": 0,
        "expected_comparison": {
            "ok": True,
            "tolerance": 1e-6,
            "compared": {
                "aime": {
                    "expected": 5.0 / 30.0,
                    "observed": 5.0 / 30.0,
                    "delta": 0.0,
                }
            },
            "mismatches": [],
        },
    }
    payload = build_step0_wandb_payload(summary)
    assert payload["step0_paper_eval/aime"] == 5.0 / 30.0
    assert payload["step0_paper_eval/amc"] == 36.0 / 83.0
    assert payload["step0_paper_eval/avg"] == 0.3
    assert payload["step0_paper_eval/pass_at_8/aime"] == 0.4
    assert payload["step0_paper_eval/pass_at_8_avg"] == 0.45
    assert payload["step0_paper_eval/mean_at_8/amc"] == 0.3
    assert payload["step0_paper_eval/mean_at_8_avg"] == 0.25
    assert payload["step0_paper_eval/process_exit_code"] == 0
    assert payload["step0_paper_eval/expected_ok"] is True
    assert payload["step0_paper_eval/expected_mismatch_count"] == 0
    assert (
        payload["step0_paper_eval_expected/aime/expected"]
        == 5.0 / 30.0
    )


def test_build_seed_paper_eval_payload_supports_custom_prefix() -> None:
    summary = {
        "results": {"math": 0.618},
        "avg": 0.33,
        "pass_at_8": {"math": 0.72},
        "pass_at_8_avg": 0.72,
        "mean_at_8": {"math": 0.41},
        "mean_at_8_avg": 0.41,
        "process_exit_code": 0,
    }
    payload = build_seed_paper_eval_payload(
        summary,
        prefix="seed_paper_eval_live",
    )
    assert payload["seed_paper_eval_live/math"] == 0.618
    assert payload["seed_paper_eval_live/avg"] == 0.33
    assert payload["seed_paper_eval_live/pass_at_8/math"] == 0.72
    assert payload["seed_paper_eval_live/pass_at_8_avg"] == 0.72
    assert payload["seed_paper_eval_live/mean_at_8/math"] == 0.41
    assert payload["seed_paper_eval_live/mean_at_8_avg"] == 0.41
    assert payload["seed_paper_eval_live/process_exit_code"] == 0


def test_parse_official_eval_summary_reads_results_blocks() -> None:
    log_text = """
Using template: qwen_math
{'aime': 0.16666666666666666, 'amc': 0.43373493975903615, 'math': 0.618}
avg: 0.4061338688085676
pass_at_8: {'aime': 0.3, 'amc': 0.5, 'math': 0.7}
pass_at_8_avg: 0.5
mean_at_8: {'aime': 0.2, 'amc': 0.25, 'math': 0.4}
mean_at_8_avg: 0.2833333333333333
avg_lens: {'aime': 1401.6}
max_lens: {'aime': 3000}
formatted: {'aime': 0.7333333333333333}
""".strip()
    summary = parse_official_eval_summary(log_text)
    assert summary["results"] == {
        "aime": 0.16666666666666666,
        "amc": 0.43373493975903615,
        "math": 0.618,
    }
    assert summary["avg"] == 0.4061338688085676
    assert summary["pass_at_8"] == {"aime": 0.3, "amc": 0.5, "math": 0.7}
    assert summary["pass_at_8_avg"] == 0.5
    assert summary["mean_at_8"] == {"aime": 0.2, "amc": 0.25, "math": 0.4}
    assert summary["mean_at_8_avg"] == 0.2833333333333333
    assert summary["avg_lens"] == {"aime": 1401.6}
    assert summary["max_lens"] == {"aime": 3000.0}
    assert summary["formatted"] == {"aime": 0.7333333333333333}


def test_parse_official_eval_summary_strips_numpy_scalar_wrappers() -> None:
    log_text = """
{'aime': np.float64(0.16666666666666666), 'math': np.float64(0.618)}
avg: 0.3923333333333333
max_lens: {'aime': np.int64(3000)}
""".strip()
    summary = parse_official_eval_summary(log_text)
    assert summary["results"] == {
        "aime": 0.16666666666666666,
        "math": 0.618,
    }
    assert summary["max_lens"] == {"aime": 3000.0}


def test_child_env_drops_pythonpath(monkeypatch) -> None:
    monkeypatch.setenv("PYTHONPATH", "/tmp/foreign/site-packages")
    env = child_env()
    assert env["PYTHONNOUSERSITE"] == "0"
    assert env["MAXENT_SEED_PAPER_SUPPRESS_GRADER_LOGS"] == "1"
    assert "PYTHONPATH" not in env


def test_compare_against_expected_profile_accepts_table2_qwen_math_1_5b() -> None:
    summary = {
        "results": {
            "aime": 5.0 / 30.0,
            "amc": 36.0 / 83.0,
            "math": 309.0 / 500.0,
            "minerva": 41.0 / 272.0,
            "olympiad_bench": 192.0 / 675.0,
        },
        "avg": (
            (5.0 / 30.0)
            + (36.0 / 83.0)
            + (309.0 / 500.0)
            + (41.0 / 272.0)
            + (192.0 / 675.0)
        )
        / 5.0,
    }
    comparison = compare_against_expected_profile(
        summary,
        "table2_qwen2_5_math_1_5b",
        tolerance=1e-9,
    )
    assert comparison["ok"] is True
    assert comparison["mismatches"] == []


def test_run_vllm_server_eval_passes_tokenizer_to_safe_generate(
    monkeypatch, tmp_path: Path
) -> None:
    captured: dict[str, object] = {}

    class DummyTokenizer:
        def decode(self, ids, skip_special_tokens=True):
            return "".join(str(x) for x in ids)

    def _fake_from_pretrained(model_name: str):
        captured["model_name"] = model_name
        return DummyTokenizer()

    def _fake_safe_generate(**kwargs):
        captured["tokenizer"] = kwargs.get("tokenizer")
        return [[r"\boxed{1}"]], [[None]], 1.0

    class FakeDataset(dict):
        pass

    def _fake_load_from_disk(path: str):
        captured["dataset_path"] = path
        return FakeDataset(
            {
                "aime": {
                    "problem": ["What is 1?"],
                    "answer": ["1"],
                }
            }
        )

    def _fake_reward_fn(model_output, gt, fast=False):
        del gt, fast
        return {"formatted": True}, float(model_output == r"\boxed{1}")

    import sys

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoTokenizer=SimpleNamespace(from_pretrained=_fake_from_pretrained)
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "datasets",
        SimpleNamespace(load_from_disk=_fake_load_from_disk),
    )
    understand_pkg = ModuleType("understand_r1_zero")
    grader_mod = ModuleType("understand_r1_zero.math_grader")
    grader_mod.answer_tag_reward_fn = _fake_reward_fn
    grader_mod.answer_tag_reward_fn_for_orz = _fake_reward_fn
    grader_mod.boxed_reward_fn = _fake_reward_fn
    training_pkg = ModuleType("maxent_grpo.training")
    patches_pkg = ModuleType("maxent_grpo.training.patches")
    vllm_patch_mod = ModuleType("maxent_grpo.training.patches.vllm")
    vllm_patch_mod.safe_generate = _fake_safe_generate
    patches_pkg.vllm = vllm_patch_mod
    training_pkg.patches = patches_pkg
    monkeypatch.setitem(sys.modules, "understand_r1_zero", understand_pkg)
    monkeypatch.setitem(sys.modules, "understand_r1_zero.math_grader", grader_mod)
    monkeypatch.setitem(sys.modules, "maxent_grpo.training", training_pkg)
    monkeypatch.setitem(sys.modules, "maxent_grpo.training.patches", patches_pkg)
    monkeypatch.setitem(sys.modules, "maxent_grpo.training.patches.vllm", vllm_patch_mod)

    config = SeedPaperEvalConfig(
        python_executable=tmp_path / "python",
        workspace_dir=tmp_path / "workspace",
        seed_repo_dir=tmp_path / "seed",
        dataset_dir=tmp_path / "seed" / "datasets" / "evaluation_suite",
        results_dir=tmp_path / "results",
        requirements_file=tmp_path / "requirements.txt",
        seed_repo_url="https://example.com/seed.git",
        seed_repo_commit="deadbeef",
        model_name="Qwen/Qwen2.5-Math-1.5B",
        template="no",
        tasks=("aime",),
        temperature=0.0,
        top_p=1.0,
        max_tokens=3000,
        max_model_len=4096,
        vllm_url="http://127.0.0.1:8000/generate",
        vllm_batch_size=32,
        vllm_use_rollout_token_guard=False,
        vllm_stop_sequences=None,
        n_samples=1,
        max_test=1,
        prompt_start=None,
        prompt_end=None,
        save_outputs=False,
        auto_install=True,
        prepare_only=False,
        use_srun=False,
        srun_args=(),
        expected_profile=None,
        expected_tolerance=1e-6,
        enforce_expected=False,
        runtime_enforce_eager=False,
        runtime_disable_async_output_proc=False,
        runtime_gpu_memory_utilization=None,
        runtime_swap_space=None,
        wandb_enabled=False,
        wandb_project=None,
        wandb_entity=None,
        wandb_group=None,
        wandb_run_name=None,
        wandb_job_type=None,
    )
    summary = _run_vllm_server_eval(config)
    assert captured["model_name"] == "Qwen/Qwen2.5-Math-1.5B"
    assert isinstance(captured["tokenizer"], DummyTokenizer)
    assert summary["results"] == {"aime": 1.0}


def test_run_vllm_server_eval_uses_oat_r1_stop_controls(
    monkeypatch, tmp_path: Path
) -> None:
    captured: dict[str, object] = {}

    class DummyTokenizer:
        pass

    def _fake_from_pretrained(model_name: str):
        captured["model_name"] = model_name
        return DummyTokenizer()

    def _fake_safe_generate(**kwargs):
        captured["stop"] = kwargs.get("stop")
        captured["include_stop_str_in_output"] = kwargs.get(
            "include_stop_str_in_output"
        )
        return [["</think> <answer>1</answer>"]], [[None]], 1.0

    class FakeDataset(dict):
        pass

    def _fake_load_from_disk(path: str):
        del path
        return FakeDataset(
            {
                "aime": {
                    "problem": ["What is 1?"],
                    "answer": ["1"],
                }
            }
        )

    def _fake_reward_fn(model_output, gt, fast=False):
        del fast
        return {"formatted": True}, float(model_output == "</think> <answer>1</answer>")

    import sys

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoTokenizer=SimpleNamespace(from_pretrained=_fake_from_pretrained)
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "datasets",
        SimpleNamespace(load_from_disk=_fake_load_from_disk),
    )
    understand_pkg = ModuleType("understand_r1_zero")
    grader_mod = ModuleType("understand_r1_zero.math_grader")
    grader_mod.answer_tag_reward_fn = _fake_reward_fn
    grader_mod.answer_tag_reward_fn_for_orz = _fake_reward_fn
    grader_mod.boxed_reward_fn = _fake_reward_fn
    training_pkg = ModuleType("maxent_grpo.training")
    patches_pkg = ModuleType("maxent_grpo.training.patches")
    vllm_patch_mod = ModuleType("maxent_grpo.training.patches.vllm")
    vllm_patch_mod.safe_generate = _fake_safe_generate
    patches_pkg.vllm = vllm_patch_mod
    training_pkg.patches = patches_pkg
    monkeypatch.setitem(sys.modules, "understand_r1_zero", understand_pkg)
    monkeypatch.setitem(sys.modules, "understand_r1_zero.math_grader", grader_mod)
    monkeypatch.setitem(sys.modules, "maxent_grpo.training", training_pkg)
    monkeypatch.setitem(sys.modules, "maxent_grpo.training.patches", patches_pkg)
    monkeypatch.setitem(sys.modules, "maxent_grpo.training.patches.vllm", vllm_patch_mod)

    config = SeedPaperEvalConfig(
        python_executable=tmp_path / "python",
        workspace_dir=tmp_path / "workspace",
        seed_repo_dir=tmp_path / "seed",
        dataset_dir=tmp_path / "seed" / "datasets" / "evaluation_suite",
        results_dir=tmp_path / "results",
        requirements_file=tmp_path / "requirements.txt",
        seed_repo_url="https://example.com/seed.git",
        seed_repo_commit="deadbeef",
        model_name="Qwen/Qwen2.5-Math-1.5B",
        template="r1",
        tasks=("aime",),
        temperature=0.0,
        top_p=1.0,
        max_tokens=3000,
        max_model_len=4096,
        vllm_url="http://127.0.0.1:8000/generate",
        vllm_batch_size=32,
        vllm_use_rollout_token_guard=False,
        vllm_stop_sequences=None,
        n_samples=1,
        max_test=1,
        prompt_start=None,
        prompt_end=None,
        save_outputs=False,
        auto_install=True,
        prepare_only=False,
        use_srun=False,
        srun_args=(),
        expected_profile=None,
        expected_tolerance=1e-6,
        enforce_expected=False,
        runtime_enforce_eager=False,
        runtime_disable_async_output_proc=False,
        runtime_gpu_memory_utilization=None,
        runtime_swap_space=None,
        wandb_enabled=False,
        wandb_project=None,
        wandb_entity=None,
        wandb_group=None,
        wandb_run_name=None,
        wandb_job_type=None,
    )
    _run_vllm_server_eval(config)

    assert captured["model_name"] == "Qwen/Qwen2.5-Math-1.5B"
    assert captured["stop"] == ["</answer>"]
    assert captured["include_stop_str_in_output"] is True


def test_run_vllm_server_eval_emits_pass_at_8_and_mean_at_8(
    monkeypatch, tmp_path: Path
) -> None:
    captured_calls: list[dict[str, object]] = []

    class DummyTokenizer:
        pass

    def _fake_from_pretrained(model_name: str):
        assert model_name == "Qwen/Qwen2.5-Math-1.5B"
        return DummyTokenizer()

    def _fake_safe_generate(**kwargs):
        captured_calls.append(dict(kwargs))
        n = int(kwargs["n"])
        if n == 1:
            return [[r"\boxed{1}"], [r"\boxed{0}"]], [[None], [None]], 1.0
        if n == 8:
            return (
                [
                    [r"\boxed{1}"] * 8,
                    [r"\boxed{0}"] * 7 + [r"\boxed{1}"],
                ],
                None,
                1.0,
            )
        raise AssertionError(f"unexpected n={n}")

    class FakeDataset(dict):
        pass

    def _fake_load_from_disk(path: str):
        del path
        return FakeDataset(
            {
                "aime": {
                    "problem": ["What is 1?", "What is 1 again?"],
                    "answer": ["1", "1"],
                }
            }
        )

    def _fake_reward_fn(model_output, gt, fast=False):
        del fast
        return {"formatted": True}, float(model_output == rf"\boxed{{{gt}}}")

    import sys

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoTokenizer=SimpleNamespace(from_pretrained=_fake_from_pretrained)
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "datasets",
        SimpleNamespace(load_from_disk=_fake_load_from_disk),
    )
    understand_pkg = ModuleType("understand_r1_zero")
    grader_mod = ModuleType("understand_r1_zero.math_grader")
    grader_mod.answer_tag_reward_fn = _fake_reward_fn
    grader_mod.answer_tag_reward_fn_for_orz = _fake_reward_fn
    grader_mod.boxed_reward_fn = _fake_reward_fn
    training_pkg = ModuleType("maxent_grpo.training")
    patches_pkg = ModuleType("maxent_grpo.training.patches")
    vllm_patch_mod = ModuleType("maxent_grpo.training.patches.vllm")
    vllm_patch_mod.safe_generate = _fake_safe_generate
    patches_pkg.vllm = vllm_patch_mod
    training_pkg.patches = patches_pkg
    monkeypatch.setitem(sys.modules, "understand_r1_zero", understand_pkg)
    monkeypatch.setitem(sys.modules, "understand_r1_zero.math_grader", grader_mod)
    monkeypatch.setitem(sys.modules, "maxent_grpo.training", training_pkg)
    monkeypatch.setitem(sys.modules, "maxent_grpo.training.patches", patches_pkg)
    monkeypatch.setitem(sys.modules, "maxent_grpo.training.patches.vllm", vllm_patch_mod)

    config = SeedPaperEvalConfig(
        python_executable=tmp_path / "python",
        workspace_dir=tmp_path / "workspace",
        seed_repo_dir=tmp_path / "seed",
        dataset_dir=tmp_path / "seed" / "datasets" / "evaluation_suite",
        results_dir=tmp_path / "results",
        requirements_file=tmp_path / "requirements.txt",
        seed_repo_url="https://example.com/seed.git",
        seed_repo_commit="deadbeef",
        model_name="Qwen/Qwen2.5-Math-1.5B",
        template="no",
        tasks=("aime",),
        temperature=0.0,
        top_p=1.0,
        max_tokens=3000,
        max_model_len=4096,
        vllm_url="http://127.0.0.1:8000/generate",
        vllm_batch_size=32,
        vllm_use_rollout_token_guard=False,
        vllm_stop_sequences=None,
        n_samples=1,
        max_test=2,
        prompt_start=None,
        prompt_end=None,
        save_outputs=False,
        auto_install=True,
        prepare_only=False,
        use_srun=False,
        srun_args=(),
        expected_profile=None,
        expected_tolerance=1e-6,
        enforce_expected=False,
        runtime_enforce_eager=False,
        runtime_disable_async_output_proc=False,
        runtime_gpu_memory_utilization=None,
        runtime_swap_space=None,
        wandb_enabled=False,
        wandb_project=None,
        wandb_entity=None,
        wandb_group=None,
        wandb_run_name=None,
        wandb_job_type=None,
        pass_at_8_enabled=True,
        pass_at_8_samples=8,
        pass_at_8_temperature=1.0,
        pass_at_8_top_p=1.0,
    )
    summary = _run_vllm_server_eval(config)
    assert len(captured_calls) == 2
    assert captured_calls[0]["n"] == 1
    assert captured_calls[1]["n"] == 8
    assert summary["results"] == {"aime": 0.5}
    assert summary["pass_at_8"] == {"aime": 1.0}
    assert summary["pass_at_8_avg"] == 1.0
    assert summary["mean_at_8"] == {"aime": 0.5625}
    assert summary["mean_at_8_avg"] == 0.5625


def test_run_vllm_server_eval_saves_single_and_pass_at_8_outputs(
    monkeypatch, tmp_path: Path
) -> None:
    class DummyTokenizer:
        pass

    def _fake_from_pretrained(model_name: str):
        assert model_name == "Qwen/Qwen2.5-Math-1.5B"
        return DummyTokenizer()

    def _fake_logprob(token_count: int, logprob_sum: float, token_ids: list[int]):
        return SimpleNamespace(
            token_count=token_count,
            logprob_sum=logprob_sum,
            raw_output={"token_ids": token_ids},
        )

    def _fake_safe_generate(**kwargs):
        n = int(kwargs["n"])
        want_logprobs = bool(kwargs.get("return_logprobs"))
        if n == 1:
            logs = [[_fake_logprob(3, -1.25, [1, 2, 3])]] if want_logprobs else None
            return [[r"\boxed{1}"]], logs, 1.0
        if n == 8:
            logs = (
                [[_fake_logprob(2, -0.5 - idx, [idx, idx + 1]) for idx in range(8)]]
                if want_logprobs
                else None
            )
            return [[r"\boxed{1}"] * 4 + [r"\boxed{0}"] * 4], logs, 1.0
        raise AssertionError(f"unexpected n={n}")

    class FakeDataset(dict):
        pass

    def _fake_load_from_disk(path: str):
        del path
        return FakeDataset(
            {
                "aime": {
                    "problem": ["What is 1?"],
                    "answer": ["1"],
                }
            }
        )

    def _fake_reward_fn(model_output, gt, fast=False):
        del fast
        return {"formatted": True}, float(model_output == rf"\boxed{{{gt}}}")

    import sys

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoTokenizer=SimpleNamespace(from_pretrained=_fake_from_pretrained)
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "datasets",
        SimpleNamespace(load_from_disk=_fake_load_from_disk),
    )
    understand_pkg = ModuleType("understand_r1_zero")
    grader_mod = ModuleType("understand_r1_zero.math_grader")
    grader_mod.answer_tag_reward_fn = _fake_reward_fn
    grader_mod.answer_tag_reward_fn_for_orz = _fake_reward_fn
    grader_mod.boxed_reward_fn = _fake_reward_fn
    training_pkg = ModuleType("maxent_grpo.training")
    patches_pkg = ModuleType("maxent_grpo.training.patches")
    vllm_patch_mod = ModuleType("maxent_grpo.training.patches.vllm")
    vllm_patch_mod.safe_generate = _fake_safe_generate
    patches_pkg.vllm = vllm_patch_mod
    training_pkg.patches = patches_pkg
    monkeypatch.setitem(sys.modules, "understand_r1_zero", understand_pkg)
    monkeypatch.setitem(sys.modules, "understand_r1_zero.math_grader", grader_mod)
    monkeypatch.setitem(sys.modules, "maxent_grpo.training", training_pkg)
    monkeypatch.setitem(sys.modules, "maxent_grpo.training.patches", patches_pkg)
    monkeypatch.setitem(sys.modules, "maxent_grpo.training.patches.vllm", vllm_patch_mod)

    results_dir = tmp_path / "results"
    config = SeedPaperEvalConfig(
        python_executable=tmp_path / "python",
        workspace_dir=tmp_path / "workspace",
        seed_repo_dir=tmp_path / "seed",
        dataset_dir=tmp_path / "seed" / "datasets" / "evaluation_suite",
        results_dir=results_dir,
        requirements_file=tmp_path / "requirements.txt",
        seed_repo_url="https://example.com/seed.git",
        seed_repo_commit="deadbeef",
        model_name="Qwen/Qwen2.5-Math-1.5B",
        template="no",
        tasks=("aime",),
        temperature=0.0,
        top_p=1.0,
        max_tokens=3000,
        max_model_len=4096,
        vllm_url="http://127.0.0.1:8000/generate",
        vllm_batch_size=32,
        vllm_use_rollout_token_guard=False,
        vllm_stop_sequences=None,
        n_samples=1,
        max_test=1,
        prompt_start=None,
        prompt_end=None,
        save_outputs=True,
        auto_install=True,
        prepare_only=False,
        use_srun=False,
        srun_args=(),
        expected_profile=None,
        expected_tolerance=1e-6,
        enforce_expected=False,
        runtime_enforce_eager=False,
        runtime_disable_async_output_proc=False,
        runtime_gpu_memory_utilization=None,
        runtime_swap_space=None,
        wandb_enabled=False,
        wandb_project=None,
        wandb_entity=None,
        wandb_group=None,
        wandb_run_name=None,
        wandb_job_type=None,
        pass_at_8_enabled=True,
        pass_at_8_samples=8,
        pass_at_8_temperature=1.0,
        pass_at_8_top_p=1.0,
    )
    summary = _run_vllm_server_eval(config)
    paths = summary.get("saved_output_paths")
    assert isinstance(paths, dict)
    single_path = Path(paths["single"])
    pass8_path = Path(paths["pass_at_8"])
    assert single_path.exists()
    assert pass8_path.exists()
    single_payload = json.loads(single_path.read_text(encoding="utf-8"))
    pass8_payload = json.loads(pass8_path.read_text(encoding="utf-8"))
    assert single_payload[0]["samples"][0]["logprob_sum"] == -1.25
    assert single_payload[0]["samples"][0]["token_ids"] == [1, 2, 3]
    assert len(pass8_payload[0]["samples"]) == 8
    assert pass8_payload[0]["samples"][0]["token_count"] == 2
