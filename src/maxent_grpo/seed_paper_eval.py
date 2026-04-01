"""Run the official SEED-GRPO paper evaluation from inside this repo.

This wrapper is intentionally thin. It bootstraps a pinned checkout of the
official SEED-GRPO repository under ``var/``, installs the small set of
missing grader CLI dependencies into a repo-local Python environment, and then
invokes the official ``evaluate_model.py`` script with the paper settings.
"""

from __future__ import annotations

import argparse
import ast
import fcntl
import json
import logging
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence
import re

from maxent_grpo.prompt_templates import (
    apply_no_template,
    apply_qwen_math_template,
    apply_r1_template,
    resolve_generation_stop_settings,
)

SEED_GRPO_REPO_URL = "https://github.com/Dreamer312/SEED-GRPO.git"
SEED_GRPO_REPO_COMMIT = "325cb1a20bb60f8efd4cdc77a1565491c29fd289"
DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-Math-1.5B"
DEFAULT_TEMPLATE = "qwen_math"
DEFAULT_TASKS = ("aime", "amc", "math", "minerva", "olympiad_bench")
DEFAULT_MAX_TOKENS = 3000
DEFAULT_MAX_MODEL_LEN = 4096
DEFAULT_VLLM_BATCH_SIZE = 32
DEFAULT_PASS_AT_8_SAMPLES = 8
DEFAULT_PASS_AT_8_TEMPERATURE = 1.0
DEFAULT_PASS_AT_8_TOP_P = 1.0
DEFAULT_REQUIRED_MODULES = (
    "datasets",
    "fire",
    "jinja2",
    "latex2sympy2_extended",
    "math_verify",
    "numpy",
    "pylatexenc",
    "sympy",
    "vllm",
)
EXPECTED_RESULT_PROFILES: dict[str, dict[str, float]] = {
    "table2_qwen2_5_math_1_5b": {
        "aime": 5.0 / 30.0,
        "amc": 36.0 / 83.0,
        "math": 309.0 / 500.0,
        "minerva": 41.0 / 272.0,
        "olympiad_bench": 192.0 / 675.0,
        "avg": (
            (5.0 / 30.0)
            + (36.0 / 83.0)
            + (309.0 / 500.0)
            + (41.0 / 272.0)
            + (192.0 / 675.0)
        )
        / 5.0,
    }
}
EXPECTED_PROFILE_DEFAULT_TEMPLATES: dict[str, str] = {
    # The published Table 2 baseline for Qwen2.5-Math-1.5B matches the
    # official SEED eval only when the base model is evaluated as a plain
    # completion model, not with the qwen_math chat wrapper.
    "table2_qwen2_5_math_1_5b": "no",
}
EXPECTED_RUNTIME_REQUIREMENTS: dict[str, str] = {
    "datasets": "2.16.1",
    "fire": "0.7.0",
    "latex2sympy2_extended": "1.10.1",
    "math_verify": "0.7.0",
    "sympy": "1.13.1",
    "torch": "2.5.1+cu121",
    "transformers": "4.48.2",
    "vllm": "0.7.2",
}
RUNTIME_PROBE_PREFIX = "SEED_PAPER_EVAL_RUNTIME_PROBE="
RUNTIME_PROBE_DISTRIBUTIONS: dict[str, str] = {
    "math_verify": "math-verify",
}


@dataclass(frozen=True)
class SeedPaperEvalConfig:
    python_executable: Path
    workspace_dir: Path
    seed_repo_dir: Path
    dataset_dir: Path
    results_dir: Path
    requirements_file: Path
    seed_repo_url: str
    seed_repo_commit: str
    model_name: str
    template: str
    tasks: tuple[str, ...]
    temperature: float
    top_p: float
    max_tokens: int
    max_model_len: int
    vllm_url: str | None
    vllm_batch_size: int
    vllm_use_rollout_token_guard: bool
    vllm_stop_sequences: tuple[str, ...] | None
    n_samples: int
    max_test: int
    prompt_start: int | None
    prompt_end: int | None
    save_outputs: bool
    auto_install: bool
    prepare_only: bool
    use_srun: bool
    srun_args: tuple[str, ...]
    expected_profile: str | None
    expected_tolerance: float
    enforce_expected: bool
    runtime_enforce_eager: bool
    runtime_disable_async_output_proc: bool
    runtime_gpu_memory_utilization: float | None
    runtime_swap_space: float | None
    wandb_enabled: bool
    wandb_project: str | None
    wandb_entity: str | None
    wandb_group: str | None
    wandb_run_name: str | None
    wandb_job_type: str | None
    seed_paper_reward_fast: bool = False
    pass_at_8_enabled: bool = False
    pass_at_8_samples: int = DEFAULT_PASS_AT_8_SAMPLES
    pass_at_8_temperature: float = DEFAULT_PASS_AT_8_TEMPERATURE
    pass_at_8_top_p: float = DEFAULT_PASS_AT_8_TOP_P


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def absolute_path(path: Path) -> Path:
    """Return an absolute path without resolving symlinks.

    Python venv interpreters are typically symlinks into a base interpreter.
    Resolving them would silently escape the venv, which breaks reproducibility.
    """
    return path.expanduser().absolute()


def default_workspace_dir(root: Path | None = None) -> Path:
    base = root or repo_root()
    return base / "var" / "seed_paper_eval"


def seed_repo_dir_for_workspace(workspace_dir: Path) -> Path:
    return workspace_dir / "external" / "SEED-GRPO"


def default_seed_repo_dir(root: Path | None = None) -> Path:
    return seed_repo_dir_for_workspace(default_workspace_dir(root))


def results_dir_for_workspace(workspace_dir: Path) -> Path:
    return workspace_dir / "results"


def default_results_dir(root: Path | None = None) -> Path:
    return results_dir_for_workspace(default_workspace_dir(root))


def default_requirements_file(root: Path | None = None) -> Path:
    base = root or repo_root()
    return base / "tools" / "requirements-seed-paper-eval.txt"


def default_python_executable(root: Path | None = None) -> Path:
    base = root or repo_root()
    candidates = (
        base / "var" / "seed_paper_eval" / "paper_venv" / "bin" / "python",
        base / "var" / "e2e-venv" / "bin" / "python",
        base / "var" / "openr1" / "bin" / "python",
        Path(sys.executable),
    )
    for candidate in candidates:
        if candidate.exists() and os.access(candidate, os.X_OK):
            return candidate
    raise FileNotFoundError(
        "No usable Python interpreter found. Pass --python explicitly."
    )


def parse_task_list(raw: str | None) -> tuple[str, ...]:
    if raw is None or not raw.strip():
        return DEFAULT_TASKS
    tasks = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not tasks:
        raise ValueError("At least one task is required when --tasks is provided.")
    return tasks


def parse_bool_flag(raw: str | None, *, default: bool = False) -> bool:
    if raw is None:
        return default
    lowered = raw.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    return default


def resolve_template(
    raw_template: str | None,
    expected_profile: str | None,
) -> str:
    if raw_template is not None:
        return raw_template
    if expected_profile is not None:
        profile_default = EXPECTED_PROFILE_DEFAULT_TEMPLATES.get(expected_profile)
        if profile_default is not None:
            return profile_default
    return DEFAULT_TEMPLATE


def _parse_stop_sequences_arg(raw: object) -> tuple[str, ...] | None:
    if raw is None:
        return None
    if isinstance(raw, (list, tuple)):
        cleaned = [str(item) for item in raw if item is not None and str(item).strip()]
        return tuple(cleaned) if cleaned else None
    if isinstance(raw, str):
        stripped = raw.strip()
        if not stripped:
            return None
        parsed_val: object | None
        try:
            parsed_val = json.loads(stripped)
        except (TypeError, ValueError):
            parsed_val = None
        if isinstance(parsed_val, list):
            cleaned = [
                str(item) for item in parsed_val if item is not None and str(item).strip()
            ]
            return tuple(cleaned) if cleaned else None
        if isinstance(parsed_val, str):
            stripped = parsed_val.strip()
        if "||" in stripped:
            parts = [part.strip() for part in stripped.split("||")]
            cleaned = [part for part in parts if part]
            return tuple(cleaned) if cleaned else None
        return (stripped,)
    return (str(raw),)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the official SEED-GRPO evaluate_model.py script with the paper "
            "settings from inside this repository."
        )
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument(
        "--template",
        help=(
            "Prompt template forwarded to the official eval. Defaults to "
            "qwen_math, except paper profiles may resolve to a different "
            "template when that is what reproduces the published baseline."
        ),
    )
    parser.add_argument(
        "--tasks",
        help=(
            "Comma-separated task list. Defaults to the Table 2 suite: "
            "aime,amc,math,minerva,olympiad_bench."
        ),
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        help=(
            "Override the evaluation dataset directory. Defaults to the official "
            "SEED evaluation suite under the checked-out repo."
        ),
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--max-model-len", type=int, default=DEFAULT_MAX_MODEL_LEN)
    parser.add_argument(
        "--vllm-url",
        help=(
            "Optional live vLLM /generate endpoint. When set, the wrapper runs "
            "the official dataset/template/scorer path against the served model "
            "instead of instantiating a standalone vLLM LLM process."
        ),
    )
    parser.add_argument(
        "--vllm-batch-size",
        type=int,
        default=DEFAULT_VLLM_BATCH_SIZE,
        help="Prompt batch size used when --vllm-url is evaluating a live server.",
    )
    parser.add_argument(
        "--vllm-use-rollout-token-guard",
        action="store_true",
        help=(
            "When --vllm-url is set, forward the same tokenizer/model token-boundary "
            "guarding used by rollout generation."
        ),
    )
    parser.add_argument(
        "--vllm-stop-sequences",
        help=(
            "Optional stop sequences forwarded to the live vLLM server. Accepts a "
            "JSON list or '||'-delimited string to match training config parsing."
        ),
    )
    parser.add_argument(
        "--seed-paper-reward-fast",
        action="store_true",
        help=(
            "Use the OAT/SEED fast verifier path for answer grading instead of "
            "the slower math_verify-style fallback."
        ),
    )
    parser.add_argument("--n-samples", type=int, default=1)
    parser.add_argument("--max-test", type=int, default=999999)
    parser.add_argument(
        "--prompt-start",
        type=int,
        help=(
            "Optional inclusive prompt start index within each selected task. "
            "Used for prompt-range subshards."
        ),
    )
    parser.add_argument(
        "--prompt-end",
        type=int,
        help=(
            "Optional exclusive prompt end index within each selected task. "
            "Used for prompt-range subshards."
        ),
    )
    parser.add_argument(
        "--pass-at-8",
        action="store_true",
        help=(
            "Also run a sampled n=8-style SEED eval pass and report pass_at_8 / "
            "mean_at_8 alongside the default single-sample results."
        ),
    )
    parser.add_argument(
        "--pass-at-8-samples",
        type=int,
        default=DEFAULT_PASS_AT_8_SAMPLES,
        help="Number of samples used for the sampled pass_at_8 / mean_at_8 eval.",
    )
    parser.add_argument(
        "--pass-at-8-temperature",
        type=float,
        default=DEFAULT_PASS_AT_8_TEMPERATURE,
        help="Sampling temperature used for the sampled pass_at_8 / mean_at_8 eval.",
    )
    parser.add_argument(
        "--pass-at-8-top-p",
        type=float,
        default=DEFAULT_PASS_AT_8_TOP_P,
        help="Top-p used for the sampled pass_at_8 / mean_at_8 eval.",
    )
    parser.add_argument(
        "--save-outputs",
        action="store_true",
        help="Forward save=True to the official script so it stores raw generations.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Clone the official repo and install missing eval deps, then stop.",
    )
    parser.add_argument(
        "--no-install",
        action="store_true",
        help="Do not auto-install missing grader dependencies.",
    )
    parser.add_argument(
        "--python",
        type=Path,
        help=(
            "Python interpreter to use for the official eval. Defaults to "
            "var/e2e-venv/bin/python when present."
        ),
    )
    parser.add_argument(
        "--workspace-dir",
        type=Path,
        help="Workspace root for the official checkout and logs.",
    )
    parser.add_argument(
        "--seed-repo-dir",
        type=Path,
        help="Existing SEED-GRPO checkout to reuse instead of the default var path.",
    )
    parser.add_argument(
        "--seed-repo-url",
        default=SEED_GRPO_REPO_URL,
        help="Official SEED-GRPO repository URL.",
    )
    parser.add_argument(
        "--seed-repo-commit",
        default=SEED_GRPO_REPO_COMMIT,
        help="Pinned SEED-GRPO commit to checkout for reproducible paper parity.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        help="Directory for wrapper metadata and logs. Defaults under var/seed_paper_eval.",
    )
    parser.add_argument(
        "--requirements-file",
        type=Path,
        help="Pinned dependency file for the official scorer wrapper.",
    )
    parser.add_argument(
        "--srun",
        action="store_true",
        help="Launch the official eval under srun so it runs on a GPU node.",
    )
    parser.add_argument(
        "--srun-args",
        default="--partition=all --gres=gpu:a6000:1 --cpus-per-task=4 --mem=24G",
        help=(
            "Extra args passed to srun when --srun is enabled. Time limits can be "
            "added here. The official SEED script hardcodes bfloat16, so the GPU "
            "must have compute capability >= 8.0. Example: --srun-args='--partition=all "
            "--gres=gpu:a6000:1 --cpus-per-task=4 --mem=24G --time=08:00:00'."
        ),
    )
    parser.add_argument(
        "--expected-profile",
        choices=tuple(sorted(EXPECTED_RESULT_PROFILES)),
        help=(
            "Optional expected benchmark profile to compare against after the "
            "official eval finishes."
        ),
    )
    parser.add_argument(
        "--expected-tolerance",
        type=float,
        default=1e-6,
        help="Absolute tolerance used when comparing against --expected-profile.",
    )
    parser.add_argument(
        "--enforce-expected",
        action="store_true",
        help="Exit non-zero when the observed metrics do not match --expected-profile.",
    )
    parser.add_argument(
        "--runtime-enforce-eager",
        action="store_true",
        help=(
            "Pass enforce_eager=True into the official SEED vLLM constructor. "
            "This is a runtime workaround for CUDA graph warmup hangs and does "
            "not change prompt, dataset, scorer, or decode settings."
        ),
    )
    parser.add_argument(
        "--runtime-disable-async-output-proc",
        action="store_true",
        help=(
            "Pass disable_async_output_proc=True into the official SEED vLLM "
            "constructor as a runtime-only workaround."
        ),
    )
    parser.add_argument(
        "--runtime-gpu-memory-utilization",
        type=float,
        help=(
            "Override the official vLLM gpu_memory_utilization as a runtime-only "
            "workaround. Lower values reduce KV-cache allocation size without "
            "changing prompt, scorer, or decode settings."
        ),
    )
    parser.add_argument(
        "--runtime-swap-space",
        type=float,
        help=(
            "Override the official vLLM swap_space (GiB) as a runtime-only "
            "workaround. Lower values reduce CPU KV-cache allocation size."
        ),
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        default=parse_bool_flag(
            os.environ.get("MAXENT_STEP0_PAPER_EVAL_WANDB_ENABLED"),
            default=False,
        ),
        help="Log the Step 0 paper eval summary to Weights & Biases.",
    )
    parser.add_argument(
        "--wandb-project",
        default=(
            os.environ.get("MAXENT_STEP0_PAPER_EVAL_WANDB_PROJECT")
            or os.environ.get("WANDB_PROJECT")
        ),
        help="Weights & Biases project for the Step 0 eval run.",
    )
    parser.add_argument(
        "--wandb-entity",
        default=(
            os.environ.get("MAXENT_STEP0_PAPER_EVAL_WANDB_ENTITY")
            or os.environ.get("WANDB_ENTITY")
        ),
        help="Weights & Biases entity for the Step 0 eval run.",
    )
    parser.add_argument(
        "--wandb-group",
        default=(
            os.environ.get("MAXENT_STEP0_PAPER_EVAL_WANDB_GROUP")
            or os.environ.get("WANDB_RUN_GROUP")
        ),
        help="Weights & Biases group for the Step 0 eval run.",
    )
    parser.add_argument(
        "--wandb-run-name",
        default=os.environ.get("MAXENT_STEP0_PAPER_EVAL_WANDB_RUN_NAME"),
        help="Explicit Weights & Biases run name for the Step 0 eval run.",
    )
    parser.add_argument(
        "--wandb-job-type",
        default=os.environ.get("MAXENT_STEP0_PAPER_EVAL_WANDB_JOB_TYPE")
        or "step0_paper_eval",
        help="Weights & Biases job type for the Step 0 eval run.",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> SeedPaperEvalConfig:
    args = build_parser().parse_args(argv)
    if int(args.pass_at_8_samples) < 1:
        raise ValueError("--pass-at-8-samples must be >= 1.")
    if args.prompt_start is not None and int(args.prompt_start) < 0:
        raise ValueError("--prompt-start must be >= 0 when provided.")
    if args.prompt_end is not None and int(args.prompt_end) < 0:
        raise ValueError("--prompt-end must be >= 0 when provided.")
    if (
        args.prompt_start is not None
        and args.prompt_end is not None
        and int(args.prompt_end) < int(args.prompt_start)
    ):
        raise ValueError("--prompt-end must be >= --prompt-start.")
    vllm_stop_sequences = _parse_stop_sequences_arg(args.vllm_stop_sequences)
    root = repo_root()
    workspace_dir = (
        args.workspace_dir.resolve()
        if args.workspace_dir is not None
        else default_workspace_dir(root)
    )
    seed_repo_dir = (
        args.seed_repo_dir.resolve()
        if args.seed_repo_dir is not None
        else seed_repo_dir_for_workspace(workspace_dir)
    )
    results_dir = (
        args.results_dir.resolve()
        if args.results_dir is not None
        else results_dir_for_workspace(workspace_dir)
    )
    python_executable = (
        absolute_path(args.python)
        if args.python is not None
        else default_python_executable(root)
    )
    requirements_file = (
        args.requirements_file.resolve()
        if args.requirements_file is not None
        else default_requirements_file(root)
    )
    return SeedPaperEvalConfig(
        python_executable=python_executable,
        workspace_dir=workspace_dir,
        seed_repo_dir=seed_repo_dir,
        dataset_dir=(
            args.dataset_dir.resolve()
            if args.dataset_dir is not None
            else seed_repo_dir / "datasets" / "evaluation_suite"
        ),
        results_dir=results_dir,
        requirements_file=requirements_file,
        seed_repo_url=args.seed_repo_url,
        seed_repo_commit=args.seed_repo_commit,
        model_name=args.model_name,
        template=resolve_template(args.template, args.expected_profile),
        tasks=parse_task_list(args.tasks),
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        max_model_len=args.max_model_len,
        vllm_url=str(args.vllm_url).strip() if args.vllm_url else None,
        vllm_batch_size=int(args.vllm_batch_size),
        vllm_use_rollout_token_guard=bool(args.vllm_use_rollout_token_guard),
        vllm_stop_sequences=vllm_stop_sequences,
        n_samples=args.n_samples,
        max_test=args.max_test,
        prompt_start=(
            int(args.prompt_start) if args.prompt_start is not None else None
        ),
        prompt_end=int(args.prompt_end) if args.prompt_end is not None else None,
        save_outputs=bool(args.save_outputs),
        auto_install=not args.no_install,
        prepare_only=bool(args.prepare_only),
        use_srun=bool(args.srun),
        srun_args=tuple(shlex.split(args.srun_args)),
        expected_profile=args.expected_profile,
        expected_tolerance=float(args.expected_tolerance),
        enforce_expected=bool(args.enforce_expected),
        runtime_enforce_eager=bool(args.runtime_enforce_eager),
        runtime_disable_async_output_proc=bool(args.runtime_disable_async_output_proc),
        runtime_gpu_memory_utilization=(
            float(args.runtime_gpu_memory_utilization)
            if args.runtime_gpu_memory_utilization is not None
            else None
        ),
        runtime_swap_space=(
            float(args.runtime_swap_space)
            if args.runtime_swap_space is not None
            else None
        ),
        wandb_enabled=bool(args.wandb),
        wandb_project=str(args.wandb_project) if args.wandb_project else None,
        wandb_entity=str(args.wandb_entity) if args.wandb_entity else None,
        wandb_group=str(args.wandb_group) if args.wandb_group else None,
        wandb_run_name=str(args.wandb_run_name) if args.wandb_run_name else None,
        wandb_job_type=str(args.wandb_job_type) if args.wandb_job_type else None,
        seed_paper_reward_fast=bool(args.seed_paper_reward_fast),
        pass_at_8_enabled=bool(args.pass_at_8),
        pass_at_8_samples=int(args.pass_at_8_samples),
        pass_at_8_temperature=float(args.pass_at_8_temperature),
        pass_at_8_top_p=float(args.pass_at_8_top_p),
    )


def shell_join(parts: Iterable[object]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def apply_prime_zero_template(question: str) -> str:
    question = question + "\n\nPresent the answer in LaTex format: \\boxed{Your answer}"
    return (
        "A conversation between User and Assistant. The user asks a question, and "
        "the Assistant solves it. The assistant first thinks about the reasoning "
        "process in the mind and then provides the user with the answer. The "
        "reasoning process and answer are enclosed within <think> </think> and "
        "<answer> </answer> tags, respectively, i.e., <think> reasoning process "
        "here </think> <answer> answer here </answer>. User: "
        f"{question}. Assistant:"
    )


def apply_open_reasoner_zero_template(question: str) -> str:
    from jinja2 import Template

    prompt_template_jinja = """\
{{bos_token}}A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. \
The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. User: {{prompt}}
Assistant: <think>\
"""
    prompt_instruction_template_jinja = """\
You must put your answer inside <answer> </answer> tags, i.e., <answer> answer here </answer>. And your final answer will be extracted automatically by the \\boxed{} tag.
This is the problem:
{{prompt}}
"""
    prompt_instruction_template = Template(prompt_instruction_template_jinja)
    prompt_instruction = prompt_instruction_template.render(prompt=question)
    prompt_template = Template(prompt_template_jinja)
    return prompt_template.render(bos_token="", prompt=prompt_instruction)


def child_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "0"
    env["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
    env.setdefault("MAXENT_SEED_PAPER_SUPPRESS_GRADER_LOGS", "1")
    # The launcher environment may prepend unrelated site-packages onto
    # PYTHONPATH. The official eval must run only against the pinned paper env.
    env.pop("PYTHONPATH", None)
    return env


def eval_env(config: SeedPaperEvalConfig) -> dict[str, str]:
    env = child_env()
    cache_root = config.workspace_dir / "hf_cache"
    hub_cache = cache_root / "hub"
    datasets_cache = cache_root / "datasets"
    transformers_cache = cache_root / "transformers"
    xet_cache = cache_root / "xet"
    vllm_cache_root = config.workspace_dir / "vllm_cache"
    xdg_cache_home = config.workspace_dir / "xdg_cache"
    xdg_config_home = config.workspace_dir / "xdg_config"
    torchinductor_cache = config.workspace_dir / "torchinductor_cache"
    for path in (
        cache_root,
        hub_cache,
        datasets_cache,
        transformers_cache,
        xet_cache,
        vllm_cache_root,
        xdg_cache_home,
        xdg_config_home,
        torchinductor_cache,
    ):
        path.mkdir(parents=True, exist_ok=True)
    env["HF_HOME"] = str(cache_root)
    env["HF_HUB_CACHE"] = str(hub_cache)
    env["HUGGINGFACE_HUB_CACHE"] = str(hub_cache)
    env["HF_DATASETS_CACHE"] = str(datasets_cache)
    env["TRANSFORMERS_CACHE"] = str(transformers_cache)
    env["HF_XET_CACHE"] = str(xet_cache)
    env["VLLM_CACHE_ROOT"] = str(vllm_cache_root)
    env["VLLM_NO_USAGE_STATS"] = "1"
    env["XDG_CACHE_HOME"] = str(xdg_cache_home)
    env["XDG_CONFIG_HOME"] = str(xdg_config_home)
    env["TORCHINDUCTOR_CACHE_DIR"] = str(torchinductor_cache)
    return env


def run_checked(
    cmd: Sequence[object],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    capture_output: bool = False,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(part) for part in cmd],
        cwd=str(cwd) if cwd is not None else None,
        env=env,
        check=True,
        text=True,
        capture_output=capture_output,
    )


def ensure_seed_repo(config: SeedPaperEvalConfig) -> None:
    config.workspace_dir.mkdir(parents=True, exist_ok=True)
    lock_path = config.workspace_dir / ".seed_repo.lock"
    with lock_path.open("w", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        config.seed_repo_dir.parent.mkdir(parents=True, exist_ok=True)
        if not config.seed_repo_dir.exists():
            run_checked(["git", "clone", config.seed_repo_url, config.seed_repo_dir])
        current_url = run_checked(
            ["git", "-C", config.seed_repo_dir, "remote", "get-url", "origin"],
            capture_output=True,
        ).stdout.strip()
        if current_url != config.seed_repo_url:
            raise RuntimeError(
                f"Existing checkout at {config.seed_repo_dir} points to {current_url}, "
                f"expected {config.seed_repo_url}."
            )
        run_checked(["git", "-C", config.seed_repo_dir, "fetch", "origin", "--tags"])
        run_checked(
            [
                "git",
                "-C",
                config.seed_repo_dir,
                "checkout",
                "--detach",
                config.seed_repo_commit,
            ]
        )
        eval_script = config.seed_repo_dir / "evaluate_model.py"
        if not config.dataset_dir.exists():
            raise FileNotFoundError(f"Official evaluation suite missing at {config.dataset_dir}")
        if not eval_script.exists():
            raise FileNotFoundError(f"Official eval script missing at {eval_script}")


def missing_modules(
    python_executable: Path,
    required_modules: Sequence[str] = DEFAULT_REQUIRED_MODULES,
) -> tuple[str, ...]:
    probe = (
        "import importlib.util, json; "
        f"mods={list(required_modules)!r}; "
        "missing=[name for name in mods if importlib.util.find_spec(name) is None]; "
        "print(json.dumps(missing))"
    )
    completed = run_checked(
        [python_executable, "-c", probe],
        env=child_env(),
        capture_output=True,
    )
    return tuple(json.loads(completed.stdout.strip() or "[]"))


def probe_runtime_stack(config: SeedPaperEvalConfig) -> dict[str, dict[str, str | None]]:
    module_names = tuple(sorted(EXPECTED_RUNTIME_REQUIREMENTS))
    probe = """
import importlib
import importlib.metadata
import json
mods = %s
dist_names = %s
payload = {}
for name in mods:
    module = importlib.import_module(name)
    version = getattr(module, "__version__", None)
    if version is None:
        try:
            version = importlib.metadata.version(dist_names.get(name, name))
        except importlib.metadata.PackageNotFoundError:
            version = None
    payload[name] = {
        "version": version,
        "file": getattr(module, "__file__", None),
    }
print(%r + json.dumps(payload, sort_keys=True))
""" % (
        repr(module_names),
        repr(RUNTIME_PROBE_DISTRIBUTIONS),
        RUNTIME_PROBE_PREFIX,
    )
    completed = run_checked(
        [config.python_executable, "-c", probe],
        env=eval_env(config),
        capture_output=True,
    )
    payload_line = None
    for raw_line in completed.stdout.splitlines():
        line = raw_line.strip()
        if line.startswith(RUNTIME_PROBE_PREFIX):
            payload_line = line[len(RUNTIME_PROBE_PREFIX) :]
    if payload_line is None:
        raise RuntimeError(
            "Runtime probe did not emit a parsable payload. Stdout was:\n"
            f"{completed.stdout}"
        )
    data = json.loads(payload_line)
    out: dict[str, dict[str, str | None]] = {}
    for name in module_names:
        entry = data.get(name, {})
        out[name] = {
            "version": entry.get("version"),
            "file": entry.get("file"),
        }
    return out


def validate_runtime_stack(
    config: SeedPaperEvalConfig,
    runtime_probe: dict[str, dict[str, str | None]],
) -> list[dict[str, str | None]]:
    mismatches: list[dict[str, str | None]] = []
    venv_root = config.python_executable.parent.parent.resolve()
    for module_name, expected_version in EXPECTED_RUNTIME_REQUIREMENTS.items():
        observed = runtime_probe.get(module_name, {})
        observed_version = observed.get("version")
        observed_file = observed.get("file")
        if observed_version != expected_version:
            mismatches.append(
                {
                    "module": module_name,
                    "kind": "version",
                    "expected": expected_version,
                    "observed": observed_version,
                    "file": observed_file,
                }
            )
        if observed_file is None:
            mismatches.append(
                {
                    "module": module_name,
                    "kind": "path",
                    "expected": str(venv_root),
                    "observed": None,
                    "file": None,
                }
            )
            continue
        try:
            module_path = Path(observed_file).resolve()
        except OSError:
            module_path = Path(observed_file)
        if not module_path.is_relative_to(venv_root):
            mismatches.append(
                {
                    "module": module_name,
                    "kind": "path",
                    "expected": str(venv_root),
                    "observed": str(module_path),
                    "file": str(module_path),
                }
            )
    return mismatches


def install_missing_dependencies(config: SeedPaperEvalConfig) -> None:
    if not config.requirements_file.exists():
        raise FileNotFoundError(
            f"Pinned requirements file not found: {config.requirements_file}"
        )
    run_checked(
        [config.python_executable, "-m", "pip", "install", "-r", config.requirements_file],
        env=child_env(),
    )


def gpu_is_available(python_executable: Path) -> bool:
    probe = (
        "import json, torch; "
        "print(json.dumps({'available': bool(torch.cuda.is_available()), "
        "'count': int(torch.cuda.device_count())}))"
    )
    completed = run_checked(
        [python_executable, "-c", probe],
        env=child_env(),
        capture_output=True,
    )
    payload = json.loads(completed.stdout.strip())
    return bool(payload["available"]) and int(payload["count"]) > 0


def build_official_eval_command(config: SeedPaperEvalConfig) -> list[str]:
    command = [
        str(config.python_executable),
        "-u",
        str(config.seed_repo_dir / "evaluate_model.py"),
        f"--model_name={config.model_name}",
        f"--template={config.template}",
        f"--dataset_name={config.dataset_dir}",
        f"--temperature={config.temperature}",
        f"--top_p={config.top_p}",
        f"--max_tokens={config.max_tokens}",
        f"--max_model_len={config.max_model_len}",
        f"--n_samples={config.n_samples}",
        f"--max_test={config.max_test}",
    ]
    if config.pass_at_8_enabled:
        command.extend(
            [
                "--pass_at_8=True",
                f"--pass_at_8_samples={config.pass_at_8_samples}",
                f"--pass_at_8_temperature={config.pass_at_8_temperature}",
                f"--pass_at_8_top_p={config.pass_at_8_top_p}",
            ]
        )
    if config.tasks != DEFAULT_TASKS:
        command.append(f"--tasks={list(config.tasks)!r}")
    if config.save_outputs:
        command.append("--save=True")
    if config.runtime_enforce_eager:
        command.append("--enforce_eager=True")
    if config.runtime_disable_async_output_proc:
        command.append("--disable_async_output_proc=True")
    if config.runtime_gpu_memory_utilization is not None:
        command.append(
            f"--gpu_memory_utilization={config.runtime_gpu_memory_utilization}"
        )
    if config.runtime_swap_space is not None:
        command.append(f"--swap_space={config.runtime_swap_space}")
    return command


def build_launch_command(config: SeedPaperEvalConfig) -> list[str]:
    official_command = build_official_eval_command(config)
    if not config.use_srun:
        return official_command
    return ["srun", *config.srun_args, *official_command]


def _parse_literal_dict(line: str) -> dict[str, object] | None:
    stripped = line.strip()
    if not (stripped.startswith("{") and stripped.endswith("}")):
        return None
    scalar_wrapper = re.compile(
        r"\bnp\.(?:float16|float32|float64|int8|int16|int32|int64)\(([^()]+)\)"
    )
    previous = None
    while stripped != previous:
        previous = stripped
        stripped = scalar_wrapper.sub(r"\1", stripped)
    try:
        parsed = ast.literal_eval(stripped)
    except (SyntaxError, ValueError):
        return None
    if not isinstance(parsed, dict):
        return None
    out: dict[str, object] = {}
    for key, value in parsed.items():
        out[str(key)] = value
    return out


def parse_official_eval_summary(log_text: str) -> dict[str, object]:
    results: dict[str, float] = {}
    avg: float | None = None
    pass_at_8: dict[str, float] = {}
    pass_at_8_avg: float | None = None
    mean_at_8: dict[str, float] = {}
    mean_at_8_avg: float | None = None
    avg_lens: dict[str, float] = {}
    max_lens: dict[str, float] = {}
    formatted: dict[str, float] = {}
    for raw_line in log_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("{") and any(
            token in line for token in ("aime", "amc", "math", "minerva", "olympiad_bench")
        ):
            parsed = _parse_literal_dict(line)
            if parsed is None:
                continue
            maybe_results: dict[str, float] = {}
            for key, value in parsed.items():
                try:
                    maybe_results[str(key)] = float(value)
                except (TypeError, ValueError):
                    maybe_results = {}
                    break
            if maybe_results:
                results = maybe_results
                continue
        if line.startswith("avg:"):
            try:
                avg = float(line.split(":", 1)[1].strip())
            except (TypeError, ValueError):
                pass
            continue
        if line.startswith("pass_at_8:"):
            parsed = _parse_literal_dict(line.split(":", 1)[1].strip())
            if parsed is not None:
                pass_at_8 = {str(key): float(value) for key, value in parsed.items()}
            continue
        if line.startswith("pass_at_8_avg:"):
            try:
                pass_at_8_avg = float(line.split(":", 1)[1].strip())
            except (TypeError, ValueError):
                pass
            continue
        if line.startswith("mean_at_8:"):
            parsed = _parse_literal_dict(line.split(":", 1)[1].strip())
            if parsed is not None:
                mean_at_8 = {str(key): float(value) for key, value in parsed.items()}
            continue
        if line.startswith("mean_at_8_avg:"):
            try:
                mean_at_8_avg = float(line.split(":", 1)[1].strip())
            except (TypeError, ValueError):
                pass
            continue
        if line.startswith("avg_lens:"):
            parsed = _parse_literal_dict(line.split(":", 1)[1].strip())
            if parsed is not None:
                avg_lens = {
                    str(key): float(value)
                    for key, value in parsed.items()
                }
            continue
        if line.startswith("max_lens:"):
            parsed = _parse_literal_dict(line.split(":", 1)[1].strip())
            if parsed is not None:
                max_lens = {
                    str(key): float(value)
                    for key, value in parsed.items()
                }
            continue
        if line.startswith("formatted:"):
            parsed = _parse_literal_dict(line.split(":", 1)[1].strip())
            if parsed is not None:
                formatted = {
                    str(key): float(value)
                    for key, value in parsed.items()
                }
            continue
    return {
        "results": results,
        "avg": avg,
        "pass_at_8": pass_at_8,
        "pass_at_8_avg": pass_at_8_avg,
        "mean_at_8": mean_at_8,
        "mean_at_8_avg": mean_at_8_avg,
        "avg_lens": avg_lens,
        "max_lens": max_lens,
        "formatted": formatted,
    }


def build_seed_paper_eval_payload(
    summary: dict[str, object],
    *,
    prefix: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    results = summary.get("results", {})
    if isinstance(results, dict):
        for metric, value in results.items():
            try:
                payload[f"{prefix}/{metric}"] = float(value)
            except (TypeError, ValueError):
                continue
    avg_value = summary.get("avg")
    if avg_value is not None:
        try:
            payload[f"{prefix}/avg"] = float(avg_value)
        except (TypeError, ValueError):
            pass
    pass_at_8 = summary.get("pass_at_8", {})
    if isinstance(pass_at_8, dict):
        for metric, value in pass_at_8.items():
            try:
                payload[f"{prefix}/pass_at_8/{metric}"] = float(value)
            except (TypeError, ValueError):
                continue
    pass_at_8_avg = summary.get("pass_at_8_avg")
    if pass_at_8_avg is not None:
        try:
            payload[f"{prefix}/pass_at_8_avg"] = float(pass_at_8_avg)
        except (TypeError, ValueError):
            pass
    mean_at_8 = summary.get("mean_at_8", {})
    if isinstance(mean_at_8, dict):
        for metric, value in mean_at_8.items():
            try:
                payload[f"{prefix}/mean_at_8/{metric}"] = float(value)
            except (TypeError, ValueError):
                continue
    mean_at_8_avg = summary.get("mean_at_8_avg")
    if mean_at_8_avg is not None:
        try:
            payload[f"{prefix}/mean_at_8_avg"] = float(mean_at_8_avg)
        except (TypeError, ValueError):
            pass
    exit_code = summary.get("process_exit_code")
    if exit_code is not None:
        try:
            payload[f"{prefix}/process_exit_code"] = int(exit_code)
        except (TypeError, ValueError):
            pass
    comparison = summary.get("expected_comparison")
    if isinstance(comparison, dict):
        payload[f"{prefix}/expected_ok"] = bool(comparison.get("ok"))
        payload[f"{prefix}/expected_tolerance"] = float(comparison.get("tolerance", 0.0))
        compared = comparison.get("compared", {})
        if isinstance(compared, dict):
            for metric, entry in compared.items():
                if not isinstance(entry, dict):
                    continue
                expected_value = entry.get("expected")
                observed_value = entry.get("observed")
                delta_value = entry.get("delta")
                if expected_value is not None:
                    payload[f"{prefix}_expected/{metric}/expected"] = float(expected_value)
                if observed_value is not None:
                    payload[f"{prefix}_expected/{metric}/observed"] = float(observed_value)
                if delta_value is not None:
                    payload[f"{prefix}_expected/{metric}/delta"] = float(delta_value)
        mismatches = comparison.get("mismatches", [])
        if isinstance(mismatches, list):
            payload[f"{prefix}/expected_mismatch_count"] = len(mismatches)
    return payload


def build_step0_wandb_payload(summary: dict[str, object]) -> dict[str, Any]:
    return build_seed_paper_eval_payload(summary, prefix="step0_paper_eval")


def _ensure_seed_repo_on_sys_path(seed_repo_dir: Path) -> None:
    seed_repo_str = str(seed_repo_dir)
    if seed_repo_str not in sys.path:
        sys.path.insert(0, seed_repo_str)


def _suppress_math_verify_grader_logs() -> None:
    raw = os.environ.get("MAXENT_SEED_PAPER_SUPPRESS_GRADER_LOGS", "1")
    if str(raw).strip().lower() in {"0", "false", "no", "off"}:
        return
    for logger_name in ("math_verify", "math_verify.grader"):
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.CRITICAL)
        logger.propagate = False


def _resolve_template_runtime(
    config: SeedPaperEvalConfig,
) -> tuple[Callable[[str], str], Callable[[str, object, bool], tuple[dict[str, object], float]]]:
    _ensure_seed_repo_on_sys_path(config.seed_repo_dir)
    _suppress_math_verify_grader_logs()
    from understand_r1_zero.math_grader import (
        answer_tag_reward_fn,
        answer_tag_reward_fn_for_orz,
        boxed_reward_fn,
    )

    template = config.template
    if "prime" in config.model_name.lower():
        template = "prime-zero"
    if "open-reasoner-zero" in config.model_name.lower():
        template = "open-reasoner-zero"

    print("Using template:", template, flush=True)
    if template in {"qwen_math", "no"}:
        reward_fn = boxed_reward_fn
        apply_template = (
            apply_qwen_math_template if template == "qwen_math" else apply_no_template
        )
        return apply_template, reward_fn
    if template == "r1":
        return apply_r1_template, answer_tag_reward_fn
    if template == "prime-zero":
        return apply_prime_zero_template, boxed_reward_fn
    if template == "open-reasoner-zero":
        return apply_open_reasoner_zero_template, answer_tag_reward_fn_for_orz
    if template == "llama-instruct":
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(config.model_name)

        def _apply_template(question: str) -> str:
            return tokenizer.apply_chat_template(
                [
                    {
                        "content": (
                            f"{question}\nPlease reason step by step, and put your "
                            "final answer within \\boxed{{}}.\n\n"
                        ),
                        "role": "user",
                    }
                ],
                tokenize=False,
                add_generation_prompt=True,
            )

        return _apply_template, boxed_reward_fn
    if template == "r1d":
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(config.model_name)

        def _apply_template(question: str) -> str:
            return tokenizer.apply_chat_template(
                [{"content": question, "role": "user"}],
                tokenize=False,
                add_generation_prompt=True,
            )

        return _apply_template, boxed_reward_fn
    raise ValueError(f"Unsupported template: {template}")


def _token_count_from_logprob_entry(logprob_entry: object) -> int | None:
    """Recover a true token count from vLLM metadata when available."""

    if logprob_entry is None:
        return None
    direct_count = getattr(logprob_entry, "token_count", None)
    if direct_count is not None:
        try:
            return int(direct_count)
        except (TypeError, ValueError):
            pass
    raw_output = getattr(logprob_entry, "raw_output", None)
    if isinstance(raw_output, dict):
        raw_count = raw_output.get("token_count") or raw_output.get("num_tokens")
        if raw_count is not None:
            try:
                return int(raw_count)
            except (TypeError, ValueError):
                pass
        token_ids = raw_output.get("token_ids") or raw_output.get("output_token_ids")
        if isinstance(token_ids, list):
            return int(len(token_ids))
    return None


def _build_vllm_eval_generate_controls(
    config: SeedPaperEvalConfig,
    tokenizer: Any,
) -> dict[str, Any]:
    controls: dict[str, Any] = {}
    template_stops, template_include_stop = resolve_generation_stop_settings(
        config.template
    )
    stop_sequences = list(config.vllm_stop_sequences) if config.vllm_stop_sequences else None
    if stop_sequences is None and template_stops:
        stop_sequences = list(template_stops)
    if stop_sequences:
        controls["stop"] = stop_sequences
    if template_include_stop:
        controls["include_stop_str_in_output"] = True
    if not config.vllm_use_rollout_token_guard:
        return controls

    from transformers import AutoConfig

    try:
        tokenizer_limit = max(
            int(getattr(tokenizer, "vocab_size", 0) or 0),
            int(len(tokenizer)),
        )
    except Exception:
        tokenizer_limit = int(getattr(tokenizer, "vocab_size", 0) or 0)
    if tokenizer_limit <= 0:
        return controls

    try:
        model_config = AutoConfig.from_pretrained(config.model_name, trust_remote_code=True)
        model_limit = int(getattr(model_config, "vocab_size", 0) or 0)
    except Exception as exc:
        print(
            f"Warning: failed to resolve model vocab limit for rollout token guard: {exc}",
            flush=True,
        )
        return controls
    if model_limit <= tokenizer_limit:
        return controls

    blocked_token_ids = list(range(int(tokenizer_limit), int(model_limit)))
    controls["allowed_token_ids"] = list(range(int(tokenizer_limit)))
    controls["blocked_token_ids"] = blocked_token_ids
    print(
        "Using rollout-style vLLM token guard "
        f"(tokenizer_limit={tokenizer_limit}, model_limit={model_limit}, "
        f"blocked_tail={len(blocked_token_ids)}).",
        flush=True,
    )
    return controls


def _run_vllm_server_eval(config: SeedPaperEvalConfig) -> dict[str, object]:
    from datasets import load_from_disk
    import numpy as np
    from transformers import AutoTokenizer
    from maxent_grpo.training.patches.vllm import safe_generate

    if not config.vllm_url:
        raise ValueError("vLLM server eval requires config.vllm_url to be set.")
    if config.save_outputs:
        config.results_dir.mkdir(parents=True, exist_ok=True)
    apply_template, reward_fn = _resolve_template_runtime(config)
    eval_tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    vllm_generate_controls = _build_vllm_eval_generate_controls(config, eval_tokenizer)
    if vllm_generate_controls.get("stop"):
        print(
            f"Using vLLM stop sequences: {list(vllm_generate_controls['stop'])}",
            flush=True,
        )

    print("Loading official SEED evaluation suite from:", config.dataset_dir, flush=True)
    datasets_by_task = load_from_disk(str(config.dataset_dir))
    results: dict[str, float] = {}
    pass_at_8: dict[str, float] = {}
    mean_at_8: dict[str, float] = {}
    avg_lens: dict[str, float] = {}
    max_lens: dict[str, float] = {}
    formatted: dict[str, float] = {}
    task_prompt_counts: dict[str, int] = {}
    task_prompt_ranges: dict[str, dict[str, int]] = {}
    requested_tasks = set(config.tasks)
    saved_single_outputs: list[dict[str, object]] = []
    saved_pass_at_8_outputs: list[dict[str, object]] = []

    for task_name, dataset in datasets_by_task.items():
        if task_name not in requested_tasks:
            continue
        full_prompts_raw = list(dataset["problem"][: config.max_test])
        full_targets = list(dataset["answer"][: config.max_test])
        shard_start = int(config.prompt_start or 0)
        shard_end = (
            min(int(config.prompt_end), len(full_prompts_raw))
            if config.prompt_end is not None
            else len(full_prompts_raw)
        )
        if shard_end < shard_start:
            raise ValueError(
                f"Invalid prompt range for {task_name}: "
                f"start={shard_start} end={shard_end}"
            )
        prompts_raw = full_prompts_raw[shard_start:shard_end]
        targets = full_targets[shard_start:shard_end]
        task_prompt_counts[task_name] = len(prompts_raw)
        task_prompt_ranges[task_name] = {
            "start": shard_start,
            "end": shard_end,
            "available": len(full_prompts_raw),
        }
        prompts = [apply_template(prompt) for prompt in prompts_raw]
        print(
            f"inference for {task_name} prompts[{shard_start}:{shard_end}] "
            f"of {len(full_prompts_raw)}",
            flush=True,
        )
        batch_scores: list[float] = []
        batch_formatted: list[float] = []
        batch_lengths: list[list[int]] = []

        for start in range(0, len(prompts), config.vllm_batch_size):
            prompt_batch = prompts[start : start + config.vllm_batch_size]
            target_batch = targets[start : start + config.vllm_batch_size]
            grouped_outputs, logprob_groups, _ = safe_generate(
                prompts=prompt_batch,
                url=config.vllm_url,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                n=config.n_samples,
                tokenizer=eval_tokenizer,
                return_logprobs=True,
                timeout=max(600.0, float(config.max_tokens)),
                **vllm_generate_controls,
            )
            if len(grouped_outputs) != len(prompt_batch):
                raise RuntimeError(
                    "vLLM server returned a mismatched number of prompt groups: "
                    f"expected {len(prompt_batch)} got {len(grouped_outputs)}"
                )
            for row_idx, outputs_for_prompt in enumerate(grouped_outputs):
                rewards: list[float] = []
                infos: list[dict[str, object]] = []
                lengths: list[int] = []
                gt = target_batch[row_idx]
                logprobs_for_prompt = None
                if logprob_groups is not None and row_idx < len(logprob_groups):
                    logprobs_for_prompt = logprob_groups[row_idx]
                for sample_idx, model_output in enumerate(outputs_for_prompt):
                    info, reward = reward_fn(
                        model_output,
                        gt,
                        fast=bool(config.seed_paper_reward_fast),
                    )
                    rewards.append(float(reward))
                    infos.append(info)
                    logprob_entry = None
                    if (
                        logprobs_for_prompt is not None
                        and sample_idx < len(logprobs_for_prompt)
                    ):
                        logprob_entry = logprobs_for_prompt[sample_idx]
                    token_count = _token_count_from_logprob_entry(logprob_entry)
                    if token_count is None:
                        token_count = len(str(model_output))
                    lengths.append(int(token_count))
                batch_lengths.append(lengths)
                batch_scores.append(float(np.mean(rewards)))
                first_info = infos[0] if infos else {}
                if isinstance(first_info, dict) and first_info:
                    batch_formatted.append(
                        float(
                            np.sum(
                                [bool(info.get("formatted", False)) for info in infos]
                            )
                        )
                    )
                if config.save_outputs:
                    samples_payload: list[dict[str, object]] = []
                    for sample_idx, model_output in enumerate(outputs_for_prompt):
                        logprob_entry = None
                        if (
                            logprobs_for_prompt is not None
                            and sample_idx < len(logprobs_for_prompt)
                        ):
                            logprob_entry = logprobs_for_prompt[sample_idx]
                        raw_output = (
                            getattr(logprob_entry, "raw_output", None)
                            if logprob_entry is not None
                            else None
                        )
                        samples_payload.append(
                            {
                                "sample_index": int(sample_idx),
                                "text": str(model_output),
                                "reward": float(rewards[sample_idx]),
                                "formatted": bool(
                                    infos[sample_idx].get("formatted", False)
                                )
                                if sample_idx < len(infos)
                                and isinstance(infos[sample_idx], dict)
                                else False,
                                "token_count": int(lengths[sample_idx]),
                                "logprob_sum": (
                                    float(getattr(logprob_entry, "logprob_sum"))
                                    if logprob_entry is not None
                                    and getattr(logprob_entry, "logprob_sum", None)
                                    is not None
                                    else None
                                ),
                                "token_ids": (
                                    raw_output.get("token_ids")
                                    if isinstance(raw_output, dict)
                                    else None
                                ),
                            }
                        )
                    saved_single_outputs.append(
                        {
                            "task_name": str(task_name),
                            "prompt_index": int(shard_start + start + row_idx),
                            "prompt_raw": str(prompts_raw[start + row_idx]),
                            "prompt": str(prompt_batch[row_idx]),
                            "gt": str(gt),
                            "mode": "single",
                            "n_samples": int(config.n_samples),
                            "temperature": float(config.temperature),
                            "top_p": float(config.top_p),
                            "samples": samples_payload,
                        }
                    )

        if batch_scores:
            results[task_name] = float(np.mean(batch_scores))
        else:
            results[task_name] = 0.0
        if config.pass_at_8_enabled:
            sampled_pass_scores: list[float] = []
            sampled_mean_scores: list[float] = []
            for start in range(0, len(prompts), config.vllm_batch_size):
                prompt_batch = prompts[start : start + config.vllm_batch_size]
                target_batch = targets[start : start + config.vllm_batch_size]
                sampled_logprob_groups = None
                grouped_outputs, sampled_logprob_groups, _ = safe_generate(
                    prompts=prompt_batch,
                    url=config.vllm_url,
                    max_tokens=config.max_tokens,
                    temperature=config.pass_at_8_temperature,
                    top_p=config.pass_at_8_top_p,
                    n=config.pass_at_8_samples,
                    tokenizer=eval_tokenizer,
                    return_logprobs=bool(config.save_outputs),
                    timeout=max(600.0, float(config.max_tokens)),
                    **vllm_generate_controls,
                )
                if len(grouped_outputs) != len(prompt_batch):
                    raise RuntimeError(
                        "vLLM server returned a mismatched number of sampled prompt groups: "
                        f"expected {len(prompt_batch)} got {len(grouped_outputs)}"
                    )
                for row_idx, outputs_for_prompt in enumerate(grouped_outputs):
                    rewards: list[float] = []
                    gt = target_batch[row_idx]
                    sampled_logprobs_for_prompt = None
                    if (
                        sampled_logprob_groups is not None
                        and row_idx < len(sampled_logprob_groups)
                    ):
                        sampled_logprobs_for_prompt = sampled_logprob_groups[row_idx]
                    for model_output in outputs_for_prompt:
                        _, reward = reward_fn(
                            model_output,
                            gt,
                            fast=bool(config.seed_paper_reward_fast),
                        )
                        rewards.append(float(reward))
                    if not rewards:
                        sampled_pass_scores.append(0.0)
                        sampled_mean_scores.append(0.0)
                        continue
                    sampled_pass_scores.append(float(max(rewards) > 0.0))
                    sampled_mean_scores.append(float(np.mean(rewards)))
                    if config.save_outputs:
                        sampled_payload: list[dict[str, object]] = []
                        for sample_idx, model_output in enumerate(outputs_for_prompt):
                            logprob_entry = None
                            if (
                                sampled_logprobs_for_prompt is not None
                                and sample_idx < len(sampled_logprobs_for_prompt)
                            ):
                                logprob_entry = sampled_logprobs_for_prompt[sample_idx]
                            raw_output = (
                                getattr(logprob_entry, "raw_output", None)
                                if logprob_entry is not None
                                else None
                            )
                            sampled_payload.append(
                                {
                                    "sample_index": int(sample_idx),
                                    "text": str(model_output),
                                    "reward": float(rewards[sample_idx]),
                                    "token_count": _token_count_from_logprob_entry(
                                        logprob_entry
                                    ),
                                    "logprob_sum": (
                                        float(getattr(logprob_entry, "logprob_sum"))
                                        if logprob_entry is not None
                                        and getattr(logprob_entry, "logprob_sum", None)
                                        is not None
                                        else None
                                    ),
                                    "token_ids": (
                                        raw_output.get("token_ids")
                                        if isinstance(raw_output, dict)
                                        else None
                                    ),
                                }
                            )
                        saved_pass_at_8_outputs.append(
                            {
                                "task_name": str(task_name),
                                "prompt_index": int(shard_start + start + row_idx),
                                "prompt_raw": str(prompts_raw[start + row_idx]),
                                "prompt": str(prompt_batch[row_idx]),
                                "gt": str(gt),
                                "mode": "pass_at_8",
                                "n_samples": int(config.pass_at_8_samples),
                                "temperature": float(config.pass_at_8_temperature),
                                "top_p": float(config.pass_at_8_top_p),
                                "samples": sampled_payload,
                            }
                        )
            pass_at_8[task_name] = (
                float(np.mean(sampled_pass_scores)) if sampled_pass_scores else 0.0
            )
            mean_at_8[task_name] = (
                float(np.mean(sampled_mean_scores)) if sampled_mean_scores else 0.0
            )
        if batch_lengths:
            avg_lens[task_name] = float(np.mean(batch_lengths))
            max_lens[task_name] = float(np.max(batch_lengths))
        else:
            avg_lens[task_name] = 0.0
            max_lens[task_name] = 0.0
        if batch_formatted:
            formatted[task_name] = float(np.mean(batch_formatted))

    avg = float(np.mean(list(results.values()))) if results else None
    pass_at_8_avg = float(np.mean(list(pass_at_8.values()))) if pass_at_8 else None
    mean_at_8_avg = float(np.mean(list(mean_at_8.values()))) if mean_at_8 else None
    summary = {
        "mode": "vllm_server",
        "results": results,
        "avg": avg,
        "pass_at_8": pass_at_8,
        "pass_at_8_avg": pass_at_8_avg,
        "mean_at_8": mean_at_8,
        "mean_at_8_avg": mean_at_8_avg,
        "avg_lens": avg_lens,
        "max_lens": max_lens,
        "formatted": formatted,
        "task_prompt_counts": task_prompt_counts,
        "task_prompt_ranges": task_prompt_ranges,
        "vllm_url": config.vllm_url,
        "vllm_batch_size": int(config.vllm_batch_size),
        "vllm_use_rollout_token_guard": bool(config.vllm_use_rollout_token_guard),
        "vllm_stop_sequences": list(config.vllm_stop_sequences)
        if config.vllm_stop_sequences is not None
        else None,
        "pass_at_8_config": {
            "enabled": bool(config.pass_at_8_enabled),
            "samples": int(config.pass_at_8_samples),
            "temperature": float(config.pass_at_8_temperature),
            "top_p": float(config.pass_at_8_top_p),
        },
    }
    if config.save_outputs:
        saved_paths: dict[str, str] = {}
        if saved_single_outputs:
            single_path = (
                config.results_dir
                / f"seed_paper_eval_outputs_single_n{int(config.n_samples)}.json"
            )
            single_path.write_text(
                json.dumps(saved_single_outputs, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            saved_paths["single"] = str(single_path)
            print(f"saved single outputs: {single_path}", flush=True)
        if saved_pass_at_8_outputs:
            pass8_path = (
                config.results_dir
                / f"seed_paper_eval_outputs_pass_at_8_n{int(config.pass_at_8_samples)}.json"
            )
            pass8_path.write_text(
                json.dumps(saved_pass_at_8_outputs, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            saved_paths["pass_at_8"] = str(pass8_path)
            print(f"saved pass@8 outputs: {pass8_path}", flush=True)
        if saved_paths:
            summary["saved_output_paths"] = saved_paths
    print(results, flush=True)
    if avg is not None:
        print("avg:", avg, flush=True)
    if pass_at_8:
        print("pass_at_8:", pass_at_8, flush=True)
    if pass_at_8_avg is not None:
        print("pass_at_8_avg:", pass_at_8_avg, flush=True)
    if mean_at_8:
        print("mean_at_8:", mean_at_8, flush=True)
    if mean_at_8_avg is not None:
        print("mean_at_8_avg:", mean_at_8_avg, flush=True)
    print("avg_lens:", avg_lens, flush=True)
    print("max_lens:", max_lens, flush=True)
    print("formatted:", formatted, flush=True)
    return summary


def compare_against_expected_profile(
    observed_summary: dict[str, object],
    profile_name: str,
    *,
    tolerance: float,
) -> dict[str, object]:
    if profile_name not in EXPECTED_RESULT_PROFILES:
        raise KeyError(f"Unknown expected profile: {profile_name}")
    expected = EXPECTED_RESULT_PROFILES[profile_name]
    observed_results = observed_summary.get("results", {})
    if not isinstance(observed_results, dict):
        observed_results = {}
    mismatches: list[dict[str, float | str]] = []
    compared: dict[str, dict[str, float]] = {}
    for key, expected_value in expected.items():
        if key == "avg" and not observed_summary.get("results"):
            continue
        if key == "avg":
            observed_value_raw = observed_summary.get("avg")
        else:
            observed_value_raw = observed_results.get(key)
        if observed_value_raw is None:
            mismatches.append(
                {
                    "metric": key,
                    "expected": float(expected_value),
                    "observed": float("nan"),
                    "delta": float("nan"),
                }
            )
            continue
        observed_value = float(observed_value_raw)
        delta = observed_value - float(expected_value)
        compared[key] = {
            "expected": float(expected_value),
            "observed": observed_value,
            "delta": delta,
        }
        if abs(delta) > tolerance:
            mismatches.append(
                {
                    "metric": key,
                    "expected": float(expected_value),
                    "observed": observed_value,
                    "delta": delta,
                }
            )
    return {
        "profile": profile_name,
        "tolerance": float(tolerance),
        "ok": not mismatches,
        "compared": compared,
        "mismatches": mismatches,
    }


def write_metadata(
    config: SeedPaperEvalConfig,
    *,
    command: Sequence[str],
    official_command: Sequence[str],
    log_path: Path,
    runtime_probe: dict[str, dict[str, str | None]],
) -> Path:
    config.results_dir.mkdir(parents=True, exist_ok=True)
    cache_root = config.workspace_dir / "hf_cache"
    vllm_cache_root = config.workspace_dir / "vllm_cache"
    metadata = {
        "seed_repo_url": config.seed_repo_url,
        "seed_repo_commit": config.seed_repo_commit,
        "seed_repo_dir": str(config.seed_repo_dir),
        "dataset_dir": str(config.dataset_dir),
        "python_executable": str(config.python_executable),
        "command": list(command),
        "official_command": list(official_command),
        "log_path": str(log_path),
        "model_name": config.model_name,
        "template": config.template,
        "tasks": list(config.tasks),
        "temperature": config.temperature,
        "top_p": config.top_p,
        "max_tokens": config.max_tokens,
        "max_model_len": config.max_model_len,
        "vllm_url": config.vllm_url,
        "vllm_batch_size": config.vllm_batch_size,
        "n_samples": config.n_samples,
        "max_test": config.max_test,
        "save_outputs": config.save_outputs,
        "pass_at_8_enabled": config.pass_at_8_enabled,
        "pass_at_8_samples": config.pass_at_8_samples,
        "pass_at_8_temperature": config.pass_at_8_temperature,
        "pass_at_8_top_p": config.pass_at_8_top_p,
        "use_srun": config.use_srun,
        "srun_args": list(config.srun_args),
        "expected_profile": config.expected_profile,
        "expected_tolerance": config.expected_tolerance,
        "enforce_expected": config.enforce_expected,
        "runtime_enforce_eager": config.runtime_enforce_eager,
        "runtime_disable_async_output_proc": config.runtime_disable_async_output_proc,
        "runtime_gpu_memory_utilization": config.runtime_gpu_memory_utilization,
        "runtime_swap_space": config.runtime_swap_space,
        "hf_cache_root": str(cache_root),
        "vllm_cache_root": str(vllm_cache_root),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_probe": runtime_probe,
        "expected_runtime_requirements": EXPECTED_RUNTIME_REQUIREMENTS,
    }
    metadata_path = log_path.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return metadata_path


def init_step0_wandb_run(
    config: SeedPaperEvalConfig,
    *,
    metadata_path: Path,
) -> Any | None:
    if not config.wandb_enabled:
        return None
    try:
        import wandb
    except ImportError:
        print(
            "Step 0 W&B logging requested but wandb is not installed in the wrapper "
            "environment; continuing without W&B.",
            flush=True,
        )
        return None
    init_kwargs: dict[str, Any] = {
        "project": config.wandb_project,
        "entity": config.wandb_entity,
        "group": config.wandb_group,
        "name": config.wandb_run_name,
        "job_type": config.wandb_job_type,
        "dir": os.environ.get("WANDB_DIR"),
        "reinit": True,
        "settings": wandb.Settings(
            init_timeout=float(os.environ.get("WANDB_INIT_TIMEOUT", "300")),
            start_method=os.environ.get("WANDB_START_METHOD") or "thread",
            x_service_wait=float(os.environ.get("WANDB__SERVICE_WAIT", "300")),
        ),
        "config": {
            "step0_paper_eval/model_name": config.model_name,
            "step0_paper_eval/template": config.template,
            "step0_paper_eval/tasks": list(config.tasks),
            "step0_paper_eval/expected_profile": config.expected_profile,
            "step0_paper_eval/metadata_path": str(metadata_path),
            "step0_paper_eval/seed_repo_commit": config.seed_repo_commit,
            "step0_paper_eval/pass_at_8_enabled": config.pass_at_8_enabled,
            "step0_paper_eval/pass_at_8_samples": config.pass_at_8_samples,
            "step0_paper_eval/pass_at_8_temperature": config.pass_at_8_temperature,
            "step0_paper_eval/pass_at_8_top_p": config.pass_at_8_top_p,
            "step0_paper_eval/runtime_swap_space": config.runtime_swap_space,
            "step0_paper_eval/runtime_enforce_eager": config.runtime_enforce_eager,
            "step0_paper_eval/runtime_disable_async_output_proc": (
                config.runtime_disable_async_output_proc
            ),
        },
    }
    init_kwargs = {
        key: value for key, value in init_kwargs.items() if value not in (None, "")
    }
    try:
        return wandb.init(**init_kwargs)
    except Exception as exc:
        print(
            f"Step 0 W&B init failed: {exc}. Continuing without W&B.",
            flush=True,
        )
        return None


def finalize_step0_wandb_run(
    run: Any | None,
    *,
    summary: dict[str, object] | None,
    metadata_path: Path,
    log_path: Path,
    summary_path: Path | None,
    error_message: str | None,
) -> None:
    if run is None:
        return
    try:
        for prefix in ("step0_paper_eval", "paper_eval"):
            run.summary[f"{prefix}/metadata_path"] = str(metadata_path)
            run.summary[f"{prefix}/log_path"] = str(log_path)
            run.summary[f"{prefix}/latest_step"] = 0
        run.summary["paper_eval/source"] = "step0"
        if summary_path is not None:
            run.summary["step0_paper_eval/summary_path"] = str(summary_path)
            run.summary["paper_eval/summary_path"] = str(summary_path)
        if summary is not None:
            payload: dict[str, Any] = {}
            payload.update(build_step0_wandb_payload(summary))
            payload.update(build_seed_paper_eval_payload(summary, prefix="paper_eval"))
            if payload:
                run.log(payload, step=0)
                for key, value in payload.items():
                    run.summary[key] = value
            run.summary["step0_paper_eval/status"] = "ok"
            run.summary["paper_eval/status"] = "ok"
            warning = summary.get("process_warning")
            if warning is not None:
                run.summary["step0_paper_eval/process_warning"] = str(warning)
                run.summary["paper_eval/process_warning"] = str(warning)
            comparison = summary.get("expected_comparison")
            if isinstance(comparison, dict) and not bool(comparison.get("ok")):
                run.summary["step0_paper_eval/status"] = "expected_mismatch"
                run.summary["paper_eval/status"] = "expected_mismatch"
        elif error_message is not None:
            run.summary["step0_paper_eval/status"] = "failed_before_summary"
            run.summary["step0_paper_eval/error"] = error_message
            run.summary["paper_eval/status"] = "failed_before_summary"
            run.summary["paper_eval/error"] = error_message
    except Exception as exc:
        print(
            f"Step 0 W&B logging failed: {exc}.",
            flush=True,
        )
    finally:
        try:
            run.finish(exit_code=0 if error_message is None else 1)
        except Exception:
            pass


def run_and_tee(
    command: Sequence[str],
    *,
    cwd: Path,
    env: dict[str, str],
    log_path: Path,
) -> tuple[int, str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    transcript_parts: list[str] = []
    with log_path.open("wb") as log_handle:
        process = subprocess.Popen(
            list(command),
            cwd=str(cwd),
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=False,
            bufsize=0,
        )
        if process.stdin is not None:
            try:
                process.stdin.write(b"y\n")
                process.stdin.flush()
            except BrokenPipeError:
                pass
            finally:
                try:
                    process.stdin.close()
                except Exception:
                    pass
        assert process.stdout is not None
        read_chunk = getattr(process.stdout, "read1", process.stdout.read)
        while True:
            chunk = read_chunk(8192)
            if not chunk:
                break
            text_chunk = chunk.decode("utf-8", errors="replace")
            print(text_chunk, end="")
            log_handle.write(chunk)
            log_handle.flush()
            transcript_parts.append(text_chunk)
        return process.wait(), "".join(transcript_parts)


def prepare_environment(config: SeedPaperEvalConfig) -> tuple[str, ...]:
    ensure_seed_repo(config)
    missing = missing_modules(config.python_executable)
    if missing and config.auto_install:
        install_missing_dependencies(config)
        missing = missing_modules(config.python_executable)
    return missing


def run_seed_paper_eval(config: SeedPaperEvalConfig) -> int:
    missing = prepare_environment(config)
    if missing:
        missing_list = ", ".join(missing)
        raise RuntimeError(
            "Official SEED eval dependencies are still missing from "
            f"{config.python_executable}: {missing_list}"
        )
    if config.prepare_only:
        print("SEED paper eval environment is ready.", flush=True)
        print(f"Official repo: {config.seed_repo_dir}", flush=True)
        print(f"Python: {config.python_executable}", flush=True)
        return 0
    if (
        config.vllm_url is None
        and not config.use_srun
        and not gpu_is_available(config.python_executable)
    ):
        raise RuntimeError(
            "No CUDA device is visible from this shell. Run the wrapper on a GPU "
            "node or pass --srun to launch the exact SEED eval through Slurm."
        )
    runtime_probe = probe_runtime_stack(config)
    runtime_mismatches = validate_runtime_stack(config, runtime_probe)
    if runtime_mismatches:
        mismatch_lines = [
            f"{entry['module']} {entry['kind']} expected={entry['expected']} "
            f"observed={entry['observed']}"
            for entry in runtime_mismatches
        ]
        raise RuntimeError(
            "Official SEED eval runtime stack does not match the pinned paper "
            "environment:\n- " + "\n- ".join(mismatch_lines)
        )
    official_command = build_official_eval_command(config)
    command = build_launch_command(config)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = config.results_dir / f"seed_paper_eval_{timestamp}.log"
    metadata_path = write_metadata(
        config,
        command=command,
        official_command=official_command,
        log_path=log_path,
        runtime_probe=runtime_probe,
    )
    wandb_run = init_step0_wandb_run(config, metadata_path=metadata_path)
    summary_path: Path | None = None
    summary: dict[str, object] | None = None
    error_message: str | None = None
    print("Resolved runtime stack:", flush=True)
    for module_name in sorted(runtime_probe):
        entry = runtime_probe[module_name]
        print(
            f"  {module_name}=={entry.get('version')} "
            f"({entry.get('file')})",
            flush=True,
        )
    print("Running official SEED paper eval command:", flush=True)
    print(shell_join(command), flush=True)
    print(f"Wrapper metadata: {metadata_path}", flush=True)
    try:
        if config.vllm_url is None:
            exit_code, transcript = run_and_tee(
                command,
                cwd=config.seed_repo_dir,
                env=eval_env(config),
                log_path=log_path,
            )
            summary = parse_official_eval_summary(transcript)
            if not summary.get("results") and log_path.exists():
                summary = parse_official_eval_summary(log_path.read_text(encoding="utf-8"))
            if not summary.get("results"):
                if exit_code != 0:
                    raise RuntimeError(
                        f"Official SEED eval exited with code {exit_code}. See {log_path}."
                    )
                raise RuntimeError(
                    "Official SEED eval completed but no benchmark summary was parsed "
                    f"from {log_path}."
                )
            summary["process_exit_code"] = exit_code
            if exit_code != 0:
                summary["process_warning"] = (
                    "Official SEED eval returned a non-zero exit code after printing a "
                    "benchmark summary."
                )
        else:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("w", encoding="utf-8") as log_handle:
                old_stdout = sys.stdout
                old_stderr = sys.stderr

                class _Tee:
                    def __init__(self, *streams: Any) -> None:
                        self._streams = streams

                    def write(self, data: str) -> int:
                        for stream in self._streams:
                            stream.write(data)
                            stream.flush()
                        return len(data)

                    def flush(self) -> None:
                        for stream in self._streams:
                            stream.flush()

                tee = _Tee(old_stdout, log_handle)
                sys.stdout = tee
                sys.stderr = tee
                try:
                    summary = _run_vllm_server_eval(config)
                finally:
                    sys.stdout = old_stdout
                    sys.stderr = old_stderr
            exit_code = 0
            summary["process_exit_code"] = 0
        if config.expected_profile is not None:
            summary["expected_comparison"] = compare_against_expected_profile(
                summary,
                config.expected_profile,
                tolerance=config.expected_tolerance,
            )
        summary_path = log_path.with_suffix(".summary.json")
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(f"Official eval log: {log_path}", flush=True)
        print(f"Official eval summary: {summary_path}", flush=True)
        if exit_code != 0:
            print(
                f"Official eval process exit code was {exit_code}, but a summary was "
                "captured.",
                flush=True,
            )
        comparison = summary.get("expected_comparison")
        if isinstance(comparison, dict):
            print(
                "Expected-profile comparison: "
                + ("ok" if bool(comparison.get("ok")) else "mismatch"),
                flush=True,
            )
            if config.enforce_expected and not bool(comparison.get("ok")):
                raise RuntimeError(
                    "Official SEED eval did not match the expected profile. "
                    f"See {summary_path}."
                )
        return 0
    except Exception as exc:
        error_message = str(exc)
        raise
    finally:
        finalize_step0_wandb_run(
            wandb_run,
            summary=summary,
            metadata_path=metadata_path,
            log_path=log_path,
            summary_path=summary_path,
            error_message=error_message,
        )


def main(argv: Sequence[str] | None = None) -> int:
    config = parse_args(argv)
    return run_seed_paper_eval(config)


if __name__ == "__main__":
    raise SystemExit(main())
