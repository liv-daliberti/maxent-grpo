# Copyright 2025 Liv d'Aliberti
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""LightEval task registration and Slurm launch utilities.

This module provides helpers to:

- Define a compact string specification per benchmark task and register common
  tasks in a dictionary consumable by launchers.
- Compute the proper vLLM Slurm submission command and spawn evaluations as
  jobs using ``subprocess.run``.

It also exposes ``SUPPORTED_BENCHMARKS`` and convenience functions to list
registered tasks. vLLM launch on Slurm requires a specific environment
bootstrap (see ``VLLM_SLURM_PREFIX``) to source system profiles and set ``$HOME``.
"""


from __future__ import annotations

import base64
import os
import subprocess
from typing import TYPE_CHECKING, Dict, List, Literal

from .hub import get_gpu_count_for_vllm, get_param_count_from_repo_id


if TYPE_CHECKING:
    from trl import GRPOConfig, ModelConfig


# We need a special environment setup to launch vLLM from within Slurm training jobs.
# - Reference code: https://github.com/huggingface/brrr/blob/c55ba3505686d690de24c7ace6487a5c1426c0fd/brrr/lighteval/one_job_runner.py#L105
# - Slack thread: https://huggingface.slack.com/archives/C043JTYE1MJ/p1726566494958269
user_home_directory: str = os.path.expanduser("~")
VLLM_SLURM_PREFIX: List[str] = [
    "env",
    "-i",
    "bash",
    "-c",
    f"for f in /etc/profile.d/*.sh; do source $f; done; export HOME={user_home_directory}; mkdir -p logs; sbatch ",
]

# Type aliases for task configuration
TaskSpec = str  # e.g. "lighteval|math_500|0|0"
TaskName = str  # e.g. "math_500"
TaskName = str
TaskSpec = str
TaskSuite = Literal["lighteval", "extended"]
BenchmarkKey = Literal["bbh", "mt-bench", "TruthfulQA"]


def register_lighteval_task(
    configs: Dict[TaskName, TaskSpec],
    eval_suite: TaskSuite,
    task_name: TaskName,
    task_list: str,
    num_fewshot: int = 0,
) -> None:
    """Register a LightEval task configuration in ``configs``.

    - Core tasks table: https://github.com/huggingface/lighteval/blob/main/src/lighteval/tasks/tasks_table.jsonl
    - Custom tasks should live under your project (ops/scripts/evaluation/...).

    :param configs: Mapping where the serialized task spec is stored.
    :type configs: dict[str, str]
    :param eval_suite: Suite prefix, e.g. ``"lighteval"`` or ``"extended"``.
    :type eval_suite: str
    :param task_name: Key to store the task under.
    :type task_name: str
    :param task_list: Comma‑separated list of tasks, without suite prefix.
    :type task_list: str
    :param num_fewshot: Number of few‑shot examples per task.
    :type num_fewshot: int
    :returns: None
    :rtype: None
    """
    # Format task list in lighteval format
    task_list = ",".join(
        f"{eval_suite}|{task}|{num_fewshot}|0" for task in task_list.split(",")
    )
    configs[task_name] = task_list


LIGHTEVAL_TASKS: Dict[TaskName, TaskSpec] = {}

register_lighteval_task(LIGHTEVAL_TASKS, "lighteval", "math_500", "math_500", 0)
register_lighteval_task(LIGHTEVAL_TASKS, "lighteval", "aime24", "aime24", 0)
register_lighteval_task(LIGHTEVAL_TASKS, "lighteval", "aime25", "aime25", 0)
register_lighteval_task(LIGHTEVAL_TASKS, "lighteval", "gpqa", "gpqa:diamond", 0)
register_lighteval_task(LIGHTEVAL_TASKS, "extended", "lcb", "lcb:codegeneration", 0)
register_lighteval_task(
    LIGHTEVAL_TASKS, "extended", "lcb_v4", "lcb:codegeneration_v4", 0
)


def get_lighteval_tasks() -> List[TaskName]:
    """Return the list of registered LightEval task names.

    :returns: Available benchmark keys.
    :rtype: list[str]
    """
    return list(LIGHTEVAL_TASKS.keys())


SUPPORTED_BENCHMARKS = get_lighteval_tasks()


def _build_slurm_gpu_flag(num_gpus: int) -> List[str]:
    """Return sbatch GPU flag(s) based on env policy.

    The behaviour is controlled by ``SLURM_GPU_FLAG_STYLE``:

    - ``\"none\"`` (default): do not add a GPU flag; rely on script headers.
    - ``\"gpus\"``: append ``--gpus={num_gpus}``.
    - ``\"gres\"``: append ``--gres=gpu:{num_gpus}``.

    :param num_gpus: Requested number of GPUs for the evaluation job.
    :type num_gpus: int
    :returns: List of flag strings suitable for ``sbatch``.
    :rtype: list[str]
    """
    style = os.getenv("SLURM_GPU_FLAG_STYLE", "none").lower()
    if style == "gpus":
        return [f"--gpus={num_gpus}"]
    if style == "gres":
        return [f"--gres=gpu:{num_gpus}"]
    return []


def run_lighteval_job(
    benchmark: TaskName,
    training_args: "GRPOConfig",
    model_args: "ModelConfig",
) -> None:
    """Launch a LightEval job under Slurm with vLLM decoding.

    :param benchmark: Registered benchmark key.
    :type benchmark: str
    :param training_args: Training configuration containing Hub model info.
    :type training_args: GRPOConfig
    :param model_args: Model configuration (trust flags).
    :type model_args: ModelConfig
    :returns: None
    :rtype: None
    """
    task_list = LIGHTEVAL_TASKS[benchmark]
    model_name = training_args.hub_model_id
    model_revision = training_args.hub_model_revision
    # For large models >= 30b params or those running the MATH benchmark, we need to shard them across the GPUs to avoid OOM
    num_gpus = get_gpu_count_for_vllm(model_name, model_revision)
    if get_param_count_from_repo_id(model_name) >= 30_000_000_000:
        tensor_parallel = True
    else:
        num_gpus = 2  # Hack while cluster is full
        tensor_parallel = False

    cmd = VLLM_SLURM_PREFIX.copy()
    gpu_flags = _build_slurm_gpu_flag(num_gpus)
    cmd_args = [
        *gpu_flags,
        f"--job-name=or1_{benchmark}_{model_name.split('/')[-1]}_{model_revision}",
        "ops/slurm/evaluate.slurm",
        benchmark,
        f'"{task_list}"',
        model_name,
        model_revision,
        f"{tensor_parallel}",
        f"{model_args.trust_remote_code}",
    ]
    if training_args.system_prompt is not None:
        # encode to base64 to avoid issues with special characters
        # we decode in the sbatch script
        prompt_encoded = base64.b64encode(training_args.system_prompt.encode()).decode()
        cmd_args.append(prompt_encoded)
    cmd[-1] += " " + " ".join(cmd_args)
    subprocess.run(cmd, check=True)


def run_benchmark_jobs(training_args: "GRPOConfig", model_args: "ModelConfig") -> None:
    """Launch one or more benchmarks as Slurm jobs.

    :param training_args: Training configuration (reads ``benchmarks`` list).
    :type training_args: GRPOConfig
    :param model_args: Model configuration.
    :type model_args: ModelConfig
    :returns: None
    :rtype: None
    :raises ValueError: If an unknown benchmark name is supplied.
    """
    benchmarks = training_args.benchmarks
    if len(benchmarks) == 1 and benchmarks[0] == "all":
        benchmarks = get_lighteval_tasks()
        # Evaluate on all supported benchmarks. Later we may want to include a `chat` option
        # that just evaluates on `ifeval` and `mt_bench` etc.

    for benchmark in benchmarks:
        print(f"Launching benchmark `{benchmark}`")
        if benchmark in get_lighteval_tasks():
            run_lighteval_job(benchmark, training_args, model_args)
        else:
            raise ValueError(f"Unknown benchmark {benchmark}")
