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

from __future__ import annotations

from dataclasses import dataclass
from argparse import Namespace
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

from maxent_grpo.generation.common import (
    AggregatedGenerationState,
    append_completion_group,
    determine_retry_limit,
    drop_empty_prompt_groups,
    flatten_ref_metadata,
    pending_generation_indices,
    retry_incomplete_prompts,
    seed_generation_groups,
    truncate_to_expected_counts,
)

if TYPE_CHECKING:  # pragma: no cover - hints only
    from distilabel.pipeline import Pipeline
    from maxent_grpo.training.types import GenerationBatch, PromptCompletionBatch


def flatten_prompt_completions(
    gen_batch: "GenerationBatch",
) -> Tuple["PromptCompletionBatch", List[str]]:
    """Return flattened prompt/completion pairs and aligned answers.

    :param gen_batch: Aggregated generation results.
    :type gen_batch: maxent_grpo.training.types.GenerationBatch
    :returns: Tuple of flattened prompt/completion batch and answer list.
    :rtype: tuple[maxent_grpo.training.types.PromptCompletionBatch, list[str]]
    """
    from maxent_grpo.training.types import (
        PromptCompletionBatch,
    )  # Local import to avoid cycles

    prompts: List[str] = []
    completions: List[str] = []
    answers: List[str] = []
    metadata: List[dict] = []
    info_groups = getattr(gen_batch, "grouped_completion_info", None) or []
    if info_groups and len(info_groups) < len(gen_batch.grouped_completions):
        # Pad with empty lists so zip does not drop prompts.
        info_groups = list(info_groups) + [
            [] for _ in range(len(gen_batch.grouped_completions) - len(info_groups))
        ]
    elif not info_groups:
        info_groups = [[] for _ in gen_batch.grouped_completions]
    for prompt_text, answer_text, comp_group, info_group in zip(
        gen_batch.prompts, gen_batch.answers, gen_batch.grouped_completions, info_groups
    ):
        info_group = info_group or []
        for idx, completion_text in enumerate(comp_group):
            prompts.append(prompt_text)
            completions.append(completion_text)
            answers.append(answer_text)
            if idx < len(info_group):
                metadata.append(info_group[idx])
            elif info_group:
                metadata.append({})
    if metadata and len(metadata) != len(completions):
        # Align lengths defensively; downstream consumers can treat empty dict
        # as "no metadata" without failing.
        target_len = min(len(completions), len(metadata))
        metadata = metadata[:target_len]
        completions = completions[:target_len]
        prompts = prompts[:target_len]
        answers = answers[:target_len]
    min_len = min(len(prompts), len(completions), len(answers))
    if min_len == 0:
        try:
            return PromptCompletionBatch([], []), []
        except TypeError:
            return PromptCompletionBatch([], [], None), []
    prompts = prompts[:min_len]
    completions = completions[:min_len]
    answers = answers[:min_len]
    metadata_out = metadata[:min_len] if metadata else None
    batch_args = (prompts, completions)
    if metadata_out:
        try:
            return (
                PromptCompletionBatch(*batch_args, metadata=metadata_out),
                answers,
            )
        except TypeError:
            return PromptCompletionBatch(*batch_args, metadata_out), answers
    return PromptCompletionBatch(*batch_args), answers


@dataclass
class DistilabelPipelineConfig:
    """Configuration for building a distilabel generation pipeline.

    :param model: Model identifier served by the OpenAI-compatible endpoint.
    :type model: str
    :param base_url: Base URL for the OpenAI-compatible endpoint.
    :type base_url: str
    :param prompt_column: Optional dataset column to render into the template.
    :type prompt_column: str | None
    :param prompt_template: Jinja template used to render prompts.
    :type prompt_template: str
    :param temperature: Sampling temperature forwarded to the backend.
    :type temperature: float | None
    :param top_p: Nucleus sampling probability mass.
    :type top_p: float | None
    :param max_new_tokens: Maximum generated tokens per completion.
    :type max_new_tokens: int
    :param num_generations: Number of completions per prompt.
    :type num_generations: int
    :param input_batch_size: Batch size used by distilabel when grouping inputs.
    :type input_batch_size: int
    :param client_replicas: Number of OpenAI client replicas to spin up.
    :type client_replicas: int
    :param timeout: Request timeout in seconds.
    :type timeout: int
    :param retries: Maximum retry attempts per request.
    :type retries: int
    """

    model: str
    base_url: str = "http://localhost:8000/v1"
    prompt_column: Optional[str] = None
    prompt_template: str = "{{ instruction }}"
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_new_tokens: int = 8192
    num_generations: int = 1
    input_batch_size: int = 64
    client_replicas: int = 1
    timeout: int = 900
    retries: int = 0


def build_distilabel_pipeline(
    cfg: Optional[DistilabelPipelineConfig] = None,
    **kwargs: Any,
) -> "Pipeline":
    """Create and return a distilabel Pipeline based on ``cfg``.

    :param cfg: Pipeline configuration. When ``None`` ``kwargs`` are used.
    :type cfg: DistilabelPipelineConfig | None
    :param kwargs: Keyword overrides used when ``cfg`` is omitted.
    :type kwargs: Any
    :returns: Configured distilabel ``Pipeline`` ready to run.
    :rtype: distilabel.pipeline.Pipeline
    :raises RuntimeError: When the ``distilabel`` package is unavailable.
    """
    if cfg is None:
        cfg = DistilabelPipelineConfig(**kwargs)
    generation_kwargs: Dict[str, float | int] = {"max_new_tokens": cfg.max_new_tokens}
    if cfg.temperature is not None:
        generation_kwargs["temperature"] = cfg.temperature
    if cfg.top_p is not None:
        generation_kwargs["top_p"] = cfg.top_p
    try:  # pragma: no cover - exercised with stubs
        import importlib

        _disti = importlib.import_module("distilabel")
        DL_Pipeline = _disti.pipeline.Pipeline
        DL_StepResources = _disti.steps.StepResources
        DL_TextGeneration = _disti.steps.tasks.TextGeneration
        DL_OpenAILLM = _disti.llms.OpenAILLM
    except (ImportError, AttributeError) as exc:  # pragma: no cover
        raise RuntimeError("distilabel is required to build the pipeline") from exc
    with DL_Pipeline().ray() as pipe:
        DL_TextGeneration(
            llm=DL_OpenAILLM(
                base_url=cfg.base_url,
                api_key="something",
                model=cfg.model,
                timeout=cfg.timeout,
                max_retries=cfg.retries,
                generation_kwargs=generation_kwargs,
            ),
            template=cfg.prompt_template,
            input_mappings=(
                {"instruction": cfg.prompt_column}
                if cfg.prompt_column is not None
                else {}
            ),
            input_batch_size=cfg.input_batch_size,
            num_generations=cfg.num_generations,
            group_generations=True,
            resources=DL_StepResources(replicas=cfg.client_replicas),
        )
    return pipe


def run_distilabel_cli(
    args: Namespace,
    pipeline_builder: Optional[Callable[[DistilabelPipelineConfig], "Pipeline"]] = None,
) -> object:
    """Deprecated wrapper that now delegates to :func:`run_generation_job`.

    The orchestration lives in :mod:`maxent_grpo.pipelines.generation.distilabel`;
    this wrapper remains for callers that imported it from the helpers module.

    :param args: Parsed CLI namespace from the distilabel entrypoint.
    :type args: argparse.Namespace
    :param pipeline_builder: Optional factory used to construct the pipeline.
    :type pipeline_builder: Callable[[DistilabelPipelineConfig], Pipeline] | None
    :returns: Result of ``run_generation_job``.
    :rtype: object
    """
    from maxent_grpo.pipelines.generation.distilabel import (
        DistilabelGenerationConfig,
        run_generation_job,
    )

    cfg = DistilabelGenerationConfig.from_namespace(args)
    return run_generation_job(cfg, builder=pipeline_builder)


__all__ = [
    "AggregatedGenerationState",
    "DistilabelPipelineConfig",
    "append_completion_group",
    "build_distilabel_pipeline",
    "determine_retry_limit",
    "drop_empty_prompt_groups",
    "flatten_prompt_completions",
    "flatten_ref_metadata",
    "pending_generation_indices",
    "run_distilabel_cli",
    "retry_incomplete_prompts",
    "seed_generation_groups",
    "truncate_to_expected_counts",
]
