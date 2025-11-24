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

if TYPE_CHECKING:  # pragma: no cover - hints only
    from distilabel.pipeline import Pipeline
    from maxent_grpo.training.types import GenerationBatch, PromptCompletionBatch

_DEFAULT_RETRY_LIMIT = 3


@dataclass
class AggregatedGenerationState:
    """Mutable container for grouped completions and optional metadata."""

    completions: List[List[str]]
    metadata: Optional[List[List[Optional[Any]]]] = None


def append_completion_group(
    grouped_comps: List[List[str]],
    grouped_meta: Optional[List[List[Optional[Any]]]],
    prompt_idx: int,
    completions: Optional[List[str]],
    meta_group: Optional[List[Optional[Any]]],
) -> Optional[List[List[Optional[Any]]]]:
    """Append completions (and metadata) for a specific prompt index.

    :param grouped_comps: Existing completion groups aligned per prompt.
    :type grouped_comps: list[list[str]]
    :param grouped_meta: Optional metadata buffers aligned with completions.
    :type grouped_meta: list[list[Any | None]] | None
    :param prompt_idx: Prompt index whose completions are being extended.
    :type prompt_idx: int
    :param completions: Newly generated completion strings.
    :type completions: list[str] | None
    :param meta_group: Optional metadata aligned with ``completions``.
    :type meta_group: list[Any | None] | None
    :returns: Updated metadata buffers (creating the structure if necessary).
    :rtype: list[list[Any | None]] | None
    """
    if not completions:
        return grouped_meta
    entries = list(completions)
    start = len(grouped_comps[prompt_idx])
    grouped_comps[prompt_idx].extend(entries)
    if meta_group is None:
        if grouped_meta is not None:
            grouped_meta[prompt_idx].extend([None] * len(entries))
        return grouped_meta
    if grouped_meta is None:
        grouped_meta = [[None] * len(group) for group in grouped_comps]
    meta_entries = list(meta_group)
    if len(meta_entries) < len(entries):
        meta_entries.extend([None] * (len(entries) - len(meta_entries)))
    else:
        meta_entries = meta_entries[: len(entries)]
    grouped_meta[prompt_idx][start : start + len(entries)] = meta_entries
    return grouped_meta


def seed_generation_groups(
    prompt_count: int,
    grouped_comps: Optional[List[List[str]]],
    grouped_meta: Optional[List[List[Optional[Any]]]],
) -> Tuple[List[List[str]], Optional[List[List[Optional[Any]]]]]:
    """Return initial completion/meta buffers aligned with prompts.

    :param prompt_count: Number of prompts in the batch.
    :type prompt_count: int
    :param grouped_comps: Pre-existing completions per prompt.
    :type grouped_comps: list[list[str]] | None
    :param grouped_meta: Metadata aligned with ``grouped_comps``.
    :type grouped_meta: list[list[Any | None]] | None
    :returns: Tuple of initialized completion groups and metadata buffers.
    :rtype: tuple[list[list[str]], list[list[Any | None]] | None]
    """
    aggregated_comps: List[List[str]] = [[] for _ in range(prompt_count)]
    aggregated_meta: Optional[List[List[Optional[Any]]]] = None
    base_groups = grouped_comps or []
    for idx in range(prompt_count):
        comp_group: List[str] = []
        if idx < len(base_groups) and base_groups[idx]:
            comp_group = list(base_groups[idx])
        meta_group: Optional[List[Optional[Any]]] = None
        if grouped_meta is not None and idx < len(grouped_meta):
            meta_group = grouped_meta[idx]
        aggregated_meta = append_completion_group(
            aggregated_comps,
            aggregated_meta,
            idx,
            comp_group,
            meta_group,
        )
    return aggregated_comps, aggregated_meta


def pending_generation_indices(
    aggregated_comps: List[List[str]],
    expected_generations: int,
) -> List[int]:
    """Return prompt indices that still need completions.

    :param aggregated_comps: Current completion buffers per prompt.
    :type aggregated_comps: list[list[str]]
    :param expected_generations: Desired completions per prompt.
    :type expected_generations: int
    :returns: Indices corresponding to prompts missing completions.
    :rtype: list[int]
    """
    if expected_generations <= 0:
        return []
    return [
        idx
        for idx, comps in enumerate(aggregated_comps)
        if len(comps) < expected_generations
    ]


def determine_retry_limit(
    expected_generations: int,
    max_retry_rounds: Optional[int],
) -> int:
    """Return the number of retry rounds required for a batch.

    :param expected_generations: Desired completions per prompt.
    :type expected_generations: int
    :param max_retry_rounds: Optional override for retry attempts.
    :type max_retry_rounds: int | None
    :returns: Number of retry rounds to attempt.
    :rtype: int
    """
    if max_retry_rounds and max_retry_rounds > 0:
        return max_retry_rounds
    if expected_generations > 0:
        return expected_generations
    return _DEFAULT_RETRY_LIMIT


def retry_incomplete_prompts(
    prompts: List[str],
    generator: Callable[
        [List[str], int, Optional[List[int]]],
        Tuple[List[List[str]], Optional[List[List[Optional[Any]]]]],
    ],
    expected_generations: int,
    aggregated: AggregatedGenerationState,
    max_retry_rounds: Optional[int],
) -> AggregatedGenerationState:
    """Retry prompts missing completions until limits are hit.

    :param prompts: Original prompt strings.
    :type prompts: list[str]
    :param generator: Callable producing grouped completions (plus metadata).
    :type generator: Callable
    :param expected_generations: Desired completions per prompt.
    :type expected_generations: int
    :param aggregated: Mutable completion/metadata buffers gathered so far.
    :type aggregated: AggregatedGenerationState
    :param max_retry_rounds: Optional override of retry attempts.
    :type max_retry_rounds: int | None
    :returns: Aggregated state incorporating any retried completions.
    :rtype: AggregatedGenerationState
    """
    incomplete_indices = pending_generation_indices(
        aggregated.completions,
        expected_generations,
    )
    retry_limit = determine_retry_limit(expected_generations, max_retry_rounds)
    retry_round = 0
    while incomplete_indices and retry_round < retry_limit:
        retry_round += 1
        retry_groups, retry_meta = generator(
            [prompts[idx] for idx in incomplete_indices],
            expected_generations,
            [
                max(expected_generations - len(aggregated.completions[idx]), 0)
                for idx in incomplete_indices
            ],
        )
        retry_groups = retry_groups or [[] for _ in incomplete_indices]
        for local_idx, prompt_idx in enumerate(incomplete_indices):
            aggregated.metadata = append_completion_group(
                aggregated.completions,
                aggregated.metadata,
                prompt_idx,
                list(retry_groups[local_idx]),
                (
                    retry_meta[local_idx]
                    if retry_meta and local_idx < len(retry_meta)
                    else None
                ),
            )
        incomplete_indices = pending_generation_indices(
            aggregated.completions,
            expected_generations,
        )
    return aggregated


def drop_empty_prompt_groups(
    prompts: List[str],
    answers: List[str],
    aggregated_comps: List[List[str]],
    aggregated_meta: Optional[List[List[Optional[Any]]]],
    generation_stats: Dict[str, int],
) -> Tuple[
    List[str],
    List[str],
    List[List[str]],
    Optional[List[List[Optional[Any]]]],
]:
    """Remove prompts that never yielded completions.

    :param prompts: Prompt strings prior to filtering.
    :type prompts: list[str]
    :param answers: Answer strings aligned with ``prompts``.
    :type answers: list[str]
    :param aggregated_comps: Completion groups per prompt.
    :type aggregated_comps: list[list[str]]
    :param aggregated_meta: Optional metadata aligned with completions.
    :type aggregated_meta: list[list[Any | None]] | None
    :param generation_stats: Mutable statistics dictionary for drop counts.
    :type generation_stats: dict[str, int]
    :returns: Filtered prompts/answers/completions/metadata tuples.
    :rtype: tuple[list[str], list[str], list[list[str]], list[list[Any | None]] | None]
    """
    drop_indices = [idx for idx, comps in enumerate(aggregated_comps) if not comps]
    if not drop_indices:
        return prompts, answers, aggregated_comps, aggregated_meta
    generation_stats["dropped_prompts"] += len(drop_indices)
    missing_set = set(drop_indices)
    keep_indices = [idx for idx in range(len(prompts)) if idx not in missing_set]
    prompts = [prompts[idx] for idx in keep_indices]
    answers = [answers[idx] for idx in keep_indices]
    aggregated_comps = [aggregated_comps[idx] for idx in keep_indices]
    if aggregated_meta is not None:
        aggregated_meta = [aggregated_meta[idx] for idx in keep_indices]
    return prompts, answers, aggregated_comps, aggregated_meta


def truncate_to_expected_counts(
    aggregated_comps: List[List[str]],
    aggregated_meta: Optional[List[List[Optional[Any]]]],
    expected_generations: int,
) -> Tuple[
    List[List[str]],
    Optional[List[List[Optional[Any]]]],
    int,
]:
    """Trim completions/meta to requested counts and track partial prompts.

    :param aggregated_comps: Completion groups to clamp.
    :type aggregated_comps: list[list[str]]
    :param aggregated_meta: Optional metadata aligned with completions.
    :type aggregated_meta: list[list[Any | None]] | None
    :param expected_generations: Desired completions per prompt.
    :type expected_generations: int
    :returns: Tuple of (trimmed completions, trimmed metadata, partial count).
    :rtype: tuple[list[list[str]], list[list[Any | None]] | None, int]
    """
    if expected_generations <= 0:
        return aggregated_comps, aggregated_meta, 0
    partial_count = 0
    for idx, comps in enumerate(aggregated_comps):
        if len(comps) > expected_generations:
            aggregated_comps[idx] = comps[:expected_generations]
        if 0 < len(aggregated_comps[idx]) < expected_generations:
            partial_count += 1
        if aggregated_meta is None or idx >= len(aggregated_meta):
            continue
        meta_group = aggregated_meta[idx]
        if isinstance(meta_group, list) and len(meta_group) > len(
            aggregated_comps[idx]
        ):
            aggregated_meta[idx] = meta_group[: len(aggregated_comps[idx])]
    return aggregated_comps, aggregated_meta, partial_count


def flatten_ref_metadata(
    grouped_comps: List[List[str]],
    grouped_meta: Optional[List[List[Optional[Any]]]],
) -> Optional[List[Optional[Any]]]:
    """Flatten metadata to align with the flattened completions list.

    :param grouped_comps: Completion groups used to size metadata slices.
    :type grouped_comps: list[list[str]]
    :param grouped_meta: Metadata aligned per prompt/completion.
    :type grouped_meta: list[list[Any | None]] | None
    :returns: Flattened metadata or ``None`` when unavailable.
    :rtype: list[Any | None] | None
    """
    if grouped_meta is None:
        return None
    flat_meta: List[Optional[Any]] = []
    for prompt_idx, comp_group in enumerate(grouped_comps):
        meta_group: Optional[List[Optional[Any]]] = (
            grouped_meta[prompt_idx] if prompt_idx < len(grouped_meta) else None
        )
        for comp_idx in range(len(comp_group)):
            meta_entry = None
            if meta_group is not None and comp_idx < len(meta_group):
                meta_entry = meta_group[comp_idx]
                if meta_entry is not None and hasattr(meta_entry, "to_trl_payload"):
                    try:
                        meta_entry = meta_entry.to_trl_payload()
                    except TypeError:
                        pass
            flat_meta.append(meta_entry)
    return flat_meta if flat_meta else None


def flatten_prompt_completions(
    gen_batch: "GenerationBatch",
) -> Tuple["PromptCompletionBatch", List[str]]:
    """Return flattened prompt/completion pairs and aligned answers.

    :param gen_batch: Aggregated generation results.
    :type gen_batch: maxent_grpo.training.types.GenerationBatch
    :returns: Tuple of flattened prompt/completion batch and answer list.
    :rtype: tuple[maxent_grpo.training.types.PromptCompletionBatch, list[str]]
    """
    from maxent_grpo.training.types import PromptCompletionBatch  # Local import to avoid cycles

    prompts: List[str] = []
    completions: List[str] = []
    answers: List[str] = []
    for prompt_text, answer_text, comp_group in zip(
        gen_batch.prompts, gen_batch.answers, gen_batch.grouped_completions
    ):
        for completion_text in comp_group:
            prompts.append(prompt_text)
            completions.append(completion_text)
            answers.append(answer_text)
    min_len = min(len(prompts), len(completions), len(answers))
    if min_len == 0:
        return PromptCompletionBatch([], []), []
    prompts = prompts[:min_len]
    completions = completions[:min_len]
    answers = answers[:min_len]
    return PromptCompletionBatch(prompts=prompts, completions=completions), answers


@dataclass
class DistilabelPipelineConfig:
    """Configuration for building a distilabel generation pipeline."""

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
    generation_kwargs = {"max_new_tokens": cfg.max_new_tokens}
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
) -> None:
    """Execute the distilabel generation CLI using parsed arguments.

    :param args: Parsed CLI arguments (see :mod:`cli.generate`).
    :type args: argparse.Namespace
    :param pipeline_builder: Optional factory used to build the pipeline.
    :type pipeline_builder: Callable[[DistilabelPipelineConfig], distilabel.pipeline.Pipeline] | None
    :raises RuntimeError: If the resulting distilabel dataset cannot be pushed.
    """
    from datasets import load_dataset

    print("\nRunning with arguments:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print()
    print(
        (
            f"Loading '{args.hf_dataset}' (config: {args.hf_dataset_config}, "
            f"split: {args.hf_dataset_split}) dataset..."
        )
    )
    dataset = load_dataset(
        args.hf_dataset, args.hf_dataset_config, split=args.hf_dataset_split
    )
    print("Dataset loaded!")
    config = DistilabelPipelineConfig(
        model=args.model,
        base_url=args.vllm_server_url,
        prompt_template=args.prompt_template,
        prompt_column=args.prompt_column,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        num_generations=args.num_generations,
        input_batch_size=args.input_batch_size,
        client_replicas=args.client_replicas,
        timeout=args.timeout,
        retries=args.retries,
    )
    builder = pipeline_builder or build_distilabel_pipeline
    pipeline = builder(config)
    print("Running generation pipeline...")
    distiset = pipeline.run(
        dataset=dataset,
        dataset_batch_size=args.input_batch_size * 1000,
        use_cache=False,
    )
    print("Generation pipeline finished!")
    if args.hf_output_dataset:
        print(
            f"Pushing dataset to {args.hf_output_dataset} (private={args.private})..."
        )
        push_fn = getattr(distiset, "push_to_hub", None)
        if callable(push_fn):
            push_fn(args.hf_output_dataset, args.private)
        else:
            raise RuntimeError("distilabel dataset object does not expose push_to_hub")
        print("Push completed!")


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
