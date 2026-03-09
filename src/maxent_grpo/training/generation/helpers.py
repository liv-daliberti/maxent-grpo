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

from typing import List, Tuple, TYPE_CHECKING

from maxent_grpo.training.generation.common import (
    AggregatedGenerationState,
    append_completion_group,
    determine_retry_limit,
    drop_empty_prompt_groups,
    drop_incomplete_prompt_groups,
    flatten_ref_metadata,
    pending_generation_indices,
    retry_incomplete_prompts,
    seed_generation_groups,
    truncate_to_expected_counts,
)

if TYPE_CHECKING:  # pragma: no cover - hints only
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


__all__ = [
    "AggregatedGenerationState",
    "append_completion_group",
    "determine_retry_limit",
    "drop_empty_prompt_groups",
    "drop_incomplete_prompt_groups",
    "flatten_prompt_completions",
    "flatten_ref_metadata",
    "pending_generation_indices",
    "retry_incomplete_prompts",
    "seed_generation_groups",
    "truncate_to_expected_counts",
]
