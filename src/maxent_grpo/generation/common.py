"""
Shared generation utilities used by both the training stack and distilabel CLI.

This module contains the small, dependency-light helpers for grouping,
retrying, and trimming completions so higher layers can import a single source
of truth instead of maintaining divergent copies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

_DEFAULT_RETRY_LIMIT = 3


@dataclass
class AggregatedGenerationState:
    """Mutable container for grouped completions and optional metadata.

    :param completions: Nested list of completions grouped per prompt index.
    :type completions: list[list[str]]
    :param metadata: Optional nested metadata aligned to ``completions``.
    :type metadata: list[list[object | None]] | None
    """

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

    Completions and metadata are extended in place, creating a fresh metadata
    structure when needed. Missing metadata is padded with ``None`` so list
    lengths stay aligned with completions.

    :param grouped_comps: Existing grouped completions buffer.
    :type grouped_comps: list[list[str]]
    :param grouped_meta: Existing grouped metadata buffer; can be ``None`` if
        metadata is not tracked.
    :type grouped_meta: list[list[object | None]] | None
    :param prompt_idx: Index of the prompt whose completions are being appended.
    :type prompt_idx: int
    :param completions: New completions to append for the prompt.
    :type completions: list[str] | None
    :param meta_group: Metadata aligned to ``completions``. Excess entries are
        trimmed and missing entries are padded with ``None``.
    :type meta_group: list[object | None] | None
    :returns: Updated grouped metadata (may be newly created), or ``None`` when
        metadata tracking is disabled.
    :rtype: list[list[object | None]] | None
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
    end = start + len(entries)
    current_meta = grouped_meta[prompt_idx]
    if len(current_meta) < end:
        current_meta.extend([None] * (end - len(current_meta)))
    current_meta[start:end] = meta_entries
    return grouped_meta


def seed_generation_groups(
    prompt_count: int,
    grouped_comps: Optional[List[List[str]]],
    grouped_meta: Optional[List[List[Optional[Any]]]],
) -> Tuple[List[List[str]], Optional[List[List[Optional[Any]]]]]:
    """Return initial completion/meta buffers aligned with prompts.

    The helper normalizes partially filled buffers into fresh lists sized to
    ``prompt_count`` and ensures metadata stays aligned with completions.

    :param prompt_count: Number of prompts that will be processed.
    :type prompt_count: int
    :param grouped_comps: Optional preexisting completions grouped per prompt.
    :type grouped_comps: list[list[str]] | None
    :param grouped_meta: Optional preexisting metadata grouped per prompt.
    :type grouped_meta: list[list[object | None]] | None
    :returns: Tuple of initialized completions buffer and optional metadata
        buffer, both sized to ``prompt_count``.
    :rtype: tuple[list[list[str]], list[list[object | None]] | None]
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

    :param aggregated_comps: Completions grouped per prompt.
    :type aggregated_comps: list[list[str]]
    :param expected_generations: Desired number of completions per prompt.
    :type expected_generations: int
    :returns: Indices whose completion count is below ``expected_generations``.
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

    :param expected_generations: Desired completions per prompt. Used as a
        fallback retry budget when explicit retries are not provided.
    :type expected_generations: int
    :param max_retry_rounds: Explicit retry cap; overrides defaults when > 0.
    :type max_retry_rounds: int | None
    :returns: Retry limit, defaulting to ``expected_generations`` or
        ``_DEFAULT_RETRY_LIMIT`` when neither input is set.
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

    The ``generator`` callback is invoked with the list of prompts still
    missing completions, along with per-prompt deficits. Metadata returned by
    the generator is merged if available.

    :param prompts: Original prompt strings.
    :type prompts: list[str]
    :param generator: Callable performing generation for a batch of prompts.
        It should accept ``prompts``, ``expected_generations``, and optionally a
        list of per-prompt counts, returning grouped completions and metadata.
    :type generator: Callable[[list[str], int, list[int] | None], tuple[list[list[str]], list[list[object | None]] | None]]
    :param expected_generations: Number of completions requested per prompt.
    :type expected_generations: int
    :param aggregated: Aggregated state containing completions and metadata to
        be updated in place.
    :type aggregated: AggregatedGenerationState
    :param max_retry_rounds: Explicit retry cap; defaults derive from
        ``expected_generations`` when omitted.
    :type max_retry_rounds: int | None
    :returns: Updated aggregated generation state after retries are exhausted or
        all prompts are complete.
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
        meta_payload: Optional[List[List[Optional[Any]]]] = None
        if isinstance(retry_meta, list):
            meta_payload = retry_meta
        for local_idx, prompt_idx in enumerate(incomplete_indices):
            meta_group = None
            if meta_payload is not None and local_idx < len(meta_payload):
                meta_group = meta_payload[local_idx]
            group = retry_groups[local_idx] if local_idx < len(retry_groups) else []
            aggregated.metadata = append_completion_group(
                aggregated.completions,
                aggregated.metadata,
                prompt_idx,
                group,
                meta_group,
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

    Any prompt lacking completions is removed from all aligned structures and
    a ``dropped_prompts`` counter in ``generation_stats`` is incremented.

    :param prompts: Prompt texts aligned to ``answers`` and grouped completions.
    :type prompts: list[str]
    :param answers: Reference answers aligned to prompts.
    :type answers: list[str]
    :param aggregated_comps: Grouped completions per prompt (mutable).
    :type aggregated_comps: list[list[str]]
    :param aggregated_meta: Optional grouped metadata per prompt.
    :type aggregated_meta: list[list[object | None]] | None
    :param generation_stats: Mutable statistics dictionary for counters.
    :type generation_stats: dict[str, int]
    :returns: Filtered prompts, answers, completions, and metadata aligned to
        the remaining prompts.
    :rtype: tuple[list[str], list[str], list[list[str]], list[list[object | None]] | None]
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

    :param aggregated_comps: Grouped completions per prompt.
    :type aggregated_comps: list[list[str]]
    :param aggregated_meta: Optional grouped metadata per prompt.
    :type aggregated_meta: list[list[object | None]] | None
    :param expected_generations: Desired completions per prompt; values <= 0
        skip trimming.
    :type expected_generations: int
    :returns: Tuple of trimmed completions, trimmed metadata, and the number of
        prompts that still have fewer completions than requested.
    :rtype: tuple[list[list[str]], list[list[object | None]] | None, int]
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

    Metadata entries exposing ``to_trl_payload`` are converted before being
    appended. Missing metadata is filled with ``None``.

    :param grouped_comps: Grouped completions per prompt.
    :type grouped_comps: list[list[str]]
    :param grouped_meta: Grouped metadata aligned to ``grouped_comps``.
    :type grouped_meta: list[list[object | None]] | None
    :returns: Flattened metadata aligned to a flattened completions list, or
        ``None`` when no metadata exists.
    :rtype: list[object | None] | None
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


__all__ = [
    "AggregatedGenerationState",
    "append_completion_group",
    "determine_retry_limit",
    "drop_empty_prompt_groups",
    "flatten_ref_metadata",
    "pending_generation_indices",
    "retry_incomplete_prompts",
    "seed_generation_groups",
    "truncate_to_expected_counts",
]
