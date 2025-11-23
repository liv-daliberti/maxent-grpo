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

"""Reward and generation helpers extracted from the training loop."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .run_helpers import _group_softmax, require_torch
from .run_training_types import (
    AdvantageStats,
    GenerationBatch,
    GenerationFn,
    PromptCompletionBatch,
    QDistribution,
    RewardComputation,
    RewardMoments,
    RewardSpec,
)

torch = require_torch("training")
LOG = logging.getLogger(__name__)

_DEFAULT_RETRY_LIMIT = 3


@dataclass
class AggregatedGenerationState:
    """Mutable container for grouped completions and optional metadata."""

    completions: List[List[str]]
    metadata: Optional[List[List[Optional[Any]]]] = None


def _append_completion_group(
    grouped_comps: List[List[str]],
    grouped_meta: Optional[List[List[Optional[Any]]]],
    prompt_idx: int,
    completions: Optional[List[str]],
    meta_group: Optional[List[Optional[Any]]],
) -> Optional[List[List[Optional[Any]]]]:
    """Append completions (and metadata) for a specific prompt index."""
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
        grouped_meta = [
            [None] * len(group) for group in grouped_comps
        ]
    meta_entries = list(meta_group)
    if len(meta_entries) < len(entries):
        meta_entries.extend([None] * (len(entries) - len(meta_entries)))
    else:
        meta_entries = meta_entries[: len(entries)]
    grouped_meta[prompt_idx][start : start + len(entries)] = meta_entries
    return grouped_meta


def flatten_prompt_completions(
    gen_batch: GenerationBatch,
) -> Tuple[PromptCompletionBatch, List[str]]:
    """Return flattened prompt/completion pairs and aligned answers."""
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


def compute_reward_totals(
    reward_spec: RewardSpec,
    completion_batch: List[str],
    flat_answers: List[str],
) -> Tuple[List[float], Dict[str, List[float]]]:
    """Evaluate reward functions and aggregate per-sequence utilities."""
    total_utils = [0.0] * len(completion_batch)
    per_reward_values: Dict[str, List[float]] = {}
    for idx_reward, (reward_fn, reward_weight) in enumerate(
        zip(reward_spec.reward_funcs, reward_spec.reward_weights)
    ):
        reward_key = f"reward_{idx_reward}"
        reward_values = [float(val) for val in reward_fn(completion_batch, flat_answers)]
        per_reward_values[reward_key] = reward_values
        if reward_weight != 1.0:
            reward_values = [float(reward_weight) * val for val in reward_values]
        total_utils = [util + val for util, val in zip(total_utils, reward_values)]
    return total_utils, per_reward_values


def reward_moments(total_utils: List[float], device: torch.device) -> Tuple[float, float]:
    """Compute reward mean/std on CPU or current accelerator device."""
    if not total_utils:
        return 0.0, 0.0
    utils_tensor = torch.tensor(
        total_utils,
        dtype=torch.float32,
        device=device if device.type != "cpu" else torch.device("cpu"),
    )
    train_reward_mean = float(utils_tensor.mean().item())
    train_reward_std = (
        float(utils_tensor.std(unbiased=False).item())
        if utils_tensor.numel() > 1
        else 0.0
    )
    return train_reward_mean, train_reward_std


def group_advantages(
    grouped_comps: List[List[str]],
    total_utils: List[float],
) -> Tuple[List[List[float]], List[float]]:
    """Return centered advantages per prompt group and flattened samples."""
    advantage_grouped: List[List[float]] = []
    idx_utils = 0
    for comp_group in grouped_comps:
        size = len(comp_group)
        group_vals = total_utils[idx_utils : idx_utils + size]
        if size > 0:
            baseline = float(sum(group_vals) / size)
            adv_vals = [val - baseline for val in group_vals]
            advantage_grouped.append(adv_vals)
        else:
            adv_vals = []
            advantage_grouped.append(adv_vals)
        idx_utils += size
    advantage_samples: List[float] = []
    for adv_vals in advantage_grouped:
        advantage_samples.extend(adv_vals)
    return advantage_grouped, advantage_samples


def _flatten_ref_metadata(
    grouped_comps: List[List[str]],
    grouped_meta: Optional[List[List[Optional[Any]]]],
) -> Optional[List[Optional[Any]]]:
    """Flatten vLLM metadata aligning with completions."""
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
                        meta_entry = meta_entry.to_trl_payload()  # type: ignore[attr-defined]
                    except TypeError:
                        pass
            flat_meta.append(meta_entry)
    return flat_meta if flat_meta else None


def _seed_generation_groups(
    prompt_count: int,
    grouped_comps: Optional[List[List[str]]],
    grouped_meta: Optional[List[List[Optional[Any]]]],
) -> Tuple[List[List[str]], Optional[List[List[Optional[Any]]]]]:
    """Return initial completion/meta buffers aligned with prompts."""
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
        aggregated_meta = _append_completion_group(
            aggregated_comps,
            aggregated_meta,
            idx,
            comp_group,
            meta_group,
        )
    return aggregated_comps, aggregated_meta


def _pending_generation_indices(
    aggregated_comps: List[List[str]],
    expected_generations: int,
) -> List[int]:
    """Return prompt indices that still need completions."""
    if expected_generations <= 0:
        return []
    return [
        idx
        for idx, comps in enumerate(aggregated_comps)
        if len(comps) < expected_generations
    ]


def _determine_retry_limit(
    expected_generations: int,
    max_retry_rounds: Optional[int],
) -> int:
    """Return how many times we should retry generation for missing prompts."""
    if max_retry_rounds and max_retry_rounds > 0:
        return max_retry_rounds
    if expected_generations > 0:
        return expected_generations
    return _DEFAULT_RETRY_LIMIT


def _retry_incomplete_prompts(
    prompts: List[str],
    generator: GenerationFn,
    expected_generations: int,
    aggregated: AggregatedGenerationState,
    max_retry_rounds: Optional[int],
) -> AggregatedGenerationState:
    """Retry prompts missing completions until limits are hit."""
    incomplete_indices = _pending_generation_indices(
        aggregated.completions,
        expected_generations,
    )
    retry_limit = _determine_retry_limit(expected_generations, max_retry_rounds)
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
            aggregated.metadata = _append_completion_group(
                aggregated.completions,
                aggregated.metadata,
                prompt_idx,
                list(retry_groups[local_idx]),
                retry_meta[local_idx] if retry_meta and local_idx < len(retry_meta) else None,
            )
        incomplete_indices = _pending_generation_indices(
            aggregated.completions,
            expected_generations,
        )
    return aggregated


def _drop_empty_prompt_groups(
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
    """Remove prompts that never yielded completions."""
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


def _truncate_to_expected_counts(
    aggregated_comps: List[List[str]],
    aggregated_meta: Optional[List[List[Optional[Any]]]],
    expected_generations: int,
) -> Tuple[
    List[List[str]],
    Optional[List[List[Optional[Any]]]],
    int,
]:
    """Trim completions/meta to the requested counts and track partial prompts."""
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
        if isinstance(meta_group, list) and len(meta_group) > len(aggregated_comps[idx]):
            aggregated_meta[idx] = meta_group[: len(aggregated_comps[idx])]
    return aggregated_comps, aggregated_meta, partial_count


def prepare_generation_batch(
    batch: Dict[str, List[str]],
    generator: GenerationFn,
    generation_stats: Dict[str, int],
    expected_generations: int,
    max_retry_rounds: Optional[int] = None,
) -> Optional[GenerationBatch]:
    """Generate completions and retry prompts that initially returned nothing."""
    prompts: List[str] = batch["prompt"]
    answers: List[str] = batch["answer"]
    if not prompts:
        return None
    LOG.debug(
        "Starting completion generation | prompts=%d | expected_generations=%d",
        len(prompts),
        expected_generations,
    )
    grouped_comps, grouped_meta = generator(prompts, expected_generations)
    LOG.debug(
        "Generation finished | prompts=%d | groups_returned=%d",
        len(prompts),
        len(grouped_comps) if grouped_comps is not None else 0,
    )
    prompt_count = len(prompts)
    aggregated_comps, aggregated_meta = _seed_generation_groups(
        prompt_count,
        grouped_comps,
        grouped_meta,
    )
    aggregated_state = AggregatedGenerationState(aggregated_comps, aggregated_meta)
    aggregated_state = _retry_incomplete_prompts(
        prompts,
        generator,
        expected_generations,
        aggregated_state,
        max_retry_rounds,
    )
    aggregated_comps, aggregated_meta = aggregated_state.completions, aggregated_state.metadata
    prompts, answers, aggregated_comps, aggregated_meta = _drop_empty_prompt_groups(
        prompts,
        answers,
        aggregated_comps,
        aggregated_meta,
        generation_stats,
    )
    if not aggregated_comps:
        return None
    aggregated_comps, aggregated_meta, partial_count = _truncate_to_expected_counts(
        aggregated_comps,
        aggregated_meta,
        expected_generations,
    )
    if partial_count > 0:
        generation_stats.setdefault("partial_prompts", 0)
        generation_stats["partial_prompts"] += partial_count
    return GenerationBatch(
        prompts=prompts,
        answers=answers,
        grouped_completions=aggregated_comps,
        grouped_ref_meta=aggregated_meta,
    )


def _group_q_distribution(
    grouped_comps: List[List[str]],
    total_utils: List[float],
    temperature: float,
    epsilon: float,
) -> Tuple[List[List[float]], List[float]]:
    """Return per-group q distributions derived from listwise utilities."""
    q_grouped: List[List[float]] = []
    q_samples: List[float] = []
    idx_utils = 0
    for comp_group in grouped_comps:
        size = len(comp_group)
        group_vals = total_utils[idx_utils : idx_utils + size]
        if size > 0 and group_vals:
            q_vals = _group_softmax(
                group_vals,
                temperature=max(temperature, 1e-8),
                eps=epsilon,
            )
        else:
            q_vals = []
        q_grouped.append(q_vals)
        q_samples.extend(q_vals)
        idx_utils += size
    return q_grouped, q_samples


def compute_reward_statistics(
    gen_batch: GenerationBatch,
    reward_spec: RewardSpec,
    device: torch.device,
    q_temperature: float,
    q_epsilon: float,
) -> Optional[RewardComputation]:
    """Compute utilities, q-distributions, and flattened prompt/completion pairs."""
    grouped_comps = gen_batch.grouped_completions
    if not grouped_comps:
        return None
    pair_batch, flat_answers = flatten_prompt_completions(gen_batch)
    if not pair_batch.completions:
        return None
    total_utils, per_reward_values = compute_reward_totals(
        reward_spec,
        pair_batch.completions,
        flat_answers,
    )
    moments = RewardMoments(*reward_moments(total_utils, device))
    advantage_stats = AdvantageStats(*group_advantages(grouped_comps, total_utils))
    q_distribution = QDistribution(
        *_group_q_distribution(
            grouped_comps,
            total_utils,
            q_temperature,
            q_epsilon,
        )
    )
    flat_ref_meta = _flatten_ref_metadata(grouped_comps, gen_batch.grouped_ref_meta)
    return RewardComputation(
        total_utils=total_utils,
        per_reward_values=per_reward_values,
        advantage=advantage_stats,
        pairs=pair_batch,
        q_distribution=q_distribution,
        moments=moments,
        ref_logprob_meta=flat_ref_meta,
    )
