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
from typing import Dict, List, Optional, Tuple

from generation import (
    AggregatedGenerationState,
    drop_empty_prompt_groups,
    flatten_prompt_completions as _flatten_prompt_completions,
    flatten_ref_metadata as _flatten_ref_metadata,
    retry_incomplete_prompts,
    seed_generation_groups,
    truncate_to_expected_counts,
)
from .run_helpers import _group_softmax, require_torch
from .types import (
    AdvantageStats,
    GenerationBatch,
    GenerationFn,
    QDistribution,
    RewardComputation,
    RewardMoments,
    RewardSpec,
)

torch = require_torch("training")
LOG = logging.getLogger(__name__)


def compute_reward_totals(
    reward_spec: RewardSpec,
    completion_batch: List[str],
    flat_answers: List[str],
) -> Tuple[List[float], Dict[str, List[float]]]:
    """Evaluate reward functions and aggregate per-sequence utilities.

    :param reward_spec: Reward configuration specifying callables/weights.
    :type reward_spec: RewardSpec
    :param completion_batch: Flattened completion texts.
    :type completion_batch: list[str]
    :param flat_answers: Flattened answer strings aligned with completions.
    :type flat_answers: list[str]
    :returns: Tuple of total utilities and per-reward raw values.
    :rtype: tuple[list[float], dict[str, list[float]]]
    """
    total_utils = [0.0] * len(completion_batch)
    per_reward_values: Dict[str, List[float]] = {}
    for idx_reward, (reward_fn, reward_weight) in enumerate(
        zip(reward_spec.reward_funcs, reward_spec.reward_weights)
    ):
        reward_key = f"reward_{idx_reward}"
        reward_values = [
            float(val) for val in reward_fn(completion_batch, flat_answers)
        ]
        per_reward_values[reward_key] = reward_values
        if reward_weight != 1.0:
            reward_values = [float(reward_weight) * val for val in reward_values]
        total_utils = [util + val for util, val in zip(total_utils, reward_values)]
    return total_utils, per_reward_values


def reward_moments(
    total_utils: List[float], device: torch.device
) -> Tuple[float, float]:
    """Compute reward mean/std on CPU or current accelerator device.

    :param total_utils: Flattened reward totals per completion.
    :type total_utils: list[float]
    :param device: Device used for tensor computations.
    :type device: ``torch.device``
    :returns: Tuple containing ``(mean, std)`` rewards.
    :rtype: tuple[float, float]
    """
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
    """Return centered advantages per prompt group and flattened samples.

    :param grouped_comps: Completions grouped by prompt.
    :type grouped_comps: list[list[str]]
    :param total_utils: Flattened utilities aligned with completions.
    :type total_utils: list[float]
    :returns: Tuple of grouped advantages and flattened advantage samples.
    :rtype: tuple[list[list[float]], list[float]]
    """
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


def prepare_generation_batch(
    batch: Dict[str, List[str]],
    generator: GenerationFn,
    generation_stats: Dict[str, int],
    expected_generations: int,
    max_retry_rounds: Optional[int] = None,
) -> Optional[GenerationBatch]:
    """Generate completions and retry prompts that initially returned nothing.

    :param batch: Mini-batch containing ``prompt``/``answer`` lists.
    :type batch: dict[str, list[str]]
    :param generator: Callable that produces grouped completions and metadata.
    :type generator: :class:`~training.types.GenerationFn`
    :param generation_stats: Mutable statistics dictionary updated in-place.
    :type generation_stats: dict[str, int]
    :param expected_generations: Desired completions per prompt.
    :type expected_generations: int
    :param max_retry_rounds: Optional cap overriding the default retry limit.
    :type max_retry_rounds: int | None
    :returns: Populated :class:`~training.types.GenerationBatch` or ``None`` if
        generation fails after retries.
    :rtype: :class:`~training.types.GenerationBatch` | None
    """
    prompts: List[str] = batch["prompt"]
    answers: List[str] = batch["answer"]
    if not prompts:
        return None
    LOG.debug(
        "Starting completion generation | prompts=%d | expected_generations=%d",
        len(prompts),
        expected_generations,
    )
    gen_result = generator(prompts, expected_generations)
    if gen_result is None:
        return None
    if isinstance(gen_result, GenerationBatch):
        grouped_comps = gen_result.grouped_completions
        grouped_meta = getattr(gen_result, "grouped_ref_meta", None)
    elif hasattr(gen_result, "grouped_completions"):
        grouped_comps = getattr(gen_result, "grouped_completions")
        grouped_meta = getattr(gen_result, "grouped_ref_meta", None)
    else:
        grouped_comps, grouped_meta = gen_result
    LOG.debug(
        "Generation finished | prompts=%d | groups_returned=%d",
        len(prompts),
        len(grouped_comps) if grouped_comps is not None else 0,
    )
    prompt_count = len(prompts)
    aggregated_comps, aggregated_meta = seed_generation_groups(
        prompt_count,
        grouped_comps,
        grouped_meta,
    )
    aggregated_state = AggregatedGenerationState(aggregated_comps, aggregated_meta)
    aggregated_state = retry_incomplete_prompts(
        prompts,
        generator,
        expected_generations,
        aggregated_state,
        max_retry_rounds,
    )
    aggregated_comps, aggregated_meta = (
        aggregated_state.completions,
        aggregated_state.metadata,
    )
    prompts, answers, aggregated_comps, aggregated_meta = drop_empty_prompt_groups(
        prompts,
        answers,
        aggregated_comps,
        aggregated_meta,
        generation_stats,
    )
    if not aggregated_comps:
        return None
    aggregated_comps, aggregated_meta, partial_count = truncate_to_expected_counts(
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
    """Return per-group q distributions derived from listwise utilities.

    :param grouped_comps: Completion groups per prompt.
    :type grouped_comps: list[list[str]]
    :param total_utils: Flattened utility values aligned with completions.
    :type total_utils: list[float]
    :param temperature: Softmax temperature for listwise distribution.
    :type temperature: float
    :param epsilon: Minimum support value to ensure non-zero probabilities.
    :type epsilon: float
    :returns: Tuple of grouped q-values and flattened q-samples.
    :rtype: tuple[list[list[float]], list[float]]
    """
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
    """Compute utilities, q-distributions, and flattened prompt/completion pairs.

    :param gen_batch: Generation batch containing grouped completions/meta.
    :type gen_batch: :class:`~training.types.GenerationBatch`
    :param reward_spec: Reward configuration (functions + weights).
    :type reward_spec: RewardSpec
    :param device: Torch device used for reward moment computations.
    :type device: ``torch.device``
    :param q_temperature: Temperature used when forming q-distributions.
    :type q_temperature: float
    :param q_epsilon: Epsilon floor ensuring full support in q-distribution.
    :type q_epsilon: float
    :returns: Populated :class:`~training.types.RewardComputation` or ``None``
        when inputs are empty.
    :rtype: :class:`~training.types.RewardComputation` | None
    """
    grouped_comps = gen_batch.grouped_completions
    if not grouped_comps:
        return None
    pair_batch, flat_answers = _flatten_prompt_completions(gen_batch)
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
