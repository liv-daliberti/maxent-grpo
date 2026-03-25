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
# pylint: disable=broad-exception-caught

import logging
import math
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING, cast
from types import SimpleNamespace

from maxent_grpo.training.generation import (
    AggregatedGenerationState,
    drop_empty_prompt_groups,
    drop_incomplete_prompt_groups,
    flatten_prompt_completions as _flatten_prompt_completions,
    flatten_ref_metadata as _flatten_ref_metadata,
    retry_incomplete_prompts,
    seed_generation_groups,
    truncate_to_expected_counts,
)
from maxent_grpo.training.generation.errors import (
    GenerationServiceError,
    log_generation_service_error,
)
from maxent_grpo.training.runtime import require_torch
from .run_helpers import _group_softmax
from maxent_grpo.rewards.basic import (
    RewardConfig,
    _answer_pat,
    _extract_boxed_answer,
    get_reward_funcs,
)
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

if TYPE_CHECKING:
    import torch as torch_types

    TorchDevice = torch_types.device
else:  # pragma: no cover - runtime uses optional torch stub
    TorchDevice = Any


def _rank_tag() -> str:
    """Return best-effort rank string for logging."""

    try:
        dist = getattr(torch, "distributed", None)
        if dist is not None and dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            world = dist.get_world_size()
            return f"rank={rank}/{world}"
    except Exception:
        pass
    return "rank=na"


def _extract_ref_logprob_fields(meta_entry: Any) -> Tuple[Optional[Any], Optional[Any]]:
    """Return ``(logprob_sum, token_count)`` when present in metadata entries."""

    if meta_entry is None:
        return None, None
    logprob_sum = getattr(meta_entry, "logprob_sum", None)
    token_count = getattr(meta_entry, "token_count", None)
    token_logprobs = getattr(meta_entry, "token_logprobs", None)
    if isinstance(meta_entry, dict):
        if logprob_sum is None:
            logprob_sum = meta_entry.get("logprob_sum")
            if logprob_sum is None:
                logprob_sum = meta_entry.get("cumulative_logprob")
        if token_logprobs is None:
            token_logprobs = meta_entry.get("token_logprobs")
            if token_logprobs is None:
                token_logprobs = meta_entry.get("logprobs")
        if token_count is None:
            token_count = meta_entry.get("token_count")
            if token_count is None:
                token_count = meta_entry.get("num_tokens")
            if token_count is None:
                if token_logprobs is not None:
                    try:
                        token_count = len(token_logprobs)
                    except (TypeError, ValueError):
                        token_count = None
    if logprob_sum is None and token_logprobs is not None:
        try:
            logprob_sum = float(sum(float(val) for val in token_logprobs))
        except (TypeError, ValueError):
            logprob_sum = None
    return logprob_sum, token_count


def _sanitize_ref_logprob_meta(
    flat_meta: Optional[List[Optional[Any]]], total_sequences: int
) -> Optional[List[Optional[Any]]]:
    """
    Drop reference metadata when any entry is missing logprob information.

    :param flat_meta: Flattened metadata aligned to completions.
    :type flat_meta: list | None
    :param total_sequences: Expected number of completions for the batch.
    :type total_sequences: int
    :returns: Metadata when complete, otherwise ``None``.
    :rtype: list | None
    """

    if not flat_meta or total_sequences <= 0:
        return None
    if len(flat_meta) != total_sequences:
        return None
    missing_idx: List[int] = []
    saw_any_logprob_fields = False
    for idx, entry in enumerate(flat_meta):
        logprob_sum, token_count = _extract_ref_logprob_fields(entry)
        if logprob_sum is not None or token_count is not None:
            saw_any_logprob_fields = True
        if logprob_sum is None or token_count is None:
            missing_idx.append(idx)
            continue
        try:
            float(logprob_sum)
            int(token_count)
        except (TypeError, ValueError):
            missing_idx.append(idx)
    # If none of the entries advertise logprob information, treat the metadata
    # as opaque and keep it. When some entries include logprob fields but others
    # don't, drop the entire batch to avoid mixing stale/partial ref stats.
    if not saw_any_logprob_fields:
        return flat_meta
    if missing_idx:
        if not getattr(_sanitize_ref_logprob_meta, "_warned", False):
            LOG.warning(
                "Incomplete reference logprob metadata detected | missing_entries=%d/%d | first_missing_idx=%d | "
                "keeping metadata for behavior-logprob fallbacks.",
                len(missing_idx),
                total_sequences,
                missing_idx[0],
            )
            setattr(_sanitize_ref_logprob_meta, "_warned", True)
        return flat_meta
    return flat_meta


def _call_reward_fn(
    reward_fn: Any,
    completions: List[str],
    answers: List[str],
    *,
    is_eval: bool,
    split: str,
) -> List[float]:
    """Call a reward fn with backward-compatible kwargs handling."""

    try:
        return reward_fn(completions, answers, is_eval=is_eval, split=split)
    except TypeError:
        try:
            return reward_fn(completions, answers)
        except TypeError:
            return reward_fn(completions, answers, is_eval=is_eval)


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
    reward_tensors: List[Any] = []
    for idx_reward, (reward_fn, reward_weight) in enumerate(
        zip(reward_spec.reward_funcs, reward_spec.reward_weights)
    ):
        reward_key = f"reward_{idx_reward}"
        reward_values = [
            float(val)
            for val in _call_reward_fn(
                reward_fn, completion_batch, flat_answers, is_eval=False, split="train"
            )
        ]
        per_reward_values[reward_key] = reward_values
        reward_tensor = torch.tensor(
            reward_values, dtype=getattr(torch, "float32", None)
        )
        if reward_weight != 1.0:
            reward_tensor = reward_tensor * float(reward_weight)
        reward_tensors.append(reward_tensor)
    if reward_tensors:
        stack_fn = getattr(torch, "stack", None)
        if callable(stack_fn):
            stacked = stack_fn(reward_tensors, dim=0)
            try:
                total_tensor = torch.nansum(stacked, dim=0)
            except AttributeError:
                total_tensor = torch.sum(torch.nan_to_num(stacked, nan=0.0), dim=0)
            total_utils = [float(val) for val in total_tensor.tolist()]
        else:
            # Fallback path used by lightweight torch doubles in unit tests.
            total_utils = [0.0] * len(completion_batch)
            for tensor in reward_tensors:
                tolist_fn = getattr(tensor, "tolist", None)
                values = tolist_fn() if callable(tolist_fn) else tensor
                for idx, raw in enumerate(values):
                    try:
                        val = float(raw)
                    except (TypeError, ValueError):
                        val = 0.0
                    if math.isnan(val) or math.isinf(val):
                        val = 0.0
                    total_utils[idx] += val
    return total_utils, per_reward_values


def reward_moments(
    total_utils: List[float], device: TorchDevice
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
    try:
        mean_val = sum(total_utils) / len(total_utils)
        train_reward_mean = float(mean_val)
        if len(total_utils) > 1:
            var = sum((u - mean_val) ** 2 for u in total_utils) / len(total_utils)
            train_reward_std = float(math.sqrt(var))
        else:
            train_reward_std = 0.0
        return train_reward_mean, train_reward_std
    except (TypeError, ZeroDivisionError, ValueError, OverflowError, RuntimeError):
        torch_mod = require_torch("training_rewards")
        utils_tensor = torch_mod.tensor(
            total_utils,
            dtype=getattr(torch_mod, "float32", None),
            device=(
                device
                if getattr(device, "type", "cpu") != "cpu"
                else torch_mod.device("cpu")
            ),
        )
        mean_val = getattr(utils_tensor, "mean", lambda *a, **k: utils_tensor)()
        train_reward_mean = float(getattr(mean_val, "item", lambda: mean_val)())
        if utils_tensor.numel() > 1:
            std_val = getattr(utils_tensor, "std", lambda *a, **k: utils_tensor)(
                unbiased=False
            )
            train_reward_std = float(getattr(std_val, "item", lambda: std_val)())
        else:
            train_reward_std = 0.0
        return train_reward_mean, train_reward_std


def _seed_extract_answer(text: str) -> Optional[str]:
    """Return the raw final answer string used for SEED-GRPO clustering."""

    boxed = _extract_boxed_answer(text)
    if boxed is not None:
        answer = str(boxed).strip()
        return answer or None
    match = _answer_pat.search(text)
    if match is None:
        return None
    answer = str(match.group(1)).strip()
    return answer or None


def _seed_semantic_ids_by_answers(answers_list: List[str]) -> List[int]:
    """Match the official SEED-GRPO exact-answer clustering rule."""

    answer_to_id: Dict[str, int] = {}
    semantic_ids: List[int] = []
    for answer in answers_list:
        if answer not in answer_to_id:
            answer_to_id[answer] = len(answer_to_id)
        semantic_ids.append(answer_to_id[answer])
    return semantic_ids


def _seed_logsumexp(values: List[float]) -> float:
    """Return a numerically stable log-sum-exp over ``values``."""

    if not values:
        return float("-inf")
    max_val = max(values)
    if not math.isfinite(max_val):
        return max_val
    total = sum(math.exp(val - max_val) for val in values)
    if total <= 0.0:
        return float("-inf")
    return max_val + math.log(total)


def _seed_logsumexp_by_id(
    semantic_ids: List[int], log_likelihoods: List[float]
) -> List[float]:
    """Aggregate normalized cluster log-mass by semantic id."""

    if not semantic_ids or not log_likelihoods or len(semantic_ids) != len(log_likelihoods):
        return []
    norm = _seed_logsumexp(log_likelihoods)
    if not math.isfinite(norm):
        raise ValueError("SEED-GRPO requires finite completion log-likelihoods.")
    unique_ids = sorted(set(int(uid) for uid in semantic_ids))
    cluster_log_probs: List[float] = []
    for uid in unique_ids:
        cluster_vals = [
            float(log_likelihoods[idx])
            for idx, semantic_id in enumerate(semantic_ids)
            if int(semantic_id) == uid
        ]
        cluster_log_probs.append(_seed_logsumexp(cluster_vals) - norm)
    return cluster_log_probs


def _seed_predictive_entropy_rao(cluster_log_probs: List[float]) -> float:
    """Return Rao-style predictive entropy over normalized cluster mass."""

    entropy = 0.0
    for log_prob in cluster_log_probs:
        if not math.isfinite(log_prob):
            continue
        prob = math.exp(log_prob)
        entropy -= prob * log_prob
    return float(entropy)


def _compute_seed_grpo_statistics(
    gen_batch: GenerationBatch,
    *,
    alpha: float,
    normalize_by_max_entropy: bool,
    length_normalize_logprobs: bool,
    num_generations: Optional[int],
) -> Tuple[List[float], List[float], float, float]:
    """Return per-prompt semantic entropies and advantage scales for SEED-GRPO."""

    grouped_comps = list(getattr(gen_batch, "grouped_completions", []) or [])
    grouped_meta = list(getattr(gen_batch, "grouped_ref_meta", []) or [])
    if not grouped_comps:
        return [], [], 0.0, 0.0
    if len(grouped_meta) < len(grouped_comps):
        raise ValueError(
            "SEED-GRPO requires generation logprob metadata for every prompt group."
        )

    generation_count = int(num_generations or 0)
    if generation_count <= 0:
        generation_count = max((len(group) for group in grouped_comps), default=0)
    max_possible_entropy = math.log(generation_count) if generation_count > 1 else 0.0
    effective_alpha = float(alpha)
    if normalize_by_max_entropy and max_possible_entropy > 0.0:
        effective_alpha = effective_alpha / max_possible_entropy

    entropies: List[float] = []
    scales: List[float] = []
    for prompt_idx, comp_group in enumerate(grouped_comps):
        meta_group = grouped_meta[prompt_idx]
        if not isinstance(meta_group, list) or len(meta_group) < len(comp_group):
            raise ValueError(
                "SEED-GRPO requires generation logprob metadata aligned with completions."
            )
        question_answers: List[str] = []
        log_liks: List[float] = []
        for comp_text, meta_entry in zip(comp_group, meta_group):
            answer = _seed_extract_answer(str(comp_text))
            question_answers.append(answer if answer is not None else "NO_ANSWER_FOUND")
            logprob_sum, token_count = _extract_ref_logprob_fields(meta_entry)
            if logprob_sum is None:
                raise ValueError(
                    "SEED-GRPO requires per-completion generation logprob metadata."
                )
            try:
                log_lik = float(logprob_sum)
                if length_normalize_logprobs:
                    denom = float(token_count) if token_count is not None else 0.0
                    if math.isfinite(denom) and denom > 0.0:
                        log_lik = log_lik / denom
                log_liks.append(log_lik)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "SEED-GRPO requires finite generation log-probabilities."
                ) from exc

        if all(answer == "NO_ANSWER_FOUND" for answer in question_answers):
            semantic_ids = list(range(len(question_answers)))
        else:
            no_answer_indices = [
                idx
                for idx, answer in enumerate(question_answers)
                if answer == "NO_ANSWER_FOUND"
            ]
            valid_answers = [
                answer for answer in question_answers if answer != "NO_ANSWER_FOUND"
            ]
            valid_indices = [
                idx
                for idx, answer in enumerate(question_answers)
                if answer != "NO_ANSWER_FOUND"
            ]
            if no_answer_indices:
                valid_semantic_ids = _seed_semantic_ids_by_answers(valid_answers)
                semantic_ids = [-1] * len(question_answers)
                for valid_pos, question_idx in enumerate(valid_indices):
                    semantic_ids[question_idx] = valid_semantic_ids[valid_pos]
                max_id = max(valid_semantic_ids, default=-1)
                for extra_offset, question_idx in enumerate(no_answer_indices):
                    semantic_ids[question_idx] = max_id + 1 + extra_offset
            else:
                semantic_ids = _seed_semantic_ids_by_answers(question_answers)

        cluster_log_probs = _seed_logsumexp_by_id(semantic_ids, log_liks)
        entropy = _seed_predictive_entropy_rao(cluster_log_probs)
        scale = 1.0 / (1.0 + effective_alpha * entropy)
        entropies.append(float(entropy))
        scales.append(float(scale))

    return entropies, scales, float(effective_alpha), float(max_possible_entropy)


def _apply_group_scales(
    advantage_grouped: List[List[float]],
    group_scales: Optional[List[float]],
) -> Tuple[List[List[float]], List[float]]:
    """Scale grouped advantages by per-prompt multipliers."""

    if not group_scales:
        flat: List[float] = []
        for group in advantage_grouped:
            flat.extend(group)
        return advantage_grouped, flat
    scaled_grouped: List[List[float]] = []
    scaled_flat: List[float] = []
    for idx, group in enumerate(advantage_grouped):
        scale = (
            float(group_scales[idx])
            if idx < len(group_scales) and math.isfinite(float(group_scales[idx]))
            else 1.0
        )
        scaled_group = [float(scale) * float(value) for value in group]
        scaled_grouped.append(scaled_group)
        scaled_flat.extend(scaled_group)
    return scaled_grouped, scaled_flat


def group_advantages(
    grouped_comps: List[List[str]],
    total_utils: List[float],
    *,
    scale_rewards: bool = True,
) -> Tuple[List[List[float]], List[float]]:
    """Return normalized advantages per prompt group and flattened samples.

    :param grouped_comps: Completions grouped by prompt.
    :type grouped_comps: list[list[str]]
    :param total_utils: Flattened utilities aligned with completions.
    :type total_utils: list[float]
    :param scale_rewards: Whether to divide by group std (TRL default).
    :type scale_rewards: bool
    :returns: Tuple of grouped advantages and flattened advantage samples.
    :rtype: tuple[list[list[float]], list[float]]
    """
    advantage_grouped: List[List[float]] = []
    eps = 1e-4
    idx_utils = 0
    for comp_group in grouped_comps:
        size = len(comp_group)
        group_vals = total_utils[idx_utils : idx_utils + size]
        if size > 0:
            baseline = float(sum(group_vals) / size)
            if scale_rewards and size > 1:
                var = sum((val - baseline) ** 2 for val in group_vals) / float(size - 1)
                std = math.sqrt(var)
            else:
                std = 0.0
            if scale_rewards:
                denom = std + eps
                adv_vals = [(val - baseline) / denom for val in group_vals]
            else:
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
    generator: GenerationFn[Any],
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
        LOG.debug("Generation skipped | %s | reason=empty_prompts", _rank_tag())
        return None
    LOG.debug(
        "Starting completion generation | %s | prompts=%d | expected_generations=%d",
        _rank_tag(),
        len(prompts),
        expected_generations,
    )

    def _call_generator(
        prompt_batch: List[str],
        expected: int,
        per_prompt_counts: Optional[List[int]] = None,
    ) -> Any:
        import inspect

        per_prompt_repr = "none"
        if per_prompt_counts is not None:
            try:
                per_prompt_repr = (
                    f"len={len(per_prompt_counts)} first3={list(per_prompt_counts)[:3]}"
                )
            except (TypeError, ValueError):
                per_prompt_repr = str(per_prompt_counts)
        LOG.debug(
            "Invoking generator | prompts=%d | expected=%d | per_prompt_counts=%s",
            len(prompt_batch),
            expected,
            per_prompt_repr,
        )
        try:
            signature = inspect.signature(generator)
        except (TypeError, ValueError):
            signature = None

        def _supports_positional_counts() -> bool:
            if signature is None:
                return True
            params = list(signature.parameters.values())
            if any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params):
                return True
            positional = [
                p
                for p in params
                if p.kind
                in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                )
            ]
            return len(positional) >= 3

        def _supports_keyword_counts() -> bool:
            if signature is None:
                return False
            if "per_prompt_counts" in signature.parameters:
                return True
            return any(
                p.kind == inspect.Parameter.VAR_KEYWORD
                for p in signature.parameters.values()
            )

        try:
            if per_prompt_counts is None:
                if _supports_positional_counts():
                    result = generator(prompt_batch, expected, None)
                else:
                    result = generator(prompt_batch, expected)
            elif _supports_positional_counts():
                result = generator(prompt_batch, expected, per_prompt_counts)
            elif _supports_keyword_counts():
                result = generator(
                    prompt_batch,
                    expected,
                    per_prompt_counts=per_prompt_counts,
                )
            else:
                LOG.debug(
                    "Generator does not accept per_prompt_counts; invoking without it."
                )
                result = generator(prompt_batch, expected)
            LOG.debug(
                "Generator returned | result_type=%s",
                type(result).__name__,
            )
            return result
        except GenerationServiceError as exc:
            log_generation_service_error(LOG, "training", exc)
            raise

    gen_result = _call_generator(prompts, expected_generations)
    if gen_result is None:
        LOG.warning(
            "Generation skipped | %s | reason=generator_returned_none",
            _rank_tag(),
        )
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
        "Generator output unpacked | %s | grouped_type=%s | meta_present=%s",
        _rank_tag(),
        type(grouped_comps).__name__,
        grouped_meta is not None,
    )
    LOG.debug(
        "Generation finished | %s | prompts=%d | groups_returned=%d",
        _rank_tag(),
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
    LOG.debug(
        "Retrying incomplete prompts | %s | initial_groups=%d | expected=%d",
        _rank_tag(),
        len(aggregated_comps) if aggregated_comps is not None else 0,
        expected_generations,
    )
    aggregated_state = retry_incomplete_prompts(
        prompts,
        _call_generator,
        expected_generations,
        aggregated_state,
        max_retry_rounds,
    )
    aggregated_comps, aggregated_meta = (
        aggregated_state.completions,
        aggregated_state.metadata,
    )
    LOG.debug(
        "Retries done | %s | prompts=%d | groups=%d",
        _rank_tag(),
        len(prompts),
        len(aggregated_comps) if aggregated_comps is not None else 0,
    )
    pre_prompt_count = len(prompts)
    pre_group_count = len(aggregated_comps) if aggregated_comps is not None else 0
    pre_total_comps = (
        sum(len(group) for group in aggregated_comps) if aggregated_comps else 0
    )
    pre_empty_groups = (
        sum(1 for group in aggregated_comps if not group) if aggregated_comps else 0
    )
    LOG.debug(
        "Generation pre-filter | %s | prompts=%d | groups=%d | total_completions=%d | empty_groups=%d",
        _rank_tag(),
        pre_prompt_count,
        pre_group_count,
        pre_total_comps,
        pre_empty_groups,
    )
    prompts, answers, aggregated_comps, aggregated_meta = drop_empty_prompt_groups(
        prompts,
        answers,
        aggregated_comps,
        aggregated_meta,
        generation_stats,
    )
    aggregated_comps, aggregated_meta, partial_count = truncate_to_expected_counts(
        aggregated_comps,
        aggregated_meta,
        expected_generations,
    )
    if partial_count > 0:
        generation_stats.setdefault("partial_prompts", 0)
        generation_stats["partial_prompts"] += int(partial_count)
    prompts, answers, aggregated_comps, aggregated_meta, mismatch_count = (
        drop_incomplete_prompt_groups(
            prompts,
            answers,
            aggregated_comps,
            aggregated_meta,
            expected_generations,
            generation_stats,
        )
    )
    post_prompt_count = len(prompts)
    post_group_count = len(aggregated_comps) if aggregated_comps is not None else 0
    post_total_comps = (
        sum(len(group) for group in aggregated_comps) if aggregated_comps else 0
    )
    post_empty_groups = (
        sum(1 for group in aggregated_comps if not group) if aggregated_comps else 0
    )
    LOG.debug(
        "Generation post-filter | %s | prompts=%d | groups=%d | total_completions=%d | empty_groups=%d | dropped_prompts=%d",
        _rank_tag(),
        post_prompt_count,
        post_group_count,
        post_total_comps,
        post_empty_groups,
        max(pre_prompt_count - post_prompt_count, 0),
    )
    if not aggregated_comps:
        LOG.warning(
            "Generation skipped | %s | reason=no_completions_after_filter | prompts=%d",
            _rank_tag(),
            post_prompt_count,
        )
        return None
    if mismatch_count > 0:
        LOG.debug(
            "Dropped incomplete groups | %s | prompts=%d | expected=%d | dropped=%d",
            _rank_tag(),
            len(prompts),
            expected_generations,
            mismatch_count,
        )
    completion_info: List[List[dict]] = [
        [{} for _ in group] for group in aggregated_comps
    ]
    if aggregated_meta is not None:
        # Propagate token-id metadata (when available) into completion_info.
        # Some generation backends include token ids or other structured info
        # alongside reference-logprob summaries; keeping token ids here lets
        # downstream scoring avoid re-tokenizing long completions.
        def _meta_to_dict(entry: Any) -> Optional[Dict[str, Any]]:
            if entry is None:
                return None
            if hasattr(entry, "to_trl_payload"):
                try:
                    value = entry.to_trl_payload()
                    return value if isinstance(value, dict) else None
                except (AttributeError, TypeError, ValueError):
                    return None
            return entry if isinstance(entry, dict) else None

        def _extract_token_ids(
            entry_dict: Optional[Dict[str, Any]],
        ) -> Optional[List[int]]:
            if not entry_dict:
                return None
            token_ids = entry_dict.get("token_ids")
            if token_ids is None and isinstance(entry_dict.get("raw_output"), dict):
                raw = entry_dict["raw_output"]
                token_ids = raw.get("token_ids") or raw.get("output_token_ids")
            if token_ids is None:
                return None
            if hasattr(token_ids, "tolist"):
                try:
                    token_ids = token_ids.tolist()
                except (AttributeError, TypeError, ValueError) as exc:
                    LOG.debug("Failed to coerce token_ids to list: %s", exc)
            if (
                isinstance(token_ids, list)
                and token_ids
                and isinstance(token_ids[0], list)
            ):
                token_ids = token_ids[0]
            if not isinstance(token_ids, list):
                return None
            coerced: List[int] = []
            for val in token_ids:
                try:
                    coerced.append(int(val))
                except (TypeError, ValueError):
                    return None
            return coerced

        for prompt_idx, comp_group in enumerate(aggregated_comps):
            if prompt_idx >= len(aggregated_meta):
                continue
            meta_group = aggregated_meta[prompt_idx]
            if not isinstance(meta_group, list) or not meta_group:
                continue
            for comp_idx in range(len(comp_group)):
                meta_entry = (
                    meta_group[comp_idx] if comp_idx < len(meta_group) else None
                )
                meta_dict = _meta_to_dict(meta_entry)
                token_ids = _extract_token_ids(meta_dict)
                if token_ids is not None:
                    completion_info[prompt_idx][comp_idx]["token_ids"] = token_ids
    return GenerationBatch(
        prompts=prompts,
        answers=answers,
        grouped_completions=aggregated_comps,
        grouped_ref_meta=aggregated_meta,
        grouped_completion_info=completion_info,
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
    LOG.debug(
        "Computing q distribution | groups=%d | total_utils=%d | temperature=%.4f | epsilon=%.2e",
        len(grouped_comps),
        len(total_utils),
        temperature,
        epsilon,
    )
    q_grouped: List[List[float]] = []
    q_samples: List[float] = []
    idx_utils = 0
    for group_idx, comp_group in enumerate(grouped_comps):
        size = len(comp_group)
        group_vals = total_utils[idx_utils : idx_utils + size]
        if LOG.isEnabledFor(logging.DEBUG) and group_idx < 5:
            LOG.debug(
                "Softmax sampler | group_idx=%d | size=%d | temp=%.4f | eps=%.2e | util_sample=%s",
                group_idx,
                size,
                temperature,
                epsilon,
                group_vals[: min(3, len(group_vals))],
            )
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
    device: TorchDevice,
    q_temperature: float,
    q_epsilon: float,
    controller_beta: Optional[float] = None,
    controller_tau: Optional[float] = None,
    scale_rewards: bool = True,
    seed_grpo_enabled: bool = False,
    seed_grpo_alpha: float = 0.0417,
    seed_grpo_alpha_normalize_by_max_entropy: bool = True,
    seed_grpo_length_normalize_logprobs: bool = True,
    seed_grpo_num_generations: Optional[int] = None,
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
    :param controller_beta: Optional KL controller beta logged with stats.
    :type controller_beta: float | None
    :param controller_tau: Optional controller tau logged alongside q temp.
    :type controller_tau: float | None
    :returns: Populated :class:`~maxent_grpo.training.types.rewards.RewardComputation` or ``None``
        when inputs are empty.
    :rtype: :class:`~maxent_grpo.training.types.rewards.RewardComputation` | None
    """
    grouped_comps = gen_batch.grouped_completions
    if not grouped_comps:
        LOG.warning(
            "Reward stats skipped | %s | reason=empty_grouped_completions",
            _rank_tag(),
        )
        return None
    pair_batch, flat_answers = _flatten_prompt_completions(gen_batch)
    if not pair_batch.completions:
        total_groups = len(grouped_comps)
        total_comps = sum(len(group) for group in grouped_comps)
        LOG.warning(
            "Reward stats skipped | %s | reason=empty_flat_completions | groups=%d | total_completions=%d",
            _rank_tag(),
            total_groups,
            total_comps,
        )
        return None
    LOG.debug(
        "Reward stats inputs | %s | prompts=%d | completions=%d",
        _rank_tag(),
        len(grouped_comps),
        len(pair_batch.completions),
    )
    completion_metadata = getattr(pair_batch, "metadata", None)
    total_utils, per_reward_values = compute_reward_totals(
        reward_spec,
        pair_batch.completions,
        flat_answers,
    )
    moments = RewardMoments(*reward_moments(total_utils, device))
    advantage_grouped, advantage_samples = group_advantages(
        grouped_comps, total_utils, scale_rewards=scale_rewards
    )
    seed_semantic_entropies: Optional[List[float]] = None
    seed_advantage_scales: Optional[List[float]] = None
    seed_alpha_effective: Optional[float] = None
    seed_max_possible_entropy: Optional[float] = None
    if seed_grpo_enabled:
        (
            seed_semantic_entropies,
            seed_advantage_scales,
            seed_alpha_effective,
            seed_max_possible_entropy,
        ) = _compute_seed_grpo_statistics(
            gen_batch,
            alpha=seed_grpo_alpha,
            normalize_by_max_entropy=seed_grpo_alpha_normalize_by_max_entropy,
            length_normalize_logprobs=seed_grpo_length_normalize_logprobs,
            num_generations=seed_grpo_num_generations,
        )
        advantage_grouped, advantage_samples = _apply_group_scales(
            advantage_grouped,
            seed_advantage_scales,
        )
    advantage_stats = AdvantageStats(advantage_grouped, advantage_samples)
    q_distribution = QDistribution(
        *_group_q_distribution(
            grouped_comps,
            total_utils,
            q_temperature,
            q_epsilon,
        )
    )
    flat_ref_meta = _sanitize_ref_logprob_meta(
        _flatten_ref_metadata(grouped_comps, gen_batch.grouped_ref_meta),
        len(pair_batch.completions),
    )
    if LOG.isEnabledFor(logging.DEBUG):
        if isinstance(flat_ref_meta, list):
            flat_meta_list: List[Optional[Any]] = list(flat_ref_meta)
        else:
            flat_meta_list = []
        meta_len = len(flat_meta_list)
        sample = flat_meta_list[: min(2, meta_len)] if meta_len else None
        LOG.debug(
            "Ref metadata flatten | grouped_meta=%s | entries=%d | sample=%s",
            "none" if gen_batch.grouped_ref_meta is None else "present",
            meta_len,
            sample,
        )
    q_samples = q_distribution.samples or []
    if q_samples:
        q_min = min(q_samples)
        q_max = max(q_samples)
    else:
        q_min = q_max = 0.0
    beta_repr = "nan"
    try:
        if controller_beta is not None:
            beta_repr = f"{float(controller_beta):.4f}"
    except (TypeError, ValueError) as exc:
        LOG.debug("Failed to format controller_beta for logging: %s", exc)
    tau_repr = "nan"
    try:
        if controller_tau is not None:
            tau_repr = f"{float(controller_tau):.4f}"
    except (TypeError, ValueError) as exc:
        LOG.debug("Failed to format controller_tau for logging: %s", exc)
    LOG.debug(
        "Reward computation | %s | prompts=%d | completions=%d | reward_mean=%.4f | reward_std=%.4f | q_range=[%.4f, %.4f] | q_temperature=%.3f | controller_tau=%s | beta=%s | eps=%.2e",
        _rank_tag(),
        len(grouped_comps),
        len(pair_batch.completions),
        moments.mean,
        moments.std,
        q_min,
        q_max,
        q_temperature,
        tau_repr,
        beta_repr,
        q_epsilon,
    )
    return RewardComputation(
        total_utils=total_utils,
        per_reward_values=per_reward_values,
        advantage=advantage_stats,
        pairs=pair_batch,
        q_distribution=q_distribution,
        moments=moments,
        ref_logprob_meta=flat_ref_meta,
        completion_metadata=completion_metadata,
        seed_semantic_entropies=seed_semantic_entropies,
        seed_advantage_scales=seed_advantage_scales,
        seed_alpha_effective=seed_alpha_effective,
        seed_max_possible_entropy=seed_max_possible_entropy,
    )


def _coerce_reward_names(raw_names: Any) -> List[str]:
    """Return a list of reward identifiers from arbitrary inputs."""

    if not raw_names:
        return []
    if isinstance(raw_names, str):
        return [raw_names]
    try:
        sequence = list(raw_names)
    except TypeError:
        return [str(raw_names)]
    names: List[str] = []
    for name in sequence:
        if name is None:
            continue
        names.append(str(name))
    return names


def _has_recipe_path(obj: Any) -> bool:
    """Return ``True`` when the object carries a recipe path marker."""

    return bool(getattr(obj, "recipe_path", None))


def _build_reward_proxy(source: Any, reward_names: List[str]) -> RewardConfig:
    """Preserve source config attributes when instantiating reward helpers."""

    proxy_data: Dict[str, Any] = {"reward_funcs": list(reward_names)}
    if source is not None:
        try:
            source_data = vars(source)
        except TypeError:
            source_data = None
        if isinstance(source_data, dict):
            proxy_data.update(source_data)
    proxy_data["reward_funcs"] = list(reward_names)
    return cast(RewardConfig, SimpleNamespace(**proxy_data))


def load_reward_functions(
    script_args: Any, tokenizer: Any, training_args: Any = None
) -> Tuple[list, list]:
    """Resolve reward functions/weights from script or training args.

    :param script_args: Script arguments carrying reward names/weights.
    :param tokenizer: Tokenizer passed to reward function factory helpers.
    :param training_args: Optional training config that can override script rewards.
    :returns: Tuple of ``(reward_funcs, reward_weights)``.
    :rtype: tuple[list, list]
    """

    def _resolve_rewards(source: Any) -> Tuple[List[str], Optional[List[float]]]:
        if source is None:
            return [], None
        names = _coerce_reward_names(getattr(source, "reward_funcs", None))
        weights = getattr(source, "reward_weights", None)
        return names, weights

    script_names, script_weights = _resolve_rewards(script_args)
    training_names, training_weights = _resolve_rewards(training_args)
    use_training = False
    if training_names:
        if not script_names or training_names != script_names:
            use_training = True
    if use_training:
        reward_names = training_names
        weight_source = training_weights
        proxy_source = training_args
    elif script_names:
        reward_names = script_names
        weight_source = script_weights
        proxy_source = script_args
    else:
        reward_names = ["pure_accuracy_math"]
        weight_source = None
        proxy_source = script_args if script_args is not None else training_args
    proxy = _build_reward_proxy(proxy_source, reward_names)
    reward_funcs = get_reward_funcs(proxy, None, tokenizer)
    reward_weights = weight_source
    if reward_weights is None or len(reward_weights) != len(reward_funcs):
        reward_weights = [1.0] * len(reward_funcs)
    return reward_funcs, reward_weights


def load_eval_reward_functions(
    script_args: Any, tokenizer: Any, training_args: Any = None
) -> Tuple[list, list]:
    """Resolve eval reward functions/weights, defaulting to training rewards.

    :param script_args: Script arguments containing eval-specific reward settings.
    :param tokenizer: Tokenizer passed to reward function factory helpers.
    :param training_args: Optional training config with reward overrides.
    :returns: Tuple of ``(reward_funcs, reward_weights)`` for evaluation.
    :rtype: tuple[list, list]
    """

    script_eval_names = _coerce_reward_names(
        getattr(script_args, "eval_reward_funcs", None)
    )
    script_eval_weights = getattr(script_args, "eval_reward_weights", None)
    script_train_names = _coerce_reward_names(
        getattr(script_args, "reward_funcs", None)
    )
    script_train_weights = getattr(script_args, "reward_weights", None)
    training_names = (
        _coerce_reward_names(getattr(training_args, "reward_funcs", None))
        if training_args is not None
        else []
    )
    training_weights = (
        getattr(training_args, "reward_weights", None)
        if training_args is not None
        else None
    )
    if script_eval_names:
        reward_names = script_eval_names
        weight_source = script_eval_weights
        proxy_source = script_args
    else:
        use_training = False
        if training_names:
            if not script_train_names or training_names != script_train_names:
                use_training = True
        if script_train_names and not use_training:
            reward_names = script_train_names
            weight_source = script_train_weights
            proxy_source = script_args
        elif training_names:
            reward_names = training_names
            weight_source = training_weights
            proxy_source = training_args
        else:
            reward_names = ["pure_accuracy_math"]
            weight_source = None
            proxy_source = script_args if script_args is not None else training_args
    proxy = _build_reward_proxy(proxy_source, reward_names)
    reward_funcs = get_reward_funcs(proxy, None, tokenizer)

    reward_weights = weight_source
    if reward_weights is None or len(reward_weights) != len(reward_funcs):
        reward_weights = [1.0] * len(reward_funcs)
    return reward_funcs, reward_weights


__all__ = [
    "compute_reward_statistics",
    "prepare_generation_batch",
    "load_reward_functions",
    "load_eval_reward_functions",
]
