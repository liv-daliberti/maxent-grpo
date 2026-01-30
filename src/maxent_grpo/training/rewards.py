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
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING, cast
from types import SimpleNamespace

from maxent_grpo.generation import (
    AggregatedGenerationState,
    drop_empty_prompt_groups,
    flatten_prompt_completions as _flatten_prompt_completions,
    flatten_ref_metadata as _flatten_ref_metadata,
    retry_incomplete_prompts,
    seed_generation_groups,
    truncate_to_expected_counts,
)
from maxent_grpo.generation.errors import (
    GenerationServiceError,
    log_generation_service_error,
)
from maxent_grpo.training.runtime import require_torch
from .run_helpers import _group_softmax
from maxent_grpo.rewards.basic import RewardConfig, get_reward_funcs
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


def _extract_ref_logprob_fields(meta_entry: Any) -> Tuple[Optional[Any], Optional[Any]]:
    """Return ``(logprob_sum, token_count)`` when present in metadata entries."""

    if meta_entry is None:
        return None, None
    logprob_sum = getattr(meta_entry, "logprob_sum", None)
    token_count = getattr(meta_entry, "token_count", None)
    if isinstance(meta_entry, dict):
        if logprob_sum is None:
            logprob_sum = meta_entry.get("logprob_sum") or meta_entry.get(
                "cumulative_logprob"
            )
        if token_count is None:
            token_count = meta_entry.get("token_count") or meta_entry.get("num_tokens")
            if token_count is None:
                tok_logs = meta_entry.get("token_logprobs") or meta_entry.get("logprobs")
                if tok_logs is not None:
                    try:
                        token_count = len(tok_logs)
                    except (TypeError, ValueError):
                        token_count = None
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
                "Dropping incomplete reference logprob metadata | missing_entries=%d/%d | first_missing_idx=%d",
                len(missing_idx),
                total_sequences,
                missing_idx[0],
            )
            setattr(_sanitize_ref_logprob_meta, "_warned", True)
        return None
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
        if reward_weight != 1.0:
            reward_values = [float(reward_weight) * val for val in reward_values]
        total_utils = [util + val for util, val in zip(total_utils, reward_values)]
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
        import math

        mean_val = sum(total_utils) / len(total_utils)
        train_reward_mean = float(mean_val)
        if len(total_utils) > 1:
            var = sum((u - mean_val) ** 2 for u in total_utils) / len(total_utils)
            train_reward_std = float(math.sqrt(var))
        else:
            train_reward_std = 0.0
        return train_reward_mean, train_reward_std
    except (TypeError, ZeroDivisionError, ValueError, OverflowError):
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
    seed_augmentation: Optional[Any] = None,
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
    :param seed_augmentation: Optional InfoSeed augmentation config.
    :type seed_augmentation: Any | None
    :returns: Populated :class:`~training.types.GenerationBatch` or ``None`` if
        generation fails after retries.
    :rtype: :class:`~training.types.GenerationBatch` | None
    """
    prompts: List[str] = batch["prompt"]
    answers: List[str] = batch["answer"]
    if not prompts:
        return None
    seed_cfg = None
    if seed_augmentation is not None:
        if hasattr(seed_augmentation, "is_active"):
            if seed_augmentation.is_active():
                seed_cfg = seed_augmentation
        elif (
            getattr(seed_augmentation, "enabled", False)
            and getattr(seed_augmentation, "num_seeds", 0) > 0
            and getattr(seed_augmentation, "completions_per_seed", 0) > 0
        ):
            seed_cfg = seed_augmentation
    LOG.debug(
        "Starting completion generation | prompts=%d | expected_generations=%d",
        len(prompts),
        expected_generations,
    )

    def _call_generator(
        prompt_batch: List[str],
        expected: int,
        per_prompt_counts: Optional[List[int]] = None,
    ) -> Any:
        per_prompt_repr = "none"
        if per_prompt_counts is not None:
            try:
                per_prompt_repr = f"len={len(per_prompt_counts)} first3={list(per_prompt_counts)[:3]}"
            except (TypeError, ValueError):
                per_prompt_repr = str(per_prompt_counts)
        LOG.debug(
            "Invoking generator | prompts=%d | expected=%d | per_prompt_counts=%s",
            len(prompt_batch),
            expected,
            per_prompt_repr,
        )
        try:
            result = generator(prompt_batch, expected, per_prompt_counts)
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
        "Generator output unpacked | grouped_type=%s | meta_present=%s",
        type(grouped_comps).__name__,
        grouped_meta is not None,
    )
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
    LOG.debug(
        "Retrying incomplete prompts | initial_groups=%d | expected=%d",
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
        "Retries done | prompts=%d | groups=%d",
        len(prompts),
        len(aggregated_comps) if aggregated_comps is not None else 0,
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
    LOG.debug(
        "Truncated completions | prompts=%d | expected=%d | partial=%d",
        len(prompts),
        expected_generations,
        partial_count,
    )
    if partial_count > 0:
        generation_stats.setdefault("partial_prompts", 0)
        generation_stats["partial_prompts"] += partial_count
    completion_info: List[List[dict]] = [
        [{"seed_id": None, "is_seed_aug": False} for _ in group]
        for group in aggregated_comps
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

        def _extract_token_ids(entry_dict: Optional[Dict[str, Any]]) -> Optional[List[int]]:
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
            if isinstance(token_ids, list) and token_ids and isinstance(token_ids[0], list):
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
                meta_entry = meta_group[comp_idx] if comp_idx < len(meta_group) else None
                meta_dict = _meta_to_dict(meta_entry)
                token_ids = _extract_token_ids(meta_dict)
                if token_ids is not None:
                    completion_info[prompt_idx][comp_idx]["token_ids"] = token_ids
    # Optionally augment with seed-tagged completions to support InfoSeed-GRPO.
    if seed_cfg:
        try:
            import random as _random
        except ImportError:  # pragma: no cover - fallback in minimal envs
            _random = None
        rng = _random.Random() if _random is not None else None
        seed_prompts: List[str] = []
        seed_answers: List[str] = []
        seed_parent: List[int] = []
        seed_ids: List[int] = []
        num_seeds = max(int(getattr(seed_cfg, "num_seeds", 0)), 0)
        comps_per_seed = max(int(getattr(seed_cfg, "completions_per_seed", 0)), 0)
        template = getattr(seed_cfg, "template", "\n[seed={seed}]") or "{prompt}"
        for idx, prompt_text in enumerate(prompts):
            for _ in range(num_seeds):
                seed_id = (
                    rng.randint(1, num_seeds)
                    if rng is not None and num_seeds > 0
                    else 1
                )
                if "{prompt}" in template:
                    seed_prompt = template.format(prompt=prompt_text, seed=seed_id)
                else:
                    seed_prompt = f"{prompt_text}{template.format(seed=seed_id)}"
                seed_prompts.append(seed_prompt)
                seed_answers.append(answers[idx])
                seed_parent.append(idx)
                seed_ids.append(seed_id)
        if seed_prompts and comps_per_seed > 0:
            LOG.debug(
                "Invoking seed generator | prompts=%d | completions_per_seed=%d",
                len(seed_prompts),
                comps_per_seed,
            )
            seed_result = _call_generator(seed_prompts, comps_per_seed)
            if isinstance(seed_result, GenerationBatch):
                seed_grouped = seed_result.grouped_completions
                seed_meta = getattr(seed_result, "grouped_ref_meta", None)
            elif hasattr(seed_result, "grouped_completions"):
                seed_grouped = getattr(seed_result, "grouped_completions")
                seed_meta = getattr(seed_result, "grouped_ref_meta", None)
            elif seed_result is None:
                seed_grouped, seed_meta = [], None
            else:
                seed_grouped, seed_meta = seed_result
            seed_grouped, seed_meta = seed_generation_groups(
                len(seed_prompts),
                seed_grouped,
                seed_meta,
            )
            LOG.debug(
                "Seed generation finished | prompts=%d | groups_returned=%d",
                len(seed_prompts),
                len(seed_grouped) if seed_grouped is not None else 0,
            )
            seed_grouped, seed_meta, _ = truncate_to_expected_counts(
                seed_grouped, seed_meta, comps_per_seed
            )
            if len(seed_grouped) < len(seed_prompts):
                seed_grouped.extend([[]] * (len(seed_prompts) - len(seed_grouped)))
            if seed_meta is not None and len(seed_meta) < len(seed_prompts):
                seed_meta.extend(
                    [[] for _ in range(len(seed_prompts) - len(seed_meta))]
                )
            for idx, comps in enumerate(seed_grouped):
                parent_idx = seed_parent[idx] if idx < len(seed_parent) else None
                if parent_idx is None or parent_idx >= len(aggregated_comps):
                    continue
                if not comps:
                    continue
                seed_tag = seed_ids[idx] if idx < len(seed_ids) else None
                aggregated_comps[parent_idx].extend(comps)
                while completion_info and len(completion_info) <= parent_idx:
                    completion_info.append([])
                completion_info[parent_idx].extend(
                    [
                        {"seed_id": seed_tag, "is_seed_aug": True}
                        for _ in range(len(comps))
                    ]
                )
                if seed_meta is not None:
                    while aggregated_meta is None or len(aggregated_meta) <= parent_idx:
                        aggregated_meta = aggregated_meta or []
                        aggregated_meta.append([])
                    meta_group = seed_meta[idx] if idx < len(seed_meta) else None
                    meta_group = meta_group or [None] * len(comps)
                    aggregated_meta[parent_idx].extend(meta_group[: len(comps)])
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
        return None
    pair_batch, flat_answers = _flatten_prompt_completions(gen_batch)
    if not pair_batch.completions:
        return None
    completion_metadata = getattr(pair_batch, "metadata", None)
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
        "Reward computation | prompts=%d | completions=%d | reward_mean=%.4f | reward_std=%.4f | q_range=[%.4f, %.4f] | q_temperature=%.3f | controller_tau=%s | beta=%s | eps=%.2e",
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
    elif script_names:
        reward_names = script_names
        weight_source = script_weights
    else:
        reward_names = ["pure_accuracy_math"]
        weight_source = None
    proxy = cast(RewardConfig, SimpleNamespace(reward_funcs=reward_names))
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
    else:
        use_training = False
        if training_names:
            if not script_train_names or training_names != script_train_names:
                use_training = True
        if script_train_names and not use_training:
            reward_names = script_train_names
            weight_source = script_train_weights
        elif training_names:
            reward_names = training_names
            weight_source = training_weights
        else:
            reward_names = ["pure_accuracy_math"]
            weight_source = None
    proxy = cast(RewardConfig, SimpleNamespace(reward_funcs=reward_names))
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
