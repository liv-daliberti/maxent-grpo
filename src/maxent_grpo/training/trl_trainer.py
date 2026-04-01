"""Custom TRL GRPOTrainer wrapper used by the MaxEnt-GRPO pipelines.

This module is the single place where GRPO-vs-MaxEnt objective behavior should
diverge at runtime. The surrounding training pipeline (dataset mapping, reward
loading, model/tokenizer setup, trainer wiring, launch entrypoints) is kept
shared so objective comparisons stay fair and easy to audit.
"""

from __future__ import annotations
# pylint: disable=broad-exception-caught

import logging
import math
import os
import json
from contextlib import contextmanager, nullcontext
from collections.abc import Mapping
from functools import partial
import inspect
import re
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Type, cast

import torch
import torch.nn.functional as F

AutoModelForCausalLM = None  # type: ignore[assignment]

try:
    from accelerate.utils import gather
except (ImportError, ModuleNotFoundError):  # pragma: no cover - test fallback

    def gather(value: Any) -> Any:
        return value


try:
    from trl.data_utils import (
        apply_chat_template,
        is_conversational,
        maybe_apply_chat_template,
    )
except (ImportError, ModuleNotFoundError):  # pragma: no cover - test fallback

    def apply_chat_template(example: Any, _tokenizer: Any) -> Dict[str, str]:
        return {"text": str(example)}

    def is_conversational(example: Any) -> bool:
        return isinstance(example, list)

    def maybe_apply_chat_template(example: Any, _tokenizer: Any) -> Dict[str, str]:
        return {"prompt": str(example)}

_trl_create_reference_model = None
_trl_prepare_deepspeed = None
_trl_prepare_fsdp = None


from maxent_grpo.rewards.basic import (
    pure_accuracy_math_correctness,
    truncate_after_first_boxed_answer,
    uses_pure_accuracy_math_reward,
)
from maxent_grpo.methods import resolve_method_spec_from_args
from maxent_grpo.objectives import resolve_objective_routing
from maxent_grpo.training.rewards import _compute_seed_grpo_statistics
from maxent_grpo.training.controller_objective import (
    ControllerMetaContext,
    build_controller_objective,
)
from maxent_grpo.training.telemetry.trl_logging import ensure_weighting_logging
from maxent_grpo.training.weighting import (
    apply_meta_controller_update,
    collect_weight_entropy,
    maybe_update_beta,
    maybe_update_tau,
    weight_matrix_from_q,
)
from maxent_grpo.training.weighting.logic import build_weighting_settings
from maxent_grpo.training.scoring_common import (
    _coerce_optional_int,
    _get_config_value,
    _get_embedding_vocab_size,
)

LOG = logging.getLogger(__name__)
_PASS_METRIC_SUCCESS_REWARD = 1.0
_PASS_METRIC_EPS = 1e-6


@contextmanager
def _adapter_disabled_context(model: Any):
    """Disable adapters when the model exposes a supported API.

    This trainer runs both plain Transformers models and PEFT-enabled models.
    Older PEFT integrations expose ``disable_adapter()`` as a context manager,
    while newer Transformers PEFT shims expose ``disable_adapters()`` /
    ``enable_adapters()`` as imperative methods. Plain base models expose
    neither and should be treated as a no-op.
    """

    disable_adapter = getattr(model, "disable_adapter", None)
    if callable(disable_adapter):
        with disable_adapter():
            yield
        return

    disable_adapters = getattr(model, "disable_adapters", None)
    enable_adapters = getattr(model, "enable_adapters", None)
    if callable(disable_adapters) and callable(enable_adapters):
        try:
            disable_adapters()
        except ValueError as exc:
            if "PEFT is not installed" in str(exc):
                with nullcontext():
                    yield
                return
            raise
        try:
            yield
        finally:
            try:
                enable_adapters()
            except ValueError as exc:
                if "PEFT is not installed" not in str(exc):
                    raise
        return

    with nullcontext():
        yield
_LOG_DELTA_CLAMP = 5.0
_BENCHMARK_SUFFIX_SANITIZER = re.compile(r"[^A-Za-z0-9]+")
_EMA_PARAM_NAME_PREFIXES: Tuple[str, ...] = (
    "_fsdp_wrapped_module.",
    "_checkpoint_wrapped_module.",
    "base_model.model.",
    "module.",
    "model.",
)


def _mean(values: List[float]) -> float:
    """Return the arithmetic mean for a non-empty list, else 0.0."""
    return float(sum(values)) / float(len(values)) if values else 0.0


def _weighted_mean(values: List[float], weights: List[float]) -> float:
    """Return the weighted mean or 0.0 when weights are empty."""
    if not values or not weights:
        return 0.0
    total_weight = float(sum(weights))
    if total_weight <= 0.0:
        return 0.0
    return float(sum(v * w for v, w in zip(values, weights))) / total_weight


def _nanmin_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Return the min value while ignoring NaNs."""
    finite = tensor[~torch.isnan(tensor)]
    if finite.numel() == 0:
        return torch.tensor(float("nan"), dtype=tensor.dtype, device=tensor.device)
    return finite.min()


def _nanmax_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Return the max value while ignoring NaNs."""
    finite = tensor[~torch.isnan(tensor)]
    if finite.numel() == 0:
        return torch.tensor(float("nan"), dtype=tensor.dtype, device=tensor.device)
    return finite.max()


def _clamp_log_delta(delta: torch.Tensor) -> torch.Tensor:
    """Clamp log-probability deltas before exponentiating."""

    return delta.float().clamp(min=-_LOG_DELTA_CLAMP, max=_LOG_DELTA_CLAMP)


def _resolve_vocab_size_limit(model: Any) -> Optional[int]:
    """Return the smallest positive vocab-size limit exposed by the model."""

    config = getattr(model, "config", None)
    embedding_vocab_size = _get_embedding_vocab_size(model, config)
    config_vocab_size = _coerce_optional_int(_get_config_value(config, "vocab_size", None))
    attr_vocab_size = _coerce_optional_int(getattr(model, "vocab_size", None))
    candidates = [
        int(value)
        for value in (embedding_vocab_size, config_vocab_size, attr_vocab_size)
        if isinstance(value, int) and int(value) > 0
    ]
    if not candidates:
        return None
    return max(candidates)


def _resolve_tokenizer_vocab_limit(tokenizer: Any) -> Optional[int]:
    """Return the full positive vocab-size limit exposed by the tokenizer."""

    candidates: List[int] = []
    vocab_size = _coerce_optional_int(getattr(tokenizer, "vocab_size", None))
    if isinstance(vocab_size, int) and vocab_size > 0:
        candidates.append(int(vocab_size))
    try:
        tokenizer_len = _coerce_optional_int(len(tokenizer))
    except Exception:
        tokenizer_len = None
    if isinstance(tokenizer_len, int) and tokenizer_len > 0:
        candidates.append(int(tokenizer_len))
    if not candidates:
        return None
    # `tokenizer.vocab_size` often excludes added special tokens while
    # `len(tokenizer)` includes them; use the larger addressable range.
    return max(candidates)


def _resolve_token_id_upper_bound(model: Any, tokenizer: Any = None) -> Optional[int]:
    """Return a conservative upper bound for valid token IDs."""

    candidates: List[int] = []
    model_limit = _resolve_vocab_size_limit(model)
    if isinstance(model_limit, int) and model_limit > 0:
        candidates.append(int(model_limit))
    tokenizer_limit = _resolve_tokenizer_vocab_limit(tokenizer)
    if isinstance(tokenizer_limit, int) and tokenizer_limit > 0:
        candidates.append(int(tokenizer_limit))
    if not candidates:
        return None
    return min(candidates)


def _mask_invalid_logit_columns(
    logits: torch.Tensor,
    *,
    valid_vocab_size: Optional[int],
) -> torch.Tensor:
    """Mask logit columns that correspond to tokenizer-inaccessible token IDs.

    Some Qwen checkpoints expose larger output embeddings than the tokenizer can
    address. Leaving those extra columns active lets entropy-regularized losses
    push probability mass into dead token rows, which later surface as sampled
    token IDs outside the tokenizer range.
    """

    if not isinstance(valid_vocab_size, int) or valid_vocab_size <= 0:
        return logits
    if logits.ndim < 1:
        return logits
    last_dim = int(logits.size(-1))
    if last_dim <= valid_vocab_size:
        return logits
    masked = logits.clone()
    mask_value = torch.finfo(masked.dtype).min
    masked[..., valid_vocab_size:] = mask_value
    return masked


def _entropy_normalization_scale(valid_vocab_size: Optional[int]) -> float:
    """Return the log-vocab normalization constant for exact entropy metrics."""

    if not isinstance(valid_vocab_size, int) or valid_vocab_size <= 1:
        return 1.0
    try:
        scale = float(math.log(float(valid_vocab_size)))
    except (TypeError, ValueError, OverflowError):
        return 1.0
    if not math.isfinite(scale) or scale <= 0.0:
        return 1.0
    return scale


def _tokenize_for_diversity(text: str, tokenizer: Any = None) -> List[Any]:
    """Tokenize a completion for diversity metrics."""
    if not text:
        return []
    if tokenizer is not None:
        try:
            encode = getattr(tokenizer, "encode", None)
            if callable(encode):
                return list(encode(text, add_special_tokens=False))
            if callable(tokenizer):
                tokenized = tokenizer(text, add_special_tokens=False)
                if isinstance(tokenized, dict) and "input_ids" in tokenized:
                    return list(tokenized["input_ids"])
                if isinstance(tokenized, (list, tuple)):
                    return list(tokenized)
        except Exception:
            pass
    return [tok for tok in text.strip().split() if tok]


def _completion_diversity_metrics(
    grouped_completions: List[List[str]],
    *,
    tokenizer: Any = None,
    accelerator: Any = None,
) -> Dict[str, float]:
    """Return coarse diversity metrics for grouped completions."""
    if not grouped_completions:
        return {}

    def _distinct_n(tokens: List[Any], n: int) -> float:
        if n <= 0 or len(tokens) < n:
            return 0.0
        total = len(tokens) - n + 1
        if total <= 0:
            return 0.0
        ngrams = {tuple(tokens[i : i + n]) for i in range(total)}
        return float(len(ngrams)) / float(total)

    def _jaccard_distance(sets: List[set[Any]]) -> float:
        if len(sets) < 2:
            return 0.0
        total_dist = 0.0
        pairs = 0
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                a = sets[i]
                b = sets[j]
                union = a | b
                if not union:
                    dist = 0.0
                else:
                    dist = 1.0 - (len(a & b) / float(len(union)))
                total_dist += dist
                pairs += 1
        return total_dist / float(pairs) if pairs > 0 else 0.0

    group_metrics: List[Dict[str, float]] = []
    for group in grouped_completions:
        if not group:
            continue
        normalized = [comp.strip() for comp in group if comp is not None]
        group_size = len(normalized)
        if group_size <= 0:
            continue
        all_tokens: List[Any] = []
        token_sets: List[set[Any]] = []
        for comp in normalized:
            tokens = _tokenize_for_diversity(comp, tokenizer)
            if tokens:
                all_tokens.extend(tokens)
                token_sets.append(set(tokens))
            else:
                token_sets.append(set())
        group_metrics.append(
            {
                "group_size": float(group_size),
                "distinct_1": _distinct_n(all_tokens, 1),
                "distinct_2": _distinct_n(all_tokens, 2),
                "jaccard": _jaccard_distance(token_sets),
            }
        )

    if not group_metrics:
        return {}

    if accelerator is not None and getattr(accelerator, "num_processes", 1) > 1:
        gather_fn = getattr(accelerator, "gather_object", None)
        if callable(gather_fn):
            try:
                gather_fn_typed = cast(Callable[[Any], Any], gather_fn)
                gathered = gather_fn_typed(group_metrics)  # pylint: disable=not-callable
                if isinstance(gathered, list):
                    merged: List[Dict[str, float]] = []
                    for item in gathered:
                        if isinstance(item, list):
                            merged.extend([m for m in item if isinstance(m, dict)])
                        elif isinstance(item, dict):
                            merged.append(item)
                    if merged:
                        group_metrics = merged
            except Exception:
                pass
        else:
            dist = getattr(torch, "distributed", None)
            if (
                dist is not None
                and callable(getattr(dist, "is_available", None))
                and callable(getattr(dist, "is_initialized", None))
                and dist.is_available()
                and dist.is_initialized()
            ):
                try:
                    world = int(getattr(dist, "get_world_size")())
                except (TypeError, ValueError, RuntimeError):
                    world = 0
                if world > 1:
                    try:
                        gathered = [None for _ in range(world)]
                        gather_obj = getattr(dist, "all_gather_object", None)
                        if callable(gather_obj):
                            gather_obj(gathered, group_metrics)
                            merged: List[Dict[str, float]] = []
                            for item in gathered:
                                if isinstance(item, list):
                                    merged.extend(
                                        [m for m in item if isinstance(m, dict)]
                                    )
                                elif isinstance(item, dict):
                                    merged.append(item)
                            if merged:
                                group_metrics = merged
                    except (RuntimeError, ValueError, TypeError):
                        pass

    distinct1_vals = [m["distinct_1"] for m in group_metrics if "distinct_1" in m]
    distinct2_vals = [m["distinct_2"] for m in group_metrics if "distinct_2" in m]
    jaccard_vals = [m["jaccard"] for m in group_metrics if "jaccard" in m]
    weights = [m.get("group_size", 0.0) for m in group_metrics]
    return {
        "distinct_1": _mean(distinct1_vals),
        "distinct_2": _mean(distinct2_vals),
        "jaccard": _mean(jaccard_vals),
        "distinct_1_micro": _weighted_mean(distinct1_vals, weights),
        "distinct_2_micro": _weighted_mean(distinct2_vals, weights),
        "jaccard_micro": _weighted_mean(jaccard_vals, weights),
    }


def _is_main_process(trainer: Any) -> bool:
    """Return whether the active trainer rank should emit shared metrics."""

    accelerator = getattr(trainer, "accelerator", None)
    return bool(getattr(accelerator, "is_main_process", True))


def _use_lightweight_greedy_eval(trainer: Any, mode: str) -> bool:
    """Return whether training-time eval is using the lightweight greedy path."""

    if mode != "eval":
        return False
    args = getattr(trainer, "args", None)
    return bool(getattr(args, "eval_greedy_only_enabled", False))


def _use_sharded_prompt_major_greedy_eval(trainer: Any, mode: str) -> bool:
    """Return whether greedy-only eval should shard prompt-major batches across ranks."""

    if not _use_lightweight_greedy_eval(trainer, mode):
        return False
    args = getattr(trainer, "args", None)
    if not bool(getattr(args, "disable_distributed_sampler", False)):
        return False
    accelerator = getattr(trainer, "accelerator", None)
    try:
        num_processes = int(getattr(accelerator, "num_processes", 1) or 1)
    except (TypeError, ValueError):
        num_processes = 1
    return num_processes > 1


def _use_local_only_lightweight_eval_metrics(trainer: Any, mode: str) -> bool:
    """Return whether greedy-only eval should stay main-rank-only for metrics."""

    return _use_lightweight_greedy_eval(
        trainer, mode
    ) and not _use_sharded_prompt_major_greedy_eval(trainer, mode)


def _use_local_only_eval_diversity_metrics(trainer: Any, mode: str) -> bool:
    """Return whether eval diversity logging should stay local to main rank.

    Full eval still runs through the standard Trainer loop, but completion
    diversity logging is auxiliary and uses Python-object gathers that have been
    the concrete failure mode under DDP. When eval inputs are replicated across
    ranks (the stable math configs set ``disable_distributed_sampler=True``), we
    can compute those diversity summaries on the main rank only without any
    cross-rank synchronization.
    """

    if mode != "eval":
        return False
    args = getattr(trainer, "args", None)
    if not bool(getattr(args, "disable_distributed_sampler", False)):
        return False
    accelerator = getattr(trainer, "accelerator", None)
    try:
        num_processes = int(getattr(accelerator, "num_processes", 1) or 1)
    except (TypeError, ValueError):
        num_processes = 1
    return num_processes > 1


def _metric_tensor_for_logging(
    trainer: Any,
    value: Any,
    *,
    mode: str,
) -> Optional[torch.Tensor]:
    """Return a metric tensor for logging, avoiding DDP gathers in local-only eval."""

    if not isinstance(value, torch.Tensor):
        return None
    if _use_sharded_prompt_major_greedy_eval(trainer, mode):
        gathered = gather(value)
        if not isinstance(gathered, torch.Tensor):
            return None
        return gathered
    if _use_local_only_lightweight_eval_metrics(trainer, mode):
        if not _is_main_process(trainer):
            return None
        return value
    gathered = gather(value)
    return gathered if isinstance(gathered, torch.Tensor) else None


def _local_metric_tensor(value: Any) -> Optional[torch.Tensor]:
    """Return a detached local metric tensor without any distributed gather."""

    if not isinstance(value, torch.Tensor):
        return None
    if value.numel() <= 0:
        return None
    return value.detach()


def _apply_eos_completion_mask(
    completion_ids: torch.Tensor,
    eos_token_id: Optional[int],
    completion_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Mask completion tokens after the first EOS token (TRL-style)."""
    if eos_token_id is None:
        if completion_mask is not None:
            return completion_mask
        return torch.ones_like(completion_ids, dtype=getattr(torch, "long", None))

    try:
        is_eos = completion_ids == eos_token_id
        batch = int(is_eos.size(0))
        seq_len = int(is_eos.size(1))
        eos_idx = torch.full(
            (batch,),
            seq_len,
            dtype=getattr(torch, "long", None),
            device=getattr(completion_ids, "device", None),
        )
        any_eos = is_eos.any(dim=1)
        if bool(any_eos.any()):
            eos_pos = is_eos.int().argmax(dim=1)
            eos_idx = eos_idx.clone()
            eos_idx[any_eos] = eos_pos[any_eos]
        seq_idx = torch.arange(
            seq_len, device=getattr(completion_ids, "device", None)
        ).unsqueeze(0)
        seq_idx = seq_idx.expand(batch, -1)
        mask = seq_idx <= eos_idx.unsqueeze(1)
        to_fn = getattr(mask, "to", None)
        if callable(to_fn):
            mask = to_fn(dtype=getattr(torch, "long", None))
        return cast(torch.Tensor, mask)
    except Exception:
        # Defensive fallback used for test doubles that only implement a subset
        # of tensor ops; preserve prior behavior by returning an all-ones mask.
        return torch.ones_like(completion_ids, dtype=getattr(torch, "long", None))


def _normalize_text_for_prefix_match(text: str) -> str:
    """Normalize text for lightweight decode-prefix comparisons."""

    return " ".join(str(text).split()).strip()


def _build_prompt_text(example: Dict[str, Any], tokenizer: Any) -> str:
    """Render one trainer example into the exact text sent to generation."""

    if not isinstance(example, dict):
        return str(example)

    prompt = example.get("prompt", "")
    if isinstance(prompt, list) and prompt:
        first_message = prompt[0]
        last_message = prompt[-1]
        if (
            isinstance(first_message, dict)
            and isinstance(last_message, dict)
            and "role" in first_message
            and "content" in first_message
            and "role" in last_message
        ):
            apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
            if callable(apply_chat_template):
                try:
                    last_role = str(last_message.get("role", ""))
                    add_generation_prompt = last_role == "user"
                    continue_final_message = last_role == "assistant"
                    return str(
                        apply_chat_template(
                            prompt,
                            tokenize=False,
                            add_generation_prompt=add_generation_prompt,
                            continue_final_message=continue_final_message,
                        )
                    )
                except Exception:
                    pass

    conversational_example: Dict[str, Any] | None = None
    if "prompt" in example:
        conversational_example = {"prompt": prompt}
    elif "messages" in example:
        conversational_example = {"messages": example.get("messages")}

    if (
        isinstance(conversational_example, dict)
        and is_conversational(conversational_example)
    ):
        try:
            rendered = maybe_apply_chat_template(conversational_example, tokenizer)
        except Exception:
            rendered = {"prompt": str(prompt)}
        if isinstance(rendered, dict):
            prompt_text = rendered.get("prompt")
            if prompt_text is None:
                prompt_text = rendered.get("text")
            if prompt_text is not None:
                return str(prompt_text)
        return str(rendered)
    return str(prompt)


def _normalize_group_mass_proxy(values: Sequence[float]) -> List[float]:
    """Convert a per-group signal into a non-negative mass proxy."""
    cleaned: List[float] = []
    for value in values:
        try:
            cleaned.append(float(value))
        except (TypeError, ValueError):
            cleaned.append(float("nan"))
    if not cleaned:
        return []
    if all(math.isfinite(val) and val >= 0.0 for val in cleaned):
        total = sum(cleaned)
        if total > 0.0:
            return [val / total for val in cleaned]
    positives = [max(val, 0.0) if math.isfinite(val) else 0.0 for val in cleaned]
    pos_total = sum(positives)
    if pos_total > 0.0:
        return [val / pos_total for val in positives]
    return [float("nan")] * len(cleaned)


def _build_rich_rollout_rows(
    *,
    step: int,
    group_size: int,
    prompt_texts: Sequence[str],
    completion_texts: Sequence[str],
    rewards: Sequence[float],
    advantages: Sequence[float],
    q_values: Optional[Sequence[float]] = None,
) -> tuple[list[str], list[list[Any]]]:
    """Build prompt-major rollout rows for within-group distribution analysis."""
    total_rows = min(
        len(prompt_texts),
        len(completion_texts),
        len(rewards),
        len(advantages),
    )
    if total_rows <= 0:
        return [], []
    q_flat = list(q_values or [])
    columns = [
        "step",
        "prompt_index",
        "completion_index",
        "group_size",
        "reward_rank_desc",
        "prompt",
        "completion",
        "reward_total",
        "advantage",
        "q_mass",
        "update_weight_raw",
        "update_mass_proxy",
    ]
    rows: List[List[Any]] = []
    effective_group = max(int(group_size), 1)
    for start in range(0, total_rows, effective_group):
        stop = min(start + effective_group, total_rows)
        local_rewards = [float(rewards[idx]) for idx in range(start, stop)]
        local_advantages = [float(advantages[idx]) for idx in range(start, stop)]
        local_q = (
            [float(q_flat[idx]) for idx in range(start, stop)]
            if len(q_flat) >= stop
            else [float("nan")] * (stop - start)
        )
        reward_order = sorted(
            range(stop - start),
            key=lambda idx: (-local_rewards[idx], idx),
        )
        reward_rank = {local_idx: rank + 1 for rank, local_idx in enumerate(reward_order)}
        use_q_mass = all(math.isfinite(val) for val in local_q)
        local_proxy = (
            _normalize_group_mass_proxy(local_q)
            if use_q_mass
            else _normalize_group_mass_proxy(local_advantages)
        )
        prompt_index = start // effective_group
        for local_idx, row_idx in enumerate(range(start, stop)):
            q_mass = local_q[local_idx] if use_q_mass else float("nan")
            update_weight_raw = q_mass if use_q_mass else local_advantages[local_idx]
            update_mass_proxy = (
                local_proxy[local_idx]
                if local_idx < len(local_proxy)
                else float("nan")
            )
            rows.append(
                [
                    int(step),
                    int(prompt_index),
                    int(local_idx),
                    int(stop - start),
                    int(reward_rank.get(local_idx, local_idx + 1)),
                    str(prompt_texts[row_idx]),
                    str(completion_texts[row_idx]),
                    float(local_rewards[local_idx]),
                    float(local_advantages[local_idx]),
                    float(q_mass),
                    float(update_weight_raw),
                    float(update_mass_proxy),
                ]
            )
    return columns, rows


def _write_rich_rollout_sidecar(
    *,
    output_dir: str,
    table_key: str,
    step: int,
    columns: Sequence[str],
    rows: Sequence[Sequence[Any]],
) -> Optional[str]:
    """Persist prompt-major rollout rows for downstream figure generation."""
    if not output_dir:
        return None
    try:
        sidecar_dir = os.path.join(output_dir, "rich_completions")
        os.makedirs(sidecar_dir, exist_ok=True)
        path = os.path.join(sidecar_dir, f"{table_key}_step_{int(step):06d}.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(
                {"columns": list(columns), "data": [list(row) for row in rows]},
                handle,
            )
        return path
    except OSError:
        return None


def _token_prefix_search_order(target_len: int, max_len: int) -> List[int]:
    """Return a small symmetric search window around a candidate prefix length."""

    if max_len <= 0:
        return []
    bounded = max(1, min(target_len, max_len))
    order = [bounded]
    radius = 1
    while radius <= 8:
        lower = bounded - radius
        upper = bounded + radius
        if lower >= 1:
            order.append(lower)
        if upper <= max_len:
            order.append(upper)
        radius += 1
    if max_len not in order:
        order.append(max_len)
    return order


def _find_token_prefix_len_for_text(
    tokenizer: Any,
    token_ids: List[int],
    target_text: str,
) -> Optional[int]:
    """Best-effort map a decoded text prefix back onto token prefix length."""

    if not token_ids:
        return None
    normalized_target = _normalize_text_for_prefix_match(target_text)
    if not normalized_target:
        return None
    encode = getattr(tokenizer, "encode", None)
    decode = getattr(tokenizer, "decode", None)
    if not callable(encode) or not callable(decode):
        return None
    try:
        encoded = list(encode(target_text, add_special_tokens=False))
    except Exception:
        encoded = []
    search_order = _token_prefix_search_order(len(encoded), len(token_ids))
    if not search_order:
        search_order = list(range(1, len(token_ids) + 1))
    for prefix_len in search_order:
        try:
            decoded = decode(token_ids[:prefix_len], skip_special_tokens=True)
        except Exception:
            continue
        if _normalize_text_for_prefix_match(decoded) == normalized_target:
            return prefix_len
    return None


def _pad_completion_rows(
    rows: List[List[int]],
    *,
    pad_token_id: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad variable-length completion token rows and return ids + mask tensors."""

    if not rows:
        empty = torch.empty((0, 0), dtype=torch.long, device=device)
        return empty, empty
    max_len = max(len(row) for row in rows)
    if max_len <= 0:
        empty = torch.empty((len(rows), 0), dtype=torch.long, device=device)
        mask = torch.empty((len(rows), 0), dtype=torch.long, device=device)
        return empty, mask
    completion_ids = torch.full(
        (len(rows), max_len),
        int(pad_token_id),
        dtype=torch.long,
        device=device,
    )
    completion_mask = torch.zeros(
        (len(rows), max_len),
        dtype=torch.long,
        device=device,
    )
    for idx, row in enumerate(rows):
        if not row:
            continue
        width = len(row)
        completion_ids[idx, :width] = torch.tensor(
            row,
            dtype=torch.long,
            device=device,
        )
        completion_mask[idx, :width] = 1
    return completion_ids, completion_mask


def _pad_logprob_rows(
    rows: List[torch.Tensor],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Pad per-token log-prob rows with zeros to a dense tensor."""

    if not rows:
        return torch.empty((0, 0), dtype=dtype, device=device)
    max_len = max(int(row.numel()) for row in rows)
    if max_len <= 0:
        return torch.empty((len(rows), 0), dtype=dtype, device=device)
    padded = torch.zeros((len(rows), max_len), dtype=dtype, device=device)
    for idx, row in enumerate(rows):
        width = int(row.numel())
        if width <= 0:
            continue
        padded[idx, :width] = row.to(device=device, dtype=dtype)
    return padded


def _metric_suffix_from_benchmark(name: Any) -> str:
    """Return a metric-safe benchmark suffix (e.g., ``AIME24``)."""

    text = str(name).strip()
    if not text:
        return "UNKNOWN"
    cleaned = _BENCHMARK_SUFFIX_SANITIZER.sub("_", text).strip("_").upper()
    return cleaned or "UNKNOWN"


def _gather_eval_benchmark_ids_for_prompts(
    trainer: Any,
    prompt_inputs: List[Dict[str, Any]],
    *,
    device: torch.device,
    local_only: bool = False,
) -> Optional[torch.Tensor]:
    """Return gathered prompt-major benchmark ids when present."""

    if not prompt_inputs:
        return None
    keys = ("eval_benchmark_id", "benchmark_id")
    raw_vals: Optional[List[Any]] = None
    for key in keys:
        candidate = [example.get(key) for example in prompt_inputs]
        if candidate and any(val is not None for val in candidate):
            raw_vals = candidate
            break
    if not raw_vals:
        return None
    ids: List[int] = []
    for val in raw_vals:
        try:
            ids.append(int(val) if val is not None else -1)
        except (TypeError, ValueError):
            ids.append(-1)
    ids_tensor = torch.tensor(ids, dtype=torch.long, device=device)
    if local_only:
        if not _is_main_process(trainer):
            return None
        return ids_tensor
    gathered = gather(ids_tensor)
    if not isinstance(gathered, torch.Tensor) or gathered.numel() <= 0:
        return None
    return gathered.to(torch.long)


def _empty_dataset_like(dataset: Any) -> Any:
    """Return an empty dataset preserving the input dataset type when possible."""

    if dataset is None:
        return []
    select_fn = getattr(dataset, "select", None)
    if callable(select_fn):
        try:
            return select_fn([])
        except Exception:
            pass
    if isinstance(dataset, list):
        return []
    if isinstance(dataset, tuple):
        return tuple()
    try:
        return dataset[:0]
    except Exception:
        return []


def _build_seed_worker(num_workers: int, rank: int):
    """Return a worker_init_fn compatible with the active transformers seed_worker signature."""
    try:
        from transformers.trainer_utils import seed_worker as hf_seed_worker
    except Exception:  # pragma: no cover - transformers is required for training
        return None
    try:
        params = list(inspect.signature(hf_seed_worker).parameters)
    except (TypeError, ValueError):  # pragma: no cover - defensive fallback
        return hf_seed_worker
    if len(params) <= 1:
        return hf_seed_worker
    return partial(hf_seed_worker, num_workers=num_workers, rank=rank)


def _numeric_or_none(value: Any) -> Optional[float]:
    """Best-effort numeric conversion used for logging filters."""
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        item_fn = getattr(value, "item", None)
        if callable(item_fn):
            try:
                return float(item_fn())
            except (TypeError, ValueError):
                return None
    return None


_BOOL_TRUE = {"1", "true", "t", "yes", "y", "on"}
_BOOL_FALSE = {"0", "false", "f", "no", "n", "off", ""}


def _coerce_bool(value: Any, *, default: bool) -> bool:
    """Convert flexible config values to bool without surprising string truthiness."""
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in _BOOL_TRUE:
            return True
        if lowered in _BOOL_FALSE:
            return False
        return default
    try:
        return bool(value)
    except Exception:
        return default


def _coerce_non_negative_float(value: Any, *, default: float = 0.0) -> float:
    """Convert config values to a finite non-negative float."""
    numeric = _numeric_or_none(value)
    if numeric is None or not math.isfinite(numeric):
        return default
    return max(float(numeric), 0.0)


def _reshape_prompt_major_tensor(
    tensor: torch.Tensor,
    group_size: int,
) -> Optional[torch.Tensor]:
    """Reshape prompt-major flat rollouts into ``[prompts, generations, ...]``."""
    if group_size <= 0:
        return None
    total_rows = int(tensor.size(0))
    if total_rows <= 0 or total_rows % group_size != 0:
        return None
    num_prompts = total_rows // group_size
    if num_prompts <= 0:
        return None
    shape = (num_prompts, group_size) + tuple(tensor.shape[1:])
    return tensor.reshape(shape).contiguous()


def _flatten_prompt_major_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Convert a prompt-major ``[prompts, generations, ...]`` tensor to flat order."""
    if tensor.dim() < 2:
        return tensor.reshape(-1)
    shape = (-1,) + tuple(tensor.shape[2:])
    return tensor.reshape(shape).contiguous()


def _resolve_prompt_group_sizes(
    tensor_dict: Dict[str, Optional[torch.Tensor]],
    group_size: int,
) -> Tuple[int, int]:
    """Infer flat row count and prompt count for listwise prompt groups."""
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    for key in (
        "completion_ids",
        "completion_mask",
        "advantages",
        "old_per_token_logps",
        "prompt_ids",
        "prompt_mask",
    ):
        tensor = tensor_dict.get(key)
        if isinstance(tensor, torch.Tensor):
            total_rows = int(tensor.size(0))
            break
    else:
        raise ValueError("Listwise prompt grouping requires flat rollout tensors.")
    usable = (total_rows // group_size) * group_size
    if usable <= 0:
        raise ValueError("Listwise prompt grouping requires at least one full prompt group.")
    if usable != total_rows:
        raise ValueError(
            "Listwise prompt grouping requires the flat batch size to be divisible by num_generations."
        )
    return total_rows, total_rows // group_size


def _shuffle_listwise_tensor_dict(
    tensor_dict: Dict[str, Optional[torch.Tensor]],
    group_size: int,
) -> Dict[str, Optional[torch.Tensor]]:
    """Shuffle prompt groups while preserving candidate order within each group."""
    total_rows, num_prompts = _resolve_prompt_group_sizes(tensor_dict, group_size)
    permutation_device: Optional[torch.device] = None
    for tensor in tensor_dict.values():
        if isinstance(tensor, torch.Tensor):
            permutation_device = tensor.device
            break
    permutation = torch.randperm(num_prompts, device=permutation_device)
    shuffled: Dict[str, Optional[torch.Tensor]] = {}
    for key, tensor in tensor_dict.items():
        if tensor is None:
            shuffled[key] = None
        elif int(tensor.size(0)) == total_rows:
            grouped = _reshape_prompt_major_tensor(tensor, group_size)
            if grouped is None:
                raise ValueError(f"Could not reshape listwise tensor {key!r} for shuffling.")
            shuffled[key] = _flatten_prompt_major_tensor(grouped[permutation])
        elif int(tensor.size(0)) == num_prompts:
            shuffled[key] = tensor[permutation]
        else:
            shuffled[key] = tensor
    return shuffled


def _normalize_listwise_q_targets(
    q_grouped: torch.Tensor,
    *,
    num_prompts: int,
    group_size: int,
    context: str,
) -> torch.Tensor:
    """Validate listwise q targets and project them onto the simplex."""
    if q_grouped.dim() != 2:
        raise ValueError(f"{context} requires rank-2 listwise q targets.")
    expected_shape = (num_prompts, group_size)
    actual_shape = (int(q_grouped.size(0)), int(q_grouped.size(1)))
    if actual_shape != expected_shape:
        raise ValueError(
            f"{context} requires listwise q targets with shape {expected_shape}, "
            f"got {actual_shape}."
        )
    if not torch.isfinite(q_grouped).all():
        raise ValueError(f"{context} requires finite listwise q targets.")
    if (q_grouped < 0).any():
        raise ValueError(f"{context} requires non-negative listwise q targets.")
    row_sums = q_grouped.sum(dim=1, keepdim=True)
    if (row_sums <= 0).any():
        raise ValueError(f"{context} requires listwise q targets with positive mass.")
    return q_grouped / row_sums


def _split_listwise_tensor_dict(
    tensor_dict: Dict[str, Optional[torch.Tensor]],
    num_chunks: int,
    group_size: int,
) -> List[Dict[str, Optional[torch.Tensor]]]:
    """Split buffered listwise tensors by whole prompt groups."""
    if num_chunks <= 0:
        raise ValueError("num_chunks must be positive")
    total_rows, num_prompts = _resolve_prompt_group_sizes(tensor_dict, group_size)
    if num_prompts % num_chunks != 0:
        # When the local rollout only contains too few whole prompt groups to
        # split across microsteps, reuse the full prompt-group batch for each
        # microstep and attenuate each reuse so one full reuse cycle matches
        # the intended total loss contribution of a normally split rollout.
        scale = torch.tensor(1.0 / float(num_chunks), dtype=torch.float32)
        for tensor in tensor_dict.values():
            if isinstance(tensor, torch.Tensor):
                scale = scale.to(device=tensor.device)
                break
        chunks: List[Dict[str, Optional[torch.Tensor]]] = []
        for _ in range(num_chunks):
            chunk = dict(tensor_dict)
            chunk["maxent_listwise_loss_scale"] = scale
            chunks.append(chunk)
        return chunks
    prompts_per_chunk = num_prompts // num_chunks
    rows_per_chunk = prompts_per_chunk * group_size
    chunks: List[Dict[str, Optional[torch.Tensor]]] = []
    for chunk_idx in range(num_chunks):
        row_start = chunk_idx * rows_per_chunk
        row_end = (chunk_idx + 1) * rows_per_chunk
        prompt_start = chunk_idx * prompts_per_chunk
        prompt_end = (chunk_idx + 1) * prompts_per_chunk
        chunk: Dict[str, Optional[torch.Tensor]] = {}
        for key, tensor in tensor_dict.items():
            if tensor is None:
                chunk[key] = None
            elif int(tensor.size(0)) == total_rows:
                chunk[key] = tensor[row_start:row_end]
            elif int(tensor.size(0)) == num_prompts:
                chunk[key] = tensor[prompt_start:prompt_end]
            else:
                chunk[key] = tensor
        chunks.append(chunk)
    return chunks


def _strip_ema_param_prefixes(name: str) -> Tuple[str, int]:
    """Remove known wrapper prefixes used in policy/reference param names."""
    clean = str(name)
    stripped = 0
    while clean:
        matched = False
        for prefix in _EMA_PARAM_NAME_PREFIXES:
            if clean.startswith(prefix):
                clean = clean[len(prefix) :]
                stripped += 1
                matched = True
                break
        if not matched:
            break
    return clean if clean else str(name), stripped


def _build_ema_alias_index(
    params: Dict[str, torch.Tensor],
) -> Dict[str, List[Tuple[str, torch.Tensor, int]]]:
    """Index tensors by canonicalized names for alias-aware EMA matching."""
    by_canonical: Dict[str, List[Tuple[str, torch.Tensor, int]]] = {}
    for name, param in params.items():
        canonical, stripped = _strip_ema_param_prefixes(name)
        by_canonical.setdefault(canonical, []).append((name, param, stripped))
    for candidates in by_canonical.values():
        candidates.sort(key=lambda item: (item[2], len(item[0]), item[0]))
    return by_canonical


def _selected_logps_and_entropy(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    *,
    entropy_mode: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return selected token log-probs and a differentiable entropy term."""
    log_probs = F.log_softmax(logits, dim=-1)
    selected_logps = torch.gather(
        log_probs, dim=-1, index=token_ids.unsqueeze(-1)
    ).squeeze(-1)
    if entropy_mode == "sample":
        entropy = -selected_logps
    else:
        probs = log_probs.exp()
        entropy = -(probs * log_probs).sum(dim=-1)
    return selected_logps, entropy


def _resolve_ema_source_param(
    ref_name: str,
    ref_param: torch.Tensor,
    policy_params: Dict[str, torch.Tensor],
    policy_alias_index: Dict[str, List[Tuple[str, torch.Tensor, int]]],
) -> Tuple[Optional[torch.Tensor], bool]:
    """Return matching policy tensor for ``ref_name`` and whether aliasing was used."""
    direct = policy_params.get(ref_name)
    if isinstance(direct, torch.Tensor) and direct.shape == ref_param.shape:
        return direct, False
    canonical, _ = _strip_ema_param_prefixes(ref_name)
    for candidate_name, candidate, _ in policy_alias_index.get(canonical, ()):
        if isinstance(candidate, torch.Tensor) and candidate.shape == ref_param.shape:
            return candidate, candidate_name != ref_name
    return None, False


def _strip_mode_prefix(key: str, mode: str) -> str:
    """Remove a train/eval prefix from metric keys when applicable."""
    if mode == "train" and key.startswith("train/"):
        return key[len("train/") :]
    if mode == "eval" and key.startswith("eval/"):
        return key[len("eval/") :]
    return key


_CANONICAL_METRIC_KEYS: Dict[str, str] = {
    "completions/mean_length": "completions/mean_length_sampled",
    "completions/min_length": "completions/min_length_sampled",
    "completions/max_length": "completions/max_length_sampled",
    "completions/clipped_ratio": "completions/clipped_frac",
    "completions/mean_terminated_length": "completions/mean_length_terminated",
    "completions/min_terminated_length": "completions/min_length_terminated",
    "completions/max_terminated_length": "completions/max_length_terminated",
}
_LEGACY_METRIC_ALIASES: Dict[str, Tuple[str, ...]] = {
    "completions/mean_length_sampled": ("completions/mean_length",),
    "completions/min_length_sampled": ("completions/min_length",),
    "completions/max_length_sampled": ("completions/max_length",),
    "completions/clipped_frac": ("completions/clipped_ratio",),
    "completions/mean_length_terminated": ("completions/mean_terminated_length",),
    "completions/min_length_terminated": ("completions/min_terminated_length",),
    "completions/max_length_terminated": ("completions/max_terminated_length",),
}


def _canonical_metric_key(key: str) -> str:
    """Normalize metric aliases to one canonical key namespace."""
    if key.startswith("diversity/"):
        return f"completions/{key}"
    return _CANONICAL_METRIC_KEYS.get(key, key)


def _legacy_metric_aliases(key: str) -> Tuple[str, ...]:
    """Return compatibility aliases for a canonical metric key."""
    aliases: List[str] = list(_LEGACY_METRIC_ALIASES.get(key, ()))
    if key.startswith("completions/diversity/"):
        aliases.append(key[len("completions/") :])
    if not aliases:
        return ()
    return tuple(dict.fromkeys(aliases))


def _supports_adapter_disabled_reference(model: Any) -> bool:
    """Return whether the model exposes an adapter-disable reference path."""

    return callable(getattr(model, "disable_adapter", None)) or (
        callable(getattr(model, "disable_adapters", None))
        and callable(getattr(model, "enable_adapters", None))
    )


def build_custom_grpo_trainer(parent_cls: Type[Any]) -> Type[Any]:
    """Return a GRPOTrainer subclass with MaxEnt hooks enabled.

    :param parent_cls: Base TRL GRPOTrainer class.
    :returns: Wrapped GRPOTrainer subclass.
    """

    if getattr(parent_cls, "_MAXENT_CUSTOM_TRAINER", False):
        return parent_cls

    class CustomGRPOTrainer(parent_cls):
        """Thin GRPOTrainer subclass used as a future extension point."""

        _MAXENT_CUSTOM_TRAINER = True

        @staticmethod
        def _resolve_parent_training_args(
            init_args: Tuple[Any, ...],
            init_kwargs: Dict[str, Any],
        ) -> Any:
            """Best-effort retrieval of TRL trainer args from constructor inputs."""
            if "args" in init_kwargs:
                return init_kwargs.get("args")
            if len(init_args) >= 3:
                return init_args[2]
            return None

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            parent_args = self._resolve_parent_training_args(args, kwargs)
            parent_routing = resolve_objective_routing(
                objective=getattr(parent_args, "objective", None),
                train_grpo_objective=getattr(
                    parent_args, "train_grpo_objective", True
                ),
                maxent_objective_variant=getattr(
                    parent_args, "maxent_objective_variant", None
                ),
                maxent_alpha=getattr(parent_args, "maxent_alpha", None),
                policy_entropy_bonus_coef=getattr(
                    parent_args, "policy_entropy_bonus_coef", 0.0
                ),
            )
            maxent_requested = parent_routing.maxent_requested
            parent_alpha_default = 1.0 if maxent_requested else 0.0
            parent_maxent_alpha = _coerce_non_negative_float(
                (
                    getattr(parent_args, "maxent_alpha", parent_alpha_default)
                    if parent_args is not None
                    else parent_alpha_default
                ),
                default=parent_alpha_default,
            )
            super().__init__(*args, **kwargs)

            self.objective_routing = resolve_objective_routing(
                objective=getattr(getattr(self, "args", None), "objective", None),
                train_grpo_objective=getattr(
                    getattr(self, "args", None), "train_grpo_objective", True
                ),
                maxent_objective_variant=getattr(
                    getattr(self, "args", None), "maxent_objective_variant", None
                ),
                maxent_alpha=getattr(
                    getattr(self, "args", None), "maxent_alpha", parent_maxent_alpha
                ),
                policy_entropy_bonus_coef=getattr(
                    getattr(self, "args", None), "policy_entropy_bonus_coef", 0.0
                ),
            )
            self.method_spec = resolve_method_spec_from_args(
                getattr(self, "args", None)
            )
            self.maxent_enabled = self.objective_routing.maxent_requested
            self.maxent_objective_variant = (
                self.objective_routing.maxent_objective_variant
            )
            self.maxent_alpha = self.objective_routing.maxent_alpha
            self._maybe_initialize_reference_model_for_maxent()
            controller_meta_requested = bool(
                getattr(getattr(self, "args", None), "controller_meta_enabled", False)
            )
            self._controller_meta_requested = controller_meta_requested
            if self.objective_routing.uses_listwise_loss:
                configured_tau = _coerce_non_negative_float(
                    getattr(getattr(self, "args", None), "maxent_tau", 0.0),
                    default=0.0,
                )
                if configured_tau <= 0.0:
                    raise ValueError("Listwise MaxEnt requires maxent_tau > 0.")
            self._maxent_weighting = (
                build_weighting_settings(getattr(self, "args", None))
                if (
                    (self.maxent_enabled or controller_meta_requested)
                    and getattr(self, "args", None) is not None
                )
                else None
            )
            self._maxent_controller_objective = (
                build_controller_objective(
                    getattr(self, "args", None), self._maxent_weighting
                )
                if self._maxent_weighting is not None
                else None
            )
            self._sync_weighting_scalars()
            route_mode = self.objective_routing.route_mode
            LOG.info(
                "Objective routing selected | mode=%s | objective=%s | "
                "maxent_variant=%s | maxent_alpha=%s",
                route_mode,
                getattr(getattr(self, "args", None), "objective", None),
                self.maxent_objective_variant,
                self.maxent_alpha,
            )
            if self.method_spec is not None:
                LOG.info(
                    "Resolved training method | name=%s | family=%s | backend=%s | "
                    "objective=%s | seed_grpo=%s | slug=%s",
                    self.method_spec.canonical_name,
                    self.method_spec.family,
                    self.method_spec.loss_backend,
                    self.method_spec.objective,
                    self.method_spec.seed_grpo_enabled,
                    self.method_spec.slug,
                )

        def evaluation_loop(  # type: ignore[override]
            self,
            dataloader: Any,
            description: str,
            prediction_loss_only: Optional[bool] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",
        ) -> Any:
            if _use_lightweight_greedy_eval(self, "eval"):
                original_include = getattr(self.args, "include_for_metrics", None)
                filtered_include: Any = original_include
                if original_include is not None:
                    filtered_include = tuple(
                        item for item in original_include if item != "inputs"
                    )
                try:
                    if original_include is not None:
                        self.args.include_for_metrics = filtered_include
                    return super().evaluation_loop(
                        dataloader,
                        description,
                        prediction_loss_only=True,
                        ignore_keys=ignore_keys,
                        metric_key_prefix=metric_key_prefix,
                    )
                finally:
                    if original_include is not None:
                        self.args.include_for_metrics = original_include
            return super().evaluation_loop(
                dataloader,
                description,
                prediction_loss_only=prediction_loss_only,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )

        def prediction_step(  # type: ignore[override]
            self,
            model: Any,
            inputs: Dict[str, Any],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
        ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
            if _use_lightweight_greedy_eval(self, "eval") and not bool(
                getattr(model, "training", False)
            ):
                # Greedy-only eval should do the minimum work required to log
                # pass@1 metrics: prepare one greedy completion per prompt and
                # skip the extra eval loss computation that the Trainer would
                # otherwise perform for every batch.
                self._prepare_inputs(inputs)
                return None, None, None
            return super().prediction_step(
                model,
                inputs,
                prediction_loss_only,
                ignore_keys=ignore_keys,
            )
            if controller_meta_requested and not self.objective_routing.uses_listwise_loss:
                LOG.info(
                    "Controller meta enabled for objective=%s; beta updates stay active, "
                    "but tau only affects listwise MaxEnt.",
                    route_mode,
                )
            if self.objective_routing.uses_listwise_loss:
                if self.maxent_alpha > 0.0:
                    LOG.info(
                        "Listwise MaxEnt selected; maxent_alpha=%.4f is inactive in this objective.",
                        self.maxent_alpha,
                    )
            elif self.maxent_enabled and self.maxent_objective_variant == "entropy":
                if float(getattr(getattr(self, "args", None), "maxent_tau", 0.0) or 0.0) > 0.0:
                    LOG.info(
                        "Entropy-regularized MaxEnt selected; listwise tau/q weighting knobs stay inactive."
                    )
            self._step = 0
            self._buffered_inputs: Optional[List[Dict[str, Optional[torch.Tensor]]]] = None
            self._last_train_kl_for_alpha: Optional[float] = None
            self._last_grpo_debug_step: Optional[int] = None
            self._last_reference_ema_step: Optional[int] = None

        def _entropy_alpha_kl_control_requested(self) -> bool:
            """Return whether entropy-MaxEnt KL-based alpha control is active."""

            args = getattr(self, "args", None)
            return bool(
                getattr(args, "maxent_alpha_raise_on_low_kl", False)
                or getattr(args, "maxent_alpha_lower_on_high_kl", False)
                or getattr(args, "maxent_alpha_disable_outside_trust_zone", False)
            )

        def _dr_grpo_denominator_mode(self) -> str:
            """Return the normalized Dr.GRPO denominator mode."""

            args = getattr(self, "args", None)
            mode = str(
                getattr(args, "dr_grpo_denominator_mode", "fixed_max") or "fixed_max"
            ).strip().lower()
            return "active_tokens" if mode == "active_tokens" else "fixed_max"

        def _dr_grpo_loss_denominator(
            self,
            completion_mask: torch.Tensor,
            *,
            loss_tensor: torch.Tensor,
            mode: str,
        ) -> torch.Tensor:
            """Return the denominator used by the Dr.GRPO loss."""

            denominator_mode = self._dr_grpo_denominator_mode()
            if denominator_mode == "active_tokens":
                denominator = completion_mask.sum().clamp(min=1).to(loss_tensor.dtype)
            else:
                max_completion_length = int(
                    getattr(self, "max_completion_length", 0)
                    or getattr(getattr(self, "args", None), "max_completion_length", 0)
                    or completion_mask.size(1)
                    or 1
                )
                denominator = loss_tensor.new_tensor(
                    float(max(loss_tensor.size(0) * max(max_completion_length, 1), 1))
                )
            self._append_metric_value(
                mode,
                "loss/dr_grpo_denominator",
                float(denominator.detach().item()),
                include_legacy_aliases=False,
            )
            self._append_metric_value(
                mode,
                "loss/dr_grpo_denominator_active_tokens",
                1.0 if denominator_mode == "active_tokens" else 0.0,
                include_legacy_aliases=False,
            )
            return denominator

        def _should_force_reference_model_for_maxent(self) -> bool:
            """Return whether MaxEnt should materialize a frozen reference model."""

            if not bool(getattr(self, "maxent_enabled", False)):
                return False
            args = getattr(self, "args", None)
            if bool(getattr(args, "maxent_share_reference_model", False)):
                return False
            if getattr(self, "ref_model", None) is not None:
                return False
            unwrapped_model = getattr(self, "model", None)
            unwrap_fn = getattr(getattr(self, "accelerator", None), "unwrap_model", None)
            if callable(unwrap_fn):
                try:
                    unwrapped_model = unwrap_fn(unwrapped_model)
                except Exception:
                    unwrapped_model = getattr(self, "model", None)
            if _supports_adapter_disabled_reference(unwrapped_model):
                return False

            ref_source = str(
                getattr(args, "maxent_reference_logprobs_source", "auto") or "auto"
            ).strip().lower()
            force_model_reference = bool(
                getattr(args, "maxent_trl_reference_scoring", False)
            ) or ref_source in {
                "model",
                "reference",
                "reference_model",
                "ref_model",
            }
            needs_entropy_kl_measure = bool(
                self.objective_routing.uses_entropy_regularized_loss
                and (
                    float(getattr(self, "beta", 0.0) or 0.0) != 0.0
                    or self._entropy_alpha_kl_control_requested()
                )
            )
            return bool(force_model_reference or needs_entropy_kl_measure)

        def _maybe_initialize_reference_model_for_maxent(self) -> None:
            """Materialize a frozen reference model when MaxEnt needs one and TRL skipped it."""

            if not self._should_force_reference_model_for_maxent():
                return
            global AutoModelForCausalLM
            global _trl_create_reference_model
            global _trl_prepare_deepspeed
            global _trl_prepare_fsdp

            if _trl_create_reference_model is None:
                try:
                    from trl.models import create_reference_model as _create_reference_model
                    from trl.models import prepare_deepspeed as _prepare_deepspeed
                    from trl.models import prepare_fsdp as _prepare_fsdp

                    _trl_create_reference_model = _create_reference_model
                    _trl_prepare_deepspeed = _prepare_deepspeed
                    _trl_prepare_fsdp = _prepare_fsdp
                except Exception:
                    _trl_create_reference_model = None
                    _trl_prepare_deepspeed = None
                    _trl_prepare_fsdp = None
            if AutoModelForCausalLM is None:
                try:
                    from transformers import AutoModelForCausalLM as _AutoModelForCausalLM

                    AutoModelForCausalLM = _AutoModelForCausalLM  # type: ignore[assignment]
                except Exception:
                    AutoModelForCausalLM = None  # type: ignore[assignment]

            policy_model = getattr(self, "model", None)
            unwrap_fn = getattr(getattr(self, "accelerator", None), "unwrap_model", None)
            unwrapped_policy = policy_model
            if callable(unwrap_fn):
                try:
                    unwrapped_policy = unwrap_fn(policy_model)
                except Exception:
                    unwrapped_policy = policy_model

            ref_model: Any = None
            if (
                not bool(getattr(self, "is_deepspeed_enabled", False))
                and not bool(getattr(self, "is_fsdp_enabled", False))
                and callable(_trl_create_reference_model)
            ):
                try:
                    ref_model = _trl_create_reference_model(unwrapped_policy)
                except Exception as exc:
                    LOG.warning(
                        "Failed to clone a frozen reference model for MaxEnt KL measurement; "
                        "retrying from pretrained weights: %s",
                        exc,
                    )

            if ref_model is None and AutoModelForCausalLM is not None:
                args = getattr(self, "args", None)
                model_init_kwargs_raw = getattr(args, "model_init_kwargs", None)
                model_init_kwargs = (
                    dict(model_init_kwargs_raw)
                    if isinstance(model_init_kwargs_raw, Mapping)
                    else {}
                )
                ref_revision = getattr(args, "reference_model_revision", None)
                if ref_revision:
                    model_init_kwargs["revision"] = ref_revision
                model_id = (
                    getattr(args, "reference_model_name_or_path", None)
                    or getattr(getattr(unwrapped_policy, "config", None), "_name_or_path", None)
                    or getattr(unwrapped_policy, "name_or_path", None)
                )
                if model_id:
                    try:
                        ref_model = AutoModelForCausalLM.from_pretrained(
                            model_id,
                            **model_init_kwargs,
                        )
                    except Exception as exc:
                        LOG.warning(
                            "Failed to load a frozen reference model for MaxEnt KL measurement "
                            "from %s: %s",
                            model_id,
                            exc,
                        )

            if ref_model is None:
                if not bool(getattr(self, "_maxent_missing_ref_model_warned", False)):
                    LOG.warning(
                        "MaxEnt requested model-based KL measurement, but no frozen reference "
                        "model could be initialized. KL metrics may collapse to rollout behavior."
                    )
                    setattr(self, "_maxent_missing_ref_model_warned", True)
                return

            for param in getattr(ref_model, "parameters", lambda: [])():
                try:
                    param.requires_grad = False
                except (AttributeError, RuntimeError):
                    continue
            eval_fn = getattr(ref_model, "eval", None)
            if callable(eval_fn):
                eval_fn()
            if bool(getattr(self, "is_deepspeed_enabled", False)) and callable(
                _trl_prepare_deepspeed
            ):
                ref_model = _trl_prepare_deepspeed(ref_model, self.accelerator)
            elif bool(getattr(self, "is_fsdp_enabled", False)) and callable(
                _trl_prepare_fsdp
            ):
                ref_model = _trl_prepare_fsdp(ref_model, self.accelerator)
            else:
                prepare_model = getattr(self.accelerator, "prepare_model", None)
                if callable(prepare_model):
                    try:
                        ref_model = prepare_model(
                            ref_model,
                            evaluation_mode=True,
                        )
                    except TypeError:
                        ref_model = prepare_model(ref_model)
                else:
                    prepare = getattr(self.accelerator, "prepare", None)
                    if callable(prepare):
                        ref_model = prepare(ref_model)
            self.ref_model = ref_model  # pylint: disable=attribute-defined-outside-init
            LOG.info(
                "Materialized a frozen reference model for MaxEnt KL measurement despite beta=0."
            )

        def _should_use_model_reference_logprobs(
            self,
            *,
            default_to_model_reference: bool,
        ) -> bool:
            """Return whether the current loss should score against a model-based reference."""

            args = getattr(self, "args", None)
            ref_source = str(
                getattr(args, "maxent_reference_logprobs_source", "auto") or "auto"
            ).strip().lower()
            if ref_source == "none":
                ref_source = "policy"
            if bool(getattr(args, "maxent_trl_reference_scoring", False)):
                return True
            if ref_source in {"model", "reference", "reference_model", "ref_model"}:
                return True
            if ref_source == "policy":
                return False
            if ref_source == "auto":
                if getattr(self, "ref_model", None) is not None:
                    return True
                unwrap_fn = getattr(getattr(self, "accelerator", None), "unwrap_model", None)
                unwrapped_model = getattr(self, "model", None)
                if callable(unwrap_fn):
                    try:
                        unwrapped_model = unwrap_fn(unwrapped_model)
                    except Exception:
                        unwrapped_model = getattr(self, "model", None)
                if _supports_adapter_disabled_reference(unwrapped_model):
                    return True
                return default_to_model_reference
            return default_to_model_reference

        def _get_reference_per_token_logps(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            logits_to_keep: int,
            *,
            batch_size: int,
        ) -> Optional[torch.Tensor]:
            """Return per-token log-probs from the frozen/model-based reference path."""

            if getattr(self, "ref_model", None) is not None:
                return self._get_per_token_logps(
                    self.ref_model,
                    input_ids,
                    attention_mask,
                    logits_to_keep,
                    batch_size=batch_size,
                )
            unwrap_fn = getattr(getattr(self, "accelerator", None), "unwrap_model", None)
            unwrapped_model = getattr(self, "model", None)
            if callable(unwrap_fn):
                try:
                    unwrapped_model = unwrap_fn(unwrapped_model)
                except Exception:
                    unwrapped_model = getattr(self, "model", None)
            if not _supports_adapter_disabled_reference(unwrapped_model):
                return None
            with _adapter_disabled_context(unwrapped_model):
                return self._get_per_token_logps(
                    self.model,
                    input_ids,
                    attention_mask,
                    logits_to_keep,
                    batch_size=batch_size,
                )

        def _sync_weighting_scalars(self) -> None:
            """Expose controller scalars on the trainer for logging helpers."""
            weighting = getattr(self, "_maxent_weighting", None)
            if weighting is None:
                return
            tau_val = float(getattr(weighting, "tau", 0.0) or 0.0)
            beta_val = float(getattr(weighting, "beta", 0.0) or 0.0)
            denom_val = float(getattr(weighting, "denom", 1.0) or 1.0)
            self.tau = tau_val  # pylint: disable=attribute-defined-outside-init
            self.maxent_tau = tau_val  # pylint: disable=attribute-defined-outside-init
            self.beta = beta_val  # pylint: disable=attribute-defined-outside-init
            self.weight_norm_denom = (
                denom_val  # pylint: disable=attribute-defined-outside-init
            )

        def _maybe_apply_controller_meta(
            self,
            *,
            mode: str,
            kl_value: Optional[float],
            weight_entropy: Optional[float] = None,
            total_loss: Optional[float] = None,
        ) -> bool:
            """Apply one meta-controller update when the active route enables it."""
            if mode != "train":
                return False
            weighting = getattr(self, "_maxent_weighting", None)
            meta_objective = getattr(self, "_maxent_controller_objective", None)
            if weighting is None or meta_objective is None:
                return False

            update_interval = max(
                1,
                int(
                    getattr(
                        getattr(weighting, "controller_meta", None),
                        "update_interval",
                        1,
                    )
                    or 1
                ),
            )
            global_step = int(getattr(self.state, "global_step", 0) or 0)
            if global_step % update_interval != 0:
                return False

            weight_stats = SimpleNamespace()
            if isinstance(weight_entropy, (int, float)) and math.isfinite(weight_entropy):
                weight_stats = SimpleNamespace(weight_entropy=float(weight_entropy))
            loss_outputs = SimpleNamespace(
                kl_loss_scalar=(
                    float(kl_value)
                    if isinstance(kl_value, (int, float)) and math.isfinite(kl_value)
                    else None
                ),
                total_loss_scalar=(
                    float(total_loss)
                    if isinstance(total_loss, (int, float)) and math.isfinite(total_loss)
                    else None
                ),
            )
            grads = meta_objective.compute(
                ControllerMetaContext(
                    weighting=weighting,
                    weight_stats=weight_stats,
                    loss_outputs=loss_outputs,
                    global_step=global_step,
                    kl_value=kl_value,
                )
            )
            if grads is None:
                return False
            updated = apply_meta_controller_update(
                weighting,
                tau_grad=grads.tau_grad,
                beta_grad=grads.beta_grad,
            )
            if not updated:
                return False
            if isinstance(getattr(grads, "tau_grad", None), (int, float)):
                self._append_metric_value(
                    mode,
                    "meta/tau_grad",
                    float(getattr(grads, "tau_grad", 0.0) or 0.0),
                )
            if isinstance(getattr(grads, "beta_grad", None), (int, float)):
                self._append_metric_value(
                    mode,
                    "meta/beta_grad",
                    float(getattr(grads, "beta_grad", 0.0) or 0.0),
                )
            return True

        def get_train_dataloader(self):  # type: ignore[override]
            # Preserve native TRL batching/sampling behavior for GRPO/MaxEnt while
            # adapting worker_init_fn to the active transformers seed_worker signature.
            if self.train_dataset is None:
                raise ValueError("Trainer: training requires a train_dataset.")

            train_dataset = self.train_dataset
            data_collator = self.data_collator

            try:
                from transformers.utils import is_datasets_available
            except (
                Exception
            ):  # pragma: no cover - transformers is required for training

                def is_datasets_available() -> bool:
                    return False

            if is_datasets_available():
                try:
                    import datasets
                except Exception:
                    datasets = None  # type: ignore
                if datasets is not None and isinstance(train_dataset, datasets.Dataset):
                    train_dataset = self._remove_unused_columns(
                        train_dataset, description="training"
                    )
                else:
                    data_collator = self._get_collator_with_removed_columns(
                        data_collator, description="training"
                    )
            else:
                data_collator = self._get_collator_with_removed_columns(
                    data_collator, description="training"
                )

            dataloader_params = {
                "batch_size": self._train_batch_size * self.args.steps_per_generation,
                "collate_fn": data_collator,
                "num_workers": self.args.dataloader_num_workers,
                "pin_memory": self.args.dataloader_pin_memory,
                "persistent_workers": self.args.dataloader_persistent_workers,
            }

            if not isinstance(train_dataset, torch.utils.data.IterableDataset):
                dataloader_params["sampler"] = self._get_train_sampler()
                dataloader_params["drop_last"] = self.args.dataloader_drop_last
                dataloader_params["prefetch_factor"] = (
                    self.args.dataloader_prefetch_factor
                )
                worker_init_fn = _build_seed_worker(
                    self.args.dataloader_num_workers, self.args.process_index
                )
                if worker_init_fn is not None:
                    dataloader_params["worker_init_fn"] = worker_init_fn

            return self.accelerator.prepare(
                torch.utils.data.DataLoader(train_dataset, **dataloader_params)
            )

        def get_eval_dataloader(self, eval_dataset: Optional[Any] = None):  # type: ignore[override]
            """Use a prompt-major loader for greedy-only eval, sharded across ranks."""

            eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
            lightweight_eval = _use_lightweight_greedy_eval(self, "eval")
            if not lightweight_eval:
                setattr(self, "_local_only_eval_prompt_major_loader_active", False)
                setattr(self, "_sharded_eval_prompt_major_loader_active", False)
                return super().get_eval_dataloader(eval_dataset)

            if eval_dataset is None:
                raise ValueError("Trainer: evaluation requires an eval_dataset.")

            sharded_eval = _use_sharded_prompt_major_greedy_eval(self, "eval")
            setattr(self, "_local_only_eval_prompt_major_loader_active", True)
            setattr(self, "_sharded_eval_prompt_major_loader_active", bool(sharded_eval))
            prompt_major_dataset = eval_dataset
            data_collator = self.data_collator

            try:
                from transformers.utils import is_datasets_available
            except Exception:  # pragma: no cover - transformers is required for training

                def is_datasets_available() -> bool:
                    return False

            if is_datasets_available():
                try:
                    import datasets
                except Exception:
                    datasets = None  # type: ignore
                if datasets is not None and isinstance(
                    prompt_major_dataset, datasets.Dataset
                ):
                    prompt_major_dataset = self._remove_unused_columns(
                        prompt_major_dataset,
                        description="evaluation",
                    )
                else:
                    trim_collator = getattr(
                        self,
                        "_get_collator_with_removed_columns",
                        None,
                    )
                    if callable(trim_collator):
                        data_collator = trim_collator(
                            data_collator,
                            description="evaluation",
                        )
            else:
                trim_collator = getattr(
                    self,
                    "_get_collator_with_removed_columns",
                    None,
                )
                if callable(trim_collator):
                    data_collator = trim_collator(
                        data_collator,
                        description="evaluation",
                    )

            dataloader_params = {
                "batch_size": int(
                    getattr(self.args, "per_device_eval_batch_size", 0)
                    or getattr(self.args, "eval_batch_size", 0)
                    or 1
                ),
                "collate_fn": data_collator,
                "num_workers": int(getattr(self.args, "dataloader_num_workers", 0) or 0),
                "pin_memory": bool(
                    getattr(self.args, "dataloader_pin_memory", False)
                ),
                "persistent_workers": bool(
                    getattr(self.args, "dataloader_persistent_workers", False)
                ),
            }

            if not isinstance(prompt_major_dataset, torch.utils.data.IterableDataset):
                if sharded_eval:
                    accelerator = getattr(self, "accelerator", None)
                    try:
                        num_processes = int(
                            getattr(accelerator, "num_processes", 1) or 1
                        )
                    except (TypeError, ValueError):
                        num_processes = 1
                    try:
                        process_index = int(
                            getattr(accelerator, "process_index", 0) or 0
                        )
                    except (TypeError, ValueError):
                        process_index = 0
                    dataloader_params["sampler"] = (
                        torch.utils.data.distributed.DistributedSampler(
                            prompt_major_dataset,
                            num_replicas=max(num_processes, 1),
                            rank=max(process_index, 0),
                            shuffle=False,
                            drop_last=False,
                        )
                    )
                else:
                    dataloader_params["sampler"] = torch.utils.data.SequentialSampler(
                        prompt_major_dataset
                    )
                dataloader_params["drop_last"] = False
                if dataloader_params["num_workers"] > 0:
                    prefetch = getattr(self.args, "dataloader_prefetch_factor", None)
                    if prefetch is not None:
                        dataloader_params["prefetch_factor"] = int(prefetch)
                    worker_init_fn = _build_seed_worker(
                        dataloader_params["num_workers"],
                        int(getattr(self.args, "process_index", 0) or 0),
                    )
                    if worker_init_fn is not None:
                        dataloader_params["worker_init_fn"] = worker_init_fn

            # Keep this dataloader fully local. ``accelerator.prepare`` would shard or
            # wrap it again, which defeats the goal of rank-0-only prompt-major eval.
            return torch.utils.data.DataLoader(prompt_major_dataset, **dataloader_params)

        def _prepare_inputs(self, generation_batch: Any) -> Any:  # type: ignore[override]
            if not self.objective_routing.uses_listwise_loss:
                return super()._prepare_inputs(generation_batch)

            mode = "train" if self.model.training else "eval"
            if mode == "train":
                generate_every = self.args.steps_per_generation * self.num_iterations
                if self._step % generate_every == 0 or self._buffered_inputs is None:
                    generated = self._generate_and_score_completions(generation_batch)
                    group_size = max(int(getattr(self, "num_generations", 1) or 1), 1)
                    steps_per_generation = int(
                        getattr(self.args, "steps_per_generation", 1) or 1
                    )
                    _, num_prompts = _resolve_prompt_group_sizes(generated, group_size)
                    q_targets = generated.get("maxent_listwise_q")
                    if not isinstance(q_targets, torch.Tensor):
                        raise ValueError(
                            "Listwise MaxEnt rollout generation must provide maxent_listwise_q targets."
                        )
                    generated["maxent_listwise_q"] = _normalize_listwise_q_targets(
                        q_targets,
                        num_prompts=num_prompts,
                        group_size=group_size,
                        context="Listwise MaxEnt rollout generation",
                    )
                    generated = _shuffle_listwise_tensor_dict(
                        generated,
                        group_size,
                    )
                    if (
                        num_prompts % steps_per_generation != 0
                        and not bool(
                            getattr(self, "_listwise_batch_reuse_warned", False)
                        )
                    ):
                        LOG.warning(
                            "Listwise MaxEnt local rollout prompt groups (%d) do not divide "
                            "steps_per_generation (%d); reusing the full local listwise batch "
                            "across microsteps with a per-microstep loss scale of 1/%d. "
                            "Increase the local prompt-group count to avoid this fallback.",
                            num_prompts,
                            steps_per_generation,
                            steps_per_generation,
                        )
                        setattr(self, "_listwise_batch_reuse_warned", True)
                    self._buffered_inputs = _split_listwise_tensor_dict(
                        generated,
                        steps_per_generation,
                        group_size,
                    )
                inputs = self._buffered_inputs[
                    self._step % int(getattr(self.args, "steps_per_generation", 1) or 1)
                ]
                self._step += 1
                return inputs
            return self._generate_and_score_completions(generation_batch)

        def _append_metric_value(
            self,
            mode: str,
            key: str,
            value: Any,
            *,
            include_legacy_aliases: bool = True,
        ) -> None:
            numeric = _numeric_or_none(value)
            if numeric is None:
                return
            normalized = _strip_mode_prefix(str(key), mode)
            canonical = _canonical_metric_key(normalized)
            store = self._metrics[mode]
            if canonical == "num_tokens":
                store[canonical] = [numeric]
            else:
                store.setdefault(canonical, []).append(numeric)
            if mode == "train" and canonical == "kl":
                setattr(self, "_last_train_kl_for_alpha", float(numeric))
            if not include_legacy_aliases:
                return
            for alias in _legacy_metric_aliases(canonical):
                if alias == canonical:
                    continue
                if canonical == "num_tokens":
                    store[alias] = [numeric]
                else:
                    store.setdefault(alias, []).append(numeric)

        def _set_latest_metric_value(
            self,
            mode: str,
            key: str,
            value: Any,
            *,
            include_legacy_aliases: bool = True,
        ) -> None:
            """Replace the most recent metric sample for a key, appending if absent."""

            numeric = _numeric_or_none(value)
            if numeric is None:
                return
            normalized = _strip_mode_prefix(str(key), mode)
            canonical = _canonical_metric_key(normalized)
            store = self._metrics[mode]
            if canonical == "num_tokens":
                store[canonical] = [numeric]
            else:
                bucket = store.setdefault(canonical, [])
                if bucket:
                    bucket[-1] = numeric
                else:
                    bucket.append(numeric)
            if mode == "train" and canonical == "kl":
                setattr(self, "_last_train_kl_for_alpha", float(numeric))
            if not include_legacy_aliases:
                return
            for alias in _legacy_metric_aliases(canonical):
                if alias == canonical:
                    continue
                if canonical == "num_tokens":
                    store[alias] = [numeric]
                    continue
                alias_bucket = store.setdefault(alias, [])
                if alias_bucket:
                    alias_bucket[-1] = numeric
                else:
                    alias_bucket.append(numeric)

        def _recompute_completion_metrics(
            self,
            outputs: Dict[str, Any],
            *,
            mode: str,
        ) -> None:
            """Overwrite TRL completion metrics with correctly gathered values."""

            completion_ids = outputs.get("completion_ids")
            completion_mask = outputs.get("completion_mask")
            if not isinstance(completion_ids, torch.Tensor) or not isinstance(
                completion_mask, torch.Tensor
            ):
                return
            try:
                completion_lengths = completion_mask.sum(dim=1).to(torch.float32)
            except Exception:
                return
            gathered_lengths = _metric_tensor_for_logging(
                self,
                completion_lengths,
                mode=mode,
            )
            if not isinstance(gathered_lengths, torch.Tensor) or gathered_lengths.numel() <= 0:
                return
            self._set_latest_metric_value(
                mode,
                "completions/mean_length",
                float(gathered_lengths.mean().item()),
            )
            self._set_latest_metric_value(
                mode,
                "completions/min_length",
                float(gathered_lengths.min().item()),
            )
            self._set_latest_metric_value(
                mode,
                "completions/max_length",
                float(gathered_lengths.max().item()),
            )

            terminated_mask: Optional[torch.Tensor] = None
            eos_token_id = _coerce_optional_int(
                getattr(getattr(self, "processing_class", None), "eos_token_id", None)
            )
            if eos_token_id is not None:
                try:
                    active_eos = (completion_ids == int(eos_token_id)) & completion_mask.to(
                        dtype=torch.bool
                    )
                    terminated_mask = active_eos.any(dim=1)
                except Exception:
                    terminated_mask = None
            max_completion_length = int(
                getattr(self, "max_completion_length", 0)
                or getattr(getattr(self, "args", None), "max_completion_length", 0)
                or 0
            )
            if max_completion_length > 0:
                try:
                    shorter_than_cap = completion_lengths.to(torch.long) < int(
                        max_completion_length
                    )
                    terminated_mask = (
                        shorter_than_cap
                        if terminated_mask is None
                        else (terminated_mask | shorter_than_cap)
                    )
                except Exception:
                    pass
            if not isinstance(terminated_mask, torch.Tensor):
                return
            gathered_terminated = _metric_tensor_for_logging(
                self,
                terminated_mask.to(torch.bool),
                mode=mode,
            )
            if (
                not isinstance(gathered_terminated, torch.Tensor)
                or gathered_terminated.numel() <= 0
            ):
                return
            total = int(min(gathered_lengths.numel(), gathered_terminated.numel()))
            if total <= 0:
                return
            gathered_lengths = gathered_lengths[:total]
            gathered_terminated = gathered_terminated[:total].to(torch.bool)
            term_completion_lengths = gathered_lengths[gathered_terminated]
            clipped_ratio = 1.0 - (
                float(term_completion_lengths.numel()) / float(max(total, 1))
            )
            self._set_latest_metric_value(
                mode,
                "completions/clipped_ratio",
                clipped_ratio,
            )
            if term_completion_lengths.numel() <= 0:
                zero_val = 0.0
                self._set_latest_metric_value(
                    mode,
                    "completions/mean_terminated_length",
                    zero_val,
                )
                self._set_latest_metric_value(
                    mode,
                    "completions/min_terminated_length",
                    zero_val,
                )
                self._set_latest_metric_value(
                    mode,
                    "completions/max_terminated_length",
                    zero_val,
                )
                return
            self._set_latest_metric_value(
                mode,
                "completions/mean_terminated_length",
                float(term_completion_lengths.mean().item()),
            )
            self._set_latest_metric_value(
                mode,
                "completions/min_terminated_length",
                float(term_completion_lengths.min().item()),
            )
            self._set_latest_metric_value(
                mode,
                "completions/max_terminated_length",
                float(term_completion_lengths.max().item()),
            )

        def _log_grpo_debug(
            self,
            inputs: List[Dict[str, Any]],
            outputs: Dict[str, Any],
            *,
            mode: str,
        ) -> None:
            if mode != "train":
                return
            step = int(getattr(self.state, "global_step", 0))
            last_logged_step = getattr(self, "_last_grpo_debug_step", None)
            if last_logged_step == step:
                return
            self._last_grpo_debug_step = step

            completion_mask = outputs.get("completion_mask")
            advantages = outputs.get("advantages")
            completion_ids = outputs.get("completion_ids")

            token_mask_sum = None
            completion_length_mean = None
            if isinstance(completion_mask, torch.Tensor):
                try:
                    completion_lengths = completion_mask.sum(1)
                    agg_lengths = self.accelerator.gather(completion_lengths)
                    completion_length_mean = float(agg_lengths.float().mean().item())
                    token_mask_sum = float(agg_lengths.sum().item())
                except Exception:
                    completion_length_mean = None
                    token_mask_sum = None

            advantages_std = None
            if isinstance(advantages, torch.Tensor):
                try:
                    agg_adv = self.accelerator.gather(advantages)
                    advantages_std = float(agg_adv.float().std().item())
                except Exception:
                    advantages_std = None

            reward_std = None
            try:
                reward_history = self._metrics.get(mode, {}).get("reward_std")
                if reward_history:
                    reward_std = float(reward_history[-1])
            except Exception:
                reward_std = None

            local_expected = len(inputs)
            local_actual = (
                int(completion_ids.shape[0])
                if isinstance(completion_ids, torch.Tensor)
                else local_expected
            )
            try:
                counts = torch.tensor(
                    [local_expected, local_actual],
                    device=self.accelerator.device,
                    dtype=torch.long,
                )
                agg_counts = self.accelerator.gather(counts)
                expected_total = int(agg_counts[0::2].sum().item())
                actual_total = int(agg_counts[1::2].sum().item())
            except Exception:
                expected_total = local_expected
                actual_total = local_actual
            dropped_total = max(expected_total - actual_total, 0)

            if self.accelerator.is_main_process:
                LOG.info(
                    "GRPO debug | step=%d | token_mask_sum=%s | completion_length_mean=%s | "
                    "advantages_std=%s | reward_std=%s | num_sequences=%d | dropped_groups=%d",
                    step,
                    token_mask_sum,
                    completion_length_mean,
                    advantages_std,
                    reward_std,
                    expected_total,
                    dropped_total,
                )
            self._maybe_update_grpo_beta(mode)

        def _maybe_update_grpo_beta(self, mode: str) -> None:
            if self.maxent_enabled:
                return
            if getattr(self, "_maxent_controller_objective", None) is not None:
                return
            args = getattr(self, "args", None)
            if args is None:
                return
            if not bool(getattr(args, "grpo_beta_controller_enabled", False)):
                return
            kl_target = float(getattr(args, "kl_target", 0.0) or 0.0)
            kl_horizon = int(getattr(args, "kl_horizon", 0) or 0)
            kl_ctl_step_size = float(getattr(args, "kl_ctl_step_size", 0.0) or 0.0)
            if kl_target <= 0.0 or kl_horizon <= 0 or kl_ctl_step_size <= 0.0:
                return
            kl_history = self._metrics.get(mode, {}).get("kl")
            if not kl_history:
                return
            try:
                measured_kl = float(kl_history[-1])
            except (TypeError, ValueError):
                return
            if not math.isfinite(measured_kl):
                return
            current_beta = float(getattr(self, "beta", 0.0) or 0.0)
            if current_beta <= 0.0:
                return
            ratio = measured_kl / max(kl_target, 1e-8)
            error = ratio - 1.0
            if abs(error) < 1e-8:
                return
            limit = kl_ctl_step_size
            clipped_error = max(min(error, limit), -limit)
            horizon = max(1, kl_horizon)
            scale = 1.0 + clipped_error / float(horizon)
            if scale <= 0.0:
                scale = 1e-6
            new_beta = max(0.0, current_beta * scale)
            self.beta = new_beta  # pylint: disable=attribute-defined-outside-init

        def _maybe_update_reference_model_ema(self) -> None:
            """Soft-update frozen reference weights from the current policy weights."""
            args = getattr(self, "args", None)
            if args is None:
                return
            if not bool(getattr(self.model, "training", False)):
                return
            if not bool(getattr(args, "maxent_reference_ema_enabled", False)):
                return
            if bool(getattr(args, "maxent_share_reference_model", False)):
                if not bool(getattr(self, "_maxent_ref_ema_share_warned", False)):
                    LOG.warning(
                        "Reference EMA requested but maxent_share_reference_model=true; skipping EMA updates."
                    )
                    setattr(self, "_maxent_ref_ema_share_warned", True)
                return

            ref_model = getattr(self, "ref_model", None)
            if ref_model is None:
                if not bool(getattr(self, "_maxent_ref_ema_missing_warned", False)):
                    LOG.warning(
                        "Reference EMA requested but no frozen reference model is available; skipping EMA updates."
                    )
                    setattr(self, "_maxent_ref_ema_missing_warned", True)
                return

            step = int(getattr(self.state, "global_step", 0) or 0)
            if step <= 0:
                return
            if self._last_reference_ema_step == step:
                return

            warmup_raw = getattr(args, "maxent_reference_ema_warmup_steps", 0)
            interval_raw = getattr(args, "maxent_reference_ema_update_interval", 1)
            beta_raw = getattr(args, "maxent_reference_ema_beta", 0.995)
            try:
                warmup_steps = int(warmup_raw)
            except (TypeError, ValueError):
                warmup_steps = 0
            if warmup_steps < 0:
                warmup_steps = 0
            if step < warmup_steps:
                return
            try:
                update_interval = int(interval_raw)
            except (TypeError, ValueError):
                update_interval = 1
            if update_interval < 1:
                update_interval = 1
            if (step - warmup_steps) % update_interval != 0:
                return

            beta = _numeric_or_none(beta_raw)
            if beta is None or not math.isfinite(beta):
                beta = 0.995
            beta = min(max(float(beta), 0.0), 1.0)
            alpha = 1.0 - beta
            if alpha <= 0.0:
                return

            unwrap_fn = getattr(self.accelerator, "unwrap_model", None)
            policy_model = self.model
            if callable(unwrap_fn):
                try:
                    policy_model = unwrap_fn(policy_model)
                except Exception:
                    policy_model = self.model
                try:
                    ref_model = unwrap_fn(ref_model)
                except Exception:
                    ref_model = getattr(self, "ref_model", None)
            if ref_model is None:
                return
            if ref_model is policy_model:
                if not bool(getattr(self, "_maxent_ref_ema_alias_warned", False)):
                    LOG.warning(
                        "Reference EMA requested but reference model aliases the policy model; skipping EMA updates."
                    )
                    setattr(self, "_maxent_ref_ema_alias_warned", True)
                return

            policy_named = getattr(policy_model, "named_parameters", None)
            ref_named = getattr(ref_model, "named_parameters", None)
            if not callable(policy_named) or not callable(ref_named):
                return

            try:
                policy_named_fn = cast(
                    Callable[[], Iterable[tuple[str, Any]]], policy_named
                )
                ref_named_fn = cast(Callable[[], Iterable[tuple[str, Any]]], ref_named)
                policy_params = {
                    str(name): param
                    for name, param in policy_named_fn()
                    if isinstance(param, torch.Tensor)
                }
                ref_params = {
                    str(name): param
                    for name, param in ref_named_fn()  # pylint: disable=not-callable
                    if isinstance(param, torch.Tensor)
                }
                if not policy_params or not ref_params:
                    return
                policy_alias_index = _build_ema_alias_index(policy_params)
                total_ref = len(ref_params)
                updated = 0
                mismatched = 0
                alias_hits = 0
                mismatch_examples: List[str] = []
                with torch.no_grad():
                    for name, ref_param in ref_params.items():
                        src_param, alias_used = _resolve_ema_source_param(
                            name,
                            ref_param,
                            policy_params,
                            policy_alias_index,
                        )
                        if not isinstance(src_param, torch.Tensor):
                            mismatched += 1
                            if len(mismatch_examples) < 5:
                                mismatch_examples.append(name)
                            continue
                        src_tensor = src_param.detach().to(
                            device=ref_param.device, dtype=ref_param.dtype
                        )
                        ref_param.data.mul_(beta).add_(src_tensor, alpha=alpha)
                        updated += 1
                        if alias_used:
                            alias_hits += 1
            except Exception as exc:
                if not bool(getattr(self, "_maxent_ref_ema_error_warned", False)):
                    LOG.warning(
                        "Reference EMA update failed once and will be retried on later steps: %s",
                        exc,
                    )
                    setattr(self, "_maxent_ref_ema_error_warned", True)
                return

            if updated <= 0:
                if not bool(getattr(self, "_maxent_ref_ema_no_params_warned", False)):
                    LOG.warning(
                        "Reference EMA enabled but no compatible parameters were updated."
                    )
                    setattr(self, "_maxent_ref_ema_no_params_warned", True)
                return

            self._last_reference_ema_step = step
            self._append_metric_value("train", "maxent/ref_ema_applied", 1.0)
            self._append_metric_value("train", "maxent/ref_ema_beta", beta)
            self._append_metric_value(
                "train",
                "maxent/ref_ema_updated_frac",
                float(updated) / float(max(total_ref, 1)),
            )
            self._append_metric_value(
                "train",
                "maxent/ref_ema_alias_hit_frac",
                float(alias_hits) / float(max(total_ref, 1)),
            )
            if mismatched > 0 and not bool(
                getattr(self, "_maxent_ref_ema_mismatch_warned", False)
            ):
                LOG.warning(
                    "Reference EMA skipped %d/%d reference parameters due to missing/mismatched policy counterparts. "
                    "sample_missing=%s",
                    mismatched,
                    total_ref,
                    mismatch_examples,
                )
                setattr(self, "_maxent_ref_ema_mismatch_warned", True)

        def _resolve_effective_maxent_alpha(
            self,
            mode: str,
            *,
            measured_kl_override: Optional[float] = None,
        ) -> Tuple[float, float, Optional[float], float, bool, float, float, float, bool]:
            """Return effective MaxEnt alpha with optional KL-based up/down scaling.

            Returns ``(effective_alpha, multiplier, measured_kl, kl_threshold,
            kl_control_enabled, direction, min_multiplier, max_multiplier,
            trust_zone_blocked)`` where direction is ``+1`` (raised), ``-1``
            (lowered/blocked), or ``0`` (unchanged).
            """
            del mode
            base_alpha = float(getattr(self, "maxent_alpha", 0.0) or 0.0)
            if base_alpha <= 0.0:
                return 0.0, 1.0, None, 0.0, False, 0.0, 1.0, 1.0, False
            args = getattr(self, "args", None)
            raise_on_low_kl = bool(getattr(args, "maxent_alpha_raise_on_low_kl", False))
            lower_on_high_kl = bool(
                getattr(args, "maxent_alpha_lower_on_high_kl", False)
            )
            trust_zone_gate_enabled = bool(
                getattr(args, "maxent_alpha_disable_outside_trust_zone", False)
            )
            enabled = raise_on_low_kl or lower_on_high_kl or trust_zone_gate_enabled
            threshold_raw = getattr(args, "maxent_alpha_kl_threshold", 0.04)
            try:
                threshold = float(threshold_raw)
            except (TypeError, ValueError):
                threshold = 0.04
            max_mult_raw = getattr(args, "maxent_alpha_kl_max_multiplier", 2.0)
            try:
                max_multiplier = float(max_mult_raw)
            except (TypeError, ValueError):
                max_multiplier = 2.0
            if not math.isfinite(max_multiplier) or max_multiplier < 1.0:
                max_multiplier = 1.0

            min_mult_raw = getattr(args, "maxent_alpha_kl_min_multiplier", 0.5)
            try:
                min_multiplier = float(min_mult_raw)
            except (TypeError, ValueError):
                min_multiplier = 0.5
            if not math.isfinite(min_multiplier) or min_multiplier <= 0.0:
                min_multiplier = 0.5
            min_multiplier = min(max(min_multiplier, 1e-8), 1.0)
            if not math.isfinite(threshold) or threshold <= 0.0:
                return (
                    base_alpha,
                    1.0,
                    None,
                    threshold,
                    enabled,
                    0.0,
                    min_multiplier,
                    max_multiplier,
                    False,
                )
            if not enabled:
                return (
                    base_alpha,
                    1.0,
                    None,
                    threshold,
                    False,
                    0.0,
                    min_multiplier,
                    max_multiplier,
                    False,
                )

            measured_kl: Optional[float] = None
            if isinstance(measured_kl_override, (int, float)):
                measured_kl = float(measured_kl_override)
            else:
                cached_kl = getattr(self, "_last_train_kl_for_alpha", None)
                if isinstance(cached_kl, (int, float)):
                    measured_kl = float(cached_kl)
                else:
                    kl_history = self._metrics.get("train", {}).get("kl")
                    if kl_history:
                        try:
                            measured_kl = float(kl_history[-1])
                        except (TypeError, ValueError):
                            measured_kl = None
            if measured_kl is None:
                return (
                    base_alpha,
                    1.0,
                    None,
                    threshold,
                    True,
                    0.0,
                    min_multiplier,
                    max_multiplier,
                    False,
                )
            if not math.isfinite(measured_kl):
                if trust_zone_gate_enabled:
                    return (
                        0.0,
                        0.0,
                        measured_kl,
                        threshold,
                        True,
                        -1.0,
                        min_multiplier,
                        max_multiplier,
                        True,
                    )
                if lower_on_high_kl:
                    return (
                        base_alpha * min_multiplier,
                        min_multiplier,
                        measured_kl,
                        threshold,
                        True,
                        -1.0,
                        min_multiplier,
                        max_multiplier,
                        False,
                    )
                return (
                    base_alpha,
                    1.0,
                    measured_kl,
                    threshold,
                    True,
                    0.0,
                    min_multiplier,
                    max_multiplier,
                    False,
                )

            gain_raw = getattr(args, "maxent_alpha_kl_gain", 1.0)
            try:
                gain = float(gain_raw)
            except (TypeError, ValueError):
                gain = 1.0
            if not math.isfinite(gain) or gain < 0.0:
                gain = 0.0

            direction = 0.0
            multiplier = 1.0
            trust_zone_blocked = False
            if measured_kl < threshold and raise_on_low_kl:
                low_kl_frac = max(threshold - measured_kl, 0.0) / max(threshold, 1e-8)
                multiplier = 1.0 + gain * low_kl_frac
                direction = 1.0
            elif measured_kl > threshold and trust_zone_gate_enabled:
                multiplier = 0.0
                direction = -1.0
                trust_zone_blocked = True
            elif measured_kl > threshold and lower_on_high_kl:
                high_kl_frac = max(measured_kl - threshold, 0.0) / max(threshold, 1e-8)
                multiplier = 1.0 / (1.0 + gain * high_kl_frac)
                direction = -1.0
            if not math.isfinite(multiplier):
                multiplier = 1.0
                direction = 0.0
                trust_zone_blocked = False
            if trust_zone_blocked:
                effective_alpha = 0.0
            else:
                multiplier = min(max(multiplier, min_multiplier), max_multiplier)
                effective_alpha = base_alpha * multiplier
            return (
                effective_alpha,
                multiplier,
                measured_kl,
                threshold,
                True,
                direction,
                min_multiplier,
                max_multiplier,
                trust_zone_blocked,
            )

        def _log_grpo_diversity(
            self,
            outputs: Dict[str, Any],
            *,
            mode: str,
        ) -> None:
            completion_ids = outputs.get("completion_ids")
            if not isinstance(completion_ids, torch.Tensor):
                return
            tokenizer = getattr(self, "processing_class", None)
            decode = getattr(tokenizer, "batch_decode", None)
            if not callable(decode):
                return
            try:
                decode_fn = cast(Callable[..., List[str]], decode)
                completions_text = decode_fn(  # pylint: disable=not-callable
                    completion_ids, skip_special_tokens=True
                )
            except Exception:
                return
            group_size = max(int(getattr(self, "num_generations", 1) or 1), 1)
            usable = len(completions_text) - (len(completions_text) % group_size)
            if usable <= 0:
                return
            if usable != len(completions_text):
                completions_text = completions_text[:usable]
            grouped = [
                completions_text[i : i + group_size]
                for i in range(0, usable, group_size)
            ]
            local_only_eval = (
                _use_local_only_lightweight_eval_metrics(self, mode)
                or _use_local_only_eval_diversity_metrics(self, mode)
            )
            if local_only_eval and not _is_main_process(self):
                return
            use_tokenizer = (
                tokenizer
                if callable(getattr(tokenizer, "encode", None)) or callable(tokenizer)
                else None
            )
            metrics = _completion_diversity_metrics(
                grouped,
                tokenizer=use_tokenizer,
                accelerator=None if local_only_eval else self.accelerator,
            )
            if metrics:
                for key, val in metrics.items():
                    self._append_metric_value(
                        mode,
                        f"completions/diversity/{key}",
                        float(val),
                    )

        def _maybe_log_rich_rollout_sidecar(
            self,
            inputs: List[Dict[str, Any]],
            outputs: Dict[str, Any],
            *,
            mode: str,
        ) -> None:
            """Write prompt-major rollout rows for distribution figures."""
            if mode != "train":
                return
            args = getattr(self, "args", None)
            if not bool(getattr(args, "rich_log_completions", False)):
                return
            if not _is_main_process(self):
                return
            output_dir = getattr(args, "output_dir", None)
            if not isinstance(output_dir, str) or not output_dir.strip():
                return
            completion_ids = outputs.get("completion_ids")
            completion_mask = outputs.get("completion_mask")
            advantages = outputs.get("advantages")
            tokenizer = getattr(self, "processing_class", None)
            decode = getattr(tokenizer, "decode", None)
            if not isinstance(completion_ids, torch.Tensor) or not isinstance(
                completion_mask, torch.Tensor
            ):
                return
            if not isinstance(advantages, torch.Tensor):
                return
            if not callable(decode):
                return
            rewards_local = self._recompute_local_rewards_for_outputs(inputs, outputs)
            if not isinstance(rewards_local, torch.Tensor) or rewards_local.numel() <= 0:
                return
            total_rows = min(
                int(completion_ids.size(0)),
                int(completion_mask.size(0)),
                int(advantages.numel()),
                int(rewards_local.numel()),
                len(inputs),
            )
            if total_rows <= 0:
                return
            prompt_texts = [
                _build_prompt_text(example, tokenizer)
                for example in inputs[:total_rows]
            ]
            completion_texts: List[str] = []
            for row_idx in range(total_rows):
                mask_row = completion_mask[row_idx].to(torch.long)
                active_ids = [
                    int(tok.item())
                    for tok, keep in zip(completion_ids[row_idx], mask_row)
                    if int(keep.item()) != 0
                ]
                try:
                    completion_text = str(decode(active_ids, skip_special_tokens=True))
                except Exception:
                    completion_text = ""
                completion_texts.append(completion_text)
            q_grouped = outputs.get("maxent_listwise_q")
            q_values: Optional[List[float]] = None
            if isinstance(q_grouped, torch.Tensor) and q_grouped.numel() > 0:
                try:
                    q_values = [
                        float(val)
                        for val in q_grouped.detach().to(torch.float32).reshape(-1).tolist()
                    ][:total_rows]
                except Exception:
                    q_values = None
            step_value = int(getattr(getattr(self, "state", None), "global_step", 0) or 0)
            if mode == "train":
                step_value += 1
            columns, rows = _build_rich_rollout_rows(
                step=step_value,
                group_size=max(int(getattr(self, "num_generations", 1) or 1), 1),
                prompt_texts=prompt_texts,
                completion_texts=completion_texts,
                rewards=[
                    float(val)
                    for val in rewards_local[:total_rows].detach().to(torch.float32).tolist()
                ],
                advantages=[
                    float(val)
                    for val in advantages[:total_rows].detach().to(torch.float32).tolist()
                ],
                q_values=q_values,
            )
            if not rows:
                return
            table_key = str(
                getattr(args, "rich_log_completions_key", "rich_completions")
                or "rich_completions"
            ).strip()
            path = _write_rich_rollout_sidecar(
                output_dir=output_dir.strip(),
                table_key=table_key,
                step=step_value,
                columns=columns,
                rows=rows,
            )
            if path:
                LOG.info(
                    "Wrote rich rollout sidecar | step=%d rows=%d path=%s",
                    step_value,
                    len(rows),
                    path,
                )

        def _recompute_grouped_advantages(
            self,
            inputs: List[Dict[str, Any]],
            outputs: Dict[str, Any],
            *,
            group_size: Optional[int] = None,
        ) -> Optional[torch.Tensor]:
            """Recompute local GRPO-style advantages after completion postprocessing."""

            rewards_local = self._recompute_local_rewards_for_outputs(inputs, outputs)
            if not isinstance(rewards_local, torch.Tensor) or rewards_local.numel() <= 0:
                return None
            rewards = gather(rewards_local)
            if not isinstance(rewards, torch.Tensor) or rewards.numel() <= 0:
                return None
            effective_group = max(
                int(group_size or getattr(self, "num_generations", 1) or 1),
                1,
            )
            if int(rewards.numel()) % effective_group != 0:
                return None
            grouped_rewards = rewards.view(-1, effective_group)
            mean_grouped_rewards = grouped_rewards.mean(dim=1)
            std_grouped_rewards = grouped_rewards.std(dim=1)
            repeated_means = mean_grouped_rewards.repeat_interleave(
                effective_group, dim=0
            )
            repeated_stds = std_grouped_rewards.repeat_interleave(
                effective_group, dim=0
            )
            advantages = rewards - repeated_means
            if bool(getattr(self, "scale_rewards", False)):
                advantages = advantages / (repeated_stds + 1e-4)
            local_count = len(inputs)
            process_index = int(getattr(self.accelerator, "process_index", 0) or 0)
            process_slice = slice(
                process_index * local_count,
                (process_index + 1) * local_count,
            )
            outputs["advantages"] = advantages[process_slice].to(
                device=rewards_local.device,
                dtype=torch.float32,
            )
            return rewards_local

        def _maybe_backfill_old_per_token_logps(
            self,
            outputs: Dict[str, Any],
            *,
            mode: str,
        ) -> None:
            """Populate rollout behavior log-probs when the parent TRL path omits them.

            TRL intentionally skips ``old_per_token_logps`` when the current rollout can
            reuse the policy log-probs in the loss. SEED-GRPO needs those rollout
            log-probs earlier, during advantage scaling, so backfill them from the
            current policy before any truncation/postprocessing changes the sequence.
            """

            if mode != "train":
                return
            if isinstance(outputs.get("old_per_token_logps"), torch.Tensor):
                return
            args = getattr(self, "args", None)
            if not bool(getattr(args, "seed_grpo_enabled", False)):
                return
            prompt_ids = outputs.get("prompt_ids")
            prompt_mask = outputs.get("prompt_mask")
            completion_ids = outputs.get("completion_ids")
            completion_mask = outputs.get("completion_mask")
            if not isinstance(prompt_ids, torch.Tensor) or not isinstance(
                prompt_mask, torch.Tensor
            ):
                return
            if not isinstance(completion_ids, torch.Tensor) or not isinstance(
                completion_mask, torch.Tensor
            ):
                return
            logits_to_keep = int(completion_ids.size(1))
            if logits_to_keep <= 0:
                return
            input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
            attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
            configured_batch_size = int(
                getattr(args, "per_device_train_batch_size", 1) or 1
            )
            chunk_size = int(
                getattr(args, "maxent_logprob_chunk_size", 0)
                or configured_batch_size
                or 1
            )
            behavior_source = str(
                getattr(args, "behavior_logprobs_source", "model") or "model"
            ).strip().lower()
            if behavior_source not in {"", "model"} and not bool(
                getattr(self, "_seed_grpo_behavior_source_warned", False)
            ):
                LOG.warning(
                    "SEED-GRPO requested behavior_logprobs_source=%s, but the shared "
                    "trainer rollout did not return per-token behavior log-probs. "
                    "Recomputing them from the current policy for rollout parity.",
                    behavior_source,
                )
                setattr(self, "_seed_grpo_behavior_source_warned", True)
            if not bool(getattr(self, "_seed_grpo_backfill_preflight_logged", False)):
                tokenizer = getattr(self, "processing_class", None)
                upper_bound = _resolve_token_id_upper_bound(
                    getattr(self, "model", None),
                    tokenizer,
                )

                def _token_range_stats(
                    tensor: torch.Tensor,
                ) -> Tuple[Optional[int], Optional[int], int]:
                    try:
                        min_token = int(tensor.min().item())
                        max_token = int(tensor.max().item())
                    except Exception:
                        min_token = None
                        max_token = None
                    invalid_count = 0
                    if isinstance(upper_bound, int) and upper_bound > 0:
                        try:
                            invalid_count = int(
                                ((tensor < 0) | (tensor >= upper_bound))
                                .to(torch.long)
                                .sum()
                                .item()
                            )
                        except Exception:
                            invalid_count = 0
                    return min_token, max_token, invalid_count

                prompt_min, prompt_max, prompt_invalid = _token_range_stats(prompt_ids)
                completion_min, completion_max, completion_invalid = _token_range_stats(
                    completion_ids
                )
                LOG.info(
                    "SEED-GRPO backfill preflight | upper_bound=%s | "
                    "prompt_ids[min=%s max=%s invalid=%d] | "
                    "completion_ids[min=%s max=%s invalid=%d]",
                    upper_bound,
                    prompt_min,
                    prompt_max,
                    prompt_invalid,
                    completion_min,
                    completion_max,
                    completion_invalid,
                )
                setattr(self, "_seed_grpo_backfill_preflight_logged", True)
            try:
                with torch.no_grad():
                    old_per_token_logps = self._get_per_token_logps(
                        self.model,
                        input_ids,
                        attention_mask,
                        logits_to_keep,
                        batch_size=chunk_size,
                    )
            except Exception as exc:
                if not bool(getattr(self, "_seed_grpo_backfill_warned", False)):
                    LOG.warning(
                        "SEED-GRPO could not backfill rollout log-prob metadata; "
                        "falling back to unscaled GRPO advantages: %s",
                        exc,
                    )
                    setattr(self, "_seed_grpo_backfill_warned", True)
                return
            if not isinstance(old_per_token_logps, torch.Tensor):
                return
            outputs["old_per_token_logps"] = old_per_token_logps.detach().to(
                device=completion_ids.device,
                dtype=torch.float32,
            )
            self._append_metric_value(
                "train",
                "seed_grpo/behavior_logprobs_backfilled",
                1.0,
                include_legacy_aliases=False,
            )

        def _sanitize_rollout_token_ids(
            self,
            outputs: Dict[str, Any],
            *,
            mode: str,
        ) -> None:
            """Clamp rollout token ids into the train-time model vocab range.

            This keeps the shared rollout pipeline robust when tokenizer/vLLM ids
            exceed the policy model vocab, which otherwise crashes later log-prob
            gathers with CUDA index assertions.
            """

            del mode
            setattr(self, "_last_rollout_invalid_token_id_count", 0.0)
            unwrap_fn = getattr(getattr(self, "accelerator", None), "unwrap_model", None)
            base_model = getattr(self, "model", None)
            if callable(unwrap_fn):
                try:
                    base_model = unwrap_fn(base_model)
                except Exception:
                    base_model = getattr(self, "model", None)
            tokenizer = getattr(self, "processing_class", None)
            vocab_size = _resolve_token_id_upper_bound(base_model, tokenizer)
            if not isinstance(vocab_size, int) or vocab_size <= 0:
                return
            replacement_id = _coerce_optional_int(getattr(tokenizer, "pad_token_id", None))
            if replacement_id is None or replacement_id < 0 or replacement_id >= vocab_size:
                replacement_id = _coerce_optional_int(getattr(tokenizer, "eos_token_id", None))
            if replacement_id is None or replacement_id < 0 or replacement_id >= vocab_size:
                replacement_id = max(vocab_size - 1, 0)

            total_invalid = 0
            details: List[str] = []
            for key in ("prompt_ids", "completion_ids"):
                tensor = outputs.get(key)
                if not isinstance(tensor, torch.Tensor):
                    continue
                if tensor.dtype.is_floating_point or tensor.dtype == torch.bool:
                    continue
                invalid_mask = (tensor < 0) | (tensor >= vocab_size)
                invalid_count = int(invalid_mask.to(torch.long).sum().item())
                if invalid_count <= 0:
                    continue
                total_invalid += invalid_count
                try:
                    invalid_vals = tensor[invalid_mask]
                    min_invalid = int(invalid_vals.min().item())
                    max_invalid = int(invalid_vals.max().item())
                except Exception:
                    min_invalid = 0
                    max_invalid = 0
                sanitized = tensor.clone()
                sanitized[invalid_mask] = int(replacement_id)
                outputs[key] = sanitized
                details.append(
                    f"{key}:count={invalid_count}:min={min_invalid}:max={max_invalid}"
                )

            if total_invalid <= 0:
                return
            setattr(self, "_last_rollout_invalid_token_id_count", float(total_invalid))
            self._append_metric_value(
                "train",
                "rollout/invalid_token_id_count",
                float(total_invalid),
                include_legacy_aliases=False,
            )
            self._append_metric_value(
                "train",
                "rollout/invalid_token_id_replacement",
                float(replacement_id),
                include_legacy_aliases=False,
            )
            if not bool(getattr(self, "_invalid_rollout_token_ids_warned", False)):
                LOG.warning(
                    "Sanitized %d rollout token ids outside model vocab_size=%d using replacement_id=%d (%s)",
                    total_invalid,
                    vocab_size,
                    replacement_id,
                    ", ".join(details),
                )
                setattr(self, "_invalid_rollout_token_ids_warned", True)
            fatal_flag = str(
                os.getenv("MAXENT_FATAL_INVALID_ROLLOUT_TOKEN_IDS", "0")
            ).strip().lower()
            if fatal_flag in {"1", "true", "yes", "on"}:
                self._append_metric_value(
                    "train",
                    "rollout/invalid_token_id_guard_triggered",
                    1.0,
                    include_legacy_aliases=False,
                )
                raise RuntimeError(
                    "Detected rollout token ids outside the tokenizer-addressable "
                    f"range (count={total_invalid}, vocab_size={vocab_size}, "
                    f"replacement_id={replacement_id})."
                )

        def _maybe_apply_seed_grpo_advantages(
            self,
            inputs: List[Dict[str, Any]],
            outputs: Dict[str, Any],
            *,
            mode: str,
            group_size: Optional[int] = None,
        ) -> None:
            """Apply SEED-GRPO semantic-entropy scaling to prepared advantages."""

            if mode != "train":
                return
            args = getattr(self, "args", None)
            if not bool(getattr(args, "seed_grpo_enabled", False)):
                return
            advantages = outputs.get("advantages")
            completion_ids = outputs.get("completion_ids")
            completion_mask = outputs.get("completion_mask")
            old_per_token_logps = outputs.get("old_per_token_logps")
            if not isinstance(advantages, torch.Tensor):
                return
            if not isinstance(completion_ids, torch.Tensor):
                return
            if not isinstance(completion_mask, torch.Tensor):
                return
            if not isinstance(old_per_token_logps, torch.Tensor):
                if not bool(getattr(self, "_seed_grpo_missing_logprobs_warned", False)):
                    LOG.warning(
                        "SEED-GRPO enabled but rollout log-prob metadata is missing; "
                        "falling back to unscaled GRPO advantages."
                    )
                    setattr(self, "_seed_grpo_missing_logprobs_warned", True)
                return
            if old_per_token_logps.ndim != 2:
                if not bool(
                    getattr(self, "_seed_grpo_logprob_rank_mismatch_warned", False)
                ):
                    LOG.warning(
                        "SEED-GRPO requires a rank-2 per-token logprob tensor; got shape=%s. "
                        "Falling back to unscaled GRPO advantages.",
                        getattr(old_per_token_logps, "shape", None),
                    )
                    setattr(
                        self,
                        "_seed_grpo_logprob_rank_mismatch_warned",
                        True,
                    )
                return
            if int(old_per_token_logps.size(0)) != int(completion_ids.size(0)):
                if not bool(
                    getattr(self, "_seed_grpo_logprob_row_mismatch_warned", False)
                ):
                    LOG.warning(
                        "SEED-GRPO requires rollout logprobs aligned with completions; "
                        "got logprob_rows=%d completion_rows=%d. Falling back to unscaled GRPO advantages.",
                        int(old_per_token_logps.size(0)),
                        int(completion_ids.size(0)),
                    )
                    setattr(
                        self,
                        "_seed_grpo_logprob_row_mismatch_warned",
                        True,
                    )
                return
            try:
                required_width = int(
                    completion_mask.to(torch.long).sum(dim=1).max().item()
                )
            except Exception:
                required_width = 0
            if required_width > int(old_per_token_logps.size(1)):
                if not bool(
                    getattr(self, "_seed_grpo_logprob_width_mismatch_warned", False)
                ):
                    LOG.warning(
                        "SEED-GRPO requires rollout logprobs wide enough for active completion tokens; "
                        "got logprob_width=%d required_width=%d. Falling back to unscaled GRPO advantages.",
                        int(old_per_token_logps.size(1)),
                        required_width,
                    )
                    setattr(
                        self,
                        "_seed_grpo_logprob_width_mismatch_warned",
                        True,
                    )
                return
            tokenizer = getattr(self, "processing_class", None)
            decode = getattr(tokenizer, "decode", None)
            if not callable(decode):
                return
            effective_group = max(
                int(group_size or getattr(self, "num_generations", 1) or 1),
                1,
            )
            total = int(completion_ids.size(0))
            if total <= 0 or total % effective_group != 0:
                if not bool(getattr(self, "_seed_grpo_group_shape_warned", False)):
                    LOG.warning(
                        "SEED-GRPO requires local rollout batches to contain whole "
                        "prompt groups; got batch=%d with num_generations=%d. "
                        "Falling back to unscaled GRPO advantages.",
                        total,
                        effective_group,
                    )
                    setattr(self, "_seed_grpo_group_shape_warned", True)
                return

            grouped_completions: List[List[str]] = []
            grouped_ref_meta: List[List[Dict[str, Any]]] = []
            for start in range(0, total, effective_group):
                completion_group: List[str] = []
                meta_group: List[Dict[str, Any]] = []
                for row_idx in range(start, start + effective_group):
                    mask_row = completion_mask[row_idx].to(torch.long)
                    active_len = int(mask_row.sum().item())
                    active_ids = [
                        int(tok.item())
                        for tok, keep in zip(completion_ids[row_idx], mask_row)
                        if int(keep.item()) != 0
                    ]
                    try:
                        completion_text = str(
                            decode(active_ids, skip_special_tokens=True)
                        )
                    except Exception:
                        completion_text = ""
                    completion_group.append(completion_text)
                    token_logps = old_per_token_logps[row_idx, :active_len]
                    logprob_sum = (
                        float(token_logps.sum().item()) if active_len > 0 else 0.0
                    )
                    meta_group.append(
                        {
                            "logprob_sum": logprob_sum,
                            "token_count": active_len,
                        }
                    )
                grouped_completions.append(completion_group)
                grouped_ref_meta.append(meta_group)

            try:
                (
                    semantic_entropies,
                    advantage_scales,
                    alpha_effective,
                    max_possible_entropy,
                ) = _compute_seed_grpo_statistics(
                    SimpleNamespace(
                        grouped_completions=grouped_completions,
                        grouped_ref_meta=grouped_ref_meta,
                    ),
                    alpha=float(getattr(args, "seed_grpo_alpha", 0.0417) or 0.0417),
                    normalize_by_max_entropy=bool(
                        getattr(
                            args,
                            "seed_grpo_alpha_normalize_by_max_entropy",
                            True,
                        )
                    ),
                    length_normalize_logprobs=bool(
                        getattr(args, "seed_grpo_length_normalize_logprobs", True)
                    ),
                    num_generations=int(
                        getattr(args, "num_generations", effective_group)
                        or effective_group
                    ),
                )
            except Exception as exc:
                if not bool(getattr(self, "_seed_grpo_compute_warned", False)):
                    LOG.warning(
                        "SEED-GRPO scaling failed during rollout prep; falling back "
                        "to unscaled GRPO advantages. Error: %s",
                        exc,
                    )
                    setattr(self, "_seed_grpo_compute_warned", True)
                return

            if not advantage_scales:
                return
            repeated_scales = torch.tensor(
                [
                    float(scale)
                    for scale in advantage_scales
                    for _ in range(effective_group)
                ],
                device=advantages.device,
                dtype=advantages.dtype,
            )
            if int(repeated_scales.numel()) != int(advantages.numel()):
                return
            outputs["advantages"] = advantages * repeated_scales
            outputs["seed_grpo_semantic_entropies"] = torch.tensor(
                semantic_entropies,
                device=advantages.device,
                dtype=torch.float32,
            )
            outputs["seed_grpo_advantage_scales"] = torch.tensor(
                advantage_scales,
                device=advantages.device,
                dtype=torch.float32,
            )

            # Keep SEED diagnostics rank-local. These metrics are not used for
            # correctness, and cross-rank gathers here can desync collectives if
            # one rank bails out of SEED scaling earlier than another.
            local_entropies = _local_metric_tensor(outputs["seed_grpo_semantic_entropies"])
            if isinstance(local_entropies, torch.Tensor) and local_entropies.numel() > 0:
                self._append_metric_value(
                    mode,
                    "seed_grpo/semantic_entropy_mean",
                    float(local_entropies.mean().item()),
                )
                self._append_metric_value(
                    mode,
                    "seed_grpo/semantic_entropy_min",
                    float(local_entropies.min().item()),
                )
                self._append_metric_value(
                    mode,
                    "seed_grpo/semantic_entropy_max",
                    float(local_entropies.max().item()),
                )
            local_scales = _local_metric_tensor(outputs["seed_grpo_advantage_scales"])
            if isinstance(local_scales, torch.Tensor) and local_scales.numel() > 0:
                self._append_metric_value(
                    mode,
                    "seed_grpo/advantage_scale_mean",
                    float(local_scales.mean().item()),
                )
                self._append_metric_value(
                    mode,
                    "seed_grpo/advantage_scale_min",
                    float(local_scales.min().item()),
                )
                self._append_metric_value(
                    mode,
                    "seed_grpo/advantage_scale_max",
                    float(local_scales.max().item()),
                )
            self._append_metric_value(
                mode,
                "seed_grpo/alpha_effective",
                float(alpha_effective),
            )
            self._append_metric_value(
                mode,
                "seed_grpo/max_possible_entropy",
                float(max_possible_entropy),
            )

        def _maybe_apply_seed_grpo_advantages_in_loss(
            self,
            inputs: Dict[str, Any],
            *,
            completion_ids: torch.Tensor,
            completion_mask: torch.Tensor,
            behavior_logps: torch.Tensor,
            mode: str,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """Apply SEED-GRPO scaling from the normal loss-path logprobs.

            The shared rollout path can omit ``old_per_token_logps`` when the
            current policy and rollout behavior are the same. For SEED-GRPO we
            only need grouped logprob sums/token counts, so defer that scaling to
            the loss path and reuse the already-computed policy logprobs instead
            of running a second scorer pass immediately after generation.
            """

            advantages = inputs.get("advantages")
            if not isinstance(advantages, torch.Tensor):
                return advantages, behavior_logps
            args = getattr(self, "args", None)
            if mode != "train" or not bool(getattr(args, "seed_grpo_enabled", False)):
                return advantages, behavior_logps
            if not isinstance(behavior_logps, torch.Tensor):
                return advantages, behavior_logps
            scaled_batch_ids = getattr(self, "_seed_grpo_scaled_batch_ids", None)
            if not isinstance(scaled_batch_ids, set):
                scaled_batch_ids = set()
                setattr(self, "_seed_grpo_scaled_batch_ids", scaled_batch_ids)
            if id(inputs) in scaled_batch_ids:
                existing_advantages = inputs.get("advantages")
                existing_logps = inputs.get("old_per_token_logps")
                if isinstance(existing_advantages, torch.Tensor) and isinstance(
                    existing_logps, torch.Tensor
                ):
                    return existing_advantages, existing_logps
                return advantages, behavior_logps

            seed_outputs: Dict[str, Any] = {
                "advantages": advantages,
                "completion_ids": completion_ids,
                "completion_mask": completion_mask,
                "old_per_token_logps": behavior_logps,
            }
            self._maybe_apply_seed_grpo_advantages([], seed_outputs, mode=mode)
            scaled_advantages = seed_outputs.get("advantages")
            scaled_logps = seed_outputs.get("old_per_token_logps")
            if not isinstance(scaled_advantages, torch.Tensor):
                scaled_advantages = advantages
            if not isinstance(scaled_logps, torch.Tensor):
                scaled_logps = behavior_logps
            inputs["advantages"] = scaled_advantages
            inputs["old_per_token_logps"] = scaled_logps.detach()
            scaled_batch_ids.add(id(inputs))
            self._append_metric_value(
                mode,
                "seed_grpo/behavior_logprobs_deferred_to_loss",
                1.0,
                include_legacy_aliases=False,
            )
            return scaled_advantages, scaled_logps

        def _maybe_truncate_completions_at_first_boxed_answer(
            self,
            inputs: List[Dict[str, Any]],
            outputs: Dict[str, Any],
            *,
            mode: str,
            group_size: Optional[int] = None,
            update_logged_metrics: bool = True,
        ) -> None:
            """Trim generated completions at the first valid boxed answer when configured."""

            args = getattr(self, "args", None)
            if not bool(
                getattr(args, "truncate_completions_at_first_boxed_answer", False)
            ):
                return
            completion_ids = outputs.get("completion_ids")
            if not isinstance(completion_ids, torch.Tensor):
                return
            completion_mask = outputs.get("completion_mask")
            if not isinstance(completion_mask, torch.Tensor):
                completion_mask = _apply_eos_completion_mask(
                    completion_ids,
                    getattr(self.processing_class, "eos_token_id", None),
                )
            if not isinstance(completion_mask, torch.Tensor):
                return
            tokenizer = getattr(self, "processing_class", None)
            decode = getattr(tokenizer, "decode", None)
            if not callable(decode):
                return

            old_per_token_logps = outputs.get("old_per_token_logps")
            old_log_rows: List[torch.Tensor] = []
            truncated_rows: List[List[int]] = []
            trimmed = 0
            for row, mask_row in zip(completion_ids, completion_mask):
                active_len = int(mask_row.to(torch.long).sum().item())
                active_ids = [
                    int(tok.item()) for tok in row[:active_len]
                ]
                if active_len <= 0:
                    truncated_rows.append([])
                    if isinstance(old_per_token_logps, torch.Tensor):
                        old_log_rows.append(
                            old_per_token_logps.new_zeros((0,))
                        )
                    continue
                try:
                    text = str(decode(active_ids, skip_special_tokens=True))
                except Exception:
                    text = ""
                truncated_text = truncate_after_first_boxed_answer(text)
                prefix_len = _find_token_prefix_len_for_text(
                    tokenizer,
                    active_ids,
                    truncated_text,
                )
                if (
                    truncated_text
                    and truncated_text != text
                    and prefix_len is not None
                    and 0 < prefix_len < active_len
                ):
                    active_ids = active_ids[:prefix_len]
                    trimmed += 1
                truncated_rows.append(active_ids)
                if isinstance(old_per_token_logps, torch.Tensor):
                    old_log_rows.append(
                        old_per_token_logps[
                            len(old_log_rows), : len(active_ids)
                        ].detach()
                    )
            if trimmed <= 0:
                return

            pad_token_id = getattr(tokenizer, "pad_token_id", None)
            if pad_token_id is None:
                pad_token_id = getattr(tokenizer, "eos_token_id", 0)
            new_completion_ids, new_completion_mask = _pad_completion_rows(
                truncated_rows,
                pad_token_id=int(pad_token_id or 0),
                device=completion_ids.device,
            )
            outputs["completion_ids"] = new_completion_ids
            outputs["completion_mask"] = new_completion_mask
            if isinstance(old_per_token_logps, torch.Tensor):
                outputs["old_per_token_logps"] = _pad_logprob_rows(
                    old_log_rows,
                    device=old_per_token_logps.device,
                    dtype=old_per_token_logps.dtype,
                )

            skip_advantage_recompute = _use_lightweight_greedy_eval(
                self,
                mode,
            )
            rewards_local: Optional[torch.Tensor] = None
            if not skip_advantage_recompute:
                rewards_local = self._recompute_grouped_advantages(
                    inputs,
                    outputs,
                    group_size=group_size,
                )
            if update_logged_metrics:
                if isinstance(rewards_local, torch.Tensor) and rewards_local.numel() > 0:
                    gathered_rewards = _metric_tensor_for_logging(
                        self,
                        rewards_local.to(torch.float32),
                        mode=mode,
                    )
                    if (
                        isinstance(gathered_rewards, torch.Tensor)
                        and gathered_rewards.numel() > 0
                    ):
                        effective_group = max(
                            int(group_size or getattr(self, "num_generations", 1) or 1),
                            1,
                        )
                        if int(gathered_rewards.numel()) % effective_group == 0:
                            grouped_rewards = gathered_rewards.view(-1, effective_group)
                            reward_mean = grouped_rewards.mean(dim=1)
                            reward_std = grouped_rewards.std(dim=1)
                            self._set_latest_metric_value(
                                mode,
                                "reward",
                                float(reward_mean.mean().item()),
                            )
                            self._set_latest_metric_value(
                                mode,
                                "reward_std",
                                float(reward_std.mean().item()),
                            )
                            self._set_latest_metric_value(
                                mode,
                                "frac_reward_zero_std",
                                float(
                                    torch.isclose(
                                        reward_std,
                                        torch.zeros_like(reward_std),
                                    )
                                    .to(torch.float32)
                                    .mean()
                                    .item()
                                ),
                            )

                completion_lengths = new_completion_mask.sum(dim=1).to(torch.float32)
                gathered_lengths = _metric_tensor_for_logging(
                    self,
                    completion_lengths,
                    mode=mode,
                )
                if (
                    isinstance(gathered_lengths, torch.Tensor)
                    and gathered_lengths.numel() > 0
                ):
                    self._set_latest_metric_value(
                        mode,
                        "completions/mean_length",
                        float(gathered_lengths.mean().item()),
                    )
                    self._set_latest_metric_value(
                        mode,
                        "completions/min_length",
                        float(gathered_lengths.min().item()),
                    )
                    self._set_latest_metric_value(
                        mode,
                        "completions/max_length",
                        float(gathered_lengths.max().item()),
                    )
                if not (
                    _use_local_only_lightweight_eval_metrics(self, mode)
                    and not _is_main_process(self)
                ):
                    trim_ratio = float(trimmed) / float(max(len(truncated_rows), 1))
                    self._append_metric_value(
                        mode,
                        "completions/boxed_stop_ratio",
                        trim_ratio,
                    )

        def _prepare_greedy_eval_prompt_batch(
            self,
            inputs: List[Dict[str, Any]],
        ) -> Tuple[List[Dict[str, Any]], torch.Tensor, torch.Tensor]:
            """Deduplicate prompt-major eval groups and tokenize one prompt per group."""

            if bool(
                getattr(self, "_local_only_eval_prompt_major_loader_active", False)
            ):
                prompt_inputs = list(inputs)
            else:
                group_size = max(int(getattr(self, "num_generations", 1) or 1), 1)
                usable = len(inputs) - (len(inputs) % group_size)
                if usable <= 0:
                    raise ValueError(
                        "Greedy eval requires at least one full prompt group."
                    )
                prompt_inputs = list(inputs[:usable:group_size])
            if not prompt_inputs:
                raise ValueError("Greedy eval requires at least one prompt example.")
            tokenizer = getattr(self, "processing_class", None)
            prompts_text = [
                _build_prompt_text(example, tokenizer) for example in prompt_inputs
            ]
            prompt_tensors = tokenizer(
                text=prompts_text,
                return_tensors="pt",
                padding=True,
                padding_side="left",
                add_special_tokens=False,
            )
            prompt_tensors = self._move_prompt_tensors_to_device(prompt_tensors)
            prompt_ids = prompt_tensors["input_ids"]
            prompt_mask = prompt_tensors["attention_mask"]
            max_prompt_length = getattr(self, "max_prompt_length", None)
            if max_prompt_length is not None:
                prompt_ids = prompt_ids[:, -max_prompt_length :]
                prompt_mask = prompt_mask[:, -max_prompt_length :]
            return prompt_inputs, prompt_ids, prompt_mask

        def _move_prompt_tensors_to_device(self, value: Any) -> Any:
            """Move tokenizer outputs to device without re-entering TRL batch prep."""

            device = getattr(getattr(self, "accelerator", None), "device", None)
            if device is None:
                device = getattr(getattr(self, "args", None), "device", None)
            if isinstance(value, torch.Tensor):
                return value.to(device=device) if device is not None else value
            move_fn = getattr(value, "to", None)
            if callable(move_fn) and not isinstance(value, (str, bytes)):
                if device is None:
                    return value
                try:
                    return move_fn(device=device)
                except TypeError:
                    try:
                        return move_fn(device)
                    except TypeError:
                        pass
            if isinstance(value, Mapping):
                return {
                    key: self._move_prompt_tensors_to_device(item)
                    for key, item in value.items()
                }
            if isinstance(value, list):
                return [self._move_prompt_tensors_to_device(item) for item in value]
            if isinstance(value, tuple):
                return tuple(
                    self._move_prompt_tensors_to_device(item) for item in value
                )
            return value

        def _generate_greedy_eval_outputs(
            self,
            inputs: List[Dict[str, Any]],
        ) -> Dict[str, Any]:
            """Generate one greedy completion per prompt for lightweight eval."""

            prompt_inputs, prompt_ids, prompt_mask = self._prepare_greedy_eval_prompt_batch(
                inputs
            )
            lightweight_eval = _use_lightweight_greedy_eval(self, "eval")
            tokenizer = getattr(self, "processing_class", None)
            eos_token_id = getattr(tokenizer, "eos_token_id", None)
            pad_token_id = getattr(tokenizer, "pad_token_id", None)
            if pad_token_id is None:
                pad_token_id = eos_token_id
            max_new_tokens = int(
                getattr(self, "max_completion_length", 0)
                or getattr(getattr(self, "args", None), "max_completion_length", 0)
                or 0
            )
            if max_new_tokens <= 0:
                raise ValueError("Greedy eval requires a positive max_completion_length.")

            unwrap_fn = getattr(self.accelerator, "unwrap_model", None)
            gen_model = self.model
            if callable(unwrap_fn):
                try:
                    gen_model = unwrap_fn(gen_model)
                except Exception:
                    gen_model = self.model
            generate_fn = getattr(gen_model, "generate", None)
            if not callable(generate_fn):
                raise ValueError("Greedy eval requires model.generate to be available.")
            was_training = bool(getattr(gen_model, "training", False))
            if was_training:
                gen_model.eval()
            generate_kwargs: Dict[str, Any] = {
                "input_ids": prompt_ids,
                "attention_mask": prompt_mask,
                "do_sample": False,
                "max_new_tokens": max_new_tokens,
                "num_return_sequences": 1,
            }
            if pad_token_id is not None:
                generate_kwargs["pad_token_id"] = int(pad_token_id)
            if eos_token_id is not None:
                generate_kwargs["eos_token_id"] = int(eos_token_id)
            if (
                getattr(self.accelerator, "num_processes", 1) > 1
                and not lightweight_eval
            ):
                generate_kwargs["synced_gpus"] = True
            try:
                with torch.inference_mode():
                    try:
                        generated = generate_fn(**generate_kwargs)
                    except TypeError as exc:
                        if "synced_gpus" not in str(exc):
                            raise
                        generate_kwargs.pop("synced_gpus", None)
                        generated = generate_fn(**generate_kwargs)
            finally:
                if was_training:
                    gen_model.train()

            sequences = getattr(generated, "sequences", generated)
            if not isinstance(sequences, torch.Tensor):
                raise ValueError("Greedy eval generation did not return tensor sequences.")
            prompt_width = int(prompt_ids.size(1))
            completion_ids = sequences[:, prompt_width:].contiguous()
            completion_mask = _apply_eos_completion_mask(completion_ids, eos_token_id)
            outputs: Dict[str, Any] = {
                "prompt_ids": prompt_ids,
                "prompt_mask": prompt_mask,
                "completion_ids": completion_ids,
                "completion_mask": completion_mask,
                "advantages": torch.zeros(
                    (completion_ids.size(0),),
                    dtype=torch.float32,
                    device=completion_ids.device,
                ),
                "old_per_token_logps": None,
                "_greedy_eval_precomputed": True,
                "_eval_prompt_inputs": prompt_inputs,
            }
            self._maybe_truncate_completions_at_first_boxed_answer(
                prompt_inputs,
                outputs,
                mode="eval",
                group_size=1,
                update_logged_metrics=False,
            )

            rewards_local = self._recompute_local_rewards_for_outputs(
                prompt_inputs,
                outputs,
            )
            if isinstance(rewards_local, torch.Tensor) and rewards_local.numel() > 0:
                gathered_rewards = _metric_tensor_for_logging(
                    self,
                    rewards_local.to(torch.float32),
                    mode="eval",
                )
                if isinstance(gathered_rewards, torch.Tensor) and gathered_rewards.numel() > 0:
                    self._append_metric_value(
                        "eval",
                        "reward",
                        float(gathered_rewards.mean().item()),
                    )
                    self._append_metric_value("eval", "reward_std", 0.0)
                    self._append_metric_value("eval", "frac_reward_zero_std", 1.0)

            completion_lengths = outputs["completion_mask"].sum(dim=1).to(torch.float32)
            gathered_lengths = _metric_tensor_for_logging(
                self,
                completion_lengths,
                mode="eval",
            )
            if isinstance(gathered_lengths, torch.Tensor) and gathered_lengths.numel() > 0:
                self._append_metric_value(
                    "eval",
                    "completions/mean_length",
                    float(gathered_lengths.mean().item()),
                )
                self._append_metric_value(
                    "eval",
                    "completions/min_length",
                    float(gathered_lengths.min().item()),
                )
                self._append_metric_value(
                    "eval",
                    "completions/max_length",
                    float(gathered_lengths.max().item()),
                )
                self._append_metric_value("eval", "completions/clipped_ratio", 0.0)
                self._append_metric_value(
                    "eval",
                    "completions/mean_terminated_length",
                    float(gathered_lengths.mean().item()),
                )
                self._append_metric_value(
                    "eval",
                    "completions/min_terminated_length",
                    float(gathered_lengths.min().item()),
                )
                self._append_metric_value(
                    "eval",
                    "completions/max_terminated_length",
                    float(gathered_lengths.max().item()),
                )
            return outputs

        def _log_eval_pass_at_k(
            self,
            inputs: List[Dict[str, Any]],
            outputs: Dict[str, Any],
            *,
            mode: str,
        ) -> None:
            """Log global and per-benchmark pass@8/pass@1/mean@1 metrics."""
            if mode != "eval":
                return
            group_size = max(int(getattr(self, "num_generations", 1) or 1), 1)
            if group_size <= 0:
                return
            target_k = 8
            if group_size < target_k:
                if not bool(getattr(self, "_pass_at_8_warned", False)):
                    LOG.warning(
                        "Skipping eval pass_at_8 metrics because num_generations=%d < 8.",
                        group_size,
                    )
                    setattr(self, "_pass_at_8_warned", True)
                return

            def _reshape_eval_rollouts(
                flat_values: torch.Tensor,
            ) -> Optional[torch.Tensor]:
                """Reshape prompt-major flat rollouts into prompt-major groups."""
                usable = (int(flat_values.numel()) // group_size) * group_size
                if usable <= 0:
                    return None
                if usable != int(flat_values.numel()):
                    flat_values = flat_values[:usable]
                num_prompts = usable // group_size
                if num_prompts <= 0:
                    return None
                # TRL rollout order is prompt-major: [p0g0, p0g1, ..., p1g0, p1g1, ...].
                return flat_values.view(num_prompts, group_size)

            def _gather_benchmark_ids(
                expected_flat_count: int,
            ) -> Optional[torch.Tensor]:
                """Return prompt-major benchmark ids aligned with ``successes``."""
                if expected_flat_count <= 0:
                    return None
                keys = ("eval_benchmark_id", "benchmark_id")
                raw_vals: Optional[List[Any]] = None
                for key in keys:
                    candidate = [example.get(key) for example in inputs]
                    if candidate and any(val is not None for val in candidate):
                        raw_vals = candidate
                        break
                if not raw_vals:
                    return None
                usable = min(len(raw_vals), expected_flat_count)
                usable = usable - (usable % group_size)
                if usable <= 0:
                    return None
                ids: List[int] = []
                for val in raw_vals[:usable]:
                    try:
                        ids.append(int(val) if val is not None else -1)
                    except (TypeError, ValueError):
                        ids.append(-1)
                ids_tensor = torch.tensor(
                    ids,
                    dtype=torch.long,
                    device=self.accelerator.device,
                )
                ids_global = gather(ids_tensor)
                grouped_ids = _reshape_eval_rollouts(ids_global)
                if grouped_ids is None:
                    return None
                return grouped_ids[:, 0].to(torch.long)

            def _append_per_benchmark_metrics(successes_tensor: torch.Tensor) -> None:
                """Append per-benchmark pass metrics when benchmark ids are present."""
                if successes_tensor.numel() <= 0:
                    return
                total_prompts = int(successes_tensor.size(0))
                benchmark_ids = _gather_benchmark_ids(
                    total_prompts * int(successes_tensor.size(1))
                )
                if not isinstance(benchmark_ids, torch.Tensor):
                    return
                if benchmark_ids.numel() != total_prompts:
                    return
                id_to_name = getattr(self, "eval_benchmark_id_to_name", {}) or {}
                unique_ids = torch.unique(benchmark_ids)
                for bench_id_tensor in unique_ids:
                    bench_id = int(bench_id_tensor.item())
                    if bench_id < 0:
                        continue
                    mask = benchmark_ids == bench_id_tensor
                    bench_count = int(mask.to(torch.long).sum().item())
                    if bench_count <= 0:
                        continue
                    bench_successes = successes_tensor[mask]
                    bench_pass_at_8 = float(
                        bench_successes.any(dim=1).to(torch.float32).mean().item()
                    )
                    bench_pass_at_1 = float(
                        bench_successes[:, 0].to(torch.float32).mean().item()
                    )
                    bench_mean_at_1 = float(
                        bench_successes[:, :1].to(torch.float32).mean().item()
                    )
                    bench_label = id_to_name.get(bench_id, f"BENCH_{bench_id}")
                    suffix = _metric_suffix_from_benchmark(bench_label)
                    self._append_metric_value(
                        mode, f"pass_at_8_{suffix}", bench_pass_at_8
                    )
                    self._append_metric_value(
                        mode, f"pass_at_1_{suffix}", bench_pass_at_1
                    )
                    self._append_metric_value(
                        mode, f"mean_at_1_{suffix}", bench_mean_at_1
                    )

            successes: Optional[torch.Tensor] = None
            reward_funcs = list(getattr(self, "reward_funcs", []) or [])
            if uses_pure_accuracy_math_reward(reward_funcs):
                completion_ids = outputs.get("completion_ids")
                tokenizer = getattr(self, "processing_class", None)
                decode = getattr(tokenizer, "batch_decode", None)
                if isinstance(completion_ids, torch.Tensor) and callable(decode):
                    try:
                        decode_fn = cast(Callable[..., List[str]], decode)
                        completions_text = decode_fn(  # pylint: disable=not-callable
                            completion_ids, skip_special_tokens=True
                        )
                    except Exception:
                        completions_text = []
                    answers = [str(example.get("answer", "")) for example in inputs]
                    usable_local = min(len(completions_text), len(answers))
                    usable_local = usable_local - (usable_local % group_size)
                    if usable_local > 0:
                        # Paper-facing pass metrics: exact canonical answer match,
                        # allowing only a final-line exact fallback (no shaping).
                        correctness_local = pure_accuracy_math_correctness(
                            completions_text[:usable_local],
                            answers[:usable_local],
                            allow_last_line_fallback=True,
                        )
                        local_successes = torch.tensor(
                            correctness_local,
                            dtype=torch.bool,
                            device=completion_ids.device,
                        )
                        global_successes = gather(local_successes)
                        grouped_successes = _reshape_eval_rollouts(global_successes)
                        if grouped_successes is not None:
                            successes = grouped_successes[:, :target_k]
            if successes is None:
                try:
                    rewards = self._recompute_global_rewards_for_outputs(
                        inputs, outputs
                    )
                except Exception as exc:
                    LOG.debug(
                        "Skipping eval pass@k logging due to reward error: %s", exc
                    )
                    return
                if not isinstance(rewards, torch.Tensor) or rewards.numel() <= 0:
                    return
                grouped_rewards = _reshape_eval_rollouts(rewards)
                if grouped_rewards is None:
                    return
                grouped_rewards = grouped_rewards[:, :target_k]
                # Fallback for non-math/custom reward functions.
                successes = grouped_rewards >= (
                    _PASS_METRIC_SUCCESS_REWARD - _PASS_METRIC_EPS
                )
            pass_at_8 = float(successes.any(dim=1).to(torch.float32).mean().item())
            pass_at_1 = float(successes[:, 0].to(torch.float32).mean().item())
            mean_at_1 = float(successes[:, :1].to(torch.float32).mean().item())
            mean_at_8 = float(successes.to(torch.float32).mean().item())
            self._append_metric_value(mode, "pass_at_8", pass_at_8)
            self._append_metric_value(mode, "pass_at_1", pass_at_1)
            self._append_metric_value(mode, "mean_at_1", mean_at_1)
            self._append_metric_value(mode, "mean_at_8", mean_at_8)
            _append_per_benchmark_metrics(successes)

        def _log_eval_greedy_metrics(
            self,
            inputs: List[Dict[str, Any]],
            outputs: Dict[str, Any],
            *,
            mode: str,
        ) -> None:
            """Log a deterministic greedy pass@1 eval beside sampled metrics."""
            if mode != "eval":
                return
            args = getattr(self, "args", None)
            if not bool(getattr(args, "greedy_eval_enabled", False)):
                return
            lightweight_eval = _use_lightweight_greedy_eval(self, mode)
            local_only_eval = _use_local_only_lightweight_eval_metrics(self, mode)
            if local_only_eval and not _is_main_process(self):
                return
            precomputed = bool(outputs.get("_greedy_eval_precomputed", False))
            if precomputed:
                prompt_inputs = outputs.get("_eval_prompt_inputs")
                completion_ids = outputs.get("completion_ids")
                completion_mask = outputs.get("completion_mask")
                if not isinstance(prompt_inputs, list):
                    return
                if not isinstance(completion_ids, torch.Tensor) or not isinstance(
                    completion_mask, torch.Tensor
                ):
                    return
                greedy_outputs = {
                    "completion_ids": completion_ids,
                    "completion_mask": completion_mask,
                }
            else:
                prompt_ids = outputs.get("prompt_ids")
                prompt_mask = outputs.get("prompt_mask")
                if not isinstance(prompt_ids, torch.Tensor) or not isinstance(
                    prompt_mask, torch.Tensor
                ):
                    return
                group_size = max(int(getattr(self, "num_generations", 1) or 1), 1)
                usable = min(
                    len(inputs),
                    int(prompt_ids.size(0)),
                    int(prompt_mask.size(0)),
                )
                usable = usable - (usable % group_size)
                if usable <= 0:
                    return
                prompt_inputs = list(inputs[:usable:group_size])
                prompt_ids = prompt_ids[:usable:group_size].contiguous()
                prompt_mask = prompt_mask[:usable:group_size].contiguous()
                if prompt_ids.numel() <= 0 or not prompt_inputs:
                    return

                unwrap_fn = getattr(self.accelerator, "unwrap_model", None)
                gen_model = self.model
                if callable(unwrap_fn):
                    try:
                        gen_model = unwrap_fn(gen_model)
                    except Exception:
                        gen_model = self.model
                generate_fn = getattr(gen_model, "generate", None)
                if not callable(generate_fn):
                    if not bool(
                        getattr(self, "_greedy_eval_generate_warned", False)
                    ):
                        LOG.warning(
                            "Skipping greedy eval metrics because model.generate is unavailable."
                        )
                        setattr(self, "_greedy_eval_generate_warned", True)
                    return

                tokenizer = getattr(self, "processing_class", None)
                eos_token_id = getattr(tokenizer, "eos_token_id", None)
                pad_token_id = getattr(tokenizer, "pad_token_id", None)
                if pad_token_id is None:
                    pad_token_id = eos_token_id
                max_new_tokens = int(
                    getattr(self, "max_completion_length", 0)
                    or getattr(args, "max_completion_length", 0)
                    or 0
                )
                if max_new_tokens <= 0:
                    if not bool(
                        getattr(self, "_greedy_eval_length_warned", False)
                    ):
                        LOG.warning(
                            "Skipping greedy eval metrics because max_completion_length is invalid."
                        )
                        setattr(self, "_greedy_eval_length_warned", True)
                    return

                was_training = bool(getattr(gen_model, "training", False))
                if was_training:
                    gen_model.eval()
                generate_kwargs: Dict[str, Any] = {
                    "input_ids": prompt_ids,
                    "attention_mask": prompt_mask,
                    "do_sample": False,
                    "max_new_tokens": max_new_tokens,
                    "num_return_sequences": 1,
                }
                if pad_token_id is not None:
                    generate_kwargs["pad_token_id"] = int(pad_token_id)
                if eos_token_id is not None:
                    generate_kwargs["eos_token_id"] = int(eos_token_id)
                if (
                    getattr(self.accelerator, "num_processes", 1) > 1
                    and not lightweight_eval
                ):
                    generate_kwargs["synced_gpus"] = True
                try:
                    with torch.inference_mode():
                        try:
                            generated = generate_fn(**generate_kwargs)
                        except TypeError as exc:
                            if "synced_gpus" not in str(exc):
                                raise
                            generate_kwargs.pop("synced_gpus", None)
                            generated = generate_fn(**generate_kwargs)
                except Exception as exc:
                    if not bool(
                        getattr(self, "_greedy_eval_failed_warned", False)
                    ):
                        LOG.warning(
                            "Skipping greedy eval metrics because greedy generation failed: %s",
                            exc,
                        )
                        setattr(self, "_greedy_eval_failed_warned", True)
                    return
                finally:
                    if was_training:
                        gen_model.train()

                sequences = getattr(generated, "sequences", generated)
                if not isinstance(sequences, torch.Tensor):
                    return
                prompt_width = int(prompt_ids.size(1))
                if int(sequences.size(1)) < prompt_width:
                    return
                completion_ids = sequences[:, prompt_width:].contiguous()
                completion_mask = _apply_eos_completion_mask(
                    completion_ids,
                    eos_token_id,
                )
                greedy_outputs = {
                    "completion_ids": completion_ids,
                    "completion_mask": completion_mask,
                }
                self._maybe_truncate_completions_at_first_boxed_answer(
                    prompt_inputs,
                    greedy_outputs,
                    mode=mode,
                    group_size=1,
                    update_logged_metrics=False,
                )
                completion_ids = greedy_outputs["completion_ids"]
                completion_mask = greedy_outputs["completion_mask"]

            rewards = self._recompute_local_rewards_for_outputs(
                prompt_inputs,
                greedy_outputs,
            )
            if not isinstance(rewards, torch.Tensor) or rewards.numel() <= 0:
                return
            global_rewards = (
                rewards.to(torch.float32)
                if local_only_eval
                else gather(rewards.to(torch.float32))
            )
            if isinstance(global_rewards, torch.Tensor) and global_rewards.numel() > 0:
                self._append_metric_value(
                    mode,
                    "greedy/reward",
                    float(global_rewards.mean().item()),
                )

            reward_funcs = list(getattr(self, "reward_funcs", []) or [])
            successes: Optional[torch.Tensor] = None
            if uses_pure_accuracy_math_reward(reward_funcs):
                tokenizer = getattr(self, "processing_class", None)
                decode = getattr(tokenizer, "batch_decode", None)
                if callable(decode):
                    try:
                        decode_fn = cast(Callable[..., List[str]], decode)
                        completions_text = decode_fn(
                            completion_ids, skip_special_tokens=True
                        )
                    except Exception:
                        completions_text = []
                    answers = [str(example.get("answer", "")) for example in prompt_inputs]
                    usable_local = min(len(completions_text), len(answers))
                    if usable_local > 0:
                        correctness_local = pure_accuracy_math_correctness(
                            completions_text[:usable_local],
                            answers[:usable_local],
                            allow_last_line_fallback=True,
                        )
                        local_successes = torch.tensor(
                            correctness_local,
                            dtype=torch.bool,
                            device=completion_ids.device,
                        )
                        successes = (
                            local_successes
                            if local_only_eval
                            else gather(local_successes)
                        )
            if successes is None:
                local_successes = (
                    rewards >= (_PASS_METRIC_SUCCESS_REWARD - _PASS_METRIC_EPS)
                ).to(torch.bool)
                successes = (
                    local_successes
                    if local_only_eval
                    else gather(local_successes)
                )
            if not isinstance(successes, torch.Tensor) or successes.numel() <= 0:
                return
            successes = successes.to(torch.bool)
            pass_at_1 = float(successes.to(torch.float32).mean().item())
            self._append_metric_value(mode, "greedy/pass_at_1", pass_at_1)
            self._append_metric_value(mode, "greedy/mean_at_1", pass_at_1)
            if bool(getattr(args, "eval_greedy_only_enabled", False)):
                self._append_metric_value(mode, "pass_at_1", pass_at_1)
                self._append_metric_value(mode, "mean_at_1", pass_at_1)

            try:
                completion_lengths = completion_mask.sum(dim=1).to(torch.float32)
                gathered_lengths = (
                    completion_lengths if local_only_eval else gather(completion_lengths)
                )
                if (
                    isinstance(gathered_lengths, torch.Tensor)
                    and gathered_lengths.numel() > 0
                ):
                    self._append_metric_value(
                        mode,
                        "greedy/completions/mean_length",
                        float(gathered_lengths.mean().item()),
                    )
            except Exception:
                pass

            benchmark_ids = _gather_eval_benchmark_ids_for_prompts(
                self,
                prompt_inputs,
                device=completion_ids.device,
                local_only=local_only_eval,
            )
            if not isinstance(benchmark_ids, torch.Tensor):
                return
            if benchmark_ids.numel() != successes.numel():
                return
            id_to_name = getattr(self, "eval_benchmark_id_to_name", {}) or {}
            for bench_id_tensor in torch.unique(benchmark_ids):
                bench_id = int(bench_id_tensor.item())
                if bench_id < 0:
                    continue
                mask = benchmark_ids == bench_id_tensor
                bench_count = int(mask.to(torch.long).sum().item())
                if bench_count <= 0:
                    continue
                bench_pass_at_1 = float(
                    successes[mask].to(torch.float32).mean().item()
                )
                bench_label = id_to_name.get(bench_id, f"BENCH_{bench_id}")
                suffix = _metric_suffix_from_benchmark(bench_label)
                self._append_metric_value(
                    mode,
                    f"greedy/pass_at_1_{suffix}",
                    bench_pass_at_1,
                )
                if bool(getattr(args, "eval_greedy_only_enabled", False)):
                    self._append_metric_value(
                        mode,
                        f"pass_at_1_{suffix}",
                        bench_pass_at_1,
                    )
                    self._append_metric_value(
                        mode,
                        f"mean_at_1_{suffix}",
                        bench_pass_at_1,
                    )

        def _get_per_token_logps_and_entropy(
            self,
            model: Any,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            logits_to_keep: int,
            *,
            entropy_mode: str,
            batch_size: Optional[int] = None,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """Compute selected-token log-probs and entropy on the completion span."""
            chunk_size = int(batch_size or input_ids.size(0) or 1)
            all_logps: List[torch.Tensor] = []
            all_entropy: List[torch.Tensor] = []
            mode = "train" if bool(getattr(model, "training", False)) else "eval"
            unwrap_fn = getattr(getattr(self, "accelerator", None), "unwrap_model", None)
            base_model = model
            if callable(unwrap_fn):
                try:
                    base_model = unwrap_fn(model)
                except Exception:
                    base_model = model
            tokenizer = getattr(self, "processing_class", None)
            vocab_size = _resolve_token_id_upper_bound(base_model, tokenizer)
            for start in range(0, int(input_ids.size(0)), chunk_size):
                stop = start + chunk_size
                input_ids_batch = input_ids[start:stop]
                attention_mask_batch = attention_mask[start:stop]
                input_ids_batch = self._sanitize_scoring_token_ids(
                    input_ids_batch,
                    upper_bound=vocab_size,
                    mode=mode,
                    context="model_input",
                )
                logits = model(
                    input_ids=input_ids_batch,
                    attention_mask=attention_mask_batch,
                    logits_to_keep=logits_to_keep + 1,
                ).logits
                logits = logits[:, :-1, :]
                token_ids = input_ids_batch[:, -logits_to_keep:]
                logits = logits[:, -logits_to_keep:]
                logits = logits / self.temperature
                if isinstance(vocab_size, int) and int(logits.size(-1)) > vocab_size:
                    if not bool(
                        getattr(self, "_invalid_logit_columns_warned_entropy", False)
                    ):
                        LOG.warning(
                            "Masking %d tokenizer-inaccessible logit columns in exact-entropy scoring (valid_vocab_size=%d, logits_width=%d).",
                            int(logits.size(-1)) - vocab_size,
                            vocab_size,
                            int(logits.size(-1)),
                        )
                        setattr(self, "_invalid_logit_columns_warned_entropy", True)
                    logits = _mask_invalid_logit_columns(
                        logits,
                        valid_vocab_size=vocab_size,
                    )
                token_ids = self._sanitize_scoring_token_ids(
                    token_ids,
                    upper_bound=int(logits.size(-1)),
                    mode=mode,
                    context="token_select",
                )
                logps, entropy = _selected_logps_and_entropy(
                    logits,
                    token_ids,
                    entropy_mode=entropy_mode,
                )
                all_logps.append(logps)
                all_entropy.append(entropy)
            return torch.cat(all_logps, dim=0), torch.cat(all_entropy, dim=0)

        def _recompute_local_rewards_for_outputs(
            self,
            inputs: List[Dict[str, Any]],
            outputs: Dict[str, Any],
        ) -> Optional[torch.Tensor]:
            if not inputs:
                return None
            completion_ids = outputs.get("completion_ids")
            if not isinstance(completion_ids, torch.Tensor):
                return None
            completion_mask = outputs.get("completion_mask")
            if not isinstance(completion_mask, torch.Tensor):
                completion_mask = _apply_eos_completion_mask(
                    completion_ids,
                    getattr(self.processing_class, "eos_token_id", None),
                )
            if not isinstance(completion_mask, torch.Tensor):
                return None
            completion_mask = completion_mask.to(
                device=completion_ids.device, dtype=torch.long
            )

            prompts = [example["prompt"] for example in inputs]
            completions_text = self.processing_class.batch_decode(
                completion_ids, skip_special_tokens=True
            )
            if is_conversational(inputs[0]):
                completions: List[Any] = []
                for prompt, completion in zip(prompts, completions_text):
                    bootstrap = ""
                    if (
                        isinstance(prompt, list)
                        and prompt
                        and isinstance(prompt[-1], dict)
                        and prompt[-1].get("role") == "assistant"
                    ):
                        bootstrap = str(prompt[-1].get("content", ""))
                    completions.append(
                        [{"role": "assistant", "content": f"{bootstrap}{completion}"}]
                    )
            else:
                completions = completions_text

            completion_ids_list = [
                [
                    int(tok.item())
                    for tok, keep in zip(row, mask_row)
                    if int(keep.item()) != 0
                ]
                for row, mask_row in zip(completion_ids, completion_mask)
            ]
            rewards_per_func_local = torch.zeros(
                (len(prompts), len(self.reward_funcs)),
                device=completion_ids.device,
                dtype=torch.float32,
            )

            keys = [
                key
                for key in inputs[0]
                if key not in {"prompt", "completion", "completion_ids"}
            ]

            def _reward_value_for_key(example: Dict[str, Any], key: str) -> Any:
                if key in example:
                    return example[key]
                if key == "answer":
                    return example.get("solution")
                if key == "solution":
                    return example.get("answer")
                return None

            reward_kwargs = {
                key: [_reward_value_for_key(example, key) for example in inputs]
                for key in keys
            }
            if "answer" not in reward_kwargs and "solution" in reward_kwargs:
                reward_kwargs["answer"] = list(reward_kwargs["solution"])
            if "solution" not in reward_kwargs and "answer" in reward_kwargs:
                reward_kwargs["solution"] = list(reward_kwargs["answer"])
            reward_processing_classes = list(
                getattr(
                    self, "reward_processing_classes", [None] * len(self.reward_funcs)
                )
            )

            for i, reward_func in enumerate(self.reward_funcs):
                if isinstance(reward_func, torch.nn.Module):
                    reward_processing_class = (
                        reward_processing_classes[i]
                        if i < len(reward_processing_classes)
                        else None
                    )
                    if reward_processing_class is None:
                        return None
                    if is_conversational(inputs[0]):
                        messages = [
                            {"messages": p + c} for p, c in zip(prompts, completions)
                        ]
                        texts = [
                            apply_chat_template(x, reward_processing_class)["text"]
                            for x in messages
                        ]
                    else:
                        texts = [p + c for p, c in zip(prompts, completions)]
                    reward_inputs = reward_processing_class(
                        text=texts,
                        return_tensors="pt",
                        padding=True,
                        padding_side="right",
                        add_special_tokens=False,
                    )
                    reward_inputs = super()._prepare_inputs(reward_inputs)
                    with torch.inference_mode():
                        rewards_per_func_local[:, i] = reward_func(
                            **reward_inputs
                        ).logits[:, 0]
                else:
                    if not callable(reward_func):
                        return None
                    output_reward_func = reward_func(
                        prompts=prompts,
                        completions=completions,
                        completion_ids=completion_ids_list,
                        **reward_kwargs,
                    )
                    output_reward_func = [
                        reward if reward is not None else torch.nan
                        for reward in output_reward_func
                    ]
                    rewards_per_func_local[:, i] = torch.tensor(
                        output_reward_func,
                        dtype=torch.float32,
                        device=completion_ids.device,
                    )

            reward_weights = getattr(self, "reward_weights", None)
            if isinstance(reward_weights, torch.Tensor):
                weights = reward_weights.to(
                    device=rewards_per_func_local.device, dtype=torch.float32
                )
            elif isinstance(reward_weights, (list, tuple)):
                weights = torch.tensor(
                    list(reward_weights),
                    dtype=torch.float32,
                    device=rewards_per_func_local.device,
                )
            else:
                weights = torch.ones(
                    (len(self.reward_funcs),),
                    dtype=torch.float32,
                    device=rewards_per_func_local.device,
                )
            if weights.numel() != rewards_per_func_local.size(1):
                weights = torch.ones(
                    (rewards_per_func_local.size(1),),
                    dtype=torch.float32,
                    device=rewards_per_func_local.device,
                )
            rewards = (rewards_per_func_local * weights.unsqueeze(0)).nansum(dim=1)
            return torch.nan_to_num(rewards, nan=0.0, posinf=0.0, neginf=0.0)

        def _recompute_global_rewards_for_outputs(
            self,
            inputs: List[Dict[str, Any]],
            outputs: Dict[str, Any],
        ) -> Optional[torch.Tensor]:
            rewards_local = self._recompute_local_rewards_for_outputs(inputs, outputs)
            if not isinstance(rewards_local, torch.Tensor):
                return None
            rewards = gather(rewards_local)
            return torch.nan_to_num(rewards, nan=0.0, posinf=0.0, neginf=0.0)

        def _prepare_listwise_rollout_targets(
            self,
            inputs: List[Dict[str, Any]],
            outputs: Dict[str, Any],
        ) -> None:
            """Cache per-prompt listwise q targets on rollout outputs."""
            rewards = self._recompute_local_rewards_for_outputs(inputs, outputs)
            if not isinstance(rewards, torch.Tensor):
                return
            if rewards.numel() <= 0:
                return
            group_size = max(int(getattr(self, "num_generations", 1) or 1), 1)
            gathered_rewards = gather(rewards)
            if not isinstance(gathered_rewards, torch.Tensor):
                gathered_rewards = torch.as_tensor(
                    gathered_rewards,
                    device=rewards.device,
                    dtype=rewards.dtype,
                )
            grouped_rewards = _reshape_prompt_major_tensor(gathered_rewards, group_size)
            if grouped_rewards is None:
                raise ValueError(
                    "Listwise MaxEnt rollout rewards must arrive as whole prompt "
                    f"groups with flat batch size divisible by num_generations={group_size}."
                )
            temperature = _coerce_non_negative_float(
                getattr(getattr(self, "args", None), "maxent_q_temperature", 1.0),
                default=1.0,
            )
            epsilon = _coerce_non_negative_float(
                getattr(getattr(self, "args", None), "maxent_q_epsilon", 1e-6),
                default=1e-6,
            )
            q_grouped = torch.softmax(grouped_rewards / max(temperature, 1e-8), dim=1)
            if epsilon > 0.0:
                max_eps = max((1.0 / float(max(q_grouped.size(1), 1))) - 1e-8, 0.0)
                epsilon = min(epsilon, max_eps)
                if epsilon > 0.0:
                    q_grouped = q_grouped * (1.0 - epsilon * q_grouped.size(1)) + epsilon
                    q_grouped = q_grouped / q_grouped.sum(dim=1, keepdim=True).clamp(
                        min=1e-12
                    )
            local_count = int(rewards.size(0))
            process_index = int(getattr(self.accelerator, "process_index", 0) or 0)
            process_start = process_index * local_count
            process_stop = process_start + local_count
            total_count = int(gathered_rewards.size(0))
            if process_stop > total_count:
                raise ValueError(
                    "Listwise MaxEnt gathered reward totals are shorter than the "
                    "current rank slice."
                )
            if process_start % group_size != 0:
                raise ValueError(
                    "Listwise MaxEnt requires each rank slice to begin on a whole "
                    "prompt-group boundary after reward gathering."
                )
            local_rewards = gathered_rewards[process_start:process_stop]
            local_q = q_grouped.reshape(-1)[process_start:process_stop]
            local_grouped_rewards = _reshape_prompt_major_tensor(local_rewards, group_size)
            local_q_grouped = _reshape_prompt_major_tensor(local_q, group_size)
            if local_grouped_rewards is None or local_q_grouped is None:
                raise ValueError(
                    "Listwise MaxEnt local rollout slice must contain whole prompt "
                    "groups after reward gathering."
                )
            outputs["maxent_listwise_q"] = _normalize_listwise_q_targets(
                local_q_grouped.detach(),
                num_prompts=int(local_grouped_rewards.size(0)),
                group_size=group_size,
                context="Listwise MaxEnt rollout targets",
            )
            outputs["maxent_listwise_rewards"] = local_grouped_rewards.detach()

        def _resolve_listwise_reference_mode(self) -> bool:
            """Return whether listwise MaxEnt should include a reference term."""
            return self._should_use_model_reference_logprobs(
                default_to_model_reference=False
            )

        def _compute_listwise_maxent_loss(self, model: Any, inputs: Any) -> torch.Tensor:
            """Match the sampled candidate distribution to the tau/q/beta posterior."""
            if bool(getattr(self, "use_liger_loss", False)):
                raise NotImplementedError(
                    "Listwise MaxEnt loss is not implemented for liger loss."
                )

            q_grouped = inputs.get("maxent_listwise_q")
            if not isinstance(q_grouped, torch.Tensor) or q_grouped.numel() <= 0:
                raise ValueError(
                    "Listwise MaxEnt requires rollout q targets from _generate_and_score_completions."
                )

            prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
            completion_ids, completion_mask = (
                inputs["completion_ids"],
                inputs["completion_mask"],
            )
            input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
            attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
            logits_to_keep = completion_ids.size(1)
            mode = "train" if self.model.training else "eval"

            configured_batch_size = (
                int(getattr(self.args, "per_device_train_batch_size", 1) or 1)
                if self.model.training
                else int(getattr(self.args, "per_device_eval_batch_size", 1) or 1)
            )
            chunk_size = int(
                getattr(self.args, "maxent_logprob_chunk_size", 0)
                or configured_batch_size
                or 1
            )
            per_token_logps = self._get_per_token_logps(
                model,
                input_ids,
                attention_mask,
                logits_to_keep,
                batch_size=chunk_size,
            )
            seq_logps = (per_token_logps * completion_mask).sum(dim=1)
            group_size = max(int(getattr(self, "num_generations", 1) or 1), 1)
            seq_logps_grouped = _reshape_prompt_major_tensor(seq_logps, group_size)
            token_counts_grouped = _reshape_prompt_major_tensor(
                completion_mask.sum(dim=1).to(torch.float32),
                group_size,
            )
            if seq_logps_grouped is None or token_counts_grouped is None:
                raise ValueError("Listwise MaxEnt could not reshape sequence log-probs.")

            old_per_token_logps = (
                per_token_logps.detach()
                if inputs["old_per_token_logps"] is None
                else inputs["old_per_token_logps"].to(
                    device=per_token_logps.device,
                    dtype=per_token_logps.dtype,
                )
            )
            old_seq_logps_grouped = _reshape_prompt_major_tensor(
                (old_per_token_logps * completion_mask).sum(dim=1),
                group_size,
            )
            if old_seq_logps_grouped is None:
                raise ValueError("Listwise MaxEnt could not reshape behavior log-probs.")

            num_prompts = int(seq_logps_grouped.size(0))
            if int(token_counts_grouped.size(0)) != num_prompts:
                raise ValueError("Listwise MaxEnt token counts are misaligned with prompts.")
            if int(old_seq_logps_grouped.size(0)) != num_prompts:
                raise ValueError("Listwise MaxEnt behavior log-probs are misaligned with prompts.")
            policy_seq_logps_grouped = seq_logps_grouped
            behavior_seq_logps_grouped = old_seq_logps_grouped
            if bool(getattr(self.args, "maxent_length_normalize_policy", False)):
                token_denoms = token_counts_grouped.to(seq_logps_grouped.dtype).clamp(
                    min=1.0
                )
                policy_seq_logps_grouped = seq_logps_grouped / token_denoms
                behavior_seq_logps_grouped = old_seq_logps_grouped / token_denoms
            q_grouped = _normalize_listwise_q_targets(
                q_grouped.to(
                    device=seq_logps_grouped.device,
                    dtype=seq_logps_grouped.dtype,
                ),
                num_prompts=num_prompts,
                group_size=group_size,
                context="Listwise MaxEnt loss",
            )
            skip_zero_variance_groups = bool(
                getattr(self.args, "maxent_listwise_skip_zero_variance_groups", False)
            )
            neutral_group_mask = (
                (
                    q_grouped.to(torch.float32).amax(dim=1)
                    - q_grouped.to(torch.float32).amin(dim=1)
                )
                <= 1e-8
            ) if skip_zero_variance_groups else torch.zeros(
                num_prompts,
                device=q_grouped.device,
                dtype=torch.bool,
            )
            active_group_mask = ~neutral_group_mask
            active_group_count = int(active_group_mask.to(torch.int64).sum().item())

            weighting = getattr(self, "_maxent_weighting", None)
            if weighting is None:
                raise ValueError("Listwise MaxEnt requires initialized weighting settings.")

            include_reference_term = self._resolve_listwise_reference_mode()
            ref_seq_logps_grouped = torch.zeros_like(seq_logps_grouped)
            measured_kl: Optional[torch.Tensor] = None
            if include_reference_term:
                with torch.no_grad():
                    ref_per_token_logps = self._get_reference_per_token_logps(
                        input_ids,
                        attention_mask,
                        logits_to_keep,
                        batch_size=chunk_size,
                    )
                if ref_per_token_logps is not None:
                    ref_per_token_logps = ref_per_token_logps.to(
                        device=per_token_logps.device,
                        dtype=per_token_logps.dtype,
                    )
                    # Guard the exponentials against rare runaway log-prob deltas.
                    kl_delta = _clamp_log_delta(ref_per_token_logps - per_token_logps)
                    per_token_kl = (
                        torch.exp(kl_delta)
                        - kl_delta
                        - 1
                    ).to(per_token_logps.dtype)
                    measured_kl = (per_token_kl * completion_mask).sum() / completion_mask.sum().clamp(
                        min=1.0
                    )
                    ref_seq_logps = (ref_per_token_logps * completion_mask).sum(dim=1)
                    if bool(getattr(weighting, "len_norm_ref", True)):
                        ref_seq_logps = ref_seq_logps / completion_mask.sum(dim=1).clamp(
                            min=1.0
                        )
                    ref_seq_logps_grouped = _reshape_prompt_major_tensor(
                        ref_seq_logps,
                        int(q_grouped.size(1)),
                    )
                    if ref_seq_logps_grouped is None:
                        raise ValueError(
                            "Listwise MaxEnt could not reshape reference log-probs."
                        )
                    if int(ref_seq_logps_grouped.size(0)) != num_prompts:
                        raise ValueError(
                            "Listwise MaxEnt reference log-probs are misaligned with prompts."
                        )
                else:
                    if not bool(getattr(self, "_maxent_listwise_ref_warned", False)):
                        LOG.warning(
                            "Listwise MaxEnt requested reference weighting but no model-based "
                            "reference path is available; using rollout behavior log-probs "
                            "as the reference term."
                        )
                        setattr(self, "_maxent_listwise_ref_warned", True)
                    # Reuse the rollout behavior log-probs as a fixed reference term when
                    # no separate frozen model is available. This preserves listwise
                    # weighting signal on full-model runs without allocating another copy.
                    ref_seq_logps_grouped = behavior_seq_logps_grouped.detach()
                    kl_delta = _clamp_log_delta(old_per_token_logps - per_token_logps)
                    per_token_kl = (
                        torch.exp(kl_delta)
                        - kl_delta
                        - 1
                    ).to(per_token_logps.dtype)
                    measured_kl = (per_token_kl * completion_mask).sum() / completion_mask.sum().clamp(
                        min=1.0
                    )

            weights_grouped = weight_matrix_from_q(
                q_grouped,
                ref_seq_logps_grouped,
                token_counts_grouped,
                weighting,
                include_reference_term=include_reference_term,
                normalize_by_tokens=False,
            ).to(
                device=seq_logps_grouped.device,
                dtype=seq_logps_grouped.dtype,
            )
            if bool(neutral_group_mask.any().item()):
                uniform_weights = torch.full_like(
                    weights_grouped,
                    1.0 / float(max(weights_grouped.size(1), 1)),
                )
                weights_grouped = torch.where(
                    neutral_group_mask.unsqueeze(1),
                    uniform_weights,
                    weights_grouped,
                )
            weights_grouped_list = weights_grouped.detach().cpu().tolist()
            log_probs_grouped = torch.log_softmax(policy_seq_logps_grouped, dim=1)
            per_group_policy_loss = -(weights_grouped * log_probs_grouped).sum(dim=1)
            if active_group_count > 0:
                policy_loss = per_group_policy_loss[active_group_mask].mean()
            else:
                policy_loss = (per_group_policy_loss * 0.0).sum()
            loss = policy_loss

            clip_loss: Optional[torch.Tensor] = None
            if bool(getattr(self.args, "maxent_use_clip_objective", False)):
                clip_coef = _coerce_non_negative_float(
                    getattr(self.args, "maxent_clip_objective_coef", 1.0),
                    default=1.0,
                )
                if clip_coef > 0.0:
                    clip_range = getattr(self.args, "maxent_clip_range", None)
                    clip_low = (
                        _coerce_non_negative_float(clip_range, default=self.epsilon_low)
                        if clip_range is not None
                        else float(self.epsilon_low)
                    )
                    clip_high = (
                        _coerce_non_negative_float(clip_range, default=self.epsilon_high)
                        if clip_range is not None
                        else float(self.epsilon_high)
                    )
                    baseline = getattr(self.args, "maxent_clip_adv_baseline", None)
                    if baseline is None:
                        baseline_value = 1.0 / float(max(weights_grouped.size(1), 1))
                    else:
                        baseline_value = float(baseline)
                    clip_adv = weights_grouped - baseline_value
                    log_seq_ratio = _clamp_log_delta(
                        policy_seq_logps_grouped - behavior_seq_logps_grouped
                    )
                    seq_ratio = torch.exp(log_seq_ratio).to(seq_logps_grouped.dtype)
                    seq_ratio_clipped = torch.clamp(
                        seq_ratio,
                        1.0 - clip_low,
                        1.0 + clip_high,
                    )
                    clip_obj = torch.min(
                        seq_ratio * clip_adv,
                        seq_ratio_clipped * clip_adv,
                    )
                    per_group_clip_loss = -clip_obj.sum(dim=1)
                    if active_group_count > 0:
                        clip_loss = per_group_clip_loss[active_group_mask].mean()
                    else:
                        clip_loss = (per_group_clip_loss * 0.0).sum()
                    loss = loss + clip_coef * clip_loss

                    is_low_clipped = (seq_ratio < 1.0 - clip_low) & (clip_adv < 0.0)
                    is_high_clipped = (seq_ratio > 1.0 + clip_high) & (clip_adv > 0.0)
                    clip_region = is_low_clipped | is_high_clipped
                    self._append_metric_value(
                        mode,
                        "clip_ratio/low_mean",
                        is_low_clipped.to(torch.float32).mean().item(),
                    )
                    self._append_metric_value(
                        mode,
                        "clip_ratio/high_mean",
                        is_high_clipped.to(torch.float32).mean().item(),
                    )
                    self._append_metric_value(
                        mode,
                        "clip_ratio/region_mean",
                        clip_region.to(torch.float32).mean().item(),
                    )

            weight_entropy, entropy_min, entropy_max, _ = collect_weight_entropy(
                weights_grouped_list
            )
            loss_scale_raw = inputs.get("maxent_listwise_loss_scale")
            loss_scale_value = 1.0
            if isinstance(loss_scale_raw, torch.Tensor):
                if loss_scale_raw.numel() != 1:
                    raise ValueError("Listwise MaxEnt loss scale must be scalar.")
                loss_scale_value = float(loss_scale_raw.detach().cpu().item())
            elif isinstance(loss_scale_raw, (int, float)):
                loss_scale_value = float(loss_scale_raw)
            if not math.isfinite(loss_scale_value) or loss_scale_value <= 0.0:
                raise ValueError("Listwise MaxEnt loss scale must be finite and positive.")
            if loss_scale_value != 1.0:
                loss = loss * loss.new_tensor(loss_scale_value)
            self._append_metric_value(mode, "loss/policy", float(policy_loss.item()))
            self._append_metric_value(
                mode, "weight_entropy", float(weight_entropy), include_legacy_aliases=False
            )
            self._append_metric_value(
                mode, "weight_entropy_min", float(entropy_min), include_legacy_aliases=False
            )
            self._append_metric_value(
                mode, "weight_entropy_max", float(entropy_max), include_legacy_aliases=False
            )
            self._append_metric_value(mode, "maxent/objective_variant_listwise", 1.0)
            self._append_metric_value(mode, "maxent/objective_variant_entropy", 0.0)
            self._append_metric_value(
                mode,
                "maxent/listwise_weight_mean",
                float(weights_grouped.mean().item()),
            )
            self._append_metric_value(
                mode,
                "maxent/listwise_weight_std",
                float(weights_grouped.to(torch.float32).std(unbiased=False).item()),
            )
            self._append_metric_value(
                mode,
                "maxent/listwise_neutral_group_frac",
                float(neutral_group_mask.to(torch.float32).mean().item()),
            )
            self._append_metric_value(
                mode,
                "maxent/listwise_active_group_frac",
                float(active_group_mask.to(torch.float32).mean().item()),
            )
            self._append_metric_value(
                mode,
                "maxent/listwise_loss_scale",
                loss_scale_value,
            )
            if clip_loss is not None:
                self._append_metric_value(mode, "loss/clip", float(clip_loss.item()))
            if measured_kl is not None:
                gathered_kl = _metric_tensor_for_logging(self, measured_kl, mode=mode)
                if isinstance(gathered_kl, torch.Tensor) and gathered_kl.numel() > 0:
                    kl_value = float(gathered_kl.nanmean().item())
                    self._append_metric_value(mode, "kl", kl_value)
                else:
                    kl_value = None
            else:
                kl_value = None

            if mode == "train":
                meta_objective = getattr(self, "_maxent_controller_objective", None)
                beta_controller_enabled = bool(
                    getattr(self.args, "maxent_beta_controller_enabled", False)
                )
                if meta_objective is not None:
                    self._maybe_apply_controller_meta(
                        mode=mode,
                        kl_value=kl_value,
                        weight_entropy=weight_entropy,
                        total_loss=float(loss.item()),
                    )
                else:
                    maybe_update_tau(
                        weighting,
                        SimpleNamespace(weight_entropy=weight_entropy),
                        global_step=int(getattr(self.state, "global_step", 0) or 0),
                    )
                    if (
                        beta_controller_enabled
                        and include_reference_term
                        and kl_value is not None
                    ):
                        maybe_update_beta(weighting, measured_kl=kl_value)
                self._sync_weighting_scalars()
                self._append_metric_value(
                    mode,
                    "kl_controller/enabled",
                    1.0 if beta_controller_enabled else 0.0,
                    include_legacy_aliases=False,
                )
                self._append_metric_value(
                    mode,
                    "kl_controller_enabled",
                    1.0 if beta_controller_enabled else 0.0,
                    include_legacy_aliases=False,
                )
                self._append_metric_value(mode, "tau", float(self.tau))
                self._append_metric_value(mode, "beta", float(self.beta))
                self._append_metric_value(
                    mode,
                    "weight_norm_denom",
                    float(getattr(weighting, "denom", 1.0)),
                    include_legacy_aliases=False,
                )

            return loss

        def _sanitize_scoring_token_ids(
            self,
            token_ids: torch.Tensor,
            *,
            upper_bound: Optional[int],
            mode: str,
            context: str,
        ) -> torch.Tensor:
            """Clamp scorer token ids into range before model/gather indexing."""

            if not isinstance(token_ids, torch.Tensor):
                return token_ids
            if token_ids.dtype.is_floating_point or token_ids.dtype == torch.bool:
                return token_ids
            if not isinstance(upper_bound, int) or upper_bound <= 0:
                return token_ids

            tokenizer = getattr(self, "processing_class", None)
            replacement_id = _coerce_optional_int(getattr(tokenizer, "pad_token_id", None))
            if (
                replacement_id is None
                or replacement_id < 0
                or replacement_id >= upper_bound
            ):
                replacement_id = _coerce_optional_int(
                    getattr(tokenizer, "eos_token_id", None)
                )
            if (
                replacement_id is None
                or replacement_id < 0
                or replacement_id >= upper_bound
            ):
                replacement_id = max(upper_bound - 1, 0)

            invalid_mask = (token_ids < 0) | (token_ids >= upper_bound)
            invalid_count = int(invalid_mask.to(torch.long).sum().item())
            if invalid_count <= 0:
                return token_ids

            try:
                invalid_vals = token_ids[invalid_mask]
                min_invalid = int(invalid_vals.min().item())
                max_invalid = int(invalid_vals.max().item())
            except Exception:
                min_invalid = 0
                max_invalid = 0

            sanitized = token_ids.clone()
            sanitized[invalid_mask] = int(replacement_id)
            self._append_metric_value(
                mode,
                "scoring/invalid_token_id_count",
                float(invalid_count),
                include_legacy_aliases=False,
            )
            self._append_metric_value(
                mode,
                f"scoring/{context}_invalid_token_id_count",
                float(invalid_count),
                include_legacy_aliases=False,
            )
            self._append_metric_value(
                mode,
                "scoring/invalid_token_id_replacement",
                float(replacement_id),
                include_legacy_aliases=False,
            )
            warned_contexts = getattr(self, "_invalid_scoring_token_ids_warned_contexts", None)
            if not isinstance(warned_contexts, set):
                warned_contexts = set()
                setattr(self, "_invalid_scoring_token_ids_warned_contexts", warned_contexts)
            if context not in warned_contexts:
                LOG.warning(
                    "Sanitized %d scoring token ids for %s outside upper_bound=%d using replacement_id=%d (min=%d max=%d)",
                    invalid_count,
                    context,
                    upper_bound,
                    replacement_id,
                    min_invalid,
                    max_invalid,
                )
                warned_contexts.add(context)
            return sanitized

        def _get_per_token_logps(self, *args: Any, **kwargs: Any) -> torch.Tensor:  # type: ignore[override]
            model = args[0] if len(args) >= 1 else kwargs.get("model")
            input_ids = args[1] if len(args) >= 2 else kwargs.get("input_ids")
            attention_mask = args[2] if len(args) >= 3 else kwargs.get("attention_mask")
            logits_to_keep = args[3] if len(args) >= 4 else kwargs.get("logits_to_keep")
            batch_size = args[4] if len(args) >= 5 else kwargs.get("batch_size")
            if not isinstance(input_ids, torch.Tensor) or not isinstance(attention_mask, torch.Tensor):
                logps = super()._get_per_token_logps(*args, **kwargs)
            else:
                chunk_size = int(batch_size or input_ids.size(0) or 1)
                mode = "train" if bool(getattr(model, "training", False)) else "eval"
                unwrap_fn = getattr(getattr(self, "accelerator", None), "unwrap_model", None)
                base_model = model
                if callable(unwrap_fn):
                    try:
                        base_model = unwrap_fn(model)
                    except Exception:
                        base_model = model
                tokenizer = getattr(self, "processing_class", None)
                vocab_size = _resolve_token_id_upper_bound(base_model, tokenizer)
                all_logps: List[torch.Tensor] = []
                for start in range(0, int(input_ids.size(0)), chunk_size):
                    stop = start + chunk_size
                    input_ids_batch = input_ids[start:stop]
                    attention_mask_batch = attention_mask[start:stop]
                    input_ids_batch = self._sanitize_scoring_token_ids(
                        input_ids_batch,
                        upper_bound=vocab_size,
                        mode=mode,
                        context="model_input",
                    )
                    try:
                        outputs = model(
                            input_ids=input_ids_batch,
                            attention_mask=attention_mask_batch,
                            logits_to_keep=int(logits_to_keep) + 1,
                        )
                    except TypeError:
                        outputs = model(
                            input_ids=input_ids_batch,
                            attention_mask=attention_mask_batch,
                        )
                    logits = getattr(outputs, "logits", outputs)
                    logits = logits[:, :-1, :]
                    token_ids = input_ids_batch[:, -int(logits_to_keep) :]
                    logits = logits[:, -int(logits_to_keep) :]
                    logits = logits / self.temperature
                    if isinstance(vocab_size, int) and int(logits.size(-1)) > vocab_size:
                        if not bool(
                            getattr(self, "_invalid_logit_columns_warned_logps", False)
                        ):
                            LOG.warning(
                                "Masking %d tokenizer-inaccessible logit columns in shared logprob scoring (valid_vocab_size=%d, logits_width=%d).",
                                int(logits.size(-1)) - vocab_size,
                                vocab_size,
                                int(logits.size(-1)),
                            )
                            setattr(self, "_invalid_logit_columns_warned_logps", True)
                        logits = _mask_invalid_logit_columns(
                            logits,
                            valid_vocab_size=vocab_size,
                        )
                    token_ids = self._sanitize_scoring_token_ids(
                        token_ids,
                        upper_bound=int(logits.size(-1)),
                        mode=mode,
                        context="token_select",
                    )
                    log_probs = F.log_softmax(logits, dim=-1)
                    chunk_logps = torch.gather(
                        log_probs,
                        dim=-1,
                        index=token_ids.unsqueeze(-1),
                    ).squeeze(-1)
                    all_logps.append(chunk_logps)
                logps = torch.cat(all_logps, dim=0)
            if self.maxent_enabled:
                return logps
            if self.accelerator.is_main_process:
                step = int(getattr(self.state, "global_step", 0))
                try:
                    requires_grad = bool(getattr(logps, "requires_grad", False))
                except Exception:
                    requires_grad = False
                LOG.info(
                    "GRPO debug | step=%d | token_logp_requires_grad=%s | grad_enabled=%s | logps_shape=%s",
                    step,
                    requires_grad,
                    torch.is_grad_enabled(),
                    getattr(logps, "shape", None),
                )
            return logps

        def _compute_maxent_loss(self, model: Any, inputs: Any) -> torch.Tensor:
            """TRL-style GRPO loss with a true entropy regularizer in the loss."""
            if bool(getattr(self, "use_liger_loss", False)):
                raise NotImplementedError(
                    "MaxEnt loss regularization is not implemented for liger loss."
                )

            prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
            completion_ids, completion_mask = (
                inputs["completion_ids"],
                inputs["completion_mask"],
            )
            input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
            attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
            logits_to_keep = completion_ids.size(1)
            mode = "train" if self.model.training else "eval"

            configured_batch_size = (
                int(getattr(self.args, "per_device_train_batch_size", 1) or 1)
                if self.model.training
                else int(getattr(self.args, "per_device_eval_batch_size", 1) or 1)
            )
            chunk_size = int(
                getattr(self.args, "maxent_logprob_chunk_size", 0)
                or configured_batch_size
                or 1
            )
            requested_entropy_mode = str(
                getattr(self.args, "maxent_policy_entropy_mode", "exact") or "exact"
            )
            entropy_mode = requested_entropy_mode.strip().lower() or "exact"
            args = getattr(self, "args", None)
            if entropy_mode != "exact":
                if not bool(
                    getattr(self, "_maxent_sample_entropy_loss_warned", False)
                ):
                    LOG.warning(
                        "Entropy-regularized MaxEnt requested maxent_policy_entropy_mode=%s, "
                        "but the training loss uses exact entropy. The sample estimator is "
                        "only valid for logging or GRPO reward bonuses, not direct "
                        "entropy-loss gradients.",
                        requested_entropy_mode,
                    )
                    setattr(self, "_maxent_sample_entropy_loss_warned", True)
                entropy_mode = "exact"

            unwrap_fn = getattr(getattr(self, "accelerator", None), "unwrap_model", None)
            base_model = model
            if callable(unwrap_fn):
                try:
                    base_model = unwrap_fn(model)
                except Exception:
                    base_model = model
            tokenizer = getattr(self, "processing_class", None)
            valid_vocab_size = _resolve_token_id_upper_bound(base_model, tokenizer)
            entropy_normalization_scale = _entropy_normalization_scale(valid_vocab_size)

            per_token_logps, per_token_entropy = self._get_per_token_logps_and_entropy(
                model,
                input_ids,
                attention_mask,
                logits_to_keep,
                entropy_mode=entropy_mode,
                batch_size=chunk_size,
            )

            per_token_kl: Optional[torch.Tensor] = None
            kl_value: Optional[float] = None
            alpha_kl_control_requested = self._entropy_alpha_kl_control_requested()
            if self.beta != 0.0 or alpha_kl_control_requested:
                use_model_reference = self._should_use_model_reference_logprobs(
                    default_to_model_reference=alpha_kl_control_requested
                )
                with torch.no_grad():
                    if use_model_reference:
                        ref_per_token_logps = self._get_reference_per_token_logps(
                            input_ids,
                            attention_mask,
                            logits_to_keep,
                            batch_size=chunk_size,
                        )
                    else:
                        ref_per_token_logps = None
                    if ref_per_token_logps is not None:
                        ref_per_token_logps = ref_per_token_logps.to(
                            device=per_token_logps.device,
                            dtype=per_token_logps.dtype,
                        )
                    else:
                        old_ref = inputs.get("old_per_token_logps")
                        if isinstance(old_ref, torch.Tensor):
                            ref_per_token_logps = old_ref.to(
                                device=per_token_logps.device,
                                dtype=per_token_logps.dtype,
                            )
                        else:
                            ref_per_token_logps = per_token_logps.detach()
                # Guard the exponentials against rare runaway log-prob deltas.
                kl_delta = _clamp_log_delta(ref_per_token_logps - per_token_logps)
                per_token_kl = (
                    torch.exp(kl_delta)
                    - kl_delta
                    - 1
                ).to(per_token_logps.dtype)
            current_batch_kl_measure: Optional[float] = None
            if mode == "train" and per_token_kl is not None:
                mean_kl_for_alpha = (
                    (per_token_kl.detach() * completion_mask).sum()
                    / completion_mask.sum().clamp(min=1.0)
                ).to(torch.float32)
                gathered_kl_for_alpha = self.accelerator.gather(mean_kl_for_alpha)
                if torch.isfinite(gathered_kl_for_alpha).all():
                    current_batch_kl_measure = float(gathered_kl_for_alpha.mean().item())
                else:
                    current_batch_kl_measure = float("inf")

            old_per_token_logps = (
                per_token_logps.detach()
                if inputs["old_per_token_logps"] is None
                else inputs["old_per_token_logps"]
            )
            advantages = inputs["advantages"]
            advantages, old_per_token_logps = self._maybe_apply_seed_grpo_advantages_in_loss(
                inputs,
                completion_ids=completion_ids,
                completion_mask=completion_mask,
                behavior_logps=old_per_token_logps.detach(),
                mode=mode,
            )
            log_ratio = _clamp_log_delta(per_token_logps - old_per_token_logps)
            coef_1 = torch.exp(log_ratio).to(per_token_logps.dtype)
            coef_2 = torch.clamp(
                coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high
            )

            if self.args.delta is not None:
                coef_1 = torch.clamp(coef_1, max=self.args.delta)

            per_token_loss1 = coef_1 * advantages.unsqueeze(1)
            per_token_loss2 = coef_2 * advantages.unsqueeze(1)
            per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
            if per_token_kl is not None:
                per_token_loss = per_token_loss + self.beta * per_token_kl

            (
                alpha,
                alpha_multiplier,
                alpha_kl_measure,
                alpha_kl_threshold,
                alpha_kl_control_enabled,
                alpha_direction,
                alpha_kl_min_multiplier,
                alpha_kl_max_multiplier,
                alpha_trust_zone_blocked,
            ) = self._resolve_effective_maxent_alpha(
                mode,
                measured_kl_override=current_batch_kl_measure,
            )
            completion_mask_f = completion_mask.to(
                device=per_token_entropy.device,
                dtype=per_token_entropy.dtype,
            )
            token_count_per_seq = completion_mask_f.sum(dim=1).clamp(min=1.0)
            mean_entropy = (per_token_entropy * completion_mask_f).sum() / completion_mask_f.sum().clamp(
                min=1.0
            )
            entropy_per_seq = (
                (per_token_entropy * completion_mask_f).sum(dim=1) / token_count_per_seq
            )
            mean_entropy_per_seq = entropy_per_seq.mean()
            if self.loss_type == "grpo":
                loss = (
                    (per_token_loss * completion_mask).sum(-1)
                    / completion_mask.sum(-1).clamp(min=1.0)
                ).mean()
                entropy_bonus_basis = mean_entropy_per_seq / entropy_normalization_scale
            elif self.loss_type == "bnpo":
                loss = (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp(
                    min=1.0
                )
                entropy_bonus_basis = mean_entropy / entropy_normalization_scale
            elif self.loss_type == "dr_grpo":
                loss = (per_token_loss * completion_mask).sum() / self._dr_grpo_loss_denominator(
                    completion_mask,
                    loss_tensor=per_token_loss,
                    mode=mode,
                )
                # Under fixed-denominator Dr.GRPO, a raw per-token entropy bonus
                # creates a spurious incentive to emit longer completions. Apply
                # the bonus on sequence-mean entropy instead so each sample gets
                # equal entropy weight regardless of realized length.
                entropy_bonus_basis = mean_entropy_per_seq / entropy_normalization_scale
            else:
                raise ValueError(f"Unknown loss type: {self.loss_type}")
            alpha_before_invalid_guard = float(alpha)
            invalid_rollout_token_count = float(
                getattr(self, "_last_rollout_invalid_token_id_count", 0.0) or 0.0
            )
            invalid_rollout_bonus_blocked = (
                invalid_rollout_token_count > 0.0 and alpha_before_invalid_guard > 0.0
            )
            if invalid_rollout_bonus_blocked:
                alpha = 0.0
            entropy_bonus = alpha * entropy_bonus_basis
            loss = loss - entropy_bonus

            if per_token_kl is not None:
                mean_kl = (per_token_kl * completion_mask).sum() / completion_mask.sum()
                gathered_kl = _metric_tensor_for_logging(self, mean_kl, mode=mode)
                if isinstance(gathered_kl, torch.Tensor) and gathered_kl.numel() > 0:
                    kl_value = float(gathered_kl.nanmean().item())
                    self._append_metric_value(
                        mode,
                        "kl",
                        kl_value,
                    )

            is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (
                advantages.unsqueeze(1) < 0
            )
            is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (
                advantages.unsqueeze(1) > 0
            )
            is_region_clipped = is_low_clipped | is_high_clipped

            low_clip = (is_low_clipped * completion_mask).sum() / completion_mask.sum()
            high_clip = (is_high_clipped * completion_mask).sum() / completion_mask.sum()
            clip_ratio = (is_region_clipped * completion_mask).sum() / completion_mask.sum()

            gathered_low_clip = _metric_tensor_for_logging(self, low_clip, mode=mode)
            if isinstance(gathered_low_clip, torch.Tensor) and gathered_low_clip.numel() > 0:
                self._append_metric_value(
                    mode, "clip_ratio/low_mean", gathered_low_clip.nanmean().item()
                )
                self._append_metric_value(
                    mode, "clip_ratio/low_min", _nanmin_tensor(gathered_low_clip).item()
                )
            gathered_high_clip = _metric_tensor_for_logging(self, high_clip, mode=mode)
            if isinstance(gathered_high_clip, torch.Tensor) and gathered_high_clip.numel() > 0:
                self._append_metric_value(
                    mode, "clip_ratio/high_mean", gathered_high_clip.nanmean().item()
                )
                self._append_metric_value(
                    mode, "clip_ratio/high_max", _nanmax_tensor(gathered_high_clip).item()
                )
            gathered_clip_ratio = _metric_tensor_for_logging(self, clip_ratio, mode=mode)
            if isinstance(gathered_clip_ratio, torch.Tensor) and gathered_clip_ratio.numel() > 0:
                self._append_metric_value(
                    mode, "clip_ratio/region_mean", gathered_clip_ratio.nanmean().item()
                )

            gathered_entropy = _metric_tensor_for_logging(self, mean_entropy, mode=mode)
            gathered_entropy_per_seq = _metric_tensor_for_logging(
                self, entropy_per_seq, mode=mode
            )
            self._append_metric_value(mode, "maxent/alpha", alpha)
            self._append_metric_value(mode, "maxent/alpha_base", float(self.maxent_alpha))
            self._append_metric_value(
                mode,
                "maxent/alpha_before_invalid_token_guard",
                alpha_before_invalid_guard,
                include_legacy_aliases=False,
            )
            self._append_metric_value(mode, "maxent/alpha_multiplier", alpha_multiplier)
            self._append_metric_value(mode, "maxent/objective_variant_entropy", 1.0)
            self._append_metric_value(mode, "maxent/objective_variant_listwise", 0.0)
            self._append_metric_value(
                mode,
                "maxent/invalid_rollout_bonus_blocked",
                1.0 if invalid_rollout_bonus_blocked else 0.0,
                include_legacy_aliases=False,
            )
            self._append_metric_value(
                mode,
                "maxent/rollout_invalid_token_id_count",
                invalid_rollout_token_count,
                include_legacy_aliases=False,
            )
            self._append_metric_value(
                mode,
                "maxent/alpha_kl_control_enabled",
                1.0 if alpha_kl_control_enabled else 0.0,
            )
            self._append_metric_value(
                mode,
                "maxent/alpha_trust_zone_blocked",
                1.0 if alpha_trust_zone_blocked else 0.0,
            )
            self._append_metric_value(mode, "maxent/alpha_kl_direction", alpha_direction)
            self._append_metric_value(mode, "maxent/alpha_kl_threshold", alpha_kl_threshold)
            self._append_metric_value(
                mode, "maxent/alpha_kl_min_multiplier", alpha_kl_min_multiplier
            )
            self._append_metric_value(
                mode, "maxent/alpha_kl_max_multiplier", alpha_kl_max_multiplier
            )
            if alpha_kl_measure is not None:
                self._append_metric_value(
                    mode, "maxent/alpha_kl_measure", alpha_kl_measure
                )
            if isinstance(gathered_entropy, torch.Tensor) and gathered_entropy.numel() > 0:
                raw_entropy_metric = (
                    gathered_entropy_per_seq.nanmean()
                    if self.loss_type in {"grpo", "dr_grpo"}
                    and isinstance(gathered_entropy_per_seq, torch.Tensor)
                    and gathered_entropy_per_seq.numel() > 0
                    else gathered_entropy.nanmean()
                )
                normalized_entropy_metric = raw_entropy_metric / entropy_normalization_scale
                normalized_entropy_token = gathered_entropy.nanmean() / entropy_normalization_scale
                self._append_metric_value(
                    mode, "maxent/policy_entropy_mean", raw_entropy_metric.item()
                )
                self._append_metric_value(
                    mode,
                    "maxent/policy_entropy_mean_token",
                    gathered_entropy.nanmean().item(),
                    include_legacy_aliases=False,
                )
                self._append_metric_value(
                    mode,
                    "maxent/policy_entropy_mean_normalized",
                    normalized_entropy_metric.item(),
                    include_legacy_aliases=False,
                )
                self._append_metric_value(
                    mode,
                    "maxent/policy_entropy_mean_token_normalized",
                    normalized_entropy_token.item(),
                    include_legacy_aliases=False,
                )
                if (
                    isinstance(gathered_entropy_per_seq, torch.Tensor)
                    and gathered_entropy_per_seq.numel() > 0
                ):
                    normalized_entropy_seq = (
                        gathered_entropy_per_seq.nanmean() / entropy_normalization_scale
                    )
                    self._append_metric_value(
                        mode,
                        "maxent/policy_entropy_mean_seq",
                        gathered_entropy_per_seq.nanmean().item(),
                        include_legacy_aliases=False,
                    )
                    self._append_metric_value(
                        mode,
                        "maxent/policy_entropy_mean_seq_normalized",
                        normalized_entropy_seq.item(),
                        include_legacy_aliases=False,
                    )
                self._append_metric_value(
                    mode,
                    "maxent/entropy_bonus_length_normalized",
                    1.0 if self.loss_type in {"grpo", "dr_grpo"} else 0.0,
                    include_legacy_aliases=False,
                )
                self._append_metric_value(
                    mode,
                    "maxent/entropy_normalization_log_vocab",
                    entropy_normalization_scale,
                    include_legacy_aliases=False,
                )
                self._append_metric_value(
                    mode,
                    "maxent/valid_vocab_size",
                    float(valid_vocab_size) if valid_vocab_size is not None else 0.0,
                    include_legacy_aliases=False,
                )
                self._append_metric_value(
                    mode,
                    "maxent/loss_entropy_bonus",
                    (-alpha * normalized_entropy_metric).item(),
                )
            if (
                isinstance(gathered_entropy_per_seq, torch.Tensor)
                and gathered_entropy_per_seq.numel() > 0
            ):
                self._append_metric_value(
                    mode,
                    "maxent/policy_entropy_std",
                    gathered_entropy_per_seq.to(torch.float32).std(unbiased=False).item(),
                )
            if mode == "train" and getattr(self, "_maxent_controller_objective", None) is not None:
                self._maybe_apply_controller_meta(
                    mode=mode,
                    kl_value=kl_value,
                    total_loss=float(loss.item()),
                )
                self._sync_weighting_scalars()
                self._append_metric_value(mode, "tau", float(self.tau))
                self._append_metric_value(mode, "beta", float(self.beta))
                self._append_metric_value(
                    mode,
                    "weight_norm_denom",
                    float(getattr(self, "weight_norm_denom", 1.0)),
                    include_legacy_aliases=False,
                )
            return loss

        def _compute_grpo_native_loss(
            self,
            *,
            model: Any,
            inputs: Any,
            return_outputs: bool,
            num_items_in_batch: Any = None,
        ) -> Any:
            """Run GRPO through the parent TRL loss implementation only."""
            try:
                return super().compute_loss(
                    model,
                    inputs,
                    return_outputs=return_outputs,
                    num_items_in_batch=num_items_in_batch,
                )
            except TypeError as exc:
                # Older TRL signatures may not accept num_items_in_batch.
                if "num_items_in_batch" not in str(exc):
                    raise
                return super().compute_loss(
                    model,
                    inputs,
                    return_outputs=return_outputs,
                )

        def _compute_stable_grpo_loss(self, model: Any, inputs: Any) -> torch.Tensor:
            """GRPO loss using the same stabilized exponentials as the MaxEnt path."""
            if bool(getattr(self, "use_liger_loss", False)):
                raise NotImplementedError(
                    "Stable GRPO loss is not implemented for liger loss."
                )

            prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
            completion_ids, completion_mask = (
                inputs["completion_ids"],
                inputs["completion_mask"],
            )
            input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
            attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
            logits_to_keep = completion_ids.size(1)
            mode = "train" if self.model.training else "eval"

            configured_batch_size = (
                int(getattr(self.args, "per_device_train_batch_size", 1) or 1)
                if self.model.training
                else int(getattr(self.args, "per_device_eval_batch_size", 1) or 1)
            )
            chunk_size = int(
                getattr(self.args, "maxent_logprob_chunk_size", 0)
                or configured_batch_size
                or 1
            )
            per_token_logps = self._get_per_token_logps(
                model,
                input_ids,
                attention_mask,
                logits_to_keep,
                batch_size=chunk_size,
            )

            per_token_kl: Optional[torch.Tensor] = None
            if self.beta != 0.0:
                use_model_reference = self._should_use_model_reference_logprobs(
                    default_to_model_reference=False
                )
                with torch.no_grad():
                    if use_model_reference:
                        ref_per_token_logps = self._get_reference_per_token_logps(
                            input_ids,
                            attention_mask,
                            logits_to_keep,
                            batch_size=chunk_size,
                        )
                    else:
                        ref_per_token_logps = None
                    if ref_per_token_logps is not None:
                        ref_per_token_logps = ref_per_token_logps.to(
                            device=per_token_logps.device,
                            dtype=per_token_logps.dtype,
                        )
                    else:
                        old_ref = inputs.get("old_per_token_logps")
                        if isinstance(old_ref, torch.Tensor):
                            ref_per_token_logps = old_ref.to(
                                device=per_token_logps.device,
                                dtype=per_token_logps.dtype,
                            )
                        else:
                            ref_per_token_logps = per_token_logps.detach()
                kl_delta = _clamp_log_delta(ref_per_token_logps - per_token_logps)
                per_token_kl = (
                    torch.exp(kl_delta)
                    - kl_delta
                    - 1
                ).to(per_token_logps.dtype)

            old_per_token_logps = (
                per_token_logps.detach()
                if inputs["old_per_token_logps"] is None
                else inputs["old_per_token_logps"]
            )
            advantages = inputs["advantages"]
            advantages, old_per_token_logps = self._maybe_apply_seed_grpo_advantages_in_loss(
                inputs,
                completion_ids=completion_ids,
                completion_mask=completion_mask,
                behavior_logps=old_per_token_logps.detach(),
                mode=mode,
            )
            log_ratio = _clamp_log_delta(per_token_logps - old_per_token_logps)
            coef_1 = torch.exp(log_ratio).to(per_token_logps.dtype)
            coef_2 = torch.clamp(
                coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high
            )

            if self.args.delta is not None:
                coef_1 = torch.clamp(coef_1, max=self.args.delta)

            per_token_loss1 = coef_1 * advantages.unsqueeze(1)
            per_token_loss2 = coef_2 * advantages.unsqueeze(1)
            per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
            if per_token_kl is not None:
                per_token_loss = per_token_loss + self.beta * per_token_kl

            if self.loss_type == "grpo":
                loss = (
                    (per_token_loss * completion_mask).sum(-1)
                    / completion_mask.sum(-1).clamp(min=1.0)
                ).mean()
            elif self.loss_type == "bnpo":
                loss = (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp(
                    min=1.0
                )
            elif self.loss_type == "dr_grpo":
                loss = (per_token_loss * completion_mask).sum() / self._dr_grpo_loss_denominator(
                    completion_mask,
                    loss_tensor=per_token_loss,
                    mode=mode,
                )
            else:
                raise ValueError(f"Unknown loss type: {self.loss_type}")

            if per_token_kl is not None:
                mean_kl = (per_token_kl * completion_mask).sum() / completion_mask.sum()
                gathered_kl = _metric_tensor_for_logging(self, mean_kl, mode=mode)
                if isinstance(gathered_kl, torch.Tensor) and gathered_kl.numel() > 0:
                    kl_value = float(gathered_kl.nanmean().item())
                    self._append_metric_value(mode, "kl", kl_value)

            is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (
                advantages.unsqueeze(1) < 0
            )
            is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (
                advantages.unsqueeze(1) > 0
            )
            is_region_clipped = is_low_clipped | is_high_clipped

            low_clip = (is_low_clipped * completion_mask).sum() / completion_mask.sum()
            high_clip = (is_high_clipped * completion_mask).sum() / completion_mask.sum()
            clip_ratio = (is_region_clipped * completion_mask).sum() / completion_mask.sum()

            gathered_low_clip = _metric_tensor_for_logging(self, low_clip, mode=mode)
            if isinstance(gathered_low_clip, torch.Tensor) and gathered_low_clip.numel() > 0:
                self._append_metric_value(
                    mode, "clip_ratio/low_mean", gathered_low_clip.nanmean().item()
                )
                self._append_metric_value(
                    mode, "clip_ratio/low_min", _nanmin_tensor(gathered_low_clip).item()
                )
            gathered_high_clip = _metric_tensor_for_logging(self, high_clip, mode=mode)
            if isinstance(gathered_high_clip, torch.Tensor) and gathered_high_clip.numel() > 0:
                self._append_metric_value(
                    mode, "clip_ratio/high_mean", gathered_high_clip.nanmean().item()
                )
                self._append_metric_value(
                    mode, "clip_ratio/high_max", _nanmax_tensor(gathered_high_clip).item()
                )
            gathered_clip_ratio = _metric_tensor_for_logging(self, clip_ratio, mode=mode)
            if isinstance(gathered_clip_ratio, torch.Tensor) and gathered_clip_ratio.numel() > 0:
                self._append_metric_value(
                    mode, "clip_ratio/region_mean", gathered_clip_ratio.nanmean().item()
                )
            self._append_metric_value(mode, "maxent/objective_variant_entropy", 0.0)
            self._append_metric_value(mode, "maxent/objective_variant_listwise", 0.0)
            return loss

        def compute_loss(  # type: ignore[override]
            self,
            model: Any,
            inputs: Any,
            return_outputs: bool = False,
            num_items_in_batch: Any = None,
        ) -> Any:
            native_grpo_route = False
            trl_prepared_inputs = isinstance(inputs, dict) and all(
                key in inputs
                for key in (
                    "prompt_ids",
                    "prompt_mask",
                    "completion_ids",
                    "completion_mask",
                    "advantages",
                )
            )
            lightweight_eval = bool(
                (not getattr(self.model, "training", False))
                and trl_prepared_inputs
                and getattr(
                    getattr(self, "args", None),
                    "eval_greedy_only_enabled",
                    False,
                )
            )
            if lightweight_eval:
                if return_outputs:
                    raise ValueError(
                        "The lightweight greedy eval path does not support returning outputs"
                    )
                loss = self._compute_stable_grpo_loss(model=model, inputs=inputs)
            elif self.objective_routing.uses_listwise_loss and trl_prepared_inputs:
                if return_outputs:
                    raise ValueError(
                        "The custom listwise MaxEnt GRPOTrainer does not support returning outputs"
                    )
                loss = self._compute_listwise_maxent_loss(model=model, inputs=inputs)
            elif (
                self.objective_routing.uses_entropy_regularized_loss
                and trl_prepared_inputs
            ):
                if return_outputs:
                    raise ValueError(
                        "The custom MaxEnt GRPOTrainer does not support returning outputs"
                    )
                loss = self._compute_maxent_loss(model=model, inputs=inputs)
            else:
                if trl_prepared_inputs and not return_outputs:
                    loss = self._compute_stable_grpo_loss(model=model, inputs=inputs)
                else:
                    native_grpo_route = True
                    loss = self._compute_grpo_native_loss(
                        model=model,
                        inputs=inputs,
                        return_outputs=return_outputs,
                        num_items_in_batch=num_items_in_batch,
                    )
            # Cache the latest train KL immediately after native TRL loss
            # computation so adaptive MaxEnt alpha can be evaluated every
            # optimizer step (the next rollout consumes this cached value).
            if bool(getattr(self.model, "training", False)):
                train_metrics = self._metrics.get("train", {})
                kl_history = (
                    train_metrics.get("kl") if isinstance(train_metrics, dict) else None
                )
                if isinstance(kl_history, list) and kl_history:
                    kl_value = _numeric_or_none(kl_history[-1])
                    if kl_value is not None:
                        setattr(self, "_last_train_kl_for_alpha", float(kl_value))
                else:
                    kl_value = None
                if (
                    native_grpo_route
                    and getattr(self, "_maxent_controller_objective", None) is not None
                ):
                    loss_value = loss[0] if isinstance(loss, tuple) else loss
                    self._maybe_apply_controller_meta(
                        mode="train",
                        kl_value=kl_value,
                        total_loss=_numeric_or_none(loss_value),
                    )
                    self._sync_weighting_scalars()
                    self._append_metric_value("train", "tau", float(self.tau))
                    self._append_metric_value("train", "beta", float(self.beta))
                    self._append_metric_value(
                        "train",
                        "weight_norm_denom",
                        float(getattr(self, "weight_norm_denom", 1.0)),
                        include_legacy_aliases=False,
                    )
                if self.maxent_enabled:
                    self._sync_weighting_scalars()
                    self._maybe_update_reference_model_ema()
            return loss

        def _generate_and_score_completions(  # type: ignore[override]
            self, inputs: List[Dict[str, Any]]
        ) -> Dict[str, Any]:
            mode = "train" if self.model.training else "eval"
            if mode == "eval" and bool(
                getattr(getattr(self, "args", None), "eval_greedy_only_enabled", False)
            ):
                outputs = self._generate_greedy_eval_outputs(inputs)
                self._log_eval_greedy_metrics(inputs, outputs, mode=mode)
                return outputs

            outputs = super()._generate_and_score_completions(inputs)
            self._sanitize_rollout_token_ids(outputs, mode=mode)
            self._maybe_truncate_completions_at_first_boxed_answer(
                inputs,
                outputs,
                mode=mode,
            )
            defer_seed_scaling = mode == "train" and bool(
                getattr(getattr(self, "args", None), "seed_grpo_enabled", False)
            )
            if not defer_seed_scaling:
                self._maybe_backfill_old_per_token_logps(outputs, mode=mode)
                self._maybe_apply_seed_grpo_advantages(
                    inputs,
                    outputs,
                    mode=mode,
                )
            if self.objective_routing.uses_listwise_loss:
                self._prepare_listwise_rollout_targets(inputs, outputs)
            self._recompute_completion_metrics(outputs, mode=mode)
            self._maybe_log_rich_rollout_sidecar(inputs, outputs, mode=mode)
            self._log_grpo_diversity(outputs, mode=mode)
            self._log_eval_pass_at_k(inputs, outputs, mode=mode)
            self._log_eval_greedy_metrics(inputs, outputs, mode=mode)
            if not self.maxent_enabled:
                self._log_grpo_debug(inputs, outputs, mode=mode)
            return outputs

    CustomGRPOTrainer.__name__ = "CustomGRPOTrainer"
    return ensure_weighting_logging(CustomGRPOTrainer)


def wrap_trl_trainer(trainer_cls: Type[Any]) -> Type[Any]:
    """Ensure a trainer class emits TRL-style logs and metrics."""

    return ensure_weighting_logging(trainer_cls)


__all__ = ["build_custom_grpo_trainer", "wrap_trl_trainer"]
