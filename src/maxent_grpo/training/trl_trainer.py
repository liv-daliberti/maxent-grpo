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
from functools import partial
import inspect
import re
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Type, cast

import torch
import torch.nn.functional as F

try:
    from accelerate.utils import gather
except (ImportError, ModuleNotFoundError):  # pragma: no cover - test fallback

    def gather(value: Any) -> Any:
        return value


try:
    from trl.data_utils import apply_chat_template, is_conversational
except (ImportError, ModuleNotFoundError):  # pragma: no cover - test fallback

    def apply_chat_template(example: Any, _tokenizer: Any) -> Dict[str, str]:
        return {"text": str(example)}

    def is_conversational(example: Any) -> bool:
        return isinstance(example, list)


from maxent_grpo.rewards.basic import (
    pure_accuracy_math_correctness,
    uses_pure_accuracy_math_reward,
)
from maxent_grpo.objectives import resolve_objective_routing
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

LOG = logging.getLogger(__name__)
_PASS_METRIC_SUCCESS_REWARD = 1.0
_PASS_METRIC_EPS = 1e-6
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


def _metric_suffix_from_benchmark(name: Any) -> str:
    """Return a metric-safe benchmark suffix (e.g., ``AIME24``)."""

    text = str(name).strip()
    if not text:
        return "UNKNOWN"
    cleaned = _BENCHMARK_SUFFIX_SANITIZER.sub("_", text).strip("_").upper()
    return cleaned or "UNKNOWN"


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
            self.maxent_enabled = self.objective_routing.maxent_requested
            self.maxent_objective_variant = (
                self.objective_routing.maxent_objective_variant
            )
            self.maxent_alpha = self.objective_routing.maxent_alpha
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
            if self._last_grpo_debug_step == step:
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
            use_tokenizer = (
                tokenizer
                if callable(getattr(tokenizer, "encode", None)) or callable(tokenizer)
                else None
            )
            metrics = _completion_diversity_metrics(
                grouped,
                tokenizer=use_tokenizer,
                accelerator=self.accelerator,
            )
            if metrics:
                for key, val in metrics.items():
                    self._append_metric_value(
                        mode,
                        f"completions/diversity/{key}",
                        float(val),
                    )

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
            for start in range(0, int(input_ids.size(0)), chunk_size):
                stop = start + chunk_size
                input_ids_batch = input_ids[start:stop]
                attention_mask_batch = attention_mask[start:stop]
                logits = model(
                    input_ids=input_ids_batch,
                    attention_mask=attention_mask_batch,
                    logits_to_keep=logits_to_keep + 1,
                ).logits
                logits = logits[:, :-1, :]
                token_ids = input_ids_batch[:, -logits_to_keep:]
                logits = logits[:, -logits_to_keep:]
                logits = logits / self.temperature
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
            reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
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

        def _resolve_listwise_reference_model(self) -> Tuple[Optional[Any], bool]:
            """Return the frozen reference model and whether to use it in weights."""
            args = getattr(self, "args", None)
            ref_source = str(
                getattr(args, "maxent_reference_logprobs_source", "auto") or "auto"
            ).strip().lower()
            if ref_source == "none":
                ref_source = "policy"
            if bool(getattr(args, "maxent_trl_reference_scoring", False)):
                ref_source = "model"
            ref_model = getattr(self, "ref_model", None)
            if ref_source == "model":
                if ref_model is not None and ref_model is not self.model:
                    return ref_model, True
                if not bool(getattr(self, "_maxent_listwise_ref_warned", False)):
                    LOG.warning(
                        "Listwise MaxEnt requested reference weighting but no independent frozen reference model is available; falling back to q-only weights."
                    )
                    setattr(self, "_maxent_listwise_ref_warned", True)
                return None, False
            if ref_source == "auto" and ref_model is not None and ref_model is not self.model:
                return ref_model, True
            return None, False

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

            weighting = getattr(self, "_maxent_weighting", None)
            if weighting is None:
                raise ValueError("Listwise MaxEnt requires initialized weighting settings.")

            ref_model, include_reference_term = self._resolve_listwise_reference_model()
            ref_seq_logps_grouped = torch.zeros_like(seq_logps_grouped)
            measured_kl: Optional[torch.Tensor] = None
            if include_reference_term and ref_model is not None:
                with torch.no_grad():
                    ref_per_token_logps = self._get_per_token_logps(
                        ref_model,
                        input_ids,
                        attention_mask,
                        logits_to_keep,
                        batch_size=chunk_size,
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
            weights_grouped_list = weights_grouped.detach().cpu().tolist()
            log_probs_grouped = torch.log_softmax(policy_seq_logps_grouped, dim=1)
            policy_loss = -(weights_grouped * log_probs_grouped).sum(dim=1).mean()
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
                    clip_loss = -clip_obj.sum(dim=1).mean()
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
                "maxent/listwise_loss_scale",
                loss_scale_value,
            )
            if clip_loss is not None:
                self._append_metric_value(mode, "loss/clip", float(clip_loss.item()))
            if measured_kl is not None:
                kl_value = float(self.accelerator.gather(measured_kl).nanmean().item())
                self._append_metric_value(mode, "kl", kl_value)
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

        def _get_per_token_logps(self, *args: Any, **kwargs: Any) -> torch.Tensor:  # type: ignore[override]
            logps = super()._get_per_token_logps(*args, **kwargs)
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
            if self.beta != 0.0:
                with torch.no_grad():
                    if self.ref_model is not None:
                        ref_per_token_logps = self._get_per_token_logps(
                            self.ref_model,
                            input_ids,
                            attention_mask,
                            logits_to_keep,
                            batch_size=chunk_size,
                        )
                    else:
                        with self.accelerator.unwrap_model(self.model).disable_adapter():
                            ref_per_token_logps = self._get_per_token_logps(
                                self.model,
                                input_ids,
                                attention_mask,
                                logits_to_keep,
                                batch_size=chunk_size,
                            )
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

            advantages = inputs["advantages"]
            old_per_token_logps = (
                per_token_logps.detach()
                if inputs["old_per_token_logps"] is None
                else inputs["old_per_token_logps"]
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
            per_token_loss = per_token_loss - alpha * per_token_entropy

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
                loss = (per_token_loss * completion_mask).sum() / (
                    per_token_loss.size(0) * self.max_completion_length
                )
            else:
                raise ValueError(f"Unknown loss type: {self.loss_type}")

            if per_token_kl is not None:
                mean_kl = (per_token_kl * completion_mask).sum() / completion_mask.sum()
                kl_value = float(self.accelerator.gather(mean_kl).nanmean().item())
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

            gathered_low_clip = self.accelerator.gather(low_clip)
            self._append_metric_value(
                mode, "clip_ratio/low_mean", gathered_low_clip.nanmean().item()
            )
            self._append_metric_value(
                mode, "clip_ratio/low_min", _nanmin_tensor(gathered_low_clip).item()
            )
            gathered_high_clip = self.accelerator.gather(high_clip)
            self._append_metric_value(
                mode, "clip_ratio/high_mean", gathered_high_clip.nanmean().item()
            )
            self._append_metric_value(
                mode, "clip_ratio/high_max", _nanmax_tensor(gathered_high_clip).item()
            )
            gathered_clip_ratio = self.accelerator.gather(clip_ratio)
            self._append_metric_value(
                mode, "clip_ratio/region_mean", gathered_clip_ratio.nanmean().item()
            )

            entropy_mask = completion_mask.to(
                device=per_token_entropy.device, dtype=per_token_entropy.dtype
            )
            mean_entropy = (per_token_entropy * entropy_mask).sum() / entropy_mask.sum().clamp(
                min=1.0
            )
            entropy_per_seq = (
                (per_token_entropy * entropy_mask).sum(dim=1)
                / entropy_mask.sum(dim=1).clamp(min=1.0)
            )
            gathered_entropy = self.accelerator.gather(mean_entropy)
            gathered_entropy_per_seq = self.accelerator.gather(entropy_per_seq)
            self._append_metric_value(mode, "maxent/alpha", alpha)
            self._append_metric_value(mode, "maxent/alpha_base", float(self.maxent_alpha))
            self._append_metric_value(mode, "maxent/alpha_multiplier", alpha_multiplier)
            self._append_metric_value(mode, "maxent/objective_variant_entropy", 1.0)
            self._append_metric_value(mode, "maxent/objective_variant_listwise", 0.0)
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
            self._append_metric_value(
                mode, "maxent/policy_entropy_mean", gathered_entropy.nanmean().item()
            )
            self._append_metric_value(
                mode,
                "maxent/policy_entropy_std",
                gathered_entropy_per_seq.to(torch.float32).std(unbiased=False).item(),
            )
            self._append_metric_value(
                mode,
                "maxent/loss_entropy_bonus",
                (-alpha * gathered_entropy.nanmean()).item(),
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
                with torch.no_grad():
                    if self.ref_model is not None:
                        ref_per_token_logps = self._get_per_token_logps(
                            self.ref_model,
                            input_ids,
                            attention_mask,
                            logits_to_keep,
                            batch_size=chunk_size,
                        )
                    else:
                        with self.accelerator.unwrap_model(self.model).disable_adapter():
                            ref_per_token_logps = self._get_per_token_logps(
                                self.model,
                                input_ids,
                                attention_mask,
                                logits_to_keep,
                                batch_size=chunk_size,
                            )
                kl_delta = _clamp_log_delta(ref_per_token_logps - per_token_logps)
                per_token_kl = (
                    torch.exp(kl_delta)
                    - kl_delta
                    - 1
                ).to(per_token_logps.dtype)

            advantages = inputs["advantages"]
            old_per_token_logps = (
                per_token_logps.detach()
                if inputs["old_per_token_logps"] is None
                else inputs["old_per_token_logps"]
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
                loss = (per_token_loss * completion_mask).sum() / (
                    per_token_loss.size(0) * self.max_completion_length
                )
            else:
                raise ValueError(f"Unknown loss type: {self.loss_type}")

            if per_token_kl is not None:
                mean_kl = (per_token_kl * completion_mask).sum() / completion_mask.sum()
                kl_value = float(self.accelerator.gather(mean_kl).nanmean().item())
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

            gathered_low_clip = self.accelerator.gather(low_clip)
            self._append_metric_value(
                mode, "clip_ratio/low_mean", gathered_low_clip.nanmean().item()
            )
            self._append_metric_value(
                mode, "clip_ratio/low_min", _nanmin_tensor(gathered_low_clip).item()
            )
            gathered_high_clip = self.accelerator.gather(high_clip)
            self._append_metric_value(
                mode, "clip_ratio/high_mean", gathered_high_clip.nanmean().item()
            )
            self._append_metric_value(
                mode, "clip_ratio/high_max", _nanmax_tensor(gathered_high_clip).item()
            )
            gathered_clip_ratio = self.accelerator.gather(clip_ratio)
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
            if self.objective_routing.uses_listwise_loss and trl_prepared_inputs:
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
            outputs = super()._generate_and_score_completions(inputs)
            mode = "train" if self.model.training else "eval"
            if self.objective_routing.uses_listwise_loss:
                self._prepare_listwise_rollout_targets(inputs, outputs)
            self._log_grpo_diversity(outputs, mode=mode)
            self._log_eval_pass_at_k(inputs, outputs, mode=mode)
            if not self.maxent_enabled:
                self._log_grpo_debug(inputs, outputs, mode=mode)
            return outputs

    CustomGRPOTrainer.__name__ = "CustomGRPOTrainer"
    return ensure_weighting_logging(CustomGRPOTrainer)


def wrap_trl_trainer(trainer_cls: Type[Any]) -> Type[Any]:
    """Ensure a trainer class emits TRL-style logs and metrics."""

    return ensure_weighting_logging(trainer_cls)


__all__ = ["build_custom_grpo_trainer", "wrap_trl_trainer"]
