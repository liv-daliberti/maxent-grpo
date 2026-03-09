"""Custom TRL GRPOTrainer wrapper used by the MaxEnt-GRPO pipelines.

This module is the single place where GRPO-vs-MaxEnt objective behavior should
diverge at runtime. The surrounding training pipeline (dataset mapping, reward
loading, model/tokenizer setup, trainer wiring, launch entrypoints) is kept
shared so objective comparisons stay fair and easy to audit.
"""

from __future__ import annotations

import logging
import math
from functools import partial
import inspect
import re
from typing import Any, Dict, List, Optional, Tuple, Type, cast

import torch
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
from maxent_grpo.training.telemetry.trl_logging import ensure_weighting_logging

LOG = logging.getLogger(__name__)
_PASS_METRIC_SUCCESS_REWARD = 1.0
_PASS_METRIC_EPS = 1e-6
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
                gathered = gather_fn(group_metrics)
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
                                    merged.extend([m for m in item if isinstance(m, dict)])
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
    params: Dict[str, torch.Tensor]
) -> Dict[str, List[Tuple[str, torch.Tensor, int]]]:
    """Index tensors by canonicalized names for alias-aware EMA matching."""
    by_canonical: Dict[str, List[Tuple[str, torch.Tensor, int]]] = {}
    for name, param in params.items():
        canonical, stripped = _strip_ema_param_prefixes(name)
        by_canonical.setdefault(canonical, []).append((name, param, stripped))
    for candidates in by_canonical.values():
        candidates.sort(key=lambda item: (item[2], len(item[0]), item[0]))
    return by_canonical


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

        @staticmethod
        def _is_maxent_requested(training_args: Any) -> bool:
            """Return True when requested objective is MaxEnt (not vanilla GRPO)."""
            return not _coerce_bool(
                getattr(training_args, "train_grpo_objective", True),
                default=True,
            )

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            parent_args = self._resolve_parent_training_args(args, kwargs)
            maxent_requested = self._is_maxent_requested(parent_args)
            parent_alpha_default = 1.0 if maxent_requested else 0.0
            parent_maxent_alpha = _coerce_non_negative_float(
                getattr(parent_args, "maxent_alpha", parent_alpha_default)
                if parent_args is not None
                else parent_alpha_default,
                default=parent_alpha_default,
            )
            super().__init__(*args, **kwargs)

            self.maxent_enabled = not _coerce_bool(
                getattr(getattr(self, "args", None), "train_grpo_objective", True),
                default=True,
            )
            self.maxent_alpha = _coerce_non_negative_float(
                getattr(getattr(self, "args", None), "maxent_alpha", parent_maxent_alpha),
                default=parent_maxent_alpha,
            )
            if not self.maxent_enabled:
                route_mode = "grpo_native"
            elif self.maxent_alpha > 0.0:
                route_mode = "maxent_native_plus_alpha"
            else:
                route_mode = "maxent_native_alpha0"
            LOG.info(
                "Objective routing selected | mode=%s | train_grpo_objective=%s | maxent_alpha=%s",
                route_mode,
                getattr(getattr(self, "args", None), "train_grpo_objective", None),
                self.maxent_alpha,
            )
            self._last_train_kl_for_alpha: Optional[float] = None
            self._last_grpo_debug_step: Optional[int] = None
            self._last_maxent_flow_log_step: Optional[int] = None
            self._last_reference_ema_step: Optional[int] = None

        def get_train_dataloader(self):  # type: ignore[override]
            # Preserve native TRL batching/sampling behavior for GRPO/MaxEnt while
            # adapting worker_init_fn to the active transformers seed_worker signature.
            if self.train_dataset is None:
                raise ValueError("Trainer: training requires a train_dataset.")

            train_dataset = self.train_dataset
            data_collator = self.data_collator

            try:
                from transformers.utils import is_datasets_available
            except Exception:  # pragma: no cover - transformers is required for training
                is_datasets_available = lambda: False  # type: ignore

            if is_datasets_available():
                try:
                    import datasets
                except Exception:
                    datasets = None  # type: ignore
                if datasets is not None and isinstance(train_dataset, datasets.Dataset):
                    train_dataset = self._remove_unused_columns(train_dataset, description="training")
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
                dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor
                worker_init_fn = _build_seed_worker(
                    self.args.dataloader_num_workers, self.args.process_index
                )
                if worker_init_fn is not None:
                    dataloader_params["worker_init_fn"] = worker_init_fn

            return self.accelerator.prepare(
                torch.utils.data.DataLoader(train_dataset, **dataloader_params)
            )

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
            local_dropped = max(local_expected - local_actual, 0)
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
            args = getattr(self, "args", None)
            if args is None:
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
            self.beta = new_beta

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
                policy_params = {
                    str(name): param for name, param in policy_named() if isinstance(param, torch.Tensor)
                }
                ref_params = {
                    str(name): param for name, param in ref_named() if isinstance(param, torch.Tensor)
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
        ) -> Tuple[float, float, Optional[float], float, bool, float, float, float]:
            """Return effective MaxEnt alpha with optional KL-based up/down scaling.

            Returns ``(effective_alpha, multiplier, measured_kl, kl_threshold,
            kl_control_enabled, direction, min_multiplier, max_multiplier)`` where
            direction is ``+1`` (raised), ``-1`` (lowered), or ``0`` (unchanged).
            """
            del mode
            base_alpha = float(getattr(self, "maxent_alpha", 0.0) or 0.0)
            if base_alpha <= 0.0:
                return 0.0, 1.0, None, 0.0, False, 0.0, 1.0, 1.0
            args = getattr(self, "args", None)
            raise_on_low_kl = bool(
                getattr(args, "maxent_alpha_raise_on_low_kl", False)
            )
            lower_on_high_kl = bool(
                getattr(args, "maxent_alpha_lower_on_high_kl", False)
            )
            enabled = raise_on_low_kl or lower_on_high_kl
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
                )

            measured_kl: Optional[float] = None
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
                )
            if not math.isfinite(measured_kl):
                return (
                    base_alpha,
                    1.0,
                    None,
                    threshold,
                    True,
                    0.0,
                    min_multiplier,
                    max_multiplier,
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
            if measured_kl < threshold and raise_on_low_kl:
                low_kl_frac = max(threshold - measured_kl, 0.0) / max(threshold, 1e-8)
                multiplier = 1.0 + gain * low_kl_frac
                direction = 1.0
            elif measured_kl > threshold and lower_on_high_kl:
                high_kl_frac = max(measured_kl - threshold, 0.0) / max(
                    threshold, 1e-8
                )
                multiplier = 1.0 / (1.0 + gain * high_kl_frac)
                direction = -1.0
            if not math.isfinite(multiplier):
                multiplier = 1.0
                direction = 0.0
            multiplier = min(max(multiplier, min_multiplier), max_multiplier)
            return (
                base_alpha * multiplier,
                multiplier,
                measured_kl,
                threshold,
                True,
                direction,
                min_multiplier,
                max_multiplier,
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
                completions_text = decode(
                    completion_ids, skip_special_tokens=True
                )
            except Exception:
                return
            group_size = max(
                int(getattr(self, "num_generations", 1) or 1), 1
            )
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
                """Reshape generation-major rollouts into prompt-major groups."""
                usable = (int(flat_values.numel()) // group_size) * group_size
                if usable <= 0:
                    return None
                if usable != int(flat_values.numel()):
                    flat_values = flat_values[:usable]
                num_prompts = usable // group_size
                if num_prompts <= 0:
                    return None
                # TRL rollout order is [gen0 prompts..., gen1 prompts..., ...].
                return flat_values.view(group_size, num_prompts).transpose(0, 1)

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
                    self._append_metric_value(mode, f"pass_at_8_{suffix}", bench_pass_at_8)
                    self._append_metric_value(mode, f"pass_at_1_{suffix}", bench_pass_at_1)
                    self._append_metric_value(mode, f"mean_at_1_{suffix}", bench_mean_at_1)

            successes: Optional[torch.Tensor] = None
            reward_funcs = list(getattr(self, "reward_funcs", []) or [])
            if uses_pure_accuracy_math_reward(reward_funcs):
                completion_ids = outputs.get("completion_ids")
                tokenizer = getattr(self, "processing_class", None)
                decode = getattr(tokenizer, "batch_decode", None)
                if isinstance(completion_ids, torch.Tensor) and callable(decode):
                    try:
                        completions_text = decode(
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
                    rewards = self._recompute_global_rewards_for_outputs(inputs, outputs)
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

        def _compute_policy_maxent_term(
            self,
            outputs: Dict[str, Any],
        ) -> Optional[torch.Tensor]:
            """Return sequence-level novelty under the frozen reference policy.

            Computes ``-log pi_ref(a|s)`` (token-averaged over completion tokens)
            for actions sampled from the current policy, then gathers across ranks.
            """
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
                device=completion_ids.device, dtype=torch.float32
            )
            prompt_ids = outputs.get("prompt_ids")
            prompt_mask = outputs.get("prompt_mask")
            if not isinstance(prompt_ids, torch.Tensor) or not isinstance(
                prompt_mask, torch.Tensor
            ):
                return None
            ref_model = getattr(self, "ref_model", None)
            input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
            attention_mask = torch.cat([prompt_mask, completion_mask.long()], dim=1)
            logits_to_keep = completion_ids.size(1)
            batch_size = (
                int(getattr(self.args, "per_device_train_batch_size", 1) or 1)
                if self.model.training
                else int(getattr(self.args, "per_device_eval_batch_size", 1) or 1)
            )
            per_token_logps: Optional[torch.Tensor]
            if ref_model is not None:
                with torch.no_grad():
                    per_token_logps = self._get_per_token_logps(
                        ref_model,
                        input_ids,
                        attention_mask,
                        logits_to_keep,
                        batch_size=batch_size,
                    )
            else:
                if not bool(getattr(self, "_maxent_ref_missing_warned", False)):
                    LOG.warning(
                        "No frozen reference model found; falling back to rollout-policy log-probs for MaxEnt novelty shaping."
                    )
                    setattr(self, "_maxent_ref_missing_warned", True)
                per_token_logps = outputs.get("old_per_token_logps")
                if not isinstance(per_token_logps, torch.Tensor):
                    with torch.no_grad():
                        per_token_logps = self._get_per_token_logps(
                            self.model,
                            input_ids,
                            attention_mask,
                            logits_to_keep,
                            batch_size=batch_size,
                        )
                    if isinstance(per_token_logps, torch.Tensor):
                        outputs["old_per_token_logps"] = per_token_logps.detach()
            if not isinstance(per_token_logps, torch.Tensor):
                return None
            if per_token_logps.ndim != 2:
                return None
            if (
                per_token_logps.size(0) != completion_mask.size(0)
                or per_token_logps.size(1) != completion_mask.size(1)
            ):
                usable_rows = min(per_token_logps.size(0), completion_mask.size(0))
                usable_cols = min(per_token_logps.size(1), completion_mask.size(1))
                if usable_rows <= 0 or usable_cols <= 0:
                    return None
                per_token_logps = per_token_logps[:usable_rows, :usable_cols]
                completion_mask = completion_mask[:usable_rows, :usable_cols]
            token_counts = completion_mask.sum(dim=1).clamp(min=1.0)
            novelty_local = -(
                per_token_logps.to(torch.float32) * completion_mask
            ).sum(dim=1) / token_counts
            novelty = gather(novelty_local)
            return torch.nan_to_num(novelty, nan=0.0, posinf=0.0, neginf=0.0)

        def _recompute_global_rewards_for_outputs(
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
                [int(tok.item()) for tok, keep in zip(row, mask_row) if int(keep.item()) != 0]
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
                getattr(self, "reward_processing_classes", [None] * len(self.reward_funcs))
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
                        messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
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
                        rewards_per_func_local[:, i] = reward_func(**reward_inputs).logits[:, 0]
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

            rewards_per_func = gather(rewards_per_func_local)
            reward_weights = getattr(self, "reward_weights", None)
            if isinstance(reward_weights, torch.Tensor):
                weights = reward_weights.to(device=rewards_per_func.device, dtype=torch.float32)
            elif isinstance(reward_weights, (list, tuple)):
                weights = torch.tensor(
                    list(reward_weights),
                    dtype=torch.float32,
                    device=rewards_per_func.device,
                )
            else:
                weights = torch.ones(
                    (len(self.reward_funcs),),
                    dtype=torch.float32,
                    device=rewards_per_func.device,
                )
            if weights.numel() != rewards_per_func.size(1):
                weights = torch.ones(
                    (rewards_per_func.size(1),),
                    dtype=torch.float32,
                    device=rewards_per_func.device,
                )
            rewards = (rewards_per_func * weights.unsqueeze(0)).nansum(dim=1)
            return torch.nan_to_num(rewards, nan=0.0, posinf=0.0, neginf=0.0)

        def _apply_maxent_reward_shaping(
            self,
            inputs: List[Dict[str, Any]],
            outputs: Dict[str, Any],
            *,
            mode: str,
        ) -> None:
            if not self.maxent_enabled or self.maxent_alpha <= 0.0:
                return
            advantages = outputs.get("advantages")
            if not isinstance(advantages, torch.Tensor):
                return
            try:
                rewards = self._recompute_global_rewards_for_outputs(inputs, outputs)
                maxent_term = self._compute_policy_maxent_term(outputs)
            except Exception as exc:
                LOG.warning("Skipping MaxEnt reward shaping due to error: %s", exc)
                return
            if rewards is None or maxent_term is None:
                return
            if rewards.numel() != maxent_term.numel():
                return

            args = getattr(self, "args", None)
            (
                alpha,
                alpha_multiplier,
                alpha_kl_measure,
                alpha_kl_threshold,
                alpha_kl_control_enabled,
                alpha_direction,
                alpha_kl_min_multiplier,
                alpha_kl_max_multiplier,
            ) = self._resolve_effective_maxent_alpha(mode)
            group_size = max(int(getattr(self, "num_generations", 1) or 1), 1)
            if rewards.numel() % group_size != 0:
                return
            num_prompts = int(rewards.numel()) // group_size
            if num_prompts <= 0:
                return

            def _reshape_prompt_groups(flat_values: torch.Tensor) -> torch.Tensor:
                """Convert generation-major flat rollouts to prompt-major groups."""
                return flat_values.view(group_size, num_prompts).transpose(0, 1)

            def _flatten_prompt_groups(grouped_values: torch.Tensor) -> torch.Tensor:
                """Convert prompt-major grouped values back to generation-major flat order."""
                return grouped_values.transpose(0, 1).reshape(-1)

            grouped_rewards_raw = _reshape_prompt_groups(rewards)
            grouped_novelty = _reshape_prompt_groups(
                maxent_term.to(device=rewards.device, dtype=rewards.dtype)
            )
            alpha_low_kl_enabled = bool(
                getattr(args, "maxent_alpha_raise_on_low_kl", False)
            )
            alpha_high_kl_enabled = bool(
                getattr(args, "maxent_alpha_lower_on_high_kl", False)
            )

            gate_by_signal = bool(
                getattr(args, "maxent_reward_signal_gate", False)
            )
            if gate_by_signal:
                min_group_max = float(
                    getattr(args, "maxent_reward_signal_min_max", 0.0) or 0.0
                )
                std_threshold = float(
                    getattr(args, "maxent_reward_signal_std_threshold", 0.0) or 0.0
                )
                group_max = grouped_rewards_raw.max(dim=1).values
                group_std = grouped_rewards_raw.std(dim=1, unbiased=False)
                eligible_groups = (group_max > min_group_max) | (
                    group_std > std_threshold
                )
            else:
                eligible_groups = torch.ones(
                    grouped_rewards_raw.size(0),
                    dtype=torch.bool,
                    device=grouped_rewards_raw.device,
                )

            reward_bonus_mask = eligible_groups.unsqueeze(1).expand_as(
                grouped_rewards_raw
            )
            positive_only = bool(
                getattr(args, "maxent_bonus_positive_only", False)
            )
            if positive_only:
                min_reward = float(getattr(args, "maxent_bonus_min_reward", 0.0) or 0.0)
                reward_bonus_mask = reward_bonus_mask & (grouped_rewards_raw > min_reward)
            cusp_gate_enabled = bool(getattr(args, "maxent_cusp_gate", False))
            cusp_threshold_raw = getattr(args, "maxent_cusp_reward_threshold", 0.4)
            try:
                cusp_threshold = float(cusp_threshold_raw)
            except (TypeError, ValueError):
                cusp_threshold = 0.4
            if not math.isfinite(cusp_threshold):
                cusp_threshold = 0.4

            bonus_mask = reward_bonus_mask.to(grouped_novelty.dtype)
            if cusp_gate_enabled:
                group_success = (grouped_rewards_raw >= cusp_threshold).to(
                    grouped_novelty.dtype
                )
                group_success_rate = group_success.mean(dim=1, keepdim=True)
                cusp_factor = 4.0 * group_success_rate * (1.0 - group_success_rate)
                cusp_factor = cusp_factor.clamp(min=0.0, max=1.0)
            else:
                group_success_rate = torch.zeros(
                    (grouped_rewards_raw.size(0), 1),
                    dtype=grouped_novelty.dtype,
                    device=grouped_novelty.device,
                )
                cusp_factor = torch.ones(
                    (grouped_rewards_raw.size(0), 1),
                    dtype=grouped_novelty.dtype,
                    device=grouped_novelty.device,
                )
            mean_grouped_rewards = grouped_rewards_raw.mean(dim=1)
            std_grouped_rewards = grouped_rewards_raw.std(dim=1)
            is_std_zero = torch.isclose(
                std_grouped_rewards,
                torch.zeros_like(std_grouped_rewards),
            )
            gathered_advantages: Optional[torch.Tensor]
            try:
                gathered_advantages = gather(
                    advantages.to(device=rewards.device, dtype=rewards.dtype)
                )
            except Exception:
                gathered_advantages = None
            using_native_advantages = bool(
                isinstance(gathered_advantages, torch.Tensor)
                and gathered_advantages.numel() == rewards.numel()
            )
            if using_native_advantages:
                base_advantages = cast(torch.Tensor, gathered_advantages)
            else:
                grouped_base_advantages = grouped_rewards_raw - mean_grouped_rewards.unsqueeze(1)
                if bool(getattr(self, "scale_rewards", False)):
                    grouped_base_advantages = grouped_base_advantages / (
                        std_grouped_rewards.unsqueeze(1) + 1e-4
                    )
                base_advantages = _flatten_prompt_groups(grouped_base_advantages)
            # Apply the MaxEnt term after base GRPO advantage scaling so the
            # entropy signal is not attenuated/amplified by reward std.
            # Build a per-sample normalized novelty signal and gate it with
            # the configured correctness/signal mask:
            #   adv_i += alpha * 1[mask_i] * zscore_group(novelty_i)
            novelty_centered = grouped_novelty - grouped_novelty.mean(
                dim=1, keepdim=True
            )
            novelty_spread = grouped_novelty.std(dim=1, unbiased=False, keepdim=True)
            novelty_zscore = novelty_centered / (novelty_spread + 1e-4)
            reward_bonus_grouped = alpha * novelty_zscore * bonus_mask * cusp_factor
            reward_bonus = _flatten_prompt_groups(reward_bonus_grouped)
            all_advantages = base_advantages + reward_bonus
            reward_bonus_abs_mean = float(reward_bonus.abs().mean().item())
            reward_bonus_std = float(reward_bonus.std(unbiased=False).item())
            reward_bonus_nonzero_frac = float(
                (reward_bonus.abs() > 1e-8).to(torch.float32).mean().item()
            )
            base_adv_abs_mean = float(base_advantages.abs().mean().item())
            final_adv_abs_mean = float(all_advantages.abs().mean().item())
            flow_bonus_to_base_ratio = reward_bonus_abs_mean / max(base_adv_abs_mean, 1e-8)
            flow_bonus_to_final_ratio = reward_bonus_abs_mean / max(final_adv_abs_mean, 1e-8)
            flow_final_minus_base = all_advantages - base_advantages

            local_batch_size = int(advantages.size(0))
            start = int(getattr(self.accelerator, "process_index", 0) or 0) * local_batch_size
            end = start + local_batch_size
            local_adv_before = advantages.detach().to(torch.float32)
            local_advantages = all_advantages[start:end].to(
                device=advantages.device,
                dtype=advantages.dtype,
            )
            if local_advantages.numel() != advantages.numel():
                return
            local_adv_after = local_advantages.detach().to(torch.float32)
            local_adv_delta = local_adv_after - local_adv_before
            outputs["advantages"] = local_advantages

            metric_store = self._metrics[mode]
            reward_val = float(mean_grouped_rewards.mean().item())
            reward_std_val = float(std_grouped_rewards.mean().item())
            reward_zero_std = float(is_std_zero.float().mean().item())
            for key, value in (
                ("reward", reward_val),
                ("reward_std", reward_std_val),
                ("frac_reward_zero_std", reward_zero_std),
            ):
                bucket = metric_store.get(key)
                if isinstance(bucket, list) and bucket:
                    bucket[-1] = value
                else:
                    metric_store.setdefault(key, []).append(value)

            self._append_metric_value(mode, "maxent/alpha", alpha)
            self._append_metric_value(
                mode,
                "maxent/base_advantages_from_outputs",
                1.0 if using_native_advantages else 0.0,
            )
            self._append_metric_value(
                mode,
                "maxent/alpha_base",
                float(self.maxent_alpha),
            )
            self._append_metric_value(
                mode,
                "maxent/alpha_multiplier",
                alpha_multiplier,
            )
            self._append_metric_value(
                mode,
                "maxent/alpha_low_kl_enabled",
                1.0 if alpha_low_kl_enabled else 0.0,
            )
            self._append_metric_value(
                mode,
                "maxent/alpha_high_kl_enabled",
                1.0 if alpha_high_kl_enabled else 0.0,
            )
            self._append_metric_value(
                mode,
                "maxent/alpha_kl_control_enabled",
                1.0 if alpha_kl_control_enabled else 0.0,
            )
            self._append_metric_value(
                mode,
                "maxent/alpha_kl_direction",
                alpha_direction,
            )
            self._append_metric_value(
                mode,
                "maxent/alpha_kl_threshold",
                alpha_kl_threshold,
            )
            self._append_metric_value(
                mode,
                "maxent/alpha_kl_min_multiplier",
                alpha_kl_min_multiplier,
            )
            self._append_metric_value(
                mode,
                "maxent/alpha_kl_max_multiplier",
                alpha_kl_max_multiplier,
            )
            if alpha_kl_measure is not None:
                self._append_metric_value(
                    mode,
                    "maxent/alpha_kl_measure",
                    alpha_kl_measure,
                )
            self._append_metric_value(
                mode,
                "maxent/reward_signal_gate",
                1.0 if gate_by_signal else 0.0,
            )
            self._append_metric_value(
                mode,
                "maxent/positive_only",
                1.0 if positive_only else 0.0,
            )
            self._append_metric_value(
                mode,
                "maxent/cusp_gate",
                1.0 if cusp_gate_enabled else 0.0,
            )
            self._append_metric_value(
                mode,
                "maxent/cusp_threshold",
                cusp_threshold,
            )
            self._append_metric_value(
                mode,
                "maxent/cusp_factor_mean",
                float(cusp_factor.mean().item()),
            )
            self._append_metric_value(
                mode,
                "maxent/cusp_success_rate_mean",
                float(group_success_rate.mean().item()),
            )
            self._append_metric_value(
                mode,
                "maxent/eligible_group_frac",
                float(eligible_groups.float().mean().item()),
            )
            self._append_metric_value(
                mode,
                "maxent/eligible_completion_frac",
                float(reward_bonus_mask.float().mean().item()),
            )
            self._append_metric_value(
                mode,
                "maxent/policy_surprisal_mean",
                float(maxent_term.mean().item()),
            )
            self._append_metric_value(
                mode,
                "maxent/policy_surprisal_std",
                float(maxent_term.std(unbiased=False).item()),
            )
            self._append_metric_value(
                mode,
                "maxent/ref_novelty_mean",
                float(maxent_term.mean().item()),
            )
            self._append_metric_value(
                mode,
                "maxent/ref_novelty_std",
                float(maxent_term.std(unbiased=False).item()),
            )
            self._append_metric_value(
                mode,
                "maxent/reward_bonus_mean",
                float(reward_bonus.mean().item()),
            )
            self._append_metric_value(
                mode,
                "maxent/reward_bonus_abs_mean",
                reward_bonus_abs_mean,
            )
            self._append_metric_value(
                mode,
                "maxent/reward_bonus_std",
                reward_bonus_std,
            )
            self._append_metric_value(
                mode,
                "maxent/reward_bonus_nonzero_frac",
                reward_bonus_nonzero_frac,
            )
            self._append_metric_value(
                mode,
                "maxent/flow/base_adv_mean",
                float(base_advantages.mean().item()),
            )
            self._append_metric_value(
                mode,
                "maxent/flow/base_adv_std",
                float(base_advantages.std(unbiased=False).item()),
            )
            self._append_metric_value(
                mode,
                "maxent/flow/base_adv_abs_mean",
                base_adv_abs_mean,
            )
            self._append_metric_value(
                mode,
                "maxent/flow/final_adv_mean",
                float(all_advantages.mean().item()),
            )
            self._append_metric_value(
                mode,
                "maxent/flow/final_adv_std",
                float(all_advantages.std(unbiased=False).item()),
            )
            self._append_metric_value(
                mode,
                "maxent/flow/final_adv_abs_mean",
                final_adv_abs_mean,
            )
            self._append_metric_value(
                mode,
                "maxent/flow/final_minus_base_mean",
                float(flow_final_minus_base.mean().item()),
            )
            self._append_metric_value(
                mode,
                "maxent/flow/final_minus_base_abs_mean",
                float(flow_final_minus_base.abs().mean().item()),
            )
            self._append_metric_value(
                mode,
                "maxent/flow/bonus_to_base_abs_ratio",
                flow_bonus_to_base_ratio,
            )
            self._append_metric_value(
                mode,
                "maxent/flow/bonus_to_final_abs_ratio",
                flow_bonus_to_final_ratio,
            )
            self._append_metric_value(
                mode,
                "maxent/flow/local_adv_before_mean",
                float(local_adv_before.mean().item()),
            )
            self._append_metric_value(
                mode,
                "maxent/flow/local_adv_after_mean",
                float(local_adv_after.mean().item()),
            )
            self._append_metric_value(
                mode,
                "maxent/flow/local_adv_delta_mean",
                float(local_adv_delta.mean().item()),
            )
            self._append_metric_value(
                mode,
                "maxent/flow/local_adv_delta_abs_mean",
                float(local_adv_delta.abs().mean().item()),
            )
            self._append_metric_value(
                mode,
                "maxent/flow/local_adv_delta_nonzero_frac",
                float(
                    (local_adv_delta.abs() > 1e-8).to(torch.float32).mean().item()
                ),
            )
            prev_loss = _numeric_or_none(getattr(self, "_last_loss_scalar", None))
            if prev_loss is not None:
                self._append_metric_value(
                    mode,
                    "maxent/flow/prev_loss_total",
                    float(prev_loss),
                )

            advantage_log = self._textual_logs.get("advantages")
            if isinstance(advantage_log, list):
                updated = all_advantages.detach().cpu().tolist()
                if len(updated) <= len(advantage_log):
                    advantage_log[-len(updated) :] = updated

            if mode == "train" and self.accelerator.is_main_process:
                step = int(getattr(self.state, "global_step", 0))
                interval_raw = getattr(args, "maxent_flow_log_interval", 25) if args else 25
                try:
                    interval = int(interval_raw)
                except (TypeError, ValueError):
                    interval = 25
                if interval <= 0:
                    interval = 25
                should_emit = (
                    self._last_maxent_flow_log_step != step and step % interval == 0
                )
                if should_emit:
                    self._last_maxent_flow_log_step = step
                    LOG.info(
                        "MaxEnt flow | step=%d | alpha=%.6f (x%.3f) | kl_for_alpha=%s | reward_mean=%.6f reward_std=%.6f | "
                        "bonus_abs_mean=%.6f bonus_nonzero_frac=%.3f | base_adv_abs_mean=%.6f final_adv_abs_mean=%.6f | prev_loss=%s",
                        step,
                        alpha,
                        alpha_multiplier,
                        alpha_kl_measure,
                        reward_val,
                        reward_std_val,
                        reward_bonus_abs_mean,
                        reward_bonus_nonzero_frac,
                        base_adv_abs_mean,
                        final_adv_abs_mean,
                        prev_loss,
                    )
                if (
                    alpha > 0.0
                    and float(reward_bonus_mask.float().mean().item()) > 0.0
                    and float(cusp_factor.mean().item()) > 0.0
                    and float(maxent_term.std(unbiased=False).item()) > 1e-6
                    and reward_bonus_abs_mean <= 1e-8
                ):
                    LOG.warning(
                        "MaxEnt flow anomaly | step=%d | alpha=%.6f but reward_bonus_abs_mean is near zero despite active masks/gates.",
                        step,
                        alpha,
                    )

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

        def compute_loss(  # type: ignore[override]
            self,
            model: Any,
            inputs: Any,
            return_outputs: bool = False,
            num_items_in_batch: Any = None,
        ) -> Any:
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
                kl_history = train_metrics.get("kl") if isinstance(train_metrics, dict) else None
                if isinstance(kl_history, list) and kl_history:
                    kl_value = _numeric_or_none(kl_history[-1])
                    if kl_value is not None:
                        setattr(self, "_last_train_kl_for_alpha", float(kl_value))
                self._maybe_update_reference_model_ema()
            return loss

        def _generate_and_score_completions(  # type: ignore[override]
            self, inputs: List[Dict[str, Any]]
        ) -> Dict[str, Any]:
            outputs = super()._generate_and_score_completions(inputs)
            mode = "train" if self.model.training else "eval"
            self._log_grpo_diversity(outputs, mode=mode)
            self._log_eval_pass_at_k(inputs, outputs, mode=mode)
            self._apply_maxent_reward_shaping(inputs, outputs, mode=mode)
            if not self.maxent_enabled:
                self._log_grpo_debug(inputs, outputs, mode=mode)
            return outputs

    CustomGRPOTrainer.__name__ = "CustomGRPOTrainer"
    return ensure_weighting_logging(CustomGRPOTrainer)


def wrap_trl_trainer(trainer_cls: Type[Any]) -> Type[Any]:
    """Ensure a trainer class emits TRL-style logs and metrics."""

    return ensure_weighting_logging(trainer_cls)


__all__ = ["build_custom_grpo_trainer", "wrap_trl_trainer"]
