"""Shared helpers for building custom MaxEnt/InfoSeed training loops."""

from __future__ import annotations

import copy
import logging
import math
import os
from collections.abc import Iterable
from dataclasses import FrozenInstanceError, MISSING, fields
from functools import lru_cache
from types import SimpleNamespace
from typing import Any, Dict, Optional, Callable, Tuple, cast

from maxent_grpo.config import GRPOConfig, GRPOScriptArguments
from maxent_grpo.core.model import get_model, get_tokenizer
from maxent_grpo.training import (
    apply_info_seed,
    GenerationSettings,
    EvaluationSettings,
    OptimizationSettings,
    OptimizationSchedule,
    RewardSpec,
    RuntimeHandles,
    ScoringSettings,
    ClipSettings,
    BatchingSettings,
    LoopSettings,
    TrainingLoopContext,
)
from maxent_grpo.training.data import load_datasets, resolve_dataloader_kwargs
from maxent_grpo.training.rewards import (
    load_reward_functions,
    load_eval_reward_functions,
)
from maxent_grpo.training.runtime.config import VLLMClientConfig
from maxent_grpo.training.runtime.deps import require_accelerator, require_dataloader
from maxent_grpo.training.runtime.prompts import GenerationPenaltyConfig
from maxent_grpo.training.runtime.logging import _maybe_init_wandb_run
from maxent_grpo.training.controller_objective import build_controller_objective
from maxent_grpo.training.controller_optimizer import ControllerMetaManager
from maxent_grpo.training.optim import detect_deepspeed_state
from maxent_grpo.utils.deps_guard import ensure_real_dependencies
from maxent_grpo.training.state import (
    build_checkpoint_saver,
    build_training_state,
    load_trainer_state_metadata,
    resolve_resume_checkpoint,
)
from maxent_grpo.training.types import ControllerPaths, PromptCacheEntry
from maxent_grpo.training.weighting.logic import build_weighting_settings
from maxent_grpo.training.weighting.types import WeightingSettings

LOG = logging.getLogger(__name__)

_MIN_RETRY_SLEEP = 0.0
_MAX_RETRY_SLEEP = 3600.0
_MIN_BACKOFF = 0.0
_MAX_BACKOFF = 3600.0
_MIN_BACKOFF_MULT = 1.0
_MAX_BACKOFF_MULT = 10.0
_MIN_MAX_RETRIES = 1
_MAX_MAX_RETRIES = 50
_PROMPT_CACHE_MIN = 10000
_PROMPT_CACHE_MAX = 50000
_PROMPT_CACHE_SCALE = 250


def _default_prompt_cache_size() -> int:
    """Return the declared default for maxent_prompt_cache_size."""

    try:
        for field in fields(GRPOConfig):
            if field.name == "maxent_prompt_cache_size":
                if field.default is not MISSING:
                    return int(field.default)
                if field.default_factory is not MISSING:
                    return int(field.default_factory())
                break
    except (TypeError, ValueError) as exc:
        LOG.debug("Unable to read default maxent_prompt_cache_size: %s", exc)
    return _PROMPT_CACHE_MIN


_PROMPT_CACHE_DEFAULT = _default_prompt_cache_size()


def _resolve_prompt_cache_size(cfg: GRPOConfig) -> int:
    """Return a prompt cache size with adaptive defaults."""

    raw = getattr(cfg, "maxent_prompt_cache_size", _PROMPT_CACHE_DEFAULT)
    try:
        requested = int(raw or 0)
    except (TypeError, ValueError):
        LOG.warning("Invalid maxent_prompt_cache_size=%s; disabling cache.", raw)
        return 0
    if requested <= 0:
        return 0
    if requested != _PROMPT_CACHE_DEFAULT:
        return requested
    bsz = int(getattr(cfg, "per_device_train_batch_size", 1) or 1)
    gens = int(getattr(cfg, "num_generations", 1) or 1)
    derived = bsz * max(1, gens) * _PROMPT_CACHE_SCALE
    resolved = max(_PROMPT_CACHE_MIN, min(_PROMPT_CACHE_MAX, derived))
    if resolved != requested:
        LOG.debug(
            "Auto-sized prompt cache | requested=%d | resolved=%d | bsz=%d | gens=%d",
            requested,
            resolved,
            bsz,
            gens,
        )
    return resolved


def _build_prompt_cache_fn(
    tokenizer: Any,
    max_prompt_len: int,
    cache_size: int,
) -> Optional[Callable[[str], PromptCacheEntry]]:
    """Return a callable that tokenizes prompts once and caches the results."""

    if tokenizer is None:
        return None
    model_max_len = getattr(tokenizer, "model_max_length", 0) or 0
    effective_len = int(max_prompt_len or 0)
    if effective_len <= 0:
        effective_len = model_max_len if model_max_len > 0 else 1
    elif model_max_len > 0:
        effective_len = min(effective_len, int(model_max_len))

    def _encode(prompt: str) -> PromptCacheEntry:
        call_kwargs = dict(
            truncation=True,
            max_length=effective_len,
            padding=False,
            add_special_tokens=False,
            return_attention_mask=True,
        )
        try:
            encoding = tokenizer(prompt, **call_kwargs)
        except TypeError:
            # Some lightweight test tokenizers do not accept attention/max kwargs.
            fallback_kwargs = {
                key: value
                for key, value in call_kwargs.items()
                if key not in {"max_length", "return_attention_mask"}
            }
            encoding = tokenizer(prompt, **fallback_kwargs)
        ids = encoding.get("input_ids")
        if ids is None:
            ids = []
        attn = encoding.get("attention_mask")

        def _normalize_tokens(value):
            if hasattr(value, "tolist"):
                try:
                    value = value.tolist()
                except (TypeError, ValueError, AttributeError, RuntimeError) as exc:
                    LOG.debug(
                        "Prompt cache token normalization failed; keeping raw tokens: %s",
                        exc,
                    )
            if isinstance(value, list) and value and isinstance(value[0], list):
                return value[0]
            return value

        ids = _normalize_tokens(ids)
        if attn is None:
            attn = [1] * len(ids)
        else:
            attn = _normalize_tokens(attn)
        return PromptCacheEntry(
            input_ids=list(ids),
            attention_mask=list(attn),
        )

    if cache_size and cache_size > 0:
        cached_encode = lru_cache(maxsize=cache_size)(_encode)

        def _cached(prompt: str) -> PromptCacheEntry:
            return cached_encode(prompt)

        return _cached
    return _encode


def _clone_model_args(model_args: Any) -> Optional[Any]:
    """Return a deep clone of the model config used for reference loading."""

    if model_args is None:
        return None
    try:
        return copy.deepcopy(model_args)
    except (TypeError, ValueError, AttributeError, RuntimeError) as exc:
        LOG.debug("Unable to deepcopy model args; falling back to __dict__: %s", exc)
    attrs = getattr(model_args, "__dict__", None)
    if attrs is not None:
        return SimpleNamespace(**attrs)
    return None


def _prepare_reference_model(model: Any, device: Any) -> Any:
    """Freeze the reference model and move it onto the target device."""

    if model is None:
        return None
    eval_fn = getattr(model, "eval", None)
    if callable(eval_fn):
        try:
            eval_fn()
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            LOG.debug("Reference model eval() failed; continuing: %s", exc)
    requires_grad_fn = getattr(model, "requires_grad_", None)
    if callable(requires_grad_fn):
        try:
            requires_grad_fn(False)
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            LOG.debug("Reference model requires_grad_(False) failed: %s", exc)
    else:
        params = getattr(model, "parameters", None)
        if callable(params):
            try:
                params_iter = params()
                if isinstance(params_iter, Iterable):
                    for param in params_iter:
                        requires_grad = getattr(param, "requires_grad", True)
                        if requires_grad and hasattr(param, "requires_grad_"):
                            param.requires_grad_(False)
            except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
                LOG.debug("Reference model parameter freeze failed: %s", exc)
    move_fn = getattr(model, "to", None)
    if callable(move_fn) and device is not None:
        try:
            moved = move_fn(device)
            if moved is not None:
                model = moved
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            LOG.debug("Reference model device move failed; using original: %s", exc)
    return model

def _seed_generation_metadata(
    stats: Dict[str, Any], cfg: GRPOConfig
) -> Dict[str, Any]:
    """Populate dataset/model metadata for downstream telemetry."""

    dataset_label = getattr(cfg, "dataset_name", None)
    if not dataset_label:
        mixture = getattr(cfg, "dataset_mixture", None)
        if mixture:
            dataset_label = str(mixture)
    if dataset_label:
        stats.setdefault("dataset_name", dataset_label)
    model_label = getattr(cfg, "hub_model_id", None) or getattr(
        cfg, "model_name_or_path", None
    )
    if model_label:
        stats.setdefault("model_id", model_label)
    return stats


def _sanitize_float(
    value: Any,
    *,
    default: float,
    minimum: float,
    maximum: float,
    name: str,
) -> float:
    """Ensure floating-point configuration values stay within safe bounds."""

    if value is None:
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        LOG.warning(
            "Invalid %s=%s; falling back to default %.3f", name, value, default
        )
        return default
    clamped = parsed
    if parsed < minimum:
        clamped = minimum
    elif parsed > maximum:
        clamped = maximum
    if clamped != parsed:
        LOG.warning(
            "Clamped %s from %.3f to %.3f (allowed %.3f–%.3f)",
            name,
            parsed,
            clamped,
            minimum,
            maximum,
        )
    return clamped


def _sanitize_int(
    value: Any,
    *,
    default: int,
    minimum: int,
    maximum: int,
    name: str,
) -> int:
    """Clamp integer configuration values to sensible ranges."""

    if value is None:
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        LOG.warning(
            "Invalid %s=%s; falling back to default %d", name, value, default
        )
        return default
    clamped = parsed
    if parsed < minimum:
        clamped = minimum
    elif parsed > maximum:
        clamped = maximum
    if clamped != parsed:
        LOG.warning(
            "Clamped %s from %d to %d (allowed %d–%d)",
            name,
            parsed,
            clamped,
            minimum,
            maximum,
        )
    return clamped


def _fallback_optim_handles(training_args: GRPOConfig) -> SimpleNamespace:
    lr = float(getattr(training_args, "learning_rate", 0.0))

    class _DummyOpt:
        def __init__(self, base_lr: float):
            self.param_groups = [{"lr": base_lr}]

        def step(self):
            return None

        def zero_grad(self, set_to_none: bool = True):
            _ = set_to_none
            return None

        def state_dict(self):
            return {}

        def load_state_dict(self, _state):
            return None

    dummy_opt = _DummyOpt(lr)
    return SimpleNamespace(
        optimizer=dummy_opt,
        lr_scheduler=None,
        base_optimizer=dummy_opt,
        learning_rate=lr,
    )


def _build_optim_handles(model: Any, training_args: GRPOConfig) -> Any:
    try:
        from maxent_grpo.training.optim import build_optimization_handles
    except ImportError:  # pragma: no cover - optional dependency
        return _fallback_optim_handles(training_args)
    try:
        return build_optimization_handles(model, training_args)
    except ImportError:  # pragma: no cover - optional dependency
        return _fallback_optim_handles(training_args)


def _controller_paths(training_args: GRPOConfig) -> ControllerPaths:
    state_path = getattr(training_args, "controller_state_path", None)
    if not state_path:
        output_dir = getattr(training_args, "output_dir", None)
        if isinstance(output_dir, str) and output_dir:
            os.makedirs(output_dir, exist_ok=True)
            state_path = os.path.join(output_dir, "controller_state.json")
    resume_from = getattr(training_args, "controller_resume_from", None)
    return ControllerPaths(
        state_path=state_path,
        resume_from=resume_from,
        overwrite_existing=bool(getattr(training_args, "overwrite_output_dir", False)),
    )


def _maybe_build_distributed_sampler(
    dataset: Any, training_args: GRPOConfig, accelerator: Any
) -> Optional[Any]:
    disable_sampler = bool(getattr(training_args, "disable_distributed_sampler", False))
    if disable_sampler:
        return None
    num_processes = getattr(accelerator, "num_processes", 1)
    if num_processes <= 1:
        return None
    if not hasattr(dataset, "__len__"):
        return None
    try:
        dataset_len = len(dataset)
    except (AttributeError, TypeError, OSError):
        return None
    if not dataset_len:
        return None
    try:
        from torch.utils.data.distributed import DistributedSampler
    except ImportError:  # pragma: no cover - depends on torch availability
        return None
    return DistributedSampler(dataset, shuffle=True)


def _prepare_runtime_components(
    accelerator: Any,
    model: Any,
    optimizer: Any,
    train_loader: Any,
    lr_scheduler: Optional[Any],
) -> Tuple[Any, Any, Any, Optional[Any]]:
    """Wrap core training objects with ``accelerator.prepare`` when available."""

    prepare_fn = getattr(accelerator, "prepare", None)
    if not callable(prepare_fn):
        return model, optimizer, train_loader, lr_scheduler
    prepare_args = [model, optimizer, train_loader]
    include_scheduler = lr_scheduler is not None
    if include_scheduler:
        prepare_args.append(lr_scheduler)
    prepared = prepare_fn(*prepare_args)
    if isinstance(prepared, (list, tuple)):
        prepared_items = list(prepared)
    else:
        prepared_items = [prepared]
    if len(prepared_items) != len(prepare_args):
        LOG.warning(
            "accelerator.prepare returned %d objects for %d inputs; skipping wrapping.",
            len(prepared_items),
            len(prepare_args),
        )
        return model, optimizer, train_loader, lr_scheduler
    prepared_model = prepared_items[0]
    prepared_optimizer = prepared_items[1]
    prepared_loader = prepared_items[2]
    prepared_scheduler: Optional[Any] = lr_scheduler
    if include_scheduler:
        prepared_scheduler = prepared_items[3]
    return prepared_model, prepared_optimizer, prepared_loader, prepared_scheduler


def _loader_steps(loader: Any) -> Optional[int]:
    try:
        return len(loader)
    except (TypeError, AttributeError):
        return None


def build_generation_settings(cfg: GRPOConfig) -> GenerationSettings:
    """Construct generation settings for the custom training loop.

    :param cfg: Training configuration carrying generation/vLLM knobs.
    :type cfg: GRPOConfig
    :returns: Generation settings with vLLM client config and stats initialized.
    :rtype: GenerationSettings
    """
    retry_sleep = _sanitize_float(
        getattr(cfg, "vllm_retry_sleep", 1.0),
        default=1.0,
        minimum=_MIN_RETRY_SLEEP,
        maximum=_MAX_RETRY_SLEEP,
        name="vllm_retry_sleep",
    )
    backoff = _sanitize_float(
        getattr(cfg, "vllm_backoff", 1.0),
        default=1.0,
        minimum=_MIN_BACKOFF,
        maximum=_MAX_BACKOFF,
        name="vllm_backoff",
    )
    backoff_multiplier = _sanitize_float(
        getattr(cfg, "vllm_backoff_multiplier", 2.0),
        default=2.0,
        minimum=_MIN_BACKOFF_MULT,
        maximum=_MAX_BACKOFF_MULT,
        name="vllm_backoff_multiplier",
    )
    max_retries = _sanitize_int(
        getattr(cfg, "vllm_max_retries", 3),
        default=3,
        minimum=_MIN_MAX_RETRIES,
        maximum=_MAX_MAX_RETRIES,
        name="vllm_max_retries",
    )
    vllm_cfg = VLLMClientConfig(
        url=getattr(cfg, "vllm_url", ""),
        rounds_cfg=getattr(
            cfg,
            "vllm_max_completion_rounds",
            getattr(cfg, "vllm_rounds_cfg", 0),
        ),
        retry_sleep=retry_sleep,
        backfill_local=bool(
            getattr(
                cfg,
                "vllm_backfill_with_model",
                getattr(cfg, "vllm_backfill_local", False),
            )
        ),
        request_logprobs=bool(
            getattr(cfg, "vllm_return_logprobs", getattr(cfg, "vllm_request_logprobs", True))
        ),
        best_of=getattr(cfg, "gen_best_of", None),
        frequency_penalty=float(getattr(cfg, "gen_frequency_penalty", 0.0)),
        presence_penalty=float(getattr(cfg, "gen_presence_penalty", 0.0)),
        top_k=getattr(cfg, "gen_top_k", None),
        stop_sequences=getattr(cfg, "vllm_stop_sequences", None),
        timeout=float(
            getattr(cfg, "vllm_server_timeout", getattr(cfg, "vllm_request_timeout", 120.0))
        ),
        max_retries=max_retries,
        backoff=backoff,
        backoff_multiplier=backoff_multiplier,
        guided_json=getattr(cfg, "vllm_guided_json", None),
        guided_regex=getattr(cfg, "vllm_guided_regex", None),
        logit_bias=getattr(cfg, "vllm_logit_bias", None),
        request_id_prefix=getattr(cfg, "vllm_request_id_prefix", None),
        sync_weights=bool(getattr(cfg, "vllm_sync_weights", False)),
    )
    settings = GenerationSettings(
        max_prompt_len=int(getattr(cfg, "max_prompt_length", 0) or 0),
        max_completion_len=int(getattr(cfg, "max_completion_length", 0) or 0),
        gen_temperature=cfg.gen_temperature,
        gen_top_p=cfg.gen_top_p,
        use_vllm=cfg.use_vllm,
        vllm=vllm_cfg,
        penalty=GenerationPenaltyConfig(),
    )
    stats = _seed_generation_metadata(settings.generation_stats, cfg)
    for key in (
        "vllm_retry_rounds",
        "vllm_backfilled_prompts",
        "vllm_failed_prompts",
        "dropped_prompts",
        "partial_prompts",
        "vllm_excess_prompts",
        "vllm_excess_completions",
    ):
        stats.setdefault(key, 0)
    settings.generation_stats = stats
    return settings


def build_scoring_settings(
    cfg: GRPOConfig, weighting: Optional[WeightingSettings] = None
) -> ScoringSettings:
    """Build scoring settings including weighting/clipping configuration.

    :param cfg: Training configuration providing scoring-related fields.
    :type cfg: GRPOConfig
    :param weighting: Optional precomputed weighting settings; built when absent.
    :type weighting: WeightingSettings | None
    :returns: Scoring settings object used by the custom loop.
    :rtype: ScoringSettings
    """
    if weighting is None:
        weighting = build_weighting_settings(cfg)

    clip_range = _sanitize_float(
        getattr(cfg, "ppo_clip_range", 0.0),
        default=0.0,
        minimum=0.0,
        maximum=1.0,
        name="ppo_clip_range",
    )
    use_clip_objective = bool(
        getattr(cfg, "maxent_use_clip_objective", False) or clip_range > 0.0
    )
    if clip_range > 0.0 and not getattr(cfg, "maxent_use_clip_objective", False):
        LOG.info(
            "Enabling clip objective in custom loop because ppo_clip_range=%.3f was set.",
            clip_range,
        )
    score_tail_tokens = getattr(cfg, "maxent_score_tail_tokens", None)
    if score_tail_tokens is not None and score_tail_tokens <= 0:
        score_tail_tokens = None
    if score_tail_tokens is not None and not weighting.len_norm_ref:
        LOG.info(
            "Ignoring maxent_score_tail_tokens=%s because length-normalized reference scoring is disabled.",
            score_tail_tokens,
        )
        score_tail_tokens = None
    batching = BatchingSettings(
        logprob_chunk_size=cfg.maxent_logprob_chunk_size,
        score_slice=cfg.maxent_logprob_chunk_size,
        prompt_length_cache_get=None,
        score_tail_tokens=score_tail_tokens,
        slice_prefetch=getattr(cfg, "maxent_score_slice_prefetch", 0),
        prompt_cache_size=_resolve_prompt_cache_size(cfg),
    )
    clipping = ClipSettings(
        clip_range=clip_range,
        use_clip_objective=use_clip_objective,
        clip_objective_coef=float(getattr(cfg, "maxent_clip_objective_coef", 1.0)),
        clip_adv_baseline=getattr(cfg, "maxent_clip_adv_baseline", None),
    )
    ref_source = str(
        getattr(cfg, "maxent_reference_logprobs_source", "auto") or "auto"
    )
    if not getattr(cfg, "train_grpo_objective", True) and ref_source.strip().lower() == "auto":
        LOG.info(
            "MaxEnt run detected (train_grpo_objective=false); forcing "
            "maxent_reference_logprobs_source=model to use the frozen reference model."
        )
        ref_source = "model"
    return ScoringSettings(
        weighting=weighting,
        clipping=clipping,
        batching=batching,
        reference_logprobs_source=ref_source,
        allow_stale_reference_logprobs=bool(
            getattr(cfg, "maxent_allow_stale_reference_logprobs", False)
        ),
    )


def build_evaluation_settings(cfg: GRPOConfig) -> EvaluationSettings:
    """Return evaluation settings derived from training arguments.

    :param cfg: Training configuration containing eval flags and cadence.
    :type cfg: GRPOConfig
    :returns: Evaluation settings for the custom loop.
    :rtype: EvaluationSettings
    """
    strategy = str(getattr(cfg, "eval_strategy", "") or "").lower()
    eval_steps = getattr(cfg, "eval_steps", None)
    enabled = cfg.do_eval
    every_n_steps = eval_steps
    if strategy:
        if strategy in {"steps", "step"}:
            enabled = True
            if not eval_steps or int(eval_steps) <= 0:
                LOG.warning(
                    "eval_strategy=steps but eval_steps is not set; disabling eval."
                )
                enabled = False
        elif strategy in {"no", "none", "off"}:
            enabled = False
        elif strategy in {"epoch", "epochs"}:
            LOG.warning("eval_strategy=epoch is not supported in custom loop; skipping eval.")
            enabled = False
    if enabled and every_n_steps is None:
        every_n_steps = getattr(cfg, "logging_steps", None)
    if enabled and (every_n_steps is None or every_n_steps <= 0):
        enabled = False
    rows = getattr(cfg, "eval_rows", []) if hasattr(cfg, "eval_rows") else []
    return EvaluationSettings(
        enabled=enabled,
        rows=rows,
        batch_size=cfg.per_device_eval_batch_size,
        every_n_steps=every_n_steps,
    )


def build_training_loop_context(
    script_args: GRPOScriptArguments,
    training_args: GRPOConfig,
    model_args: Any,
    *,
    deps_namespace: str,
    apply_info_seed_cfg: bool,
    force_grpo_objective: Optional[bool],
) -> TrainingLoopContext:
    """Return a TrainingLoopContext configured for the custom MaxEnt loop.

    :param script_args: Script configuration with dataset/reward options.
    :type script_args: GRPOScriptArguments
    :param training_args: Training configuration (GRPO/MaxEnt knobs).
    :type training_args: GRPOConfig
    :param model_args: Model configuration used for loading policy/reference models.
    :type model_args: Any
    :param deps_namespace: Namespace hint for dependency resolution/logging.
    :type deps_namespace: str
    :param apply_info_seed_cfg: Whether to apply InfoSeed config to settings.
    :type apply_info_seed_cfg: bool
    :param force_grpo_objective: Optional override for ``train_grpo_objective``.
    :type force_grpo_objective: bool | None
    :returns: Fully constructed training loop context.
    :rtype: TrainingLoopContext
    """

    share_reference_model = bool(
        getattr(training_args, "maxent_share_reference_model", False)
    )
    reference_model_args = None
    if not share_reference_model:
        reference_model_args = _clone_model_args(model_args)
        if reference_model_args is None:
            share_reference_model = True
        else:
            ref_name = getattr(training_args, "reference_model_name_or_path", None)
            if ref_name:
                setattr(reference_model_args, "model_name_or_path", ref_name)
            ref_revision = getattr(training_args, "reference_model_revision", None)
            if ref_revision is not None:
                setattr(reference_model_args, "model_revision", ref_revision)
    if force_grpo_objective is not None:
        training_args.train_grpo_objective = force_grpo_objective
    resume_checkpoint, resume_requested = resolve_resume_checkpoint(training_args)
    resume_state = load_trainer_state_metadata(resume_checkpoint)
    if resume_checkpoint:
        LOG.info("Resuming from checkpoint %s", resume_checkpoint)
        # When resuming, prefer restoring model weights via `accelerator.load_state()`
        # (called later in the training loop) instead of re-initializing the model
        # from `resume_checkpoint`. Older ZeRO-3/FSDP checkpoints may not contain
        # consolidated HF weights loadable via `from_pretrained()`.
        try:
            training_args.resume_from_checkpoint = resume_checkpoint
        except (AttributeError, FrozenInstanceError, TypeError, ValueError):
            setattr(training_args, "resume_from_checkpoint", resume_checkpoint)
    elif resume_requested:
        try:
            training_args.resume_from_checkpoint = None
        except (AttributeError, FrozenInstanceError, TypeError, ValueError):
            setattr(training_args, "resume_from_checkpoint", None)
    accelerator_cls_or_obj = require_accelerator(deps_namespace)
    # Defensive: reset any lingering Accelerate shared state (e.g., from stubs or
    # prior imports) so the first Accelerator() call can safely set attributes.
    try:  # pragma: no cover - runtime guard
        from accelerate.state import AcceleratorState  # type: ignore[reportMissingTypeStubs]

        reset_fn = getattr(AcceleratorState, "_reset_state", None)
        if callable(reset_fn):
            reset_fn(reset_partial_state=True)
    except (ImportError, ModuleNotFoundError, AttributeError, TypeError, RuntimeError) as exc:
        LOG.debug("Unable to reset Accelerate shared state: %s", exc)

    def _init_accelerator(cls_or_obj: Any) -> Any:
        if not callable(cls_or_obj):
            return cls_or_obj
        try:
            return cls_or_obj(kwargs_handlers=None)
        except TypeError:
            # Older/newer Accelerate versions may not accept kwargs_handlers; retry bare.
            return cls_or_obj()

    accelerator = _init_accelerator(accelerator_cls_or_obj)
    dataloader_cls = require_dataloader(deps_namespace)
    model = get_model(model_args, training_args)
    reference_model = None
    if not share_reference_model and reference_model_args is not None:
        try:
            reference_model = get_model(reference_model_args, training_args)
        except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:  # pragma: no cover - defensive logging
            LOG.warning(
                "Unable to load frozen reference model; falling back to the live policy | error=%s",
                exc,
            )
            share_reference_model = True
    if reference_model is not None:
        reference_model = _prepare_reference_model(reference_model, accelerator.device)
    tokenizer = get_tokenizer(model_args, training_args)
    ensure_real_dependencies(
        context="MaxEnt-GRPO training loop",
        model=model,
        tokenizer=tokenizer,
    )
    train_dataset, eval_rows = load_datasets(
        script_args, training_args, tokenizer, accelerator=accelerator
    )
    setattr(training_args, "eval_rows", eval_rows)
    reward_funcs, reward_weights = load_reward_functions(
        script_args, tokenizer, training_args
    )
    reward_spec = RewardSpec(reward_funcs=reward_funcs, reward_weights=reward_weights)
    eval_reward_funcs, eval_reward_weights = load_eval_reward_functions(
        script_args, tokenizer, training_args
    )
    eval_reward_spec = RewardSpec(
        reward_funcs=eval_reward_funcs, reward_weights=eval_reward_weights
    )
    weighting = build_weighting_settings(training_args)
    controller_objective = build_controller_objective(training_args, weighting)
    controller_manager = ControllerMetaManager(training_args, weighting)
    if not controller_manager.enabled:
        controller_manager = None
    scoring = build_scoring_settings(training_args, weighting)
    generation = build_generation_settings(training_args)
    evaluation = build_evaluation_settings(training_args)
    if apply_info_seed_cfg and getattr(training_args, "info_seed_enabled", False):
        generation, scoring, evaluation = apply_info_seed(
            generation, scoring, evaluation, training_args
        )
    sampler = _maybe_build_distributed_sampler(
        train_dataset, training_args, accelerator
    )
    loader_kwargs = {
        "batch_size": training_args.per_device_train_batch_size,
        "shuffle": sampler is None,
    }
    loader_kwargs.update(resolve_dataloader_kwargs(training_args))
    if sampler is not None:
        loader_kwargs["sampler"] = sampler
    train_loader = dataloader_cls(train_dataset, **loader_kwargs)

    optim_handles = _build_optim_handles(model, training_args)
    (
        model,
        prepared_optimizer,
        train_loader,
        prepared_scheduler,
    ) = _prepare_runtime_components(
        accelerator,
        model,
        optim_handles.optimizer,
        train_loader,
        optim_handles.lr_scheduler,
    )
    optim_handles.optimizer = prepared_optimizer
    if prepared_scheduler is not None:
        optim_handles.lr_scheduler = prepared_scheduler

    def _resolve_ref_model():
        if reference_model is not None:
            return reference_model
        # Under DeepSpeed ZeRO (and some other wrappers), unwrapping returns the
        # underlying module with partitioned parameters. Reference scoring uses
        # forward passes and may invoke ZeRO gather contexts; using the wrapped
        # model keeps those interactions consistent with the training engine.
        ds_state = detect_deepspeed_state(accelerator)
        if ds_state.use_deepspeed or ds_state.zero_stage >= 2:
            return model
        unwrap = getattr(accelerator, "unwrap_model", None)
        if callable(unwrap):
            try:
                return unwrap(model)
            except (AttributeError, RuntimeError, TypeError, ValueError):
                return model
        return model

    runtime_handles = RuntimeHandles(
        accelerator=accelerator,
        model=model,
        tokenizer=cast(Any, tokenizer),
        train_loader=train_loader,
        train_sampler=sampler,
        device=accelerator.device,
        get_ref_model=cast(Callable[[], Any], _resolve_ref_model),
        reference_model=reference_model,
    )
    prompt_cache = _build_prompt_cache_fn(
        tokenizer,
        int(getattr(training_args, "max_prompt_length", 0) or 0),
        _resolve_prompt_cache_size(training_args),
    )
    if callable(prompt_cache):
        runtime_handles.prompt_cache_get = prompt_cache
        scoring.batching.prompt_length_cache_get = prompt_cache
    max_grad_norm = _sanitize_float(
        getattr(training_args, "max_grad_norm", 0.0),
        default=0.0,
        minimum=0.0,
        maximum=float("inf"),
        name="max_grad_norm",
    )
    num_epochs_raw = float(getattr(training_args, "num_train_epochs", 0) or 0)
    num_epochs = int(math.ceil(num_epochs_raw)) if num_epochs_raw > 0 else 0
    num_generations = int(getattr(training_args, "num_generations", 1) or 1)
    if num_generations <= 0:
        num_generations = 1
    max_steps_cfg = int(getattr(training_args, "max_steps", 0) or 0)
    warmup_limit = max_steps_cfg if max_steps_cfg > 0 else (1 << 31) - 1
    warmup_steps_cfg = _sanitize_int(
        getattr(training_args, "warmup_steps", 0),
        default=0,
        minimum=0,
        maximum=warmup_limit,
        name="warmup_steps",
    )
    steps_per_epoch = _loader_steps(train_loader)
    if max_steps_cfg > 0:
        total_training_steps = max_steps_cfg
    elif steps_per_epoch and num_epochs_raw > 0:
        total_training_steps = int(math.ceil(steps_per_epoch * num_epochs_raw))
    else:
        total_training_steps = 0
    warmup_steps = warmup_steps_cfg
    if warmup_steps <= 0 and total_training_steps > 0:
        warmup_ratio = float(getattr(training_args, "warmup_ratio", 0.0) or 0.0)
        warmup_steps = int(warmup_ratio * total_training_steps)
    if warmup_steps < 0:
        warmup_steps = 0
    if total_training_steps > 0 and warmup_steps > total_training_steps:
        warmup_steps = total_training_steps
    lr_scheduler_type = getattr(training_args, "lr_scheduler_type", None) or "cosine"
    schedule = OptimizationSchedule(
        num_epochs=num_epochs,
        num_generations=num_generations,
        grad_accum_steps=training_args.gradient_accumulation_steps,
        max_grad_norm=max_grad_norm,
        steps_per_epoch=steps_per_epoch,
        total_training_steps=int(total_training_steps or 0),
        warmup_steps=int(warmup_steps or 0),
        lr_scheduler_type=str(lr_scheduler_type),
    )
    optimization = OptimizationSettings(schedule=schedule, handles=optim_handles)
    controller = _controller_paths(training_args)
    logging_handles = build_training_state(training_args)
    checkpoint_state_ref: Dict[str, Any] = {"state": None}
    logging_handles.save_checkpoint = build_checkpoint_saver(
        training_args,
        runtime_handles,
        optim_handles,
        tokenizer,
        state_ref=checkpoint_state_ref,
        base_trainer_state=resume_state,
        controller_cfg=controller,
    )
    logging_handles.checkpoint_state_ref = checkpoint_state_ref
    loop_settings = LoopSettings(
        generation=generation,
        evaluation=evaluation,
        optimization=optimization,
        scoring=scoring,
        controller=controller,
        controller_objective=controller_objective,
        controller_meta_manager=controller_manager,
    )
    # Initialize W&B run if requested so downstream metrics logging is enabled.
    try:
        wandb_run = _maybe_init_wandb_run(
            accelerator,
            training_args,
            {"run/config_path": getattr(training_args, "recipe_path", None)},
        )
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:  # pragma: no cover - network/env guard
        LOG.warning("Unable to initialize W&B run: %s | continuing without wandb", exc)
        wandb_run = None
    logging_handles.wandb_run = wandb_run
    ctx = TrainingLoopContext(
        runtime=runtime_handles,
        reward=reward_spec,
        eval_reward=eval_reward_spec,
        settings=loop_settings,
        logging=logging_handles,
    )
    ctx.resume_checkpoint = resume_checkpoint
    ctx.resume_state = resume_state
    ctx.checkpoint_state_ref = checkpoint_state_ref
    ctx.training_args = training_args
    return ctx


__all__ = [
    "build_generation_settings",
    "build_scoring_settings",
    "build_evaluation_settings",
    "build_training_loop_context",
]
