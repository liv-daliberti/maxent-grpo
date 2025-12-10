"""Shared helpers for building custom MaxEnt/InfoSeed training loops."""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from types import SimpleNamespace
from typing import Any, Dict, Optional, Callable

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
from maxent_grpo.training.data import load_datasets
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
                except Exception:
                    pass
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


def _loader_steps(loader: Any) -> Optional[int]:
    try:
        return len(loader)
    except (TypeError, AttributeError):
        return None


def build_generation_settings(cfg: GRPOConfig) -> GenerationSettings:
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
        rounds_cfg=getattr(cfg, "vllm_rounds_cfg", 0),
        retry_sleep=retry_sleep,
        backfill_local=bool(getattr(cfg, "vllm_backfill_local", False)),
        request_logprobs=bool(getattr(cfg, "vllm_request_logprobs", True)),
        best_of=getattr(cfg, "gen_best_of", None),
        frequency_penalty=float(getattr(cfg, "gen_frequency_penalty", 0.0)),
        presence_penalty=float(getattr(cfg, "gen_presence_penalty", 0.0)),
        top_k=getattr(cfg, "gen_top_k", None),
        stop_sequences=getattr(cfg, "vllm_stop_sequences", None),
        timeout=float(getattr(cfg, "vllm_request_timeout", 120.0)),
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
        max_prompt_len=cfg.max_prompt_length,
        max_completion_len=cfg.max_completion_length,
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
    if weighting is None:
        weighting = build_weighting_settings(cfg)

    clip_range = _sanitize_float(
        getattr(cfg, "clip_range", 0.0),
        default=0.0,
        minimum=0.0,
        maximum=1.0,
        name="clip_range",
    )
    use_clip_objective = bool(
        getattr(cfg, "maxent_use_clip_objective", False) or clip_range > 0.0
    )
    if clip_range > 0.0 and not getattr(cfg, "maxent_use_clip_objective", False):
        LOG.info(
            "Enabling clip objective in custom loop because clip_range=%.3f was set.",
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
        prompt_cache_size=int(getattr(cfg, "maxent_prompt_cache_size", 0) or 0),
    )
    clipping = ClipSettings(
        clip_range=clip_range,
        use_clip_objective=use_clip_objective,
        clip_objective_coef=float(getattr(cfg, "maxent_clip_objective_coef", 1.0)),
        clip_adv_baseline=getattr(cfg, "maxent_clip_adv_baseline", None),
    )
    return ScoringSettings(weighting=weighting, clipping=clipping, batching=batching)


def build_evaluation_settings(cfg: GRPOConfig) -> EvaluationSettings:
    strategy = str(getattr(cfg, "evaluation_strategy", "") or "").lower()
    eval_steps = getattr(cfg, "eval_steps", None)
    enabled = cfg.do_eval
    every_n_steps = eval_steps
    if strategy:
        if strategy in {"steps", "step"}:
            enabled = True
            if not eval_steps or int(eval_steps) <= 0:
                LOG.warning(
                    "evaluation_strategy=steps but eval_steps is not set; disabling eval."
                )
                enabled = False
        elif strategy in {"no", "none", "off"}:
            enabled = False
        elif strategy in {"epoch", "epochs"}:
            LOG.warning("evaluation_strategy=epoch is not supported in custom loop; skipping eval.")
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
    """Return a TrainingLoopContext configured for the custom MaxEnt loop."""

    if force_grpo_objective is not None:
        training_args.train_grpo_objective = force_grpo_objective
    resume_checkpoint, resume_requested = resolve_resume_checkpoint(training_args)
    resume_state = load_trainer_state_metadata(resume_checkpoint)
    if resume_checkpoint:
        LOG.info("Resuming from checkpoint %s", resume_checkpoint)
        try:
            model_args.model_name_or_path = resume_checkpoint
        except Exception:
            pass
        try:
            training_args.model_name_or_path = resume_checkpoint
        except Exception:
            pass
        try:
            training_args.resume_from_checkpoint = resume_checkpoint
        except Exception:
            setattr(training_args, "resume_from_checkpoint", resume_checkpoint)
    elif resume_requested:
        try:
            training_args.resume_from_checkpoint = False
        except Exception:
            setattr(training_args, "resume_from_checkpoint", False)
    accelerator_cls_or_obj = require_accelerator(deps_namespace)
    # Defensive: reset any lingering Accelerate shared state (e.g., from stubs or
    # prior imports) so the first Accelerator() call can safely set attributes.
    try:  # pragma: no cover - runtime guard
        from accelerate.state import AcceleratorState

        reset_fn = getattr(AcceleratorState, "_reset_state", None)
        if callable(reset_fn):
            reset_fn(reset_partial_state=True)
    except Exception:
        pass

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
    tokenizer = get_tokenizer(model_args, training_args)
    train_dataset, eval_rows = load_datasets(
        script_args, training_args, tokenizer, accelerator=accelerator
    )
    training_args.eval_rows = eval_rows
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
    if sampler is not None:
        loader_kwargs["sampler"] = sampler
    train_loader = dataloader_cls(train_dataset, **loader_kwargs)
    runtime_handles = RuntimeHandles(
        accelerator=accelerator,
        model=model,
        tokenizer=tokenizer,
        train_loader=train_loader,
        train_sampler=None,
        device=accelerator.device,
        get_ref_model=lambda: model,
    )
    prompt_cache = _build_prompt_cache_fn(
        tokenizer,
        getattr(training_args, "max_prompt_length", 0),
        int(getattr(training_args, "maxent_prompt_cache_size", 0) or 0),
    )
    if callable(prompt_cache):
        runtime_handles.prompt_cache_get = prompt_cache
        scoring.batching.prompt_length_cache_get = prompt_cache
    optim_handles = _build_optim_handles(model, training_args)
    max_grad_norm = _sanitize_float(
        getattr(training_args, "max_grad_norm", 0.0),
        default=0.0,
        minimum=0.0,
        maximum=float("inf"),
        name="max_grad_norm",
    )
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
    elif steps_per_epoch and training_args.num_train_epochs > 0:
        total_training_steps = steps_per_epoch * training_args.num_train_epochs
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
        num_epochs=training_args.num_train_epochs,
        num_generations=training_args.num_generations,
        grad_accum_steps=training_args.gradient_accumulation_steps,
        max_grad_norm=max_grad_norm,
        steps_per_epoch=steps_per_epoch,
        total_training_steps=total_training_steps,
        warmup_steps=warmup_steps,
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
    logging_handles._checkpoint_state_ref = checkpoint_state_ref
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
    except Exception as exc:  # pragma: no cover - network/env guard
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
