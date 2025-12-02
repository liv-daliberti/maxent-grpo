"""InfoSeed-GRPO training entrypoint using the custom training loop.

This mirrors the baseline data/model loading but wires InfoSeed-specific
settings (seed augmentation + auxiliary loss) into the runtime context and
executes the custom ``training.run_training_loop`` instead of TRL's trainer.
"""

from __future__ import annotations

import logging

from typing import Any
from types import SimpleNamespace

from maxent_grpo.config import GRPOConfig, GRPOScriptArguments
from maxent_grpo.core.model import get_model, get_tokenizer
from maxent_grpo.training import (
    apply_info_seed,
    run_training_loop,
    GenerationSettings,
    EvaluationSettings,
    OptimizationSettings,
    OptimizationSchedule,
    ScoringSettings,
    ClipSettings,
    BatchingSettings,
    LoopSettings,
    RuntimeHandles,
    RewardSpec,
    TrainingLoopContext,
)
from maxent_grpo.training.weighting.types import WeightingSettings
from maxent_grpo.training.types.runtime import ControllerPaths
from maxent_grpo.training.runtime.deps import require_dataloader, require_accelerator
from maxent_grpo.training.runtime.prompts import GenerationPenaltyConfig
from maxent_grpo.training.runtime.config import VLLMClientConfig
from maxent_grpo.training.data import load_datasets
from maxent_grpo.training.rewards import (
    load_reward_functions,
    load_eval_reward_functions,
)
from maxent_grpo.training.weighting.logic import build_weighting_settings
from maxent_grpo.training.state import build_training_state
from maxent_grpo.training.types import PromptCacheEntry


LOG = logging.getLogger(__name__)


def _build_generation_settings(cfg: GRPOConfig) -> GenerationSettings:
    vllm_cfg = VLLMClientConfig(
        url=getattr(cfg, "vllm_url", ""),
        rounds_cfg=getattr(cfg, "vllm_rounds_cfg", 0),
        retry_sleep=getattr(cfg, "vllm_retry_sleep", 1.0),
        backfill_local=bool(getattr(cfg, "vllm_backfill_local", False)),
        request_logprobs=bool(getattr(cfg, "vllm_request_logprobs", True)),
        best_of=getattr(cfg, "gen_best_of", None),
        frequency_penalty=float(getattr(cfg, "gen_frequency_penalty", 0.0)),
        presence_penalty=float(getattr(cfg, "gen_presence_penalty", 0.0)),
        top_k=getattr(cfg, "gen_top_k", None),
        stop_sequences=getattr(cfg, "vllm_stop_sequences", None),
        timeout=float(getattr(cfg, "vllm_request_timeout", 120.0)),
        max_retries=int(getattr(cfg, "vllm_max_retries", 3)),
        backoff=float(getattr(cfg, "vllm_backoff", 1.0)),
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
    stats = settings.generation_stats
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


def _build_scoring_settings(
    cfg: GRPOConfig, weighting: WeightingSettings
) -> ScoringSettings:

    def _prompt_cache_get(_prompt: str) -> PromptCacheEntry:
        # Minimal placeholder; callers using scoring rely on .length attribute.
        return PromptCacheEntry(input_ids=[], attention_mask=[])

    batching = BatchingSettings(
        logprob_chunk_size=cfg.maxent_logprob_chunk_size,
        score_slice=cfg.maxent_logprob_chunk_size,
        prompt_length_cache_get=_prompt_cache_get,
    )
    clipping = ClipSettings(
        clip_range=cfg.clip_range,
        use_clip_objective=bool(getattr(cfg, "maxent_use_clip_objective", False)),
        clip_objective_coef=float(getattr(cfg, "maxent_clip_objective_coef", 1.0)),
        clip_adv_baseline=getattr(cfg, "maxent_clip_adv_baseline", None),
    )
    return ScoringSettings(weighting=weighting, clipping=clipping, batching=batching)


def _build_evaluation_settings(cfg: GRPOConfig) -> EvaluationSettings:
    rows = getattr(cfg, "eval_rows", []) if hasattr(cfg, "eval_rows") else []
    return EvaluationSettings(
        enabled=cfg.do_eval,
        rows=rows,
        batch_size=cfg.per_device_eval_batch_size,
        every_n_steps=getattr(cfg, "eval_steps", None),
    )


def run_infoseed_training(
    script_args: GRPOScriptArguments,
    training_args: GRPOConfig,
    model_args: "Any",
) -> None:
    """Run InfoSeed-GRPO training via the custom loop."""

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler()],
    )
    LOG.setLevel(training_args.get_process_log_level())

    accelerator_cls_or_obj = require_accelerator("infoseed")
    accelerator = (
        accelerator_cls_or_obj()
        if callable(accelerator_cls_or_obj)
        else accelerator_cls_or_obj
    )
    dataloader_cls = require_dataloader("infoseed")

    # Force GRPO objective (no MaxEnt weighting) unless explicitly overridden.
    training_args.train_grpo_objective = True
    model = get_model(model_args, training_args)
    tokenizer = get_tokenizer(model_args, training_args)

    # Dataset + reward loading mirrors baseline path.
    train_dataset, eval_rows = load_datasets(script_args, training_args, tokenizer)
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

    # Weighting from GRPOConfig (MaxEnt params set to neutral values).
    weighting = build_weighting_settings(training_args)
    scoring = _build_scoring_settings(training_args, weighting)
    generation = _build_generation_settings(training_args)
    evaluation = _build_evaluation_settings(training_args)
    generation, scoring, evaluation = apply_info_seed(
        generation, scoring, evaluation, training_args
    )

    # Runtime handles
    train_loader = dataloader_cls(
        train_dataset, batch_size=training_args.per_device_train_batch_size
    )
    runtime_handles = RuntimeHandles(
        accelerator=accelerator,
        model=model,
        tokenizer=tokenizer,
        train_loader=train_loader,
        train_sampler=None,
        device=accelerator.device,
        get_ref_model=lambda: model,
    )

    # Optimization schedule/handles
    def _fallback_optim_handles() -> SimpleNamespace:
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

    try:
        from maxent_grpo.training.optim import build_optimization_handles
    except ImportError:

        def build_optimization_handles(*_args, **_kwargs):
            return _fallback_optim_handles()

    try:
        optim_handles = build_optimization_handles(model, training_args)
    except ImportError:
        optim_handles = _fallback_optim_handles()
    schedule = OptimizationSchedule(
        num_epochs=training_args.num_train_epochs,
        num_generations=training_args.num_generations,
        grad_accum_steps=training_args.gradient_accumulation_steps,
        max_grad_norm=training_args.max_grad_norm,
        steps_per_epoch=None,
        total_training_steps=training_args.max_steps,
        warmup_steps=int(training_args.warmup_ratio * training_args.max_steps),
    )
    optimization = OptimizationSettings(schedule=schedule, handles=optim_handles)

    # Controller + logging handles (reuse training.state helpers).
    controller = ControllerPaths(
        state_path=None, resume_from=None, overwrite_existing=False
    )
    logging_handles = build_training_state(training_args)

    loop_settings = LoopSettings(
        generation=generation,
        evaluation=evaluation,
        optimization=optimization,
        scoring=scoring,
        controller=controller,
    )
    ctx = TrainingLoopContext(
        runtime=runtime_handles,
        reward=reward_spec,
        eval_reward=eval_reward_spec,
        settings=loop_settings,
        logging=logging_handles,
    )

    LOG.info(
        "Starting InfoSeed-GRPO training | seeds=%s | lambda=%s",
        getattr(training_args, "info_seed_num_seeds", 0),
        getattr(training_args, "info_seed_lambda", 0.0),
    )
    run_training_loop(ctx)


__all__ = ["run_infoseed_training"]
