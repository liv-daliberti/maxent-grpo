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

"""Training loop utilities for the MaxEnt-GRPO runner.

The functions in this module orchestrate the end-to-end training flow:

``run_training_loop``
    Entry point that constructs reusable helpers (generation, validation, step
    resources) and drives epoch iteration.
``_run_epoch``
    Consumes the PyTorch DataLoader to process each batch and stops early when
    the optimizer signals convergence/termination.
``_train_step``
    Encapsulates the per-batch workflow of generation, reward computation,
    sequence scoring, loss evaluation, gradient accumulation, controller
    updates, and logging hooks.

The helpers favor explicit parameters over implicit globals so that unit tests
can stub individual pieces (generation, checkpointing, metrics) without running
the full distributed stack.  All docstrings are Sphinx-friendly to power the
reference documentation for the training CLI.
"""

from __future__ import annotations

import logging
import os
from types import SimpleNamespace
from collections.abc import MutableMapping
import math
from contextlib import nullcontext

from dataclasses import replace
from typing import Optional

from .generation import CompletionGenerator, GenerationContext
from maxent_grpo.training.runtime import require_accelerator, require_torch
from maxent_grpo.core.hub import push_to_hub_revision
from .eval import run_validation_step
from .weighting.loss import LossInputConfig, build_loss_inputs, evaluate_losses
from . import pipeline as pipeline_mod
from .metrics import log_local_step, log_training_step, summarize_weight_stats
from .optim import (
    configure_accumulation_steps,
    detect_deepspeed_state,
    epoch_progress,
    optimizer_step,
    require_accumulation_context,
    scheduled_learning_rate,
    sync_gradients_enabled,
)
from .state import (
    check_stop_condition,
    load_controller_state_chain,
    maybe_checkpoint,
    maybe_load_accelerator_state,
)
from .controller_objective import ControllerMetaContext
from .weighting.logic import (
    apply_meta_controller_update,
    broadcast_controller_state,
    maybe_update_beta,
    maybe_update_tau,
    save_controller_state,
    _sync_controller_state,
)
from .types import (
    EvaluationSettings,
    LogStepArtifacts,
    StepBatchInfo,
    StepResources,
    TrainingLoopContext,
    TrainingLoopState,
    ValidationContext,
)
from .zero_utils import _maybe_patch_zero_no_sync

torch = require_torch("training")
Tensor = torch.Tensor
Accelerator = require_accelerator("training")
prepare_training_batch = pipeline_mod.prepare_training_batch

LOG = logging.getLogger(__name__)
_scheduled_learning_rate = scheduled_learning_rate
_epoch_progress = epoch_progress
_optimizer_step = optimizer_step


def _maybe_save_seed_heatmap(seed_heatmap: Optional[dict], step: int) -> None:
    """Persist a correlation heatmap when enabled via env flag.

    Controlled by ``INFOSEED_SAVE_HEATMAP=1``; writes a JSON file under
    ``INFOSEED_HEATMAP_DIR`` (default: ``logs/seed_heatmaps``).
    """

    if not seed_heatmap:
        return
    import json
    import os

    if os.environ.get("INFOSEED_SAVE_HEATMAP", "0") not in {"1", "true", "True"}:
        return
    out_dir = os.environ.get("INFOSEED_HEATMAP_DIR", "logs/seed_heatmaps")
    try:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"seed_heatmap_step{int(step)}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(seed_heatmap, f)
    except (OSError, TypeError, ValueError):  # pragma: no cover - best-effort
        LOG.warning("Failed to save seed heatmap at step %d", step)


def _maybe_validate(
    evaluation_cfg: EvaluationSettings,
    validation_ctx: ValidationContext,
    global_step: int,
) -> None:
    """Optionally trigger the evaluation loop.

    :param evaluation_cfg: Evaluation schedule/settings.
    :type evaluation_cfg: EvaluationSettings
    :param validation_ctx: Handles required to run validation.
    :type validation_ctx: ValidationContext
    :param global_step: Current optimizer step.
    :type global_step: int
    """
    if (
        evaluation_cfg.enabled
        and evaluation_cfg.every_n_steps
        and (global_step % evaluation_cfg.every_n_steps == 0)
    ):
        run_validation_step(global_step, validation_ctx)


def _maybe_overwrite_controller_state_from_config(
    ctx: TrainingLoopContext, controller_resumed: bool = False
) -> None:
    """Optionally force controller scalars to match the active recipe."""

    training_args = getattr(ctx, "training_args", None)
    if training_args is None:
        return
    if not getattr(training_args, "controller_overwrite_from_config", False):
        return
    if controller_resumed:
        return
    weighting = getattr(getattr(ctx, "scoring", None), "weighting", None)
    if weighting is None:
        return

    def _coerce_scalar(value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _first_defined(*values):
        for candidate in values:
            if candidate is not None:
                return candidate
        return None

    tau_override = _coerce_scalar(getattr(training_args, "maxent_tau", None))
    beta_source = _first_defined(
        getattr(training_args, "init_kl_coeff", None),
        getattr(training_args, "init_kl_coef", None),
        getattr(training_args, "kl_penalty_beta", None),
        getattr(training_args, "beta", None),
    )
    beta_override = _coerce_scalar(beta_source)
    updated = False
    prev_tau = getattr(weighting, "tau", None)
    prev_beta = getattr(weighting, "beta", None)
    if tau_override is not None:
        weighting.tau = tau_override
        updated = True
    if beta_override is not None:
        weighting.beta = beta_override
        updated = True
    if not updated:
        return
    if getattr(weighting, "train_grpo_objective", False):
        weighting.denom = 1.0
    else:
        denom_sum = float(weighting.tau) + float(weighting.beta)
        weighting.denom = denom_sum if denom_sum > 0 else 1.0
    try:
        setattr(weighting, "_tau_entropy_ema", float(weighting.tau))
        setattr(weighting, "_tau_log", math.log(max(float(weighting.tau), 1e-8)))
    except (TypeError, ValueError):
        pass
    _sync_controller_state(weighting)
    accelerator = getattr(getattr(ctx, "runtime", None), "accelerator", None)
    if accelerator is not None:
        broadcast_controller_state(accelerator, weighting)
    prev_tau_float = _coerce_scalar(prev_tau)
    prev_beta_float = _coerce_scalar(prev_beta)
    LOG.info(
        "Overwrote controller state from config | tau=%.4f (prev=%s) | beta=%.4f (prev=%s)",
        float(weighting.tau),
        "nan" if prev_tau_float is None else f"{prev_tau_float:.4f}",
        float(weighting.beta),
        "nan" if prev_beta_float is None else f"{prev_beta_float:.4f}",
    )


def _apply_weighting_overrides_from_config(ctx: TrainingLoopContext) -> None:
    """Apply non-controller weighting toggles from the active training config."""

    training_args = getattr(ctx, "training_args", None)
    if training_args is None:
        return
    scoring_cfg = getattr(ctx, "scoring", None)
    weighting = getattr(scoring_cfg, "weighting", None) if scoring_cfg else None
    if weighting is None:
        return
    fallback_flag = getattr(training_args, "maxent_allow_empty_weight_fallback", None)
    if fallback_flag is not None:
        weighting.allow_empty_weight_fallback = bool(fallback_flag)


def _train_step(
    ctx: TrainingLoopContext,
    state: TrainingLoopState,
    step_info: StepBatchInfo,
    resources: StepResources,
) -> bool:
    """Process a single batch and update training state.

    :param ctx: Training loop context containing configs and runtime handles.
    :type ctx: :class:`~training.types.TrainingLoopContext`
    :param state: Mutable counters shared across batches and epochs.
    :type state: :class:`~training.types.TrainingLoopState`
    :param step_info: Metadata describing the current batch (epoch index,
        micro-step, and raw data).
    :type step_info: :class:`~training.types.StepBatchInfo`
    :param resources: Reusable handles for generation and validation.
    :type resources: :class:`~training.types.StepResources`
    :returns: ``True`` when training should stop (schedule exhausted or
        controller requested halt), otherwise ``False``.
    :rtype: bool
    """
    schedule = ctx.optimization.schedule
    accelerator = ctx.runtime.accelerator
    ds_state = detect_deepspeed_state(accelerator)
    LOG.debug(
        "Train step begin | epoch=%d | step_in_epoch=%d | global_step=%d",
        step_info.epoch,
        step_info.step_in_epoch,
        state.global_step,
    )
    gen_stats = getattr(ctx.generation, "generation_stats", None)
    if isinstance(gen_stats, MutableMapping):
        gen_stats["current_step"] = int(state.global_step)
    prepared = prepare_training_batch(ctx, resources.generator, step_info.batch)
    if prepared is None:
        skip_stage = getattr(ctx.runtime, "_last_skip_stage", "unknown")
        LOG.warning(
            "Skipping training batch | epoch=%d | step_in_epoch=%d | global_step=%d | stage=%s",
            step_info.epoch,
            step_info.step_in_epoch,
            state.global_step,
            skip_stage,
        )
        try:
            delattr(ctx.runtime, "_last_skip_stage")
        except Exception:
            pass
        return False
    state.num_input_tokens_seen += float(prepared.total_input_tokens)
    loss_outputs, diagnostics = evaluate_losses(
        *build_loss_inputs(
            prepared.grouped_completions,
            prepared.weight_stats,
            prepared.scores,
            LossInputConfig(
                clip_cfg=ctx.scoring.clipping,
                weighting_cfg=ctx.scoring.weighting,
                ref_stats=prepared.ref_stats,
            ),
        ),
        seed_inputs=(
            prepared.scores.seed_aux if hasattr(prepared.scores, "seed_aux") else None
        ),
        info_seed_lambda=getattr(ctx.scoring, "info_seed_lambda", 0.0),
        info_seed_temperature=getattr(ctx.scoring, "info_seed_temperature", 0.1),
        info_seed_loss_type=getattr(ctx.scoring, "info_seed_loss_type", "infonce"),
        info_seed_alpha_entropy=getattr(ctx.scoring, "info_seed_alpha_entropy", 0.0),
    )
    _maybe_save_seed_heatmap(
        getattr(prepared, "seed_heatmap", None),
        state.global_step,
    )
    current_lr = _scheduled_learning_rate(
        schedule, ctx.optimization.handles, state.global_step
    )
    log_artifacts = LogStepArtifacts(
        loss_outputs=loss_outputs,
        diagnostics=diagnostics,
        grad_norm_scalar=None,
        epoch_progress=_epoch_progress(
            schedule, step_info.epoch, step_info.step_in_epoch
        ),
    )
    scalars = getattr(loss_outputs, "scalars", None)
    if scalars is not None:
        LOG.debug(
            "Loss scalars | total=%.4f | policy=%.4f | kl=%.4f",
            getattr(scalars, "total_loss", 0.0),
            getattr(scalars, "policy_loss", 0.0),
            getattr(scalars, "kl_loss", 0.0),
        )
    if accelerator.__class__.__name__ == "SimpleNamespace":
        accumulation_ctx = nullcontext()
    else:
        accumulation_ctx = require_accumulation_context(
            accelerator,
            ctx.runtime.model,
        )
        if not hasattr(accumulation_ctx, "__enter__"):
            accumulation_ctx = nullcontext()
    grad_accum_total = int(getattr(schedule, "grad_accum_steps", 1) or 1)
    accum_position = (step_info.step_in_epoch % grad_accum_total) + 1
    grad_norm_scalar: Optional[float] = None
    LOG.debug(
        "Entering accumulate context | grad_accum_steps=%d | accum_progress=%d/%d | step_in_epoch=%d",
        schedule.grad_accum_steps,
        accum_position,
        grad_accum_total,
        step_info.step_in_epoch,
    )
    with accumulation_ctx:
        # Some test stubs may produce a loss tensor that does not require
        # gradients (e.g., detached or aggregated into a float). Guard the
        # backward call to avoid RuntimeError when autograd is not active.
        loss_val = loss_outputs.loss
        if getattr(loss_val, "requires_grad", False):
            accelerator.backward(loss_val)
        else:
            LOG.debug("Skipping backward: loss does not require grad")
        if ds_state.use_deepspeed and ds_state.zero_stage >= 2:
            if (step_info.step_in_epoch + 1) % grad_accum_total != 0:
                LOG.debug(
                    "DeepSpeed accumulate | deferring optimizer step | epoch=%d | step_in_epoch=%d | accum_progress=%d/%d",
                    step_info.epoch,
                    step_info.step_in_epoch,
                    accum_position,
                    grad_accum_total,
                )
                return False
            grad_norm_scalar = _optimizer_step(ctx, state, current_lr)
            LOG.debug(
                "Optimizer step executed | global_step=%d | grad_norm=%s",
                state.global_step,
                grad_norm_scalar,
            )
        else:
            if not sync_gradients_enabled(accelerator, state.global_step):
                LOG.debug(
                    "Deferring optimizer step until sync | epoch=%d | step_in_epoch=%d | accum_progress=%d/%d",
                    step_info.epoch,
                    step_info.step_in_epoch,
                    accum_position,
                    grad_accum_total,
                )
                return False
            grad_norm_scalar = _optimizer_step(ctx, state, current_lr)
            LOG.debug(
                "Optimizer step executed | global_step=%d | grad_norm=%s",
                state.global_step,
                grad_norm_scalar,
            )
    LOG.debug("Exiting accumulate context | synced_step=%d", state.global_step)
    log_artifacts.grad_norm_scalar = grad_norm_scalar
    weight_stats_global = summarize_weight_stats(
        accelerator, prepared.weight_stats
    )  # Aggregate entropy across ranks once per step.
    log_local_step(
        ctx,
        state,
        prepared,
        log_artifacts,
        current_lr,
        weight_view=weight_stats_global,
    )
    log_training_step(
        ctx,
        state,
        prepared,
        log_artifacts,
        current_lr,
        weight_view=weight_stats_global,
    )
    maybe_update_beta(ctx.scoring.weighting, loss_outputs.kl_loss_scalar)
    base_lr = max(float(getattr(ctx.optimization.handles, "learning_rate", 0.0)), 1e-12)
    lr_scale = float(current_lr) / base_lr if base_lr > 0 else 1.0
    weight_stats_for_tau = getattr(prepared, "weight_stats", weight_stats_global)
    try:
        maybe_update_tau(
            ctx.scoring.weighting,
            weight_stats_for_tau,
            state.global_step,
            lr_scale=lr_scale,
        )
    except TypeError:
        maybe_update_tau(ctx.scoring.weighting, weight_stats_for_tau, state.global_step)
    settings_obj = getattr(ctx, "settings", None)
    controller_objective = getattr(settings_obj, "controller_objective", None)
    controller_manager = getattr(settings_obj, "controller_meta_manager", None)
    if controller_manager is None:
        controller_manager = getattr(ctx, "controller_meta_manager", None)
    if controller_objective is None:
        controller_objective = getattr(ctx, "controller_objective", None)
    if controller_objective is not None:
        should_run_meta = (
            controller_manager.should_run(state.global_step)
            if controller_manager
            else True
        )
    else:
        should_run_meta = False
    if controller_objective is not None and should_run_meta:
        _cache_meta_stats(ctx.scoring.weighting, weight_stats_global, loss_outputs)
        meta_ctx = ControllerMetaContext(
            weighting=ctx.scoring.weighting,
            weight_stats=weight_stats_global,
            loss_outputs=loss_outputs,
            prepared_batch=prepared,
            global_step=state.global_step,
            lr_scale=lr_scale,
            kl_value=loss_outputs.kl_loss_scalar,
            backprop_fn=controller_manager.make_backprop_fn()
            if controller_manager
            else None,
        )
        try:
            gradients = controller_objective.compute(meta_ctx)
        except (RuntimeError, ValueError, TypeError) as exc:  # pragma: no cover - defensive logging
            gradients = None
            LOG.warning("Controller objective failed: %s", exc)
        if controller_manager:
            controller_manager.apply_gradients(gradients, lr_scale=lr_scale)
        elif gradients and gradients.has_updates():
            apply_meta_controller_update(
                ctx.scoring.weighting,
                tau_grad=gradients.tau_grad,
                beta_grad=gradients.beta_grad,
                lr_scale=lr_scale,
            )
    broadcast_controller_state(accelerator, ctx.scoring.weighting)
    if ctx.runtime.accelerator.is_main_process:
        save_controller_state(ctx.controller.state_path, ctx.scoring.weighting)
    _maybe_validate(ctx.evaluation, resources.validation_ctx, state.global_step)
    maybe_checkpoint(ctx.logging, ctx.runtime.accelerator, state.global_step)
    check_stop_condition(ctx.optimization.schedule, state)
    return state.stop_training


def _run_epoch(
    ctx: TrainingLoopContext,
    state: TrainingLoopState,
    epoch: int,
    resources: StepResources,
) -> bool:
    """Run a full epoch and return ``True`` when training should stop.

    :param ctx: Training loop context with loaders, configs, and handles.
    :type ctx: TrainingLoopContext
    :param state: Mutable loop counters shared across epochs.
    :type state: TrainingLoopState
    :param epoch: Epoch index used for schedulers/samplers.
    :type epoch: int
    :param resources: Reusable generation/validation handles.
    :type resources: StepResources
    :returns: ``True`` if the loop signaled an early stop.
    :rtype: bool
    """
    sampler = getattr(ctx.runtime, "train_sampler", None)
    if sampler is not None:
        set_epoch = getattr(sampler, "set_epoch", None)
        if callable(set_epoch):
            try:
                set_epoch(epoch)
            except TypeError:
                set_epoch(int(epoch))
    for step_in_epoch, batch in enumerate(ctx.runtime.train_loader):
        step_info = StepBatchInfo(epoch=epoch, step_in_epoch=step_in_epoch, batch=batch)
        if _train_step(ctx, state, step_info, resources):
            return True
    return state.stop_training


def run_training_loop(ctx: TrainingLoopContext) -> None:
    """Execute the training loop using the supplied context.

    :param ctx: Fully-populated :class:`~training.types.TrainingLoopContext`
        describing runtime handles, configurations, logging hooks, and controller paths.
    :type ctx: :class:`~training.types.TrainingLoopContext`
    :returns: ``None``. Side effects are driven through the provided handles.
    :rtype: None
    """

    def _maybe_load_optimizer_state(path: Optional[str]) -> None:
        opt = getattr(ctx.optimization.handles, "optimizer", None)
        if not path or opt is None:
            return
        opt_path = os.path.join(path, "optimizer.pt")
        if not os.path.isfile(opt_path):
            return
        try:
            state_dict = torch.load(opt_path, map_location=runtime.device)
            opt.load_state_dict(state_dict)
            LOG.info("Loaded optimizer state from %s", opt_path)
        except (OSError, RuntimeError, ValueError) as exc:
            LOG.warning("Failed to load optimizer state from %s: %s", opt_path, exc)

    def _maybe_push_final() -> None:
        training_args = getattr(ctx, "training_args", None)
        if training_args is None:
            return
        push_enabled = bool(
            getattr(training_args, "push_to_hub", False)
            or getattr(training_args, "push_to_hub_revision", False)
        )
        if not push_enabled:
            return
        hub_strategy = str(getattr(training_args, "hub_strategy", "end") or "end").lower()
        if hub_strategy not in {"end", "last", "final"}:
            return
        if not runtime.accelerator.is_main_process:
            return
        try:
            push_args = SimpleNamespace(**getattr(training_args, "__dict__", {}))
            push_args.push_to_hub_revision = True
            push_to_hub_revision(
                push_args,
                extra_ignore_patterns=[],
                include_checkpoints=True,
            )
        except Exception as exc:  # pragma: no cover - optional hub deps
            LOG.warning("Failed to push final output_dir to Hub: %s", exc)

    runtime = ctx.runtime
    generation_cfg = ctx.generation
    generation_ctx = GenerationContext(
        accelerator=runtime.accelerator,
        model=runtime.model,
        tokenizer=runtime.tokenizer,
        generation_stats=generation_cfg.generation_stats,
        device=runtime.device,
        max_prompt_len=generation_cfg.max_prompt_len,
        max_completion_len=generation_cfg.max_completion_len,
        gen_temperature=generation_cfg.gen_temperature,
        gen_top_p=generation_cfg.gen_top_p,
        use_vllm=generation_cfg.use_vllm,
        vllm=generation_cfg.vllm,
        penalty=replace(generation_cfg.penalty),
    )
    completion_generator = CompletionGenerator(generation_ctx)
    validation_ctx = ValidationContext(
        evaluation=ctx.evaluation,
        accelerator=runtime.accelerator,
        model=runtime.model,
        tokenizer=runtime.tokenizer,
        reward=ctx.reward,
        eval_reward=getattr(ctx, "eval_reward", None),
        generator=completion_generator.generate,
        logging=ctx.logging,
    )
    configure_accumulation_steps(
        runtime.accelerator, ctx.optimization.schedule.grad_accum_steps
    )
    _maybe_patch_zero_no_sync(runtime.model)
    resume_state = getattr(ctx, "resume_state", {}) or {}
    starting_step = int(resume_state.get("global_step", 0) or 0)
    starting_tokens = float(resume_state.get("num_input_tokens_seen", 0.0) or 0.0)
    state = TrainingLoopState(
        global_step=starting_step,
        num_input_tokens_seen=starting_tokens,
    )
    if runtime.accelerator.is_main_process:
        LOG.info(
            "Training schedule | steps_per_epoch=%s | total_training_steps=%s | grad_accum_steps=%s",
            ctx.optimization.schedule.steps_per_epoch,
            ctx.optimization.schedule.total_training_steps,
            ctx.optimization.schedule.grad_accum_steps,
        )
        if starting_step > 0:
            LOG.info(
                "Resumed training state | checkpoint=%s | global_step=%d | num_input_tokens_seen=%.0f",
                getattr(ctx, "resume_checkpoint", None),
                starting_step,
                starting_tokens,
            )
    resources = StepResources(
        generator=completion_generator.generate,
        validation_ctx=validation_ctx,
    )
    state_ref = getattr(ctx, "checkpoint_state_ref", None)
    if isinstance(state_ref, dict):
        state_ref["state"] = state
    accel_state_path = getattr(ctx, "resume_checkpoint", None) or getattr(
        ctx.controller, "resume_from", None
    )
    controller_loaded = load_controller_state_chain(
        ctx.controller,
        runtime.accelerator,
        ctx.scoring.weighting,
    )
    _maybe_overwrite_controller_state_from_config(
        ctx, controller_resumed=bool(controller_loaded)
    )
    _apply_weighting_overrides_from_config(ctx)
    maybe_load_accelerator_state(accel_state_path, runtime.accelerator)
    _maybe_load_optimizer_state(accel_state_path)
    try:
        ctx.optimization.handles.optimizer.zero_grad(set_to_none=True)
        for epoch in range(ctx.optimization.schedule.num_epochs):
            if _run_epoch(ctx, state, epoch, resources):
                break
    finally:
        if runtime.accelerator.is_main_process:
            LOG.info(
                (
                    "Generation stats | retries=%d | backfilled_prompts=%d | "
                    "vllm_failures=%d | dropped_prompts=%d | partial_prompts=%d | "
                    "excess_prompts=%d | excess_completions=%d"
                ),
                generation_cfg.generation_stats["vllm_retry_rounds"],
                generation_cfg.generation_stats["vllm_backfilled_prompts"],
                generation_cfg.generation_stats["vllm_failed_prompts"],
                generation_cfg.generation_stats["dropped_prompts"],
                generation_cfg.generation_stats.get("partial_prompts", 0),
                generation_cfg.generation_stats.get("vllm_excess_prompts", 0),
                generation_cfg.generation_stats.get("vllm_excess_completions", 0),
            )
            _maybe_push_final()
        if ctx.logging.wandb_run is not None and runtime.accelerator.is_main_process:
            try:
                ctx.logging.wandb_run.finish()
            except (
                RuntimeError,
                ValueError,
            ) as exc:  # pragma: no cover - defensive logging
                LOG.warning("Failed to close W&B run cleanly: %s", exc)


def _cache_meta_stats(weighting_cfg, weight_view, loss_outputs):
    entropy_val = getattr(weight_view, "weight_entropy", None)
    if entropy_val is None:
        entropy_val = getattr(weight_view, "entropy", None)
    if isinstance(entropy_val, (int, float)):
        setattr(weighting_cfg, "_meta_entropy_value", float(entropy_val))
    kl_val = getattr(loss_outputs, "kl_loss_scalar", None)
    if isinstance(kl_val, (int, float)):
        setattr(weighting_cfg, "_meta_kl_value", float(kl_val))
    meta_loss = 0.0
    target_entropy = getattr(weighting_cfg, "tau_target_entropy", None)
    if target_entropy is not None and isinstance(entropy_val, (int, float)):
        entropy_error = float(entropy_val) - float(target_entropy)
        meta_loss += 0.5 * entropy_error * entropy_error
    kl_target = getattr(weighting_cfg, "kl_target", None)
    if kl_target and isinstance(kl_val, (int, float)):
        kl_error = float(kl_val) - float(kl_target)
        meta_loss += 0.5 * kl_error * kl_error
    setattr(weighting_cfg, "_meta_last_loss", float(meta_loss))
