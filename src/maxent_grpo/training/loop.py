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
from collections.abc import MutableMapping

from dataclasses import replace
from typing import Optional

from .generation import CompletionGenerator, GenerationContext
from maxent_grpo.training.runtime import require_accelerator, require_torch
from .eval import run_validation_step
from .weighting.loss import LossInputConfig, build_loss_inputs, evaluate_losses
from .pipeline import prepare_training_batch
from .metrics import log_local_step, log_training_step
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
from .weighting.logic import (
    maybe_update_beta,
    maybe_update_tau,
    save_controller_state,
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

LOG = logging.getLogger(__name__)
_scheduled_learning_rate = scheduled_learning_rate
_epoch_progress = epoch_progress
_optimizer_step = optimizer_step


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
        )
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
    accumulation_ctx = require_accumulation_context(
        accelerator,
        ctx.runtime.model,
    )
    grad_norm_scalar: Optional[float] = None
    LOG.debug(
        "Entering accumulate context | grad_accum_steps=%d | step_in_epoch=%d",
        schedule.grad_accum_steps,
        step_info.step_in_epoch,
    )
    with accumulation_ctx:
        accelerator.backward(loss_outputs.loss)
        if ds_state.use_deepspeed and ds_state.zero_stage >= 2:
            if (step_info.step_in_epoch + 1) % max(schedule.grad_accum_steps, 1) != 0:
                LOG.debug(
                    "DeepSpeed accumulate | deferring optimizer step | epoch=%d | step_in_epoch=%d",
                    step_info.epoch,
                    step_info.step_in_epoch,
                )
                return False
            grad_norm_scalar = _optimizer_step(ctx, state, current_lr)
        else:
            if not sync_gradients_enabled(accelerator, state.global_step):
                LOG.debug(
                    "Deferring optimizer step until sync | epoch=%d | step_in_epoch=%d",
                    step_info.epoch,
                    step_info.step_in_epoch,
                )
                return False
            grad_norm_scalar = _optimizer_step(ctx, state, current_lr)
    LOG.debug("Exiting accumulate context | synced_step=%d", state.global_step)
    log_artifacts.grad_norm_scalar = grad_norm_scalar
    log_local_step(ctx, state, prepared, log_artifacts, current_lr)
    log_training_step(ctx, state, prepared, log_artifacts, current_lr)
    maybe_update_beta(ctx.scoring.weighting, loss_outputs.kl_loss_scalar)
    maybe_update_tau(ctx.scoring.weighting, prepared.weight_stats, state.global_step)
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
        reward=ctx.reward,
        generator=completion_generator.generate,
        logging=ctx.logging,
    )
    configure_accumulation_steps(
        runtime.accelerator, ctx.optimization.schedule.grad_accum_steps
    )
    _maybe_patch_zero_no_sync(runtime.model)
    state = TrainingLoopState()
    if runtime.accelerator.is_main_process:
        LOG.info(
            "Training schedule | steps_per_epoch=%s | total_training_steps=%s | grad_accum_steps=%s",
            ctx.optimization.schedule.steps_per_epoch,
            ctx.optimization.schedule.total_training_steps,
            ctx.optimization.schedule.grad_accum_steps,
        )
    resources = StepResources(
        generator=completion_generator.generate,
        validation_ctx=validation_ctx,
    )
    load_controller_state_chain(
        ctx.controller,
        runtime.accelerator,
        ctx.scoring.weighting,
    )
    maybe_load_accelerator_state(ctx.controller.resume_from, runtime.accelerator)
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
        if ctx.logging.wandb_run is not None and runtime.accelerator.is_main_process:
            try:
                ctx.logging.wandb_run.finish()
            except (
                RuntimeError,
                ValueError,
            ) as exc:  # pragma: no cover - defensive logging
                LOG.warning("Failed to close W&B run cleanly: %s", exc)
