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
import math
from collections.abc import MutableMapping
from contextlib import nullcontext

try:  # accelerate is optional in the unit-test environment
    from accelerate.state import DistributedType  # type: ignore
except ImportError:  # pragma: no cover - test fallback when accelerate is absent
    class DistributedType:  # type: ignore[misc]
        DEEPSPEED = "deepspeed"
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional

from . import run_generation
from .run_helpers import (
    require_accelerator,
    require_torch,
)
from .run_training_eval import run_validation_step
from .run_training_loss import (
    LossInputConfig,
    build_loss_inputs,
    evaluate_losses,
)
from .run_training_pipeline import prepare_training_batch
from .run_training_metrics import LogStepArtifacts, log_local_step, log_training_step
from .run_training_state import (
    check_stop_condition,
    load_controller_state_chain,
    maybe_checkpoint,
    maybe_load_accelerator_state,
)
from .run_training_weighting import (
    maybe_update_beta,
    maybe_update_tau,
    save_controller_state,
)
from .run_training_types import (
    EvaluationSettings,
    GenerationFn,
    OptimizationSchedule,
    OptimizerHandles,
    TrainingLoopContext,
    ValidationContext,
)
from .zero_utils import _maybe_patch_zero_no_sync
torch = require_torch("training")
Tensor = torch.Tensor
Accelerator = require_accelerator("training")

LOG = logging.getLogger(__name__)
_TWO_NORM = 2.0


@dataclass
class TrainingLoopState:
    """Mutable counters that track training progress.

    Parameters
    ----------
    global_step:
        Number of optimizer steps that have completed successfully.
    stop_training:
        Flag toggled by :func:`check_stop_condition` when schedules finish.
    num_input_tokens_seen:
        Running total of prompt+completion tokens processed; logged for cost
        diagnostics.
    metric_sums / metric_counts:
        Accumulators used by :func:`~maxent_helpers.run_training_metrics.accumulate_metrics`.
    """

    global_step: int = 0
    stop_training: bool = False
    num_input_tokens_seen: float = 0.0
    metric_sums: Dict[str, float] = field(default_factory=dict)
    metric_counts: Dict[str, int] = field(default_factory=dict)


@dataclass
class StepBatchInfo:
    """Metadata describing the in-progress batch."""

    epoch: int
    step_in_epoch: int
    batch: Dict[str, List[str]]


@dataclass
class StepResources:
    """Reusable handles for each train step."""

    generator: GenerationFn
    validation_ctx: ValidationContext


@dataclass
class _DeepspeedState:
    """Runtime flags describing the current DeepSpeed strategy."""

    use_deepspeed: bool
    zero_stage: int


def _clip_grad_norm_local(
    model: Any,
    accelerator: Accelerator,
    max_grad_norm: float,
) -> Optional[float]:
    """Clip gradients via Accelerate when possible and return the norm."""
    if max_grad_norm <= 0.0:
        return None
    params = [param for param in model.parameters() if param.grad is not None]
    if not params:
        return None
    grad_norm: Optional[float] = None
    clip_fn = getattr(accelerator, "clip_grad_norm_", None)
    if callable(clip_fn):
        try:
            grad_norm = clip_fn(params, max_grad_norm, norm_type=_TWO_NORM)
        except TypeError:
            grad_norm = clip_fn(params, max_grad_norm)
    if grad_norm is None:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            params,
            max_grad_norm,
            norm_type=_TWO_NORM,
        )
    if isinstance(grad_norm, torch.Tensor):
        grad_norm = grad_norm.detach().float().cpu().item()
    try:
        return float(grad_norm)
    except (TypeError, ValueError):
        return None


def _optimizer_step(
    ctx: TrainingLoopContext,
    state: TrainingLoopState,
    current_lr: float,
) -> Optional[float]:
    """Perform the optimizer + scheduler step and update ``global_step``.

    Parameters
    ----------
    ctx:
        Training context containing optimizer handles.
    state:
        Mutable loop state that tracks step/metrics.
    current_lr:
        Learning rate to apply before stepping the optimizer.

    Returns
    -------
    Optional[float]
        Gradient norm (if computed) for metrics/logging.
    """
    accelerator = ctx.runtime.accelerator
    schedule = ctx.optimization.schedule
    handles = ctx.optimization.handles
    grad_norm_scalar = _clip_grad_norm_local(
        ctx.runtime.model,
        accelerator,
        float(schedule.max_grad_norm),
    )
    LOG.debug("Optimizer step starting | scheduled_step=%d", state.global_step + 1)
    _apply_learning_rate(handles, current_lr)
    optimizer_step_fn = getattr(accelerator, "optimizer_step", None)
    if callable(optimizer_step_fn):
        optimizer_step_fn(handles.optimizer)
    else:
        handles.optimizer.step()
    handles.optimizer.zero_grad(set_to_none=True)
    state.global_step += 1
    LOG.debug("Optimizer step complete | new_global_step=%d", state.global_step)
    return grad_norm_scalar


def _epoch_progress(schedule: OptimizationSchedule, epoch: int, step_in_epoch: int) -> float:
    """Return floating-point epoch progress for logging."""
    if schedule.steps_per_epoch and schedule.steps_per_epoch > 0:
        return float(epoch) + float(step_in_epoch + 1) / float(schedule.steps_per_epoch)
    return float(epoch + 1)


def _scheduled_learning_rate(schedule: OptimizationSchedule, handles: OptimizerHandles, step: int) -> float:
    """Return the learning rate for the given optimizer step."""
    base_lr = handles.learning_rate
    warmup_steps = max(int(getattr(schedule, "warmup_steps", 0)), 0)
    total_steps = max(int(schedule.total_training_steps), 1)
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * (float(step) / float(warmup_steps))
    decay_steps = max(total_steps - warmup_steps, 1)
    progress = min(max(step - warmup_steps, 0), decay_steps) / float(decay_steps)
    return 0.5 * base_lr * (1.0 + math.cos(math.pi * progress))


def _apply_learning_rate(handles: OptimizerHandles, learning_rate: float) -> None:
    """Set the provided learning rate on all optimizer parameter groups."""
    for optimizer in (handles.optimizer, handles.base_optimizer):
        param_groups = getattr(optimizer, "param_groups", None)
        if not param_groups:
            continue
        for group in param_groups:
            group["lr"] = learning_rate



def _maybe_validate(
    evaluation_cfg: EvaluationSettings,
    validation_ctx: ValidationContext,
    global_step: int,
) -> None:
    """Optionally trigger the evaluation loop."""
    if (
        evaluation_cfg.enabled
        and evaluation_cfg.every_n_steps
        and (global_step % evaluation_cfg.every_n_steps == 0)
    ):
        run_validation_step(global_step, validation_ctx)


def _configure_accumulation_steps(accelerator: Accelerator, grad_accum_steps: int) -> None:
    """Pass gradient accumulation steps to Accelerate when supported."""
    if grad_accum_steps <= 1:
        return
    grad_state = getattr(accelerator, "gradient_state", None)
    setters = []
    for target in (accelerator, grad_state):
        if target is None:
            continue
        setters.extend(
            [
                getattr(target, "set_gradient_accumulation_steps", None),
                getattr(target, "set_accumulation_steps", None),
            ]
        )
    for setter in setters:
        if callable(setter):
            try:
                setter(int(grad_accum_steps))
                return
            except TypeError:
                continue
    for target in (accelerator, grad_state):
        if target is None:
            continue
        if hasattr(target, "gradient_accumulation_steps"):
            try:
                setattr(target, "gradient_accumulation_steps", int(grad_accum_steps))
                return
            except (AttributeError, TypeError):
                continue


def _detect_deepspeed_state(accelerator: Accelerator) -> _DeepspeedState:
    """Return DeepSpeed usage flags derived from the accelerator state."""
    accelerator_state = getattr(accelerator, "state", None)
    ds_plugin = getattr(accelerator_state, "deepspeed_plugin", None)
    try:
        zero_stage = int(getattr(ds_plugin, "zero_stage", 0) or 0)
    except (TypeError, ValueError):
        zero_stage = 0
    distributed_type = getattr(accelerator_state, "distributed_type", None)
    use_deepspeed = distributed_type == DistributedType.DEEPSPEED
    return _DeepspeedState(use_deepspeed=use_deepspeed, zero_stage=zero_stage)


def _sync_gradients_enabled(accelerator: Accelerator, global_step: int) -> bool:
    """Return the sync_gradients flag and log it for debugging."""
    syncing = bool(getattr(accelerator, "sync_gradients", True))
    LOG.debug(
        "Backprop complete | sync_gradients=%s | global_step=%d",
        syncing,
        global_step,
    )
    return syncing


def _require_accumulation_context(accelerator: Accelerator, model: Any) -> Any:
    """Return an accumulation context compatible with current strategy."""
    ds_state = _detect_deepspeed_state(accelerator)
    # Deepspeed is incompatible with Accelerate's no_sync-based accumulate wrapper
    # when gradients are partitioned. Rely on DeepSpeed's own grad accumulation.
    if ds_state.use_deepspeed or ds_state.zero_stage >= 2:
        return nullcontext()
    accumulate_fn = getattr(accelerator, "accumulate", None)
    if callable(accumulate_fn):
        return accumulate_fn(model)
    raise RuntimeError(
        "Accelerator.accumulate is unavailable; upgrade Accelerate to match TRL's control flow."
    )


def _train_step(
    ctx: TrainingLoopContext,
    state: TrainingLoopState,
    step_info: StepBatchInfo,
    resources: StepResources,
) -> bool:
    """Process a single batch and update training state.

    Parameters
    ----------
    ctx:
        Training loop context containing configs and runtime handles.
    state:
        Mutable counters shared across batches and epochs.
    step_info:
        Metadata describing the current batch (epoch index, micro-step, raw
        batch data).
    resources:
        Reusable handles for generation and validation.

    Returns
    -------
    bool
        ``True`` when training should stop (schedule exhausted or controller
        requested halt), otherwise ``False``.
    """
    schedule = ctx.optimization.schedule
    accelerator = ctx.runtime.accelerator
    ds_state = _detect_deepspeed_state(accelerator)
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
    current_lr = _scheduled_learning_rate(schedule, ctx.optimization.handles, state.global_step)
    log_artifacts = LogStepArtifacts(
        loss_outputs=loss_outputs,
        diagnostics=diagnostics,
        grad_norm_scalar=None,
        epoch_progress=_epoch_progress(schedule, step_info.epoch, step_info.step_in_epoch),
    )
    accumulation_ctx = _require_accumulation_context(
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
            if not _sync_gradients_enabled(accelerator, state.global_step):
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

    The loop iterates over ``ctx.runtime.train_loader`` and calls
    :func:`_train_step` for each batch.  Early termination is propagated when
    ``_train_step`` signals that the schedule or controller requested a stop.
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

    Parameters
    ----------
    ctx:
        Fully-populated :class:`~maxent_helpers.run_training_types.TrainingLoopContext`
        describing runtime handles, configurations, logging hooks, and
        controller paths.

    Notes
    -----
    The function is intentionally side-effect free outside of the provided
    handles.  Callers are expected to prepare the dataloader, optimizer, model
    shards, and logging sinks before invoking this entry point.
    """

    runtime = ctx.runtime
    generation_cfg = ctx.generation
    generation_ctx = run_generation.GenerationContext(
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
    completion_generator = run_generation.CompletionGenerator(generation_ctx)
    validation_ctx = ValidationContext(
        evaluation=ctx.evaluation,
        accelerator=runtime.accelerator,
        model=runtime.model,
        reward=ctx.reward,
        generator=completion_generator.generate,
        logging=ctx.logging,
    )
    _configure_accumulation_steps(runtime.accelerator, ctx.optimization.schedule.grad_accum_steps)
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
            except (RuntimeError, ValueError) as exc:  # pragma: no cover - defensive logging
                LOG.warning("Failed to close W&B run cleanly: %s", exc)
