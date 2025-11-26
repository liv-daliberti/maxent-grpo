"""
Copyright 2025 Liv d'Aliberti

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Optimizer and gradient utilities shared across the training loop.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from contextlib import nullcontext
from typing import Any, Optional

try:  # Optional dependency in unit tests
    from accelerate.state import DistributedType
except ImportError:  # pragma: no cover - fallback when accelerate absent

    class DistributedType:
        DEEPSPEED = "deepspeed"


from maxent_grpo.training.runtime import require_torch
from .types import (
    Accelerator,
    OptimizationSchedule,
    OptimizerHandles,
    TrainingLoopContext,
    TrainingLoopState,
)

LOG = logging.getLogger(__name__)
_TWO_NORM = 2.0
torch = require_torch("training_optim")


@dataclass
class DeepspeedState:
    """Describe whether the current accelerator session uses DeepSpeed."""

    use_deepspeed: bool
    zero_stage: int


def clip_grad_norm_local(
    model: Any,
    accelerator: Accelerator,
    max_grad_norm: float,
) -> Optional[float]:
    """Clip gradients via Accelerate when possible and return the norm.

    :param model: Model whose gradients should be clipped.
    :type model: torch.nn.Module
    :param accelerator: Accelerate handle providing ``clip_grad_norm_``.
    :type accelerator: accelerate.Accelerator
    :param max_grad_norm: Maximum norm applied during clipping.
    :type max_grad_norm: float
    :returns: Gradient norm when clipping occurs, otherwise ``None``.
    :rtype: float | None
    """
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
            params, max_grad_norm, norm_type=_TWO_NORM
        )
    if isinstance(grad_norm, torch.Tensor):
        grad_norm = grad_norm.detach().float().cpu().item()
    try:
        return float(grad_norm)
    except (TypeError, ValueError):
        return None


def apply_learning_rate(handles: OptimizerHandles, learning_rate: float) -> None:
    """Set the provided learning rate on all optimizer parameter groups.

    :param handles: Wrapper containing the primary/base optimizers.
    :type handles: training.types.OptimizerHandles
    :param learning_rate: Learning rate to apply across all parameter groups.
    :type learning_rate: float
    """
    for optimizer in (handles.optimizer, handles.base_optimizer):
        param_groups = getattr(optimizer, "param_groups", None)
        if not param_groups:
            continue
        for group in param_groups:
            group["lr"] = learning_rate


def scheduled_learning_rate(
    schedule: OptimizationSchedule, handles: OptimizerHandles, step: int
) -> float:
    """Return the learning rate for the given optimizer step.

    :param schedule: Optimization schedule describing warmup/total steps.
    :type schedule: training.types.OptimizationSchedule
    :param handles: Optimizer handles (used to read base LR).
    :type handles: training.types.OptimizerHandles
    :param step: Current optimizer step index.
    :type step: int
    :returns: Learning rate for this step.
    :rtype: float
    """
    base_lr = handles.learning_rate
    warmup_steps = max(int(getattr(schedule, "warmup_steps", 0)), 0)
    total_steps = max(int(schedule.total_training_steps), 1)
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * (float(step) / float(warmup_steps))
    decay_steps = max(total_steps - warmup_steps, 1)
    progress = min(max(step - warmup_steps, 0), decay_steps) / float(decay_steps)
    return 0.5 * base_lr * (1.0 + math.cos(math.pi * progress))


def optimizer_step(
    ctx: "TrainingLoopContext",
    state: "TrainingLoopState",
    current_lr: float,
) -> Optional[float]:
    """Perform an optimizer step and advance ``state.global_step``.

    :param ctx: Training context containing optimizer handles.
    :type ctx: training.types.TrainingLoopContext
    :param state: Mutable training state tracking global steps.
    :type state: :class:`~training.types.TrainingLoopState`
    :param current_lr: Learning rate to apply before stepping.
    :type current_lr: float
    :returns: Gradient norm (if available) for metrics/logging.
    :rtype: float | None
    """
    accelerator = ctx.runtime.accelerator
    schedule = ctx.optimization.schedule
    handles = ctx.optimization.handles
    grad_norm_scalar = clip_grad_norm_local(
        ctx.runtime.model,
        accelerator,
        float(schedule.max_grad_norm),
    )
    LOG.debug("Optimizer step starting | scheduled_step=%d", state.global_step + 1)
    apply_learning_rate(handles, current_lr)
    optimizer_step_fn = getattr(accelerator, "optimizer_step", None)
    if callable(optimizer_step_fn):
        optimizer_step_fn(handles.optimizer)
    else:
        handles.optimizer.step()
    handles.optimizer.zero_grad(set_to_none=True)
    state.global_step += 1
    LOG.debug("Optimizer step complete | new_global_step=%d", state.global_step)
    return grad_norm_scalar


def epoch_progress(
    schedule: OptimizationSchedule, epoch: int, step_in_epoch: int
) -> float:
    """Return floating-point epoch progress for logging.

    :param schedule: Optimization schedule describing steps per epoch.
    :type schedule: OptimizationSchedule
    :param epoch: Current epoch index (zero-based).
    :type epoch: int
    :param step_in_epoch: Step index inside the current epoch.
    :type step_in_epoch: int
    :returns: Floating-point epoch progress suitable for logs.
    :rtype: float
    """
    if schedule.steps_per_epoch and schedule.steps_per_epoch > 0:
        return float(epoch) + float(step_in_epoch + 1) / float(schedule.steps_per_epoch)
    return float(epoch + 1)


def configure_accumulation_steps(
    accelerator: Accelerator, grad_accum_steps: int
) -> None:
    """Pass gradient accumulation steps to Accelerate when supported.

    :param accelerator: Accelerate handle used to configure accumulation.
    :type accelerator: Accelerator
    :param grad_accum_steps: Desired gradient accumulation steps.
    :type grad_accum_steps: int
    """
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


def detect_deepspeed_state(accelerator: Accelerator) -> DeepspeedState:
    """Return DeepSpeed usage flags derived from the accelerator state."""
    accelerator_state = getattr(accelerator, "state", None)
    ds_plugin = getattr(accelerator_state, "deepspeed_plugin", None)
    zero_stage = int(getattr(ds_plugin, "zero_stage", 0) or 0)
    distributed_type = getattr(accelerator_state, "distributed_type", None)
    use_deepspeed = False
    if distributed_type is not None:
        use_deepspeed = distributed_type == DistributedType.DEEPSPEED
        if not use_deepspeed:
            use_deepspeed = str(distributed_type).lower() == "deepspeed"
    return DeepspeedState(use_deepspeed=use_deepspeed, zero_stage=zero_stage)


def sync_gradients_enabled(accelerator: Accelerator, global_step: int) -> bool:
    """Return the ``sync_gradients`` flag and log it for debugging."""
    syncing = bool(getattr(accelerator, "sync_gradients", True))
    LOG.debug(
        "Backprop complete | sync_gradients=%s | global_step=%d",
        syncing,
        global_step,
    )
    return syncing


def require_accumulation_context(accelerator: Accelerator, model: Any) -> Any:
    """Return an accumulation context compatible with the current strategy."""
    ds_state = detect_deepspeed_state(accelerator)
    if ds_state.use_deepspeed or ds_state.zero_stage >= 2:
        return nullcontext()
    accumulate_fn = getattr(accelerator, "accumulate", None)
    if callable(accumulate_fn):
        return accumulate_fn(model)
    raise RuntimeError(
        "Accelerator.accumulate is unavailable; upgrade Accelerate to match TRL's control flow."
    )


def build_optimization_handles(model: Any, cfg: Any) -> OptimizerHandles:
    """Construct a minimal optimizer/scheduler bundle for the custom runner."""

    optimizer_cls = getattr(torch.optim, "AdamW", None)
    if optimizer_cls is None:
        raise ImportError("torch.optim.AdamW is required for optimization.")
    lr = float(getattr(cfg, "learning_rate", 1e-5))
    params = (
        model.parameters()
        if hasattr(model, "parameters") and callable(getattr(model, "parameters"))
        else []
    )
    optimizer = optimizer_cls(params, lr=lr)
    return OptimizerHandles(
        optimizer=optimizer,
        lr_scheduler=None,
        base_optimizer=optimizer,
        learning_rate=lr,
    )


__all__ = [
    "DeepspeedState",
    "apply_learning_rate",
    "clip_grad_norm_local",
    "configure_accumulation_steps",
    "detect_deepspeed_state",
    "epoch_progress",
    "optimizer_step",
    "require_accumulation_context",
    "scheduled_learning_rate",
    "sync_gradients_enabled",
    "build_optimization_handles",
]
