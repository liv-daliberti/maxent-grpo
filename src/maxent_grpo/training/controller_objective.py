"""Meta-controller objectives for tau/beta adaptation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional

from maxent_grpo.config import GRPOConfig
from .weighting.types import WeightingSettings

LOG = logging.getLogger(__name__)


@dataclass
class ControllerGradients:
    """Gradient bundle returned by controller objectives."""

    tau_grad: Optional[float] = None
    beta_grad: Optional[float] = None

    def has_updates(self) -> bool:
        return any(
            isinstance(val, (int, float)) for val in (self.tau_grad, self.beta_grad)
        )


@dataclass
class ControllerMetaContext:
    """Inputs made available to controller objectives."""

    weighting: WeightingSettings
    weight_stats: Any
    loss_outputs: Any
    global_step: int
    lr_scale: float = 1.0
    prepared_batch: Any = None
    kl_value: Optional[float] = None
    backprop_fn: Optional[Callable[[int], Optional[ControllerGradients]]] = None

    def entropy_value(self) -> Optional[float]:
        """Return the batch entropy used for tau updates (handles logging views)."""

        for attr in ("weight_entropy", "entropy"):
            value = getattr(self.weight_stats, attr, None)
            if isinstance(value, (int, float)):
                return float(value)
        return None

    def kl_metric(self) -> Optional[float]:
        """Return the KL metric supplied by the loss or fallback to cached value."""

        if isinstance(self.kl_value, (int, float)):
            return float(self.kl_value)
        val = getattr(self.loss_outputs, "kl_loss_scalar", None)
        if isinstance(val, (int, float)):
            return float(val)
        return None


class ControllerObjective:
    """Base class for controller objectives."""

    name = "base"

    def compute(self, meta_ctx: ControllerMetaContext) -> Optional[ControllerGradients]:
        raise NotImplementedError


class AnalyticControllerObjective(ControllerObjective):
    """Closed-form gradients based on entropy/KL targets."""

    name = "analytic"

    def compute(self, meta_ctx: ControllerMetaContext) -> Optional[ControllerGradients]:
        gradients = ControllerGradients()
        entropy_val = meta_ctx.entropy_value()
        target_entropy = meta_ctx.weighting.tau_target_entropy
        if entropy_val is not None and target_entropy is not None:
            gradients.tau_grad = entropy_val - float(target_entropy)
        kl_val = meta_ctx.kl_metric()
        target_kl = meta_ctx.weighting.kl_target
        if kl_val is not None and target_kl > 0:
            gradients.beta_grad = kl_val - float(target_kl)
        return gradients if gradients.has_updates() else None


class TruncatedBackpropControllerObjective(ControllerObjective):
    """Truncated meta-gradient objective relying on a user-supplied callback."""

    name = "truncated_backprop"

    def __init__(self, steps: int = 1):
        self.steps = max(1, int(steps))

    def compute(self, meta_ctx: ControllerMetaContext) -> Optional[ControllerGradients]:
        backprop_fn = meta_ctx.backprop_fn
        if callable(backprop_fn):
            try:
                result = backprop_fn(self.steps)
            except RuntimeError as exc:
                LOG.warning("Controller backprop callback failed: %s", exc)
                result = None
            if result and result.has_updates():
                return result
        # Fallback to analytic gradients so the controller still makes progress.
        return AnalyticControllerObjective().compute(meta_ctx)


def build_controller_objective(
    cfg: GRPOConfig, weighting: WeightingSettings
) -> Optional[ControllerObjective]:
    """Return the configured controller objective for the current run."""

    del cfg  # legacy argument retained for compatibility
    meta_cfg = getattr(weighting, "controller_meta", None)
    if meta_cfg is None or not getattr(meta_cfg, "enabled", False):
        return None
    method = str(getattr(meta_cfg, "method", "analytic") or "analytic").lower()
    if method in ("analytic", "analytic_grad", "potential"):
        return AnalyticControllerObjective()
    if method in ("truncated", "truncated_backprop", "backprop"):
        steps = getattr(meta_cfg, "truncation_steps", None)
        if steps is None or steps <= 0:
            steps = getattr(meta_cfg, "analytic_steps", 1)
        return TruncatedBackpropControllerObjective(steps=steps)
    LOG.warning(
        "Unknown controller_meta_method=%s; falling back to analytic gradients.",
        method,
    )
    return AnalyticControllerObjective()


__all__ = [
    "AnalyticControllerObjective",
    "ControllerGradients",
    "ControllerMetaContext",
    "ControllerObjective",
    "TruncatedBackpropControllerObjective",
    "build_controller_objective",
]
