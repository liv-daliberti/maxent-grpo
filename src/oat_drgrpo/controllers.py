"""Adaptive controller helpers for listwise tau and beta."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import torch

_TAU_METRIC_EMA_DECAY = 0.9


@dataclass
class ListwiseControllerState:
    """Mutable controller history used by adaptive tau updates."""

    tau_metric_ema: float | None = None
    tau_log: float | None = None


def resolve_listwise_target_entropy(
    *,
    target_entropy: float | None,
    target_entropy_start: float | None,
    target_entropy_peak: float | None,
    target_entropy_peak_step: int,
    target_entropy_final: float | None,
    target_entropy_horizon: int,
    global_step: int,
) -> Optional[float]:
    """Return the active target entropy, honoring optional annealing settings."""

    if (
        target_entropy is None
        and target_entropy_start is None
        and target_entropy_peak is None
        and target_entropy_final is None
    ):
        return None
    if (
        target_entropy_start is None
        and target_entropy_peak is None
        and target_entropy_final is None
    ):
        return float(target_entropy) if target_entropy is not None else None
    start = (
        float(target_entropy_start)
        if target_entropy_start is not None
        else float(target_entropy)
    )
    peak = float(target_entropy_peak) if target_entropy_peak is not None else None
    final = (
        float(target_entropy_final)
        if target_entropy_final is not None
        else float(target_entropy)
    )
    if not math.isfinite(start) or not math.isfinite(final):
        return None
    if peak is not None and not math.isfinite(peak):
        return None
    horizon = max(int(target_entropy_horizon), 0)
    peak_step = max(int(target_entropy_peak_step), 0)
    step = max(int(global_step), 0)
    if peak is None:
        if horizon <= 0:
            return final
        frac = min(step, horizon) / float(horizon)
        return start + (final - start) * frac
    if horizon <= 0:
        return final
    if peak_step <= 0:
        if step >= horizon:
            return final
        down_frac = min(step, horizon) / float(horizon)
        return peak + (final - peak) * down_frac
    if step <= peak_step:
        up_frac = min(step, peak_step) / float(peak_step)
        return start + (peak - start) * up_frac
    if horizon <= peak_step:
        return peak
    down_frac = min(step - peak_step, horizon - peak_step) / float(horizon - peak_step)
    return peak + (final - peak) * down_frac


def update_listwise_tau_metric_ema(
    state: ListwiseControllerState | None,
    *,
    measured_metric: float | None,
) -> float | None:
    """Update and return the smoothed controller statistic for tau adaptation."""

    if state is None:
        return None
    if not isinstance(measured_metric, (int, float)) or not math.isfinite(
        float(measured_metric)
    ):
        return None
    if not isinstance(state.tau_metric_ema, (int, float)) or not math.isfinite(
        state.tau_metric_ema
    ):
        state.tau_metric_ema = float(measured_metric)
    else:
        state.tau_metric_ema = _TAU_METRIC_EMA_DECAY * float(state.tau_metric_ema) + (
            1.0 - _TAU_METRIC_EMA_DECAY
        ) * float(measured_metric)
    return float(state.tau_metric_ema)


def update_listwise_tau_entropy_ema(
    state: ListwiseControllerState | None,
    *,
    measured_entropy: float | None,
) -> float | None:
    """Backward-compatible alias for the old tau-entropy EMA helper."""

    return update_listwise_tau_metric_ema(
        state,
        measured_metric=measured_entropy,
    )


def clamp_listwise_tau(
    current_tau: float,
    *,
    tau_min: float,
    tau_max: float,
) -> float:
    """Project tau into the configured positive range."""

    new_tau = max(float(current_tau), max(float(tau_min), 1e-8))
    safe_tau_max = float(tau_max)
    if math.isfinite(safe_tau_max) and safe_tau_max > 0.0:
        new_tau = min(new_tau, safe_tau_max)
    return float(new_tau)


def compute_learnable_tau_loss(
    tau_log: torch.Tensor,
    *,
    measured_metric: float | torch.Tensor | None,
    target_metric: float | torch.Tensor | None,
) -> torch.Tensor | None:
    """Return the SAC-style log-tau objective for scalar-metric matching."""

    if measured_metric is None or target_metric is None:
        return None
    if tau_log.numel() != 1:
        raise ValueError("tau_log must contain exactly one scalar value")

    measured = torch.as_tensor(
        measured_metric,
        device=tau_log.device,
        dtype=tau_log.dtype,
    )
    target = torch.as_tensor(
        target_metric,
        device=tau_log.device,
        dtype=tau_log.dtype,
    )
    if not bool(torch.isfinite(measured).all()) or not bool(
        torch.isfinite(target).all()
    ):
        return None
    return tau_log.reshape(()) * (measured.detach() - target.detach())


def maybe_update_listwise_tau(
    current_tau: float,
    *,
    measured_metric: float | None,
    global_step: int,
    state: ListwiseControllerState | None,
    target_metric: float | None,
    target_metric_start: float | None,
    target_metric_peak: float | None,
    target_metric_peak_step: int,
    target_metric_final: float | None,
    target_metric_horizon: int,
    tau_lr: float,
    tau_min: float,
    tau_max: float,
    tau_warmup_steps: int,
) -> float:
    """Return the next tau under the simple scalar-target controller."""

    active_target = resolve_listwise_target_entropy(
        target_entropy=target_metric,
        target_entropy_start=target_metric_start,
        target_entropy_peak=target_metric_peak,
        target_entropy_peak_step=target_metric_peak_step,
        target_entropy_final=target_metric_final,
        target_entropy_horizon=target_metric_horizon,
        global_step=global_step,
    )
    if active_target is None:
        return float(current_tau)
    if global_step <= max(0, int(tau_warmup_steps)):
        return float(current_tau)
    if not isinstance(measured_metric, (int, float)) or not math.isfinite(
        float(measured_metric)
    ):
        return float(current_tau)
    safe_tau_lr = float(tau_lr)
    if not math.isfinite(safe_tau_lr) or safe_tau_lr <= 0.0:
        return float(current_tau)

    if state is None:
        state = ListwiseControllerState()
    if not isinstance(state.tau_log, (int, float)) or not math.isfinite(state.tau_log):
        state.tau_log = math.log(max(float(current_tau), 1e-8))
    ema_metric = update_listwise_tau_metric_ema(
        state,
        measured_metric=float(measured_metric),
    )
    if ema_metric is None:
        return float(current_tau)

    error = float(active_target) - float(ema_metric)
    if abs(error) < 1e-12:
        return float(current_tau)
    tau_log = float(state.tau_log) + safe_tau_lr * error
    new_tau = clamp_listwise_tau(
        math.exp(tau_log),
        tau_min=tau_min,
        tau_max=tau_max,
    )
    state.tau_log = math.log(max(new_tau, 1e-8))
    return float(new_tau)


def maybe_update_listwise_beta(
    current_beta: float,
    *,
    measured_kl: float | None,
    kl_target: float,
    kl_horizon: int,
    kl_ctl_step_size: float,
) -> float:
    """Return the next beta under the simple KL controller."""

    safe_target = float(kl_target)
    safe_horizon = int(kl_horizon)
    safe_step = float(kl_ctl_step_size)
    if safe_target <= 0.0 or safe_horizon <= 0 or safe_step <= 0.0:
        return float(current_beta)
    if not isinstance(measured_kl, (int, float)) or not math.isfinite(
        float(measured_kl)
    ):
        return float(current_beta)

    ratio = float(measured_kl) / max(safe_target, 1e-8)
    error = ratio - 1.0
    if abs(error) < 1e-8:
        return float(current_beta)
    clipped_error = max(min(error, safe_step), -safe_step)
    scale = 1.0 + clipped_error / float(max(safe_horizon, 1))
    if scale <= 0.0:
        scale = 1e-6
    return max(0.0, float(current_beta) * scale)
