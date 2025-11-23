"""Logging utilities for the MaxEnt-GRPO training loop."""

from __future__ import annotations

from typing import Any, Dict

from .run_training_types import (
    BatchDiagnostics,
    LengthStats,
    LoggingHandles,
    TrainingMetricsPayload,
)


def _base_metric_block(payload: TrainingMetricsPayload, global_step: int) -> Dict[str, Any]:
    """Return loss/optimizer scalars that mirror the TRL trainer."""
    scalars = payload.scalars
    metrics: Dict[str, Any] = {
        "train/loss": payload.loss_outputs.total_loss_scalar,
        "train/learning_rate": scalars.current_lr,
        "train/epoch": scalars.epoch_progress,
        "train/global_step": float(global_step),
        "train/num_tokens": scalars.num_input_tokens,
        "train/avg_completion_tokens": scalars.avg_completion_tokens,
        "train/beta": payload.config.weighting.beta,
        "train/tau": payload.config.weighting.tau,
        "train/grpo_objective": 1.0
        if getattr(payload.config.weighting, "train_grpo_objective", False)
        else 0.0,
    }
    if scalars.num_completion_tokens > 0:
        kl_scalar = getattr(payload.loss_outputs, "kl_loss_scalar", None)
        if kl_scalar is None:
            kl_scalar = getattr(getattr(payload.loss_outputs, "scalars", None), "kl_loss", None)
        if kl_scalar is not None:
            kl_per_token = float(kl_scalar) / float(scalars.num_completion_tokens)
            metrics["train/kl_per_completion_token"] = max(0.0, kl_per_token)
        loss_per_token = (
            float(payload.loss_outputs.total_loss_scalar) / float(scalars.num_completion_tokens)
        )
        metrics["train/loss_per_completion_token"] = max(0.0, loss_per_token)
    if scalars.grad_norm_scalar is not None:
        metrics["train/grad_norm"] = scalars.grad_norm_scalar
    if scalars.vllm_latency_ms is not None:
        metrics["train/vllm_latency_ms"] = scalars.vllm_latency_ms
    return metrics


def _length_metric_block(length_stats: LengthStats) -> Dict[str, float]:
    """Metrics summarizing completion lengths."""
    return {
        "train/completions/mean_length": length_stats.mean_length,
        "train/completions/min_length": length_stats.min_length,
        "train/completions/max_length": length_stats.max_length,
        "train/completions/clipped_ratio": length_stats.clipped_ratio,
        "train/completions/mean_terminated_length": length_stats.mean_terminated,
        "train/completions/min_terminated_length": length_stats.min_terminated,
        "train/completions/max_terminated_length": length_stats.max_terminated,
    }


def _reward_metric_block(payload: TrainingMetricsPayload) -> Dict[str, float]:
    reward_stats = payload.reward_stats
    metrics: Dict[str, float] = {
        "train/reward": reward_stats.reward_mean,
        "train/reward_std": reward_stats.reward_std,
        "train/frac_reward_zero_std": reward_stats.frac_zero_std,
    }
    for reward_key, stats in reward_stats.per_reward.items():
        metrics[f"train/rewards/{reward_key}/mean"] = stats.mean
        metrics[f"train/rewards/{reward_key}/std"] = stats.std
    return metrics


def _clip_metric_block(diagnostics: BatchDiagnostics) -> Dict[str, float]:
    """Return PPO-style clipping diagnostics."""
    metrics: Dict[str, float] = {
        "train/clip_ratio": diagnostics.clip_ratio,
        "train/clip_ratio/low_mean": diagnostics.clip_ratio_low_mean,
        "train/clip_ratio/low_min": diagnostics.clip_ratio_low_min,
        "train/clip_ratio/high_mean": diagnostics.clip_ratio_high_mean,
        "train/clip_ratio/high_max": diagnostics.clip_ratio_high_max,
        "train/clip_ratio/region_mean": diagnostics.clip_ratio_region_mean,
    }
    if diagnostics.kl_value is not None:
        metrics["train/kl"] = diagnostics.kl_value
    return metrics


def _weight_metric_block(payload: TrainingMetricsPayload) -> Dict[str, float]:
    """Entropy diagnostics for the MaxEnt weighting distribution."""
    weight_stats = payload.weight_stats
    metrics = {
        "train/weight_entropy": weight_stats.entropy,
        "train/weight_entropy_min": weight_stats.entropy_min,
        "train/weight_entropy_max": weight_stats.entropy_max,
        "train/advantage_entropy_mean": weight_stats.advantage_entropy_mean,
        "train/advantage_entropy_std": weight_stats.advantage_entropy_std,
    }
    entropy_ema = getattr(payload.config.weighting, "_tau_entropy_ema", None)
    if isinstance(entropy_ema, (int, float)):
        metrics["train/weight_entropy_ema"] = float(entropy_ema)
    return metrics


def build_training_metrics_dict(
    payload: TrainingMetricsPayload,
    global_step: int,
) -> Dict[str, Any]:
    """Return the flattened metrics dictionary for logging."""
    metrics: Dict[str, Any] = {}
    metrics.update(_base_metric_block(payload, global_step))
    metrics.update(_length_metric_block(payload.length_stats))
    metrics.update(_reward_metric_block(payload))
    metrics.update(_weight_metric_block(payload))
    metrics.update(_clip_metric_block(payload.diagnostics))
    return metrics


def log_training_metrics(
    logging_cfg: LoggingHandles,
    global_step: int,
    payload: TrainingMetricsPayload,
) -> Dict[str, Any]:
    """Emit scalar metrics to logging callbacks and return them."""
    metrics = build_training_metrics_dict(payload, global_step)
    logging_cfg.log_metrics(metrics, global_step)
    return metrics


__all__ = ["log_training_metrics", "build_training_metrics_dict"]
