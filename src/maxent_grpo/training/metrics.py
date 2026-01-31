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

"""Metrics and logging helpers for the MaxEnt-GRPO training loop.

Key entry points
----------------

``log_local_step``
    Emits per-rank metrics for debugging and updates the accumulator used for
    windowed averages.
``log_training_step``
    Aggregates metrics across processes, forwards them to ``wandb`` and/or the
    ``accelerate`` logger, and dumps a structured log line.
``LogStepArtifacts``
    Lightweight container that bundles loss outputs, diagnostics, gradient
    norms, and epoch progress.

The module also exposes helpers for building W&B sample tables,
gathering statistics across ranks, and summarizing reward/weighting diagnostics.
Docstrings follow Sphinx conventions so the documentation clearly describes the
available metrics and their shapes.
"""

from __future__ import annotations

import logging
import sys
import math
import json
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Callable, TYPE_CHECKING

from maxent_grpo.training.runtime.logging import _log_wandb
from maxent_grpo.telemetry.trl_logging import _normalize_prefixes
from .runtime import resolve_run_metadata
from .types import (
    Accelerator,
    BatchDiagnostics,
    LengthStats,
    LogStepArtifacts,
    LoggingHandles,
    LoggingConfigView,
    MetricState,
    OptimizationSchedule,
    RewardComponentStats,
    RewardComputation,
    RewardLoggingView,
    TrainingLoopContext,
    TrainingMetricsPayload,
    TrainingScalarStats,
    TokenUsageStats,
)
from .weighting import WeightLoggingView, WeightStats

if TYPE_CHECKING:  # Avoid importing heavy pipeline/scoring deps at runtime
    from .pipeline import PreparedBatch
    from .weighting.loss import LossOutputs

LOG = logging.getLogger(__name__)
_WANDB_SAMPLE_ROWS = 4
_LOG_STRATEGY_WARNED = {"epoch": False}
_DEBUG_METRIC_FIELDS = (
    ("train/loss", "loss"),
    ("train/reward", "reward"),
    ("train/reward_std", "reward_std"),
    ("train/q_entropy_mean", "q_entropy"),
    ("train/weight_entropy", "w_entropy"),
    ("train/kl", "kl"),
    ("train/tau", "tau"),
    ("train/beta", "beta"),
)


def _log_like_grpo_enabled(training_args: Any) -> bool:
    """Return ``True`` when GRPO-style per-rank logging is requested."""

    flag_val = getattr(training_args, "log_like_grpo", False) if training_args else False
    if isinstance(flag_val, bool):
        return flag_val
    try:
        return bool(flag_val)
    except (TypeError, ValueError):
        return False


def _logging_controls(ctx: TrainingLoopContext) -> tuple[str, int, bool]:
    """Return logging cadence (strategy, steps, first-step flag)."""
    training_args = getattr(ctx, "training_args", None)
    strategy = str(getattr(training_args, "logging_strategy", "steps") or "steps").lower()
    steps = int(getattr(training_args, "logging_steps", 1) or 1)
    first_step = bool(getattr(training_args, "logging_first_step", True))
    if steps <= 0:
        steps = 1
    return strategy, steps, first_step


def _should_log(ctx: TrainingLoopContext, step: int) -> bool:
    """Return True when metrics should be emitted for this step."""
    strategy, steps, first_step = _logging_controls(ctx)
    if strategy in {"no", "none", "off"}:
        return False
    if strategy in {"epoch", "epochs"}:
        if not _LOG_STRATEGY_WARNED["epoch"]:
            LOG.warning("logging_strategy=epoch is not supported in the custom loop; disabling step logs.")
            _LOG_STRATEGY_WARNED["epoch"] = True
        return False
    if step == 0:
        return first_step
    return (step % steps) == 0


def _log_debug_metrics(step: int, metrics: Dict[str, Any]) -> None:
    """Emit a concise debug line with key metrics for the current step."""

    if not LOG.isEnabledFor(logging.DEBUG):
        return
    parts: List[str] = []
    for key, label in _DEBUG_METRIC_FIELDS:
        value = metrics.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            parts.append(f"{label}={float(value):.6f}")
    reward_components: List[str] = []
    for key in sorted(metrics):
        if not key.startswith("train/rewards/"):
            continue
        if not key.endswith("/mean"):
            continue
        short_name = key.split("/")[-2]
        value = metrics.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            reward_components.append(f"{short_name}={float(value):.4f}")
    if reward_components:
        parts.append("rewards[" + ", ".join(reward_components) + "]")
    if not parts:
        parts.append("no-metrics")
    LOG.debug("debug metrics step %d | %s", step, " ".join(parts))

try:  # Optional dependency
    import wandb
except ImportError:  # pragma: no cover - optional logging backend
    wandb = None


class _FallbackWandbError(RuntimeError):
    """Fallback error used when wandb is unavailable."""


WandbError: type[BaseException]
if wandb is not None:
    WandbError = getattr(getattr(wandb, "errors", None), "Error", _FallbackWandbError)
else:
    WandbError = _FallbackWandbError


def _get_wandb() -> Optional[Any]:
    """Return the wandb module when available (facilitates testing)."""
    return wandb


def _mean_std(values: Sequence[float]) -> Tuple[float, float]:
    """Compute mean/std for a list of values.

    :param values: Sequence of numeric samples.
    :type values: Sequence[float]
    :returns: Tuple containing ``(mean, std)``.
    :rtype: tuple[float, float]
    """
    if not values:
        return 0.0, 0.0
    mean_val = float(sum(values) / len(values))
    if len(values) > 1:
        variance = sum((val - mean_val) ** 2 for val in values) / len(values)
        std_val = float(math.sqrt(max(variance, 0.0)))
    else:
        std_val = 0.0
    return mean_val, std_val


def _gather_list_for_metrics(
    accelerator: Accelerator,
    values: Sequence[float],
    *,
    skip_global: bool = False,
) -> List[float]:
    """Gather a sequence of floats across processes.

    :param accelerator: Accelerate handle used for distributed comms.
    :type accelerator: Accelerator
    :param values: Local float values to gather.
    :type values: Sequence[float]
    :returns: Flattened list containing values from all ranks.
    :rtype: list[float]
    """
    local = [float(v) for v in values]
    if skip_global or getattr(accelerator, "num_processes", 1) <= 1:
        return local
    gather_fn = getattr(accelerator, "gather_object", None)
    if not callable(gather_fn):
        return local
    gathered = gather_fn(local)
    if not isinstance(gathered, list):
        return local
    merged: List[float] = []
    for chunk in gathered:
        merged.extend(float(v) for v in chunk)
    return merged


def _gather_dict_of_lists_for_metrics(
    accelerator: Accelerator,
    values: Mapping[str, Sequence[float]],
    *,
    skip_global: bool = False,
) -> Dict[str, List[float]]:
    """Gather dict-of-list structures across processes.

    :param accelerator: Accelerate handle used for distributed comms.
    :type accelerator: Accelerator
    :param values: Mapping of metric name to local float sequence.
    :type values: Mapping[str, Sequence[float]]
    :returns: Mapping where each metric key contains concatenated lists.
    :rtype: dict[str, list[float]]
    """
    if skip_global or getattr(accelerator, "num_processes", 1) <= 1:
        return {key: [float(v) for v in seq] for key, seq in values.items()}
    gather_fn = getattr(accelerator, "gather_object", None)
    if not callable(gather_fn):
        return {key: [float(v) for v in seq] for key, seq in values.items()}
    payload = {key: [float(v) for v in seq] for key, seq in values.items()}
    gathered = gather_fn(payload)
    if not isinstance(gathered, list):
        return {key: [float(v) for v in seq] for key, seq in values.items()}
    merged: Dict[str, List[float]] = {}
    for shard in gathered:
        if not isinstance(shard, dict):
            continue
        for key, seq in shard.items():
            merged.setdefault(key, []).extend(float(v) for v in seq)
    return merged


def _sum_scalar_for_metrics(
    accelerator: Accelerator,
    value: float,
    *,
    skip_global: bool = False,
) -> float:
    """Sum a scalar across all processes.

    :param accelerator: Accelerate handle used for reductions.
    :type accelerator: Accelerator
    :param value: Scalar value contributed by the local rank.
    :type value: float
    :returns: Sum of the scalar across all processes.
    :rtype: float
    """
    return float(
        sum(_gather_list_for_metrics(accelerator, [value], skip_global=skip_global))
    )


def _base_metric_block(
    payload: TrainingMetricsPayload, global_step: int
) -> Dict[str, Any]:
    """Return loss/optimizer scalars that mirror the TRL trainer."""
    scalars = payload.scalars
    total_loss = payload.loss_outputs.total_loss_scalar
    metrics: Dict[str, Any] = {
        "train/loss": total_loss,
        "train/loss/total": total_loss,
        "train/learning_rate": scalars.current_lr,
        "train/epoch": scalars.epoch_progress,
        "train/global_step": float(global_step),
        "train/num_tokens": scalars.num_input_tokens,
        "train/avg_completion_tokens": scalars.avg_completion_tokens,
        "train/ref_logp_mean": scalars.ref_logp_mean,
        "train/beta": payload.config.weighting.beta,
        "train/tau": payload.config.weighting.tau,
        "train/kl_coeff": payload.config.weighting.beta,
        "train/grpo_objective": (
            1.0
            if getattr(payload.config.weighting, "train_grpo_objective", False)
            else 0.0
        ),
    }
    if scalars.num_completion_tokens > 0:
        kl_scalar = getattr(payload.loss_outputs, "kl_loss_scalar", None)
        if kl_scalar is None:
            kl_scalar = getattr(
                getattr(payload.loss_outputs, "scalars", None), "kl_loss", None
            )
        if kl_scalar is not None:
            kl_per_token = float(kl_scalar) / float(scalars.num_completion_tokens)
            metrics["train/kl_per_completion_token"] = max(0.0, kl_per_token)
        loss_per_token = float(payload.loss_outputs.total_loss_scalar) / float(
            scalars.num_completion_tokens
        )
        metrics["train/loss_per_completion_token"] = max(0.0, loss_per_token)
    if scalars.grad_norm_scalar is not None:
        metrics["train/grad_norm"] = scalars.grad_norm_scalar
    if scalars.vllm_latency_ms is not None:
        metrics["train/vllm_latency_ms"] = scalars.vllm_latency_ms
    return metrics


def _loss_component_block(loss_outputs: "LossOutputs") -> Dict[str, float]:
    """Break down the loss into individual components."""
    metrics: Dict[str, float] = {
        "train/loss/policy": loss_outputs.policy_loss_scalar,
        "train/loss/kl": loss_outputs.kl_loss_scalar,
        "train/loss/weighted_kl": loss_outputs.weighted_kl_loss_scalar,
    }
    clip_loss = loss_outputs.clip_loss_scalar
    if clip_loss is not None:
        metrics["train/loss/clip"] = clip_loss
    seed_loss = getattr(loss_outputs, "seed_loss_value", None)
    if seed_loss is not None:
        metrics["train/loss/seed"] = seed_loss
    info_entropy = getattr(loss_outputs, "info_seed_entropy_scalar", None)
    if info_entropy is not None:
        metrics["train/info_seed/entropy"] = info_entropy
    return metrics


def _length_metric_block(length_stats: LengthStats) -> Dict[str, float]:
    """Metrics summarizing completion lengths."""
    # Clamp clipped_ratio to the valid [0, 1] range to avoid noisy negatives.
    clipped_ratio = max(0.0, min(1.0, float(length_stats.clipped_ratio)))
    return {
        "train/completions/mean_length_sampled": length_stats.mean_length,
        "train/completions/min_length_sampled": length_stats.min_length,
        "train/completions/max_length_sampled": length_stats.max_length,
        "train/completions/clipped_frac": clipped_ratio,
        "train/completions/mean_length_terminated": length_stats.mean_terminated,
        "train/completions/min_length_terminated": length_stats.min_terminated,
        "train/completions/max_length_terminated": length_stats.max_terminated,
    }


def _reward_metric_block(payload: TrainingMetricsPayload) -> Dict[str, float]:
    reward_stats = payload.reward_stats
    metrics: Dict[str, float] = {
        "train/reward": reward_stats.reward_mean,
        "train/reward_std": reward_stats.reward_std,
        "train/frac_reward_zero_std": reward_stats.frac_zero_std,
        "train/q_entropy_mean": reward_stats.q_entropy_mean,
        "train/q_entropy_std": reward_stats.q_entropy_std,
        "train/q_entropy_min": reward_stats.q_entropy_min,
        "train/q_entropy_max": reward_stats.q_entropy_max,
    }
    for reward_key, stats in reward_stats.per_reward.items():
        metrics[f"train/rewards/{reward_key}/mean"] = stats.mean
        metrics[f"train/rewards/{reward_key}/std"] = stats.std
    return metrics


def _clip_metric_block(diagnostics: "BatchDiagnostics") -> Dict[str, float]:
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
    bucket_means = getattr(diagnostics, "kl_per_token_by_len_bucket", {}) or {}
    bucket_token_counts = getattr(diagnostics, "kl_token_count_by_len_bucket", {}) or {}
    for bucket in sorted(bucket_means.keys()):
        metrics[f"train/kl_per_token_bucket/{bucket}"] = bucket_means[bucket]
        metrics[f"train/kl_per_token_bucket_tokens/{bucket}"] = bucket_token_counts.get(
            bucket, 0.0
        )
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


def _weighting_config_block(
    payload: TrainingMetricsPayload, global_step: int
) -> Dict[str, float]:
    """Log controller hyperparameters for both GRPO and MaxEnt-GRPO."""
    weighting = payload.config.weighting
    prev_tau = getattr(weighting, "_prev_tau", None)
    prev_beta = getattr(weighting, "_prev_beta", None)
    delta_tau = float(weighting.tau) - float(prev_tau) if prev_tau is not None else 0.0
    delta_beta = (
        float(weighting.beta) - float(prev_beta) if prev_beta is not None else 0.0
    )
    meta_cfg = getattr(weighting, "controller_meta", None)
    meta_enabled = bool(getattr(meta_cfg, "enabled", False))
    tau_lr_effective = getattr(weighting, "_tau_lr_effective", weighting.tau_lr)
    metrics: Dict[str, float] = {
        "train/weight_norm_denom": weighting.denom,
        "train/weighting/tau": float(weighting.tau),
        "train/weighting/beta": float(weighting.beta),
        "train/tau_log": float(
            getattr(weighting, "_tau_log", math.log(max(weighting.tau, 1e-8)))
        ),
        "train/q_temperature": weighting.q_temperature,
        "train/q_epsilon": weighting.q_epsilon,
        "train/tau_lr": float(tau_lr_effective),
        "train/tau_min": weighting.tau_min,
        "train/tau_max": weighting.tau_max,
        "train/tau_warmup_steps": float(weighting.tau_warmup_steps),
        "train/tau_target_entropy": float(
            weighting.tau_target_entropy
            if weighting.tau_target_entropy is not None
            else 0.0
        ),
        "train/tau_target_enabled": (
            1.0 if weighting.tau_target_entropy is not None else 0.0
        ),
        "train/tau_schedule_active": (
            1.0
            if (
                (not meta_enabled)
                and
                weighting.tau_target_entropy is not None
                and global_step > max(0, weighting.tau_warmup_steps)
            )
            else 0.0
        ),
        "train/kl_controller_target": weighting.kl_target,
        "train/kl_controller_horizon": float(weighting.kl_horizon),
        "train/kl_controller_step_size": weighting.kl_ctl_step_size,
        "train/kl_controller_enabled": (
            1.0
            if (not meta_enabled)
            and weighting.kl_target > 0.0
            and weighting.kl_horizon > 0
            and weighting.kl_ctl_step_size > 0.0
            else 0.0
        ),
        "train/len_norm_ref": 1.0 if weighting.len_norm_ref else 0.0,
        "train/maxent_objective": 0.0 if weighting.train_grpo_objective else 1.0,
        "train/delta_tau": delta_tau,
        "train/delta_tau_abs": abs(delta_tau),
        "train/delta_beta": delta_beta,
        "train/delta_beta_abs": abs(delta_beta),
    }
    metrics["train/weighting/weight_norm_denom"] = metrics["train/weight_norm_denom"]
    metrics["train/weighting/tau_log"] = metrics["train/tau_log"]
    metrics["train/weighting/q_temperature"] = metrics["train/q_temperature"]
    metrics["train/weighting/q_epsilon"] = metrics["train/q_epsilon"]
    metrics["train/weighting/tau_lr"] = metrics["train/tau_lr"]
    metrics["train/weighting/tau_min"] = metrics["train/tau_min"]
    metrics["train/weighting/tau_max"] = metrics["train/tau_max"]
    metrics[
        "train/weighting/tau_warmup_steps"
    ] = metrics["train/tau_warmup_steps"]
    metrics[
        "train/weighting/tau_target_entropy"
    ] = metrics["train/tau_target_entropy"]
    metrics[
        "train/weighting/tau_schedule_active"
    ] = metrics["train/tau_schedule_active"]
    metrics["train/weighting/delta_tau"] = metrics["train/delta_tau"]
    metrics["train/weighting/delta_tau_abs"] = metrics["train/delta_tau_abs"]
    metrics["train/weighting/delta_beta"] = metrics["train/delta_beta"]
    metrics["train/weighting/delta_beta_abs"] = metrics["train/delta_beta_abs"]
    # Error-to-target signals for KL and weight entropy controllers.
    kl_measured = payload.diagnostics.kl_value
    if kl_measured is None:
        kl_measured = getattr(payload.loss_outputs, "kl_loss_scalar", None)
    if (
        isinstance(kl_measured, (int, float))
        and weighting.kl_target
        and weighting.kl_target > 0.0
    ):
        metrics["train/kl_error_to_target"] = float(kl_measured) - weighting.kl_target
        metrics["train/kl_ratio_to_target"] = float(kl_measured) / max(
            weighting.kl_target, 1e-8
        )
    target_entropy = weighting.tau_target_entropy
    if target_entropy is not None:
        entropy_error = payload.weight_stats.entropy - float(target_entropy)
        metrics["train/weight_entropy_error"] = entropy_error
        metrics["train/weight_entropy_abs_error"] = abs(entropy_error)
        # Treat the squared error as a simple controller "loss" so it shows up alongside
        # the main model loss in dashboards (e.g., W&B).
        metrics["train/tau_loss"] = 0.5 * entropy_error * entropy_error
    meta_cfg = getattr(weighting, "controller_meta", None)
    meta_enabled = bool(getattr(meta_cfg, "enabled", False))
    metrics["train/meta/enabled"] = 1.0 if meta_enabled else 0.0
    metrics["train/meta/lr"] = float(getattr(meta_cfg, "learning_rate", 0.0)) if meta_cfg else 0.0
    metrics["train/meta/update_interval"] = float(
        getattr(meta_cfg, "update_interval", 0.0) if meta_cfg else 0.0
    )
    metrics["train/meta/truncation_steps"] = float(
        getattr(meta_cfg, "truncation_steps", getattr(meta_cfg, "analytic_steps", 0)) if meta_cfg else 0.0
    )
    metrics["train/meta/use_hessian"] = (
        1.0 if meta_cfg and getattr(meta_cfg, "use_hessian", False) else 0.0
    )
    tau_grad = float(getattr(weighting, "_meta_last_tau_grad", 0.0))
    beta_grad = float(getattr(weighting, "_meta_last_beta_grad", 0.0))
    metrics["train/meta/tau_grad"] = tau_grad
    metrics["train/meta/beta_grad"] = beta_grad
    metrics["train/meta/grad_norm"] = math.sqrt(tau_grad * tau_grad + beta_grad * beta_grad)
    metrics["train/meta/loss"] = float(getattr(weighting, "_meta_last_loss", 0.0))
    metrics["train/meta/tau_projected"] = (
        1.0 if getattr(weighting, "_meta_tau_projected", False) else 0.0
    )
    metrics["train/meta/beta_projected"] = (
        1.0 if getattr(weighting, "_meta_beta_projected", False) else 0.0
    )
    metrics.setdefault(
        "train/weighting/tau_loss", metrics.get("train/tau_loss", 0.0)
    )
    metrics["train/kl_controller/target"] = metrics["train/kl_controller_target"]
    metrics["train/kl_controller/horizon"] = metrics["train/kl_controller_horizon"]
    metrics["train/kl_controller/step_size"] = metrics["train/kl_controller_step_size"]
    metrics["train/kl_controller/enabled"] = metrics["train/kl_controller_enabled"]
    return metrics


def build_training_metrics_dict(
    payload: TrainingMetricsPayload,
    global_step: int,
) -> Dict[str, Any]:
    """Return the flattened metrics dictionary for logging.

    :param payload: Structured metrics payload produced by the training loop.
    :type payload: TrainingMetricsPayload
    :param global_step: Current optimizer step used for logging context.
    :type global_step: int
    :returns: Flat mapping of scalar metrics keyed by name.
    :rtype: dict[str, Any]
    """
    metrics: Dict[str, Any] = {}
    metrics.update(resolve_run_metadata())
    metrics.update(_base_metric_block(payload, global_step))
    metrics.update(_loss_component_block(payload.loss_outputs))
    metrics.update(_length_metric_block(payload.length_stats))
    metrics.update(_reward_metric_block(payload))
    metrics.update(_weight_metric_block(payload))
    metrics.update(_weighting_config_block(payload, global_step))
    metrics.update(_clip_metric_block(payload.diagnostics))
    if "train/kl" not in metrics:
        kl_fallback = getattr(payload.loss_outputs, "kl_loss_scalar", None)
        if isinstance(kl_fallback, (int, float)):
            metrics["train/kl"] = float(kl_fallback)
    if payload.seed_metrics:
        metrics.update({f"train/seed/{k}": v for k, v in payload.seed_metrics.items()})
    return metrics


def log_training_metrics(
    logging_cfg: LoggingHandles,
    global_step: int,
    payload: TrainingMetricsPayload,
) -> Dict[str, Any]:
    """Emit scalar metrics to logging callbacks and return them.

    :param logging_cfg: Logging handles (W&B, tensorboard, stdout, etc.).
    :type logging_cfg: LoggingHandles
    :param global_step: Current optimizer step.
    :type global_step: int
    :param payload: Structured metrics payload to log.
    :type payload: TrainingMetricsPayload
    :returns: Flattened metrics dictionary emitted to loggers.
    :rtype: dict[str, Any]
    """
    metrics = build_training_metrics_dict(payload, global_step)
    logging_cfg.log_metrics(metrics, global_step)
    writer = getattr(logging_cfg, "metric_writer", None)
    flush = getattr(writer, "flush", None)
    if callable(flush):
        flush()
    return metrics


def _reward_component_stats(
    per_reward_values: Mapping[str, Sequence[float]],
) -> Dict[str, RewardComponentStats]:
    """Convert raw reward samples into summary statistics.

    :param per_reward_values: Mapping of reward key to local samples.
    :type per_reward_values: Mapping[str, Sequence[float]]
    :returns: Mapping of reward key to mean/std dataclasses.
    :rtype: dict[str, RewardComponentStats]
    """
    stats: Dict[str, RewardComponentStats] = {}
    for key, values in per_reward_values.items():
        mean_val, std_val = _mean_std([float(v) for v in values])
        stats[key] = RewardComponentStats(mean=mean_val, std=std_val)
    return stats


def _fraction_zero_std_groups(
    accelerator: Accelerator,
    advantage_groups: Sequence[Sequence[float]],
    *,
    skip_global: bool = False,
) -> float:
    """Return the global fraction of zero-variance advantage groups.

    :param accelerator: Accelerate handle used for reductions.
    :type accelerator: Accelerator
    :param advantage_groups: Advantage samples grouped per prompt.
    :type advantage_groups: Sequence[Sequence[float]]
    :returns: Fraction of groups whose advantages are (near) zero variance.
    :rtype: float
    """
    zero_std_local = 0.0
    total_groups_local = 0.0
    for adv_group in advantage_groups:
        if not adv_group:
            continue
        total_groups_local += 1.0
        if all(abs(val) < 1e-8 for val in adv_group):
            zero_std_local += 1.0
    zero_std_total = _sum_scalar_for_metrics(
        accelerator, zero_std_local, skip_global=skip_global
    )
    group_total = _sum_scalar_for_metrics(
        accelerator, total_groups_local, skip_global=skip_global
    )
    return zero_std_total / group_total if group_total > 0 else 0.0


def _summarize_reward_stats(
    accelerator: Accelerator,
    reward_comp: RewardComputation,
    *,
    skip_global: bool = False,
) -> RewardLoggingView:
    """Aggregate reward/advantage stats into a lightweight view.

    :param accelerator: Accelerate handle used for reductions.
    :type accelerator: Accelerator
    :param reward_comp: Reward computation outputs from the batch.
    :type reward_comp: RewardComputation
    :returns: Lightweight logging view containing aggregated stats.
    :rtype: RewardLoggingView
    """
    all_rewards = _gather_list_for_metrics(
        accelerator, reward_comp.total_utils, skip_global=skip_global
    )
    reward_mean, reward_std = _mean_std(all_rewards)
    adv_samples = _gather_list_for_metrics(
        accelerator, reward_comp.advantage_samples, skip_global=skip_global
    )
    adv_mean, adv_std = _mean_std(adv_samples)
    per_reward_values = _gather_dict_of_lists_for_metrics(
        accelerator, reward_comp.per_reward_values, skip_global=skip_global
    )
    # Q-distribution entropy captures how sharp the ranking is per prompt.
    q_grouped = reward_comp.q_grouped
    q_entropies = []
    for q_vals in q_grouped:
        if not q_vals:
            continue
        # Clamp for numerical stability before log.
        entropy = 0.0
        for q in q_vals:
            q_clamped = max(float(q), 1e-12)
            entropy -= q_clamped * math.log(q_clamped)
        q_entropies.append(entropy)
    q_entropies = _gather_list_for_metrics(
        accelerator, q_entropies, skip_global=skip_global
    )
    q_entropy_mean, q_entropy_std = _mean_std(q_entropies)
    q_entropy_min = min(q_entropies) if q_entropies else 0.0
    q_entropy_max = max(q_entropies) if q_entropies else 0.0
    return RewardLoggingView(
        reward_mean=reward_mean,
        reward_std=reward_std,
        frac_zero_std=_fraction_zero_std_groups(
            accelerator, reward_comp.advantage.grouped, skip_global=skip_global
        ),
        advantage_mean=adv_mean,
        advantage_std=adv_std,
        advantage_count=len(adv_samples),
        per_reward=_reward_component_stats(per_reward_values),
        q_entropy_mean=q_entropy_mean,
        q_entropy_std=q_entropy_std,
        q_entropy_min=q_entropy_min,
        q_entropy_max=q_entropy_max,
    )


def summarize_reward_stats(
    accelerator: Accelerator,
    reward_comp: Optional[RewardComputation],
    *,
    log_like_grpo: bool = False,
) -> RewardLoggingView:
    """Aggregate reward statistics across all ranks.

    Exposes the internal helper so that training code can gather reward
    diagnostics even on non-main ranks before metrics are logged.

    :param accelerator: Accelerate handle used for reductions.
    :type accelerator: Accelerator
    :param reward_comp: Reward computation outputs for the current batch.
    :type reward_comp: RewardComputation | None
    :param log_like_grpo: When ``True``, skip global reductions and keep local
        statistics for GRPO-style logging.
    :type log_like_grpo: bool
    :returns: Aggregated reward statistics for logging.
    :rtype: RewardLoggingView
    """

    if reward_comp is None:
        return RewardLoggingView(
            reward_mean=0.0,
            reward_std=0.0,
            frac_zero_std=0.0,
            advantage_mean=0.0,
            advantage_std=0.0,
            advantage_count=0,
            per_reward={},
            q_entropy_mean=0.0,
            q_entropy_std=0.0,
            q_entropy_min=0.0,
            q_entropy_max=0.0,
        )
    return _summarize_reward_stats(accelerator, reward_comp, skip_global=log_like_grpo)


def _summarize_weight_stats(
    accelerator: Accelerator,
    weight_stats: WeightStats,
    *,
    skip_global: bool = False,
) -> WeightLoggingView:
    """Summarize entropy statistics for logging.

    :param accelerator: Accelerate handle used for distributed reductions.
    :type accelerator: accelerate.Accelerator
    :param weight_stats: Per-batch weight diagnostics.
    :type weight_stats: training.types.WeightStats
    :returns: Aggregated entropy metrics per batch.
    :rtype: WeightLoggingView
    """
    weights_grouped = getattr(weight_stats, "weights_grouped", []) or []
    prompt_count = len(weights_grouped)
    entropy_val = float(getattr(weight_stats, "weight_entropy", 0.0))
    entropy_sum = _sum_scalar_for_metrics(
        accelerator, float(entropy_val * max(prompt_count, 0)), skip_global=skip_global
    )
    prompt_total = _sum_scalar_for_metrics(
        accelerator, float(prompt_count), skip_global=skip_global
    )
    entropy_mean = entropy_sum / prompt_total if prompt_total > 0 else entropy_val
    entropy_min_vals = _gather_list_for_metrics(
        accelerator,
        [getattr(weight_stats, "weight_entropy_min", 0.0)],
        skip_global=skip_global,
    )
    entropy_max_vals = _gather_list_for_metrics(
        accelerator,
        [getattr(weight_stats, "weight_entropy_max", 0.0)],
        skip_global=skip_global,
    )
    ent_adv_values = _gather_list_for_metrics(
        accelerator,
        getattr(weight_stats, "advantage_entropy", []),
        skip_global=skip_global,
    )
    ent_adv_mean, ent_adv_std = _mean_std(ent_adv_values)
    return WeightLoggingView(
        entropy=entropy_mean,
        entropy_min=min(entropy_min_vals) if entropy_min_vals else 0.0,
        entropy_max=max(entropy_max_vals) if entropy_max_vals else 0.0,
        advantage_entropy_mean=ent_adv_mean,
        advantage_entropy_std=ent_adv_std,
    )


def summarize_weight_stats(
    accelerator: Accelerator,
    weight_stats: WeightStats,
    *,
    log_like_grpo: bool = False,
) -> WeightLoggingView:
    """Aggregate per-batch weight statistics across all processes.

    Exposes the internal summarization helper so controller logic can rely on
    the same cross-rank entropy measurement used for logging.

    :param accelerator: Accelerate handle used for reductions.
    :type accelerator: Accelerator
    :param weight_stats: Weight statistics for the current batch.
    :type weight_stats: WeightStats
    :param log_like_grpo: When ``True``, skip global reductions and keep local
        statistics for GRPO-style logging.
    :type log_like_grpo: bool
    :returns: Aggregated weight statistics for logging.
    :rtype: WeightLoggingView
    """

    return _summarize_weight_stats(
        accelerator, weight_stats, skip_global=log_like_grpo
    )


def _build_metrics_payload(
    ctx: TrainingLoopContext,
    state: MetricState,
    prepared: PreparedBatch,
    log_artifacts: LogStepArtifacts,
    current_lr: float,
    *,
    reward_view: Optional[RewardLoggingView] = None,
    weight_view: Optional[WeightLoggingView] = None,
) -> TrainingMetricsPayload:
    """Return a structured payload describing the current step.

    :param ctx: Full training loop context.
    :type ctx: training.types.TrainingLoopContext
    :param state: Metric accumulator providing token counts and step numbers.
    :type state: MetricState
    :param prepared: Batch artifacts containing reward/weight stats.
    :type prepared: PreparedBatch
    :param log_artifacts: Loss/diagnostic bundle produced by
        :func:`log_local_step`.
    :type log_artifacts: LogStepArtifacts
    :param current_lr: Learning rate applied for the step (logged for reference).
    :type current_lr: float
    :param reward_view: Optional pre-aggregated reward statistics. When
        ``None`` the helper gathers them across all ranks.
    :type reward_view: RewardLoggingView | None
    :param weight_view: Optional pre-aggregated weight statistics.
    :type weight_view: WeightLoggingView | None
    :returns: Aggregated metrics suitable for logging.
    :rtype: training.types.TrainingMetricsPayload
    """
    accelerator = ctx.runtime.accelerator
    training_args = getattr(ctx, "training_args", None)
    log_like_grpo = _log_like_grpo_enabled(training_args)
    config_view = LoggingConfigView(
        weighting=ctx.scoring.weighting,
        clipping=ctx.scoring.clipping,
        schedule=ctx.optimization.schedule,
    )
    scalar_stats = TrainingScalarStats(
        ref_logp_mean=prepared.ref_stats.ref_logp_mean,
        tokens=TokenUsageStats(
            avg_completion_tokens=prepared.ref_stats.avg_completion_tokens,
            num_completion_tokens=prepared.num_completion_tokens,
            num_input_tokens=state.num_input_tokens_seen,
        ),
        current_lr=current_lr,
        grad_norm_scalar=log_artifacts.grad_norm_scalar,
        epoch_progress=log_artifacts.epoch_progress,
        vllm_latency_ms=(
            float(ctx.generation.generation_stats.get("vllm_last_latency_ms", 0.0))
            if ctx.generation.use_vllm
            else None
        ),
    )
    if log_like_grpo:
        reward_stats_payload = _summarize_reward_stats(
            accelerator, prepared.reward_comp, skip_global=True
        )
        weight_stats_payload = _summarize_weight_stats(
            accelerator, prepared.weight_stats, skip_global=True
        )
    else:
        reward_stats_payload = (
            reward_view
            if reward_view is not None
            else _summarize_reward_stats(accelerator, prepared.reward_comp)
        )
        weight_stats_payload = (
            weight_view
            if weight_view is not None
            else _summarize_weight_stats(accelerator, prepared.weight_stats)
        )
    return TrainingMetricsPayload(
        reward_stats=reward_stats_payload,
        weight_stats=weight_stats_payload,
        loss_outputs=log_artifacts.loss_outputs,
        diagnostics=log_artifacts.diagnostics,
        length_stats=prepared.length_stats,
        config=config_view,
        scalars=scalar_stats,
        seed_metrics=prepared.seed_metrics,
    )


def _epoch_from_global_step(schedule: OptimizationSchedule, global_step: int) -> float:
    """Return the current epoch progress given the training schedule.

    :param schedule: Optimization schedule containing step/epoch metadata.
    :type schedule: OptimizationSchedule
    :param global_step: Current optimizer step.
    :type global_step: int
    :returns: Fractional epoch progress.
    :rtype: float
    """
    steps_per_epoch = getattr(schedule, "steps_per_epoch", None)
    if steps_per_epoch and steps_per_epoch > 0:
        return float(global_step) / float(steps_per_epoch)
    num_generations = getattr(schedule, "num_generations", 0)
    num_epochs = getattr(schedule, "num_epochs", 0)
    total_steps = getattr(schedule, "total_training_steps", 0)
    if num_generations and num_generations > 0:
        return float(global_step) / float(num_generations)
    if total_steps > 0 and num_epochs > 0:
        return float(global_step) * float(num_epochs) / float(total_steps)
    return float(global_step)


def _emit_metrics(
    ctx: TrainingLoopContext,
    metrics: Dict[str, Any],
    global_step: int,
    *,
    log_to_wandb: bool,
    tag: str,
    metric_logger: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """Emit metrics to stdout, Accelerate, and optionally W&B.

    :param ctx: Training context containing logging handles.
    :type ctx: training.types.TrainingLoopContext
    :param metrics: Dictionary of scalar metrics to log.
    :type metrics: dict[str, Any]
    :param global_step: Current optimizer step.
    :type global_step: int
    :param log_to_wandb: Whether to forward the metrics to W&B via
        ``logging_handles``.
    :type log_to_wandb: bool
    :param tag: Human-readable prefix for the info log line.
    :type tag: str
    :returns: The metrics dictionary (unchanged) for convenience.
    :rtype: dict[str, Any]
    """
    accelerator = ctx.runtime.accelerator
    logging_handles = ctx.logging
    if log_to_wandb:
        if metric_logger is not None:
            metric_logger(metrics)
        else:
            logging_handles.log_metrics(metrics, global_step)
        _log_wandb(getattr(logging_handles, "wandb_run", None), metrics, global_step)
        accelerator_log = getattr(accelerator, "log", None)
        if callable(accelerator_log):
            try:
                accelerator_log(metrics, step=global_step)
            except TypeError:
                accelerator_log(metrics)
    elif metric_logger is not None:
        metric_logger(metrics)
    try:
        kv_pairs = " | ".join(f"{key}={metrics[key]}" for key in sorted(metrics.keys()))
    except (TypeError, ValueError, KeyError):
        kv_pairs = str(metrics)
    LOG.info("%s metrics step %d | %s", tag, global_step, kv_pairs)
    return metrics


def _pretty_print_metrics(metrics: Dict[str, Any]) -> str:
    """Return a deterministic, pretty JSON string for human-readable logs."""
    try:
        return json.dumps(metrics, indent=2, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return str(metrics)


def _update_weighting_history(weighting: Any, global_step: int) -> None:
    """Cache the last-seen tau/beta for delta logging."""
    try:
        setattr(weighting, "_prev_tau", float(weighting.tau))
        setattr(weighting, "_prev_beta", float(weighting.beta))
        setattr(weighting, "_prev_step", int(global_step))
    except (AttributeError, TypeError, ValueError):
        return


def accumulate_metrics(state: MetricState, metrics: Dict[str, Any]) -> None:
    """Accumulate per-batch metrics so the global log can show running averages.

    :param state: Mutable metric accumulator storing sums/counts.
    :type state: MetricState
    :param metrics: Scalar metrics emitted for the current step.
    :type metrics: dict[str, Any]
    """
    for key, value in metrics.items():
        if key in {"train/global_step", "train/epoch"}:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        state.metric_sums[key] = state.metric_sums.get(key, 0.0) + numeric
        state.metric_counts[key] = state.metric_counts.get(key, 0) + 1


def flush_metric_averages(state: MetricState) -> Dict[str, float]:
    """Return averaged metrics and clear the accumulator.

    :param state: Metric accumulator to flush.
    :type state: MetricState
    :returns: Mapping of metric name to averaged value.
    :rtype: dict[str, float]
    """
    averaged: Dict[str, float] = {}
    for key, total in state.metric_sums.items():
        count = max(state.metric_counts.get(key, 1), 1)
        averaged[key] = total / float(count)
    state.metric_sums.clear()
    state.metric_counts.clear()
    return averaged


def log_local_step(
    ctx: TrainingLoopContext,
    state: MetricState,
    prepared: PreparedBatch,
    log_artifacts: LogStepArtifacts,
    current_lr: float,
    *,
    reward_view: Optional[RewardLoggingView] = None,
    weight_view: Optional[WeightLoggingView] = None,
    emit: bool = True,
) -> None:
    """Log metrics for the current step on the main process only.

    :param ctx: Full training loop context containing runtime/logging handles.
    :type ctx: training.types.TrainingLoopContext
    :param state: Metric accumulator tracking sums and counts.
    :type state: MetricState
    :param prepared: Prepared batch with reward and weighting statistics.
    :type prepared: PreparedBatch
    :param log_artifacts: Loss outputs and diagnostics emitted by the optimizer step.
    :type log_artifacts: LogStepArtifacts
    :param current_lr: Learning rate applied for the current step.
    :type current_lr: float
    :param reward_view: Optional reward statistics aggregated across ranks.
    :type reward_view: RewardLoggingView | None
    :param weight_view: Optional weight statistics aggregated across ranks.
    :type weight_view: WeightLoggingView | None
    :param emit: When ``False``, skip emitting logs and only accumulate averages.
    :type emit: bool
    """
    accelerator = ctx.runtime.accelerator
    if not accelerator.is_main_process:
        return
    training_args = getattr(ctx, "training_args", None)
    log_like_grpo = _log_like_grpo_enabled(training_args)
    payload = _build_metrics_payload(
        ctx,
        state,
        prepared,
        log_artifacts,
        current_lr,
        reward_view=reward_view,
        weight_view=weight_view,
    )
    metrics = build_training_metrics_dict(payload, state.global_step)
    metrics["train/global_step"] = float(state.global_step)
    accumulate_metrics(state, metrics)
    if not emit:
        return
    if log_like_grpo:
        if not _should_log(ctx, state.global_step):
            return
        averaged_metrics = flush_metric_averages(state)
        if averaged_metrics:
            metrics_to_emit = dict(metrics)
            metrics_to_emit.update(averaged_metrics)
        else:
            metrics_to_emit = dict(metrics)
        if "train/epoch" not in metrics_to_emit:
            metrics_to_emit["train/epoch"] = _epoch_from_global_step(
                ctx.optimization.schedule,
                state.global_step,
            )
        metrics_to_emit.setdefault("train/global_step", float(state.global_step))
        _log_debug_metrics(state.global_step, metrics_to_emit)
        normalized_metrics = _normalize_prefixes(dict(metrics_to_emit), is_eval=False)
        with ctx.logging.step_logger(state.global_step, enabled=True) as step_logger:
            _emit_metrics(
                ctx,
                normalized_metrics,
                state.global_step,
                log_to_wandb=True,
                tag="Global",
                metric_logger=getattr(step_logger, "log", None),
            )
        _update_weighting_history(ctx.scoring.weighting, state.global_step)
        if training_args is None:
            log_completions = True
        else:
            log_completions = getattr(training_args, "log_completions", False)
        if log_completions:
            _log_sample_table(ctx, state, prepared)
        return
    _log_debug_metrics(state.global_step, metrics)
    if not _should_log(ctx, state.global_step):
        return
    with ctx.logging.step_logger(state.global_step, enabled=True) as step_logger:
        _emit_metrics(
            ctx,
            metrics,
            state.global_step,
            log_to_wandb=False,
            tag="Local",
            metric_logger=getattr(step_logger, "log", None),
        )


def _build_sample_table(
    prepared: PreparedBatch,
    step: int,
    max_rows: int,
) -> Tuple[List[str], List[List[Any]]]:
    """Return W&B table columns/rows for sample completions.

    :param prepared: Batch artifacts used to extract prompts/completions.
    :type prepared: PreparedBatch
    :param step: Global step used in the W&B table rows.
    :type step: int
    :param max_rows: Maximum number of rows to include in the table.
    :type max_rows: int
    :returns: Tuple containing table columns and row data.
    :rtype: tuple[list[str], list[list[Any]]]
    """
    pairs = prepared.reward_comp.pairs
    prompts = pairs.prompts
    completions = pairs.completions
    reward_values = prepared.reward_comp.per_reward_values
    reward_keys = sorted(reward_values.keys())
    advantages = prepared.reward_comp.advantage_samples
    columns = ["step", "prompt", "completion", "advantage"] + [
        f"reward/{key}" for key in reward_keys
    ]
    rows: List[List[Any]] = []
    for idx in range(max_rows):
        prompt = prompts[idx]
        completion = completions[idx]
        advantage_val = (
            float(advantages[idx]) if idx < len(advantages) else float("nan")
        )
        row: List[Any] = [step, prompt, completion, advantage_val]
        for key in reward_keys:
            values = reward_values.get(key, [])
            reward_val = float(values[idx]) if idx < len(values) else float("nan")
            row.append(reward_val)
        rows.append(row)
    return columns, rows


def _log_sample_table(
    ctx: TrainingLoopContext,
    state: MetricState,
    prepared: PreparedBatch,
) -> None:
    """Log a W&B table with prompt/completion samples when enabled.

    :param ctx: Training context providing logging handles and accelerator state.
    :type ctx: training.types.TrainingLoopContext
    :param state: Metric state containing the global step for table rows.
    :type state: MetricState
    :param prepared: Batch artifacts whose ``RewardComputation`` holds the prompts,
        completions, and reward components to display.
    :type prepared: PreparedBatch
    """
    wandb_run = ctx.logging.wandb_run
    accelerator = ctx.runtime.accelerator
    wandb_mod = _get_wandb()
    if wandb_mod is None and "wandb" in sys.modules:
        wandb_mod = sys.modules["wandb"]
    if wandb_mod is None:

        class _FallbackWandb:
            def Table(
                self,
                columns: Any = None,
                rows: Any = None,
                **_kwargs: Any,
            ) -> Dict[str, Any]:
                return {"columns": columns, "rows": rows}

        wandb_mod = _FallbackWandb()
    if wandb_run is None or not accelerator.is_main_process:
        return
    pairs = prepared.reward_comp.pairs
    if not pairs.prompts or not pairs.completions:
        return
    max_rows = min(len(pairs.prompts), len(pairs.completions), _WANDB_SAMPLE_ROWS)
    if max_rows <= 0:
        return
    columns, rows = _build_sample_table(prepared, state.global_step, max_rows)
    if not rows:
        return
    table = wandb_mod.Table(columns=columns, rows=rows)
    try:
        wandb_run.log({"completions": table}, step=state.global_step)
    except WandbError:
        return


def log_training_step(
    ctx: TrainingLoopContext,
    state: MetricState,
    prepared: PreparedBatch,
    log_artifacts: LogStepArtifacts,
    current_lr: float,
    *,
    reward_view: Optional[RewardLoggingView] = None,
    weight_view: Optional[WeightLoggingView] = None,
) -> None:
    """Emit global metrics (including optional W&B logging).

    :param ctx: Training context containing runtime/logging handles.
    :type ctx: training.types.TrainingLoopContext
    :param state: Metric accumulator tracking running averages.
    :type state: MetricState
    :param prepared: Batch artifacts with reward/weight statistics.
    :type prepared: PreparedBatch
    :param log_artifacts: Loss outputs and diagnostics for the step.
    :type log_artifacts: LogStepArtifacts
    :param current_lr: Learning rate applied for the current step.
    :type current_lr: float
    :param reward_view: Optional reward statistics aggregated across ranks.
    :type reward_view: RewardLoggingView | None
    :param weight_view: Optional weight statistics aggregated across ranks.
    :type weight_view: WeightLoggingView | None
    """
    training_args = getattr(ctx, "training_args", None)
    if _log_like_grpo_enabled(training_args):
        return
    if not _should_log(ctx, state.global_step):
        return
    accelerator = ctx.runtime.accelerator
    if accelerator.is_main_process:
        LOG.info(
            "step %d | epoch %.2f | loss=%.4f | tau=%.3f beta=%.3f",
            state.global_step,
            log_artifacts.epoch_progress,
            log_artifacts.loss_outputs.total_loss_scalar,
            ctx.scoring.weighting.tau,
            ctx.scoring.weighting.beta,
        )
    averaged_metrics = flush_metric_averages(state)
    if averaged_metrics:
        averaged_metrics["train/global_step"] = float(state.global_step)
        averaged_metrics["train/epoch"] = _epoch_from_global_step(
            ctx.optimization.schedule,
            state.global_step,
        )
        metrics = averaged_metrics
    else:
        payload = _build_metrics_payload(
            ctx,
            state,
            prepared,
            log_artifacts,
            current_lr,
            reward_view=reward_view,
            weight_view=weight_view,
        )
        metrics = build_training_metrics_dict(payload, state.global_step)
        metrics["train/global_step"] = float(state.global_step)
    with ctx.logging.step_logger(state.global_step, enabled=True) as step_logger:
        _emit_metrics(
            ctx,
            metrics,
            state.global_step,
            log_to_wandb=True,
            tag="Global",
            metric_logger=getattr(step_logger, "log", None),
        )
    if accelerator.is_main_process:
        pretty = _pretty_print_metrics(metrics)
        LOG.info("Global metrics (pretty) step %d\n%s", state.global_step, pretty)
    _update_weighting_history(ctx.scoring.weighting, state.global_step)
    if training_args is None:
        log_completions = True
    else:
        log_completions = getattr(training_args, "log_completions", False)
    if log_completions:
        _log_sample_table(ctx, state, prepared)


__all__ = [
    "LogStepArtifacts",
    "accumulate_metrics",
    "build_training_metrics_dict",
    "flush_metric_averages",
    "log_local_step",
    "log_training_metrics",
    "log_training_step",
    "summarize_reward_stats",
    "summarize_weight_stats",
]
