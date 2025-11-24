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
from typing import Any, Dict, List, Optional, Sequence, Tuple, Callable

from .pipeline import PreparedBatch
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

LOG = logging.getLogger(__name__)
_WANDB_SAMPLE_ROWS = 4

try:  # Optional dependency
    import wandb
    from wandb.errors import Error as WandbError
except ImportError:  # pragma: no cover - optional logging backend
    wandb = None

    class WandbError(RuntimeError):
        """Fallback error used when wandb is unavailable."""


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
    if getattr(accelerator, "num_processes", 1) <= 1:
        return local
    gather_fn = getattr(accelerator, "gather_object", None)
    if not callable(gather_fn):
        return local
    gathered = gather_fn(local)
    merged: List[float] = []
    for chunk in gathered:
        merged.extend(float(v) for v in chunk)
    return merged


def _gather_dict_of_lists_for_metrics(
    accelerator: Accelerator,
    values: Dict[str, Sequence[float]],
) -> Dict[str, List[float]]:
    """Gather dict-of-list structures across processes.

    :param accelerator: Accelerate handle used for distributed comms.
    :type accelerator: Accelerator
    :param values: Mapping of metric name to local float sequence.
    :type values: dict[str, Sequence[float]]
    :returns: Mapping where each metric key contains concatenated lists.
    :rtype: dict[str, list[float]]
    """
    if getattr(accelerator, "num_processes", 1) <= 1:
        return {key: [float(v) for v in seq] for key, seq in values.items()}
    gather_fn = getattr(accelerator, "gather_object", None)
    if not callable(gather_fn):
        return {key: [float(v) for v in seq] for key, seq in values.items()}
    payload = {key: [float(v) for v in seq] for key, seq in values.items()}
    gathered = gather_fn(payload)
    merged: Dict[str, List[float]] = {}
    for shard in gathered:
        for key, seq in shard.items():
            merged.setdefault(key, []).extend(float(v) for v in seq)
    return merged


def _sum_scalar_for_metrics(accelerator: Accelerator, value: float) -> float:
    """Sum a scalar across all processes.

    :param accelerator: Accelerate handle used for reductions.
    :type accelerator: Accelerator
    :param value: Scalar value contributed by the local rank.
    :type value: float
    :returns: Sum of the scalar across all processes.
    :rtype: float
    """
    return float(sum(_gather_list_for_metrics(accelerator, [value])))


def _base_metric_block(
    payload: TrainingMetricsPayload, global_step: int
) -> Dict[str, Any]:
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
    writer = getattr(logging_cfg, "metric_writer", None)
    flush = getattr(writer, "flush", None)
    if callable(flush):
        flush()
    return metrics


def _reward_component_stats(
    per_reward_values: Dict[str, Sequence[float]],
) -> Dict[str, RewardComponentStats]:
    """Convert raw reward samples into summary statistics.

    :param per_reward_values: Mapping of reward key to local samples.
    :type per_reward_values: dict[str, Sequence[float]]
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
    zero_std_total = _sum_scalar_for_metrics(accelerator, zero_std_local)
    group_total = _sum_scalar_for_metrics(accelerator, total_groups_local)
    return zero_std_total / group_total if group_total > 0 else 0.0


def _summarize_reward_stats(
    accelerator: Accelerator,
    reward_comp: RewardComputation,
) -> RewardLoggingView:
    """Aggregate reward/advantage stats into a lightweight view.

    :param accelerator: Accelerate handle used for reductions.
    :type accelerator: Accelerator
    :param reward_comp: Reward computation outputs from the batch.
    :type reward_comp: RewardComputation
    :returns: Lightweight logging view containing aggregated stats.
    :rtype: RewardLoggingView
    """
    all_rewards = _gather_list_for_metrics(accelerator, reward_comp.total_utils)
    reward_mean, reward_std = _mean_std(all_rewards)
    adv_samples = _gather_list_for_metrics(accelerator, reward_comp.advantage_samples)
    adv_mean, adv_std = _mean_std(adv_samples)
    per_reward_values = _gather_dict_of_lists_for_metrics(
        accelerator, reward_comp.per_reward_values
    )
    return RewardLoggingView(
        reward_mean=reward_mean,
        reward_std=reward_std,
        frac_zero_std=_fraction_zero_std_groups(
            accelerator, reward_comp.advantage.grouped
        ),
        advantage_mean=adv_mean,
        advantage_std=adv_std,
        advantage_count=len(adv_samples),
        per_reward=_reward_component_stats(per_reward_values),
    )


def _summarize_weight_stats(
    accelerator: Accelerator,
    weight_stats: WeightStats,
) -> WeightLoggingView:
    """Summarize entropy statistics for logging.

    :param accelerator: Accelerate handle used for distributed reductions.
    :type accelerator: accelerate.Accelerator
    :param weight_stats: Per-batch weight diagnostics.
    :type weight_stats: training.types.WeightStats
    :returns: Aggregated entropy metrics per batch.
    :rtype: WeightLoggingView
    """
    prompt_count = len(weight_stats.weights_grouped)
    entropy_sum = _sum_scalar_for_metrics(
        accelerator, float(weight_stats.weight_entropy * max(prompt_count, 0))
    )
    prompt_total = _sum_scalar_for_metrics(accelerator, float(prompt_count))
    entropy_mean = (
        entropy_sum / prompt_total if prompt_total > 0 else weight_stats.weight_entropy
    )
    entropy_min_vals = _gather_list_for_metrics(
        accelerator, [weight_stats.weight_entropy_min]
    )
    entropy_max_vals = _gather_list_for_metrics(
        accelerator, [weight_stats.weight_entropy_max]
    )
    ent_adv_values = _gather_list_for_metrics(
        accelerator, weight_stats.advantage_entropy
    )
    ent_adv_mean, ent_adv_std = _mean_std(ent_adv_values)
    return WeightLoggingView(
        entropy=entropy_mean,
        entropy_min=min(entropy_min_vals) if entropy_min_vals else 0.0,
        entropy_max=max(entropy_max_vals) if entropy_max_vals else 0.0,
        advantage_entropy_mean=ent_adv_mean,
        advantage_entropy_std=ent_adv_std,
    )


def _build_metrics_payload(
    ctx: TrainingLoopContext,
    state: MetricState,
    prepared: PreparedBatch,
    log_artifacts: LogStepArtifacts,
    current_lr: float,
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
    :returns: Aggregated metrics suitable for logging.
    :rtype: training.types.TrainingMetricsPayload
    """
    accelerator = ctx.runtime.accelerator
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
    return TrainingMetricsPayload(
        reward_stats=_summarize_reward_stats(accelerator, prepared.reward_comp),
        weight_stats=_summarize_weight_stats(accelerator, prepared.weight_stats),
        loss_outputs=log_artifacts.loss_outputs,
        diagnostics=log_artifacts.diagnostics,
        length_stats=prepared.length_stats,
        config=config_view,
        scalars=scalar_stats,
    )


def _epoch_from_global_step(schedule: OptimizationSchedule, global_step: int) -> float:
    """Return the current epoch progress given the training schedule."""
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
    """
    if not ctx.runtime.accelerator.is_main_process:
        return
    payload = _build_metrics_payload(ctx, state, prepared, log_artifacts, current_lr)
    metrics = build_training_metrics_dict(payload, state.global_step)
    with ctx.logging.step_logger(state.global_step, enabled=False) as step_logger:
        _emit_metrics(
            ctx,
            metrics,
            state.global_step,
            log_to_wandb=False,
            tag="Local",
            metric_logger=getattr(step_logger, "log", None),
        )
    accumulate_metrics(state, metrics)


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
            def Table(self, columns=None, rows=None, **_kwargs):
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
    """
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
        )
        metrics = build_training_metrics_dict(payload, state.global_step)
    with ctx.logging.step_logger(state.global_step, enabled=True) as step_logger:
        _emit_metrics(
            ctx,
            metrics,
            state.global_step,
            log_to_wandb=True,
            tag="Global",
            metric_logger=getattr(step_logger, "log", None),
        )
    _log_sample_table(ctx, state, prepared)


__all__ = [
    "LogStepArtifacts",
    "accumulate_metrics",
    "build_training_metrics_dict",
    "flush_metric_averages",
    "log_local_step",
    "log_training_metrics",
    "log_training_step",
]
