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

"""Validation helpers for the MaxEnt-GRPO training loop."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterator, List, Tuple

from .run_training_types import RewardSpec, ValidationContext


@dataclass
class _EvalShardInfo:
    """Metadata describing the evaluation shard for the current rank."""

    rows: List[dict]
    total_rows: int
    shard_total: int
    world_size: int
    log_every: int
    is_main: bool


def _iter_eval_batches(
    evaluation_rows: List[dict],
    batch_size: int,
) -> Iterator[Tuple[List[str], List[str]]]:
    """Yield prompt/answer lists for evaluation rows."""
    for batch_start in range(0, len(evaluation_rows), batch_size):
        batch_rows = evaluation_rows[batch_start : batch_start + batch_size]
        prompts = [row["prompt"] for row in batch_rows]
        if not prompts:
            continue
        answers = [row.get("answer", "") for row in batch_rows]
        yield prompts, answers


def _compute_eval_rewards(
    completions: List[str],
    answers: List[str],
    reward_spec: RewardSpec,
) -> List[float]:
    """Return aggregated reward scores for completions."""
    total_rewards = [0.0] * len(completions)
    for reward_weight, reward_fn in zip(
        reward_spec.reward_weights, reward_spec.reward_funcs
    ):
        reward_scores = reward_fn(completions, answers)
        if reward_weight != 1.0:
            reward_scores = [float(reward_weight) * float(score) for score in reward_scores]
        total_rewards = [
            running + float(score)
            for running, score in zip(total_rewards, reward_scores)
        ]
    return total_rewards


def _build_eval_shard(
    evaluation_rows: List[dict],
    accelerator: Any,
) -> _EvalShardInfo:
    """Return shard metadata describing which rows this rank evaluates."""
    world_size = max(int(getattr(accelerator, "num_processes", 1)), 1)
    rank = int(getattr(accelerator, "process_index", 0))
    shard_rows = evaluation_rows[rank::world_size] if world_size > 1 else evaluation_rows
    shard_total = len(shard_rows)
    total_rows = len(evaluation_rows)
    log_every = max(1, shard_total // 10) if shard_total else 1
    is_main = bool(getattr(accelerator, "is_main_process", True))
    return _EvalShardInfo(
        rows=shard_rows,
        total_rows=total_rows,
        shard_total=shard_total,
        world_size=world_size,
        log_every=log_every,
        is_main=is_main,
    )


def _log_eval_start(step: int, shard: _EvalShardInfo, batch_size: int) -> None:
    """Log the evaluation plan when running on the main rank."""
    if not shard.is_main:
        return
    logging.getLogger(__name__).info(
        (
            "eval step %d starting | total_rows=%d | shard_rows=%d | "
            "world_size=%d | batch_size=%d"
        ),
        step,
        shard.total_rows,
        shard.shard_total,
        shard.world_size,
        batch_size,
    )


def _run_eval_batches(
    shard: _EvalShardInfo,
    batch_size: int,
    ctx: ValidationContext,
    step: int,
) -> List[float]:
    """Generate completions for the shard rows and log periodic progress."""
    eval_scores: List[float] = []
    processed = 0
    for batch_idx, (prompts, answers) in enumerate(
        _iter_eval_batches(shard.rows, batch_size)
    ):
        target_counts = [1] * len(prompts)
        grouped, _ = ctx.generator(prompts, 1, target_counts)
        if grouped:
            completions = [grp[0] if grp else "" for grp in grouped]
            eval_scores.extend(_compute_eval_rewards(completions, answers, ctx.reward))
        processed += len(prompts)
        should_log = shard.shard_total and (
            processed >= shard.shard_total or (batch_idx + 1) % shard.log_every == 0
        )
        if should_log and shard.is_main:
            running_mean = float(sum(eval_scores) / max(len(eval_scores), 1))
            logging.getLogger(__name__).info(
                "eval step %d progress | shard_processed=%d/%d | running_mean=%.4f",
                step,
                processed,
                shard.shard_total,
                running_mean,
            )
    return eval_scores


def _gather_eval_stats(
    accelerator: Any,
    eval_scores: List[float],
) -> Tuple[float, float]:
    """Gather mean reward statistics across all ranks."""
    local_sum = float(sum(eval_scores))
    local_count = float(len(eval_scores))
    gather_fn = getattr(accelerator, "gather_object", None)
    gathered = gather_fn((local_sum, local_count)) if callable(gather_fn) else None
    if not gathered:
        gathered = [(local_sum, local_count)]
    total_sum = sum(pair[0] for pair in gathered)
    total_count = sum(pair[1] for pair in gathered)
    return total_sum, total_count


def run_validation_step(step: int, ctx: ValidationContext) -> None:
    """Generate single completions on the eval set and log mean reward."""
    evaluation_cfg = ctx.evaluation
    if not evaluation_cfg.enabled or not evaluation_cfg.rows:
        return
    accelerator = ctx.accelerator
    wait_for_all = getattr(accelerator, "wait_for_everyone", None)
    if callable(wait_for_all):
        wait_for_all()
    shard = _build_eval_shard(evaluation_cfg.rows, accelerator)
    _log_eval_start(step, shard, evaluation_cfg.batch_size)

    model = ctx.model
    prev_mode = getattr(model, "training", False)
    model.eval()

    try:
        eval_scores = _run_eval_batches(
            shard,
            evaluation_cfg.batch_size,
            ctx,
            step,
        )
        total_sum, total_count = _gather_eval_stats(accelerator, eval_scores)
        if shard.is_main:
            mean_reward = total_sum / max(total_count, 1.0)
            logging.getLogger(__name__).info(
                "eval step %d | mean_reward=%.4f | samples=%d",
                step,
                mean_reward,
                int(total_count),
            )
            ctx.logging.log_metrics({"eval/mean_reward": mean_reward}, step)
    finally:
        if prev_mode:
            model.train()
        if callable(wait_for_all):
            wait_for_all()


__all__ = ["run_validation_step"]
