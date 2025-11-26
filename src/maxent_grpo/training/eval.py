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
from types import SimpleNamespace
from typing import Any, Dict, Iterator, List, Optional, Tuple

from .types import RewardSpec, ValidationContext
from .run_helpers import _batch_tokenize_pairs, _prepare_labels_for_ce
from .scoring import _refresh_torch


@dataclass
class _EvalShardInfo:
    """Metadata describing the evaluation shard for the current rank."""

    rows: List[dict]
    total_rows: int
    shard_total: int
    world_size: int
    log_every: int
    is_main: bool


@dataclass
class _SeedEvalConfig:
    """Parsed seed-eval options."""

    enabled: bool
    num_seeds: int
    samples_per_seed: int
    template: str
    pooling: str


def _iter_eval_batches(
    evaluation_rows: List[dict],
    batch_size: int,
) -> Iterator[Tuple[List[str], List[str]]]:
    """Yield prompt/answer lists for evaluation rows.

    :param evaluation_rows: Serialized evaluation records containing prompts
        and optional answers.
    :type evaluation_rows: list[dict]
    :param batch_size: Number of rows per batch.
    :type batch_size: int
    :yields: Tuple containing batched prompts and answers.
    :rtype: Iterator[tuple[list[str], list[str]]]
    :returns: Iterator over prompt and answer batches.
    :rtype: collections.abc.Iterator[tuple[list[str], list[str]]]
    """
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
    """Return aggregated reward scores for completions.

    :param completions: Generated completions to score.
    :type completions: list[str]
    :param answers: Reference answers aligned with ``completions``.
    :type answers: list[str]
    :param reward_spec: Reward configuration used for evaluation.
    :type reward_spec: RewardSpec
    :returns: Aggregated reward values per completion.
    :rtype: list[float]
    """
    total_rewards = [0.0] * len(completions)
    for reward_weight, reward_fn in zip(
        reward_spec.reward_weights, reward_spec.reward_funcs
    ):
        reward_scores = reward_fn(completions, answers)
        if reward_weight != 1.0:
            reward_scores = [
                float(reward_weight) * float(score) for score in reward_scores
            ]
        total_rewards = [
            running + float(score)
            for running, score in zip(total_rewards, reward_scores)
        ]
    return total_rewards


def _build_eval_shard(
    evaluation_rows: List[dict],
    accelerator: Any,
) -> _EvalShardInfo:
    """Return shard metadata describing which rows this rank evaluates.

    :param evaluation_rows: Full evaluation dataset rows.
    :type evaluation_rows: list[dict]
    :param accelerator: Accelerate handle providing world size/rank info.
    :type accelerator: Any
    :returns: Metadata describing the rows allocated to the current rank.
    :rtype: _EvalShardInfo
    """
    world_size = max(int(getattr(accelerator, "num_processes", 1)), 1)
    rank = int(getattr(accelerator, "process_index", 0))
    shard_rows = (
        evaluation_rows[rank::world_size] if world_size > 1 else evaluation_rows
    )
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
    """Log the evaluation plan when running on the main rank.

    :param step: Training step at which evaluation is triggered.
    :type step: int
    :param shard: Partition metadata for the current rank.
    :type shard: _EvalShardInfo
    :param batch_size: Evaluation batch size.
    :type batch_size: int
    """
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
    """Generate completions for the shard rows and log periodic progress.

    :param shard: Evaluation shard metadata for the current rank.
    :type shard: _EvalShardInfo
    :param batch_size: Evaluation batch size.
    :type batch_size: int
    :param ctx: Validation context containing generator, reward spec, etc.
    :type ctx: ValidationContext
    :param step: Current training step (for logging tags).
    :type step: int
    :returns: Flattened list of reward scores produced for the shard.
    :rtype: list[float]
    """
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
    """Gather mean reward statistics across all ranks.

    :param accelerator: Accelerate handle used to gather objects.
    :type accelerator: Any
    :param eval_scores: Reward samples produced locally.
    :type eval_scores: list[float]
    :returns: Tuple containing ``(total_sum, total_count)`` across ranks.
    :rtype: tuple[float, float]
    """
    local_sum = float(sum(eval_scores))
    local_count = float(len(eval_scores))
    gather_fn = getattr(accelerator, "gather_object", None)
    gathered = gather_fn((local_sum, local_count)) if callable(gather_fn) else None
    if not gathered:
        gathered = [(local_sum, local_count)]
    total_sum = sum(pair[0] for pair in gathered)
    total_count = sum(pair[1] for pair in gathered)
    return total_sum, total_count


def _render_seed_prompts(
    prompts: List[str], num_seeds: int, template: str
) -> Tuple[List[str], List[int], List[int]]:
    """Expand prompts with seed template; return prompts, seeds, base indices."""

    rendered: List[str] = []
    seed_ids: List[int] = []
    base_idx: List[int] = []
    for base_idx_val, prompt in enumerate(prompts):
        for seed in range(1, num_seeds + 1):
            if "{prompt}" in template:
                rendered_prompt = template.format(prompt=prompt, seed=seed)
            else:
                rendered_prompt = f"{prompt}{template.format(seed=seed)}"
            rendered.append(rendered_prompt)
            seed_ids.append(seed)
            base_idx.append(base_idx_val)
    return rendered, seed_ids, base_idx


def _pool_hidden(hidden: Any, mask: Any, pooling: str) -> Any:
    """Pool hidden states according to the configured mode."""

    if pooling == "last":
        return hidden[:, -1, :]
    mask = mask.unsqueeze(-1).type_as(hidden)
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)


def _run_seed_eval(
    ctx: ValidationContext,
    shard: _EvalShardInfo,
    seed_cfg: _SeedEvalConfig,
) -> Optional[Dict[str, float]]:
    """Run multi-seed eval to measure pass@k, predictability, and diversity."""

    if (
        not seed_cfg.enabled
        or seed_cfg.num_seeds <= 0
        or seed_cfg.samples_per_seed <= 0
    ):
        return None
    torch_mod = _refresh_torch()
    prompts = [row["prompt"] for row in shard.rows]
    answers = [row.get("answer", "") for row in shard.rows]
    if not prompts:
        return None
    rendered_prompts, seed_ids, base_indices = _render_seed_prompts(
        prompts, seed_cfg.num_seeds, seed_cfg.template
    )
    per_prompt_counts = [seed_cfg.samples_per_seed] * len(rendered_prompts)
    grouped, _ = ctx.generator(
        rendered_prompts,
        seed_cfg.samples_per_seed,
        per_prompt_counts,
    )
    if not grouped:
        return None
    # Flatten completions, align answers/seed ids.
    flat_completions: List[str] = []
    flat_answers: List[str] = []
    flat_seed_ids: List[int] = []
    flat_base_idx: List[int] = []
    flat_prompts_for_pairs: List[str] = []
    for comps, seed_id, base_idx_val, rendered_prompt in zip(
        grouped, seed_ids, base_indices, rendered_prompts
    ):
        for comp in comps:
            flat_completions.append(comp)
            flat_answers.append(answers[base_idx_val])
            flat_seed_ids.append(seed_id)
            flat_base_idx.append(base_idx_val)
            flat_prompts_for_pairs.append(rendered_prompt)
    if not flat_completions:
        return None
    rewards = _compute_eval_rewards(flat_completions, flat_answers, ctx.reward)
    # Pass@K per base prompt
    pass_counts: Dict[int, int] = {}
    total_per_prompt: Dict[int, int] = {}
    for r, base_idx_val in zip(rewards, flat_base_idx):
        total_per_prompt[base_idx_val] = total_per_prompt.get(base_idx_val, 0) + 1
        if r > 0:
            pass_counts[base_idx_val] = 1
    pass_at_1 = sum(pass_counts.values()) / max(len(prompts), 1)
    # Seed predictability via seed head if available
    seed_pred_acc = None
    diversity_l2 = None
    seed_head = getattr(ctx.model, "seed_head", None)
    tokenizer = getattr(ctx, "tokenizer", None)
    if callable(seed_head) and tokenizer is not None:
        input_ids, attn, prompt_lengths = _batch_tokenize_pairs(
            tokenizer, flat_prompts_for_pairs, flat_completions
        )
        labels = _prepare_labels_for_ce(input_ids.clone(), prompt_lengths)
        input_ids = input_ids.to(ctx.model.device)
        attn = attn.to(ctx.model.device)
        labels = labels.to(ctx.model.device)
        with torch_mod.no_grad():
            outputs = ctx.model(
                input_ids=input_ids,
                attention_mask=attn,
                labels=labels,
                output_hidden_states=True,
            )
            hidden = outputs.hidden_states[-1]
            pooled = _pool_hidden(hidden, attn, seed_cfg.pooling)
            logits = seed_head(pooled)
            preds = logits.argmax(dim=-1).cpu()
            sid_tensor = torch_mod.tensor(flat_seed_ids, device=preds.device)
            valid_mask = sid_tensor >= 0
            if valid_mask.any():
                seed_pred_acc = (
                    (preds[valid_mask] == sid_tensor[valid_mask]).float().mean().item()
                )
                # Diversity across seeds (mean pooled representations per seed)
                unique_sids = torch_mod.unique(sid_tensor[valid_mask])
                if unique_sids.numel() > 1:
                    means = []
                    for sid in unique_sids:
                        m = pooled[sid_tensor == sid].mean(dim=0)
                        means.append(m)
                    if len(means) > 1:
                        stacked = torch_mod.stack(means)
                        pdist_fn = getattr(
                            getattr(torch_mod, "nn", SimpleNamespace()),
                            "functional",
                            None,
                        )
                        pdist = getattr(pdist_fn, "pdist", None) if pdist_fn else None
                        if callable(pdist):
                            try:
                                diversity_l2 = pdist(stacked, p=2).mean().item()
                            except (TypeError, ValueError, RuntimeError):
                                diversity_l2 = None
    metrics = {
        "eval_seed/pass_at_1": pass_at_1,
    }
    if seed_pred_acc is not None:
        metrics["eval_seed/pred_acc"] = float(seed_pred_acc)
    if diversity_l2 is not None:
        metrics["eval_seed/diversity_l2"] = float(diversity_l2)
    return metrics


def run_validation_step(step: int, ctx: ValidationContext) -> None:
    """Generate single completions on the eval set and log mean reward.

    :param step: Training step identifier passed to logging hooks.
    :type step: int
    :param ctx: Validation context providing evaluation rows and handles.
    :type ctx: ValidationContext
    :returns: None. Logs metrics through the provided handles.
    :rtype: None
    """
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
            seed_cfg_raw = getattr(evaluation_cfg, "seed_eval", None)
            if isinstance(seed_cfg_raw, dict):
                seed_cfg = _SeedEvalConfig(
                    enabled=bool(seed_cfg_raw.get("enabled", False)),
                    num_seeds=int(seed_cfg_raw.get("num_seeds", 0)),
                    samples_per_seed=int(seed_cfg_raw.get("samples_per_seed", 0)),
                    template=str(seed_cfg_raw.get("template", "\n[seed={seed}]")),
                    pooling=str(seed_cfg_raw.get("pooling", "mean")),
                )
                seed_metrics = _run_seed_eval(ctx, shard, seed_cfg)
                if seed_metrics:
                    logging.getLogger(__name__).info(
                        "eval seed metrics @ step %d | %s",
                        step,
                        seed_metrics,
                    )
                    ctx.logging.log_metrics(seed_metrics, step)
    finally:
        if prev_mode:
            model.train()
        if callable(wait_for_all):
            wait_for_all()


__all__ = ["run_validation_step"]
