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
import math
import os
import time
from contextlib import nullcontext
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, Iterator, List, Optional, Tuple, cast

from maxent_grpo.generation.errors import (
    GenerationServiceError,
    log_generation_service_error,
)
from maxent_grpo.rewards.basic import _answer_pat, _format_pat
from maxent_grpo.training.runtime.logging import _log_wandb
from .run_helpers import _batch_tokenize_pairs, _prepare_labels_for_ce
from .scoring import (
    _refresh_torch,
    _to_numpy_array,
    build_score_batch,
    gather_reference_logprobs,
    reference_from_vllm_meta,
    reference_stats_from_policy_logprobs,
    score_model_outputs,
    token_counts_from_score_batch,
    vllm_meta_has_logprobs,
)
from .types import PromptCompletionBatch, RewardSpec, ValidationContext

LOG = logging.getLogger(__name__)
_EVAL_LOGPROBS_ENV = "MAXENT_EVAL_LOGPROBS"
_EVAL_LOGPROBS_WARNED = False


def _progress_log_enabled() -> bool:
    raw = os.getenv("MAXENT_PROGRESS_LOG")
    if raw is None or not str(raw).strip():
        return False
    return str(raw).strip().lower() not in {"0", "false", "no", "off"}


def _eval_rank_tag() -> str:
    for key in ("RANK", "LOCAL_RANK", "SLURM_PROCID", "SLURM_LOCALID"):
        val = os.getenv(key)
        if val is not None and str(val).strip():
            return f"{key}={val}"
    return "rank=unknown"


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def _eval_logprobs_enabled() -> bool:
    return _env_flag(_EVAL_LOGPROBS_ENV, False)


def _warn_eval_logprobs_unavailable(reason: str) -> None:
    global _EVAL_LOGPROBS_WARNED
    if _EVAL_LOGPROBS_WARNED:
        return
    LOG.warning("Eval logprob metrics disabled: %s", reason)
    _EVAL_LOGPROBS_WARNED = True


def _deepspeed_zero_stage(accelerator: Any) -> int:
    """Return DeepSpeed ZeRO stage from Accelerate plugin state when present."""
    state = getattr(accelerator, "state", None)
    ds_plugin = getattr(state, "deepspeed_plugin", None)
    try:
        return int(getattr(ds_plugin, "zero_stage", 0) or 0)
    except (TypeError, ValueError):
        return 0


def _as_tensor_1d(value: Any, device: Any = None) -> Any:
    """Best-effort conversion to a 1D float tensor."""
    torch_mod = _refresh_torch()
    if value is None:
        return torch_mod.zeros(
            (0,),
            dtype=getattr(torch_mod, "float32", None),
            device=device,
        )
    tensor_cls = getattr(torch_mod, "Tensor", None)
    if tensor_cls is not None and isinstance(value, tensor_cls):
        tensor = value
    else:
        tensor = torch_mod.tensor(
            getattr(value, "arr", value),
            dtype=getattr(torch_mod, "float32", None),
            device=device,
        )
    try:
        return tensor.view(-1)
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return tensor


def _match_tensor_length(tensor: Any, target_len: int, device: Any = None) -> Any:
    """Pad or truncate a 1D tensor to ``target_len``."""
    torch_mod = _refresh_torch()
    if target_len <= 0:
        return torch_mod.zeros(
            (0,),
            dtype=getattr(torch_mod, "float32", None),
            device=device,
        )
    tensor = _as_tensor_1d(tensor, device=device)
    try:
        cur_len = int(getattr(tensor, "numel", lambda: 0)())
    except (TypeError, ValueError):
        cur_len = 0
    if cur_len == target_len:
        return tensor
    if cur_len == 0:
        return torch_mod.zeros(
            (target_len,),
            dtype=getattr(torch_mod, "float32", None),
            device=device,
        )
    if cur_len == 1:
        try:
            scalar_val = float(getattr(tensor[0], "item", lambda: tensor[0])())
        except (TypeError, ValueError):
            scalar_val = float(tensor[0])
        return torch_mod.full(
            (target_len,),
            scalar_val,
            dtype=getattr(torch_mod, "float32", None),
            device=device,
        )
    min_len = min(cur_len, target_len)
    try:
        tensor = tensor[:min_len]
    except (TypeError, ValueError):
        pass
    if min_len == target_len:
        return tensor
    try:
        pad_val = float(getattr(tensor[-1], "item", lambda: tensor[-1])())
    except (TypeError, ValueError):
        pad_val = float(tensor[-1])
    pad = torch_mod.full(
        (target_len - min_len,),
        pad_val,
        dtype=getattr(torch_mod, "float32", None),
        device=device,
    )
    try:
        return torch_mod.cat([tensor, pad], dim=0)
    except (TypeError, ValueError):
        return pad


def _flatten_eval_meta(
    grouped_meta: Optional[List[List[Optional[Any]]]],
    expected_len: int,
) -> Optional[List[Optional[Any]]]:
    if not grouped_meta or len(grouped_meta) != expected_len:
        return None
    flat: List[Optional[Any]] = []
    for group in grouped_meta:
        if group:
            flat.append(group[0])
        else:
            flat.append(None)
    return flat


def _compute_eval_kl_tensor(
    cur_logp_sum: Any,
    tok_counts: Any,
    ref_stats: Any,
    *,
    len_norm_ref: bool,
) -> Optional[Any]:
    torch_mod = _refresh_torch()
    cur_tensor = _as_tensor_1d(cur_logp_sum)
    denom = _as_tensor_1d(tok_counts, device=getattr(cur_tensor, "device", None))
    try:
        cur_len = int(getattr(cur_tensor, "numel", lambda: 0)())
    except (TypeError, ValueError):
        cur_len = 0
    if cur_len <= 0:
        return None
    denom = _match_tensor_length(denom, cur_len, device=getattr(cur_tensor, "device", None))
    try:
        denom = denom.clamp(min=1.0)
    except (AttributeError, RuntimeError, TypeError, ValueError):
        pass
    ref_source = getattr(ref_stats, "ref_logp_sum", None) if len_norm_ref else getattr(
        ref_stats, "ref_logp_sum_raw", None
    )
    if ref_source is None:
        ref_source = getattr(ref_stats, "ref_logp_sum", None)
    if ref_source is None:
        return None
    ref_tensor = _match_tensor_length(
        ref_source, cur_len, device=getattr(cur_tensor, "device", None)
    )
    try:
        cur_per_tok = cur_tensor / denom
    except (RuntimeError, TypeError, ValueError):
        return None
    if len_norm_ref:
        ref_per_tok = ref_tensor
    else:
        try:
            ref_per_tok = ref_tensor / denom
        except (RuntimeError, TypeError, ValueError):
            return None
    try:
        delta = (ref_per_tok - cur_per_tok).clamp(min=-60.0, max=60.0)
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return None
    try:
        return delta.exp() - delta - 1.0
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return None


def _init_eval_score_stats() -> Dict[str, float]:
    return {
        "kl_weighted_sum": 0.0,
        "kl_token_sum": 0.0,
        "kl_sum": 0.0,
        "kl_count": 0.0,
        "entropy_sum": 0.0,
        "entropy_token_sum": 0.0,
        "length_sum": 0.0,
        "length_count": 0.0,
        "clipped_count": 0.0,
        "terminated_sum": 0.0,
        "terminated_count": 0.0,
    }


def _update_eval_score_stats(
    target: Dict[str, float], update: Dict[str, float]
) -> None:
    for key, value in update.items():
        target[key] = target.get(key, 0.0) + float(value)


def _score_eval_batch(
    ctx: ValidationContext,
    prompts: List[str],
    completions: List[str],
    grouped_meta: Optional[List[List[Optional[Any]]]],
) -> Optional[Dict[str, float]]:
    if not _eval_logprobs_enabled():
        return None
    progress_log = _progress_log_enabled()
    rank_tag = _eval_rank_tag()
    if progress_log:
        LOG.info(
            "eval logprobs batch start | %s | prompts=%d completions=%d",
            rank_tag,
            len(prompts),
            len(completions),
        )
        batch_start = time.monotonic()
    runtime = getattr(ctx, "runtime", None)
    scoring_cfg = getattr(ctx, "scoring", None)
    generation_cfg = getattr(ctx, "generation", None)
    if runtime is None or scoring_cfg is None or generation_cfg is None:
        _warn_eval_logprobs_unavailable("missing runtime/scoring/generation context")
        return None
    if not prompts or not completions:
        return None
    batching_cfg = getattr(scoring_cfg, "batching", None)
    if batching_cfg is None:
        _warn_eval_logprobs_unavailable("missing scoring.batching")
        return None
    if not getattr(batching_cfg, "prompt_length_cache_get", None):
        prompt_cache = getattr(runtime, "prompt_cache_get", None)
        if callable(prompt_cache):
            batching_cfg.prompt_length_cache_get = prompt_cache
    pairs = PromptCompletionBatch(prompts=prompts, completions=completions)
    reward_stub = SimpleNamespace(pairs=pairs, completion_metadata=None)
    if progress_log:
        LOG.info("eval logprobs build_score_batch start | %s", rank_tag)
        build_start = time.monotonic()
    score_batch = build_score_batch(
        reward_stub,
        ctx.tokenizer,
        generation_cfg,
        batching_cfg,
    )
    if progress_log:
        LOG.info(
            "eval logprobs build_score_batch done | %s | seconds=%.2f | ok=%s",
            rank_tag,
            time.monotonic() - build_start,
            bool(score_batch is not None),
        )
    if score_batch is None:
        return None
    torch_mod = _refresh_torch()
    no_grad_ctx = getattr(torch_mod, "no_grad", None) or nullcontext
    entropy_mode = str(
        getattr(scoring_cfg, "policy_entropy_mode", "exact") or "exact"
    )
    if progress_log:
        LOG.info(
            "eval logprobs policy score start | %s | total_sequences=%s slice_size=%s",
            rank_tag,
            getattr(score_batch, "total_sequences", None),
            getattr(score_batch, "slice_size", None),
        )
        policy_start = time.monotonic()
    with no_grad_ctx():
        result = score_model_outputs(
            ctx.model,
            score_batch,
            batching_cfg,
            runtime,
            return_entropy=True,
            entropy_mode=entropy_mode,
        )
    if progress_log:
        LOG.info(
            "eval logprobs policy score done | %s | seconds=%.2f | ok=%s",
            rank_tag,
            time.monotonic() - policy_start,
            bool(result is not None),
        )
    if result is None:
        return None
    if isinstance(result, tuple) and len(result) == 3:
        cur_logp_sum, _pooled, policy_entropy_sum = result
    else:
        cur_logp_sum, _pooled = result  # type: ignore[misc]
        policy_entropy_sum = None
    tok_counts = token_counts_from_score_batch(score_batch, runtime, batching_cfg)
    try:
        cur_len = int(getattr(cur_logp_sum, "numel", lambda: 0)())
    except (TypeError, ValueError):
        cur_len = 0
    if cur_len > 0:
        tok_counts = _match_tensor_length(
            tok_counts, cur_len, device=getattr(cur_logp_sum, "device", None)
        )
    ref_source = str(
        getattr(scoring_cfg, "reference_logprobs_source", "auto") or "auto"
    ).strip().lower()
    trl_reference_scoring = bool(
        getattr(scoring_cfg, "trl_reference_scoring", False)
    )
    force_ref_model = ref_source in {
        "model",
        "reference",
        "reference_model",
        "ref_model",
    }
    if trl_reference_scoring:
        force_ref_model = True
    if _env_flag("MAXENT_FORCE_REF_MODEL", False):
        force_ref_model = True
    flat_meta = _flatten_eval_meta(grouped_meta, len(completions))
    if progress_log:
        LOG.info(
            "eval logprobs reference source | %s | ref_source=%s force_ref_model=%s vllm_meta=%s",
            rank_tag,
            ref_source,
            force_ref_model,
            bool(flat_meta),
        )
    ref_stats = None
    if flat_meta and not force_ref_model and vllm_meta_has_logprobs(
        flat_meta, getattr(score_batch, "total_sequences", None)
    ):
        if progress_log:
            LOG.info("eval logprobs reference from vLLM meta | %s", rank_tag)
        ref_stats = reference_from_vllm_meta(
            flat_meta,
            int(getattr(score_batch, "total_sequences", len(completions))),
            runtime.device,
        )
    if ref_stats is None and not force_ref_model:
        if progress_log:
            LOG.info("eval logprobs reference from policy logprobs start | %s", rank_tag)
        try:
            ref_stats = reference_stats_from_policy_logprobs(cur_logp_sum, tok_counts)
        except (TypeError, ValueError, RuntimeError):
            ref_stats = None
        if progress_log:
            LOG.info(
                "eval logprobs reference from policy logprobs done | %s | ok=%s",
                rank_tag,
                bool(ref_stats is not None),
            )
    if ref_stats is None:
        if progress_log:
            LOG.info("eval logprobs reference gather start | %s", rank_tag)
            ref_start = time.monotonic()
        ref_stats = gather_reference_logprobs(
            score_batch,
            runtime,
            batching_cfg,
            trl_reference_scoring=trl_reference_scoring,
            temperature=getattr(generation_cfg, "gen_temperature", None),
        )
        if progress_log:
            LOG.info(
                "eval logprobs reference gather done | %s | seconds=%.2f | ok=%s",
                rank_tag,
                time.monotonic() - ref_start,
                bool(ref_stats is not None),
            )
    if ref_stats is None:
        return None
    weighting_cfg = getattr(scoring_cfg, "weighting", None)
    len_norm_ref = bool(getattr(weighting_cfg, "len_norm_ref", False))
    kl_tensor = _compute_eval_kl_tensor(
        cur_logp_sum,
        tok_counts,
        ref_stats,
        len_norm_ref=len_norm_ref,
    )
    if kl_tensor is None:
        return None
    stats: Dict[str, float] = {}
    if progress_log:
        LOG.info(
            "eval logprobs batch done | %s | seconds=%.2f",
            rank_tag,
            time.monotonic() - batch_start,
        )
    try:
        kl_sum = float(kl_tensor.detach().float().sum().cpu().item())
        kl_count = float(getattr(kl_tensor, "numel", lambda: 0)())
    except (AttributeError, RuntimeError, TypeError, ValueError):
        kl_sum = float(kl_tensor.sum())
        kl_count = float(getattr(kl_tensor, "numel", lambda: 0)())
    try:
        tok_sum = float(tok_counts.detach().float().sum().cpu().item())
    except (AttributeError, RuntimeError, TypeError, ValueError):
        tok_sum = float(tok_counts.sum())
    try:
        kl_weighted = float((kl_tensor * tok_counts).detach().float().sum().cpu().item())
    except (AttributeError, RuntimeError, TypeError, ValueError):
        kl_weighted = float((kl_tensor * tok_counts).sum())
    stats["kl_sum"] = kl_sum
    stats["kl_count"] = kl_count
    stats["kl_token_sum"] = tok_sum
    stats["kl_weighted_sum"] = kl_weighted
    if policy_entropy_sum is not None:
        try:
            entropy_sum = float(policy_entropy_sum.detach().float().sum().cpu().item())
        except (AttributeError, RuntimeError, TypeError, ValueError):
            entropy_sum = float(policy_entropy_sum.sum())
        stats["entropy_sum"] = entropy_sum
        stats["entropy_token_sum"] = tok_sum
    max_len = int(getattr(generation_cfg, "max_completion_len", 0) or 0)
    try:
        length_sum = float(tok_counts.detach().float().sum().cpu().item())
        length_count = float(getattr(tok_counts, "numel", lambda: 0)())
    except (AttributeError, RuntimeError, TypeError, ValueError):
        length_sum = float(tok_counts.sum())
        length_count = float(getattr(tok_counts, "numel", lambda: 0)())
    stats["length_sum"] = length_sum
    stats["length_count"] = length_count
    if max_len > 0:
        try:
            clipped = (tok_counts >= max_len).float()
            clipped_count = float(clipped.sum().detach().cpu().item())
            terminated_mask = tok_counts < max_len
            terminated_sum = float(
                (tok_counts * terminated_mask).detach().float().sum().cpu().item()
            )
            terminated_count = float(terminated_mask.detach().float().sum().cpu().item())
        except (AttributeError, RuntimeError, TypeError, ValueError):
            clipped_count = 0.0
            terminated_sum = 0.0
            terminated_count = 0.0
        stats["clipped_count"] = clipped_count
        stats["terminated_sum"] = terminated_sum
        stats["terminated_count"] = terminated_count
    return stats


@dataclass
class _EvalShardInfo:
    """Metadata describing the evaluation shard for the current rank."""

    rows: List[dict]
    total_rows: int
    shard_total: int
    world_size: int
    log_every: int
    is_main: bool
    rank: int


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
    progress_log = _progress_log_enabled()
    rank_tag = _eval_rank_tag() if progress_log else ""
    for idx, (reward_weight, reward_fn) in enumerate(
        zip(reward_spec.reward_weights, reward_spec.reward_funcs)
    ):
        if progress_log:
            LOG.info(
                "eval reward fn start | %s | idx=%d | weight=%.3f | fn=%s | n=%d",
                rank_tag,
                idx,
                float(reward_weight),
                getattr(reward_fn, "__name__", reward_fn.__class__.__name__),
                len(completions),
            )
            fn_start = time.monotonic()
        try:
            reward_scores = reward_fn(completions, answers, is_eval=True, split="eval")
        except TypeError:
            try:
                reward_scores = reward_fn(completions, answers)
            except TypeError:
                reward_scores = reward_fn(completions, answers, is_eval=True)
        if progress_log:
            LOG.info(
                "eval reward fn done | %s | idx=%d | seconds=%.2f",
                rank_tag,
                idx,
                time.monotonic() - fn_start,
            )
        if reward_weight != 1.0:
            reward_scores = [
                float(reward_weight) * float(score) for score in reward_scores
            ]
        total_rewards = [
            running + float(score)
            for running, score in zip(total_rewards, reward_scores)
        ]
    return total_rewards


def _tally_format_issues(
    completions: List[str],
) -> Dict[str, float]:
    """Return format issue counts for a batch of completions."""

    stats = {"missing_answer": 0.0, "missing_format": 0.0, "total": 0.0}
    for comp in completions:
        stats["total"] += 1.0
        has_answer = bool(_answer_pat.search(comp))
        has_format = bool(_format_pat.search(comp))
        if not has_answer:
            stats["missing_answer"] += 1.0
        # For eval metrics, treat <answer> alone as acceptable format.
        if not has_format and not has_answer:
            stats["missing_format"] += 1.0
    return stats


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
        rank=rank,
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


def _maybe_presync_vllm_for_eval(ctx: ValidationContext, eval_only_rank0: bool) -> None:
    """Synchronize vLLM weights across all ranks before rank-0-only eval."""
    if not eval_only_rank0:
        return
    if os.getenv("MAXENT_SKIP_VLLM_EVAL_PRESYNC"):
        logging.getLogger(__name__).info(
            "eval pre-sync vLLM weights skipped | reason=MAXENT_SKIP_VLLM_EVAL_PRESYNC"
        )
        return
    generator = getattr(ctx, "generator", None)
    generator_self = getattr(generator, "__self__", None)
    vllm_helper = getattr(generator_self, "_vllm_helper", None) if generator_self else None
    maybe_sync = getattr(vllm_helper, "maybe_sync_weights", None)
    if not callable(maybe_sync):
        return
    logging.getLogger(__name__).info(
        "eval pre-sync vLLM weights | mode=eval_only_rank0"
    )
    try:
        sync_model = getattr(vllm_helper, "_sync_model_params_to_vllm", None)
        if callable(sync_model):
            maybe_sync(sync_model=sync_model)
        else:
            maybe_sync()
    except Exception as exc:
        logging.getLogger(__name__).warning(
            "eval pre-sync vLLM weights failed: %s", exc
        )
        raise


def _run_eval_batches(
    shard: _EvalShardInfo,
    batch_size: int,
    ctx: ValidationContext,
    step: int,
) -> Tuple[List[float], Dict[str, float], Optional[Dict[str, float]]]:
    """Generate completions for the shard rows and log periodic progress.

    :param shard: Evaluation shard metadata for the current rank.
    :type shard: _EvalShardInfo
    :param batch_size: Evaluation batch size.
    :type batch_size: int
    :param ctx: Validation context containing generator, reward spec, etc.
    :type ctx: ValidationContext
    :param step: Current training step (for logging tags).
    :type step: int
    :returns: Flattened list of reward scores produced for the shard, format
        counts, and optional aggregated logprob stats.
    :rtype: tuple[list[float], dict[str, float], dict[str, float] | None]
    """
    eval_scores: List[float] = []
    fmt_counts: Dict[str, float] = {
        "missing_answer": 0.0,
        "missing_format": 0.0,
        "total": 0.0,
    }
    score_stats: Optional[Dict[str, float]] = (
        _init_eval_score_stats() if _eval_logprobs_enabled() else None
    )
    timeout_s = 0.0
    try:
        timeout_s = float(os.getenv("MAXENT_EVAL_BATCH_TIMEOUT_S", "0") or 0)
    except (TypeError, ValueError):
        timeout_s = 0.0
    num_batches = (
        math.ceil(shard.shard_total / batch_size) if batch_size > 0 else 0
    )
    reward_spec = ctx.eval_reward or ctx.reward
    processed = 0
    for batch_idx, (prompts, answers) in enumerate(
        _iter_eval_batches(shard.rows, batch_size)
    ):
        batch_num = batch_idx + 1
        request_id_prefix = f"eval-s{step}-b{batch_num}-r{shard.rank}"
        progress_log = _progress_log_enabled()
        logging.getLogger(__name__).info(
            "eval step %d rank %d/%d batch %d/%d start | batch_size=%d",
            step,
            shard.rank,
            max(shard.world_size, 1),
            batch_num,
            max(num_batches, 1),
            len(prompts),
        )
        logging.getLogger(__name__).info(
            "eval step %d rank %d batch %d request_id_prefix=%s",
            step,
            shard.rank,
            batch_num,
            request_id_prefix,
        )
        batch_start = time.time()
        target_counts = [1] * len(prompts)
        prev_request_id_prefix = getattr(ctx, "vllm_request_id_prefix", None)
        setattr(ctx, "vllm_request_id_prefix", request_id_prefix)
        try:
            grouped, grouped_meta = ctx.generator(prompts, 1, target_counts)
        except GenerationServiceError as exc:
            log_generation_service_error(
                logging.getLogger(__name__), "evaluation", exc
            )
            raise
        finally:
            if prev_request_id_prefix is None:
                try:
                    delattr(ctx, "vllm_request_id_prefix")
                except AttributeError:
                    pass
            else:
                setattr(ctx, "vllm_request_id_prefix", prev_request_id_prefix)
        elapsed_s = time.time() - batch_start
        logging.getLogger(__name__).info(
            "eval step %d rank %d/%d batch %d/%d done | batch_size=%d | elapsed_s=%.2f",
            step,
            shard.rank,
            max(shard.world_size, 1),
            batch_num,
            max(num_batches, 1),
            len(prompts),
            elapsed_s,
        )
        if timeout_s > 0.0 and elapsed_s > timeout_s:
            logging.getLogger(__name__).warning(
                (
                    "eval step %d rank %d/%d batch %d/%d slow | elapsed_s=%.2f "
                    "> timeout_s=%.2f"
                ),
                step,
                shard.rank,
                max(shard.world_size, 1),
                batch_num,
                max(num_batches, 1),
                elapsed_s,
                timeout_s,
            )
        if grouped:
            completions = [grp[0] if grp else "" for grp in grouped]
            if progress_log:
                logging.getLogger(__name__).info(
                    "eval step %d rank %d/%d batch %d reward start | completions=%d",
                    step,
                    shard.rank,
                    max(shard.world_size, 1),
                    batch_num,
                    len(completions),
                )
                reward_start = time.monotonic()
            eval_scores.extend(_compute_eval_rewards(completions, answers, reward_spec))
            if progress_log:
                logging.getLogger(__name__).info(
                    "eval step %d rank %d/%d batch %d reward done | seconds=%.2f",
                    step,
                    shard.rank,
                    max(shard.world_size, 1),
                    batch_num,
                    time.monotonic() - reward_start,
                )
            batch_fmt = _tally_format_issues(completions)
            for k, v in batch_fmt.items():
                fmt_counts[k] = fmt_counts.get(k, 0.0) + float(v)
            if score_stats is not None:
                if progress_log:
                    logging.getLogger(__name__).info(
                        "eval step %d rank %d/%d batch %d logprobs start",
                        step,
                        shard.rank,
                        max(shard.world_size, 1),
                        batch_num,
                    )
                    score_start = time.monotonic()
                batch_stats = _score_eval_batch(
                    ctx, prompts, completions, grouped_meta
                )
                if progress_log:
                    logging.getLogger(__name__).info(
                        "eval step %d rank %d/%d batch %d logprobs done | seconds=%.2f | ok=%s",
                        step,
                        shard.rank,
                        max(shard.world_size, 1),
                        batch_num,
                        time.monotonic() - score_start,
                        bool(batch_stats is not None),
                    )
                if batch_stats is not None:
                    _update_eval_score_stats(score_stats, batch_stats)
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
    return eval_scores, fmt_counts, score_stats


def _gather_eval_stats(
    accelerator: Any,
    eval_scores: List[float],
    fmt_counts: Optional[Dict[str, float]] = None,
    score_stats: Optional[Dict[str, float]] = None,
) -> Tuple[float, float] | Tuple[float, float, Dict[str, float]] | Tuple[
    float, float, Dict[str, float], Dict[str, float]
]:
    """Gather mean reward statistics across all ranks.

    :param accelerator: Accelerate handle used to gather objects.
    :type accelerator: Any
    :param eval_scores: Reward samples produced locally.
    :type eval_scores: list[float]
    :returns: Tuple containing ``(total_sum, total_count)`` across ranks, and
        optionally the aggregated ``fmt_counts``/``score_stats`` when provided.
    :rtype: tuple[float, float] | tuple[float, float, dict[str, float]] |
        tuple[float, float, dict[str, float], dict[str, float]]
    """
    local_sum = float(sum(eval_scores))
    local_count = float(len(eval_scores))
    provided_fmt = fmt_counts is not None
    fmt_counts = fmt_counts or {
        "missing_answer": 0.0,
        "missing_format": 0.0,
        "total": 0.0,
    }
    gather_fn = getattr(accelerator, "gather_object", None)
    payload = (local_sum, local_count, fmt_counts, score_stats)
    gathered_raw = gather_fn(payload) if callable(gather_fn) else None
    if isinstance(gathered_raw, (list, tuple)):
        gathered = list(gathered_raw)
    else:
        gathered = [payload]
    total_sum = 0.0
    total_count = 0.0
    total_fmt: Dict[str, float] = {
        "missing_answer": 0.0,
        "missing_format": 0.0,
        "total": 0.0,
    }
    total_score: Dict[str, float] = {}
    for item in gathered:
        # Accept payloads of length 2 (sum, count) or 3 (sum, count, fmt_counts)
        if not isinstance(item, (list, tuple)):
            continue
        if len(item) >= 2:
            total_sum += float(item[0])
            total_count += float(item[1])
        if len(item) >= 3 and item[2] is not None:
            for k, v in item[2].items():
                total_fmt[k] = total_fmt.get(k, 0.0) + float(v)
        if len(item) >= 4 and item[3] is not None:
            for k, v in item[3].items():
                total_score[k] = total_score.get(k, 0.0) + float(v)
    if not provided_fmt:
        return total_sum, total_count
    if score_stats is None:
        return total_sum, total_count, total_fmt
    return total_sum, total_count, total_fmt, total_score


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
    reward_spec = ctx.eval_reward or ctx.reward
    rewards = _compute_eval_rewards(flat_completions, flat_answers, reward_spec)
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
            call_target = ctx.model if callable(ctx.model) else getattr(ctx.model, "forward", None)
            if not callable(call_target):
                raise TypeError("Model is not callable and lacks a forward method")
            outputs = cast(
                Any,
                call_target(
                input_ids=input_ids,
                attention_mask=attn,
                labels=labels,
                output_hidden_states=True,
            ),
            )
            hidden_states = getattr(outputs, "hidden_states", None)
            if not hidden_states:
                raise TypeError("Model outputs missing hidden_states")
            hidden = hidden_states[-1]
            pooled = _pool_hidden(hidden, attn, seed_cfg.pooling)
            logits = seed_head(pooled)
            logits_any = cast(Any, logits)
            preds = getattr(torch_mod, "as_tensor", torch_mod.tensor)(
                logits_any.argmax(dim=-1)
            ).view(-1)
            sid_tensor = getattr(torch_mod, "as_tensor", torch_mod.tensor)(
                flat_seed_ids, device=preds.device
            ).view(-1)
            seq_len = 0
            shape = getattr(pooled, "shape", None)
            if shape and len(shape) > 0:
                seq_len = int(shape[0])
            valid_len = int(
                min(preds.numel(), sid_tensor.numel(), seq_len or preds.numel())
            )
            if valid_len > 0:
                preds = preds[:valid_len]
                sid_tensor = sid_tensor[:valid_len]
                try:
                    pooled = pooled[:valid_len]
                except (
                    TypeError,
                    AttributeError,
                    RuntimeError,
                ):  # pragma: no cover - defensive
                    LOG.debug("Failed to trim pooled seed predictions for valid length.")
            valid_mask = sid_tensor >= 0
            if valid_mask.any() and preds.numel() > 0:
                try:
                    seed_pred_acc = (
                        (preds[valid_mask] == sid_tensor[valid_mask])
                        .float()
                        .mean()
                        .item()
                    )
                except (TypeError, ValueError, RuntimeError):
                    # Fall back to numpy-style masking when stubs are present.
                    import numpy as _np

                    preds_arr = _to_numpy_array(preds)
                    mask_arr = _to_numpy_array(valid_mask).astype(bool)
                    sid_arr = _to_numpy_array(sid_tensor)
                    try:
                        seed_pred_acc = float(
                            _np.mean(preds_arr[mask_arr] == sid_arr[mask_arr])
                        )
                    except (TypeError, ValueError, RuntimeError):
                        seed_pred_acc = None
                # Diversity across seeds (mean pooled representations per seed)
                unique_sids = torch_mod.unique(sid_tensor[valid_mask])
                if unique_sids.numel() > 1:
                    means = []
                    pooled_any = cast(Any, pooled)
                    for sid in unique_sids:
                        m = pooled_any[sid_tensor == sid].mean(dim=0)
                        means.append(m)
                    if len(means) > 1:
                        try:
                            stacked = torch_mod.stack(means)
                        except TypeError:
                            stacked = torch_mod.stack(
                                [torch_mod.tensor(getattr(m, "arr", m)) for m in means]
                            )
                        pdist_fn = getattr(
                            getattr(torch_mod, "nn", SimpleNamespace()),
                            "functional",
                            None,
                        )
                        pdist = getattr(pdist_fn, "pdist", None) if pdist_fn else None
                        if callable(pdist):
                            try:
                                pdist_val = cast(Any, pdist(stacked, p=2))
                                diversity_l2 = float(pdist_val.mean().item())
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
    eval_only_rank0 = str(
        os.getenv("MAXENT_EVAL_ONLY_RANK0", "0") or "0"
    ).strip().lower() in {"1", "true", "yes", "y", "on"}
    is_main = bool(getattr(accelerator, "is_main_process", True))
    if eval_only_rank0:
        zero_stage = _deepspeed_zero_stage(accelerator)
        if zero_stage >= 3 and int(getattr(accelerator, "num_processes", 1) or 1) > 1:
            use_vllm = True
            generator = getattr(ctx, "generator", None)
            generator_self = getattr(generator, "__self__", None)
            if generator_self is not None:
                gen_ctx = getattr(generator_self, "ctx", None)
                if gen_ctx is not None:
                    use_vllm = bool(getattr(gen_ctx, "use_vllm", False))
            if not use_vllm:
                if is_main:
                    LOG.warning(
                        "Disabling MAXENT_EVAL_ONLY_RANK0 under ZeRO-3 local eval generation "
                        "(zero_stage=%s, use_vllm=%s).",
                        zero_stage,
                        use_vllm,
                    )
                eval_only_rank0 = False
    _maybe_presync_vllm_for_eval(ctx, eval_only_rank0)
    if eval_only_rank0 and callable(wait_for_all):
        wait_for_all()
    if eval_only_rank0 and not is_main:
        if callable(wait_for_all):
            # Barrier before eval starts.
            wait_for_all()
            # Barrier after rank 0 finishes eval.
            wait_for_all()
        return
    shard = _build_eval_shard(evaluation_cfg.rows, accelerator)
    if eval_only_rank0:
        total_rows = len(evaluation_cfg.rows)
        shard = _EvalShardInfo(
            rows=evaluation_cfg.rows,
            total_rows=total_rows,
            shard_total=total_rows,
            world_size=1,
            log_every=max(1, total_rows // 10) if total_rows else 1,
            is_main=True,
            rank=0,
        )
    _log_eval_start(step, shard, evaluation_cfg.batch_size)

    model = ctx.model
    prev_mode = getattr(model, "training", False)
    model.eval()

    try:
        eval_scores, fmt_counts, score_stats = _run_eval_batches(
            shard,
            evaluation_cfg.batch_size,
            ctx,
            step,
        )
        if eval_only_rank0:
            total_sum = float(sum(eval_scores))
            total_count = float(len(eval_scores))
            total_fmt = fmt_counts
            total_score = score_stats or {}
        else:
            gathered_stats = _gather_eval_stats(
                accelerator, eval_scores, fmt_counts, score_stats
            )
            total_sum, total_count = gathered_stats[:2]
            total_fmt = (
                gathered_stats[2]
                if len(gathered_stats) >= 3
                else {
                    "missing_answer": 0.0,
                    "missing_format": 0.0,
                    "total": float(total_count),
                }
            )
            total_score = (
                gathered_stats[3] if len(gathered_stats) >= 4 else {}
            )
        if shard.is_main:
            mean_reward = total_sum / max(total_count, 1.0)
            sample_total = int(shard.total_rows)
            logging.getLogger(__name__).info(
                "eval step %d | mean_reward=%.4f | samples=%d | missing_answer_frac=%.4f | missing_format_frac=%.4f",
                step,
                mean_reward,
                sample_total,
                float(total_fmt.get("missing_answer", 0.0))
                / max(float(total_fmt.get("total", 0.0)), 1.0),
                float(total_fmt.get("missing_format", 0.0))
                / max(float(total_fmt.get("total", 0.0)), 1.0),
            )
            ctx.logging.log_metrics(
                {
                    "eval/mean_reward": mean_reward,
                    "eval/samples": sample_total,
                    "eval/format/missing_answer_frac": float(
                        total_fmt.get("missing_answer", 0.0)
                    )
                    / max(float(total_fmt.get("total", 0.0)), 1.0),
                    "eval/format/missing_format_frac": float(
                        total_fmt.get("missing_format", 0.0)
                    )
                    / max(float(total_fmt.get("total", 0.0)), 1.0),
                },
                step,
            )
            eval_metrics = {
                "eval/mean_reward": mean_reward,
                "eval/samples": sample_total,
                "eval/format/missing_answer_frac": float(
                    total_fmt.get("missing_answer", 0.0)
                )
                / max(float(total_fmt.get("total", 0.0)), 1.0),
                "eval/format/missing_format_frac": float(
                    total_fmt.get("missing_format", 0.0)
                )
                / max(float(total_fmt.get("total", 0.0)), 1.0),
            }
            if total_score:
                kl_token_sum = float(total_score.get("kl_token_sum", 0.0))
                kl_sum = float(total_score.get("kl_sum", 0.0))
                kl_count = float(total_score.get("kl_count", 0.0))
                kl_weighted = float(total_score.get("kl_weighted_sum", 0.0))
                if kl_token_sum > 0:
                    eval_metrics["eval/kl"] = kl_weighted / kl_token_sum
                if kl_count > 0:
                    eval_metrics["eval/kl_mean"] = kl_sum / kl_count
                entropy_sum = float(total_score.get("entropy_sum", 0.0))
                entropy_token_sum = float(total_score.get("entropy_token_sum", 0.0))
                if entropy_token_sum > 0:
                    eval_metrics["eval/policy_entropy"] = entropy_sum / entropy_token_sum
                length_sum = float(total_score.get("length_sum", 0.0))
                length_count = float(total_score.get("length_count", 0.0))
                if length_count > 0:
                    eval_metrics["eval/avg_completion_tokens"] = length_sum / length_count
                    eval_metrics["eval/completions/clipped_frac"] = float(
                        total_score.get("clipped_count", 0.0)
                    ) / length_count
                terminated_sum = float(total_score.get("terminated_sum", 0.0))
                terminated_count = float(total_score.get("terminated_count", 0.0))
                if terminated_count > 0:
                    eval_metrics["eval/completions/mean_length_terminated"] = (
                        terminated_sum / terminated_count
                    )
            _log_wandb(getattr(ctx.logging, "wandb_run", None), eval_metrics, step)
            accel_log = getattr(accelerator, "log", None)
            if callable(accel_log):
                try:
                    accel_log(eval_metrics, step=step)
                except TypeError:
                    accel_log(eval_metrics)
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
                    _log_wandb(
                        getattr(ctx.logging, "wandb_run", None), seed_metrics, step
                    )
                    accel_log = getattr(accelerator, "log", None)
                    if callable(accel_log):
                        try:
                            accel_log(seed_metrics, step=step)
                        except TypeError:
                            accel_log(seed_metrics)
        if eval_only_rank0 and callable(wait_for_all):
            wait_for_all()
    finally:
        if prev_mode:
            model.train()
        if callable(wait_for_all):
            wait_for_all()
        # Reduce CUDA fragmentation after eval.
        torch_mod = _refresh_torch()
        cuda_mod = getattr(torch_mod, "cuda", None)
        if cuda_mod is not None and getattr(cuda_mod, "is_available", lambda: False)():
            try:
                cuda_mod.empty_cache()
            except (AttributeError, RuntimeError):
                LOG.debug("Failed to empty CUDA cache after eval.")
        if callable(wait_for_all):
            wait_for_all()


__all__ = ["run_validation_step"]
