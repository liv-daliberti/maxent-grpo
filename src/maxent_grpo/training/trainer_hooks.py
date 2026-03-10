"""Trainer helper hooks used by the active TRL/HF training path.

This module intentionally contains only helper utilities still used by
``CustomGRPOTrainer``. Legacy custom-loop execution code lives nowhere in the
runtime path.
"""

from __future__ import annotations

import logging
import math
import os
from typing import Any, Dict, List, Optional

import torch

from .types import TrainingLoopContext
from .weighting.logic import _sync_controller_state, broadcast_controller_state

LOG = logging.getLogger(__name__)

_PROMPT_OBJECTIVE_ENV_VAR = "MAXENT_LOG_PROMPT_OBJECTIVE"
_PROMPT_OBJECTIVE_PREVIEW_LEN = 160


def _prompt_objective_logging_enabled(ctx: TrainingLoopContext) -> bool:
    """Return ``True`` when per-prompt objective logging is explicitly enabled."""

    training_args = getattr(ctx, "training_args", None)
    args_enabled = bool(getattr(training_args, "log_prompt_objective", False))
    env_val = os.environ.get(_PROMPT_OBJECTIVE_ENV_VAR, "")
    if isinstance(env_val, str):
        env_enabled = env_val.strip().lower() in {"1", "true", "yes", "on"}
    else:
        env_enabled = False
    return args_enabled or env_enabled


def _to_cpu_tensor(value: Any) -> torch.Tensor:
    """Best-effort conversion of tensor-like inputs to 1D CPU float tensors."""

    if value is None:
        return torch.tensor([], dtype=torch.float32)
    if isinstance(value, torch.Tensor):
        try:
            return value.detach().float().cpu().view(-1)
        except (RuntimeError, TypeError, ValueError):
            return torch.tensor([], dtype=torch.float32)
    arr = getattr(value, "arr", value)
    try:
        tensor = torch.tensor(arr, dtype=torch.float32)
    except (TypeError, ValueError):
        return torch.tensor([], dtype=torch.float32)
    return tensor.view(-1)


def _entropy_from_probs(probs: Optional[List[float]]) -> float:
    """Return natural-log entropy for a probability vector."""

    if not probs:
        return 0.0
    filtered = [max(float(p), 1e-12) for p in probs if isinstance(p, (int, float))]
    if not filtered:
        return 0.0
    total = sum(filtered)
    if total <= 0.0:
        return 0.0
    normalized = [val / total for val in filtered]
    return float(-sum(val * math.log(val) for val in normalized))


def _per_sequence_kl_values(
    scores: Any, ref_stats: Any, weighting_cfg: Any
) -> List[float]:
    """Return per-sequence KL estimates for prompt-level diagnostics."""

    if scores is None or ref_stats is None or weighting_cfg is None:
        return []
    cur_logp = _to_cpu_tensor(getattr(scores, "cur_logp_sum", None))
    denom = _to_cpu_tensor(getattr(scores, "denom_tok_tensor", None)).clamp(min=1.0)
    count = min(cur_logp.numel(), denom.numel())
    if count <= 0:
        return []
    cur_logp = cur_logp[:count]
    denom = denom[:count]
    cur_per_tok = cur_logp / denom

    ref_source = (
        getattr(ref_stats, "ref_logp_sum", None)
        if getattr(weighting_cfg, "len_norm_ref", False)
        else getattr(ref_stats, "ref_logp_sum_raw", None)
    )
    if ref_source is None:
        ref_source = getattr(ref_stats, "ref_logp_sum", None)
    ref_tensor = _to_cpu_tensor(ref_source)
    if ref_tensor.numel() == 0:
        return []
    if ref_tensor.numel() < count:
        pad_val = float(ref_tensor[-1])
        pad = torch.full((count - ref_tensor.numel(),), pad_val, dtype=ref_tensor.dtype)
        ref_tensor = torch.cat([ref_tensor, pad], dim=0)
    ref_tensor = ref_tensor[:count]
    if getattr(weighting_cfg, "len_norm_ref", False):
        ref_per_tok = ref_tensor
    else:
        ref_per_tok = ref_tensor / denom

    delta = (ref_per_tok - cur_per_tok).clamp(min=-60.0, max=60.0)
    per_seq = delta.exp() - delta - 1.0
    return per_seq.detach().cpu().tolist()


def _prompt_preview(text: str) -> str:
    """Return a compact prompt preview for logs."""

    if not text:
        return ""
    compact = " ".join(str(text).strip().split())
    if len(compact) <= _PROMPT_OBJECTIVE_PREVIEW_LEN:
        return compact
    return compact[: _PROMPT_OBJECTIVE_PREVIEW_LEN - 1] + "..."


def _build_prompt_objective_entries(
    prepared: Any, weighting_cfg: Any
) -> List[Dict[str, Any]]:
    """Return per-prompt summaries of reward, KL, and entropy."""

    if prepared is None or weighting_cfg is None:
        return []
    grouped = getattr(prepared, "grouped_completions", None) or []
    if not grouped:
        return []

    reward_comp = getattr(prepared, "reward_comp", None)
    weight_stats = getattr(prepared, "weight_stats", None)
    if reward_comp is None:
        return []

    rewards_flat = list(getattr(reward_comp, "total_utils", []) or [])
    q_grouped = getattr(reward_comp, "q_grouped", None)
    if q_grouped is None:
        q_dist = getattr(reward_comp, "q_distribution", None)
        q_grouped = getattr(q_dist, "grouped", None)
    if q_grouped is None:
        q_grouped = []

    prompt_pairs = getattr(reward_comp, "pairs", None)
    prompt_texts = list(getattr(prompt_pairs, "prompts", []) or [])
    weight_groups = getattr(weight_stats, "weights_grouped", None) or []
    use_weight_entropy = not (
        weighting_cfg is not None
        and getattr(weighting_cfg, "train_grpo_objective", False)
    )

    kl_values = _per_sequence_kl_values(
        getattr(prepared, "scores", None),
        getattr(prepared, "ref_stats", None),
        weighting_cfg,
    )

    entries: List[Dict[str, Any]] = []
    offset = 0
    for idx, comp_group in enumerate(grouped):
        size = len(comp_group)
        if size <= 0:
            continue
        reward_slice = rewards_flat[offset : offset + size]
        kl_slice = kl_values[offset : offset + size] if kl_values else []
        reward_mean = (
            float(sum(reward_slice) / len(reward_slice)) if reward_slice else 0.0
        )
        kl_mean = float(sum(kl_slice) / len(kl_slice)) if kl_slice else 0.0
        q_entropy = _entropy_from_probs(q_grouped[idx] if idx < len(q_grouped) else [])
        weight_entropy = (
            _entropy_from_probs(weight_groups[idx] if idx < len(weight_groups) else [])
            if use_weight_entropy
            else 0.0
        )
        prompt_text = prompt_texts[offset] if offset < len(prompt_texts) else ""
        entries.append(
            {
                "index": idx,
                "prompt": prompt_text,
                "reward": reward_mean,
                "kl": kl_mean,
                "q_entropy": q_entropy,
                "weight_entropy": weight_entropy,
                "objective": reward_mean + kl_mean + q_entropy,
                "group_size": size,
            }
        )
        offset += size
    return entries


def _log_prompt_objective(ctx: TrainingLoopContext, prepared: Any, step: int) -> None:
    """Emit per-prompt objective breakdown when requested."""

    if not _prompt_objective_logging_enabled(ctx):
        return
    weighting_cfg = getattr(getattr(ctx, "scoring", None), "weighting", None)
    entries = _build_prompt_objective_entries(prepared, weighting_cfg)
    if not entries:
        return
    for entry in entries:
        LOG.info(
            (
                "Prompt objective | step=%d | idx=%d | comps=%d | reward=%.4f | "
                "kl=%.4f | entropy=%.4f | weight_entropy=%.4f | objective=%.4f | prompt=%s"
            ),
            step,
            entry["index"],
            entry["group_size"],
            entry["reward"],
            entry["kl"],
            entry["q_entropy"],
            entry["weight_entropy"],
            entry["objective"],
            _prompt_preview(entry.get("prompt", "")),
        )


def _maybe_overwrite_controller_state_from_config(
    ctx: TrainingLoopContext, controller_resumed: bool = False
) -> None:
    """Optionally force controller scalars to match recipe values."""

    training_args = getattr(ctx, "training_args", None)
    if training_args is None:
        return
    if not getattr(training_args, "controller_overwrite_from_config", False):
        return
    if controller_resumed:
        LOG.info("Controller state resumed from checkpoint; skipping config overwrite.")
        return

    weighting = getattr(getattr(ctx, "scoring", None), "weighting", None)
    if weighting is None:
        return

    def _coerce_scalar(value: Any) -> Optional[float]:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    tau_override = _coerce_scalar(getattr(training_args, "maxent_tau", None))
    beta_override = _coerce_scalar(getattr(training_args, "beta", None))
    if beta_override is None:
        beta_override = _coerce_scalar(getattr(training_args, "init_kl_coeff", None))

    updated = False
    prev_tau = getattr(weighting, "tau", None)
    prev_beta = getattr(weighting, "beta", None)
    if tau_override is not None:
        weighting.tau = tau_override
        updated = True
    if beta_override is not None:
        weighting.beta = beta_override
        updated = True
    if not updated:
        return

    if getattr(weighting, "train_grpo_objective", False):
        weighting.denom = 1.0
    else:
        denom_sum = float(weighting.tau) + float(weighting.beta)
        weighting.denom = denom_sum if denom_sum > 0 else 1.0
    try:
        setattr(weighting, "_tau_entropy_ema", float(weighting.tau))
        setattr(weighting, "_tau_log", math.log(max(float(weighting.tau), 1e-8)))
    except (TypeError, ValueError):
        LOG.debug("Failed to refresh weighting tau tracking fields.")
    _sync_controller_state(weighting)

    accelerator = getattr(getattr(ctx, "runtime", None), "accelerator", None)
    if accelerator is not None:
        broadcast_controller_state(accelerator, weighting)

    prev_tau_float = _coerce_scalar(prev_tau)
    prev_beta_float = _coerce_scalar(prev_beta)
    LOG.info(
        "Overwrote controller state from config | tau=%.4f (prev=%s) | beta=%.4f (prev=%s)",
        float(weighting.tau),
        "nan" if prev_tau_float is None else f"{prev_tau_float:.4f}",
        float(weighting.beta),
        "nan" if prev_beta_float is None else f"{prev_beta_float:.4f}",
    )


def _apply_weighting_overrides_from_config(ctx: TrainingLoopContext) -> None:
    """Apply non-controller weighting toggles from active training config."""

    training_args = getattr(ctx, "training_args", None)
    if training_args is None:
        return
    scoring_cfg = getattr(ctx, "scoring", None)
    weighting = getattr(scoring_cfg, "weighting", None) if scoring_cfg else None
    if weighting is None:
        return
    fallback_flag = getattr(training_args, "maxent_allow_empty_weight_fallback", None)
    if fallback_flag is not None:
        weighting.allow_empty_weight_fallback = bool(fallback_flag)


def _cache_meta_stats(weighting_cfg: Any, weight_view: Any, loss_outputs: Any) -> None:
    """Cache scalar summaries used by meta-controller objectives."""

    entropy_val = getattr(weight_view, "weight_entropy", None)
    if entropy_val is None:
        entropy_val = getattr(weight_view, "entropy", None)
    if isinstance(entropy_val, (int, float)):
        setattr(weighting_cfg, "_meta_entropy_value", float(entropy_val))

    kl_val = getattr(loss_outputs, "kl_loss_scalar", None)
    if isinstance(kl_val, (int, float)):
        setattr(weighting_cfg, "_meta_kl_value", float(kl_val))

    meta_loss = 0.0
    target_entropy = getattr(weighting_cfg, "tau_target_entropy", None)
    if target_entropy is not None and isinstance(entropy_val, (int, float)):
        entropy_error = float(entropy_val) - float(target_entropy)
        meta_loss += 0.5 * entropy_error * entropy_error
    kl_target = getattr(weighting_cfg, "kl_target", None)
    if kl_target and isinstance(kl_val, (int, float)):
        kl_error = float(kl_val) - float(kl_target)
        meta_loss += 0.5 * kl_error * kl_error
    setattr(weighting_cfg, "_meta_last_loss", float(meta_loss))


__all__ = [
    "_apply_weighting_overrides_from_config",
    "_cache_meta_stats",
    "_log_prompt_objective",
    "_maybe_overwrite_controller_state_from_config",
]
