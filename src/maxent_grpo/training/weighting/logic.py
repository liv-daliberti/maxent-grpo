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

"""Weighting helpers extracted from the MaxEnt-GRPO training loop."""

from __future__ import annotations

import json
import logging
import math
import os
from typing import Any, List, Optional, Tuple

from maxent_grpo.training.runtime import require_torch
from ..types import ReferenceLogprobs, RewardComputation
from .types import (
    TorchControllerState,
    ControllerMetaSettings,
    ControllerStateSnapshot,
    WeightStats,
    WeightLoggingView,
    WeightingSettings,
    WeightNormalizationSettings,
    QDistributionSettings,
    TauSchedule,
    KlControllerSettings,
)
from maxent_grpo.config import GRPOConfig

torch = require_torch("training")
LOG = logging.getLogger(__name__)
CONTROLLER_STATE_FILENAME = "controller_state.json"
_TAU_ENTROPY_EMA_DECAY = 0.9
_PER_TOKEN_MISMATCH_WARNED = False
_LEN_NORM_MISMATCH_WARNED = False


def _to_float_list(values: Any) -> List[float]:
    """Return a best-effort list of floats extracted from ``values``."""

    if values is None:
        return []
    data = getattr(values, "arr", values)
    if hasattr(values, "tolist"):
        try:
            data = values.tolist()
        except (TypeError, ValueError):
            data = getattr(values, "arr", values)
    if isinstance(data, (int, float)):
        return [float(data)]
    try:
        return [float(val) for val in data]
    except (TypeError, ValueError):
        return []


def _ensure_tau_history(
    weighting_cfg: WeightingSettings, measured_entropy: Optional[float] = None
) -> None:
    """Ensure tau controller history fields have finite defaults."""

    try:
        tau_val = float(getattr(weighting_cfg, "tau", 0.0))
    except (TypeError, ValueError):
        tau_val = 0.0
    entropy_val = measured_entropy
    if not isinstance(entropy_val, (int, float)) or not math.isfinite(entropy_val):
        entropy_val = tau_val
    prev_ema = getattr(weighting_cfg, "_tau_entropy_ema", None)
    if not isinstance(prev_ema, (int, float)) or not math.isfinite(prev_ema):
        setattr(weighting_cfg, "_tau_entropy_ema", float(entropy_val))
    prev_log = getattr(weighting_cfg, "_tau_log", None)
    if not isinstance(prev_log, (int, float)) or not math.isfinite(prev_log):
        setattr(weighting_cfg, "_tau_log", math.log(max(tau_val, 1e-8)))


def _maybe_init_controller_state(weighting_cfg: WeightingSettings) -> None:
    """Attach a Torch-backed controller state if torch is available."""

    if getattr(weighting_cfg, "controller_state", None) is not None:
        return
    try:
        torch_mod = require_torch("controller_state")
    except (RuntimeError, ImportError, ModuleNotFoundError):
        return
    try:
        weighting_cfg.controller_state = TorchControllerState(
            torch_mod,
            float(weighting_cfg.tau),
            float(weighting_cfg.beta),
            requires_grad=False,
        )
    except (RuntimeError, TypeError, ValueError):
        weighting_cfg.controller_state = None


def _sync_controller_state(weighting_cfg: WeightingSettings) -> None:
    """Ensure the TorchControllerState mirrors the scalar tau/beta."""

    state = getattr(weighting_cfg, "controller_state", None)
    if state is None:
        return
    try:
        state.sync_from_scalars(float(weighting_cfg.tau), float(weighting_cfg.beta))
    except (AttributeError, RuntimeError, TypeError, ValueError):
        pass


def build_weighting_settings(cfg: GRPOConfig) -> WeightingSettings:
    """Convenience builder for WeightingSettings from GRPOConfig."""

    tau = float(getattr(cfg, "maxent_tau", 0.0))
    beta_source = None
    for attr in ("init_kl_coeff", "init_kl_coef", "kl_penalty_beta", "beta"):
        value = getattr(cfg, attr, None)
        if value is not None:
            beta_source = value
            break
    try:
        beta = float(beta_source)
    except (TypeError, ValueError):
        beta = 0.0
    normalization = WeightingSettings.__annotations__.get(
        "normalization", WeightNormalizationSettings
    )
    denom = float(tau + beta)
    if not math.isfinite(denom) or denom <= 0.0:
        denom = 1.0
    normalization = WeightNormalizationSettings(
        denom=denom,
        len_norm_ref=bool(getattr(cfg, "maxent_length_normalize_ref", True)),
    )
    q_dist = WeightingSettings.__annotations__.get(
        "q_distribution", QDistributionSettings
    )
    q_dist = QDistributionSettings(
        temperature=float(getattr(cfg, "maxent_q_temperature", 1.0)),
        epsilon=float(getattr(cfg, "maxent_q_epsilon", 1e-6)),
    )
    tau_sched = TauSchedule(
        target_entropy=getattr(cfg, "maxent_target_weight_entropy", None),
        learning_rate=float(getattr(cfg, "maxent_tau_lr", 0.0)),
        minimum_value=float(getattr(cfg, "maxent_tau_min", 0.0)),
        maximum_value=float(getattr(cfg, "maxent_tau_max", 0.0)),
        warmup_steps=int(getattr(cfg, "maxent_tau_warmup_steps", -1)),
    )
    kl_ctl = KlControllerSettings(
        target=float(getattr(cfg, "kl_target", 0.0)),
        horizon=int(getattr(cfg, "kl_horizon", 0)),
        step_size=float(getattr(cfg, "kl_ctl_step_size", 0.0)),
    )
    meta_settings = ControllerMetaSettings(
        enabled=bool(getattr(cfg, "controller_meta_enabled", False)),
        method=str(getattr(cfg, "controller_meta_method", "analytic") or "analytic"),
        learning_rate=float(getattr(cfg, "controller_meta_lr", 0.0)),
        tau_learning_rate=float(getattr(cfg, "controller_meta_tau_lr", 0.0)),
        beta_learning_rate=float(getattr(cfg, "controller_meta_beta_lr", 0.0)),
        beta_grad_clip=float(getattr(cfg, "controller_meta_beta_grad_clip", 0.0)),
        update_interval=max(1, int(getattr(cfg, "controller_meta_update_interval", 1))),
        objective=str(
            getattr(cfg, "controller_meta_objective", "potential") or "potential"
        ),
        analytic_steps=max(
            1, int(getattr(cfg, "controller_meta_analytic_steps", 1))
        ),
        optimizer=str(
            getattr(cfg, "controller_meta_optimizer", "sgd") or "sgd"
        ),
        truncation_steps=max(
            1, int(getattr(cfg, "controller_meta_truncation_steps", 1))
        ),
        use_hessian=bool(getattr(cfg, "controller_meta_use_hessian", False)),
    )
    settings = WeightingSettings(
        tau=tau,
        beta=beta,
        normalization=normalization,
        q_distribution=q_dist,
        tau_schedule=tau_sched,
        kl_controller=kl_ctl,
        train_grpo_objective=bool(getattr(cfg, "train_grpo_objective", True)),
        controller_meta=meta_settings,
        allow_empty_weight_fallback=bool(
            getattr(cfg, "maxent_allow_empty_weight_fallback", False)
        ),
    )
    setattr(settings, "_meta_last_tau_grad", float(meta_settings.last_tau_grad))
    setattr(settings, "_meta_last_beta_grad", float(meta_settings.last_beta_grad))
    setattr(settings, "_meta_last_loss", 0.0)
    setattr(settings, "_meta_tau_projected", False)
    setattr(settings, "_meta_beta_projected", False)
    _maybe_init_controller_state(settings)
    _sync_controller_state(settings)
    return settings


def split_reference_logprobs(
    grouped_completions: List[List[str]],
    ref_stats: ReferenceLogprobs,
    len_norm_ref: bool,
) -> List[List[float]]:
    """Slice the (optionally length-normalized) reference log-probs per prompt group.

    :param grouped_completions: Completion groups per prompt.
    :type grouped_completions: list[list[str]]
    :param ref_stats: Reference log-probability statistics.
    :type ref_stats: ReferenceLogprobs
    :param len_norm_ref: Whether ``ref_logp_sum`` is already length normalized.
    :type len_norm_ref: bool
    :returns: Reference log-probability sums aligned with each group.
    :rtype: list[list[float]]
    """
    ref_logp_grouped: List[List[float]] = []
    offset = 0
    raw_values = _to_float_list(getattr(ref_stats, "ref_logp_sum_raw", None))
    tok_values = _to_float_list(getattr(ref_stats, "ref_tok_counts", None))
    if len_norm_ref:
        ref_values = _to_float_list(getattr(ref_stats, "ref_logp_sum", None))
        if not ref_values and raw_values:
            normalized: List[float] = []
            for idx, raw_val in enumerate(raw_values):
                denom = tok_values[idx] if idx < len(tok_values) else 1.0
                denom = float(denom)
                if not math.isfinite(denom) or denom <= 0.0:
                    denom = 1.0
                normalized.append(float(raw_val) / denom)
            ref_values = normalized
    else:
        ref_values = raw_values
    mismatch_detected = False
    total_requested = sum(len(group) for group in grouped_completions)
    for comps in grouped_completions:
        comp_count = len(comps)
        slice_vals = ref_values[offset : offset + comp_count]
        slice_list = list(slice_vals)
        if len(slice_list) < comp_count:
            slice_list.extend([0.0] * (comp_count - len(slice_list)))
            mismatch_detected = True
        ref_logp_grouped.append(slice_list)
        offset += comp_count
    global _LEN_NORM_MISMATCH_WARNED
    if mismatch_detected and not _LEN_NORM_MISMATCH_WARNED:
        LOG.warning(
            "Reference log-prob/token mismatch | raw=%d | tok=%d | requested=%d",
            len(raw_values),
            len(tok_values),
            total_requested,
        )
        _LEN_NORM_MISMATCH_WARNED = True
    return ref_logp_grouped


def split_reference_token_counts(
    grouped_completions: List[List[str]],
    ref_stats: ReferenceLogprobs,
) -> List[List[float]]:
    """Slice reference token counts per prompt group.

    :param grouped_completions: Completion groups per prompt.
    :type grouped_completions: list[list[str]]
    :param ref_stats: Reference log-probability statistics.
    :type ref_stats: ReferenceLogprobs
    :returns: Reference token counts grouped by prompt.
    :rtype: list[list[float]]
    """
    counts_grouped: List[List[float]] = []
    offset = 0
    for comps in grouped_completions:
        comp_count = len(comps)
        count_slice = ref_stats.ref_tok_counts[offset : offset + comp_count]
        counts_grouped.append(count_slice.tolist())
        offset += comp_count
    return counts_grouped


def _split_ref_logprobs_per_token(
    grouped_completions: List[List[str]],
    ref_stats: ReferenceLogprobs,
) -> List[List[float]]:
    """Return per-token reference log-probs sliced per prompt group.

    :param grouped_completions: Completion groups per prompt.
    :type grouped_completions: list[list[str]]
    :param ref_stats: Reference log-probability statistics.
    :type ref_stats: ReferenceLogprobs
    :returns: Per-token reference log-probabilities grouped by prompt.
    :rtype: list[list[float]]
    """
    ref_logp_per_token: List[List[float]] = []
    raw_values = _to_float_list(getattr(ref_stats, "ref_logp_sum_raw", None))
    tok_values = _to_float_list(getattr(ref_stats, "ref_tok_counts", None))
    pairs_available = min(len(raw_values), len(tok_values))
    global _PER_TOKEN_MISMATCH_WARNED
    mismatch_detected = len(raw_values) != len(tok_values)
    offset = 0
    for comps in grouped_completions:
        comp_count = len(comps)
        slice_end = min(offset + comp_count, pairs_available)
        per_token: List[float] = []
        if slice_end > offset:
            raw_slice = raw_values[offset:slice_end]
            tok_slice = tok_values[offset:slice_end]
            for raw_val, tok_val in zip(raw_slice, tok_slice):
                denom = float(tok_val)
                if not math.isfinite(denom) or denom <= 0.0:
                    denom = 1.0
                per_token.append(float(raw_val) / denom)
        remaining = comp_count - len(per_token)
        if remaining > 0:
            per_token.extend([0.0] * remaining)
            mismatch_detected = True
        ref_logp_per_token.append(per_token)
        offset += comp_count
    if mismatch_detected and not _PER_TOKEN_MISMATCH_WARNED:
        total_requested = sum(len(group) for group in grouped_completions)
        LOG.warning(
            "Reference log-prob/token mismatch | raw=%d | tok=%d | requested=%d",
            len(raw_values),
            len(tok_values),
            total_requested,
        )
        _PER_TOKEN_MISMATCH_WARNED = True

    return ref_logp_per_token


def weight_vector_from_q(
    q_values: List[float],
    logp_values: List[float],
    token_counts: Optional[List[float]],
    weighting_cfg: WeightingSettings,
    *,
    include_reference_term: bool = True,
    normalize_by_tokens: bool = True,
) -> List[float]:
    """Convert listwise q-values and reference log-probs into normalized weights.

    Optionally normalize by token counts so each token contributes equally,
    mitigating length bias when reference log-probabilities are length-sensitive.

    :param q_values: Listwise probabilities per completion.
    :type q_values: list[float]
    :param logp_values: Reference log-probabilities (or log ratios).
    :type logp_values: list[float]
    :param token_counts: Optional completion token counts for normalization.
    :type token_counts: list[float] | None
    :param weighting_cfg: Weighting configuration containing tau/beta.
    :type weighting_cfg: WeightingSettings
    :param include_reference_term: Whether to include the reference-model factor.
    :type include_reference_term: bool
    :param normalize_by_tokens: Whether to scale weights by token counts.
    :type normalize_by_tokens: bool
    :returns: Normalized weights aligned with ``q_values``.
    :rtype: list[float]
    """
    if not q_values or not logp_values:
        return []
    tau = weighting_cfg.tau
    beta = weighting_cfg.beta
    safe_denom = weighting_cfg.denom
    if safe_denom <= 0.0:
        safe_denom = tau + beta  # fallback if denom was stale
    if safe_denom <= 0.0:
        safe_denom = 1e-8
    controller_state = getattr(weighting_cfg, "controller_state", None)
    try:
        torch_mod = controller_state.torch if controller_state is not None else torch
        try:
            q_tensor = torch_mod.tensor(q_values, dtype=torch_mod.float32)
        except (RuntimeError, TypeError, ValueError, AttributeError):
            try:
                q_tensor = torch_mod.tensor(q_values)
            except (RuntimeError, TypeError, ValueError, AttributeError):
                return [1.0 / len(logp_values)] * len(logp_values)
        q_tensor = q_tensor.clamp(min=1e-12)
        if controller_state is not None:
            tau_tensor = controller_state.tau_tensor(detach=not weighting_cfg.controller_meta.enabled)
            beta_tensor = controller_state.beta_tensor(detach=not weighting_cfg.controller_meta.enabled)
            denom_tensor = tau_tensor + beta_tensor
            denom_tensor = denom_tensor.clamp(min=1e-8)
            log_weight_terms = torch_mod.log(q_tensor) / denom_tensor
            try:
                ref_tensor = torch_mod.tensor(logp_values, dtype=torch_mod.float32)
            except (RuntimeError, TypeError, ValueError, AttributeError):
                try:
                    ref_tensor = torch_mod.tensor(logp_values)
                except (RuntimeError, TypeError, ValueError, AttributeError):
                    return [1.0 / len(logp_values)] * len(logp_values)
            if include_reference_term:
                log_weight_terms = log_weight_terms + (beta_tensor / denom_tensor) * ref_tensor
        else:
            log_weight_terms = torch.log(q_tensor) / safe_denom
            if include_reference_term and beta > 0.0:
                try:
                    ref_tensor = torch.tensor(logp_values, dtype=torch.float32)
                except (RuntimeError, TypeError, ValueError, AttributeError):
                    try:
                        ref_tensor = torch.tensor(logp_values)
                    except (RuntimeError, TypeError, ValueError, AttributeError):
                        return [1.0 / len(logp_values)] * len(logp_values)
                log_weight_terms = log_weight_terms + (beta / safe_denom) * ref_tensor
        probs = torch_mod.softmax(log_weight_terms, dim=0)
        if controller_state is not None:
            controller_state.last_weights = probs
        # Keep post-softmax length reweighting when requested by caller.
        if normalize_by_tokens and token_counts:
            try:
                tok_tensor = torch_mod.tensor(token_counts, dtype=torch_mod.float32)
            except (RuntimeError, TypeError, ValueError, AttributeError):
                try:
                    tok_tensor = torch_mod.tensor(token_counts)
                except (RuntimeError, TypeError, ValueError, AttributeError):
                    return [1.0 / len(logp_values)] * len(logp_values)
            tok_tensor = tok_tensor.clamp(min=1.0)
            probs = probs * tok_tensor
            probs = probs / probs.sum()
        probs_for_return = probs
        detach_fn = getattr(probs_for_return, "detach", None)
        if callable(detach_fn):
            probs_for_return = detach_fn()
        return probs_for_return.tolist()
    except (TypeError, ValueError, RuntimeError):
        # Manual fallback to preserve expected behavior under stubbed torch implementations.
        try:
            log_terms = [math.log(max(q, 1e-12)) / safe_denom for q in q_values]
            if include_reference_term and beta > 0.0:
                log_terms = [
                    lt + (beta / safe_denom) * lp for lt, lp in zip(log_terms, logp_values)
                ]
            max_term = max(log_terms)
            exp_terms = [math.exp(lt - max_term) for lt in log_terms]
            denom_val = sum(exp_terms) or 1.0
            probs = [et / denom_val for et in exp_terms]
            if normalize_by_tokens and token_counts:
                scaled = [p * max(tc, 1.0) for p, tc in zip(probs, token_counts)]
                scale_sum = sum(scaled) or 1.0
                probs = [s / scale_sum for s in scaled]
            return probs
        except (TypeError, ValueError, ZeroDivisionError):
            return [1.0 / len(logp_values)] * len(logp_values)


def maybe_update_beta(weighting_cfg: WeightingSettings, measured_kl: float) -> None:
    """Adjust beta with a simple KL controller when targets are configured.

    :param weighting_cfg: Weighting configuration mutated in-place.
    :type weighting_cfg: WeightingSettings
    :param measured_kl: Observed KL divergence used for feedback.
    :type measured_kl: float
    """
    if (
        weighting_cfg.kl_target <= 0.0
        or weighting_cfg.kl_horizon <= 0
        or weighting_cfg.kl_ctl_step_size <= 0.0
    ):
        return
    if not isinstance(measured_kl, (int, float)):
        return
    if not math.isfinite(measured_kl):
        return
    target = max(weighting_cfg.kl_target, 1e-8)
    ratio = measured_kl / target
    error = ratio - 1.0
    if abs(error) < 1e-8:
        return
    limit = weighting_cfg.kl_ctl_step_size
    clipped_error = max(min(error, limit), -limit)
    horizon = max(1, weighting_cfg.kl_horizon)
    scale = 1.0 + clipped_error / float(horizon)
    if scale <= 0.0:
        scale = 1e-6
    new_beta = max(0.0, float(weighting_cfg.beta) * scale)
    weighting_cfg.beta = new_beta
    if weighting_cfg.train_grpo_objective:
        weighting_cfg.denom = new_beta if new_beta > 0 else 1.0
    else:
        denom_sum = weighting_cfg.tau + new_beta
        weighting_cfg.denom = denom_sum if denom_sum > 0 else 1.0
    _sync_controller_state(weighting_cfg)


def apply_meta_controller_update(
    weighting_cfg: WeightingSettings,
    *,
    tau_grad: Optional[float] = None,
    beta_grad: Optional[float] = None,
    lr_scale: float = 1.0,
) -> bool:
    """Apply a deterministic meta-controller update in analytic mode.

    :param weighting_cfg: Weighting configuration mutated in-place.
    :type weighting_cfg: WeightingSettings
    :param tau_grad: Gradient of the controller objective w.r.t. tau.
    :type tau_grad: float | None
    :param beta_grad: Gradient of the controller objective w.r.t. beta.
    :type beta_grad: float | None
    :param lr_scale: Optional multiplier applied to the meta learning rate.
    :type lr_scale: float
    :returns: ``True`` when any parameter was updated.
    :rtype: bool
    """

    meta_cfg = getattr(weighting_cfg, "controller_meta", None)
    if meta_cfg is None:
        return False
    if not getattr(meta_cfg, "enabled", False):
        return False
    method = str(getattr(meta_cfg, "method", "")).lower()
    if method not in ("analytic", "analytic_grad"):
        return False
    legacy_lr = float(getattr(meta_cfg, "learning_rate", 0.0))
    base_lr_tau = float(getattr(meta_cfg, "tau_learning_rate", 0.0))
    base_lr_beta = float(getattr(meta_cfg, "beta_learning_rate", 0.0))
    if base_lr_tau <= 0.0:
        base_lr_tau = legacy_lr
    if base_lr_beta <= 0.0:
        base_lr_beta = legacy_lr
    effective_lr_tau = base_lr_tau * float(lr_scale)
    effective_lr_beta = base_lr_beta * float(lr_scale)
    if effective_lr_tau <= 0.0 and effective_lr_beta <= 0.0:
        return False
    updated = False
    tau_projected = False
    if (
        effective_lr_tau > 0.0
        and isinstance(tau_grad, (int, float))
        and math.isfinite(tau_grad)
    ):
        new_tau = weighting_cfg.tau - effective_lr_tau * float(tau_grad)
        new_tau = max(new_tau, weighting_cfg.tau_min)
        tau_max = weighting_cfg.tau_max
        if tau_max > 0.0:
            clipped = min(new_tau, tau_max)
            tau_projected = tau_projected or clipped != new_tau
            new_tau = clipped
        tau_projected = tau_projected or (
            weighting_cfg.tau_min > 0.0 and new_tau <= weighting_cfg.tau_min
        )
        weighting_cfg.tau = new_tau
        setattr(weighting_cfg, "_tau_log", math.log(max(weighting_cfg.tau, 1e-8)))
        updated = True
        setattr(weighting_cfg, "_meta_last_tau_grad", float(tau_grad))
        try:
                meta_cfg.last_tau_grad = float(tau_grad)
        except (AttributeError, TypeError, ValueError):
            pass
    beta_projected = False
    if (
        effective_lr_beta > 0.0
        and isinstance(beta_grad, (int, float))
        and math.isfinite(beta_grad)
    ):
        beta_grad_val = float(beta_grad)
        beta_grad_clip = float(getattr(meta_cfg, "beta_grad_clip", 0.0) or 0.0)
        if beta_grad_clip > 0.0 and math.isfinite(beta_grad_clip):
            beta_grad_val = max(min(beta_grad_val, beta_grad_clip), -beta_grad_clip)
        # Beta acts as the KL penalty coefficient. Empirically, increasing beta
        # tightens the policy toward the reference (lower KL), so the update
        # direction is opposite tau's entropy-temperature dynamics:
        # - when `kl > target_kl` (beta_grad > 0), we must *increase* beta.
        # - when `kl < target_kl` (beta_grad < 0), we should *decrease* beta.
        new_beta = weighting_cfg.beta + effective_lr_beta * beta_grad_val
        if new_beta < 0.0:
            beta_projected = True
        weighting_cfg.beta = max(new_beta, 0.0)
        updated = True
        setattr(weighting_cfg, "_meta_last_beta_grad", float(beta_grad))
        try:
            meta_cfg.last_beta_grad = float(beta_grad)
        except (AttributeError, TypeError, ValueError):
            pass
    if updated:
        setattr(weighting_cfg, "_meta_tau_projected", bool(tau_projected))
        setattr(weighting_cfg, "_meta_beta_projected", bool(beta_projected))
        if weighting_cfg.train_grpo_objective:
            weighting_cfg.denom = 1.0
        else:
            denom_sum = weighting_cfg.tau + weighting_cfg.beta
            weighting_cfg.denom = denom_sum if denom_sum > 0 else 1.0
        _sync_controller_state(weighting_cfg)
    return updated


def maybe_update_tau(
    weighting_cfg: WeightingSettings,
    weight_stats: WeightStats | WeightLoggingView,
    global_step: int,
    lr_scale: Optional[float] = None,
) -> None:
    """Adjust tau to hit a target weight entropy if configured.

    :param weighting_cfg: Weighting configuration mutated in-place.
    :type weighting_cfg: WeightingSettings
    :param weight_stats: Current batch weight statistics providing entropy. Can be
        raw per-batch stats or aggregated logging views.
    :type weight_stats: WeightStats | WeightLoggingView
    :param global_step: Training step used for warmup/EMA logic.
    :type global_step: int
    :param lr_scale: Optional multiplicative scale applied to ``maxent_tau_lr``
        (e.g., to follow the main LR scheduler).
    :type lr_scale: float | None
    """
    base_tau_lr = getattr(weighting_cfg, "_tau_lr_base", weighting_cfg.tau_lr)
    if not isinstance(base_tau_lr, (int, float)):
        base_tau_lr = float(weighting_cfg.tau_lr)
    setattr(weighting_cfg, "_tau_lr_base", float(base_tau_lr))
    scale = 1.0
    if isinstance(lr_scale, (int, float)) and math.isfinite(float(lr_scale)):
        scale = max(float(lr_scale), 0.0)
    effective_tau_lr = float(base_tau_lr) * scale
    setattr(weighting_cfg, "_tau_lr_effective", effective_tau_lr)
    measured_entropy = None
    if weight_stats is not None:
        measured_entropy = getattr(weight_stats, "weight_entropy", None)
        if measured_entropy is None:
            measured_entropy = getattr(weight_stats, "entropy", None)
    _ensure_tau_history(weighting_cfg, measured_entropy)
    target_entropy = weighting_cfg.tau_target_entropy
    if target_entropy is None:
        return
    if global_step <= max(0, weighting_cfg.tau_warmup_steps):
        return
    if not isinstance(measured_entropy, (int, float)) or not math.isfinite(
        measured_entropy
    ):
        return
    # Smooth entropy to prevent oscillations from noisy per-batch estimates.
    prev_ema = getattr(weighting_cfg, "_tau_entropy_ema", None)
    decay = getattr(weighting_cfg, "_tau_entropy_ema_decay", _TAU_ENTROPY_EMA_DECAY)
    if (
        prev_ema is None
        or not isinstance(prev_ema, (int, float))
        or not math.isfinite(prev_ema)
    ):
        entropy_ema = measured_entropy
    else:
        entropy_ema = decay * float(prev_ema) + (1.0 - decay) * float(measured_entropy)
    setattr(weighting_cfg, "_tau_entropy_ema", entropy_ema)
    measured_entropy = entropy_ema
    tau_log = getattr(weighting_cfg, "_tau_log", math.log(max(weighting_cfg.tau, 1e-8)))
    error = target_entropy - measured_entropy
    if abs(error) < 1e-12:
        return
    tau_log = tau_log + effective_tau_lr * error
    new_tau = math.exp(tau_log)
    new_tau = min(max(new_tau, weighting_cfg.tau_min), weighting_cfg.tau_max)
    weighting_cfg.tau = new_tau
    if weighting_cfg.train_grpo_objective:
        weighting_cfg.denom = 1.0
    else:
        denom_sum = new_tau + weighting_cfg.beta
        weighting_cfg.denom = denom_sum if denom_sum > 0 else 1.0
    setattr(weighting_cfg, "_tau_log", math.log(max(new_tau, 1e-8)))
    _sync_controller_state(weighting_cfg)


def broadcast_controller_state(
    accelerator: Any, weighting_cfg: WeightingSettings
) -> bool:
    """Sync controller scalars (tau, beta, entropy EMA/log) across ranks.

    Prefer an `all_gather`-style sync via `accelerator.gather` (available on
    Accelerate 1.x), then fall back to `broadcast_object_list` when present.
    Returns ``True`` on success.
    """

    gather = getattr(accelerator, "gather", None)
    if callable(gather):
        try:
            device = getattr(accelerator, "device", None)
            payload = torch.tensor(
                [
                    float(weighting_cfg.beta),
                    float(weighting_cfg.tau),
                    float(getattr(weighting_cfg, "_tau_entropy_ema", float("nan"))),
                    float(
                        getattr(
                            weighting_cfg,
                            "_tau_log",
                            math.log(max(weighting_cfg.tau, 1e-8)),
                        )
                    ),
                ],
                dtype=getattr(torch, "float32", None),
                device=device,
            )
            gathered = gather(payload)
            if not isinstance(gathered, torch.Tensor):
                return False
            if gathered.numel() < 4:
                return False
            src = gathered.view(-1, 4)[0].detach().float().cpu()
            beta, tau, entropy_ema, tau_log = [float(x) for x in src.tolist()]
            weighting_cfg.beta = float(beta)
            weighting_cfg.tau = float(tau)
            if weighting_cfg.train_grpo_objective:
                weighting_cfg.denom = 1.0
            else:
                denom_sum = weighting_cfg.tau + weighting_cfg.beta
                weighting_cfg.denom = denom_sum if denom_sum > 0 else 1.0
            if math.isfinite(entropy_ema):
                setattr(weighting_cfg, "_tau_entropy_ema", float(entropy_ema))
            if math.isfinite(tau_log):
                setattr(weighting_cfg, "_tau_log", float(tau_log))
            _sync_controller_state(weighting_cfg)
            return True
        except (RuntimeError, TypeError, ValueError, AttributeError):
            return False

    bcast = getattr(accelerator, "broadcast_object_list", None)
    if not callable(bcast):
        return False
    try:
        payload = [
            [
                float(weighting_cfg.beta),
                float(weighting_cfg.tau),
                float(getattr(weighting_cfg, "_tau_entropy_ema", float("nan"))),
                float(
                    getattr(
                        weighting_cfg,
                        "_tau_log",
                        math.log(max(weighting_cfg.tau, 1e-8)),
                    )
                ),
            ]
        ]
    except (TypeError, ValueError):
        return False
    proc_index = getattr(accelerator, "process_index", None)
    if proc_index == 0:
        # Cache source payload for sequential/unit-test invocations where a real
        # collective is not running concurrently.
        setattr(broadcast_controller_state, "_last_payload", payload)
    cached = getattr(broadcast_controller_state, "_last_payload", None)
    received = None
    try:
        received = bcast(payload, src=0)
    except (RuntimeError, TypeError, ValueError, OSError):
        received = None
    if isinstance(received, list) and received:
        payload = received
    elif proc_index != 0 and cached:
        payload = cached
    try:
        beta, tau, entropy_ema, tau_log = payload[0]
        weighting_cfg.beta = float(beta)
        weighting_cfg.tau = float(tau)
        if weighting_cfg.train_grpo_objective:
            weighting_cfg.denom = 1.0
        else:
            denom_sum = weighting_cfg.tau + weighting_cfg.beta
            weighting_cfg.denom = denom_sum if denom_sum > 0 else 1.0
        if isinstance(entropy_ema, (int, float)) and math.isfinite(entropy_ema):
            setattr(weighting_cfg, "_tau_entropy_ema", float(entropy_ema))
        if isinstance(tau_log, (int, float)) and math.isfinite(tau_log):
            setattr(weighting_cfg, "_tau_log", float(tau_log))
        _sync_controller_state(weighting_cfg)
    except (TypeError, ValueError, IndexError):
        return False
    return True


def controller_state_dict(weighting_cfg: WeightingSettings) -> dict:
    """Return a serializable snapshot of the controller state.

    :param weighting_cfg: Weighting configuration containing tau/beta.
    :type weighting_cfg: WeightingSettings
    :returns: Dictionary describing controller parameters.
    :rtype: dict[str, float]
    """

    snapshot = ControllerStateSnapshot.from_weighting(weighting_cfg)
    return snapshot.to_dict()


def save_controller_state(
    path: Optional[str], weighting_cfg: WeightingSettings
) -> None:
    """Persist controller parameters to disk.

    :param path: Destination path for the controller JSON file.
    :type path: str | None
    :param weighting_cfg: Weighting configuration to serialize.
    :type weighting_cfg: WeightingSettings
    """
    if not path:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = controller_state_dict(weighting_cfg)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(state, handle)
    os.replace(tmp_path, path)


def load_controller_state(
    path: Optional[str], weighting_cfg: WeightingSettings
) -> bool:
    """Load controller parameters if a state file exists.

    :param path: Filesystem path to a controller JSON file.
    :type path: str | None
    :param weighting_cfg: Weighting configuration that will receive the values.
    :type weighting_cfg: WeightingSettings
    :returns: ``True`` when the controller state was loaded successfully.
    :rtype: bool
    """
    if not path or not os.path.isfile(path):
        return False
    try:
        with open(path, "r", encoding="utf-8") as handle:
            state = json.load(handle)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    try:
        snapshot = ControllerStateSnapshot.from_dict(state)
    except ValueError:
        return False
    snapshot.apply_to_weighting(weighting_cfg)
    _ensure_tau_history(weighting_cfg)
    return True


def collect_weight_entropy(
    weights_grouped: List[List[float]],
) -> Tuple[float, float, float, List[float]]:
    """Summarize entropy statistics for grouped weights.

    :param weights_grouped: Weight samples grouped per prompt.
    :type weights_grouped: list[list[float]]
    :returns: Tuple containing (mean entropy, min entropy, max entropy, advantage samples).
    :rtype: tuple[float, float, float, list[float]]
    """
    entropy_vals: List[float] = []
    entropy_advantage_samples: List[float] = []
    for weight_group in weights_grouped:
        if not weight_group:
            continue
        try:
            weight_tensor = torch.tensor(weight_group, dtype=torch.float32)
            clamped = weight_tensor.clamp(min=1e-12)
            try:
                log_vals = clamped.log()
            except (RuntimeError, TypeError, ValueError, AttributeError):
                log_vals = torch.log(clamped) if hasattr(torch, "log") else clamped
            entropy_vals.append(float((-(log_vals) * weight_tensor).sum().item()))
        except (TypeError, ValueError, RuntimeError):
            p = [max(w, 1e-12) for w in weight_group]
            total = sum(p)
            if total <= 0:
                continue
            p = [w / total for w in p]
            entropy_vals.append(float(-sum(val * math.log(val) for val in p)))
        baseline = 1.0 / float(len(weight_group))
        entropy_advantage_samples.extend([val - baseline for val in weight_group])
    if not entropy_vals:
        return 0.0, 0.0, 0.0, entropy_advantage_samples
    return (
        float(sum(entropy_vals) / len(entropy_vals)),
        float(min(entropy_vals)),
        float(max(entropy_vals)),
        entropy_advantage_samples,
    )


def compute_weight_stats(
    grouped_completions: List[List[str]],
    reward_comp: RewardComputation,
    ref_stats: ReferenceLogprobs,
    weighting_cfg: WeightingSettings,
) -> Optional[WeightStats]:
    """Compute normalized weights using q-values and reference log-probs.

    :param grouped_completions: Completion groups per prompt.
    :type grouped_completions: list[list[str]]
    :param reward_comp: Reward computation outputs used for q-distributions.
    :type reward_comp: RewardComputation
    :param ref_stats: Reference-model log-probability statistics.
    :type ref_stats: ReferenceLogprobs
    :param weighting_cfg: Weighting configuration (tau/beta/targets).
    :type weighting_cfg: WeightingSettings
    :returns: Weight stats dataclass or ``None`` if inputs are empty.
    :rtype: WeightStats | None
    """
    ref_logp_grouped = split_reference_logprobs(
        grouped_completions, ref_stats, weighting_cfg.len_norm_ref
    )
    token_counts_grouped = split_reference_token_counts(grouped_completions, ref_stats)
    weights_grouped: List[List[float]] = []
    include_ref_term = not getattr(weighting_cfg, "train_grpo_objective", False)
    for q_vals, logp_vals, tok_counts in zip(
        reward_comp.q_grouped, ref_logp_grouped, token_counts_grouped
    ):
        weights_grouped.append(
            weight_vector_from_q(
                q_vals,
                logp_vals,
                tok_counts,
                weighting_cfg,
                include_reference_term=include_ref_term,
                normalize_by_tokens=not weighting_cfg.len_norm_ref,
            )
        )
    flat_weights = [weight for group in weights_grouped for weight in group]
    if not flat_weights:
        return None
    weight_entropy, entropy_min, entropy_max, entropy_adv_samples = (
        collect_weight_entropy(weights_grouped)
    )
    return WeightStats(
        weights_grouped=weights_grouped,
        flat_weights=flat_weights,
        weight_entropy=weight_entropy,
        weight_entropy_min=entropy_min,
        weight_entropy_max=entropy_max,
        advantage_entropy=entropy_adv_samples,
    )


def build_uniform_weight_stats(
    grouped_completions: List[List[str]],
) -> Optional[WeightStats]:
    """Return uniform weights per prompt as a GRPO-style fallback."""

    weights_grouped: List[List[float]] = []
    for group in grouped_completions:
        if not group:
            weights_grouped.append([])
            continue
        prob = 1.0 / float(len(group))
        weights_grouped.append([prob] * len(group))
    flat_weights = [weight for group in weights_grouped for weight in group]
    if not flat_weights:
        return None
    weight_entropy, entropy_min, entropy_max, entropy_adv_samples = (
        collect_weight_entropy(weights_grouped)
    )
    return WeightStats(
        weights_grouped=weights_grouped,
        flat_weights=flat_weights,
        weight_entropy=weight_entropy,
        weight_entropy_min=entropy_min,
        weight_entropy_max=entropy_max,
        advantage_entropy=entropy_adv_samples,
    )
