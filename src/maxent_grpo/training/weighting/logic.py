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
import math
import os
from typing import List, Optional, Tuple

from maxent_grpo.training.runtime import require_torch
from ..types import ReferenceLogprobs, RewardComputation
from .types import WeightStats, WeightingSettings

torch = require_torch("training")
CONTROLLER_STATE_FILENAME = "controller_state.json"
_TAU_ENTROPY_EMA_DECAY = 0.9


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
    ref_values = ref_stats.ref_logp_sum if len_norm_ref else ref_stats.ref_logp_sum_raw
    for comps in grouped_completions:
        comp_count = len(comps)
        ref_slice = ref_values[offset : offset + comp_count]
        ref_logp_grouped.append(ref_slice.tolist())
        offset += comp_count
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
    offset = 0
    for comps in grouped_completions:
        comp_count = len(comps)
        raw_slice = ref_stats.ref_logp_sum_raw[offset : offset + comp_count]
        tok_slice = ref_stats.ref_tok_counts[offset : offset + comp_count].clamp(
            min=1.0
        )
        per_token = (raw_slice / tok_slice).tolist()
        ref_logp_per_token.append(per_token)
        offset += comp_count
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
    q_tensor = torch.tensor(q_values, dtype=torch.float32).clamp(min=1e-12)
    log_weight_terms = torch.log(q_tensor) / safe_denom
    if include_reference_term and beta > 0.0:
        ref_tensor = torch.tensor(logp_values, dtype=torch.float32)
        log_weight_terms = log_weight_terms + (beta / safe_denom) * ref_tensor
    probs = torch.softmax(log_weight_terms, dim=0)
    if normalize_by_tokens and token_counts:
        tok_tensor = torch.tensor(token_counts, dtype=torch.float32).clamp(min=1.0)
        probs = probs * tok_tensor
        probs = probs / probs.sum()
    return probs.tolist()


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


def maybe_update_tau(
    weighting_cfg: WeightingSettings,
    weight_stats: WeightStats,
    global_step: int,
) -> None:
    """Adjust tau to hit a target weight entropy if configured.

    :param weighting_cfg: Weighting configuration mutated in-place.
    :type weighting_cfg: WeightingSettings
    :param weight_stats: Current batch weight statistics providing entropy.
    :type weight_stats: WeightStats
    :param global_step: Training step used for warmup/EMA logic.
    :type global_step: int
    """
    target_entropy = weighting_cfg.tau_target_entropy
    if target_entropy is None:
        return
    if global_step <= max(0, weighting_cfg.tau_warmup_steps):
        return
    measured_entropy = weight_stats.weight_entropy
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
    tau_log = tau_log + weighting_cfg.tau_lr * error
    new_tau = math.exp(tau_log)
    new_tau = min(max(new_tau, weighting_cfg.tau_min), weighting_cfg.tau_max)
    weighting_cfg.tau = new_tau
    if weighting_cfg.train_grpo_objective:
        weighting_cfg.denom = 1.0
    else:
        denom_sum = new_tau + weighting_cfg.beta
        weighting_cfg.denom = denom_sum if denom_sum > 0 else 1.0
    setattr(weighting_cfg, "_tau_log", math.log(max(new_tau, 1e-8)))


def controller_state_dict(weighting_cfg: WeightingSettings) -> dict:
    """Return a serializable snapshot of the controller state.

    :param weighting_cfg: Weighting configuration containing tau/beta.
    :type weighting_cfg: WeightingSettings
    :returns: Dictionary describing controller parameters.
    :rtype: dict[str, float]
    """
    tau_log = getattr(weighting_cfg, "_tau_log", math.log(max(weighting_cfg.tau, 1e-8)))
    return {
        "beta": float(weighting_cfg.beta),
        "tau": float(weighting_cfg.tau),
        "tau_log": float(tau_log),
        "tau_entropy_ema": float(
            getattr(weighting_cfg, "_tau_entropy_ema", float("nan"))
        ),
    }


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
    beta = state.get("beta")
    tau = state.get("tau")
    tau_log = state.get("tau_log")
    if not isinstance(beta, (int, float)) or not isinstance(tau, (int, float)):
        return False
    weighting_cfg.beta = float(beta)
    weighting_cfg.tau = float(tau)
    if weighting_cfg.train_grpo_objective:
        weighting_cfg.denom = 1.0
    else:
        denom_sum = weighting_cfg.tau + weighting_cfg.beta
        weighting_cfg.denom = denom_sum if denom_sum > 0 else 1.0
    if isinstance(tau_log, (int, float)):
        setattr(weighting_cfg, "_tau_log", float(tau_log))
    else:
        setattr(weighting_cfg, "_tau_log", math.log(max(weighting_cfg.tau, 1e-8)))
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
        weight_tensor = torch.tensor(weight_group, dtype=torch.float32)
        entropy_vals.append(
            float((-weight_tensor.clamp(min=1e-12).log() * weight_tensor).sum().item())
        )
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
    ref_logp_grouped = (
        split_reference_logprobs(grouped_completions, ref_stats, True)
        if weighting_cfg.len_norm_ref
        else _split_ref_logprobs_per_token(grouped_completions, ref_stats)
    )
    token_counts_grouped = split_reference_token_counts(grouped_completions, ref_stats)
    weights_grouped: List[List[float]] = []
    for q_vals, logp_vals, tok_counts in zip(
        reward_comp.q_grouped, ref_logp_grouped, token_counts_grouped
    ):
        weights_grouped.append(
            weight_vector_from_q(
                q_vals,
                logp_vals,
                tok_counts,
                weighting_cfg,
                include_reference_term=True,
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
