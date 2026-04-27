from __future__ import annotations

from dataclasses import dataclass


def _clip_unit_interval(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)


def _lerp(start: float, end: float, t: float) -> float:
    return float(start) + (float(end) - float(start)) * float(t)


@dataclass(frozen=True)
class CorrectnessScheduleSettings:
    batch_any_correct_mean: float
    any_correct_ema: float
    exploration_level: float
    consolidation_level: float
    budget_max: float
    prompt_select_min_alpha_frac: float
    mode_tau: float
    intra_tau: float


def update_correctness_ema(
    *,
    previous_ema: float | None,
    batch_any_correct_mean: float,
    ema_decay: float,
) -> float:
    batch_value = _clip_unit_interval(batch_any_correct_mean)
    decay = _clip_unit_interval(ema_decay)
    if previous_ema is None:
        return batch_value
    return _clip_unit_interval(
        decay * float(previous_ema) + (1.0 - decay) * batch_value
    )


def build_correctness_scheduled_settings(
    *,
    enabled: bool,
    previous_ema: float | None,
    batch_any_correct_mean: float,
    ema_decay: float,
    correctness_low: float,
    correctness_high: float,
    static_budget_max: float,
    static_prompt_select_min_alpha_frac: float,
    static_mode_tau: float,
    static_intra_tau: float,
    budget_max_early: float,
    budget_max_late: float,
    prompt_select_min_alpha_frac_early: float,
    prompt_select_min_alpha_frac_late: float,
    mode_tau_early: float,
    mode_tau_late: float,
    intra_tau_early: float,
    intra_tau_late: float,
) -> CorrectnessScheduleSettings:
    any_correct_ema = update_correctness_ema(
        previous_ema=previous_ema,
        batch_any_correct_mean=batch_any_correct_mean,
        ema_decay=ema_decay,
    )
    low = _clip_unit_interval(correctness_low)
    high = _clip_unit_interval(correctness_high)
    if high <= low:
        consolidation_level = 1.0 if any_correct_ema >= high else 0.0
    else:
        consolidation_level = _clip_unit_interval(
            (any_correct_ema - low) / (high - low)
        )
    exploration_level = 1.0 - consolidation_level

    if enabled:
        budget_max = _lerp(budget_max_late, budget_max_early, exploration_level)
        prompt_select_min_alpha_frac = _lerp(
            prompt_select_min_alpha_frac_late,
            prompt_select_min_alpha_frac_early,
            exploration_level,
        )
        mode_tau = _lerp(mode_tau_late, mode_tau_early, exploration_level)
        intra_tau = _lerp(intra_tau_late, intra_tau_early, exploration_level)
    else:
        budget_max = float(static_budget_max)
        prompt_select_min_alpha_frac = float(static_prompt_select_min_alpha_frac)
        mode_tau = float(static_mode_tau)
        intra_tau = float(static_intra_tau)

    return CorrectnessScheduleSettings(
        batch_any_correct_mean=_clip_unit_interval(batch_any_correct_mean),
        any_correct_ema=any_correct_ema,
        exploration_level=exploration_level,
        consolidation_level=consolidation_level,
        budget_max=float(budget_max),
        prompt_select_min_alpha_frac=float(prompt_select_min_alpha_frac),
        mode_tau=float(mode_tau),
        intra_tau=float(intra_tau),
    )
