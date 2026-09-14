#!/usr/bin/env python3
"""Paired-vector statistics for the E117 successor development screen."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
import math
import re
from typing import Any


ARMS = ("c", "p", "f")
SENTINELS = (
    "qwen05b/countdown",
    "qwen05b/graph_coloring",
    "qwen05b/python_factors",
    "falcon1b/mathir",
)
PRIMITIVES = ("pass_at_8", "raw_distinct_at_8")
DERIVED = "adjusted_breadth_at_8"
IDENTITY_FIELDS = ("prompt_surface_sha256", "request_surface_sha256")
COMPONENTS = {
    "proposal_replay": ("p", "c"),
    "semantic_increment": ("f", "p"),
}
ACTIONABLE_RAW_DISTINCT_EFFECT = 0.05
MIN_MEAN_PASS_EFFECT = -0.03
MIN_SEED_PASS_EFFECT = -0.10
STAGE1_TRAINING_SEEDS = (201, 202, 203)
EVALUATION_DRAWS = tuple(range(16))
CHECKPOINTS = tuple(range(0, 3072 + 1, 192))
STAGE1_START_ORDERS = {
    201: "C-P-F",
    202: "P-F-C",
    203: "F-C-P",
}
CONFIRMATION_TRAINING_SEEDS = tuple(range(301, 307))
CONFIRMATION_START_ORDERS = {
    301: "C-P-F",
    302: "P-F-C",
    303: "F-C-P",
    304: "C-P-F",
    305: "P-F-C",
    306: "F-C-P",
}


def mean(values: Sequence[float]) -> float:
    if not values:
        raise RuntimeError("cannot average an empty sequence")
    return sum(values) / len(values)


def sample_se(values: Sequence[float]) -> float:
    if len(values) < 2:
        raise RuntimeError("a standard error requires at least two values")
    center = mean(values)
    variance = sum((value - center) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance / len(values))


def normalized_auc(checkpoints: Sequence[int], values: Sequence[float]) -> float:
    """Trapezoidal AUC divided by the complete registered horizon."""

    if len(checkpoints) != len(values) or len(checkpoints) < 2:
        raise RuntimeError("AUC requires one value per checkpoint")
    if any(right <= left for left, right in zip(checkpoints, checkpoints[1:])):
        raise RuntimeError("checkpoints must be strictly increasing")
    horizon = checkpoints[-1] - checkpoints[0]
    return (
        sum(
            (right - left) * (left_value + right_value) / 2.0
            for left, right, left_value, right_value in zip(
                checkpoints, checkpoints[1:], values, values[1:]
            )
        )
        / horizon
    )


def registered(values: Sequence[Any], *, name: str, minimum: int) -> tuple[Any, ...]:
    result = tuple(values)
    if len(result) < minimum or len(set(result)) != len(result):
        raise RuntimeError(
            f"{name} must contain at least {minimum} unique registered values"
        )
    return result


def metric(vector: tuple[float, float], name: str) -> float:
    sampled_pass, raw_distinct = vector
    if name == PRIMITIVES[0]:
        return sampled_pass
    if name == PRIMITIVES[1]:
        return raw_distinct
    if name == DERIVED:
        return raw_distinct - sampled_pass
    raise KeyError(name)


def sha256_value(value: Any, *, where: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise RuntimeError(f"{where} must be a lowercase SHA-256 digest")
    return value


def endpoint_value(value: Any, *, where: str) -> float:
    """Accept native JSON numbers only; never coerce strings or Booleans."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{where} must be a native JSON number")
    return float(value)


def surface_summary(
    surface: dict[tuple[int, int], float],
    seeds: tuple[int, ...],
    draws: tuple[int, ...],
) -> dict[str, Any]:
    """Keep seed and evaluation Monte Carlo variability visibly separate."""

    per_seed = []
    seed_effects = []
    for seed in seeds:
        draw_effects = [surface[(seed, draw)] for draw in draws]
        estimate = mean(draw_effects)
        seed_effects.append(estimate)
        per_seed.append(
            {
                "training_seed": seed,
                "estimate": estimate,
                "evaluation_mc_se": sample_se(draw_effects),
                "draw_effects": draw_effects,
            }
        )
    draw_means = [mean([surface[(seed, draw)] for seed in seeds]) for draw in draws]
    return {
        "estimate": mean(seed_effects),
        "training_seed_se": sample_se(seed_effects),
        "evaluation_mc_se": sample_se(draw_means),
        "positive_training_seeds": sum(value > 0.0 for value in seed_effects),
        "n_training_seeds": len(seeds),
        "n_evaluation_draws": len(draws),
        "per_training_seed": per_seed,
        "draw_mean_effects": draw_means,
    }


def effect_surface(
    index: dict[tuple[str, int, str, int, int], tuple[float, float]],
    sentinel: str,
    numerator: str,
    denominator: str,
    endpoint: str,
    summary: str,
    seeds: tuple[int, ...],
    draws: tuple[int, ...],
    checkpoints: tuple[int, ...],
) -> dict[tuple[int, int], float]:
    result = {}
    for seed in seeds:
        for draw in draws:
            numerator_curve = [
                metric(index[(sentinel, seed, numerator, step, draw)], endpoint)
                for step in checkpoints
            ]
            denominator_curve = [
                metric(index[(sentinel, seed, denominator, step, draw)], endpoint)
                for step in checkpoints
            ]
            if summary == "terminal":
                value = numerator_curve[-1] - denominator_curve[-1]
            elif summary == "normalized_auc":
                value = normalized_auc(checkpoints, numerator_curve) - normalized_auc(
                    checkpoints, denominator_curve
                )
            else:
                raise KeyError(summary)
            result[(seed, draw)] = value
    return result


def contrast_summary(
    index: dict[tuple[str, int, str, int, int], tuple[float, float]],
    sentinel: str,
    numerator: str,
    denominator: str,
    seeds: tuple[int, ...],
    draws: tuple[int, ...],
    checkpoints: tuple[int, ...],
) -> dict[str, Any]:
    result = {}
    for summary in ("terminal", "normalized_auc"):
        result[summary] = {}
        for endpoint in (*PRIMITIVES, DERIVED):
            surface = effect_surface(
                index,
                sentinel,
                numerator,
                denominator,
                endpoint,
                summary,
                seeds,
                draws,
                checkpoints,
            )
            result[summary][endpoint] = surface_summary(surface, seeds, draws)
    return result


def pass_safety(
    contrast: dict[str, Any],
    *,
    label: str,
) -> tuple[dict[str, bool], dict[str, Any]]:
    checks = {}
    summaries = {}
    for summary in ("terminal", "normalized_auc"):
        pass_effect = contrast[summary][PRIMITIVES[0]]
        seed_effects = [row["estimate"] for row in pass_effect["per_training_seed"]]
        checks[f"{summary}_mean_pass_safety_{label}"] = (
            pass_effect["estimate"] >= MIN_MEAN_PASS_EFFECT
        )
        checks[f"{summary}_every_seed_pass_safety_{label}"] = (
            min(seed_effects) >= MIN_SEED_PASS_EFFECT
        )
        summaries[summary] = {
            "pass_effect": pass_effect["estimate"],
            "minimum_paired_seed_pass_effect": min(seed_effects),
        }
    return checks, summaries


def actionability(
    component: dict[str, Any],
    safety_vs_c: dict[str, Any],
    *,
    minimum_positive_training_seeds: int = 2,
    require_two_training_seed_se: bool = False,
) -> dict[str, Any]:
    checks = {}
    for summary in ("terminal", "normalized_auc"):
        raw_distinct = component[summary][PRIMITIVES[1]]
        checks[f"{summary}_raw_distinct_effect_gt_0p05"] = (
            raw_distinct["estimate"] > ACTIONABLE_RAW_DISTINCT_EFFECT
        )
        checks[f"{summary}_raw_distinct_effect_gt_two_mc_se"] = (
            raw_distinct["estimate"] > 2.0 * raw_distinct["evaluation_mc_se"]
        )
        positive_label = (
            "two_of_three"
            if minimum_positive_training_seeds == 2
            and raw_distinct["n_training_seeds"] == 3
            else "five_of_six"
        )
        checks[f"{summary}_raw_distinct_positive_in_{positive_label}_seeds"] = (
            raw_distinct["positive_training_seeds"] >= minimum_positive_training_seeds
        )
        if require_two_training_seed_se:
            checks[f"{summary}_raw_distinct_effect_gt_two_training_seed_se"] = (
                raw_distinct["estimate"] > 2.0 * raw_distinct["training_seed_se"]
            )
    component_checks, component_safety = pass_safety(
        component,
        label="vs_component_denominator",
    )
    overall_checks, overall_safety = pass_safety(
        safety_vs_c,
        label="vs_c",
    )
    checks.update(component_checks)
    checks.update(overall_checks)
    return {
        "actionable": all(checks.values()),
        "advancement_endpoint": PRIMITIVES[1],
        "derived_adjusted_breadth_veto": False,
        "checks": checks,
        "component_pass_safety": component_safety,
        "overall_pass_safety_vs_c": overall_safety,
    }


def _analyze_registered_table(
    rows: Iterable[dict[str, Any]],
    *,
    training_seeds: Sequence[int],
    evaluation_draws: Sequence[int],
    checkpoints: Sequence[int],
    sentinels: Sequence[str] = SENTINELS,
    k: int = 8,
    analysis_kind: str,
    selected_scopes: dict[str, str] | None,
) -> dict[str, Any]:
    """Validate and analyze the exact registered C/P/F seed-draw grid."""

    if analysis_kind not in {"development", "confirmation"}:
        raise RuntimeError(f"unknown E117 analysis kind: {analysis_kind}")
    expected_seeds = (
        STAGE1_TRAINING_SEEDS
        if analysis_kind == "development"
        else CONFIRMATION_TRAINING_SEEDS
    )
    seeds = registered(
        training_seeds,
        name="training_seeds",
        minimum=len(expected_seeds),
    )
    if seeds != expected_seeds:
        label = "201,202,203" if analysis_kind == "development" else "301..306"
        raise RuntimeError(f"the {analysis_kind} analysis requires exact seeds {label}")
    draws = registered(evaluation_draws, name="evaluation_draws", minimum=16)
    if draws != EVALUATION_DRAWS:
        raise RuntimeError(
            "the Stage-1 development screen requires exact draw labels 0..15"
        )
    steps = registered(checkpoints, name="checkpoints", minimum=2)
    if steps != CHECKPOINTS:
        raise RuntimeError(
            "the Stage-1 development screen requires exact checkpoints 0:192:3072"
        )
    sentinel_names = registered(sentinels, name="sentinels", minimum=1)
    if sentinel_names != SENTINELS:
        raise RuntimeError(
            "the Stage-1 development screen requires the exact four "
            "registered sentinels"
        )
    for name, values in (
        ("training seeds", seeds),
        ("evaluation draws", draws),
        ("checkpoints", steps),
    ):
        if any(
            not isinstance(value, int) or isinstance(value, bool) for value in values
        ):
            raise RuntimeError(f"{name} must be integers")
    if steps[0] != 0 or any(right <= left for left, right in zip(steps, steps[1:])):
        raise RuntimeError("checkpoints must start at zero and increase")
    if k != 8 or isinstance(k, bool):
        raise RuntimeError("the E117 successor analysis requires K=8")
    allowed_scopes = {
        "broad_development_candidate",
        "countdown_domain_specific_candidate",
    }
    if analysis_kind == "development":
        if selected_scopes is not None:
            raise RuntimeError("development analysis cannot select confirmation scopes")
    else:
        if (
            not isinstance(selected_scopes, dict)
            or not selected_scopes
            or not set(selected_scopes).issubset(COMPONENTS)
            or any(scope not in allowed_scopes for scope in selected_scopes.values())
        ):
            raise RuntimeError(
                "confirmation requires nonempty Stage-1-selected component scopes"
            )

    allowed = {
        (sentinel, seed, arm, step, draw)
        for sentinel in sentinel_names
        for seed in seeds
        for arm in ARMS
        for step in steps
        for draw in draws
    }
    index = {}
    identities = {}
    for number, row in enumerate(rows, start=1):
        try:
            if (
                not isinstance(row["sentinel"], str)
                or not isinstance(row["arm"], str)
                or any(
                    not isinstance(row[field], int) or isinstance(row[field], bool)
                    for field in (
                        "training_seed",
                        "checkpoint",
                        "evaluation_draw",
                    )
                )
            ):
                raise TypeError("invalid Stage-1 identifier type")
            key = (
                row["sentinel"],
                row["training_seed"],
                row["arm"],
                row["checkpoint"],
                row["evaluation_draw"],
            )
            sampled_pass = endpoint_value(
                row[PRIMITIVES[0]], where=f"Stage-1 row {number} {PRIMITIVES[0]}"
            )
            raw_distinct = endpoint_value(
                row[PRIMITIVES[1]], where=f"Stage-1 row {number} {PRIMITIVES[1]}"
            )
            prompt_surface = sha256_value(
                row[IDENTITY_FIELDS[0]],
                where=f"Stage-1 row {number} {IDENTITY_FIELDS[0]}",
            )
            request_surface = sha256_value(
                row[IDENTITY_FIELDS[1]],
                where=f"Stage-1 row {number} {IDENTITY_FIELDS[1]}",
            )
        except (KeyError, OverflowError, TypeError, ValueError) as exc:
            raise RuntimeError(f"invalid Stage-1 row {number}") from exc
        if key not in allowed:
            raise RuntimeError(f"unregistered Stage-1 row {number}: {key}")
        if key in index:
            raise RuntimeError(f"duplicate Stage-1 row: {key}")
        if not math.isfinite(sampled_pass) or not math.isfinite(raw_distinct):
            raise RuntimeError(f"non-finite Stage-1 endpoint: {key}")
        if not 0.0 <= sampled_pass <= 1.0:
            raise RuntimeError(f"pass@8 outside [0,1]: {key}")
        if not 0.0 <= raw_distinct <= float(k):
            raise RuntimeError(f"raw distinct@8 outside [0,K]: {key}")
        if raw_distinct + 1e-12 < sampled_pass:
            raise RuntimeError(f"raw distinct@8 is below pass@8: {key}")
        index[key] = (sampled_pass, raw_distinct)
        identities[key] = (prompt_surface, request_surface)
    missing = allowed.difference(index)
    if missing:
        raise RuntimeError(
            "incomplete Stage-1 grid: "
            f"missing {len(missing)} rows; first={min(missing)}"
        )

    for sentinel in sentinel_names:
        prompt_surfaces = {
            identities[(sentinel, seed, arm, step, draw)][0]
            for seed in seeds
            for arm in ARMS
            for step in steps
            for draw in draws
        }
        if len(prompt_surfaces) != 1:
            raise RuntimeError(f"response-free prompt surface mismatch: {sentinel}")
        request_surface_by_draw = {}
        for draw in draws:
            request_surfaces = {
                identities[(sentinel, seed, arm, step, draw)][1]
                for seed in seeds
                for arm in ARMS
                for step in steps
            }
            if len(request_surfaces) != 1:
                raise RuntimeError(
                    "common-random-number request surface mismatch: "
                    f"{sentinel}, draw={draw}"
                )
            request_surface_by_draw[draw] = next(iter(request_surfaces))
        if len(set(request_surface_by_draw.values())) != len(draws):
            raise RuntimeError(f"evaluation draws reuse a request surface: {sentinel}")
        for draw in draws:
            vectors = [
                index[(sentinel, seed, arm, 0, draw)] for seed in seeds for arm in ARMS
            ]
            if any(vector != vectors[0] for vector in vectors[1:]):
                raise RuntimeError(
                    "step-zero endpoint mismatch across paired seeds/C/P/F: "
                    f"{sentinel}, draw={draw}"
                )

    sentinel_results = {}
    actionable = {name: [] for name in COMPONENTS}
    for sentinel in sentinel_names:
        components = {}
        for name, (numerator, denominator) in COMPONENTS.items():
            component = contrast_summary(
                index, sentinel, numerator, denominator, seeds, draws, steps
            )
            safety_vs_c = contrast_summary(
                index, sentinel, numerator, "c", seeds, draws, steps
            )
            component["advancement"] = actionability(component, safety_vs_c)
            if analysis_kind == "confirmation":
                component["advancement"] = actionability(
                    component,
                    safety_vs_c,
                    minimum_positive_training_seeds=5,
                    require_two_training_seed_se=True,
                )
            if component["advancement"]["actionable"]:
                actionable[name].append(sentinel)
            components[name] = component
        sentinel_results[sentinel] = {"components": components}

    decisions = {}
    registered_model_families = {
        sentinel.split("/", 1)[0] for sentinel in sentinel_names
    }
    for name, passing in actionable.items():
        passing_model_families = {sentinel.split("/", 1)[0] for sentinel in passing}
        cross_model_support = len(
            registered_model_families
        ) >= 2 and registered_model_families.issubset(passing_model_families)
        broad_candidate = len(passing) >= 3 and cross_model_support
        if set(passing) == {"qwen05b/countdown"}:
            scope = "countdown_domain_specific_candidate"
        elif broad_candidate:
            scope = "broad_development_candidate"
        else:
            scope = "do_not_advance"
        decisions[name] = {
            "actionable_sentinels": passing,
            "actionable_count": len(passing),
            "actionable_model_families": sorted(passing_model_families),
            "cross_model_support": cross_model_support,
            "cross_model_support_meaning": (
                "actionable contexts include both registered model-family labels"
            ),
            "model_domain_factorially_crossed": False,
            "model_family_main_effect_identified": False,
            "broad_candidate": broad_candidate,
            "scope": scope,
        }

    confirmation_decisions = None
    if analysis_kind == "confirmation":
        assert selected_scopes is not None
        confirmation_decisions = {}
        for name, selected_scope in selected_scopes.items():
            if selected_scope == "broad_development_candidate":
                reproduced = bool(decisions[name]["broad_candidate"])
            else:
                reproduced = "qwen05b/countdown" in actionable[name]
            confirmation_decisions[name] = {
                "stage1_selected_scope": selected_scope,
                "scope_reproduced": reproduced,
                "confirmation_scope": selected_scope if reproduced else "not_confirmed",
                "scope_upgrade_permitted": False,
            }

    start_orders = (
        STAGE1_START_ORDERS
        if analysis_kind == "development"
        else CONFIRMATION_START_ORDERS
    )
    prompt_block = "development" if analysis_kind == "development" else "confirmation"

    return {
        "schema": (
            "e117_stage1_paired_vector_statistics_v9"
            if analysis_kind == "development"
            else "e117_confirmation_paired_vector_statistics_v1"
        ),
        "analysis_kind": analysis_kind,
        "development_only": analysis_kind == "development",
        "estimands": {
            "proposal_replay": {
                "contrast": "P-C",
                "kind": "total effect of enabling proposal admission and replay",
                "realized_post_treatment_support_held_fixed": False,
            },
            "semantic_increment": {
                "contrast": "F-P",
                "kind": (
                    "total effect of enabling v7 semantic PPO on the "
                    "proposal-replay system"
                ),
                "controlled_direct_semantic_effect": False,
                "realized_post_treatment_support_held_fixed": False,
                "semantic_without_proposal_arm_required_for_decomposition": True,
            },
        },
        "endpoint_basis": list(PRIMITIVES),
        "derived_endpoint": {
            "name": DERIVED,
            "formula": "raw_distinct_at_8 - pass_at_8",
            "independent_coordinate": False,
            "advancement_veto": False,
        },
        "advancement_basis": {
            "endpoint": PRIMITIVES[1],
            "minimum_effect": ACTIONABLE_RAW_DISTINCT_EFFECT,
            "pass_safety_endpoint": PRIMITIVES[0],
            "pass_safety_summaries": ["terminal", "normalized_auc"],
            "minimum_mean_pass_effect": MIN_MEAN_PASS_EFFECT,
            "minimum_seed_pass_effect": MIN_SEED_PASS_EFFECT,
            "pass_safety_contrasts": [
                "component numerator versus component denominator",
                "candidate numerator versus C",
            ],
            "reason": (
                "advance on the primitive endpoint vector; the derived "
                "subtraction cannot veto simultaneous gains in pass and raw distinct"
            ),
        },
        "uncertainty": {
            "training_seed_se": "SE over draw-averaged paired seed effects",
            "evaluation_mc_se": "SE over seed-averaged common-draw effects",
            "prompt_population_se": None,
            "conditional_prompt_estimand": True,
            "combined_interval": None,
            "p_value": None,
        },
        "inferential_scope": {
            "prompt_target": f"registered finite 128-prompt {prompt_block} bank",
            "prompt_population_inference": False,
            "independent_confirmation_prompt_block_required": True,
            "replication_unit": "registered model-domain context",
            "model_domain_factorially_crossed": False,
            "model_family_main_effect_identified": False,
            "domain_main_effect_identified": False,
            "model_by_domain_interaction_identified": False,
        },
        "design": {
            "arms": list(ARMS),
            "sentinels": list(sentinel_names),
            "registered_model_families": sorted(registered_model_families),
            "training_seeds": list(seeds),
            "actual_start_orders": start_orders,
            "evaluation_draws": list(draws),
            "checkpoints": list(steps),
            "k": k,
            "expected_rows": len(allowed),
            "evaluation_identity": {
                "fields": list(IDENTITY_FIELDS),
                "prompt_surface": "exact within sentinel across all rows",
                "request_surface": (
                    "exact within sentinel and evaluation draw across "
                    "training seeds, arms, and checkpoints"
                ),
                "request_surfaces_distinct_across_draws": True,
                "step_zero_exact_across_training_seeds_and_arms": True,
            },
        },
        "locked_confirmation": {
            "training_seeds": list(CONFIRMATION_TRAINING_SEEDS),
            "actual_start_orders": CONFIRMATION_START_ORDERS,
            "evaluation_draws": list(EVALUATION_DRAWS),
            "checkpoints": list(CHECKPOINTS),
            "minimum_positive_training_seeds": 5,
            "raw_distinct_must_exceed_two_training_seed_se": True,
            "scope_cannot_exceed_stage1_selection": True,
            "development_values_enter_confirmation_estimates": False,
        }
        if analysis_kind == "development"
        else None,
        "confirmation_selection": (
            {
                "selected_scopes": selected_scopes,
                "decisions": confirmation_decisions,
                "development_endpoint_values_used": False,
            }
            if analysis_kind == "confirmation"
            else None
        ),
        "sentinels": sentinel_results,
        "advancement": decisions,
    }


def analyze_registered_table(
    rows: Iterable[dict[str, Any]],
    *,
    training_seeds: Sequence[int] = STAGE1_TRAINING_SEEDS,
    evaluation_draws: Sequence[int] = EVALUATION_DRAWS,
    checkpoints: Sequence[int] = CHECKPOINTS,
    sentinels: Sequence[str] = SENTINELS,
    k: int = 8,
) -> dict[str, Any]:
    """Validate and analyze the exact pre-outcome Stage-1 development grid."""

    return _analyze_registered_table(
        rows,
        training_seeds=training_seeds,
        evaluation_draws=evaluation_draws,
        checkpoints=checkpoints,
        sentinels=sentinels,
        k=k,
        analysis_kind="development",
        selected_scopes=None,
    )


def analyze_confirmation_table(
    rows: Iterable[dict[str, Any]],
    *,
    selected_scopes: dict[str, str],
    training_seeds: Sequence[int] = CONFIRMATION_TRAINING_SEEDS,
    evaluation_draws: Sequence[int] = EVALUATION_DRAWS,
    checkpoints: Sequence[int] = CHECKPOINTS,
    sentinels: Sequence[str] = SENTINELS,
    k: int = 8,
) -> dict[str, Any]:
    """Apply the locked six-seed rule without importing development values."""

    return _analyze_registered_table(
        rows,
        training_seeds=training_seeds,
        evaluation_draws=evaluation_draws,
        checkpoints=checkpoints,
        sentinels=sentinels,
        k=k,
        analysis_kind="confirmation",
        selected_scopes=selected_scopes,
    )
