#!/usr/bin/env python3
"""Build the frozen terminal/AUC result surface for E61-R1/E64/E66/E68."""

from __future__ import annotations

from collections import defaultdict
import csv
from datetime import datetime, timezone
import io
import json
import math
import os
from pathlib import Path
import re
import statistics
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[2]


CONTROL = "grpo"
TREATMENT = "verified_first_global_replay_canonical"
REPAIR = "verified_entropy_gated_singleton_escape_canonical"
PLUMBING_CONTROL = "same_plumbing_actuator_off_control"
SEEDS = (43, 44, 45)
MODEBENCH_PASSES = (0, 1, 2, 3, 4, 5, 6, 8, 10, 12)
MATH_PASSES = (0, 2, 4, 6, 8, 10, 12)
QUALITY = ("greedy", "mean8", "pass8", "distinct8")
MODEBENCH = (
    (
        "Graph coloring",
        "gce61r1_e58_vs_grpo_05b_12ep",
        "gce68_separated_support_actuator_05b_12ep",
        192,
    ),
    (
        "Countdown",
        "cde61r1_e58_vs_grpo_05b_12ep",
        "cde68_separated_support_actuator_05b_12ep",
        384,
    ),
    (
        "Python factors",
        "pye61r1_e58_vs_grpo_05b_12ep",
        "pye68_separated_support_actuator_05b_12ep",
        384,
    ),
    (
        "MathIR action menu",
        "mie61r1_e58_vs_grpo_05b_12ep",
        "mie68_separated_support_actuator_05b_12ep",
        384,
    ),
)
PLUMBING_PREFIX = {
    "Graph coloring": "gce66_same_plumbing_control_05b_12ep",
    "Countdown": "cde66_same_plumbing_control_05b_12ep",
    "Python factors": "pye66_same_plumbing_control_05b_12ep",
    "MathIR action menu": "mie66_same_plumbing_control_05b_12ep",
}
MECHANISM_DOMAIN_LABELS = {
    "graph_coloring": "Graph coloring",
    "countdown": "Countdown",
    "python_factor": "Python factors",
    "mathir": "MathIR action menu",
}
MATH_DOMAIN = "Held-out MATH-500 transfer"
OUT_JSON = (
    ROOT / "var/artifacts/e65_five_domain_confirmation_results_latest.json"
)
OUT_MD = ROOT / "paper/results/e65_five_domain_confirmation_live.md"
OUT_CSV = (
    ROOT
    / "paper/results/e65_five_domain_confirmation_fixed_checkpoints_live.csv"
)
AUDIT = ROOT / "var/artifacts/e65_five_domain_confirmation_audit_latest.json"
E61_AUDIT = (
    ROOT / "var/artifacts/e61r1_e58_vs_grpo_12pass_audit_latest.json"
)
E64_AUDIT = ROOT / "var/artifacts/e64_math500_realism_matched_audit_latest.json"
E64_SENSITIVITY = (
    ROOT / "var/artifacts/e64_math500_verifier_sensitivity_latest.json"
)
E68_AUDIT = (
    ROOT / "var/artifacts/e68_separated_support_actuator_ablation_audit_latest.json"
)
E68_EQUIVALENCE = (
    ROOT / "var/artifacts/e68_preintervention_equivalence_audit_latest.json"
)
E68_CHECKPOINT = (
    ROOT / "var/artifacts/e68_checkpoint_separation_audit_latest.json"
)
E68_PAIRED_PROMPT_UNCERTAINTY = (
    ROOT
    / "var/artifacts/"
    "e68_paired_prompt_uncertainty_secondary_v2_latest.json"
)
E65_INVALIDATION = (
    ROOT / "var/artifacts/e65r1_objective_mismatch_invalidation.json"
)
E67_INVALIDATION = (
    ROOT / "var/artifacts/e67_preoptimizer_invalidation.json"
)
E66_AUDIT = (
    ROOT / "var/artifacts/e66_same_plumbing_actuator_ablation_audit_latest.json"
)
EVAL_CADENCE_AUDIT = (
    ROOT / "var/artifacts/e65_eval_cadence_audit_latest.json"
)
CHECKPOINT_COVERAGE_AUDIT = (
    ROOT / "var/artifacts/e65_fixed_checkpoint_coverage_audit_latest.json"
)
E62R10_AUDIT = ROOT / "var/artifacts/e62r10_python_pilot_audit_latest.json"


def _finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _load_json(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _raw_metric(record: dict[str, Any], name: str) -> Any:
    for prefix in ("train/", "actor/", ""):
        value = record.get(f"{prefix}{name}")
        if value is not None:
            return value
    return None


def _latest_jsonl_record(path: Path) -> dict[str, Any]:
    latest: dict[str, Any] = {}
    if not path.is_file():
        return latest
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                candidate = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict):
                latest = candidate
    return latest


def _job_number(path: Path) -> int:
    match = re.search(r"debug_job(\d+)", str(path))
    return int(match.group(1)) if match else -1


def _e58_python_controller_diagnosis() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        pattern = (
            "var/data/"
            "xdr_qwen25_0p5b_instruct_verified_first_global_replay_canonical_"
            "pye61r1_e58_vs_grpo_05b_12ep_"
            f"verified_first_global_replay_canonical_s{seed}/"
            "debug_job*/train_metrics.jsonl"
        )
        paths = sorted(ROOT.glob(pattern), key=_job_number)
        if not paths:
            continue
        record = _latest_jsonl_record(paths[-1])
        entropy = _raw_metric(
            record,
            "semantic_shannon_success_conditioned_signed_open_set_entropy_ema",
        )
        reference = _raw_metric(
            record,
            "semantic_shannon_success_conditioned_signed_open_set_reference_entropy",
        )
        coefficient = _raw_metric(
            record,
            "semantic_shannon_success_conditioned_signed_open_set_coefficient_used",
        )
        support = _raw_metric(
            record,
            "verified_discovery_mean_support_per_prompt",
        )
        outcomes = _raw_metric(record, "verified_discovery_cumulative_outcomes")
        projection = _raw_metric(
            record,
            "canonical_replay_mass_projection_active",
        )
        observations = _raw_metric(
            record,
            "semantic_shannon_success_conditioned_signed_open_set_observations",
        )
        if observations is None:
            observations = _raw_metric(
                record,
                "canonical_replay_mass_observations",
            )
        step = (
            record.get("trainer/global_step")
            or record.get("misc/global_step")
            or record.get("trainer/step")
        )
        ratio = (
            float(entropy) / float(reference)
            if _finite(entropy)
            and _finite(reference)
            and float(reference) > 0
            else None
        )
        rows.append(
            {
                "seed": seed,
                "step": int(step) if _finite(step) else None,
                "entropy_ema": float(entropy) if _finite(entropy) else None,
                "entropy_reference": (
                    float(reference) if _finite(reference) else None
                ),
                "entropy_ratio": ratio,
                "coefficient": (
                    float(coefficient) if _finite(coefficient) else None
                ),
                "projection_active": (
                    int(projection) if _finite(projection) else None
                ),
                "mean_discovered_support": (
                    float(support) if _finite(support) else None
                ),
                "cumulative_discovered_outcomes": (
                    int(outcomes) if _finite(outcomes) else None
                ),
                "controller_detects_entropy_deficit": (
                    ratio is not None and ratio < 1.0
                ),
                "bank_still_singleton": (
                    _finite(support) and float(support) <= 1.000001
                ),
                "controller_observations": (
                    int(observations) if _finite(observations) else None
                ),
            }
        )
    return rows


def _e62r10_engineering_evidence() -> dict[str, Any] | None:
    audit = _load_json(E62R10_AUDIT, {})
    arm = audit.get("arms", {}).get("verified_counterfactual_canonical")
    if not isinstance(arm, dict):
        return None
    proposal = arm.get("proposal", {})
    evaluations = arm.get("evaluations", [])
    terminal = evaluations[-1] if evaluations else {}
    proposal_rows_to_ppo = 0
    run_root = Path(str(arm.get("run_root", "")))
    metrics_path = run_root / "train_metrics.jsonl"
    if metrics_path.is_file():
        with metrics_path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for metric in (
                    "counterfactual_proposal_conditioned_rows_sent_to_ppo",
                    "counterfactual_proposal_transform_rows_sent_to_ppo",
                ):
                    value = _raw_metric(record, metric)
                    if _finite(value):
                        proposal_rows_to_ppo += int(value)
    return {
        "evidential_role": "engineering_pilot_only",
        "seed_count": 1,
        "training_passes": 1,
        "maximum_discovered_mean_support": arm.get(
            "maximum_discovered_mean_support"
        ),
        "admitted_new_outcomes": proposal.get("admitted_new_outcomes"),
        "admissions_per_100_updates": (
            100.0 * float(proposal.get("admitted_new_outcomes", 0))
            / float(arm["latest_step"])
            if _finite(proposal.get("admitted_new_outcomes"))
            and _finite(arm.get("latest_step"))
            and float(arm["latest_step"]) > 0
            else None
        ),
        "proposal_rows_sent_to_ppo": proposal_rows_to_ppo,
        "terminal_pass8": terminal.get("pass8"),
        "terminal_distinct8": terminal.get("distinct8"),
    }


def _mechanism_evidence(repair_audit: dict[str, Any]) -> dict[str, Any]:
    runs = [
        {"domain": domain_name, **run}
        for domain_name, domain in repair_audit.get("domains", {}).items()
        for run in domain.get("runs", [])
    ]
    intervention_events = [
        {
            "domain": run["domain"],
            "seed": run.get("seed"),
            "job_id": run.get("job_id"),
            **event,
        }
        for run in runs
        for event in run.get("intervention_events", [])
    ]
    by_domain: dict[str, dict[str, Any]] = {}
    for domain_name in sorted(repair_audit.get("domains", {})):
        domain_runs = [
            run for run in runs if run["domain"] == domain_name
        ]
        metric_records = sum(
            int(run.get("metric_records", 0)) for run in domain_runs
        )
        interventions = sum(
            int(run.get("interventions", 0)) for run in domain_runs
        )
        by_domain[domain_name] = {
            "metric_records": metric_records,
            "entropy_gated_interventions": interventions,
            "interventions_per_100_updates": (
                100.0 * interventions / metric_records
                if metric_records > 0
                else None
            ),
        }
    total_metric_records = sum(
        int(run.get("metric_records", 0)) for run in runs
    )
    total_interventions = int(
        repair_audit.get("summary", {}).get(
            "entropy_gated_interventions",
            0,
        )
    )
    return {
        "e58_python_live_controller_diagnosis": (
            _e58_python_controller_diagnosis()
        ),
        "prior_support_only_actuator_pilot": _e62r10_engineering_evidence(),
        "corrected_repair_live": {
            "audit_status": repair_audit.get("status", "not_available"),
            "entropy_gated_interventions": total_interventions,
            "metric_records": total_metric_records,
            "interventions_per_100_updates": (
                100.0 * total_interventions / total_metric_records
                if total_metric_records > 0
                else None
            ),
            "by_domain": by_domain,
            "maximum_admitted_per_group": max(
                (
                    int(run.get("maximum_admitted_per_group", 0))
                    for run in runs
                ),
                default=0,
            ),
            "intervention_events": intervention_events,
            "terminal_runs": int(
                repair_audit.get("summary", {}).get("terminal_runs", 0)
            ),
        },
        "interpretation": (
            "E58 can detect an entropy deficit and increase its unprojected "
            "coefficient while a singleton canonical bank still lacks a "
            "second valid target. E62R10 establishes actuator feasibility "
            "but is a one-seed, overactive engineering pilot. E68 is the "
            "corrected confirmatory repair: it retains literal E58's novelty "
            "objective, and the same support-only class of actuator is "
            "eligible only after entropy warmup, measured entropy deficit, "
            "inverse-controller activation, and singleton support, and can "
            "admit at most one alternate to each singleton prompt bank before "
            "that prompt bank switches off."
        ),
    }


def _load_modebench_points(
    prefix: str,
    *,
    steps_per_pass: int,
    allowed_arms: tuple[str, ...],
    arm_alias: str | None = None,
) -> list[dict[str, Any]]:
    path = ROOT / f"var/artifacts/{prefix}_scaling_curve.json"
    rows = _load_json(path, [])
    deduplicated: dict[tuple[str, int, int], dict[str, Any]] = {}
    for row in rows:
        arm = row.get("arm")
        seed = row.get("seed")
        step = row.get("step")
        if (
            arm not in allowed_arms
            or seed not in SEEDS
            or not _finite(step)
            or row.get("split") != "multi_answer"
        ):
            continue
        step = int(step)
        if step not in {
            int(value * steps_per_pass) for value in MODEBENCH_PASSES
        }:
            continue
        resolved_arm = arm_alias or str(arm)
        key = (resolved_arm, int(seed), step)
        point = deduplicated.setdefault(
            key,
            {
                "arm": resolved_arm,
                "seed": int(seed),
                "step": step,
                "passes": step / steps_per_pass,
            },
        )
        for metric, value in row.items():
            if _finite(value):
                point[metric] = float(value)
    return [deduplicated[key] for key in sorted(deduplicated)]


def _load_math_points() -> list[dict[str, Any]]:
    # Keep plotting dependencies out of this module's import surface so the
    # frozen AUC/gate utilities remain testable in the lean training venv.
    import sys

    sys.path.insert(0, str(ROOT / "ops/exp_scaling"))
    from plot_e64_math500_realism import load_e64_aggregate_points

    allowed_steps = {int(value * 384) for value in MATH_PASSES}
    return [
        point
        for point in load_e64_aggregate_points()
        if int(point["step"]) in allowed_steps
        and point["arm"] in (CONTROL, TREATMENT)
        and int(point["seed"]) in SEEDS
    ]


def _trapezoid(points: list[dict[str, Any]]) -> float:
    total = 0.0
    for left, right in zip(points, points[1:]):
        width = float(right["pass"]) - float(left["pass"])
        total += width * (float(left["mean"]) + float(right["mean"])) / 2.0
    return total


def _metric_summary(
    points: list[dict[str, Any]],
    *,
    arm: str,
    metric: str,
    fixed_passes: tuple[int, ...],
    steps_per_pass: int,
) -> dict[str, Any]:
    by_step: dict[int, dict[int, float]] = defaultdict(dict)
    for point in points:
        if point["arm"] != arm or not _finite(point.get(metric)):
            continue
        by_step[int(point["step"])][int(point["seed"])] = float(point[metric])
    complete: list[dict[str, Any]] = []
    for training_pass in fixed_passes:
        step = int(training_pass * steps_per_pass)
        seed_values = by_step.get(step, {})
        if set(seed_values) != set(SEEDS):
            continue
        values = [seed_values[seed] for seed in SEEDS]
        complete.append(
            {
                "pass": training_pass,
                "step": step,
                "seeds": {str(seed): seed_values[seed] for seed in SEEDS},
                "mean": statistics.fmean(values),
                "min": min(values),
                "max": max(values),
            }
        )
    complete_passes = [int(point["pass"]) for point in complete]
    terminal = next(
        (point for point in complete if int(point["pass"]) == 12),
        None,
    )
    full = complete_passes == list(fixed_passes)
    return {
        "complete_checkpoints": complete_passes,
        "checkpoint_count": len(complete_passes),
        "expected_checkpoint_count": len(fixed_passes),
        "points": complete,
        "latest_complete": complete[-1] if complete else None,
        "terminal": terminal,
        "auc": _trapezoid(complete) if full else None,
        "auc_normalized": _trapezoid(complete) / 12.0 if full else None,
        "partial_auc": _trapezoid(complete) if len(complete) >= 2 else None,
        "partial_auc_last_pass": (
            int(complete[-1]["pass"]) if len(complete) >= 2 else None
        ),
    }


def _domain_summary(
    points: list[dict[str, Any]],
    *,
    arms: tuple[str, ...],
    fixed_passes: tuple[int, ...],
    steps_per_pass: int,
    metrics: tuple[str, ...] = QUALITY,
) -> dict[str, Any]:
    return {
        "fixed_checkpoints": list(fixed_passes),
        "arms": {
            arm: {
                "metrics": {
                    metric: _metric_summary(
                        points,
                        arm=arm,
                        metric=metric,
                        fixed_passes=fixed_passes,
                        steps_per_pass=steps_per_pass,
                    )
                    for metric in metrics
                }
            }
            for arm in arms
        },
    }


def _terminal_metric(
    domains: dict[str, Any],
    domain: str,
    arm: str,
    metric: str,
) -> dict[str, Any] | None:
    return domains[domain]["arms"][arm]["metrics"][metric]["terminal"]


def _auc_metric(
    domains: dict[str, Any],
    domain: str,
    arm: str,
    metric: str,
) -> float | None:
    return domains[domain]["arms"][arm]["metrics"][metric]["auc"]


def _paired_delta(
    left: dict[str, Any],
    right: dict[str, Any],
) -> dict[str, float]:
    return {
        str(seed): float(left["seeds"][str(seed)])
        - float(right["seeds"][str(seed)])
        for seed in SEEDS
    }


def _latest_shared_comparison(
    domains: dict[str, Any],
    domain: str,
    left_arm: str,
    right_arm: str,
    metrics: tuple[str, ...],
) -> dict[str, Any] | None:
    shared: set[int] | None = None
    for arm in (left_arm, right_arm):
        for metric in metrics:
            checkpoints = set(
                domains[domain]["arms"][arm]["metrics"][metric][
                    "complete_checkpoints"
                ]
            )
            shared = checkpoints if shared is None else shared & checkpoints
    if not shared:
        return None
    training_pass = max(shared)
    comparison: dict[str, Any] = {"pass": training_pass, "metrics": {}}
    for metric in metrics:
        values: dict[str, Any] = {}
        for arm in (left_arm, right_arm):
            values[arm] = next(
                point
                for point in domains[domain]["arms"][arm]["metrics"][metric][
                    "points"
                ]
                if int(point["pass"]) == training_pass
            )
        comparison["metrics"][metric] = {
            left_arm: values[left_arm],
            right_arm: values[right_arm],
            "mean_delta": (
                float(values[right_arm]["mean"])
                - float(values[left_arm]["mean"])
            ),
            "paired_seed_deltas": _paired_delta(
                values[right_arm],
                values[left_arm],
            ),
        }
    return comparison


def _e58_gate(domains: dict[str, Any], terminal_runs: int) -> dict[str, Any]:
    details: dict[str, Any] = {}
    ready = terminal_runs == 24
    positive = 0
    safe = True
    for domain, *_ in MODEBENCH:
        comparisons: dict[str, Any] = {}
        domain_ready = True
        for metric in ("pass8", "distinct8"):
            treatment_terminal = _terminal_metric(
                domains, domain, TREATMENT, metric
            )
            control_terminal = _terminal_metric(domains, domain, CONTROL, metric)
            treatment_auc = _auc_metric(domains, domain, TREATMENT, metric)
            control_auc = _auc_metric(domains, domain, CONTROL, metric)
            if (
                treatment_terminal is None
                or control_terminal is None
                or treatment_auc is None
                or control_auc is None
            ):
                domain_ready = False
                comparisons[metric] = None
                continue
            comparisons[metric] = {
                "terminal_delta": (
                    float(treatment_terminal["mean"])
                    - float(control_terminal["mean"])
                ),
                "auc_delta": float(treatment_auc) - float(control_auc),
            }
        terminal_pass8 = comparisons.get("pass8")
        no_large_loss = (
            terminal_pass8 is not None
            and float(terminal_pass8["terminal_delta"]) >= -0.05
        )
        domain_positive = domain_ready and all(
            comparisons[metric]["terminal_delta"] > 0
            and comparisons[metric]["auc_delta"] > 0
            for metric in ("pass8", "distinct8")
        )
        positive += int(domain_positive)
        safe = safe and no_large_loss
        ready = ready and domain_ready
        details[domain] = {
            "ready": domain_ready,
            "positive_on_terminal_and_auc_for_both_metrics": domain_positive,
            "terminal_pass8_noninferiority_margin": 0.05,
            "terminal_pass8_no_large_loss": no_large_loss,
            "comparisons": comparisons,
        }
    passed = positive >= 3 and safe
    return {
        "status": "pass" if ready and passed else "fail" if ready else "pending",
        "ready": ready,
        "positive_domains": positive,
        "required_positive_domains": 3,
        "all_domains_within_terminal_pass8_loss_margin": safe,
        "details": details,
    }


def _corrected_repair_gate(
    domains: dict[str, Any],
    *,
    repair_terminal_runs: int,
    comparator_terminal_runs: int,
    interventions: int,
    repair_audit_status: str,
    comparator_audit_status: str,
    equivalence_audit_status: str,
    checkpoint_audit_status: str,
) -> dict[str, Any]:
    ready = repair_terminal_runs == 12 and comparator_terminal_runs == 12
    details: dict[str, Any] = {}
    python_repair = _terminal_metric(
        domains,
        "Python factors",
        REPAIR,
        "pass8",
    )
    python_control = _terminal_metric(
        domains, "Python factors", PLUMBING_CONTROL, "pass8"
    )
    python_ready = python_repair is not None and python_control is not None
    python_pass = False
    if python_ready:
        python_pass = (
            float(python_repair["mean"]) > float(python_control["mean"])
            and float(python_repair["min"]) > float(python_control["min"])
        )
        details["Python factors"] = {
            "mean_delta": float(python_repair["mean"])
            - float(python_control["mean"]),
            "worst_seed_delta": float(python_repair["min"])
            - float(python_control["min"]),
            "mean_and_worst_seed_improve": python_pass,
        }
    else:
        details["Python factors"] = None
    ready = ready and python_ready
    safety_pass = True
    for domain in ("Graph coloring", "Countdown", "MathIR action menu"):
        repair = _terminal_metric(domains, domain, REPAIR, "pass8")
        plumbing_control = _terminal_metric(
            domains,
            domain,
            PLUMBING_CONTROL,
            "pass8",
        )
        domain_ready = repair is not None and plumbing_control is not None
        ready = ready and domain_ready
        if not domain_ready:
            details[domain] = None
            safety_pass = False
            continue
        paired = _paired_delta(repair, plumbing_control)
        mean_delta = float(repair["mean"]) - float(plumbing_control["mean"])
        domain_pass = mean_delta >= -0.05 and min(paired.values()) >= -0.15
        safety_pass = safety_pass and domain_pass
        details[domain] = {
            "mean_delta": mean_delta,
            "paired_seed_deltas": paired,
            "mean_noninferiority_margin": 0.05,
            "paired_seed_loss_margin": 0.15,
            "noninferior": domain_pass,
        }
    mechanism_pass = (
        interventions > 0
        and repair_audit_status == "pass"
        and comparator_audit_status == "pass"
        and equivalence_audit_status == "pass"
        and checkpoint_audit_status == "pass"
    )
    passed = python_pass and safety_pass and mechanism_pass
    return {
        "status": "pass" if ready and passed else "fail" if ready else "pending",
        "ready": ready,
        "causal_comparator": PLUMBING_CONTROL,
        "entropy_gated_interventions": interventions,
        "repair_audit_status": repair_audit_status,
        "comparator_audit_status": comparator_audit_status,
        "preintervention_equivalence_audit_status": (
            equivalence_audit_status
        ),
        "checkpoint_separation_audit_status": checkpoint_audit_status,
        "mechanism_gate_pass": mechanism_pass,
        "python_improvement_pass": python_pass,
        "other_domain_safety_pass": safety_pass if ready else None,
        "details": details,
    }


def _plumbing_consistency_gate(
    domains: dict[str, Any],
    *,
    historical_terminal_runs: int,
    plumbing_terminal_runs: int,
) -> dict[str, Any]:
    ready = historical_terminal_runs == 24 and plumbing_terminal_runs == 12
    details: dict[str, Any] = {}
    passed = True
    for domain, *_ in MODEBENCH:
        historical = _terminal_metric(
            domains,
            domain,
            TREATMENT,
            "pass8",
        )
        plumbing = _terminal_metric(
            domains,
            domain,
            PLUMBING_CONTROL,
            "pass8",
        )
        domain_ready = historical is not None and plumbing is not None
        ready = ready and domain_ready
        if not domain_ready:
            details[domain] = None
            passed = False
            continue
        paired = _paired_delta(plumbing, historical)
        mean_delta = float(plumbing["mean"]) - float(historical["mean"])
        domain_pass = mean_delta >= -0.05 and min(paired.values()) >= -0.15
        passed = passed and domain_pass
        details[domain] = {
            "mean_delta": mean_delta,
            "paired_seed_deltas": paired,
            "mean_noninferiority_margin": 0.05,
            "paired_seed_loss_margin": 0.15,
            "within_descriptive_margin": domain_pass,
        }
    return {
        "status": "pass" if ready and passed else "fail" if ready else "pending",
        "ready": ready,
        "evidential_role": (
            "execution-plumbing sensitivity; cannot replace the causal "
            "E68-versus-E66 gate"
        ),
        "details": details,
    }


def _math_gate(domains: dict[str, Any], terminal_runs: int) -> dict[str, Any]:
    deltas: dict[str, float] = {}
    ready = terminal_runs == 6
    for metric in ("greedy", "mean8"):
        treatment = _terminal_metric(domains, MATH_DOMAIN, TREATMENT, metric)
        control = _terminal_metric(domains, MATH_DOMAIN, CONTROL, metric)
        if treatment is None or control is None:
            ready = False
            continue
        deltas[metric] = float(treatment["mean"]) - float(control["mean"])
    passed = (
        set(deltas) == {"greedy", "mean8"}
        and all(value >= -0.02 for value in deltas.values())
        and any(value > 0 for value in deltas.values())
    )
    return {
        "status": "pass" if ready and passed else "fail" if ready else "pending",
        "ready": ready,
        "terminal_mean_deltas": deltas,
        "noninferiority_margin": 0.02,
        "at_least_one_directionally_higher": (
            any(value > 0 for value in deltas.values()) if ready else None
        ),
    }


def _fmt(value: Any) -> str:
    return "—" if value is None else f"{float(value):.3f}"


def _render_markdown(payload: dict[str, Any]) -> str:
    domains = payload["domains"]
    mechanism = payload["mechanism_evidence"]
    lines = [
        "# Five-domain confirmation — live frozen-checkpoint results",
        "",
        (
            f"Generated `{payload['generated_at']}`. Campaign audit: "
            f"`{payload['campaign']['status']}`; terminal runs "
            f"`{payload['campaign']['terminal_runs']}/"
            f"{payload['campaign']['expected_runs']}`; integrity "
            f"violations `{payload['campaign']['violation_count']}`."
        ),
        "",
        "Only three-seed-complete frozen checkpoints are summarized. "
        "AUC is emitted only after every registered checkpoint lands; "
        "partial AUC is descriptive and cannot pass a gate.",
        "",
        "## Confirmation readiness",
        "",
        "| Requirement | Live evidence | Terminal requirement |",
        "|---|---:|---:|",
        (
            "| Independent registered runs | "
            f"{payload['campaign']['terminal_runs']}/"
            f"{payload['campaign']['expected_runs']} terminal | "
            f"{payload['campaign']['expected_runs']}/"
            f"{payload['campaign']['expected_runs']} |"
        ),
        (
            "| Fixed arm/domain checkpoint cells | "
            f"{payload['fixed_checkpoint_coverage']['landed_checkpoint_cells']}/"
            f"{payload['fixed_checkpoint_coverage']['expected_checkpoint_cells']} "
            "| complete surface |"
        ),
        (
            "| Required seed-metric values | "
            f"{payload['fixed_checkpoint_coverage']['landed_metric_seed_values']}/"
            f"{payload['fixed_checkpoint_coverage']['expected_metric_seed_values']} "
            "| complete surface |"
        ),
        (
            "| Evaluation cadence | "
            f"{payload['eval_cadence']['status']} "
            f"({payload['eval_cadence']['violation_count']} violations) | "
            "pass, ≤1 epoch gap |"
        ),
        (
            "| E66/E68 pre-intervention paired seeds | "
            f"{payload['preintervention_equivalence']['ready_pairs']}/12 "
            f"(`{payload['preintervention_equivalence']['status']}`) | "
            "12/12, pass |"
        ),
        (
            "| E68 durable support-separation checkpoints | "
            f"{payload['checkpoint_separation']['checkpointed_runs']}/12 "
            f"(`{payload['checkpoint_separation']['status']}`) | "
            "12/12, pass |"
        ),
        (
            "| Primary interpretation gates | "
            + ", ".join(
                f"{name}: `{gate['status']}`"
                for name, gate in payload["gates"].items()
            )
            + " | all evaluated only at the frozen terminal surface |"
        ),
        "",
        (
            "**Objective-correction disclosure:** E65R1 is excluded from all "
            "confirmatory gates because its runtime forced E58's novelty beta "
            "from `0.50` to `0.0` before the singleton actuator could fire. "
            "Those trajectories remain archived as engineering evidence. "
            "E67 is also excluded: its shared proposal/on-policy bank was "
            "rejected before optimization, with "
            f"`{payload['invalidated_e67']['optimizer_metric_files']}` "
            "optimizer metric files. E68 is the prospectively frozen "
            "separated-support treatment and must "
            "pass an explicit runtime same-objective audit against E66. "
            "Machine-readable invalidation audits: E65R1 "
            f"`{payload['invalidated_e65r1']['status']}`, E67 "
            f"`{payload['invalidated_e67']['status']}`."
        ),
        "",
        (
            "**Infrastructure disclosure:** after scheduler preemption, the "
            "nine pending non-Math E66 controls were moved from single-node "
            "`lowprio` placement to broader same-accelerator-family "
            "`pvl-lowprio` pools. Run IDs, checkpoints, frozen source, "
            "objective, seeds, and all scientific settings are unchanged; "
            "the E66 audit fails if any trace regresses below its recorded "
            "pre-amendment step. The same hash-bound, same-family resume "
            "rule was first applied to three pending E61-R1 jobs. After eight "
            "additional RTX 3090 jobs were preempted, a second hash-bound "
            "placement-only amendment moved those pending checkpoint resumes "
            "to the same accelerator-family `pvl-lowprio` pool; the E61-R1 "
            "audit verifies both amendment identities and rejects trace "
            "regression. After both registered "
            "A6000 nodes entered a health-check drain for overheated GPUs, "
            "all six paired E66/E68 Graph jobs were moved together to "
            "non-MLTheory `lowprio` nodes that advertise the same A6000 "
            "accelerator family. This second hash-bound amendment changes "
            "placement only and both prospective audits reject trace "
            "regression. When two zero-step E68 jobs were then assigned a "
            "node206 GPU already holding 48.3 GiB from foreign processes, a "
            "third paired amendment removed that pool and moved all six Graph "
            "jobs together to healthy `mltheory/pvl-lowprio` A6000 nodes. "
            "Only the exact pre-amendment log prefixes are classified as "
            "infrastructure interruptions; later OOMs remain hard failures."
        ),
        "",
        (
            "Pre-intervention E66/E68 trajectory equivalence audit: "
            f"`{payload['preintervention_equivalence']['status']}`; "
            f"ready paired seeds "
            f"`{payload['preintervention_equivalence']['ready_pairs']}/12`; "
            f"paired updates compared "
            f"`{payload['preintervention_equivalence']['compared_steps']}`; "
            f"violations "
            f"`{payload['preintervention_equivalence']['violation_count']}`."
        ),
        "",
        (
            "E68 checkpoint separation audit: "
            f"`{payload['checkpoint_separation']['status']}`; "
            f"latest durable checkpoints audited "
            f"`{payload['checkpoint_separation']['checkpointed_runs']}/12`; "
            f"proposal-only outcomes checked "
            f"`{payload['checkpoint_separation']['proposal_only_outcomes']}`; "
            f"graduated to neutral on-policy discoveries "
            f"`{payload['checkpoint_separation']['graduated_outcomes']}`; "
            f"objective-count overlaps "
            f"`{payload['checkpoint_separation']['objective_overlaps']}`; "
            f"violations "
            f"`{payload['checkpoint_separation']['violation_count']}`."
        ),
        "",
        (
            "Evaluation-cadence audit: "
            f"`{payload['eval_cadence']['status']}`; "
            f"materialized runs with optimizer progress audited "
            f"`{payload['eval_cadence']['audited_runs']}`; "
            "maximum permitted gap `1` prompt epoch; "
            f"violations `{payload['eval_cadence']['violation_count']}`."
        ),
        "",
        (
            "Fixed paper-checkpoint coverage audit: "
            f"`{payload['fixed_checkpoint_coverage']['status']}`; complete "
            "arm/domain checkpoint cells "
            f"`{payload['fixed_checkpoint_coverage']['landed_checkpoint_cells']}/"
            f"{payload['fixed_checkpoint_coverage']['expected_checkpoint_cells']}`; "
            "landed seed-metric values "
            f"`{payload['fixed_checkpoint_coverage']['landed_metric_seed_values']}/"
            f"{payload['fixed_checkpoint_coverage']['expected_metric_seed_values']}`; "
            "complete full comparison surfaces by domain: "
            + "; ".join(
                f"{domain} `{record['complete_checkpoint_count']}/"
                f"{record['maximum_checkpoints']}`"
                for domain, record in payload[
                    "fixed_checkpoint_coverage"
                ]["domains"].items()
            )
            + "."
        ),
        "",
        (
            "The complete seed-level numerical surface is refreshed at "
            "[fixed-checkpoint CSV]"
            "(e65_five_domain_confirmation_fixed_checkpoints_live.csv)."
        ),
        "",
        "## Latest shared E58 versus Dr.GRPO checkpoint",
        "",
        "These are interim three-seed means at the latest fixed checkpoint "
        "landed by both arms in each domain; they are not terminal claims.",
        "",
        "| Domain | pass | Dr.GRPO pass@8 | E58 pass@8 | Δ | Dr.GRPO distinct@8 | E58 distinct@8 | Δ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for domain, comparison in payload["interim_shared_e58"].items():
        if comparison is None:
            continue
        pass8 = comparison["metrics"]["pass8"]
        distinct8 = comparison["metrics"]["distinct8"]
        lines.append(
            "| "
            + " | ".join(
                (
                    domain,
                    str(comparison["pass"]),
                    _fmt(pass8[CONTROL]["mean"]),
                    _fmt(pass8[TREATMENT]["mean"]),
                    _fmt(pass8["mean_delta"]),
                    _fmt(distinct8[CONTROL]["mean"]),
                    _fmt(distinct8[TREATMENT]["mean"]),
                    _fmt(distinct8["mean_delta"]),
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Latest shared separated-support E68 versus E66 checkpoint",
            "",
            "This is the direct causal comparison at the latest fixed "
            "three-seed checkpoint shared by both prospective arms. It is "
            "interim only; the frozen repair gate uses pass 12.",
            "",
            "| Domain | pass | E66 pass@8 | E68 pass@8 | Δ | E66 distinct@8 | E68 distinct@8 | Δ |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for domain, comparison in payload["interim_shared_repair"].items():
        if comparison is None:
            continue
        pass8 = comparison["metrics"]["pass8"]
        distinct8 = comparison["metrics"]["distinct8"]
        lines.append(
            "| "
            + " | ".join(
                (
                    domain,
                    str(comparison["pass"]),
                    _fmt(pass8[PLUMBING_CONTROL]["mean"]),
                    _fmt(pass8[REPAIR]["mean"]),
                    _fmt(pass8["mean_delta"]),
                    _fmt(distinct8[PLUMBING_CONTROL]["mean"]),
                    _fmt(distinct8[REPAIR]["mean"]),
                    _fmt(distinct8["mean_delta"]),
                )
            )
            + " |"
        )
    uncertainty = payload.get("paired_prompt_uncertainty", {})
    uncertainty_summary = uncertainty.get("summary", {})
    lines.extend(
        [
            "",
            "## Secondary paired prompt-level uncertainty",
            "",
            (
                "This post-specified analysis is descriptive and cannot "
                "change any primary gate. It resamples the three paired "
                "training seeds and the shared evaluation prompts as crossed "
                "units after averaging the four fixed K=8 draws. Complete "
                "domain-checkpoints: "
                f"`{uncertainty_summary.get('complete_domain_checkpoints', 0)}`/"
                f"{uncertainty_summary.get('expected_domain_checkpoints', 40)}; "
                "integrity violations: "
                f"`{uncertainty_summary.get('violation_count', 0)}`. The "
                "[source-aligned method]"
                "(../preregistration/"
                "e68_paired_prompt_uncertainty_secondary_v2_20260727.md) "
                "was frozen before integration into this report; it computes "
                "no p-values."
            ),
            "",
            "| Domain | pass | metric | E68 − E66 | descriptive crossed-bootstrap 95% | seed 43 / 44 / 45 deltas |",
            "|---|---:|---|---:|---:|---:|",
        ]
    )
    for domain in uncertainty.get("domains", {}).values():
        latest = domain.get("latest_complete")
        if not latest:
            continue
        for metric in ("greedy", "mean8", "pass8", "distinct8"):
            row = latest.get("metrics", {}).get(metric)
            if not row:
                continue
            interval = row["descriptive_crossed_bootstrap_95"]
            seed_deltas = row["paired_seed_deltas"]
            lines.append(
                "| "
                + " | ".join(
                    (
                        str(domain.get("label", "unknown")),
                        str(latest["training_pass"]),
                        metric,
                        _fmt(row["e68_minus_e66"]),
                        f"[{_fmt(interval[0])}, {_fmt(interval[1])}]",
                        " / ".join(
                            _fmt(seed_deltas.get(str(seed)))
                            for seed in SEEDS
                        ),
                    )
                )
                + " |"
            )
    repeatability_rows = []
    for domain in uncertainty.get("domains", {}).values():
        latest = domain.get("latest_complete")
        if not latest:
            continue
        for seed_rows in latest.get(
            "primary_vs_repeated_greedy_sensitivity",
            {},
        ).values():
            repeatability_rows.extend(seed_rows.values())
    if repeatability_rows:
        differing = sum(
            int(row.get("different_prompt_scores", 0))
            for row in repeatability_rows
        )
        comparisons = sum(
            int(
                uncertainty["domains"][domain_key]["latest_complete"][
                    "prompts_per_seed"
                ]
            )
            * len(SEEDS)
            * 2
            for domain_key in uncertainty.get("domains", {})
            if uncertainty["domains"][domain_key].get("latest_complete")
        )
        shifts = [
            float(row.get("repeated_minus_primary", 0.0))
            for row in repeatability_rows
        ]
        lines.extend(
            [
                "",
                (
                    "Greedy repeatability sensitivity at the latest displayed "
                    "checkpoints: the separate repeated temperature-zero call "
                    f"changed `{differing}/{comparisons}` prompt scores; "
                    "per-run mean shifts ranged from "
                    f"`{min(shifts):+.3f}` to `{max(shifts):+.3f}`. The table "
                    "is centered on the primary greedy result used by the "
                    "paper figure."
                ),
                "",
                "These intervals include finite-prompt uncertainty but do not "
                "turn three training seeds into more than three independent "
                "runs. Intermediate intervals remain interim.",
            ]
        )
    lines.extend(
        [
            "",
            "## Mechanism diagnosis and repair",
            "",
            "These are live mechanism diagnostics, not selected performance "
            "endpoints. They make the repair hypothesis falsifiable: the "
            "controller must first detect an entropy deficit, and E68 must "
            "then demonstrate a clean, support-only intervention before its "
            "terminal gate can pass.",
            "",
            "| Python E58 seed | step | entropy EMA / own reference | ratio | unprojected coefficient | projection active | mean bank support | outcomes |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in mechanism["e58_python_live_controller_diagnosis"]:
        entropy_pair = (
            f"{_fmt(row['entropy_ema'])} / "
            f"{_fmt(row['entropy_reference'])}"
        )
        lines.append(
            "| "
            + " | ".join(
                (
                    str(row["seed"]),
                    str(row["step"]) if row["step"] is not None else "—",
                    entropy_pair,
                    _fmt(row["entropy_ratio"]),
                    _fmt(row["coefficient"]),
                    (
                        str(row["projection_active"])
                        if row["projection_active"] is not None
                        else "—"
                    ),
                    _fmt(row["mean_discovered_support"]),
                    (
                        str(row["cumulative_discovered_outcomes"])
                        if row["cumulative_discovered_outcomes"] is not None
                        else "—"
                    ),
                )
            )
            + " |"
        )
    pilot = mechanism["prior_support_only_actuator_pilot"]
    if pilot is not None:
        lines.extend(
            [
                "",
                (
                    "The prior E62R10 **one-seed engineering pilot only** "
                    "showed actuator feasibility: maximum mean support "
                    f"{_fmt(pilot['maximum_discovered_mean_support'])}, "
                    f"{pilot['admitted_new_outcomes']} admitted outcomes, "
                    f"{_fmt(pilot['admissions_per_100_updates'])} admissions "
                    "per 100 updates, "
                    f"{pilot['proposal_rows_sent_to_ppo']} proposal rows sent "
                    "to PPO, and terminal pass@8 "
                    f"{_fmt(pilot['terminal_pass8'])} after one pass. Its "
                    "breadth is precisely why it is not confirmatory evidence."
                ),
            ]
        )
    repair_live = mechanism["corrected_repair_live"]
    lines.extend(
        [
            "",
            (
                "Separated-support E68 currently has "
                f"`{repair_live['entropy_gated_interventions']}` audited "
                f"entropy-gated interventions over "
                f"`{repair_live['metric_records']}` metric-bearing updates "
                f"({_fmt(repair_live['interventions_per_100_updates'])} per 100 "
                "updates); "
                "maximum admitted in any proposal group "
                f"is `{repair_live['maximum_admitted_per_group']}` and the "
                f"mechanism audit is `{repair_live['audit_status']}`. Zero is "
                "expected before a registered collapse condition occurs; the "
                "frozen terminal gate cannot pass without at least one clean "
                "intervention."
            ),
            "",
            "| E68 domain | metric-bearing updates | interventions | interventions / 100 updates |",
            "|---|---:|---:|---:|",
        ]
    )
    for domain, row in repair_live["by_domain"].items():
        lines.append(
            "| "
            + " | ".join(
                (
                    MECHANISM_DOMAIN_LABELS.get(domain, domain),
                    str(row["metric_records"]),
                    str(row["entropy_gated_interventions"]),
                    _fmt(row["interventions_per_100_updates"]),
                )
            )
            + " |"
        )
    lines.extend(
        [
        "",
        "## ModeBench pass@8",
        "",
        "| Domain | Arm | checkpoints | latest pass | latest mean [range] | terminal | AUC/12 |",
        "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    labels = {
        CONTROL: "Dr.GRPO",
        TREATMENT: "E58",
        PLUMBING_CONTROL: "E66 same-plumbing control",
        REPAIR: "E68 separated-support actuator",
    }
    for domain, *_ in MODEBENCH:
        for arm in (CONTROL, TREATMENT, PLUMBING_CONTROL, REPAIR):
            metric = domains[domain]["arms"][arm]["metrics"]["pass8"]
            latest = metric["latest_complete"]
            latest_text = (
                "—"
                if latest is None
                else (
                    f"{_fmt(latest['mean'])} "
                    f"[{_fmt(latest['min'])}, {_fmt(latest['max'])}]"
                )
            )
            lines.append(
                "| "
                + " | ".join(
                    (
                        domain,
                        labels[arm],
                        (
                            f"{metric['checkpoint_count']}/"
                            f"{metric['expected_checkpoint_count']}"
                        ),
                        (
                            "—"
                            if latest is None
                            else str(latest["pass"])
                        ),
                        latest_text,
                        _fmt(
                            metric["terminal"]["mean"]
                            if metric["terminal"]
                            else None
                        ),
                        _fmt(metric["auc_normalized"]),
                    )
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "## Held-out MATH-500",
            "",
            "The named paper figure, table below, and primary realism gate "
            "remain frozen to the seven even-pass anchors "
            "`0, 2, 4, 6, 8, 10, 12`. The separate "
            "[all-epoch diagnostic figure]"
            "(../figures/"
            "e61r1_e58_vs_grpo_05b_12ep_all_epoch_diagnostic_live.pdf) "
            "displays every complete integer epoch from the already-retained "
            "quarter-epoch evaluation traces.",
            "",
            "| Arm | metric | checkpoints | latest pass | latest mean [range] | terminal | AUC/12 |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for arm in (CONTROL, TREATMENT):
        for metric_name in ("greedy", "mean8", "pass8"):
            metric = domains[MATH_DOMAIN]["arms"][arm]["metrics"][metric_name]
            latest = metric["latest_complete"]
            latest_text = (
                "—"
                if latest is None
                else (
                    f"{_fmt(latest['mean'])} "
                    f"[{_fmt(latest['min'])}, {_fmt(latest['max'])}]"
                )
            )
            lines.append(
                "| "
                + " | ".join(
                    (
                        labels[arm],
                        metric_name,
                        (
                            f"{metric['checkpoint_count']}/"
                            f"{metric['expected_checkpoint_count']}"
                        ),
                        "—" if latest is None else str(latest["pass"]),
                        latest_text,
                        _fmt(
                            metric["terminal"]["mean"]
                            if metric["terminal"]
                            else None
                        ),
                        _fmt(metric["auc_normalized"]),
                    )
                )
                + " |"
            )
    verifier = payload["math500_verifier_sensitivity"]
    step_zero = verifier.get("step_zero_reproducible", {})
    step_zero_clean = bool(step_zero) and all(
        row.get("reproducible") for row in step_zero.values()
    )
    lines.extend(
        [
            "",
            (
                "Supplemental raw-trace verifier sensitivity: "
                f"`{verifier['status']}`; "
                f"{verifier['verifier_timeout_diagnostics']} caught timeout "
                "diagnostics, "
                f"{verifier['identical_response_reward_conflicts']} "
                "identical-response reward conflicts, and step-0 six-run "
                "response/reward redundancy "
                f"`{'pass' if step_zero_clean else 'fail'}`. This audit does "
                "not alter primary scores and must pass at all seven fixed "
                "checkpoints for a terminal campaign audit."
            ),
            "",
            "## Frozen interpretation gates",
            "",
            "| Claim | status |",
            "|---|---|",
            (
                "| E58 cross-domain positive | "
                f"`{payload['gates']['e58_cross_domain']['status']}` |"
            ),
            (
                "| Separated-support E68 singleton repair | "
                f"`{payload['gates']['e68_repair']['status']}` |"
            ),
            (
                "| E66 versus historical E58 plumbing sensitivity | "
                f"`{payload['gates']['plumbing_consistency']['status']}` |"
            ),
            (
                "| Held-out MATH-500 realism | "
                f"`{payload['gates']['math500_realism']['status']}` |"
            ),
            "",
            "A pending gate is not evidence of success or failure. Full seed "
            "values, paired deltas, raw AUCs, and gate diagnostics are in the "
            "machine-readable JSON artifact.",
            "",
        ]
    )
    return "\n".join(lines)


def _render_checkpoint_csv(payload: dict[str, Any]) -> str:
    output = io.StringIO()
    fieldnames = (
        "domain",
        "track",
        "arm",
        "metric",
        "training_pass",
        "optimizer_step",
        "seed_43",
        "seed_44",
        "seed_45",
        "mean",
        "min",
        "max",
        "is_terminal",
        "is_registered_checkpoint",
    )
    writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    domains = payload["domains"]
    for domain, *_ in MODEBENCH:
        for arm in (CONTROL, TREATMENT, PLUMBING_CONTROL, REPAIR):
            for metric in QUALITY:
                for point in domains[domain]["arms"][arm]["metrics"][metric][
                    "points"
                ]:
                    writer.writerow(
                        {
                            "domain": domain,
                            "track": "modebench",
                            "arm": arm,
                            "metric": metric,
                            "training_pass": point["pass"],
                            "optimizer_step": point["step"],
                            "seed_43": point["seeds"]["43"],
                            "seed_44": point["seeds"]["44"],
                            "seed_45": point["seeds"]["45"],
                            "mean": point["mean"],
                            "min": point["min"],
                            "max": point["max"],
                            "is_terminal": int(point["pass"] == 12),
                            "is_registered_checkpoint": 1,
                        }
                    )
    for arm in (CONTROL, TREATMENT):
        for metric in ("greedy", "mean8", "pass8"):
            for point in domains[MATH_DOMAIN]["arms"][arm]["metrics"][metric][
                "points"
            ]:
                writer.writerow(
                    {
                        "domain": MATH_DOMAIN,
                        "track": "external_validity",
                        "arm": arm,
                        "metric": metric,
                        "training_pass": point["pass"],
                        "optimizer_step": point["step"],
                        "seed_43": point["seeds"]["43"],
                        "seed_44": point["seeds"]["44"],
                        "seed_45": point["seeds"]["45"],
                        "mean": point["mean"],
                        "min": point["min"],
                        "max": point["max"],
                        "is_terminal": int(point["pass"] == 12),
                        "is_registered_checkpoint": 1,
                    }
                )
    return output.getvalue()


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(content)
    os.replace(temporary, path)


def main() -> None:
    domains: dict[str, Any] = {}
    for domain, base_prefix, repair_prefix, steps_per_pass in MODEBENCH:
        points = _load_modebench_points(
            base_prefix,
            steps_per_pass=steps_per_pass,
            allowed_arms=(CONTROL, TREATMENT),
        )
        points.extend(
            _load_modebench_points(
                repair_prefix,
                steps_per_pass=steps_per_pass,
                allowed_arms=(REPAIR,),
            )
        )
        points.extend(
            _load_modebench_points(
                PLUMBING_PREFIX[domain],
                steps_per_pass=steps_per_pass,
                allowed_arms=(TREATMENT,),
                arm_alias=PLUMBING_CONTROL,
            )
        )
        domains[domain] = _domain_summary(
            points,
            arms=(CONTROL, TREATMENT, PLUMBING_CONTROL, REPAIR),
            fixed_passes=MODEBENCH_PASSES,
            steps_per_pass=steps_per_pass,
        )
    domains[MATH_DOMAIN] = _domain_summary(
        _load_math_points(),
        arms=(CONTROL, TREATMENT),
        fixed_passes=MATH_PASSES,
        steps_per_pass=384,
        metrics=("greedy", "mean8", "pass8"),
    )
    campaign = _load_json(AUDIT, {})
    e61 = _load_json(E61_AUDIT, {})
    e64 = _load_json(E64_AUDIT, {})
    e64_sensitivity = _load_json(E64_SENSITIVITY, {})
    repair = _load_json(E68_AUDIT, {})
    equivalence = _load_json(E68_EQUIVALENCE, {})
    checkpoint_separation = _load_json(E68_CHECKPOINT, {})
    paired_prompt_uncertainty = _load_json(
        E68_PAIRED_PROMPT_UNCERTAINTY,
        {},
    )
    invalidation = _load_json(E65_INVALIDATION, {})
    e67_invalidation = _load_json(E67_INVALIDATION, {})
    e66 = _load_json(E66_AUDIT, {})
    eval_cadence = _load_json(EVAL_CADENCE_AUDIT, {})
    checkpoint_coverage = _load_json(CHECKPOINT_COVERAGE_AUDIT, {})
    campaign_summary = campaign.get("summary", {})
    payload = {
        "schema": "five_domain_confirmation_results_v3",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "evidential_status": campaign.get("evidential_status", {}),
        "campaign": {
            "status": campaign.get("status", "not_available"),
            "terminal_runs": int(campaign_summary.get("terminal_runs", 0)),
            "expected_runs": int(campaign_summary.get("expected_runs", 54)),
            "violation_count": len(campaign.get("violations", [])),
        },
        "domains": domains,
        "mechanism_evidence": _mechanism_evidence(repair),
        "preintervention_equivalence": {
            "status": equivalence.get("status", "not_available"),
            "ready_pairs": int(
                equivalence.get("summary", {}).get("ready_pairs", 0)
            ),
            "materialized_pairs": int(
                equivalence.get("summary", {}).get(
                    "materialized_pairs",
                    0,
                )
            ),
            "violation_count": len(equivalence.get("violations", [])),
            "compared_steps": sum(
                int(pair.get("compared_steps", 0))
                for domain in equivalence.get("domains", {}).values()
                for pair in domain.get("pairs", [])
            ),
        },
        "checkpoint_separation": {
            "status": checkpoint_separation.get(
                "status", "not_available"
            ),
            "checkpointed_runs": int(
                checkpoint_separation.get("summary", {}).get(
                    "checkpointed_runs",
                    0,
                )
            ),
            "proposal_only_outcomes": sum(
                int(record.get("proposal_only_outcomes", 0))
                for record in checkpoint_separation.get("checkpoints", [])
            ),
            "graduated_outcomes": sum(
                int(record.get("proposal_graduated_to_on_policy", 0))
                for record in checkpoint_separation.get("checkpoints", [])
            ),
            "objective_overlaps": sum(
                int(record.get("proposal_objective_overlap", 0))
                for record in checkpoint_separation.get("checkpoints", [])
            ),
            "violation_count": len(
                checkpoint_separation.get("violations", [])
            ),
        },
        "eval_cadence": {
            "status": eval_cadence.get("status", "not_available"),
            "audited_runs": int(
                eval_cadence.get("summary", {}).get("audited_runs", 0)
            ),
            "materialized_runs": int(
                eval_cadence.get("summary", {}).get("materialized_runs", 0)
            ),
            "preoptimizer_runs": int(
                eval_cadence.get("summary", {}).get("preoptimizer_runs", 0)
            ),
            "violation_count": len(eval_cadence.get("violations", [])),
        },
        "fixed_checkpoint_coverage": {
            "status": checkpoint_coverage.get("status", "not_available"),
            "landed_checkpoint_cells": int(
                checkpoint_coverage.get("summary", {}).get(
                    "landed_checkpoint_cells",
                    0,
                )
            ),
            "expected_checkpoint_cells": int(
                checkpoint_coverage.get("summary", {}).get(
                    "expected_checkpoint_cells",
                    174,
                )
            ),
            "landed_metric_seed_values": int(
                checkpoint_coverage.get("summary", {}).get(
                    "landed_metric_seed_values",
                    0,
                )
            ),
            "expected_metric_seed_values": int(
                checkpoint_coverage.get("summary", {}).get(
                    "expected_metric_seed_values",
                    2046,
                )
            ),
            "domains": {
                domain: {
                    "complete_checkpoint_count": int(
                        record.get(
                            "complete_full_surface_checkpoint_count",
                            0,
                        )
                    ),
                    "maximum_checkpoints": int(
                        record.get("maximum_registered_checkpoints", 0)
                    ),
                    "complete_passes": record.get(
                        "complete_full_surface_passes",
                        [],
                    ),
                }
                for domain, record in checkpoint_coverage.get(
                    "domains",
                    {},
                ).items()
            },
            "violation_count": len(
                checkpoint_coverage.get("violations", [])
            ),
        },
        "paired_prompt_uncertainty": paired_prompt_uncertainty,
        "invalidated_e65r1": {
            "status": invalidation.get("status", "not_available"),
            "jobs_checked": int(invalidation.get("e65_jobs_checked", 0)),
            "expected_literal_e58_novelty_beta": invalidation.get(
                "expected_literal_e58_novelty_beta"
            ),
        },
        "invalidated_e67": {
            "status": e67_invalidation.get("status", "not_available"),
            "jobs_checked": int(
                e67_invalidation.get("expected_runs", 0)
            ),
            "optimizer_metric_files": len(
                e67_invalidation.get("optimizer_metric_files", [])
            ),
        },
        "math500_verifier_sensitivity": {
            "status": e64_sensitivity.get("status", "not_available"),
            "verifier_timeout_diagnostics": int(
                e64_sensitivity.get("summary", {}).get(
                    "verifier_timeout_diagnostics",
                    0,
                )
            ),
            "caught_traceback_diagnostics": int(
                e64_sensitivity.get("summary", {}).get(
                    "caught_traceback_diagnostics",
                    0,
                )
            ),
            "identical_response_reward_conflicts": int(
                e64_sensitivity.get("summary", {}).get(
                    "identical_response_reward_conflicts",
                    0,
                )
            ),
            "unique_response_tuples_checked": int(
                e64_sensitivity.get("summary", {}).get(
                    "unique_response_tuples_checked",
                    0,
                )
            ),
            "step_zero_reproducible": e64_sensitivity.get(
                "summary",
                {},
            ).get("step_zero_reproducible", {}),
        },
        "interim_shared_e58": {
            domain: _latest_shared_comparison(
                domains,
                domain,
                CONTROL,
                TREATMENT,
                ("pass8", "distinct8"),
            )
            for domain, *_ in MODEBENCH
        },
        "interim_shared_repair": {
            domain: _latest_shared_comparison(
                domains,
                domain,
                PLUMBING_CONTROL,
                REPAIR,
                ("pass8", "distinct8"),
            )
            for domain, *_ in MODEBENCH
        },
        "gates": {
            "e58_cross_domain": _e58_gate(
                domains,
                int(e61.get("summary", {}).get("terminal_runs", 0)),
            ),
            "e68_repair": _corrected_repair_gate(
                domains,
                repair_terminal_runs=int(
                    repair.get("summary", {}).get("terminal_runs", 0)
                ),
                comparator_terminal_runs=int(
                    e66.get("summary", {}).get("terminal_runs", 0)
                ),
                interventions=int(
                    repair.get("summary", {}).get(
                        "entropy_gated_interventions",
                        0,
                    )
                ),
                repair_audit_status=str(
                    repair.get("status", "not_available")
                ),
                comparator_audit_status=str(
                    e66.get("status", "not_available")
                ),
                equivalence_audit_status=str(
                    equivalence.get("status", "not_available")
                ),
                checkpoint_audit_status=str(
                    checkpoint_separation.get("status", "not_available")
                ),
            ),
            "plumbing_consistency": _plumbing_consistency_gate(
                domains,
                historical_terminal_runs=int(
                    e61.get("summary", {}).get("terminal_runs", 0)
                ),
                plumbing_terminal_runs=int(
                    e66.get("summary", {}).get("terminal_runs", 0)
                ),
            ),
            "math500_realism": _math_gate(
                domains,
                int(e64.get("summary", {}).get("terminal_runs", 0)),
            ),
        },
    }
    _atomic_text(OUT_JSON, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _atomic_text(OUT_MD, _render_markdown(payload))
    _atomic_text(OUT_CSV, _render_checkpoint_csv(payload))
    print(
        "[five-domain-results] "
        f"campaign={payload['campaign']['status']} "
        f"terminal={payload['campaign']['terminal_runs']}/"
        f"{payload['campaign']['expected_runs']} "
        "gates="
        + ",".join(
            f"{name}:{gate['status']}"
            for name, gate in payload["gates"].items()
        )
    )


if __name__ == "__main__":
    main()
