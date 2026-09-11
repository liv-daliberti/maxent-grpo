#!/usr/bin/env python3
"""Audit E66/E67 trajectory equivalence before the singleton actuator fires."""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "var/artifacts/e67_preintervention_equivalence_audit_latest.json"
ARM_CONTROL = "verified_first_global_replay_canonical"
ARM_REPAIR = "verified_entropy_gated_singleton_escape_canonical"
SEEDS = (43, 44, 45)
MANIFESTS = {
    "graph_coloring": (
        ROOT
        / "var/artifacts/gce66_same_plumbing_control_05b_12ep_comparative_jobs.tsv",
        ROOT
        / "var/artifacts/gce67_corrected_same_objective_actuator_05b_12ep_comparative_jobs.tsv",
    ),
    "countdown": (
        ROOT
        / "var/artifacts/cde66_same_plumbing_control_05b_12ep_comparative_jobs.tsv",
        ROOT
        / "var/artifacts/cde67_corrected_same_objective_actuator_05b_12ep_comparative_jobs.tsv",
    ),
    "python_factor": (
        ROOT
        / "var/artifacts/pye66_same_plumbing_control_05b_12ep_comparative_jobs.tsv",
        ROOT
        / "var/artifacts/pye67_corrected_same_objective_actuator_05b_12ep_comparative_jobs.tsv",
    ),
    "mathir": (
        ROOT
        / "var/artifacts/mie66_same_plumbing_control_05b_12ep_comparative_jobs.tsv",
        ROOT
        / "var/artifacts/mie67_corrected_same_objective_actuator_05b_12ep_comparative_jobs.tsv",
    ),
}
EXACT_KEYS = (
    "actor/sampling_request_seed",
    "actor/rewards",
    "train/online_canonical_new_outcome_count",
    "train/online_canonical_new_outcome_row_fraction",
    "train/online_canonical_tracked_outcomes",
    "train/online_canonical_tracked_prompts",
    "train/canonical_replay_available_groups",
    "train/canonical_replay_available_modes",
    "train/canonical_replay_eligible_groups",
    "train/canonical_replay_actuator_groups",
    "train/canonical_replay_actuator_modes",
    "train/canonical_replay_observations",
    "train/canonical_replay_mass_observations",
    "train/canonical_replay_projection_active",
    "train/canonical_replay_mass_projection_active",
    "train/semantic_shannon_success_conditioned_signed_open_set_projection_active",
    "train/online_canonical_advantage_applied_after_task_centering",
)
FLOAT_KEYS = (
    "train/adv_mean",
    "train/adv_min",
    "train/adv_max",
    "train/entropy",
    "train/policy_grad_norm",
    "train/online_canonical_novelty_advantage_mean",
    "train/online_canonical_novelty_advantage_rms",
    "train/online_canonical_combined_advantage_mean",
    "train/online_canonical_combined_advantage_rms",
    "train/online_canonical_entropy_estimate_mean",
    "train/online_canonical_log_support_mean",
    "train/canonical_replay_actuator_loss",
    "train/canonical_replay_balance_loss",
    "train/canonical_replay_applied_score_gradient_l2",
    "train/canonical_replay_applied_score_gradient_sum",
    "train/canonical_replay_alpha_used",
    "train/canonical_replay_mass_alpha_used",
    "train/canonical_replay_next_alpha",
    "train/canonical_replay_mass_next_alpha",
    "train/canonical_replay_entropy_ema",
    "train/canonical_replay_mass_surprisal_ema",
    "train/semantic_shannon_success_conditioned_signed_open_set_entropy_ema",
    "train/semantic_shannon_success_conditioned_signed_open_set_reference_entropy",
    "train/semantic_shannon_success_conditioned_signed_open_set_coefficient_used",
)
REQUIRED_KEYS = {
    "actor/sampling_request_seed",
    "actor/rewards",
    "train/adv_mean",
    "train/adv_min",
    "train/adv_max",
    "train/online_canonical_new_outcome_count",
    "train/online_canonical_tracked_outcomes",
    "train/online_canonical_novelty_advantage_mean",
    "train/online_canonical_novelty_advantage_rms",
    "train/online_canonical_combined_advantage_mean",
    "train/online_canonical_combined_advantage_rms",
    "train/online_canonical_advantage_applied_after_task_centering",
}


def _finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _run_dir(run_stamp: str, job_id: int) -> Path | None:
    candidates = sorted(
        (ROOT / "var/data").glob(f"*_{run_stamp}/debug_job{job_id}")
    )
    return candidates[0] if len(candidates) == 1 else None


def _manifest_rows(path: Path, expected_arm: str) -> dict[int, dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    result = {
        int(row["seed"]): {
            "arm": row["arm"],
            "seed": int(row["seed"]),
            "job_id": int(row["job_id"]),
            "run_stamp": row["run_stamp"],
        }
        for row in rows
    }
    if set(result) != set(SEEDS):
        raise RuntimeError(f"{path}: expected seeds {SEEDS}")
    if any(row["arm"] != expected_arm for row in result.values()):
        raise RuntimeError(f"{path}: arm mismatch")
    return result


def _training_rows(path: Path) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    if not path.is_file():
        return rows
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                continue
            step = row.get("trainer/global_step")
            if (
                _finite(step)
                and "train/online_canonical_new_outcome_count" in row
            ):
                rows[int(step)] = row
    return rows


def _interventions(rows: dict[int, dict[str, Any]]) -> list[int]:
    result = []
    for step, row in rows.items():
        value = row.get(
            "actor/counterfactual_proposal_admitted_new_outcomes",
            row.get(
                "train/counterfactual_proposal_admitted_new_outcomes",
                0,
            ),
        )
        if _finite(value) and float(value) > 0:
            result.append(step)
    return sorted(result)


def _compare_rows(
    control: dict[str, Any],
    repair: dict[str, Any],
    *,
    label: str,
) -> list[str]:
    violations: list[str] = []
    for key in EXACT_KEYS:
        left = control.get(key)
        right = repair.get(key)
        if left is None and right is None:
            if key in REQUIRED_KEYS:
                violations.append(f"{label}: missing required metric {key}")
            continue
        if not (_finite(left) and _finite(right)):
            violations.append(f"{label}: missing/nonfinite exact metric {key}")
        elif not math.isclose(
            float(left),
            float(right),
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            violations.append(
                f"{label}: exact mismatch {key}: {left!r} != {right!r}"
            )
    for key in FLOAT_KEYS:
        left = control.get(key)
        right = repair.get(key)
        if left is None and right is None:
            if key in REQUIRED_KEYS:
                violations.append(f"{label}: missing required metric {key}")
            continue
        if not (_finite(left) and _finite(right)):
            violations.append(f"{label}: missing/nonfinite float metric {key}")
        elif not math.isclose(
            float(left),
            float(right),
            rel_tol=2e-5,
            abs_tol=2e-6,
        ):
            violations.append(
                f"{label}: float mismatch {key}: {left!r} != {right!r}"
            )
    return violations


def main() -> None:
    violations: list[str] = []
    domains: dict[str, Any] = {}
    ready_pairs = 0
    materialized_pairs = 0
    for domain, (control_manifest, repair_manifest) in MANIFESTS.items():
        control_jobs = _manifest_rows(control_manifest, ARM_CONTROL)
        repair_jobs = _manifest_rows(repair_manifest, ARM_REPAIR)
        pairs = []
        for seed in SEEDS:
            control_job = control_jobs[seed]
            repair_job = repair_jobs[seed]
            control_dir = _run_dir(
                control_job["run_stamp"],
                control_job["job_id"],
            )
            repair_dir = _run_dir(
                repair_job["run_stamp"],
                repair_job["job_id"],
            )
            materialized = control_dir is not None and repair_dir is not None
            materialized_pairs += int(materialized)
            control_rows = (
                _training_rows(control_dir / "train_metrics.jsonl")
                if control_dir is not None
                else {}
            )
            repair_rows = (
                _training_rows(repair_dir / "train_metrics.jsonl")
                if repair_dir is not None
                else {}
            )
            first_intervention = next(
                iter(_interventions(repair_rows)),
                None,
            )
            maximum_comparable = min(
                max(control_rows, default=0),
                max(repair_rows, default=0),
            )
            comparison_end = min(
                maximum_comparable,
                (
                    first_intervention - 1
                    if first_intervention is not None
                    else 64
                ),
            )
            expected_steps = set(range(1, comparison_end + 1))
            shared_steps = (
                set(control_rows)
                & set(repair_rows)
                & expected_steps
            )
            pair_violations: list[str] = []
            for step in sorted(shared_steps):
                pair_violations.extend(
                    _compare_rows(
                        control_rows[step],
                        repair_rows[step],
                        label=f"{domain}/s{seed}/step{step}",
                    )
                )
            missing_steps = sorted(expected_steps - shared_steps)
            if missing_steps and maximum_comparable >= comparison_end > 0:
                pair_violations.append(
                    f"{domain}/s{seed}: missing comparable steps "
                    f"{missing_steps[:8]}"
                )
            minimum_required = (
                min(64, first_intervention - 1)
                if first_intervention is not None
                else 64
            )
            ready = (
                materialized
                and comparison_end >= minimum_required
                and len(shared_steps) >= minimum_required
            )
            ready_pairs += int(ready)
            violations.extend(pair_violations)
            pairs.append(
                {
                    "seed": seed,
                    "control_job_id": control_job["job_id"],
                    "repair_job_id": repair_job["job_id"],
                    "materialized": materialized,
                    "first_repair_intervention_step": first_intervention,
                    "comparison_end_step": comparison_end,
                    "compared_steps": len(shared_steps),
                    "minimum_required_steps": minimum_required,
                    "ready": ready,
                    "violation_count": len(pair_violations),
                }
            )
        domains[domain] = {"pairs": pairs}
    status = (
        "fail"
        if violations
        else "pass"
        if ready_pairs == 12
        else "in_progress"
    )
    payload = {
        "schema": "e67_preintervention_equivalence_audit_v1",
        "status": status,
        "summary": {
            "expected_pairs": 12,
            "materialized_pairs": materialized_pairs,
            "ready_pairs": ready_pairs,
            "compared_metric_count": len(EXACT_KEYS) + len(FLOAT_KEYS),
        },
        "domains": domains,
        "violations": sorted(set(violations)),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{OUT.name}.", dir=OUT.parent)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, OUT)
    print(
        f"[e67-equivalence] status={status} "
        f"pairs={materialized_pairs}/12 ready={ready_pairs}/12 "
        f"violations={len(payload['violations'])}"
    )


if __name__ == "__main__":
    main()
