#!/usr/bin/env python3
"""Audit E102 discovery, replay priority, and retention-safe bank balance."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "var/artifacts/e102_full_open_bank_maxent_replay_05b_jobs.json"
OUT = ROOT / "var/artifacts/e102_full_open_bank_maxent_replay_05b_audit_latest.json"
SMOKE_OUT = ROOT / "var/artifacts/e102_full_open_bank_maxent_replay_05b_smoke_gate.json"
TARGET_STEPS = 3072


def finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def metric(row: dict[str, Any], suffix: str) -> float | None:
    for prefix in ("train/", "actor/"):
        value = row.get(prefix + suffix)
        if finite(value):
            return float(value)
    return None


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as sink:
            json.dump(payload, sink, indent=2, sort_keys=True)
            sink.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def metric_paths(run_dir: Path) -> list[Path]:
    return sorted(run_dir.glob("debug_job*/train_metrics.jsonl"))


def rows(paths: Iterable[Path]) -> Iterable[tuple[Path, int, dict[str, Any]]]:
    for path in paths:
        for line_number, raw in enumerate(
            path.read_text(encoding="utf-8", errors="replace").splitlines(), 1
        ):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                yield path, line_number, {"__invalid_json__": True}
                continue
            yield path, line_number, row


def parse_metrics(paths: Iterable[Path]) -> tuple[dict[str, Any], list[str]]:
    violations: list[str] = []
    last_step = -1
    train_records = 0
    proposal_groups = 0.0
    proposal_rows = 0.0
    cumulative_admissions = 0.0
    priority_groups_cumulative = 0.0
    priority_modes_cumulative = 0.0
    priority_modes_updates = 0
    replay_updates = 0
    balance_updates = 0
    retention_safe_min = 1.0
    retention_safe_seen = False
    balance_scale_min = 1.0
    balance_scale_mean_min = 1.0
    requested_positive_max = 0.0
    applied_positive_max = 0.0
    mass_weight_max = 1.0
    leakage_max = 0.0
    objective_delta_max = 0.0
    forbidden_feedback_max = 0.0
    transform_enabled_max = 0.0
    exact_transform_enabled_max = 0.0
    separate_support_min = 1.0
    separate_support_seen = False
    proposal_charged_response_token_budget = 0.0
    replay_charged_response_token_budget = 0.0
    replay_realized_prompt_tokens = 0.0
    replay_realized_response_tokens = 0.0

    for path, line_number, row in rows(paths):
        if row.get("__invalid_json__"):
            violations.append(f"invalid JSON at {path}:{line_number}")
            continue
        for key, value in row.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if not math.isfinite(float(value)):
                    violations.append(f"non-finite {key} at {path}:{line_number}")
        raw_step = row.get(
            "misc/global_step",
            row.get("trainer/global_step", row.get("trainer/step", -1)),
        )
        if finite(raw_step):
            last_step = max(last_step, int(raw_step))
        if not any(key.startswith(("train/", "actor/")) for key in row):
            continue
        train_records += int(any(key.startswith("train/") for key in row))
        groups = metric(row, "canonical_replay_actuator_groups") or 0.0
        eligible = metric(row, "canonical_replay_eligible_groups") or 0.0
        replay_updates += int(groups > 0)
        balance_updates += int(eligible > 0)

        safe = metric(row, "canonical_replay_retention_safe_balance")
        if safe is not None:
            retention_safe_seen = True
            retention_safe_min = min(retention_safe_min, safe)
        scale = metric(row, "canonical_replay_balance_scale_min")
        if scale is not None:
            balance_scale_min = min(balance_scale_min, scale)
        scale_mean = metric(row, "canonical_replay_balance_scale_mean")
        if scale_mean is not None:
            balance_scale_mean_min = min(balance_scale_mean_min, scale_mean)
        requested_positive_max = max(
            requested_positive_max,
            metric(row, "canonical_replay_requested_positive_gradient_max") or 0.0,
        )
        applied_positive_max = max(
            applied_positive_max,
            metric(row, "canonical_replay_applied_positive_gradient_max") or 0.0,
        )
        mass_weight_max = max(
            mass_weight_max,
            metric(row, "canonical_replay_mass_weight_max") or 1.0,
        )
        priority_modes = metric(row, "canonical_replay_priority_modes") or 0.0
        priority_modes_updates += int(priority_modes > 0)
        priority_groups_cumulative = max(
            priority_groups_cumulative,
            metric(row, "canonical_replay_priority_replay_groups_cumulative") or 0.0,
        )
        priority_modes_cumulative = max(
            priority_modes_cumulative,
            metric(row, "canonical_replay_priority_replay_modes_cumulative") or 0.0,
        )

        proposal_groups += metric(row, "counterfactual_proposal_groups_generated") or 0.0
        proposal_rows_this_update = (
            metric(row, "counterfactual_proposal_rows_generated") or 0.0
        )
        proposal_rows += proposal_rows_this_update
        proposal_charged_response_token_budget += proposal_rows_this_update * (
            metric(row, "sampling_max_tokens") or 0.0
        )
        replay_charged_response_token_budget += (
            metric(row, "canonical_replay_charged_response_token_budget") or 0.0
        )
        replay_realized_prompt_tokens += (
            metric(row, "canonical_replay_realized_prompt_tokens") or 0.0
        )
        replay_realized_response_tokens += (
            metric(row, "canonical_replay_realized_response_tokens") or 0.0
        )
        cumulative_admissions = max(
            cumulative_admissions,
            metric(row, "counterfactual_proposal_cumulative_new_outcomes") or 0.0,
        )
        leakage_max = max(
            leakage_max,
            abs(metric(row, "counterfactual_proposal_conditioned_rows_sent_to_ppo") or 0.0),
            abs(metric(row, "counterfactual_proposal_transform_rows_sent_to_ppo") or 0.0),
        )
        objective_delta_max = max(
            objective_delta_max,
            abs(metric(row, "counterfactual_proposal_objective_outcome_delta") or 0.0),
        )
        forbidden_feedback_max = max(
            forbidden_feedback_max,
            *[
                abs(metric(row, suffix) or 0.0)
                for suffix in (
                    "counterfactual_proposal_gold_support_feedback",
                    "counterfactual_proposal_desired_mode_count_feedback",
                    "counterfactual_proposal_eval_feedback",
                    "counterfactual_proposal_transform_gold_support_feedback",
                    "counterfactual_proposal_transform_desired_mode_count_feedback",
                    "counterfactual_proposal_transform_eval_feedback",
                    "canonical_replay_gold_support_feedback",
                )
            ],
        )
        transform_enabled_max = max(
            transform_enabled_max,
            metric(row, "counterfactual_proposal_transform_enabled") or 0.0,
        )
        exact_transform_enabled_max = max(
            exact_transform_enabled_max,
            metric(row, "counterfactual_proposal_exact_grammar_transform_enabled") or 0.0,
        )
        separate = metric(row, "counterfactual_proposal_objective_support_separated")
        if separate is not None:
            separate_support_seen = True
            separate_support_min = min(separate_support_min, separate)

    if leakage_max > 0:
        violations.append("proposal rows reached PPO")
    if objective_delta_max > 0:
        violations.append("proposal changed neutral objective support")
    if forbidden_feedback_max > 0:
        violations.append("forbidden gold/support/evaluation feedback reached training")
    if transform_enabled_max > 0 or exact_transform_enabled_max > 0:
        violations.append("a proposal transform was enabled")
    if applied_positive_max > 1e-7:
        violations.append("retention-safe replay directly lowered a verified score")
    if retention_safe_seen and retention_safe_min < 1.0:
        violations.append("retention-safe balance was disabled on a replay update")
    if separate_support_seen and separate_support_min < 1.0:
        violations.append("proposal objective support was not separated")

    return {
        "materialized": train_records > 0,
        "last_step": last_step,
        "train_records": train_records,
        "proposal_groups_generated": proposal_groups,
        "proposal_rows_generated": proposal_rows,
        "proposal_charged_response_token_budget": (
            proposal_charged_response_token_budget
        ),
        "proposal_realized_prompt_tokens": None,
        "proposal_realized_response_tokens": None,
        "proposal_realized_token_telemetry": (
            "not retained by the E102 actor path; charged response-token "
            "budget is derived per update as generated rows times the frozen "
            "sampling ceiling"
        ),
        "proposal_cumulative_admissions": cumulative_admissions,
        "replay_actuation_updates": replay_updates,
        "balance_eligible_updates": balance_updates,
        "retention_safe_seen": retention_safe_seen,
        "balance_scale_min": balance_scale_min,
        "balance_scale_mean_min": balance_scale_mean_min,
        "requested_positive_gradient_max": requested_positive_max,
        "applied_positive_gradient_max": applied_positive_max,
        "priority_modes_updates": priority_modes_updates,
        "priority_replay_groups_cumulative": priority_groups_cumulative,
        "priority_replay_modes_cumulative": priority_modes_cumulative,
        "mass_weight_max": mass_weight_max,
        "replay_charged_response_token_budget": replay_charged_response_token_budget,
        "replay_realized_prompt_tokens": replay_realized_prompt_tokens,
        "replay_realized_response_tokens": replay_realized_response_tokens,
        "proposal_rows_to_ppo_max_abs": leakage_max,
        "proposal_objective_outcome_delta_max_abs": objective_delta_max,
        "forbidden_feedback_max_abs": forbidden_feedback_max,
        "transform_enabled_max": transform_enabled_max,
        "exact_grammar_transform_enabled_max": exact_transform_enabled_max,
    }, violations


def smoke_gate(run_dir: Path, target: int) -> tuple[dict[str, Any], list[str]]:
    report, violations = parse_metrics(metric_paths(run_dir))
    if report["last_step"] < target:
        violations.append(f"smoke has only {report['last_step']}/{target} optimizer steps")
    if report["replay_actuation_updates"] <= 0:
        violations.append("smoke never actuated replay")
    if not report["retention_safe_seen"]:
        violations.append("smoke lacks retention-safe balance telemetry")
    if report["proposal_cumulative_admissions"] <= 0:
        violations.append("smoke discovered no new verified mode")
    if report["priority_replay_groups_cumulative"] <= 0:
        violations.append("smoke never replay-prioritized an admitted mode")
    if report["mass_weight_max"] <= 1.0:
        violations.append("smoke never applied a nonuniform priority mass weight")
    return report, violations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-run-dir", type=Path)
    parser.add_argument("--smoke-target", type=int, default=32)
    parser.add_argument("--smoke-job-id", type=int)
    args = parser.parse_args()

    if args.smoke_run_dir is not None:
        report, violations = smoke_gate(args.smoke_run_dir, args.smoke_target)
        payload = {
            "schema": "e102-full-open-bank-smoke-audit-v1",
            "run_dir": str(args.smoke_run_dir),
            "job_id": args.smoke_job_id,
            "report": report,
            "violations": violations,
            "passed": not violations,
        }
        atomic_json(SMOKE_OUT, payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 1 if violations else 0

    if not LEDGER.is_file():
        raise SystemExit(f"E102 ledger is absent: {LEDGER}")
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    run_reports: list[dict[str, Any]] = []
    all_violations: list[str] = []
    domain_admissions: dict[str, float] = defaultdict(float)
    domain_priority: dict[str, float] = defaultdict(float)
    compute_fields = (
        "proposal_rows_generated",
        "proposal_charged_response_token_budget",
        "replay_charged_response_token_budget",
        "replay_realized_prompt_tokens",
        "replay_realized_response_tokens",
    )
    compute_by_domain: dict[str, dict[str, float]] = defaultdict(
        lambda: {field: 0.0 for field in compute_fields}
    )
    for run in ledger["runs"]:
        report, violations = parse_metrics(metric_paths(Path(run["run_dir"])))
        label = f"{run['domain']}/s{run['seed']}"
        run_reports.append(
            {
                "domain": run["domain"],
                "seed": run["seed"],
                "job_id": run["job_id"],
                "report": report,
                "violations": violations,
            }
        )
        all_violations.extend(f"{label}: {value}" for value in violations)
        domain_admissions[str(run["domain"])] += float(
            report["proposal_cumulative_admissions"]
        )
        domain_priority[str(run["domain"])] += float(
            report["priority_replay_groups_cumulative"]
        )
        for field in compute_fields:
            compute_by_domain[str(run["domain"])][field] += float(report[field])

    terminal = all(
        int(run["report"]["last_step"]) >= int(ledger["target_steps"])
        for run in run_reports
    )
    domain_gate = {
        domain: {
            "admissions": domain_admissions[domain],
            "priority_replay_groups": domain_priority[domain],
            "passed": domain_admissions[domain] > 0 and domain_priority[domain] > 0,
        }
        for domain in ledger["domains"]
    }
    if terminal:
        all_violations.extend(
            f"{domain}: terminal campaign had no admission and priority actuation"
            for domain, value in domain_gate.items()
            if not value["passed"]
        )
    payload = {
        "schema": "e102-full-open-bank-campaign-audit-v1",
        "ledger": str(LEDGER),
        "released": ledger.get("released"),
        "terminal": terminal,
        "domain_gate": domain_gate,
        "compute_accounting": {
            "unit": "tokens summed over the five terminal seeds in each domain",
            "proposal_charge_rule": (
                "generated proposal rows times the frozen per-update response "
                "ceiling"
            ),
            "proposal_realized_tokens": (
                "not retained by the E102 actor path; this preregistered "
                "reporting quantity is unavailable and is not imputed"
            ),
            "by_domain": dict(compute_by_domain),
        },
        "runs": run_reports,
        "violations": all_violations,
        "passed_so_far": not all_violations,
    }
    atomic_json(OUT, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 1 if all_violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
