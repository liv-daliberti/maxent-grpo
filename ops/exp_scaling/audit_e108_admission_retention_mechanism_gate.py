#!/usr/bin/env python3
"""Audit E108 admission-to-retention measurement and priority actuation."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_e102_full_open_bank_maxent_replay as base  # noqa: E402
import launch_e108_admission_retention_mechanism_gate as launch  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / launch.LEDGER
OUT = ROOT / "var/artifacts/e108_admission_retention_mechanism_gate_audit_latest.json"


def retention_metric(row: dict[str, Any], name: str) -> float | None:
    for family in (
        "canonical_replay_proposal_retention_",
        "online_canonical_proposal_retention_",
    ):
        value = base.metric(row, family + name)
        if value is not None:
            return value
    return None


def parse_metrics(paths: list[Path]) -> tuple[dict[str, Any], list[str]]:
    report, violations = base.parse_metrics(paths)
    tracking_seen = False
    tracking_min = 1.0
    adaptive_seen = False
    adaptive_min = 1.0
    adaptive_max = 0.0
    latest: dict[str, float] = {}
    maxima: dict[str, float] = defaultdict(float)
    forbidden_feedback_max = 0.0
    fields = (
        "tracked_admissions",
        "rollout_eligible_admissions",
        "rollout_converted_admissions",
        "rollout_conversion_fraction",
        "rollout_row_frequency",
        "score_observed_admissions",
        "score_followup_admissions",
        "score_retained_admissions",
        "score_retained_fraction",
        "joint_eligible_admissions",
        "joint_retained_admissions",
        "joint_retained_fraction",
        "mean_logprob_drop_mean",
        "mean_logprob_drop_max",
        "sequence_logprob_drop_mean",
        "sequence_logprob_drop_max",
        "rollout_refresh_requests_cumulative",
        "score_refresh_requests_cumulative",
        "refresh_requests_cumulative",
        "priority_visits_added_cumulative",
    )
    for _path, _line, row in base.rows(paths):
        if row.get("__invalid_json__"):
            continue
        tracking = retention_metric(row, "tracking_enabled")
        if tracking is not None:
            tracking_seen = True
            tracking_min = min(tracking_min, tracking)
        adaptive = retention_metric(row, "adaptive_priority_enabled")
        if adaptive is not None:
            adaptive_seen = True
            adaptive_min = min(adaptive_min, adaptive)
            adaptive_max = max(adaptive_max, adaptive)
        for field in fields:
            value = retention_metric(row, field)
            if value is not None:
                latest[field] = value
                maxima[field] = max(maxima[field], value)
        for field in (
            "gold_support_feedback",
            "desired_mode_count_feedback",
            "eval_feedback",
        ):
            forbidden_feedback_max = max(
                forbidden_feedback_max,
                abs(retention_metric(row, field) or 0.0),
            )

    if not tracking_seen:
        violations.append("admission-retention telemetry never materialized")
    elif tracking_min < 1.0:
        violations.append("admission-retention tracking was disabled")
    if not adaptive_seen:
        violations.append("adaptive-retention configuration telemetry is absent")
    if forbidden_feedback_max > 0.0:
        violations.append("retention controller consumed forbidden feedback")
    report.update(
        {
            "retention_tracking_seen": tracking_seen,
            "retention_tracking_min": tracking_min if tracking_seen else 0.0,
            "adaptive_priority_seen": adaptive_seen,
            "adaptive_priority_min": adaptive_min if adaptive_seen else 0.0,
            "adaptive_priority_max": adaptive_max,
            "retention_forbidden_feedback_max": forbidden_feedback_max,
            "retention_latest": latest,
            "retention_maxima": dict(maxima),
        }
    )
    return report, violations


def main() -> int:
    if not LEDGER.is_file():
        raise SystemExit(f"E108 ledger is absent: {LEDGER}")
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    violations: list[str] = []
    reports: list[dict[str, Any]] = []
    arm_totals: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    expected = {(arm, domain) for arm in launch.ARMS for domain in launch.DOMAINS}
    observed = {
        (str(run.get("arm")), str(run.get("domain"))) for run in ledger.get("runs", [])
    }
    if observed != expected:
        violations.append("ledger does not contain the registered ten cells")

    for run in ledger.get("runs", []):
        arm = str(run["arm"])
        domain = str(run["domain"])
        report, run_violations = parse_metrics(
            base.metric_paths(Path(str(run["run_dir"])))
        )
        if int(report["last_step"]) < launch.TARGET_STEPS:
            run_violations.append(
                f"only {report['last_step']}/{launch.TARGET_STEPS} updates"
            )
        expected_adaptive = 1.0 if arm == "adaptive_retention" else 0.0
        if report["adaptive_priority_min"] != expected_adaptive or (
            report["adaptive_priority_max"] != expected_adaptive
        ):
            run_violations.append("adaptive-retention arm configuration drifted")
        reports.append(
            {
                "arm": arm,
                "domain": domain,
                "seed": int(run["seed"]),
                "job_id": int(run["job_id"]),
                "report": report,
                "violations": run_violations,
            }
        )
        violations.extend(f"{arm}/{domain}: {value}" for value in run_violations)
        maxima = report["retention_maxima"]
        for field in (
            "tracked_admissions",
            "score_observed_admissions",
            "score_followup_admissions",
            "rollout_converted_admissions",
            "refresh_requests_cumulative",
            "priority_visits_added_cumulative",
        ):
            arm_totals[arm][field] += float(maxima.get(field, 0.0))

    for arm in launch.ARMS:
        totals = arm_totals[arm]
        if totals["tracked_admissions"] <= 0.0:
            violations.append(f"{arm}: no proposal admission was tracked")
        if totals["score_observed_admissions"] <= 0.0:
            violations.append(f"{arm}: no admitted exemplar received a replay score")
    passive = arm_totals["retention_tracking"]
    if passive["refresh_requests_cumulative"] != 0.0:
        violations.append("passive arm emitted an adaptive refresh request")
    if passive["priority_visits_added_cumulative"] != 0.0:
        violations.append("passive arm added adaptive priority visits")
    adaptive = arm_totals["adaptive_retention"]
    if adaptive["refresh_requests_cumulative"] <= 0.0:
        violations.append("adaptive arm emitted no retention refresh request")
    if adaptive["priority_visits_added_cumulative"] <= 0.0:
        violations.append("adaptive arm added no retention-triggered priority visit")

    terminal = bool(reports) and all(
        int(item["report"]["last_step"]) >= launch.TARGET_STEPS for item in reports
    )
    payload = {
        "schema": "e108_admission_retention_mechanism_gate_audit_v1",
        "ledger": str(LEDGER),
        "terminal": terminal,
        "passed": terminal and not violations,
        "outcomes_used_for_gate": False,
        "arm_totals": {arm: dict(values) for arm, values in arm_totals.items()},
        "runs": reports,
        "violations": violations,
    }
    base.atomic_json(OUT, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
