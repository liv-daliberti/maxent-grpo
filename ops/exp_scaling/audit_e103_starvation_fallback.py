#!/usr/bin/env python3
"""Audit E103 fallback scheduling, PPO isolation, replay, and safe balance."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_e102_full_open_bank_maxent_replay as base  # noqa: E402
import launch_e103_starvation_fallback_maxent_replay_05b as launch  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / launch.LEDGER
OUT = ROOT / "var/artifacts/e103_starvation_fallback_audit_latest.json"
SMOKE_OUT = ROOT / launch.SMOKE_AUDIT


def fallback_metrics(paths: Iterable[Path]) -> dict[str, float]:
    enabled_min = 1.0
    enabled_seen = False
    eligible_updates = 0.0
    activations = 0.0
    fallback_updates = 0.0
    extra_groups = 0.0
    fallback_admissions = 0.0
    fallback_max_attempts_seen = 0.0
    base_max_attempts_seen = 0.0
    forbidden_feedback = 0.0
    for _path, _line_number, row in base.rows(paths):
        if row.get("__invalid_json__"):
            continue
        enabled = base.metric(row, "counterfactual_proposal_starvation_fallback_enabled")
        if enabled is not None:
            enabled_seen = True
            enabled_min = min(enabled_min, enabled)
        eligible_updates += (
            base.metric(row, "counterfactual_proposal_starvation_eligible_update")
            or 0.0
        )
        activations = max(
            activations,
            base.metric(
                row,
                "counterfactual_proposal_starvation_fallback_activations_cumulative",
            )
            or 0.0,
        )
        fallback_updates = max(
            fallback_updates,
            base.metric(
                row,
                "counterfactual_proposal_starvation_fallback_updates_cumulative",
            )
            or 0.0,
        )
        extra_groups += (
            base.metric(
                row,
                "counterfactual_proposal_starvation_fallback_extra_groups",
            )
            or 0.0
        )
        fallback_admissions += (
            base.metric(
                row,
                "counterfactual_proposal_starvation_fallback_admitted_new_outcomes",
            )
            or 0.0
        )
        attempts = base.metric(row, "counterfactual_proposal_max_attempts")
        active = bool(
            base.metric(
                row,
                "counterfactual_proposal_starvation_fallback_active",
            )
            or 0.0
        )
        if attempts is not None:
            if active:
                fallback_max_attempts_seen = max(
                    fallback_max_attempts_seen, attempts
                )
            else:
                base_max_attempts_seen = max(base_max_attempts_seen, attempts)
        for suffix in (
            "counterfactual_proposal_starvation_gold_support_feedback",
            "counterfactual_proposal_starvation_desired_mode_count_feedback",
            "counterfactual_proposal_starvation_eval_feedback",
        ):
            forbidden_feedback = max(
                forbidden_feedback,
                abs(base.metric(row, suffix) or 0.0),
            )
    return {
        "fallback_enabled_min": enabled_min if enabled_seen else 0.0,
        "eligible_updates": eligible_updates,
        "fallback_activations": activations,
        "fallback_updates": fallback_updates,
        "fallback_extra_groups": extra_groups,
        "fallback_admissions": fallback_admissions,
        "fallback_max_attempts_seen": fallback_max_attempts_seen,
        "base_max_attempts_seen": base_max_attempts_seen,
        "starvation_forbidden_feedback_max": forbidden_feedback,
    }


def parse_metrics(paths: list[Path]) -> tuple[dict[str, Any], list[str]]:
    report, violations = base.parse_metrics(paths)
    report.update(fallback_metrics(paths))
    report["proposal_rows_to_ppo_max"] = report[
        "proposal_rows_to_ppo_max_abs"
    ]
    report["objective_outcome_delta_max"] = report[
        "proposal_objective_outcome_delta_max_abs"
    ]
    if report["fallback_enabled_min"] < 1.0:
        violations.append("proposal starvation fallback was disabled")
    if report["starvation_forbidden_feedback_max"] > 0.0:
        violations.append("fallback consumed forbidden target/evaluation feedback")
    return report, violations


def smoke_gate(run_dir: Path, target: int) -> tuple[dict[str, Any], list[str]]:
    paths = base.metric_paths(run_dir)
    report, violations = parse_metrics(paths)
    if report["last_step"] < target:
        violations.append(f"smoke has only {report['last_step']}/{target} optimizer steps")
    if report["fallback_activations"] <= 0:
        violations.append("smoke never activated the starvation fallback")
    if report["fallback_extra_groups"] <= 0:
        violations.append("smoke generated no extra fallback proposal group")
    if report["fallback_max_attempts_seen"] != float(
        launch.STARVATION_FALLBACK_MAX_ATTEMPTS
    ):
        violations.append("smoke did not use the registered fallback attempt budget")
    if report["base_max_attempts_seen"] not in (
        0.0,
        float(launch.PROPOSAL_MAX_ATTEMPTS),
    ):
        violations.append("smoke base proposal budget drifted from E102")
    if not report["retention_safe_seen"]:
        violations.append("smoke lacks retention-safe balance telemetry")
    if report["fallback_admissions"] > 0 and report[
        "priority_replay_groups_cumulative"
    ] <= 0:
        violations.append("fallback admission did not reach prioritized replay")
    return report, violations


def parse_smoke_spec(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("smoke must be DOMAIN=RUN_DIR")
    domain, raw_path = value.split("=", 1)
    if domain not in launch.DOMAINS:
        raise argparse.ArgumentTypeError(f"unknown E103 smoke domain: {domain}")
    return domain, Path(raw_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="append", type=parse_smoke_spec, default=[])
    parser.add_argument("--smoke-target", type=int, default=launch.SMOKE_TARGET_STEPS)
    parser.add_argument("--snapshot-root", type=Path)
    args = parser.parse_args()

    if args.smoke:
        smoke_dirs = dict(args.smoke)
        violations: list[str] = []
        if len(smoke_dirs) != len(args.smoke):
            violations.append("duplicate smoke domain")
        if set(smoke_dirs) != set(launch.DOMAINS):
            violations.append("smoke gate does not contain all five E103 domains")
        reports: dict[str, dict[str, Any]] = {}
        for domain, run_dir in smoke_dirs.items():
            report, run_violations = smoke_gate(run_dir, args.smoke_target)
            reports[domain] = report
            violations.extend(f"{domain}: {value}" for value in run_violations)
        if args.snapshot_root is None:
            violations.append("smoke gate requires the frozen source snapshot")
        payload = {
            "schema": "e103-starvation-fallback-smoke-audit-v1",
            "snapshot_root": str(args.snapshot_root) if args.snapshot_root else None,
            "smoke_target_steps": args.smoke_target,
            "reports": reports,
            "violations": violations,
            "passed": not violations,
        }
        base.atomic_json(SMOKE_OUT, payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 1 if violations else 0

    if not LEDGER.is_file():
        raise SystemExit(f"E103 ledger is absent: {LEDGER}")
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    run_reports: list[dict[str, Any]] = []
    violations: list[str] = []
    domain_totals: dict[str, dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    for run in ledger["runs"]:
        report, run_violations = parse_metrics(
            base.metric_paths(Path(run["run_dir"]))
        )
        label = f"{run['domain']}/s{run['seed']}"
        run_reports.append(
            {
                "domain": run["domain"],
                "seed": run["seed"],
                "job_id": run["job_id"],
                "report": report,
                "violations": run_violations,
            }
        )
        violations.extend(f"{label}: {value}" for value in run_violations)
        for field in (
            "proposal_cumulative_admissions",
            "fallback_activations",
            "fallback_updates",
            "fallback_extra_groups",
            "fallback_admissions",
            "priority_replay_groups_cumulative",
        ):
            domain_totals[str(run["domain"])][field] += float(report[field])
    terminal = all(
        int(run["report"]["last_step"]) >= int(ledger["target_steps"])
        for run in run_reports
    )
    payload = {
        "schema": "e103-starvation-fallback-audit-v1",
        "ledger": str(LEDGER),
        "terminal": terminal,
        "passed": terminal and not violations,
        "domain_totals": {
            domain: dict(values) for domain, values in domain_totals.items()
        },
        "runs": run_reports,
        "violations": violations,
    }
    base.atomic_json(OUT, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
