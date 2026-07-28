#!/usr/bin/env python3
"""Machine-check whether the five-domain campaign supports a paper claim."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = ROOT / "var/artifacts/e65_five_domain_confirmation_audit_latest.json"
RESULTS = ROOT / "var/artifacts/e65_five_domain_confirmation_results_latest.json"
CADENCE = ROOT / "var/artifacts/e65_eval_cadence_audit_latest.json"
COVERAGE = (
    ROOT / "var/artifacts/e65_fixed_checkpoint_coverage_audit_latest.json"
)
FIGURE_PDF = (
    ROOT / "paper/figures/e61r1_e58_vs_grpo_05b_12ep_live.pdf"
)
FIGURE_PNG = (
    ROOT / "paper/figures/e61r1_e58_vs_grpo_05b_12ep_live.png"
)
DIAGNOSTIC_PDF = (
    ROOT
    / "paper/figures/"
    "e61r1_e58_vs_grpo_05b_12ep_all_epoch_diagnostic_live.pdf"
)
OUT = ROOT / "var/artifacts/e65_legitimate_result_readiness_latest.json"
EXPECTED_DOMAINS = {
    "Graph coloring",
    "Countdown",
    "Python factors",
    "MathIR action menu",
    "Held-out MATH-500 transfer",
}
REQUIRED_GATES = (
    "e58_cross_domain",
    "e68_repair",
    "plumbing_consistency",
    "math500_realism",
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _artifact_ready(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def main() -> None:
    campaign = _load(CAMPAIGN)
    results = _load(RESULTS)
    cadence = _load(CADENCE)
    coverage = _load(COVERAGE)
    violations: list[str] = []
    pending: list[str] = []
    failed: list[str] = []

    domains = set(results.get("domains", {}))
    if domains != EXPECTED_DOMAINS:
        violations.append(
            "result domain set mismatch: "
            f"observed={sorted(domains)} expected={sorted(EXPECTED_DOMAINS)}"
        )
    for path in (FIGURE_PDF, FIGURE_PNG, DIAGNOSTIC_PDF):
        if not _artifact_ready(path):
            violations.append(f"missing or empty rendered artifact: {path}")

    coverage_domains = coverage.get("domains", {})
    if set(coverage_domains) != EXPECTED_DOMAINS:
        violations.append("checkpoint coverage domain set mismatch")
    for domain, record in coverage_domains.items():
        maximum = int(record.get("maximum_registered_checkpoints", 0))
        if maximum <= 0 or maximum > 10:
            violations.append(
                f"{domain}: registered checkpoint maximum is {maximum}, "
                "expected 1..10"
            )

    cadence_status = cadence.get("status")
    if cadence_status != "pass":
        failed.append("evaluation cadence audit is not pass")
    if cadence.get("violations"):
        violations.extend(
            f"evaluation cadence: {value}"
            for value in cadence.get("violations", [])
        )

    coverage_status = coverage.get("status")
    if coverage_status == "fail":
        failed.append("fixed checkpoint coverage audit failed")
    elif coverage_status != "pass":
        pending.append("fixed seed-level checkpoint surface is incomplete")
    if coverage.get("violations"):
        violations.extend(
            f"fixed checkpoint coverage: {value}"
            for value in coverage.get("violations", [])
        )

    campaign_status = campaign.get("status")
    if campaign_status == "fail":
        failed.append("combined 54-run integrity audit failed")
    elif campaign_status != "pass":
        pending.append("combined 54-run integrity audit is not terminal")
    if campaign.get("violations"):
        violations.extend(
            f"combined campaign: {value}"
            for value in campaign.get("violations", [])
        )

    gates = results.get("gates", {})
    gate_statuses: dict[str, str] = {}
    for gate_name in REQUIRED_GATES:
        gate_status = str(gates.get(gate_name, {}).get("status", "missing"))
        gate_statuses[gate_name] = gate_status
        if gate_status == "fail":
            failed.append(f"primary gate failed: {gate_name}")
        elif gate_status != "pass":
            pending.append(f"primary gate not terminal: {gate_name}")

    summary = campaign.get("summary", {})
    terminal_runs = int(summary.get("terminal_runs", 0))
    expected_runs = int(summary.get("expected_runs", 54))
    if terminal_runs != expected_runs:
        pending.append(
            f"registered terminal runs incomplete: {terminal_runs}/{expected_runs}"
        )

    status = (
        "fail"
        if violations or failed
        else "pass"
        if not pending
        else "in_progress"
    )
    payload = {
        "schema": "e65_legitimate_result_readiness_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "requirements": {
            "five_exact_domains": domains == EXPECTED_DOMAINS,
            "maximum_ten_checkpoints_per_domain": not any(
                "checkpoint maximum" in value for value in violations
            ),
            "three_seed_fixed_surface_status": coverage_status,
            "evaluation_at_least_once_per_epoch_status": cadence_status,
            "combined_54_run_integrity_status": campaign_status,
            "primary_gate_statuses": gate_statuses,
            "paper_figure_ready": _artifact_ready(FIGURE_PDF)
            and _artifact_ready(FIGURE_PNG),
            "all_epoch_diagnostic_ready": _artifact_ready(DIAGNOSTIC_PDF),
        },
        "summary": {
            "terminal_runs": terminal_runs,
            "expected_runs": expected_runs,
            "fixed_checkpoint_cells": int(
                coverage.get("summary", {}).get(
                    "landed_checkpoint_cells",
                    0,
                )
            ),
            "expected_fixed_checkpoint_cells": int(
                coverage.get("summary", {}).get(
                    "expected_checkpoint_cells",
                    174,
                )
            ),
            "pending_requirement_count": len(set(pending)),
            "failed_requirement_count": len(set(failed)),
            "integrity_violation_count": len(set(violations)),
        },
        "pending_requirements": sorted(set(pending)),
        "failed_requirements": sorted(set(failed)),
        "violations": sorted(set(violations)),
    }
    _atomic_json(OUT, payload)
    print(
        f"[legitimate-result-readiness] status={status} "
        f"terminal={terminal_runs}/{expected_runs} "
        f"pending={len(set(pending))} failed={len(set(failed))} "
        f"violations={len(set(violations))}"
    )


if __name__ == "__main__":
    main()
