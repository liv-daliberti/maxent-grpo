#!/usr/bin/env python3
"""Fail-closed live audit for E101m2 discovery opportunity."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_e101_open_bank_countdown_pilot as parent  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "var/artifacts/e101m2_open_bank_discovery_smoke_job.json"
OUT = ROOT / "var/artifacts/e101m2_open_bank_discovery_smoke_audit_latest.json"
LAUNCHER = ROOT / "ops/exp_scaling/launch_e101m2_open_bank_discovery_smoke.py"
TARGET_STEPS = 32
WALLTIME_CAP_SECONDS = 30 * 60


def main() -> int:
    if not LEDGER.is_file():
        payload = {
            "schema": "e101m2_open_bank_discovery_smoke_audit_v1",
            "status": "not_submitted",
            "violations": [],
        }
        parent.atomic_json(OUT, payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    violations: list[str] = []
    protocol = Path(ledger["protocol"])
    data_root = Path(ledger["data_root"])
    if not protocol.is_file() or parent.digest(protocol) != ledger["protocol_sha256"]:
        violations.append("frozen protocol digest mismatch")
    if not LAUNCHER.is_file() or parent.digest(LAUNCHER) != ledger["launcher_sha256"]:
        violations.append("frozen launcher digest mismatch")
    if not data_root.is_dir() or parent.tree_digest(data_root) != ledger["data_tree_sha256"]:
        violations.append("frozen data-tree digest mismatch")

    job_id = int(ledger["job_id"])
    scheduler = parent.scheduler_rows([job_id]).get(job_id, {})
    attempt = Path(ledger["run_dir"]) / f"debug_job{job_id}"
    metrics, metric_violations = parent.parse_metrics(
        attempt / "train_metrics.jsonl"
    )
    violations.extend(metric_violations)
    elapsed = scheduler.get("elapsed_seconds")
    if elapsed is not None and int(elapsed) > WALLTIME_CAP_SECONDS:
        violations.append(f"elapsed time exceeded cap: {elapsed}s")
    for stream in (ledger["stdout"], ledger["stderr"]):
        violations.extend(
            f"runtime failure: {value}"
            for value in parent.log_failures(Path(stream))
        )
    state = str(scheduler.get("state", "UNKNOWN"))
    if state.startswith(("FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY")):
        violations.append(f"terminal scheduler state: {state}")
    if state.startswith("COMPLETED"):
        if int(metrics.get("last_step", -1)) < TARGET_STEPS:
            violations.append("completed before registered terminal step")
        evaluation_steps = {
            int(row["step"]) for row in metrics.get("evaluations", [])
        }
        if not {0, TARGET_STEPS}.issubset(evaluation_steps):
            violations.append(
                f"missing registered evaluations: have {sorted(evaluation_steps)}"
            )

    mechanism = metrics.get("mechanism", {})
    complete = state.startswith("COMPLETED")
    if violations:
        status = "invalid"
    elif not complete:
        status = "running"
    elif mechanism.get("proposal_singleton_active_updates", 0) <= 0:
        status = "complete_no_proposal_eligibility"
    elif mechanism.get("proposal_validator_positive_rows", 0) <= 0:
        status = "complete_no_validator_positive_proposal"
    elif mechanism.get("proposal_admissions", 0) <= 0:
        status = "complete_positive_but_no_admission"
    elif mechanism.get("replay_actuation_updates_at_or_after_first_admission", 0) <= 0:
        status = "complete_admission_not_actuated"
    else:
        status = "complete_mechanism_success"

    payload = {
        "schema": "e101m2_open_bank_discovery_smoke_audit_v1",
        "status": status,
        "job_id": job_id,
        "target_steps": TARGET_STEPS,
        "walltime_cap_seconds": WALLTIME_CAP_SECONDS,
        "scheduler": scheduler,
        "metrics": metrics,
        "violations": violations,
        "interpretation": "discovery_mechanism_only_no_performance_comparison",
    }
    parent.atomic_json(OUT, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
