#!/usr/bin/env python3
"""Outcome-blind audit for the corrected Qwen-3B capacity retry."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_e104_group_centered_semantic_repair as main_audit  # noqa: E402
import audit_e104_qwen3_a6000_capacity_preflight as mechanism_audit  # noqa: E402
import launch_e104_group_centered_semantic_repair_three_scale as e104  # noqa: E402
import launch_e104_qwen3_update_only_capacity_retry as launch  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / launch.LEDGER
OUT = ROOT / launch.AUDIT


def main() -> int:
    if not LEDGER.is_file():
        raise SystemExit(f"retry ledger is absent: {LEDGER}")
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    violations: list[str] = []
    artifacts = (
        ("protocol", "protocol_sha256"),
        ("launcher", "launcher_sha256"),
        ("failed_preflight_ledger", "failed_preflight_ledger_sha256"),
        ("failed_preflight_audit", "failed_preflight_audit_sha256"),
        ("e104_ledger", "e104_ledger_sha256"),
    )
    for path_key, digest_key in artifacts:
        path = Path(str(ledger.get(path_key, "")))
        if not path.is_file() or e104.digest(path) != ledger.get(digest_key):
            violations.append(f"retry {path_key} digest mismatch")
    try:
        launch.verify_overlay()
    except SystemExit as exc:
        violations.append(str(exc))
    if launch.tree_digest(launch.OPS_OVERLAY) != ledger.get(
        "ops_overlay_tree_sha256"
    ):
        violations.append("retry ops overlay tree digest mismatch")
    snapshot = Path(str(ledger.get("snapshot_root", "")))
    try:
        e104.verify_snapshot(snapshot)
    except SystemExit as exc:
        violations.append(str(exc))
    run_dir = Path(str(ledger["run_dir"]))
    report, run_violations = mechanism_audit.parse_run(run_dir)
    violations.extend(run_violations)
    evaluation_artifacts = sorted(
        str(path) for path in run_dir.glob("debug_job*/eval*") if path.exists()
    )
    if evaluation_artifacts:
        violations.append("retry produced evaluation artifacts")
    job_id = int(ledger["job_id"])
    state = main_audit.scheduler_states([job_id]).get(job_id, "UNKNOWN")
    complete = state.startswith("COMPLETED") and report["last_step"] >= 1
    if state.startswith(("FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY")):
        violations.append(f"terminal scheduler state {state}")
    for stream in ("stdout", "stderr"):
        for marker in main_audit.log_failures(Path(str(ledger[stream]))):
            violations.append(f"{stream} contains {marker!r}")
    payload = {
        "schema": "e104_qwen3_update_only_capacity_retry_gate_v1",
        "ledger": str(LEDGER),
        "snapshot_root": str(snapshot),
        "ops_overlay": str(launch.OPS_OVERLAY),
        "job_id": job_id,
        "scheduler_state": state,
        "complete": complete,
        "passed": complete and not violations,
        "outcome_metrics_inspected": False,
        "evaluation_artifacts": evaluation_artifacts,
        "report": report,
        "violations": violations,
    }
    main_audit.shared.atomic_json(OUT, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
