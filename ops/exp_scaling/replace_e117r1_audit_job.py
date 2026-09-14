#!/usr/bin/env python3
"""Replace E117-R1's pending audit with the frozen E117-A1 correction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import apply_e117s1_lowprio_partition_repair as scheduler  # noqa: E402
import launch_e117_same_plumbing_component_preflight as e117  # noqa: E402
import launch_e117r1_zero_step_lowprio_replacement as e117r1  # noqa: E402


PROTOCOL = (
    "paper/preregistration/e117a1_actuable_pressure_audit_correction_20260824.md"
)
HISTORY = "var/artifacts/e117a1_audit_job_replacement.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--snapshot-root", type=Path, required=True)
    args = parser.parse_args()
    if not args.replace:
        raise SystemExit("pass --replace to install E117-A1")
    root = e117.repo_root()
    snapshot = args.snapshot_root.resolve()
    ledger_path = root / e117r1.LEDGER
    audit_job_path = root / e117r1.AUDIT_JOB
    retirement_path = root / e117r1.RETIREMENT
    history_path = root / HISTORY
    protocol_path = root / PROTOCOL
    if history_path.exists():
        raise SystemExit(f"refusing duplicate E117-A1 installation: {history_path}")
    audit_script = (
        snapshot
        / "ops/exp_scaling/audit_e117_same_plumbing_component_preflight.py"
    )
    if not audit_script.is_file() or e117.digest(audit_script) != e117.digest(
        root / "ops/exp_scaling/audit_e117_same_plumbing_component_preflight.py"
    ):
        raise SystemExit("E117-A1 snapshot does not contain the corrected audit")

    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    audit_job = json.loads(audit_job_path.read_text(encoding="utf-8"))
    retirement = json.loads(retirement_path.read_text(encoding="utf-8"))
    runs = list(ledger.get("runs", []))
    job_ids = [int(row["job_id"]) for row in runs]
    if len(job_ids) != 12 or ledger.get("released") is not True:
        raise SystemExit("E117-R1 replacement ledger is incomplete")
    old_job_id = int(audit_job["audit_job_id"])
    old_record = scheduler.show(old_job_id)
    if (
        scheduler.field(old_record, "JobState") != "PENDING"
        or scheduler.field(old_record, "Reason") != "Dependency"
        or scheduler.field(old_record, "RunTime") != "00:00:00"
        or scheduler.field(old_record, "Restarts") != "0"
    ):
        raise SystemExit("existing E117-R1 audit is no longer replaceable")

    new_job_id = 0
    try:
        new_job_id, new_record = e117r1.schedule_audit(
            root=root,
            snapshot=snapshot,
            ledger=ledger_path,
            job_ids=job_ids,
        )
        subprocess.run(
            ["scancel", str(old_job_id)],
            capture_output=True,
            text=True,
            check=True,
        )
        old_state = e117r1.accounting([old_job_id]).get(old_job_id, {})
        if old_state.get("state") != "CANCELLED":
            raise RuntimeError("old E117-R1 audit did not become canceled")
        payload = {
            "schema": "e117a1_audit_job_replacement_v1",
            "protocol": str(protocol_path),
            "protocol_sha256": e117.digest(protocol_path),
            "previous_audit_job_id": old_job_id,
            "previous_scheduler_record": old_record,
            "replacement_audit_job_id": new_job_id,
            "replacement_scheduler_record": new_record,
            "dependency_job_ids": job_ids,
            "audit_snapshot_root": str(snapshot),
            "audit_script": str(audit_script),
            "audit_script_sha256": e117.digest(audit_script),
            "training_configuration_changed": False,
            "outcomes_inspected": False,
            "installed": True,
        }
        audit_job.update(
            {
                "schema": "e117r1_same_plumbing_component_preflight_audit_job_v2",
                "audit_job_id": new_job_id,
                "audit_snapshot_root": str(snapshot),
                "audit_script_sha256": e117.digest(audit_script),
                "scheduler_record": new_record,
                "supersedes_audit_job_id": old_job_id,
            }
        )
        ledger["audit_job_id"] = new_job_id
        ledger["audit_amendment"] = str(protocol_path)
        ledger["audit_amendment_sha256"] = e117.digest(protocol_path)
        retirement["replacement_audit_job_id"] = new_job_id
        e117.e111.e81.atomic_json(history_path, payload)
        e117.e111.e81.atomic_json(audit_job_path, audit_job)
        e117.e111.e81.atomic_json(ledger_path, ledger)
        e117.e111.e81.atomic_json(retirement_path, retirement)
    except Exception:
        if new_job_id:
            subprocess.run(
                ["scancel", str(new_job_id)],
                capture_output=True,
                text=True,
                check=False,
            )
        raise
    print(
        f"[e117a1] previous={old_job_id} replacement={new_job_id} "
        f"snapshot={snapshot}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
