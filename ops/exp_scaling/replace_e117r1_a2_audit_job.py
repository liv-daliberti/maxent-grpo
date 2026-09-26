#!/usr/bin/env python3
"""Replace E117-R1's zero-runtime audit with the frozen complete A2 audit."""

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


PROTOCOL = "paper/preregistration/e117a2_complete_mechanism_audit_20260825.md"
HISTORY = "var/artifacts/e117a2_audit_job_replacement.json"
AUDIT_SOURCE = "ops/exp_scaling/audit_e117_same_plumbing_component_preflight.py"
AMENDMENT_NAME = "E117-A2"
REPLACEMENT_SCHEMA = "e117a2_audit_job_replacement_v1"
AUDIT_JOB_SCHEMA = "e117r1_same_plumbing_component_preflight_audit_job_v3"
LOG_LABEL = "e117a2"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--snapshot-root", type=Path, required=True)
    args = parser.parse_args()
    if not args.replace:
        raise SystemExit(f"pass --replace to install {AMENDMENT_NAME}")

    root = e117.repo_root()
    snapshot = args.snapshot_root.resolve()
    ledger_path = root / e117r1.LEDGER
    audit_job_path = root / e117r1.AUDIT_JOB
    retirement_path = root / e117r1.RETIREMENT
    audit_output_path = root / e117r1.AUDIT
    history_path = root / HISTORY
    protocol_path = root / PROTOCOL
    root_audit_script = root / AUDIT_SOURCE
    snapshot_audit_script = snapshot / AUDIT_SOURCE

    if history_path.exists():
        raise SystemExit(
            f"refusing duplicate {AMENDMENT_NAME} installation: {history_path}"
        )
    if audit_output_path.exists():
        raise SystemExit(
            f"refusing {AMENDMENT_NAME} after an E117-R1 audit output exists"
        )
    if not protocol_path.is_file():
        raise SystemExit(f"{AMENDMENT_NAME} protocol is absent: {protocol_path}")
    if not snapshot_audit_script.is_file() or e117.digest(
        snapshot_audit_script
    ) != e117.digest(root_audit_script):
        raise SystemExit(
            f"{AMENDMENT_NAME} snapshot does not contain the complete audit"
        )

    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    audit_job = json.loads(audit_job_path.read_text(encoding="utf-8"))
    retirement = json.loads(retirement_path.read_text(encoding="utf-8"))
    runs = list(ledger.get("runs", []))
    job_ids = [int(row["job_id"]) for row in runs]
    if (
        len(job_ids) != 12
        or len(set(job_ids)) != 12
        or ledger.get("released") is not True
        or audit_job.get("dependency_job_ids") != job_ids
    ):
        raise SystemExit("E117-R1 effective ledger/dependency identity is incomplete")

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
    old_cancelled = False
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
        old_cancelled = True

        prior_amendment = ledger.get("audit_amendment")
        prior_superseded = list(audit_job.get("superseded_audit_job_ids", []))
        immediate_prior = audit_job.get("supersedes_audit_job_id")
        if immediate_prior is not None and int(immediate_prior) not in prior_superseded:
            prior_superseded.append(int(immediate_prior))
        prior_superseded.append(old_job_id)
        prior_superseded = list(dict.fromkeys(prior_superseded))

        payload = {
            "schema": REPLACEMENT_SCHEMA,
            "protocol": str(protocol_path),
            "protocol_sha256": e117.digest(protocol_path),
            "previous_audit_amendment": prior_amendment,
            "previous_audit_job_id": old_job_id,
            "previous_scheduler_record": old_record,
            "replacement_audit_job_id": new_job_id,
            "replacement_scheduler_record": new_record,
            "dependency_job_ids": job_ids,
            "audit_snapshot_root": str(snapshot),
            "audit_snapshot_identity_sha256": e117.digest(
                snapshot / "SNAPSHOT_IDENTITY.json"
            ),
            "audit_script": str(snapshot_audit_script),
            "audit_script_sha256": e117.digest(snapshot_audit_script),
            "superseded_audit_job_ids": prior_superseded,
            "training_configuration_changed": False,
            "training_jobs_changed": False,
            "outcomes_inspected": False,
            "installed": True,
        }
        audit_job.update(
            {
                "schema": AUDIT_JOB_SCHEMA,
                "audit_job_id": new_job_id,
                "audit_snapshot_root": str(snapshot),
                "audit_script": str(snapshot_audit_script),
                "audit_script_sha256": e117.digest(snapshot_audit_script),
                "scheduler_record": new_record,
                "supersedes_audit_job_id": old_job_id,
                "superseded_audit_job_ids": prior_superseded,
                "audit_protocol": str(protocol_path),
                "audit_protocol_sha256": e117.digest(protocol_path),
            }
        )
        amendment_history = list(ledger.get("audit_amendment_history", []))
        if prior_amendment and prior_amendment not in amendment_history:
            amendment_history.append(prior_amendment)
        amendment_history.append(str(protocol_path))
        ledger.update(
            {
                "audit_job_id": new_job_id,
                "audit_amendment": str(protocol_path),
                "audit_amendment_sha256": e117.digest(protocol_path),
                "audit_amendment_history": amendment_history,
            }
        )
        retirement["replacement_audit_job_id"] = new_job_id
        e117.e111.e81.atomic_json(history_path, payload)
        e117.e111.e81.atomic_json(audit_job_path, audit_job)
        e117.e111.e81.atomic_json(ledger_path, ledger)
        e117.e111.e81.atomic_json(retirement_path, retirement)
    except Exception:
        if new_job_id and not old_cancelled:
            subprocess.run(
                ["scancel", str(new_job_id)],
                capture_output=True,
                text=True,
                check=False,
            )
        raise

    print(
        f"[{LOG_LABEL}] previous={old_job_id} replacement={new_job_id} "
        f"snapshot={snapshot}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
