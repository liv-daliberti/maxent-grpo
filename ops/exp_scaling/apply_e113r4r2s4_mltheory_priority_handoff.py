#!/usr/bin/env python3
"""Move the 46 pending E113-R4-R2 jobs to the mltheory fair-share account."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e113r4_official_dapo_common as common  # noqa: E402
import launch_e113r4_official_dapo as launch  # noqa: E402
import release_e113r4r2s3_user_holds as release  # noqa: E402


PROTOCOL = common.ROOT / (
    "paper/preregistration/"
    "e113r4r2s4_mltheory_priority_handoff_20260829.md"
)
PRECEDING = common.ROOT / "var/artifacts/e113r4r2s3_user_hold_release.json"
ARTIFACT = common.ROOT / (
    "var/artifacts/e113r4r2s4_mltheory_priority_handoff.json"
)
APPLICATION = Path(__file__).resolve()
JOB_IDS = tuple(range(30869115, 30869161))


def validate(
    record: str,
    *,
    account: str,
    expected_environment: str,
    held: bool,
    allow_running: bool = False,
) -> None:
    allowed_states = {"PENDING", "RUNNING"} if allow_running else {"PENDING"}
    failures: dict[str, Any] = {}
    if release.field(record, "JobState") not in allowed_states:
        failures["JobState"] = release.field(record, "JobState")
    if held and release.field(record, "Reason") != "JobHeldUser":
        failures["Reason"] = (
            release.field(record, "Reason"),
            "JobHeldUser",
        )
    if not held and release.field(record, "Reason") == "JobHeldUser":
        failures["Reason"] = "JobHeldUser"
    required = {
        "Restarts": "0",
        "Partition": "all",
        "Account": account,
        "QOS": "long",
        "Dependency": "(null)",
        "Requeue": "1",
        "ReqNodeList": "(null)",
        "ReqTRES": (
            "cpu=16,mem=128G,node=1,billing=44,gres/gpu=1,gres/gpu:a6000=1"
        ),
        "TresPerNode": "gres/gpu:a6000:1",
        "TimeLimit": "12:00:00",
        "Nice": "0",
    }
    if not allow_running:
        required["RunTime"] = "00:00:00"
    for key, expected in required.items():
        if release.field(record, key) != expected:
            failures[key] = (release.field(record, key), expected)
    if release.environment(record) != expected_environment:
        failures["environment"] = "scientific export drift"
    if failures:
        raise RuntimeError(f"E113-R4-R2-S4 scheduler identity drifted: {failures}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if args.check == args.apply:
        raise SystemExit("pass exactly one of --check or --apply")
    if ARTIFACT.exists():
        raise SystemExit(f"refusing duplicate E113-R4-R2-S4: {ARTIFACT}")
    if not PROTOCOL.is_file():
        raise SystemExit(f"priority-handoff protocol is absent: {PROTOCOL}")
    if not PRECEDING.is_file():
        raise SystemExit(f"S3 release record is absent: {PRECEDING}")

    preceding = json.loads(PRECEDING.read_text(encoding="utf-8"))
    if preceding.get("schema") != "e113r4r2s3_user_hold_release_v1":
        raise RuntimeError("unexpected E113-R4-R2-S3 release schema")
    if preceding.get("released_job_ids") != list(JOB_IDS):
        raise RuntimeError("E113-R4-R2-S3 released job set drifted")

    ledger = release.load_ledger()
    rows = {int(row["job_id"]): row for row in ledger.get("runs", [])}
    if not all(job_id in rows for job_id in JOB_IDS):
        raise RuntimeError("E113-R4-R2 effective run ledger is incomplete")

    records: dict[int, dict[str, str]] = {}
    for job_id in JOB_IDS:
        row = rows[job_id]
        run_dir = Path(str(row["run_dir"]))
        if run_dir.exists():
            raise RuntimeError(f"zero-runtime E113-R4 run directory exists: {run_dir}")
        expected_environment = release.environment(str(row["held_scheduler_record"]))
        before = release.show(job_id)
        validate(
            before,
            account="allcs",
            expected_environment=expected_environment,
            held=False,
        )
        if release.field(before, "Reason") != "Priority":
            raise RuntimeError(
                f"E113-R4 job {job_id} is not priority-blocked: "
                f"{release.field(before, 'Reason')}"
            )
        records[job_id] = {
            "environment": expected_environment,
            "before": before,
            "family": str(row["family"]),
            "domain": str(row["domain"]),
            "seed": str(row["seed"]),
            "run_dir": str(run_dir),
        }

    if args.check:
        print(f"[e113r4r2s4] preflight_passed=True jobs={len(JOB_IDS)}")
        return 0

    changed: list[int] = []
    released = False
    try:
        release.run(["scontrol", "uhold", *[str(job_id) for job_id in JOB_IDS]])
        for job_id in JOB_IDS:
            held_record = release.show(job_id)
            validate(
                held_record,
                account="allcs",
                expected_environment=records[job_id]["environment"],
                held=True,
            )
            records[job_id]["held"] = held_record

        for job_id in JOB_IDS:
            release.run(
                [
                    "scontrol",
                    "update",
                    f"JobId={job_id}",
                    "Account=mltheory",
                    "Partition=all",
                ]
            )
            changed.append(job_id)
        for job_id in JOB_IDS:
            after_held = release.show(job_id)
            validate(
                after_held,
                account="mltheory",
                expected_environment=records[job_id]["environment"],
                held=True,
            )
            records[job_id]["after_held"] = after_held

        payload: dict[str, Any] = {
            "schema": "e113r4r2s4_mltheory_priority_handoff_v1",
            "recorded_at": datetime.now(timezone.utc).astimezone().isoformat(),
            "authorization": (
                "User explicitly handed campaign execution priority from "
                "terminal E117-R1 to E113-R4."
            ),
            "protocol": str(PROTOCOL),
            "protocol_sha256": launch.digest(PROTOCOL),
            "application": str(APPLICATION),
            "application_sha256": launch.digest(APPLICATION),
            "preceding_artifact": str(PRECEDING),
            "preceding_artifact_sha256": launch.digest(PRECEDING),
            "ledger": str(common.LEDGER),
            "ledger_sha256": launch.digest(common.LEDGER),
            "exact_job_ids": list(JOB_IDS),
            "old_account": "allcs",
            "new_account": "mltheory",
            "partition_retained": "all",
            "qos_retained": "long",
            "hardware_retained": "gpu:a6000:1",
            "scientific_environment_changed": False,
            "incomplete_endpoint_values_inspected": False,
            "records": [
                {
                    "job_id": job_id,
                    "family": records[job_id]["family"],
                    "domain": records[job_id]["domain"],
                    "seed": int(records[job_id]["seed"]),
                    "run_dir": records[job_id]["run_dir"],
                    "before": records[job_id]["before"],
                    "held": records[job_id]["held"],
                    "after_held": records[job_id]["after_held"],
                }
                for job_id in JOB_IDS
            ],
            "installed": False,
        }
        launch.atomic_json(ARTIFACT, payload)

        release.run(["scontrol", "release", *[str(job_id) for job_id in JOB_IDS]])
        released = True
        for row in payload["records"]:
            job_id = int(row["job_id"])
            current = release.show(job_id)
            validate(
                current,
                account="mltheory",
                expected_environment=records[job_id]["environment"],
                held=False,
                allow_running=True,
            )
            row["after_release"] = current
        payload["installed"] = True
        launch.atomic_json(ARTIFACT, payload)
    except Exception:
        if not released:
            for job_id in changed:
                subprocess.run(
                    [
                        "scontrol",
                        "update",
                        f"JobId={job_id}",
                        "Account=allcs",
                        "Partition=all",
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                )
            subprocess.run(
                ["scontrol", "release", *[str(job_id) for job_id in JOB_IDS]],
                capture_output=True,
                text=True,
                check=False,
            )
            if ARTIFACT.exists():
                ARTIFACT.unlink()
        raise

    states = {
        release.field(str(row["after_release"]), "JobState")
        for row in payload["records"]
    }
    print(
        f"[e113r4r2s4] account=mltheory jobs={len(JOB_IDS)} "
        f"states={','.join(sorted(states))}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
