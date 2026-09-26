#!/usr/bin/env python3
"""Apply E117-S1's transactional zero-runtime partition repair."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import launch_e117_same_plumbing_component_preflight as e117  # noqa: E402


PROTOCOL = "paper/preregistration/e117s1_lowprio_partition_repair_20260824.md"
ARTIFACT = "var/artifacts/e117s1_lowprio_partition_repair.json"


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"command failed {command}: {result.stderr.strip()}")
    return result


def show(job_id: int) -> str:
    return run(["scontrol", "show", "job", "-dd", "-o", str(job_id)]).stdout.strip()


def field(record: str, name: str) -> str:
    match = re.search(rf"(?:^| ){re.escape(name)}=([^ ]*)", record)
    if match is None:
        raise RuntimeError(f"scheduler record lacks {name}")
    return match.group(1)


def environment(record: str) -> str:
    match = re.search(r"(?:^| )SubmitLine=.*?--export=([^ ]+)", record)
    if match is None:
        raise RuntimeError("scheduler record lacks a SubmitLine export surface")
    return match.group(1)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def assert_zero_runtime(record: str, *, node: str, expected_environment: str) -> None:
    required = {
        "JobState": "PENDING",
        "RunTime": "00:00:00",
        "Restarts": "0",
        "Account": "mltheory",
        "ReqNodeList": node,
        "ReqTRES": "cpu=8,mem=64G,node=1,billing=28,gres/gpu=1",
        "TresPerNode": "gres/gpu:1",
        "TimeLimit": "08:00:00",
    }
    failures = {
        key: (field(record, key), value)
        for key, value in required.items()
        if field(record, key) != value
    }
    if field(record, "NodeList") not in ("", "(null)"):
        failures["NodeList"] = (field(record, "NodeList"), "")
    if environment(record) != expected_environment:
        failures["Environment"] = (
            sha256_text(environment(record)),
            sha256_text(expected_environment),
        )
    if failures:
        raise RuntimeError(f"zero-runtime E117 scheduler identity drifted: {failures}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    root = e117.repo_root()
    ledger_path = root / e117.LEDGER
    protocol_path = root / PROTOCOL
    artifact_path = root / ARTIFACT
    if not args.apply:
        raise SystemExit("pass --apply to install E117-S1")
    if artifact_path.exists():
        raise SystemExit(f"refusing duplicate E117-S1 application: {artifact_path}")
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    runs = list(ledger.get("runs", []))
    if ledger.get("released") is not True or len(runs) != 12:
        raise SystemExit("E117 release ledger is not complete")

    records: dict[int, dict[str, Any]] = {}
    for row in runs:
        job_id = int(row["job_id"])
        expected_environment = environment(str(row["held_scheduler_record"]))
        before = show(job_id)
        assert_zero_runtime(
            before,
            node=str(row["node"]),
            expected_environment=expected_environment,
        )
        if field(before, "Partition") != "mltheory":
            raise SystemExit(f"E117 job {job_id} no longer has partition mltheory")
        if field(before, "Reason") != "BadConstraints":
            raise SystemExit(f"E117 job {job_id} is not blocked by BadConstraints")
        run_dir = Path(str(row["run_dir"]))
        if run_dir.exists():
            raise SystemExit(f"E117 job {job_id} already created {run_dir}")
        records[job_id] = {
            "node": str(row["node"]),
            "environment": expected_environment,
            "environment_sha256": sha256_text(expected_environment),
            "before": before,
        }

    job_ids = sorted(records)
    changed: list[int] = []
    try:
        for job_id in job_ids:
            # Plain `hold` is administrator-owned on this site and cannot be
            # released by the submitting user. Always request a user hold.
            run(["scontrol", "uhold", str(job_id)])
        for job_id in job_ids:
            held = show(job_id)
            assert_zero_runtime(
                held,
                node=str(records[job_id]["node"]),
                expected_environment=str(records[job_id]["environment"]),
            )
            if field(held, "Reason") != "JobHeldUser":
                raise RuntimeError(f"E117 job {job_id} was not held")
            records[job_id]["held"] = held
        for job_id in job_ids:
            run(
                [
                    "scontrol",
                    "update",
                    f"JobId={job_id}",
                    "Partition=lowprio",
                    "Account=mltheory",
                ]
            )
            changed.append(job_id)
        for job_id in job_ids:
            after = show(job_id)
            assert_zero_runtime(
                after,
                node=str(records[job_id]["node"]),
                expected_environment=str(records[job_id]["environment"]),
            )
            if field(after, "Partition") != "lowprio":
                raise RuntimeError(f"E117 job {job_id} did not enter lowprio")
            if field(after, "Reason") != "JobHeldUser":
                raise RuntimeError(f"E117 job {job_id} lost its hold")
            records[job_id]["after_held"] = after
        payload = {
            "schema": "e117s1_lowprio_partition_repair_v1",
            "protocol": str(protocol_path),
            "protocol_sha256": e117.digest(protocol_path),
            "application": str(Path(__file__).resolve()),
            "application_sha256": e117.digest(Path(__file__)),
            "ledger": str(ledger_path),
            "ledger_sha256": e117.digest(ledger_path),
            "exact_job_ids": job_ids,
            "from_partition": "mltheory",
            "to_partition": "lowprio",
            "account": "mltheory",
            "scientific_environment_changed": False,
            "outcomes_inspected": False,
            "pointmaze": "excluded",
            "records": [
                {
                    "job_id": job_id,
                    "node": records[job_id]["node"],
                    "environment_sha256": records[job_id]["environment_sha256"],
                    "before": records[job_id]["before"],
                    "held": records[job_id]["held"],
                    "after_held": records[job_id]["after_held"],
                }
                for job_id in job_ids
            ],
            "installed": False,
            "released": False,
        }
        e117.e111.e81.atomic_json(artifact_path, payload)
        for job_id in job_ids:
            run(["scontrol", "release", str(job_id)])
        for row in payload["records"]:
            current = show(int(row["job_id"]))
            if field(current, "Partition") != "lowprio":
                raise RuntimeError(
                    f"E117 job {row['job_id']} left lowprio after release"
                )
            if environment(current) != records[int(row["job_id"])]["environment"]:
                raise RuntimeError(
                    f"E117 job {row['job_id']} environment changed after release"
                )
            row["after_release"] = current
        payload["installed"] = True
        payload["released"] = True
        e117.e111.e81.atomic_json(artifact_path, payload)
    except Exception:
        for job_id in changed:
            subprocess.run(
                [
                    "scontrol",
                    "update",
                    f"JobId={job_id}",
                    "Partition=mltheory",
                    "Account=mltheory",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
        for job_id in job_ids:
            subprocess.run(
                ["scontrol", "release", str(job_id)],
                capture_output=True,
                text=True,
                check=False,
            )
        if artifact_path.exists():
            artifact_path.unlink()
        raise
    print(
        f"[e117s1] jobs={len(job_ids)} partition=lowprio "
        f"released={len(job_ids)} artifact={artifact_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
