#!/usr/bin/env python3
"""Apply the transactional E117-R1-S2 drained-node repair."""

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


PROTOCOL = "paper/preregistration/e117r1s2_drained_node022_repair_20260824.md"
ARTIFACT = "var/artifacts/e117r1s2_drained_node022_repair.json"
APPLICATION = Path(__file__).resolve()
AMENDMENT_NAME = "E117-R1-S2"
SCHEMA = "e117r1s2_drained_node022_repair_v1"
SOURCE_NODE = "node022"
TARGET_NODES = {
    "python_factors": "node101",
    "mathir": "node203",
}
EXPECTED_JOB_IDS = tuple(range(30873701, 30873707))
LOG_LABEL = "e117r1s2"


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"command failed {command}: {result.stderr.strip()}")
    return result


def audit(
    record: str,
    *,
    node: str,
    expected_environment: str,
    held: bool,
) -> None:
    required = {
        "JobState": "PENDING",
        "RunTime": "00:00:00",
        "Restarts": "0",
        "Partition": "lowprio",
        "Account": "mltheory",
        "ReqNodeList": node,
        "ReqTRES": "cpu=8,mem=64G,node=1,billing=2,gres/gpu=1",
        "TresPerNode": "gres/gpu:1",
        "TimeLimit": "08:00:00",
    }
    failures = {
        key: (scheduler.field(record, key), expected)
        for key, expected in required.items()
        if scheduler.field(record, key) != expected
    }
    if held and scheduler.field(record, "Reason") != "JobHeldUser":
        failures["Reason"] = (
            scheduler.field(record, "Reason"),
            "JobHeldUser",
        )
    if scheduler.environment(record) != expected_environment:
        failures["environment_sha256"] = (
            scheduler.sha256_text(scheduler.environment(record)),
            scheduler.sha256_text(expected_environment),
        )
    if failures:
        raise RuntimeError(f"{AMENDMENT_NAME} identity drifted: {failures}")


def main() -> int:
    parser = argparse.ArgumentParser(description=f"Apply {AMENDMENT_NAME}.")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.apply == args.check:
        raise SystemExit(f"pass exactly one of --check or --apply for {AMENDMENT_NAME}")
    root = e117.repo_root()
    ledger_path = root / e117r1.LEDGER
    protocol_path = root / PROTOCOL
    artifact_path = root / ARTIFACT
    if artifact_path.exists():
        raise SystemExit(f"refusing duplicate {AMENDMENT_NAME}: {artifact_path}")
    if not protocol_path.is_file():
        raise SystemExit(f"{AMENDMENT_NAME} protocol is absent: {protocol_path}")
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    targets = [
        row for row in ledger.get("runs", []) if str(row["domain"]) in TARGET_NODES
    ]
    expected_ids = list(EXPECTED_JOB_IDS)
    if len(targets) != len(expected_ids) or ledger.get("released") is not True:
        raise SystemExit("E117-R1 target ledger is incomplete")
    if sorted(int(row["job_id"]) for row in targets) != expected_ids:
        raise SystemExit(f"{AMENDMENT_NAME} exact job set drifted")

    records: dict[int, dict[str, str]] = {}
    for row in targets:
        job_id = int(row["job_id"])
        before = scheduler.show(job_id)
        expected_environment = scheduler.environment(str(row["held_scheduler_record"]))
        audit(
            before,
            node=SOURCE_NODE,
            expected_environment=expected_environment,
            held=False,
        )
        if Path(str(row["run_dir"])).exists():
            raise SystemExit(f"E117-R1 job {job_id} already has a run directory")
        records[job_id] = {
            "domain": str(row["domain"]),
            "environment": expected_environment,
            "before": before,
        }
    if args.check:
        print(
            f"[{LOG_LABEL}] preflight_passed=True jobs={len(expected_ids)} "
            f"source={SOURCE_NODE}"
        )
        return 0

    changed: list[int] = []
    try:
        for job_id in expected_ids:
            run(["scontrol", "uhold", str(job_id)])
        for job_id in expected_ids:
            held_record = scheduler.show(job_id)
            audit(
                held_record,
                node=SOURCE_NODE,
                expected_environment=records[job_id]["environment"],
                held=True,
            )
            records[job_id]["held"] = held_record
        for job_id in expected_ids:
            target_node = TARGET_NODES[records[job_id]["domain"]]
            run(
                [
                    "scontrol",
                    "update",
                    f"JobId={job_id}",
                    f"NodeList={target_node}",
                ]
            )
            changed.append(job_id)
        for job_id in expected_ids:
            target_node = TARGET_NODES[records[job_id]["domain"]]
            after = scheduler.show(job_id)
            audit(
                after,
                node=target_node,
                expected_environment=records[job_id]["environment"],
                held=True,
            )
            records[job_id]["after_held"] = after
        payload = {
            "schema": SCHEMA,
            "protocol": str(protocol_path),
            "protocol_sha256": e117.digest(protocol_path),
            "application": str(APPLICATION),
            "application_sha256": e117.digest(APPLICATION),
            "ledger": str(ledger_path),
            "ledger_sha256_before": e117.digest(ledger_path),
            "exact_job_ids": expected_ids,
            "source_node": SOURCE_NODE,
            "target_nodes": TARGET_NODES,
            "previous_scheduler_amendment": ledger.get("scheduler_amendment"),
            "scientific_environment_changed": False,
            "outcomes_inspected": False,
            "pointmaze": "excluded",
            "records": [
                {
                    "job_id": job_id,
                    "domain": records[job_id]["domain"],
                    "target_node": TARGET_NODES[records[job_id]["domain"]],
                    "environment_sha256": scheduler.sha256_text(
                        records[job_id]["environment"]
                    ),
                    "before": records[job_id]["before"],
                    "held": records[job_id]["held"],
                    "after_held": records[job_id]["after_held"],
                }
                for job_id in expected_ids
            ],
            "installed": False,
        }
        e117.e111.e81.atomic_json(artifact_path, payload)
        for job_id in expected_ids:
            run(["scontrol", "release", str(job_id)])
        for row in payload["records"]:
            current = scheduler.show(int(row["job_id"]))
            if scheduler.field(current, "ReqNodeList") != row["target_node"]:
                raise RuntimeError(
                    f"E117-R1 job {row['job_id']} lost its repaired node"
                )
            if (
                scheduler.environment(current)
                != records[int(row["job_id"])]["environment"]
            ):
                raise RuntimeError(f"E117-R1 job {row['job_id']} environment changed")
            row["after_release"] = current
        payload["installed"] = True
        amendment_history = list(ledger.get("scheduler_amendment_history", []))
        previous_amendment = ledger.get("scheduler_amendment")
        if previous_amendment and previous_amendment not in amendment_history:
            amendment_history.append(previous_amendment)
        amendment_history.append(str(protocol_path))
        ledger["scheduler_amendment"] = str(protocol_path)
        ledger["scheduler_amendment_sha256"] = e117.digest(protocol_path)
        ledger["scheduler_amendment_history"] = amendment_history
        effective_nodes = dict(ledger.get("effective_nodes") or {})
        for row in targets:
            effective_nodes[f"{row['scale']}/{row['domain']}"] = TARGET_NODES[
                str(row["domain"])
            ]
        ledger["effective_nodes"] = effective_nodes
        e117.e111.e81.atomic_json(artifact_path, payload)
        e117.e111.e81.atomic_json(ledger_path, ledger)
    except Exception:
        for job_id in changed:
            subprocess.run(
                [
                    "scontrol",
                    "update",
                    f"JobId={job_id}",
                    f"NodeList={SOURCE_NODE}",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
        for job_id in expected_ids:
            subprocess.run(
                ["scontrol", "release", str(job_id)],
                capture_output=True,
                text=True,
                check=False,
            )
        if artifact_path.exists():
            artifact_path.unlink()
        raise
    target_summary = " ".join(
        f"{node}={sum(row['target_node'] == node for row in payload['records'])}"
        for node in sorted(set(TARGET_NODES.values()))
    )
    print(
        f"[{LOG_LABEL}] jobs={len(expected_ids)} released={len(expected_ids)} "
        f"{target_summary}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
