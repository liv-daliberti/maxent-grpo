#!/usr/bin/env python3
"""Apply the transactional E117-R1-S6 zero-step capacity repair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import apply_e117s1_lowprio_partition_repair as scheduler  # noqa: E402
import launch_e117_same_plumbing_component_preflight as e117  # noqa: E402
import launch_e117r1_zero_step_lowprio_replacement as e117r1  # noqa: E402


PROTOCOL = "paper/preregistration/e117r1s6_zero_step_capacity_repair_20260829.md"
ARTIFACT = "var/artifacts/e117r1s6_zero_step_capacity_repair.json"
APPLICATION = Path(__file__).resolve()
AMENDMENT_NAME = "E117-R1-S6"
SCHEMA = "e117r1s6_zero_step_capacity_repair_v1"
AUDIT_JOB_ID = 30874713
TARGETS = {
    "countdown": {
        "source": "node103",
        "target": "node202",
        "job_ids": (30873695, 30873696, 30873697),
    },
    "graph_coloring": {
        "source": "node104",
        "target": "node203",
        "job_ids": (30873698, 30873699, 30873700),
    },
    "python_factors": {
        "source": "node101",
        "target": "node203",
        "job_ids": (30873701, 30873702, 30873703),
    },
}
EXPECTED_JOB_IDS = tuple(
    job_id
    for domain in ("countdown", "graph_coloring", "python_factors")
    for job_id in TARGETS[domain]["job_ids"]
)


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"command failed {command}: {result.stderr.strip()}")
    return result


def show_node(node: str) -> str:
    result = run(["scontrol", "show", "node", "--oneliner", node])
    record = result.stdout.strip()
    if not record.startswith(f"NodeName={node} "):
        raise RuntimeError(f"missing scheduler record for {node}: {record!r}")
    return record


def int_field(record: str, name: str) -> int:
    value = scheduler.field(record, name)
    try:
        return int(value)
    except ValueError as error:
        raise RuntimeError(f"invalid {name} in scheduler record: {value!r}") from error


def allocated_gpu_count(record: str) -> int:
    allocated = scheduler.field(record, "AllocTRES")
    for item in allocated.split(","):
        if item.startswith("gres/gpu="):
            return int(item.split("=", 1)[1])
    return 0


def audit_node(record: str, *, node: str, required_jobs: int) -> None:
    failures: dict[str, Any] = {}
    state = scheduler.field(record, "State")
    for forbidden in ("DRAIN", "DOWN", "FAIL"):
        if forbidden in state:
            failures["State"] = state
    if scheduler.field(record, "Gres") != "gpu:a5000:10":
        failures["Gres"] = scheduler.field(record, "Gres")
    if "all" not in scheduler.field(record, "Partitions").split(","):
        failures["Partitions"] = scheduler.field(record, "Partitions")
    free_gpus = 10 - allocated_gpu_count(record)
    free_cpus = int_field(record, "CPUTot") - int_field(record, "CPUAlloc")
    free_memory_mib = int_field(record, "RealMemory") - int_field(record, "AllocMem")
    if free_gpus < required_jobs:
        failures["free_gpus"] = free_gpus
    if free_cpus < required_jobs * 8:
        failures["free_cpus"] = free_cpus
    if free_memory_mib < required_jobs * 64 * 1024:
        failures["free_memory_mib"] = free_memory_mib
    if failures:
        raise RuntimeError(f"{AMENDMENT_NAME} target {node} drifted: {failures}")


def audit_job(
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
        "Partition": "all",
        "Account": "mltheory",
        "ReqNodeList": node,
        "ReqTRES": "cpu=8,mem=64G,node=1,billing=2,gres/gpu=1",
        "TresPerNode": "gres/gpu:1",
        "TimeLimit": "06:00:00",
    }
    failures: dict[str, Any] = {
        key: (scheduler.field(record, key), expected)
        for key, expected in required.items()
        if scheduler.field(record, key) != expected
    }
    if held and scheduler.field(record, "Reason") != "JobHeldUser":
        failures["Reason"] = (scheduler.field(record, "Reason"), "JobHeldUser")
    if scheduler.environment(record) != expected_environment:
        failures["environment_sha256"] = (
            scheduler.sha256_text(scheduler.environment(record)),
            scheduler.sha256_text(expected_environment),
        )
    if failures:
        raise RuntimeError(f"{AMENDMENT_NAME} job identity drifted: {failures}")


def audit_dependency() -> str:
    record = scheduler.show(AUDIT_JOB_ID)
    dependency = scheduler.field(record, "Dependency")
    failures: dict[str, Any] = {}
    if scheduler.field(record, "JobState") != "PENDING":
        failures["JobState"] = scheduler.field(record, "JobState")
    if scheduler.field(record, "Reason") != "Dependency":
        failures["Reason"] = scheduler.field(record, "Reason")
    missing = [job_id for job_id in EXPECTED_JOB_IDS if str(job_id) not in dependency]
    if missing:
        failures["missing_dependencies"] = missing
    if failures:
        raise RuntimeError(f"{AMENDMENT_NAME} audit dependency drifted: {failures}")
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
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
    rows = {
        int(row["job_id"]): row
        for row in ledger.get("runs", [])
        if int(row["job_id"]) in EXPECTED_JOB_IDS
    }
    if ledger.get("released") is not True or sorted(rows) != list(EXPECTED_JOB_IDS):
        raise SystemExit("E117-R1 target ledger is incomplete")

    expected_nodes = {
        "qwen05b/countdown": "node103",
        "qwen05b/graph_coloring": "node104",
        "qwen05b/python_factors": "node101",
        "falcon1b/mathir": "node203",
    }
    if ledger.get("effective_nodes") != expected_nodes:
        raise SystemExit("E117-R1 effective-node map drifted before S6")

    node_records = {
        str(spec["target"]): show_node(str(spec["target"]))
        for spec in TARGETS.values()
    }
    target_counts = {
        node: sum(
            len(spec["job_ids"])
            for spec in TARGETS.values()
            if str(spec["target"]) == node
        )
        for node in node_records
    }
    for node, record in node_records.items():
        audit_node(record, node=node, required_jobs=target_counts[node])
    dependency_before = audit_dependency()

    records: dict[int, dict[str, str]] = {}
    for domain, spec in TARGETS.items():
        source = str(spec["source"])
        for job_id in spec["job_ids"]:
            row = rows[int(job_id)]
            if str(row["domain"]) != domain:
                raise SystemExit(f"E117-R1 job {job_id} domain drifted")
            expected_environment = scheduler.environment(
                str(row["held_scheduler_record"])
            )
            before = scheduler.show(int(job_id))
            audit_job(
                before,
                node=source,
                expected_environment=expected_environment,
                held=False,
            )
            if Path(str(row["run_dir"])).exists():
                raise SystemExit(f"E117-R1 job {job_id} already has a run directory")
            records[int(job_id)] = {
                "domain": domain,
                "source": source,
                "target": str(spec["target"]),
                "environment": expected_environment,
                "before": before,
            }

    if args.check:
        print(
            f"[e117r1s6] preflight_passed=True jobs={len(EXPECTED_JOB_IDS)} "
            f"targets={','.join(sorted(node_records))}"
        )
        return 0

    changed: list[int] = []
    released = False
    try:
        for job_id in EXPECTED_JOB_IDS:
            run(["scontrol", "uhold", str(job_id)])
        for job_id in EXPECTED_JOB_IDS:
            row = records[job_id]
            held_record = scheduler.show(job_id)
            audit_job(
                held_record,
                node=row["source"],
                expected_environment=row["environment"],
                held=True,
            )
            row["held"] = held_record

        for job_id in EXPECTED_JOB_IDS:
            row = records[job_id]
            run(
                [
                    "scontrol",
                    "update",
                    f"JobId={job_id}",
                    f"NodeList={row['target']}",
                ]
            )
            changed.append(job_id)
        for job_id in EXPECTED_JOB_IDS:
            row = records[job_id]
            after_held = scheduler.show(job_id)
            audit_job(
                after_held,
                node=row["target"],
                expected_environment=row["environment"],
                held=True,
            )
            row["after_held"] = after_held

        payload: dict[str, Any] = {
            "schema": SCHEMA,
            "protocol": str(protocol_path),
            "protocol_sha256": e117.digest(protocol_path),
            "application": str(APPLICATION),
            "application_sha256": e117.digest(APPLICATION),
            "ledger": str(ledger_path),
            "ledger_sha256_before": e117.digest(ledger_path),
            "exact_job_ids": list(EXPECTED_JOB_IDS),
            "target_nodes": {
                domain: str(spec["target"]) for domain, spec in TARGETS.items()
            },
            "node_capacity_before": node_records,
            "audit_job_id": AUDIT_JOB_ID,
            "audit_dependency_before": dependency_before,
            "scientific_environment_changed": False,
            "outcomes_inspected": False,
            "pointmaze": "excluded",
            "records": [
                {
                    "job_id": job_id,
                    "domain": records[job_id]["domain"],
                    "source_node": records[job_id]["source"],
                    "target_node": records[job_id]["target"],
                    "environment_sha256": scheduler.sha256_text(
                        records[job_id]["environment"]
                    ),
                    "before": records[job_id]["before"],
                    "held": records[job_id]["held"],
                    "after_held": records[job_id]["after_held"],
                }
                for job_id in EXPECTED_JOB_IDS
            ],
            "installed": False,
        }
        e117.e111.e81.atomic_json(artifact_path, payload)

        run(["scontrol", "release", *[str(job_id) for job_id in EXPECTED_JOB_IDS]])
        released = True
        for row in payload["records"]:
            current = scheduler.show(int(row["job_id"]))
            if scheduler.field(current, "ReqNodeList") != row["target_node"]:
                raise RuntimeError(
                    f"E117-R1 job {row['job_id']} lost its S6 target node"
                )
            if scheduler.environment(current) != records[int(row["job_id"])][
                "environment"
            ]:
                raise RuntimeError(f"E117-R1 job {row['job_id']} environment changed")
            if scheduler.field(current, "Reason") == "JobHeldUser":
                raise RuntimeError(f"E117-R1 job {row['job_id']} remained user-held")
            row["after_release"] = current

        payload["audit_dependency_after"] = audit_dependency()
        payload["installed"] = True
        amendment_history = list(ledger.get("scheduler_amendment_history", []))
        previous = ledger.get("scheduler_amendment")
        if previous and previous not in amendment_history:
            amendment_history.append(previous)
        amendment_history.append(str(protocol_path))
        ledger["scheduler_amendment"] = str(protocol_path)
        ledger["scheduler_amendment_sha256"] = e117.digest(protocol_path)
        ledger["scheduler_amendment_history"] = amendment_history
        effective_nodes = dict(ledger["effective_nodes"])
        for domain, spec in TARGETS.items():
            effective_nodes[f"qwen05b/{domain}"] = str(spec["target"])
        ledger["effective_nodes"] = effective_nodes
        e117.e111.e81.atomic_json(artifact_path, payload)
        e117.e111.e81.atomic_json(ledger_path, ledger)
    except Exception:
        if not released:
            for job_id in changed:
                row = records[job_id]
                subprocess.run(
                    [
                        "scontrol",
                        "update",
                        f"JobId={job_id}",
                        f"NodeList={row['source']}",
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                )
            for job_id in EXPECTED_JOB_IDS:
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
        f"[e117r1s6] jobs={len(EXPECTED_JOB_IDS)} released=True "
        "countdown=node202 graph_coloring=node203 python_factors=node203"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
