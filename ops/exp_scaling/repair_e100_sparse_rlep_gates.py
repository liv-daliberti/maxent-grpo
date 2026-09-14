#!/usr/bin/env python3
"""Repair E100 infrastructure gates without changing frozen pool outcomes."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PRIMARY = ROOT / "var/artifacts/e100_sparse_rlep_dr_falcon1b_jobs.json"
REPAIR = ROOT / "var/artifacts/e100_sparse_rlep_execution_repair_jobs.json"
PROTOCOL = ROOT / "paper/preregistration/e100_sparse_rlep_execution_repair_20260814.md"
SNAPSHOT = ROOT / "var/artifacts/source_snapshots/e100_sparse_rlep_falcon1b_89c7b7a475e2e459"
SMOKE_AUDIT_ID = 30572806
SMOKE_ROOT = ROOT / "var/data/e100_sparse_rlep_smoke_graph_s55"

RECOVERED = {
    ("graph_coloring", 58): (30572761, 30572762, 30572810),
    ("countdown", 57): (30572769, 30572770, 30572814),
    ("mathir", 55): (30572785, 30572786, 30572822),
    ("mathir", 58): (30572791, 30572792, 30572825),
}
RETRY = ("python_factors", 57)
RETRY_OLD = (30572779, 30572780, 30572819)
BLOCKED = {
    ("python_factors", 56): (30572777, 30572778, 30572818),
    ("python_factors", 59): (30572783, 30572784, 30572821),
}
ALREADY_RELEASED = (
    30572807, 30572808, 30572809, 30572811,
    30572812, 30572813, 30572815, 30572816,
    30572817, 30572823, 30572824, 30572826, 30572829,
)
PROMOTED_POOL_GRAPH = (
    30572781, 30572782, 30572797, 30572798,
    30572801, 30572802, 30572803, 30572804,
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=False, capture_output=True, text=True)


def scheduler_state(job_id: int) -> str:
    result = run([
        "sacct", "-n", "-X", "-j", str(job_id),
        "--format=State", "--parsable2",
    ])
    states = [line.split("|", 1)[0].split("+", 1)[0].strip()
              for line in result.stdout.splitlines() if line.strip()]
    if not states:
        raise RuntimeError(f"cannot resolve scheduler state for {job_id}")
    return states[0]


def scheduler_record(job_id: int) -> str:
    result = run(["scontrol", "show", "job", "-dd", "-o", str(job_id)])
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or f"cannot inspect {job_id}")
    return result.stdout.strip()


def submit_held(command: list[str]) -> int:
    result = run(command)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "E100 repair submission failed")
    return int(result.stdout.strip().split(";", 1)[0])


def audit_held(job_id: int, expected: tuple[str, ...]) -> str:
    record = scheduler_record(job_id)
    required = ("JobState=PENDING", "Reason=JobHeldUser", *expected)
    missing = [item for item in required if item not in record]
    if missing:
        raise RuntimeError(f"repair job {job_id} lacks {missing}")
    return record


def record_index(primary: dict[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    return {
        (str(row["domain"]), int(row["seed"])): row
        for row in primary["collection"]["records"]
    }


def run_index(primary: dict[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    return {
        (str(row["domain"]), int(row["seed"])): row
        for row in primary["runs"]
    }


def verify_recovered(primary: dict[str, Any]) -> list[dict[str, Any]]:
    records = record_index(primary)
    runs = run_index(primary)
    output: list[dict[str, Any]] = []
    for key, (collection_id, audit_id, science_id) in RECOVERED.items():
        record, science = records[key], runs[key]
        if (int(record["collection_job_id"]), int(record["audit_job_id"]),
                int(science["job_id"])) != (collection_id, audit_id, science_id):
            raise RuntimeError(f"{key}: E100 job identity drift")
        receipt = Path(record["pool_root"]) / "RLEP_SPARSE_POOL_COMPLETE.json"
        payload = json.loads(receipt.read_text(encoding="utf-8"))
        if int(payload["prompts"]) != 384 or int(payload["eligible_prompts"]) <= 0:
            raise RuntimeError(f"{key}: recovered pool fails frozen hard gate")
        if scheduler_state(collection_id) != "FAILED":
            raise RuntimeError(f"{key}: original collection state drifted")
        output.append({
            "domain": key[0], "seed": key[1],
            "original_collection_job_id": collection_id,
            "original_audit_job_id": audit_id,
            "science_job_id": science_id,
            "pool_root": str(record["pool_root"]),
            "receipt": str(receipt), "receipt_sha256": digest(receipt),
            "eligible_prompts": int(payload["eligible_prompts"]),
            "ineligible_prompts": int(payload["ineligible_prompts"]),
        })
    return output


def audit_command(domain: str, seed: int, pool: Path, dependency: int | None) -> list[str]:
    python = ROOT / "var/seed_paper_eval/paper310/bin/python"
    script = SNAPSHOT / "ops/exp_scaling/audit_e98_rlep_pool.py"
    command = [
        "sbatch", "--parsable", "--hold",
        f"--job-name=e100-{domain[:6]}-audit-s{seed}-rr",
        f"--export=ALL,PYTHONPATH={SNAPSHOT / 'src'}",
        "--partition=all", "--account=allcs", "--cpus-per-task=2",
        "--mem=8G", "--time=00:15:00", "--nice=0",
        f"--output={ROOT / 'var/artifacts/logs'}/%x-%j.out",
        f"--error={ROOT / 'var/artifacts/logs'}/%x-%j.err",
    ]
    if dependency is not None:
        command.append(f"--dependency=afterok:{dependency}")
    command.extend([
        "--wrap",
        shlex.join([
            str(python), str(script), "--pool-root", str(pool),
            "--expected-prompts", "384", "--allow-sparse",
        ]),
    ])
    return command


def smoke_reaudit_command() -> list[str]:
    python = ROOT / "var/seed_paper_eval/paper310/bin/python"
    script = SNAPSHOT / "ops/exp_scaling/audit_e100_sparse_rlep_smoke.py"
    return [
        "sbatch", "--parsable", "--hold",
        "--job-name=e100-rlep-smoke-reaudit",
        f"--export=ALL,PYTHONPATH={SNAPSHOT / 'src'}",
        "--partition=all", "--account=allcs", "--cpus-per-task=2",
        "--mem=8G", "--time=00:15:00", "--nice=0",
        f"--output={ROOT / 'var/artifacts/logs'}/%x-%j.out",
        f"--error={ROOT / 'var/artifacts/logs'}/%x-%j.err",
        "--wrap",
        shlex.join([
            str(python), str(script), "--run-root", str(SMOKE_ROOT),
            "--expected-terminal-step", "32",
        ]),
    ]


def submit_line(record: str) -> list[str]:
    if " SubmitLine=" not in record or " WorkDir=" not in record:
        raise RuntimeError("scheduler record lacks SubmitLine")
    return shlex.split(record.split(" SubmitLine=", 1)[1].split(" WorkDir=", 1)[0])


def retry_command(record: dict[str, Any]) -> list[str]:
    source = submit_line(str(record["held_collection_scheduler_record"]))
    command: list[str] = []
    for token in source:
        if token == "--hold" or token.startswith("--nodelist="):
            continue
        if token.startswith("--job-name="):
            command.append("--job-name=e100-python-pool-s57-rr")
        elif token.startswith("--nice="):
            command.append("--nice=0")
        else:
            command.append(token)
    if command[:2] != ["sbatch", "--parsable"]:
        raise RuntimeError("Python/s57 retry lost sbatch safeguards")
    command.insert(2, "--hold")
    command.insert(-1, "--nodelist=node205,node207")
    return command


def science_command(
    run_record: dict[str, Any], audit_id: int, smoke_audit_id: int
) -> list[str]:
    source = submit_line(str(run_record["held_scheduler_record"]))
    gpu = str(run_record["gpu"])
    nodes = "node202,node203,node204" if gpu == "a5000" else "node205,node207"
    command: list[str] = []
    for token in source:
        if (token == "--hold" or token.startswith("--dependency=")
                or token.startswith("--nodelist=")):
            continue
        if token.startswith("--job-name="):
            command.append(token + "-rr")
        elif token.startswith("--nice="):
            command.append("--nice=0")
        else:
            command.append(token)
    if command[:2] != ["sbatch", "--parsable"]:
        raise RuntimeError("science replacement lost sbatch safeguards")
    command.insert(2, "--hold")
    command.insert(-1, f"--dependency=afterok:{audit_id}:{smoke_audit_id}")
    command.insert(-1, f"--nodelist={nodes}")
    return command


def cancel(job_ids: list[int]) -> None:
    if not job_ids:
        return
    result = run(["scancel", *(str(job_id) for job_id in job_ids)])
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "cannot cancel obsolete jobs")


def release(job_ids: list[int]) -> None:
    for job_id in job_ids:
        result = run(["scontrol", "release", str(job_id)])
        if result.returncode:
            raise RuntimeError(result.stderr.strip() or f"cannot release {job_id}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    if not PRIMARY.is_file() or not PROTOCOL.is_file() or not SNAPSHOT.is_dir():
        raise SystemExit("E100 primary ledger, protocol, or snapshot is missing")
    if REPAIR.exists():
        raise SystemExit(f"refusing duplicate E100 repair: {REPAIR}")

    primary = json.loads(PRIMARY.read_text(encoding="utf-8"))
    original = copy.deepcopy(primary)
    original_hash = digest(PRIMARY)
    recovered = verify_recovered(primary)
    records = record_index(primary)
    runs = run_index(primary)

    retry_record = records[RETRY]
    retry_run = runs[RETRY]
    if (int(retry_record["collection_job_id"]), int(retry_record["audit_job_id"]),
            int(retry_run["job_id"])) != RETRY_OLD:
        raise SystemExit("Python/s57 job identity drifted")
    retry_root = Path(retry_record["pool_root"])
    if any(path.is_file() for path in retry_root.rglob("*")):
        raise SystemExit("Python/s57 retry root unexpectedly contains an artifact")
    if scheduler_state(RETRY_OLD[0]) != "TIMEOUT":
        raise SystemExit("Python/s57 original collection is not TIMEOUT")

    for key, (_pool_id, audit_id, science_id) in BLOCKED.items():
        if scheduler_state(audit_id) != "FAILED":
            raise SystemExit(f"{key}: hard-gate audit state drifted")
        if int(runs[key]["job_id"]) != science_id:
            raise SystemExit(f"{key}: science job identity drifted")
    if scheduler_state(SMOKE_AUDIT_ID) != "COMPLETED":
        raise SystemExit("E100 global smoke audit is not complete")
    smoke_receipt = SMOKE_ROOT / "E100_SMOKE_COMPLETE.json"
    if not smoke_receipt.is_file():
        raise SystemExit("E100 global smoke receipt is missing")

    retry_cmd = retry_command(retry_record)
    if not args.submit:
        for row in recovered:
            print(shlex.join(audit_command(
                str(row["domain"]), int(row["seed"]),
                Path(str(row["pool_root"])), None,
            )))
        print(shlex.join(retry_cmd))
        print(shlex.join(smoke_reaudit_command()))
        print("<Python/s57 replacement audit depends on replacement pool>")
        print("<affected science jobs depend on replacement audit + smoke audit>")
        return 0

    submitted: list[int] = []
    try:
        recovered_audits: list[dict[str, Any]] = []
        for row in recovered:
            audit_id = submit_held(audit_command(
                str(row["domain"]), int(row["seed"]),
                Path(str(row["pool_root"])), None,
            ))
            submitted.append(audit_id)
            held = audit_held(audit_id, (
                "JobName=e100-", str(row["pool_root"]), "--allow-sparse",
            ))
            recovered_audits.append({**row, "replacement_audit_job_id": audit_id,
                                     "held_scheduler_record": held})

        retry_id = submit_held(retry_cmd)
        submitted.append(retry_id)
        retry_held = audit_held(retry_id, (
            "JobName=e100-python-pool-s57-rr", "ReqNodeList=node[205,207]",
            "gres/gpu:a6000=1", "OAT_ZERO_EVAL_MODE_COVERAGE_SEED=1000257",
            f"SAVE_PATH={retry_root}", f"OAT_ZERO_SOURCE_ROOT={SNAPSHOT / 'src'}",
        ))
        retry_audit_id = submit_held(audit_command(
            RETRY[0], RETRY[1], retry_root, retry_id,
        ))
        submitted.append(retry_audit_id)
        retry_audit_held = audit_held(retry_audit_id, (
            f"Dependency=afterok:{retry_id}", str(retry_root), "--allow-sparse",
        ))

        smoke_reaudit_id = submit_held(smoke_reaudit_command())
        submitted.append(smoke_reaudit_id)
        smoke_reaudit_held = audit_held(smoke_reaudit_id, (
            "JobName=e100-rlep-smoke-reaudit", str(SMOKE_ROOT),
            "--expected-terminal-step", "32",
        ))

        audit_by_key = {
            (str(row["domain"]), int(row["seed"])):
                int(row["replacement_audit_job_id"])
            for row in recovered_audits
        }
        audit_by_key[RETRY] = retry_audit_id
        science_replacements: list[dict[str, Any]] = []
        for key, audit_id in audit_by_key.items():
            old_run = runs[key]
            job_id = submit_held(
                science_command(old_run, audit_id, smoke_reaudit_id)
            )
            submitted.append(job_id)
            normalized_nodes = (
                "node[202-204]" if str(old_run["gpu"]) == "a5000"
                else "node[205,207]"
            )
            held = audit_held(job_id, (
                "JobName=e100-",
                f"Dependency=afterok:{audit_id}",
                f"afterok:{smoke_reaudit_id}",
                f"ReqNodeList={normalized_nodes}",
                f"OAT_ZERO_RLEP_EXPERIENCE_ROOT={old_run['pool_root']}",
                "OAT_ZERO_RLEP_REPLAY_COUNT=2",
                "OAT_ZERO_RLEP_SPARSE_FALLBACK=1",
            ))
            science_replacements.append({
                "domain": key[0], "seed": key[1],
                "original_science_job_id": int(old_run["job_id"]),
                "replacement_science_job_id": job_id,
                "replacement_pool_audit_job_id": audit_id,
                "replacement_smoke_audit_job_id": smoke_reaudit_id,
                "held_scheduler_record": held,
            })

        blocked_runs: list[dict[str, Any]] = []
        blocked_ids: list[int] = []
        for key, (pool_id, audit_id, science_id) in BLOCKED.items():
            row = copy.deepcopy(runs[key])
            row.update({
                "blocked_because": (
                    "frozen sparse-RLEP pool has zero replay-eligible prompts; "
                    f"hard-gate audit job {audit_id} failed"
                ),
                "collection_job_id": pool_id,
                "failed_pool_audit_job_id": audit_id,
                "unstarted_cancelled_job_id": science_id,
            })
            blocked_runs.append(row)
            blocked_ids.append(science_id)

        obsolete_audits = [value[1] for value in RECOVERED.values()]
        obsolete_audits.append(RETRY_OLD[1])
        replaced_science_ids = [
            int(row["original_science_job_id"])
            for row in science_replacements
        ]
        cancel(obsolete_audits + blocked_ids + replaced_science_ids)

        for row in recovered_audits:
            key = (str(row["domain"]), int(row["seed"]))
            record = records[key]
            record["execution_repair"] = {
                "original_collection_job_id": int(row["original_collection_job_id"]),
                "original_collection_state": "FAILED (exit 137 during teardown)",
                "original_audit_job_id": int(row["original_audit_job_id"]),
                "replacement_audit_job_id": int(row["replacement_audit_job_id"]),
                "reused_immutable_pool": True,
                "receipt_sha256": str(row["receipt_sha256"]),
            }
            record["audit_job_id"] = int(row["replacement_audit_job_id"])
        retry_record["execution_repair"] = {
            "original_collection_job_id": RETRY_OLD[0],
            "original_collection_state": "TIMEOUT",
            "original_audit_job_id": RETRY_OLD[1],
            "replacement_collection_job_id": retry_id,
            "replacement_audit_job_id": retry_audit_id,
            "frozen_configuration": True,
        }
        retry_record["collection_job_id"] = retry_id
        retry_record["audit_job_id"] = retry_audit_id
        retry_record["held_collection_scheduler_record"] = retry_held
        retry_record["held_audit_scheduler_record"] = retry_audit_held

        for key, audit_id in audit_by_key.items():
            row = runs[key]
            replacement = next(
                item for item in science_replacements
                if (str(item["domain"]), int(item["seed"])) == key
            )
            original_science_id = int(row["job_id"])
            row["job_id"] = int(replacement["replacement_science_job_id"])
            row["pool_audit_dependency_job_id"] = audit_id
            row["held_scheduler_record"] = replacement["held_scheduler_record"]
            row["execution_repair"] = {
                "original_science_job_id": original_science_id,
                "previous_pool_audit_dependency_job_id": (
                    RETRY_OLD[1] if key == RETRY else RECOVERED[key][1]
                ),
                "replacement_pool_audit_dependency_job_id": audit_id,
                "historical_smoke_audit_job_id": SMOKE_AUDIT_ID,
                "replacement_smoke_audit_job_id": smoke_reaudit_id,
                "replacement_science_job_id": int(row["job_id"]),
            }

        primary["runs"] = [
            row for row in primary["runs"]
            if (str(row["domain"]), int(row["seed"])) not in BLOCKED
        ]
        primary["blocked_runs"] = blocked_runs
        primary["execution_amendment"] = {
            "protocol": str(PROTOCOL), "protocol_sha256": digest(PROTOCOL),
            "repair_ledger": str(REPAIR),
            "executable_cells": len(primary["runs"]),
            "blocked_pool_gate_cells": len(blocked_runs),
            "normal_priority_already_gated_jobs": list(ALREADY_RELEASED),
            "normal_priority_pending_pool_graph": list(PROMOTED_POOL_GRAPH),
        }

        repair_payload = {
            "schema": "e100_sparse_rlep_execution_repair_v1",
            "cohort": "e100", "released": False,
            "protocol": str(PROTOCOL), "protocol_sha256": digest(PROTOCOL),
            "primary_ledger": str(PRIMARY),
            "original_primary_ledger_sha256": original_hash,
            "snapshot_root": str(SNAPSHOT), "smoke_audit_job_id": SMOKE_AUDIT_ID,
            "smoke_reaudit": {
                "historical_audit_job_id": SMOKE_AUDIT_ID,
                "replacement_audit_job_id": smoke_reaudit_id,
                "receipt": str(smoke_receipt),
                "receipt_sha256": digest(smoke_receipt),
                "held_scheduler_record": smoke_reaudit_held,
            },
            "recovered_pools": recovered_audits,
            "python_s57_retry": {
                "original_collection_job_id": RETRY_OLD[0],
                "original_audit_job_id": RETRY_OLD[1],
                "science_job_id": RETRY_OLD[2],
                "replacement_collection_job_id": retry_id,
                "replacement_audit_job_id": retry_audit_id,
                "held_collection_scheduler_record": retry_held,
                "held_audit_scheduler_record": retry_audit_held,
            },
            "science_replacements": science_replacements,
            "blocked_runs": blocked_runs,
            "already_released_science_job_ids": list(ALREADY_RELEASED),
            "promoted_pending_pool_graph_job_ids": list(PROMOTED_POOL_GRAPH),
            "obsolete_audit_job_ids_cancelled": obsolete_audits,
        }
        atomic_json(PRIMARY, primary)
        atomic_json(REPAIR, repair_payload)

        release(submitted)
        repair_payload["released"] = True
        atomic_json(REPAIR, repair_payload)
    except Exception:
        cancel(submitted)
        if not REPAIR.exists():
            atomic_json(PRIMARY, original)
        raise

    print(json.dumps({
        "recovered_audits": {
            f"{row['domain']}/s{row['seed']}": row["replacement_audit_job_id"]
            for row in recovered_audits
        },
        "python_s57_retry": retry_id,
        "python_s57_audit": retry_audit_id,
        "smoke_reaudit": smoke_reaudit_id,
        "released_science_cells": len(ALREADY_RELEASED) + len(audit_by_key),
        "blocked_cells": [f"{key[0]}/s{key[1]}" for key in BLOCKED],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
