#!/usr/bin/env python3
"""Recover two empty E100 Pantry pool failures and classify Python/s58."""

from __future__ import annotations

import argparse
import copy
import json
import shlex
from pathlib import Path
from typing import Any

import repair_e100_sparse_rlep_gates as base


ROOT = Path(__file__).resolve().parents[2]
PRIMARY = ROOT / "var/artifacts/e100_sparse_rlep_dr_falcon1b_jobs.json"
R1 = ROOT / "var/artifacts/e100_sparse_rlep_execution_repair_jobs.json"
RECOVERY = ROOT / "var/artifacts/e100_pantry_infrastructure_recovery_jobs.json"
PROTOCOL = ROOT / "paper/preregistration/e100r2_pantry_infrastructure_recovery_20260819.md"
SMOKE_RECEIPT = ROOT / "var/data/e100_sparse_rlep_smoke_graph_s55/E100_SMOKE_COMPLETE.json"
PY58_AUDIT_LOG = ROOT / "var/artifacts/logs/e100-python-audit-s58-30572782.err"

RETRY = {
    55: (30572795, 30572796, 30572827),
    58: (30572801, 30572802, 30572830),
}
PASSED_HELD = {
    56: (30572797, 30572798, 30572828),
    57: (30572799, 30572800, 30572829),
    59: (30572803, 30572804, 30572831),
}
PY58_BLOCK = (30572781, 30572782, 30572820)
NODE_SET = "node205,node207"


def records(primary: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {
        int(row["seed"]): row
        for row in primary["collection"]["records"]
        if str(row["domain"]) == "pantry_plan"
    }


def runs(primary: dict[str, Any], domain: str) -> dict[int, dict[str, Any]]:
    return {
        int(row["seed"]): row
        for row in primary["runs"]
        if str(row["domain"]) == domain
    }


def empty_tree(path: Path) -> bool:
    return not path.exists() or not any(item.is_file() for item in path.rglob("*"))


def retry_collection_command(
    record: dict[str, Any], seed: int, dependency: int | None
) -> list[str]:
    source = base.submit_line(str(record["held_collection_scheduler_record"]))
    command: list[str] = []
    for token in source:
        if token == "--hold" or token.startswith("--nodelist=") or token.startswith("--dependency="):
            continue
        if token.startswith("--job-name="):
            command.append(f"--job-name=e100-pantry-pool-s{seed}-r2")
        elif token.startswith("--nice="):
            command.append("--nice=0")
        else:
            command.append(token)
    if command[:2] != ["sbatch", "--parsable"]:
        raise RuntimeError("Pantry retry lost sbatch safeguards")
    command.insert(2, "--hold")
    if dependency is not None:
        command.insert(-1, f"--dependency=afterany:{dependency}")
    command.insert(-1, f"--nodelist={NODE_SET}")
    return command


def smoke_reaudit_command() -> list[str]:
    command = base.smoke_reaudit_command()
    return [
        "--job-name=e100-rlep-smoke-reaudit-r2"
        if token.startswith("--job-name=") else token
        for token in command
    ]


def verify_passed_cells(
    pool_records: dict[int, dict[str, Any]], science_runs: dict[int, dict[str, Any]]
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for seed, expected in PASSED_HELD.items():
        record, science = pool_records[seed], science_runs[seed]
        observed = (
            int(record["collection_job_id"]),
            int(record["audit_job_id"]),
            int(science["job_id"]),
        )
        if observed != expected:
            raise RuntimeError(f"Pantry/s{seed}: identity drift {observed}")
        if base.scheduler_state(expected[0]) != "COMPLETED" or base.scheduler_state(expected[1]) != "COMPLETED":
            raise RuntimeError(f"Pantry/s{seed}: pool gate is not terminal-success")
        scheduler = base.scheduler_record(expected[2])
        if "JobState=PENDING" not in scheduler or "Reason=JobHeldUser" in scheduler:
            raise RuntimeError(f"Pantry/s{seed}: released science job is not runnable")
        receipt = Path(str(record["pool_root"])) / "RLEP_SPARSE_POOL_COMPLETE.json"
        payload = json.loads(receipt.read_text(encoding="utf-8"))
        if (int(payload["prompts"]), int(payload["eligible_prompts"]), int(payload["ineligible_prompts"])) != (384, 228, 156):
            raise RuntimeError(f"Pantry/s{seed}: frozen receipt drift")
        output.append({
            "seed": seed,
            "collection_job_id": expected[0],
            "audit_job_id": expected[1],
            "released_science_job_id": expected[2],
            "receipt": str(receipt),
            "receipt_sha256": base.digest(receipt),
            "scheduler_record_after_release": scheduler,
        })
    return output


def classify_python_s58(primary: dict[str, Any]) -> dict[str, Any]:
    python_runs = runs(primary, "python_factors")
    science = python_runs[58]
    python_records = {
        int(row["seed"]): row for row in primary["collection"]["records"]
        if str(row["domain"]) == "python_factors"
    }
    record = python_records[58]
    observed = (
        int(record["collection_job_id"]), int(record["audit_job_id"]),
        int(science["job_id"]),
    )
    if observed != PY58_BLOCK:
        raise RuntimeError(f"Python/s58: identity drift {observed}")
    if base.scheduler_state(PY58_BLOCK[0]) != "COMPLETED" or base.scheduler_state(PY58_BLOCK[1]) != "FAILED":
        raise RuntimeError("Python/s58: frozen pool/audit state drift")
    log = PY58_AUDIT_LOG.read_text(encoding="utf-8")
    marker = "sparse RLEP pool has no replay-eligible prompt"
    if marker not in log:
        raise RuntimeError("Python/s58: zero-eligibility audit marker missing")
    blocked = copy.deepcopy(science)
    blocked.update({
        "blocked_because": (
            "frozen sparse-RLEP pool has zero replay-eligible prompts; "
            f"hard-gate audit job {PY58_BLOCK[1]} failed"
        ),
        "collection_job_id": PY58_BLOCK[0],
        "failed_pool_audit_job_id": PY58_BLOCK[1],
        "unstarted_cancelled_job_id": PY58_BLOCK[2],
    })
    return blocked


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    for required in (PRIMARY, R1, PROTOCOL, SMOKE_RECEIPT, PY58_AUDIT_LOG):
        if not required.is_file():
            raise SystemExit(f"missing required E100-R2 input: {required}")
    if RECOVERY.exists():
        raise SystemExit(f"refusing duplicate E100-R2 recovery: {RECOVERY}")
    if not bool(json.loads(R1.read_text(encoding="utf-8"))["released"]):
        raise SystemExit("E100-R1 repair is not released")
    if base.scheduler_state(30580929) != "COMPLETED":
        raise SystemExit("historical E100 smoke re-audit is not complete")

    primary = json.loads(PRIMARY.read_text(encoding="utf-8"))
    original = copy.deepcopy(primary)
    original_hash = base.digest(PRIMARY)
    pool_records = records(primary)
    science_runs = runs(primary, "pantry_plan")
    passed = verify_passed_cells(pool_records, science_runs)
    python_s58 = classify_python_s58(primary)

    for seed, expected in RETRY.items():
        record, science = pool_records[seed], science_runs[seed]
        observed = (
            int(record["collection_job_id"]), int(record["audit_job_id"]),
            int(science["job_id"]),
        )
        if observed != expected or base.scheduler_state(expected[0]) != "FAILED":
            raise SystemExit(f"Pantry/s{seed}: failed collection identity/state drift")
        if not empty_tree(Path(str(record["pool_root"]))) or not empty_tree(Path(str(science["run_dir"]))):
            raise SystemExit(f"Pantry/s{seed}: retry target contains an artifact")

    if not args.submit:
        first = retry_collection_command(pool_records[55], 55, None)
        print(shlex.join(first))
        print(shlex.join(retry_collection_command(pool_records[58], 58, 55555555)))
        print(shlex.join(base.audit_command("pantry_plan", 55, Path(str(pool_records[55]["pool_root"])), 55555555)))
        print(shlex.join(smoke_reaudit_command()))
        print("<two science replacements depend on fresh pool audit + fresh smoke audit>")
        print("<three already-gated Pantry science jobs verified released; Python/s58 classified blocked>")
        return 0

    submitted: list[int] = []
    try:
        retry_rows: list[dict[str, Any]] = []
        previous_collection: int | None = None
        for seed in (55, 58):
            record = pool_records[seed]
            collection_cmd = retry_collection_command(record, seed, previous_collection)
            collection_id = base.submit_held(collection_cmd)
            submitted.append(collection_id)
            expected = [
                f"JobName=e100-pantry-pool-s{seed}-r2",
                "ReqNodeList=node[205,207]",
                "gres/gpu:a6000=1",
                f"SAVE_PATH={record['pool_root']}",
                f"OAT_ZERO_EVAL_MODE_COVERAGE_SEED={1000400 + seed}",
                f"OAT_ZERO_SOURCE_ROOT={base.SNAPSHOT / 'src'}",
            ]
            if previous_collection is not None:
                expected.append(f"Dependency=afterany:{previous_collection}")
            collection_held = base.audit_held(collection_id, tuple(expected))
            audit_id = base.submit_held(base.audit_command(
                "pantry_plan", seed, Path(str(record["pool_root"])), collection_id
            ))
            submitted.append(audit_id)
            audit_held = base.audit_held(audit_id, (
                f"Dependency=afterok:{collection_id}",
                str(record["pool_root"]), "--allow-sparse",
            ))
            retry_rows.append({
                "seed": seed,
                "original_collection_job_id": RETRY[seed][0],
                "original_audit_job_id": RETRY[seed][1],
                "original_science_job_id": RETRY[seed][2],
                "replacement_collection_job_id": collection_id,
                "replacement_audit_job_id": audit_id,
                "held_collection_scheduler_record": collection_held,
                "held_audit_scheduler_record": audit_held,
            })
            previous_collection = collection_id

        smoke_id = base.submit_held(smoke_reaudit_command())
        submitted.append(smoke_id)
        smoke_held = base.audit_held(smoke_id, (
            "JobName=e100-rlep-smoke-reaudit-r2",
            str(base.SMOKE_ROOT), "--expected-terminal-step", "32",
        ))

        science_rows: list[dict[str, Any]] = []
        for retry in retry_rows:
            seed = int(retry["seed"])
            science = science_runs[seed]
            science_id = base.submit_held(base.science_command(
                science, int(retry["replacement_audit_job_id"]), smoke_id
            ))
            submitted.append(science_id)
            science_held = base.audit_held(science_id, (
                f"Dependency=afterok:{int(retry['replacement_audit_job_id'])}",
                f"afterok:{smoke_id}",
                f"OAT_ZERO_RLEP_EXPERIENCE_ROOT={science['pool_root']}",
                "OAT_ZERO_RLEP_REPLAY_COUNT=2",
                "OAT_ZERO_RLEP_SPARSE_FALLBACK=1",
            ))
            science_rows.append({
                "seed": seed,
                "original_science_job_id": RETRY[seed][2],
                "replacement_science_job_id": science_id,
                "replacement_pool_audit_job_id": int(retry["replacement_audit_job_id"]),
                "replacement_smoke_audit_job_id": smoke_id,
                "held_scheduler_record": science_held,
            })

        for retry in retry_rows:
            seed = int(retry["seed"])
            record = pool_records[seed]
            record["collection_job_id"] = int(retry["replacement_collection_job_id"])
            record["audit_job_id"] = int(retry["replacement_audit_job_id"])
            record["held_collection_scheduler_record"] = retry["held_collection_scheduler_record"]
            record["held_audit_scheduler_record"] = retry["held_audit_scheduler_record"]
            record["execution_repair_r2"] = {
                "original_collection_job_id": RETRY[seed][0],
                "original_collection_state": "FAILED (transient GPU OOM before draws)",
                "original_audit_job_id": RETRY[seed][1],
                "replacement_collection_job_id": int(retry["replacement_collection_job_id"]),
                "replacement_audit_job_id": int(retry["replacement_audit_job_id"]),
                "frozen_configuration": True,
            }
            science = science_runs[seed]
            replacement = next(row for row in science_rows if int(row["seed"]) == seed)
            science["job_id"] = int(replacement["replacement_science_job_id"])
            science["pool_audit_dependency_job_id"] = int(replacement["replacement_pool_audit_job_id"])
            science["smoke_audit_dependency_job_id"] = smoke_id
            science["held_scheduler_record"] = replacement["held_scheduler_record"]
            science["execution_repair_r2"] = {
                "original_science_job_id": RETRY[seed][2],
                "replacement_science_job_id": int(replacement["replacement_science_job_id"]),
            }

        primary["runs"] = [
            row for row in primary["runs"]
            if not (str(row["domain"]) == "python_factors" and int(row["seed"]) == 58)
        ]
        if not any(
            str(row["domain"]) == "python_factors" and int(row["seed"]) == 58
            for row in primary.get("blocked_runs", [])
        ):
            primary.setdefault("blocked_runs", []).append(python_s58)
        primary["execution_amendment_r2"] = {
            "protocol": str(PROTOCOL),
            "protocol_sha256": base.digest(PROTOCOL),
            "recovery_ledger": str(RECOVERY),
            "executable_cells": len(primary["runs"]),
            "blocked_pool_gate_cells": len(primary.get("blocked_runs", [])),
            "released_existing_pantry_science_job_ids": sorted(PASSED_HELD[seed][2] for seed in PASSED_HELD),
        }

        payload = {
            "schema": "e100_pantry_infrastructure_recovery_v1",
            "cohort": "e100-r2",
            "released": False,
            "protocol": str(PROTOCOL),
            "protocol_sha256": base.digest(PROTOCOL),
            "launcher_sha256": base.digest(Path(__file__)),
            "primary_ledger": str(PRIMARY),
            "primary_ledger_sha256_before_recovery": original_hash,
            "r1_ledger": str(R1),
            "r1_ledger_sha256": base.digest(R1),
            "snapshot_root": str(base.SNAPSHOT),
            "released_existing_pantry_cells": passed,
            "pantry_retries": retry_rows,
            "science_replacements": science_rows,
            "python_s58_blocked_run": python_s58,
            "smoke_reaudit": {
                "job_id": smoke_id,
                "receipt": str(SMOKE_RECEIPT),
                "receipt_sha256": base.digest(SMOKE_RECEIPT),
                "held_scheduler_record": smoke_held,
            },
        }
        base.atomic_json(PRIMARY, primary)
        base.atomic_json(RECOVERY, payload)
        base.release(submitted)
        payload["released"] = True
        base.atomic_json(RECOVERY, payload)
    except Exception:
        base.cancel(submitted)
        base.atomic_json(PRIMARY, original)
        if RECOVERY.exists():
            RECOVERY.unlink()
        raise

    print(json.dumps({
        "released_existing_science_jobs": sorted(PASSED_HELD[seed][2] for seed in PASSED_HELD),
        "replacement_science_jobs": {
            str(row["seed"]): row["replacement_science_job_id"] for row in science_rows
        },
        "python_s58": "blocked_zero_eligibility",
        "released_new_jobs": len(submitted),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
