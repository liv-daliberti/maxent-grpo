#!/usr/bin/env python3
"""Apply the audited E117/DAPO scheduler-continuity amendment."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import apply_e117s1_lowprio_partition_repair as scheduler  # noqa: E402


PROTOCOL = ROOT / (
    "paper/preregistration/"
    "e117r1s4_e113r4r2s2_scheduler_continuity_20260826.md"
)
ARTIFACT = ROOT / (
    "var/artifacts/e117r1s4_e113r4r2s2_scheduler_continuity.json"
)
E117_LEDGER = ROOT / "var/artifacts/e117r1_same_plumbing_component_preflight_jobs.json"
DAPO_LEDGER = ROOT / "var/artifacts/e113r4_official_verl_dapo_jobs.json"
E117_IDS = tuple(range(30873695, 30873707))
DAPO_COMPLETE_IDS = (30869111, 30869112, 30869113, 30869114)
DAPO_PENDING_IDS = tuple(range(30869115, 30869161))
E117_OLD_PARTITION = "lowprio"
E117_NEW_PARTITION = "all"
E117_OLD_TIME = "08:00:00"
E117_NEW_TIME = "06:00:00"
DAPO_OLD_TIME = "7-00:00:00"
DAPO_NEW_TIME = "12:00:00"


def run(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if check and result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"command failed {command}: {detail}")
    return result


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(payload: dict[str, Any]) -> None:
    temporary = ARTIFACT.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(ARTIFACT)


def load_runs(path: Path) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload, {int(row["job_id"]): row for row in payload.get("runs", [])}


def validate_environment(
    job_id: int, record: str, expected_sha256: str, run_dir: str
) -> None:
    environment = scheduler.environment(record)
    observed = scheduler.sha256_text(environment)
    failures: dict[str, Any] = {}
    if observed != expected_sha256:
        failures["environment_sha256"] = (observed, expected_sha256)
    if f"SAVE_PATH={run_dir}" not in environment and f"E113R4_OUTPUT={run_dir}" not in environment:
        failures["run_dir"] = ("missing from export", run_dir)
    if failures:
        raise RuntimeError(f"job {job_id} scientific export drifted: {failures}")


def validate_e117(
    job_id: int,
    record: str,
    row: dict[str, Any],
    effective_nodes: dict[str, str],
    *,
    partition: str,
    time_limit: str,
) -> None:
    node_key = f"{row['scale']}/{row['domain']}"
    expected = {
        "JobState": "PENDING",
        "RunTime": "00:00:00",
        "Restarts": "0",
        "Partition": partition,
        "Account": "mltheory",
        "ReqNodeList": effective_nodes[node_key],
        "NumCPUs": "8",
        "MinMemoryNode": "64G",
        "TimeLimit": time_limit,
        "Nice": "0",
        "TresPerNode": "gres/gpu:1",
        "Dependency": "(null)",
    }
    failures = {
        key: (scheduler.field(record, key), value)
        for key, value in expected.items()
        if scheduler.field(record, key) != value
    }
    if failures:
        raise RuntimeError(f"E117 job {job_id} drifted: {failures}")
    validate_environment(
        job_id,
        record,
        str(row["scientific_environment_sha256"]),
        str(row["run_dir"]),
    )
    if Path(str(row["run_dir"])).exists():
        raise RuntimeError(f"zero-runtime E117 job {job_id} has a run directory")


def dapo_expected_environment(row: dict[str, Any]) -> str:
    # The post-amendment record predates R4-R2's frozen vLLM-scheduler
    # recovery; this held record is the live byte-exact science surface.
    return scheduler.environment(str(row["held_scheduler_record"]))


def validate_dapo(
    job_id: int,
    record: str,
    row: dict[str, Any],
    *,
    time_limit: str,
) -> None:
    expected = {
        "JobState": "PENDING",
        "RunTime": "00:00:00",
        "Restarts": "0",
        "Partition": "all",
        "Account": "allcs",
        "QOS": "long",
        "NumCPUs": "16",
        "MinMemoryNode": "128G",
        "TimeLimit": time_limit,
        "Nice": "0",
        "TresPerNode": "gres/gpu:a6000:1",
        "Dependency": "(null)",
    }
    failures = {
        key: (scheduler.field(record, key), value)
        for key, value in expected.items()
        if scheduler.field(record, key) != value
    }
    if failures:
        raise RuntimeError(f"DAPO job {job_id} drifted: {failures}")
    expected_env = dapo_expected_environment(row)
    validate_environment(
        job_id,
        record,
        scheduler.sha256_text(expected_env),
        str(row["run_dir"]),
    )
    if Path(str(row["run_dir"])).exists():
        raise RuntimeError(f"zero-runtime DAPO job {job_id} has a run directory")


def validate_completed_dapo(rows: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for job_id in DAPO_COMPLETE_IDS:
        row = rows[job_id]
        accounting = run(
            [
                "sacct", "-X", "-j", str(job_id), "-n", "-P",
                "-o", "JobIDRaw,JobName,State,Elapsed,Start,End,ExitCode,NodeList,ReqMem,Timelimit",
            ]
        ).stdout.strip()
        fields = accounting.split("|")
        if len(fields) < 7 or fields[2] != "COMPLETED" or fields[6] != "0:0":
            raise RuntimeError(f"DAPO runtime evidence drifted for {job_id}: {accounting}")
        receipt_path = Path(str(row["run_dir"])) / "TRAINING_COMPLETE.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        evidence.append(
            {
                "job_id": job_id,
                "accounting_record": accounting,
                "completion_receipt": str(receipt_path),
                "completion_receipt_sha256": digest(receipt_path),
            }
        )
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if not PROTOCOL.is_file():
        raise SystemExit(f"missing protocol: {PROTOCOL}")
    if ARTIFACT.exists():
        raise SystemExit(f"refusing duplicate amendment: {ARTIFACT}")

    e117_payload, e117 = load_runs(E117_LEDGER)
    dapo_payload, dapo = load_runs(DAPO_LEDGER)
    if e117_payload.get("released") is not True or set(e117) != set(E117_IDS):
        raise RuntimeError("E117 release ledger drifted")
    if dapo_payload.get("released") is not True or len(dapo) != 50:
        raise RuntimeError("DAPO release ledger drifted")
    effective_nodes = {
        str(key): str(value) for key, value in e117_payload["effective_nodes"].items()
    }
    completed_evidence = validate_completed_dapo(dapo)
    e117_before = {job_id: scheduler.show(job_id) for job_id in E117_IDS}
    dapo_before = {job_id: scheduler.show(job_id) for job_id in DAPO_PENDING_IDS}
    for job_id in E117_IDS:
        validate_e117(
            job_id, e117_before[job_id], e117[job_id], effective_nodes,
            partition=E117_OLD_PARTITION, time_limit=E117_OLD_TIME,
        )
    for job_id in DAPO_PENDING_IDS:
        validate_dapo(
            job_id, dapo_before[job_id], dapo[job_id], time_limit=DAPO_OLD_TIME
        )
    if not args.apply:
        print(
            f"[dry-run] E117 jobs={len(E117_IDS)} partition={E117_NEW_PARTITION} "
            f"time={E117_NEW_TIME}; DAPO jobs={len(DAPO_PENDING_IDS)} "
            f"time={DAPO_NEW_TIME}"
        )
        return 0

    changed_e117: list[int] = []
    changed_dapo: list[int] = []
    try:
        for job_id in E117_IDS:
            run(
                [
                    "scontrol", "update", f"JobId={job_id}",
                    f"Partition={E117_NEW_PARTITION}",
                    f"TimeLimit={E117_NEW_TIME}",
                ]
            )
            changed_e117.append(job_id)
        for job_id in DAPO_PENDING_IDS:
            run(
                [
                    "scontrol", "update", f"JobId={job_id}",
                    f"TimeLimit={DAPO_NEW_TIME}",
                ]
            )
            changed_dapo.append(job_id)
        e117_after = {job_id: scheduler.show(job_id) for job_id in E117_IDS}
        dapo_after = {job_id: scheduler.show(job_id) for job_id in DAPO_PENDING_IDS}
        for job_id in E117_IDS:
            validate_e117(
                job_id, e117_after[job_id], e117[job_id], effective_nodes,
                partition=E117_NEW_PARTITION, time_limit=E117_NEW_TIME,
            )
        for job_id in DAPO_PENDING_IDS:
            validate_dapo(
                job_id, dapo_after[job_id], dapo[job_id], time_limit=DAPO_NEW_TIME
            )
    except Exception:
        for job_id in changed_e117:
            run(
                [
                    "scontrol", "update", f"JobId={job_id}",
                    f"Partition={E117_OLD_PARTITION}",
                    f"TimeLimit={E117_OLD_TIME}",
                ],
                check=False,
            )
        for job_id in changed_dapo:
            run(
                ["scontrol", "update", f"JobId={job_id}", f"TimeLimit={DAPO_OLD_TIME}"],
                check=False,
            )
        raise

    payload = {
        "schema": "e117r1s4_e113r4r2s2_scheduler_continuity_v1",
        "applied_at": datetime.now(timezone.utc).isoformat(),
        "protocol": str(PROTOCOL.relative_to(ROOT)),
        "protocol_sha256": digest(PROTOCOL),
        "application": str(Path(__file__).resolve().relative_to(ROOT)),
        "application_sha256": digest(Path(__file__).resolve()),
        "source_ledgers": {
            str(E117_LEDGER.relative_to(ROOT)): digest(E117_LEDGER),
            str(DAPO_LEDGER.relative_to(ROOT)): digest(DAPO_LEDGER),
        },
        "scheduler_only": True,
        "scientific_environment_changed": False,
        "outcomes_inspected": False,
        "pointmaze": "excluded",
        "holds_used": False,
        "e117": {
            "job_ids": list(E117_IDS),
            "old_partition": E117_OLD_PARTITION,
            "new_partition": E117_NEW_PARTITION,
            "old_time_limit": E117_OLD_TIME,
            "new_time_limit": E117_NEW_TIME,
            "records": [
                {
                    "job_id": job_id,
                    "environment_sha256": e117[job_id]["scientific_environment_sha256"],
                    "before_scheduler_record": e117_before[job_id],
                    "after_scheduler_record": e117_after[job_id],
                }
                for job_id in E117_IDS
            ],
        },
        "dapo": {
            "completed_job_ids_untouched": list(DAPO_COMPLETE_IDS),
            "completed_runtime_evidence": completed_evidence,
            "pending_job_ids": list(DAPO_PENDING_IDS),
            "old_time_limit": DAPO_OLD_TIME,
            "new_time_limit": DAPO_NEW_TIME,
            "records": [
                {
                    "job_id": job_id,
                    "environment_sha256": scheduler.sha256_text(
                        dapo_expected_environment(dapo[job_id])
                    ),
                    "before_scheduler_record": dapo_before[job_id],
                    "after_scheduler_record": dapo_after[job_id],
                }
                for job_id in DAPO_PENDING_IDS
            ],
        },
        "installed": True,
    }
    atomic_json(payload)
    print(
        f"[applied] E117={len(E117_IDS)} partition={E117_NEW_PARTITION} "
        f"time={E117_NEW_TIME}; DAPO={len(DAPO_PENDING_IDS)} time={DAPO_NEW_TIME}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
