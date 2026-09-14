#!/usr/bin/env python3
"""Right-size host memory for five pending E113-R4 Qwen-0.5B jobs."""

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
    "e113r4r2s9_qwen05b_memory_rightsizing_20260830.md"
)
ARTIFACT = common.ROOT / (
    "var/artifacts/e113r4r2s9_qwen05b_memory_rightsizing.json"
)
APPLICATION = Path(__file__).resolve()
TARGET_IDS = tuple(range(30977262, 30977267))
TARGET_SEEDS = tuple(range(43, 48))
PRIOR_QWEN_PYTHON_IDS = tuple(range(30869121, 30869126))
OLD_MEMORY = "128G"
NEW_MEMORY = "96G"
OLD_MEMORY_MIB = str(128 * 1024)
NEW_MEMORY_MIB = str(96 * 1024)
QWEN_COMPLETED_MAX_KIB = 64 * 1024 * 1024
QWEN_PYTHON_MAX_KIB = 40 * 1024 * 1024


def run(
    command: list[str], *, check: bool = True
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if check and result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"command failed {command}: {detail}")
    return result


def normalize_state(value: str) -> str:
    return value.split()[0].split("+")[0]


def memory_kib(value: str) -> int:
    value = value.strip()
    if not value:
        raise RuntimeError("missing MaxRSS")
    suffix = value[-1].upper()
    scales = {"K": 1, "M": 1024, "G": 1024 * 1024, "T": 1024 * 1024 * 1024}
    if suffix in scales:
        return int(float(value[:-1]) * scales[suffix])
    return int(value)


def accounting(job_id: int) -> dict[str, Any]:
    output = run(
        [
            "sacct", "-j", str(job_id), "-n", "-P", "-o",
            "JobIDRaw,JobName,State,Elapsed,ExitCode,ReqMem,MaxRSS",
        ]
    ).stdout
    rows = [line.split("|") for line in output.splitlines() if line.strip()]
    parent = next((row for row in rows if row[0] == str(job_id)), None)
    batch = next((row for row in rows if row[0] == f"{job_id}.batch"), None)
    if parent is None or batch is None or len(parent) < 7 or len(batch) < 7:
        raise RuntimeError(f"incomplete accounting for job {job_id}: {output!r}")
    return {
        "job_id": job_id,
        "job_name": parent[1],
        "state": normalize_state(parent[2]),
        "elapsed": parent[3],
        "exit_code": parent[4],
        "requested_memory": parent[5],
        "batch_max_rss": batch[6],
        "batch_max_rss_kib": memory_kib(batch[6]),
        "raw": output.strip(),
    }


def req_tres(record: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for token in release.field(record, "ReqTRES").split(","):
        key, value = token.split("=", 1)
        result[key] = value
    return result


def validate(
    job_id: int,
    record: str,
    *,
    expected_environment: str,
    memory: str,
    held: bool,
    allow_running: bool = False,
) -> None:
    failures: dict[str, Any] = {}
    allowed_states = {"PENDING", "RUNNING"} if allow_running else {"PENDING"}
    if release.field(record, "JobState") not in allowed_states:
        failures["JobState"] = (
            release.field(record, "JobState"),
            sorted(allowed_states),
        )
    if not allow_running and release.field(record, "RunTime") != "00:00:00":
        failures["RunTime"] = release.field(record, "RunTime")
    if held and release.field(record, "Reason") != "JobHeldUser":
        failures["Reason"] = (release.field(record, "Reason"), "JobHeldUser")
    if not held and release.field(record, "Reason") == "JobHeldUser":
        failures["Reason"] = "JobHeldUser"
    required = {
        "Restarts": "0",
        "Partition": "all",
        "Account": "mltheory",
        "QOS": "long",
        "Dependency": "(null)",
        "Requeue": "1",
        "ReqNodeList": "(null)",
        "NumCPUs": "16",
        "MinMemoryNode": memory,
        "TresPerNode": "gres/gpu:a6000:1",
        "TimeLimit": "12:00:00",
        "Nice": "0",
    }
    for key, expected in required.items():
        observed = release.field(record, key)
        if observed != expected:
            failures[key] = (observed, expected)
    tres = req_tres(record)
    expected_tres = {
        "cpu": "16",
        "mem": memory,
        "node": "1",
        "gres/gpu": "1",
        "gres/gpu:a6000": "1",
    }
    for key, expected in expected_tres.items():
        if tres.get(key) != expected:
            failures[f"ReqTRES.{key}"] = (tres.get(key), expected)
    if release.environment(record) != expected_environment:
        failures["environment"] = "scientific export drift"
    if failures:
        raise RuntimeError(f"E113-R4-R2-S9 job {job_id} drifted: {failures}")


def memory_evidence(
    rows: dict[int, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    completed: list[dict[str, Any]] = []
    for job_id, row in sorted(rows.items()):
        if row.get("family") != "qwen05b" or job_id in TARGET_IDS:
            continue
        receipt = Path(str(row["run_dir"])) / "TRAINING_COMPLETE.json"
        if not receipt.is_file():
            raise RuntimeError(f"Qwen completion receipt absent for job {job_id}")
        item = accounting(job_id)
        if item["state"] != "COMPLETED" or item["exit_code"] != "0:0":
            raise RuntimeError(f"Qwen completion accounting drifted: {item}")
        if int(item["batch_max_rss_kib"]) >= QWEN_COMPLETED_MAX_KIB:
            raise RuntimeError(f"Qwen completed MaxRSS does not license 96G: {item}")
        item["completion_receipt"] = str(receipt)
        item["completion_receipt_sha256"] = launch.digest(receipt)
        completed.append(item)
    if len(completed) != 20:
        raise RuntimeError(f"expected 20 completed Qwen cells, found {len(completed)}")

    python_attempts: list[dict[str, Any]] = []
    for job_id in PRIOR_QWEN_PYTHON_IDS:
        item = accounting(job_id)
        if int(item["batch_max_rss_kib"]) >= QWEN_PYTHON_MAX_KIB:
            raise RuntimeError(f"prior Qwen/Python MaxRSS does not license 96G: {item}")
        python_attempts.append(item)
    return completed, python_attempts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if args.check == args.apply:
        raise SystemExit("pass exactly one of --check or --apply")
    if not PROTOCOL.is_file():
        raise SystemExit(f"memory-rightsizing protocol is absent: {PROTOCOL}")
    if ARTIFACT.exists():
        raise SystemExit(f"refusing duplicate E113-R4-R2-S9: {ARTIFACT}")

    ledger = release.load_ledger()
    rows = {int(row["job_id"]): row for row in ledger.get("runs", [])}
    if len(rows) != 50:
        raise RuntimeError("authoritative E113-R4 ledger does not contain 50 cells")
    target_rows = [rows.get(job_id) for job_id in TARGET_IDS]
    if any(row is None for row in target_rows):
        raise RuntimeError("authoritative E113-R4 ledger lacks an S9 target")
    identities = {
        (str(row["family"]), str(row["domain"]), int(row["seed"]))
        for row in target_rows
        if row is not None
    }
    expected_identities = {
        ("qwen05b", "python_factors", seed) for seed in TARGET_SEEDS
    }
    if identities != expected_identities:
        raise RuntimeError(f"S9 target identities drifted: {identities}")

    completed_evidence, python_evidence = memory_evidence(rows)
    records: dict[int, dict[str, Any]] = {}
    for job_id in TARGET_IDS:
        row = rows[job_id]
        expected_environment = release.environment(str(row["held_scheduler_record"]))
        before = release.show(job_id)
        validate(
            job_id,
            before,
            expected_environment=expected_environment,
            memory=OLD_MEMORY,
            held=False,
        )
        receipt = Path(str(row["run_dir"])) / "TRAINING_COMPLETE.json"
        if receipt.exists():
            raise RuntimeError(f"pending S9 target has a completion receipt: {receipt}")
        records[job_id] = {
            "family": str(row["family"]),
            "domain": str(row["domain"]),
            "seed": int(row["seed"]),
            "run_dir": str(row["run_dir"]),
            "environment": expected_environment,
            "before": before,
        }

    if args.check:
        qwen_peak = max(int(item["batch_max_rss_kib"]) for item in completed_evidence)
        python_peak = max(int(item["batch_max_rss_kib"]) for item in python_evidence)
        print(
            f"[e113r4r2s9] preflight_passed=True jobs={len(TARGET_IDS)} "
            f"qwen_peak_kib={qwen_peak} python_peak_kib={python_peak}"
        )
        return 0

    changed: list[int] = []
    artifact_written = False
    released = False
    try:
        run(["scontrol", "uhold", *[str(job_id) for job_id in TARGET_IDS]])
        for job_id in TARGET_IDS:
            held_record = release.show(job_id)
            validate(
                job_id,
                held_record,
                expected_environment=records[job_id]["environment"],
                memory=OLD_MEMORY,
                held=True,
            )
            records[job_id]["held"] = held_record

        for job_id in TARGET_IDS:
            run(
                [
                    "scontrol", "update", f"JobId={job_id}",
                    f"MinMemoryNode={NEW_MEMORY_MIB}",
                ]
            )
            changed.append(job_id)
        for job_id in TARGET_IDS:
            after_held = release.show(job_id)
            validate(
                job_id,
                after_held,
                expected_environment=records[job_id]["environment"],
                memory=NEW_MEMORY,
                held=True,
            )
            records[job_id]["after_held"] = after_held

        payload: dict[str, Any] = {
            "schema": "e113r4r2s9_qwen05b_memory_rightsizing_v1",
            "recorded_at": datetime.now(timezone.utc).astimezone().isoformat(),
            "authorization": (
                "User explicitly requested the proposed DAPO host-memory "
                "rightsizing on 2026-08-30."
            ),
            "protocol": str(PROTOCOL),
            "protocol_sha256": launch.digest(PROTOCOL),
            "application": str(APPLICATION),
            "application_sha256": launch.digest(APPLICATION),
            "ledger": str(common.LEDGER),
            "ledger_sha256": launch.digest(common.LEDGER),
            "exact_job_ids": list(TARGET_IDS),
            "old_memory": OLD_MEMORY,
            "new_memory": NEW_MEMORY,
            "completed_qwen_memory_evidence": completed_evidence,
            "prior_qwen_python_memory_evidence": python_evidence,
            "falcon_jobs_excluded": list(range(30977252, 30977262))
            + list(range(30978749, 30978754)),
            "scientific_environment_changed": False,
            "gpu_request_changed": False,
            "cpu_request_changed": False,
            "incomplete_endpoint_values_inspected": False,
            "records": [
                {
                    "job_id": job_id,
                    "family": records[job_id]["family"],
                    "domain": records[job_id]["domain"],
                    "seed": records[job_id]["seed"],
                    "run_dir": records[job_id]["run_dir"],
                    "before": records[job_id]["before"],
                    "held": records[job_id]["held"],
                    "after_held": records[job_id]["after_held"],
                }
                for job_id in TARGET_IDS
            ],
            "installed": False,
        }
        launch.atomic_json(ARTIFACT, payload)
        artifact_written = True

        run(["scontrol", "release", *[str(job_id) for job_id in TARGET_IDS]])
        released = True
        for row in payload["records"]:
            job_id = int(row["job_id"])
            after_release = release.show(job_id)
            validate(
                job_id,
                after_release,
                expected_environment=records[job_id]["environment"],
                memory=NEW_MEMORY,
                held=False,
                allow_running=True,
            )
            row["after_release"] = after_release
        payload["installed"] = True
        launch.atomic_json(ARTIFACT, payload)
    except Exception:
        if not released:
            for job_id in changed:
                run(
                    [
                        "scontrol", "update", f"JobId={job_id}",
                        f"MinMemoryNode={OLD_MEMORY_MIB}",
                    ],
                    check=False,
                )
            run(
                ["scontrol", "release", *[str(job_id) for job_id in TARGET_IDS]],
                check=False,
            )
            if artifact_written and ARTIFACT.exists():
                ARTIFACT.unlink()
        raise

    states = {
        release.field(str(row["after_release"]), "JobState")
        for row in payload["records"]
    }
    print(
        f"[e113r4r2s9] installed=True jobs={len(TARGET_IDS)} "
        f"memory={OLD_MEMORY}->{NEW_MEMORY} states={','.join(sorted(states))}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
