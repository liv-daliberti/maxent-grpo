#!/usr/bin/env python3
"""Submit the fail-closed E117 priority-fence restoration watcher."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / (
    "paper/preregistration/" "e117r2r1s1_priority_fence_restore_watcher_20260830.md"
)
FENCE = ROOT / "var/artifacts/e117r2r1s1_owner_backfill_priority_fence.json"
RESTORE = ROOT / ("ops/exp_scaling/apply_e117r2r1s1_owner_backfill_priority_fence.py")
ARTIFACT = ROOT / ("var/artifacts/e117r2r1s1_priority_fence_restore_watcher.json")
E117_IDS = tuple(range(30977267, 30977278))


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


def show(job_id: int) -> str:
    return run(["scontrol", "show", "job", "-dd", "-o", str(job_id)]).stdout.strip()


def field(record: str, key: str) -> str:
    match = re.search(rf"(?:^| ){re.escape(key)}=([^ ]*)", record)
    if not match:
        raise RuntimeError(f"watcher record lacks {key}")
    return match.group(1)


def validate_source() -> dict[str, Any]:
    payload = json.loads(FENCE.read_text(encoding="utf-8"))
    expected = {
        "schema": "e117r2r1s1_owner_backfill_priority_fence_v1",
        "installed": True,
        "restored": False,
        "e117_job_ids": list(E117_IDS),
    }
    failures = {
        key: (payload.get(key), value)
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if payload.get("application_sha256") != digest(RESTORE):
        failures["application_sha256"] = (
            payload.get("application_sha256"),
            digest(RESTORE),
        )
    if failures:
        raise RuntimeError(f"priority-fence source drifted: {failures}")
    return payload


def validate_watcher(record: str, *, held: bool) -> None:
    expected = {
        "JobName": "e117r2r1s1-fence-restore",
        "UserId": "od2961(363432)",
        "JobState": "PENDING",
        "Partition": "all",
        "Account": "allcs",
        "QOS": "none",
        "Requeue": "0",
        "NumCPUs": "1",
        "MinMemoryNode": "1G",
        "TimeLimit": "00:10:00",
    }
    failures = {
        key: (field(record, key), value)
        for key, value in expected.items()
        if field(record, key) != value
    }
    reason = field(record, "Reason")
    if held and reason != "JobHeldUser":
        failures["Reason"] = (reason, "JobHeldUser")
    if not held and reason not in {"Dependency", "Resources", "Priority", "None"}:
        failures["Reason"] = reason
    dependency = field(record, "Dependency")
    missing = [job_id for job_id in E117_IDS if str(job_id) not in dependency]
    if missing or "after:" not in dependency:
        failures["Dependency"] = (dependency, missing)
    if failures:
        raise RuntimeError(f"restoration watcher drifted: {failures}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if args.check == args.apply:
        raise SystemExit("pass exactly one of --check or --apply")
    if ARTIFACT.exists():
        raise SystemExit(f"refusing duplicate watcher: {ARTIFACT}")
    source = validate_source()
    dependency = "after:" + ":".join(str(job_id) for job_id in E117_IDS)
    command = [
        "sbatch",
        "--parsable",
        "--hold",
        "--no-requeue",
        "--job-name=e117r2r1s1-fence-restore",
        "--partition=all",
        "--account=allcs",
        "--qos=none",
        "--cpus-per-task=1",
        "--mem=1G",
        "--time=00:10:00",
        f"--dependency={dependency}",
        f"--output={ROOT}/var/artifacts/logs/e117r2r1s1-fence-restore-%j.out",
        f"--error={ROOT}/var/artifacts/logs/e117r2r1s1-fence-restore-%j.err",
        "--chdir=" + str(ROOT),
        "--wrap=" + f"{sys.executable} {RESTORE} --restore",
    ]
    if args.check:
        print(
            f"[check] dependencies={len(E117_IDS)} " f"restore_sha256={digest(RESTORE)}"
        )
        return 0

    job_id: int | None = None
    try:
        result = run(command)
        job_id = int(result.stdout.strip().split(";", 1)[0])
        held = show(job_id)
        validate_watcher(held, held=True)
        payload = {
            "schema": "e117r2r1s1_priority_fence_restore_watcher_v1",
            "submitted_at": datetime.now(timezone.utc).astimezone().isoformat(),
            "protocol": str(PROTOCOL.relative_to(ROOT)),
            "protocol_sha256": digest(PROTOCOL),
            "application": str(Path(__file__).resolve().relative_to(ROOT)),
            "application_sha256": digest(Path(__file__).resolve()),
            "fence_artifact": str(FENCE.relative_to(ROOT)),
            "fence_artifact_sha256": digest(FENCE),
            "restore_application": str(RESTORE.relative_to(ROOT)),
            "restore_application_sha256": digest(RESTORE),
            "e117_job_ids": list(E117_IDS),
            "watcher_job_id": job_id,
            "dependency": dependency,
            "held_scheduler_record": held,
            "released": False,
            "source_installed_at": source["installed_at"],
        }
        atomic_json(payload)
        run(["scontrol", "release", str(job_id)])
        released = show(job_id)
        validate_watcher(released, held=False)
        payload["released_scheduler_record"] = released
        payload["released"] = True
        atomic_json(payload)
    except Exception:
        if job_id is not None:
            run(["scancel", str(job_id)], check=False)
        if ARTIFACT.exists():
            ARTIFACT.unlink()
        raise
    print(f"[submitted] watcher={job_id} dependencies={len(E117_IDS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
