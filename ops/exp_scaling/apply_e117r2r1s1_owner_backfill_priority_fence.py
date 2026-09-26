#!/usr/bin/env python3
"""Install or restore E117-R2-R1-S1's owner backfill priority fence."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / (
    "paper/preregistration/" "e117r2r1s1_owner_backfill_priority_fence_20260830.md"
)
ARTIFACT = ROOT / ("var/artifacts/e117r2r1s1_owner_backfill_priority_fence.json")
RECOVERY = ROOT / "var/artifacts/e113r4_e117r2_signal53_wave_recovery.json"
E117_IDS = tuple(range(30977267, 30977278))
E113_IDS = tuple(range(30977240, 30977267))
E115_IDS = tuple(range(30790293, 30790303)) + tuple(range(30790304, 30790319))
COMPETITOR_IDS = E113_IDS + E115_IDS
FENCE_NICE = "1000"


def baseline_nice(job_id: int) -> str:
    return "0" if job_id in E113_IDS else "100"


def run(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if check and result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"command failed {command}: {detail}")
    return result


def show(job_id: int) -> str:
    return run(["scontrol", "show", "job", "-dd", "-o", str(job_id)]).stdout.strip()


def field(record: str, key: str) -> str:
    match = re.search(rf"(?:^| )({re.escape(key)})=([^ ]*)", record)
    if not match:
        raise RuntimeError(f"scheduler record lacks {key}")
    return match.group(2)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(payload: dict[str, Any]) -> None:
    temporary = ARTIFACT.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(ARTIFACT)


def validate_recovery() -> dict[str, Any]:
    payload = json.loads(RECOVERY.read_text(encoding="utf-8"))
    e117 = payload.get("e117", {})
    observed = tuple(int(value) for value in e117.get("replacement_job_ids", []))
    if observed != E117_IDS:
        raise RuntimeError(f"E117 recovery job set drifted: {observed}")
    e113 = payload.get("e113", {})
    observed_e113 = tuple(int(value) for value in e113.get("replacement_job_ids", []))
    if observed_e113 != E113_IDS:
        raise RuntimeError(f"E113 recovery job set drifted: {observed_e113}")
    return payload


def validate_e117(record: str) -> None:
    expected = {
        "UserId": "od2961(363432)",
        "JobState": "PENDING",
        "RunTime": "00:00:00",
        "Restarts": "0",
        "Partition": "all",
        "Account": "mltheory",
        "QOS": "none",
        "TimeLimit": "01:00:00",
        "Nice": "0",
        "Dependency": "(null)",
        "TresPerNode": "gres/gpu:1",
    }
    failures = {
        key: (field(record, key), value)
        for key, value in expected.items()
        if field(record, key) != value
    }
    if field(record, "ReqNodeList") not in {"node202", "node203"}:
        failures["ReqNodeList"] = field(record, "ReqNodeList")
    if failures:
        raise RuntimeError(f"E117 scheduler identity drifted: {failures}")


def validate_competitor(record: str, *, nice: str, held: bool) -> None:
    failures: dict[str, Any] = {}
    expected = {
        "UserId": "od2961(363432)",
        "JobState": "PENDING",
        "Nice": nice,
    }
    for key, value in expected.items():
        if field(record, key) != value:
            failures[key] = (field(record, key), value)
    reason = field(record, "Reason")
    if held and reason != "JobHeldUser":
        failures["Reason"] = (reason, "JobHeldUser")
    if not held and reason == "JobHeldUser":
        failures["Reason"] = reason
    if failures:
        raise RuntimeError(f"priority competitor drifted: {failures}")


def pending_above_e117() -> tuple[int, list[str]]:
    output = run(
        [
            "squeue",
            "-h",
            "-u",
            "od2961",
            "-t",
            "PD",
            "-o",
            "%i|%j|%r|%Q",
        ]
    ).stdout
    eligible: list[str] = []
    for line in output.splitlines():
        parts = line.split("|")
        if len(parts) != 4:
            continue
        reason, priority = parts[2], int(parts[3])
        if (
            reason not in {"Dependency", "DependencyNeverSatisfied", "JobHeldUser"}
            and priority > 8075
        ):
            eligible.append(line)
    return len(eligible), eligible


def install(*, check_only: bool) -> int:
    if ARTIFACT.exists():
        raise SystemExit(f"refusing duplicate priority fence: {ARTIFACT}")
    if not PROTOCOL.is_file():
        raise SystemExit(f"missing protocol: {PROTOCOL}")
    validate_recovery()
    e117_before = {job_id: show(job_id) for job_id in E117_IDS}
    for record in e117_before.values():
        validate_e117(record)
    before = {job_id: show(job_id) for job_id in COMPETITOR_IDS}
    for job_id, record in before.items():
        validate_competitor(record, nice=baseline_nice(job_id), held=False)
    count_before, rows_before = pending_above_e117()
    if count_before < 83:
        raise RuntimeError(
            f"expected at least 83 eligible owner jobs above E117, observed {count_before}"
        )
    if check_only:
        print(
            f"[check] E117={len(E117_IDS)} competitors={len(COMPETITOR_IDS)} "
            f"eligible_above={count_before} projected_after={count_before - len(COMPETITOR_IDS)}"
        )
        return 0

    changed: list[int] = []
    released = False
    try:
        run(["scontrol", "uhold", *[str(job_id) for job_id in COMPETITOR_IDS]])
        for job_id in COMPETITOR_IDS:
            validate_competitor(show(job_id), nice=baseline_nice(job_id), held=True)
        for job_id in COMPETITOR_IDS:
            run(["scontrol", "update", f"JobId={job_id}", f"Nice={FENCE_NICE}"])
            changed.append(job_id)
        held_after = {job_id: show(job_id) for job_id in COMPETITOR_IDS}
        for record in held_after.values():
            validate_competitor(record, nice=FENCE_NICE, held=True)

        payload: dict[str, Any] = {
            "schema": "e117r2r1s1_owner_backfill_priority_fence_v1",
            "installed_at": datetime.now(timezone.utc).astimezone().isoformat(),
            "protocol": str(PROTOCOL.relative_to(ROOT)),
            "protocol_sha256": digest(PROTOCOL),
            "application": str(Path(__file__).resolve().relative_to(ROOT)),
            "application_sha256": digest(Path(__file__).resolve()),
            "recovery_artifact": str(RECOVERY.relative_to(ROOT)),
            "recovery_artifact_sha256": digest(RECOVERY),
            "e117_job_ids": list(E117_IDS),
            "competitor_job_ids": list(COMPETITOR_IDS),
            "e113_competitor_job_ids": list(E113_IDS),
            "e115_competitor_job_ids": list(E115_IDS),
            "baseline_nice": {
                str(job_id): int(baseline_nice(job_id)) for job_id in COMPETITOR_IDS
            },
            "fence_nice": int(FENCE_NICE),
            "bf_max_job_user": 64,
            "eligible_above_e117_before": count_before,
            "eligible_above_e117_rows_before": rows_before,
            "e117_before": {str(key): value for key, value in e117_before.items()},
            "competitor_before": {str(key): value for key, value in before.items()},
            "competitor_held_after": {
                str(key): value for key, value in held_after.items()
            },
            "scheduler_only": True,
            "e117_changed": False,
            "other_users_changed": False,
            "installed": False,
            "restored": False,
        }
        atomic_json(payload)
        run(["scontrol", "release", *[str(job_id) for job_id in COMPETITOR_IDS]])
        released = True

        after = {job_id: show(job_id) for job_id in COMPETITOR_IDS}
        for record in after.values():
            validate_competitor(record, nice=FENCE_NICE, held=False)
        for job_id in E117_IDS:
            validate_e117(show(job_id))
        count_after, rows_after = pending_above_e117()
        if count_after > 53:
            raise RuntimeError(
                f"priority fence did not open full E117 window: {count_after} jobs remain above"
            )
        payload["eligible_above_e117_after"] = count_after
        payload["eligible_above_e117_rows_after"] = rows_after
        payload["competitor_after_release"] = {
            str(key): value for key, value in after.items()
        }
        payload["installed"] = True
        atomic_json(payload)
    except Exception:
        if not released:
            for job_id in changed:
                run(
                    [
                        "scontrol",
                        "update",
                        f"JobId={job_id}",
                        f"Nice={baseline_nice(job_id)}",
                    ],
                    check=False,
                )
            run(
                ["scontrol", "release", *[str(job_id) for job_id in COMPETITOR_IDS]],
                check=False,
            )
            if ARTIFACT.exists():
                ARTIFACT.unlink()
        raise

    print(
        f"[installed] E117={len(E117_IDS)} competitors={len(COMPETITOR_IDS)} "
        f"eligible_above={count_before}->{count_after}"
    )
    return 0


def accounting_start(job_id: int) -> str:
    row = (
        run(["sacct", "-X", "-j", str(job_id), "-n", "-P", "-o", "Start"])
        .stdout.strip()
        .splitlines()
    )
    return row[0].strip() if row else ""


def restore() -> int:
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    if payload.get("installed") is not True:
        raise RuntimeError("priority fence is not installed")
    if payload.get("restored") is True:
        print("[restore] already restored")
        return 0
    starts = {job_id: accounting_start(job_id) for job_id in E117_IDS}
    missing = [
        job_id for job_id, value in starts.items() if not value or value == "Unknown"
    ]
    if missing:
        raise SystemExit(
            f"E117 jobs have not all started; refusing restoration: {missing}"
        )

    restored: dict[int, str] = {}
    skipped: dict[int, str] = {}
    for job_id in COMPETITOR_IDS:
        current = show(job_id)
        if field(current, "JobState") == "PENDING":
            if field(current, "Nice") != FENCE_NICE:
                raise RuntimeError(
                    f"competitor {job_id} Nice drifted before restoration"
                )
            run(
                [
                    "scontrol",
                    "update",
                    f"JobId={job_id}",
                    f"Nice={baseline_nice(job_id)}",
                ]
            )
            restored[job_id] = show(job_id)
        else:
            skipped[job_id] = current
    payload["restored_at"] = datetime.now(timezone.utc).astimezone().isoformat()
    payload["e117_start_evidence"] = {str(key): value for key, value in starts.items()}
    payload["restored_scheduler_records"] = {
        str(key): value for key, value in restored.items()
    }
    payload["nonpending_competitor_records"] = {
        str(key): value for key, value in skipped.items()
    }
    payload["restored"] = True
    atomic_json(payload)
    print(f"[restored] pending={len(restored)} nonpending={len(skipped)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--check", action="store_true")
    modes.add_argument("--apply", action="store_true")
    modes.add_argument("--restore", action="store_true")
    args = parser.parse_args()
    if args.restore:
        return restore()
    return install(check_only=args.check)


if __name__ == "__main__":
    raise SystemExit(main())
