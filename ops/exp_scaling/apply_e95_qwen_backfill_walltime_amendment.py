#!/usr/bin/env python3
"""Shorten nine zero-step E95 Qwen allocations to enable safe backfill."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / (
    "paper/preregistration/e95_qwen_backfill_walltime_amendment_20260818.md"
)
SOURCE = ROOT / "var/artifacts/e95_completion_placement_amendment.json"
LEDGER = ROOT / "var/artifacts/e95_plain_grpo_Qwen25-05B_jobs.json"
OUT = ROOT / "var/artifacts/e95_qwen_backfill_walltime_amendment.json"
TARGETS = {
    30516445: ("graph_coloring", 43),
    30516453: ("countdown", 46),
    30516454: ("countdown", 47),
    30516455: ("python_factors", 43),
    30516456: ("python_factors", 44),
    30516463: ("mathir", 46),
    30516464: ("mathir", 47),
    30516465: ("pantry_plan", 43),
    30516466: ("pantry_plan", 44),
}
NODES = "node[105,202-204]"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def command(argv: list[str]) -> str:
    result = subprocess.run(argv, capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return result.stdout.strip()


def load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"missing E95 artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def scheduler_record(job_id: int) -> str:
    return command(["scontrol", "show", "job", "-dd", "-o", str(job_id)])


def selected_runs() -> dict[int, dict[str, Any]]:
    source = load(SOURCE)
    if source.get("released") is not True:
        raise RuntimeError("E95 placement amendment was not released")
    ledger = load(LEDGER)
    selected: dict[int, dict[str, Any]] = {}
    for job_id, (domain, seed) in TARGETS.items():
        matches = [
            row
            for row in ledger["runs"]
            if int(row["job_id"]) == job_id
            and str(row["domain"]) == domain
            and int(row["seed"]) == seed
        ]
        if len(matches) != 1:
            raise RuntimeError(f"E95 Qwen target identity drifted: {job_id}")
        selected[job_id] = matches[0]
    return selected


def require_job(run: dict[str, Any], record: str, *, shortened: bool) -> None:
    required = [
        "JobState=PENDING",
        "RunTime=00:00:00",
        "Partition=all",
        "Account=allcs",
        f"ReqNodeList={NODES}",
        "Nice=0",
        "TresPerNode=gres/gpu:a5000:1",
        "MinMemoryNode=64G",
        f"OAT_ZERO_SEED={run['seed']}",
        "OAT_ZERO_VARIANT=grpo_plain_control",
        "OAT_ZERO_MAX_TRAIN=384",
        "OAT_ZERO_SAVE_STEPS=192",
        "OAT_ZERO_AUTO_RESUME=1",
        "OAT_ZERO_WATCHDOG_REQUEUE=1",
        f"SAVE_PATH={run['run_dir']}",
        "TimeLimit=12:00:00" if shortened else "TimeLimit=1-12:00:00",
    ]
    missing = [needle for needle in required if needle not in record]
    if missing:
        raise RuntimeError(f"E95 Qwen job {run['job_id']} lacks {missing}")


def update_ledger(amendment: dict[str, Any]) -> None:
    payload = load(LEDGER)
    entries = payload.setdefault("scheduler_amendments", [])
    if any(str(item.get("artifact")) == str(OUT) for item in entries):
        raise RuntimeError(f"E95 Qwen ledger already records {OUT}")
    entries.append(amendment)
    atomic_json(LEDGER, payload)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if not PROTOCOL.is_file():
        raise SystemExit(f"missing E95 Qwen protocol: {PROTOCOL}")
    if args.apply and OUT.exists():
        raise SystemExit(f"refusing duplicate E95 Qwen amendment: {OUT}")

    runs = selected_runs()
    before = {job_id: scheduler_record(job_id) for job_id in runs}
    for job_id, run in runs.items():
        require_job(run, before[job_id], shortened=False)

    if not args.apply:
        print(f"hold {len(runs)} zero-step E95 Qwen jobs")
        print("TimeLimit 1-12:00:00 -> 12:00:00")
        print("audit, record, release")
        return 0

    held: list[int] = []
    changed: list[int] = []
    try:
        for job_id in runs:
            command(["scontrol", "hold", str(job_id)])
            held.append(job_id)
        for job_id in runs:
            command(
                ["scontrol", "update", f"JobId={job_id}", "TimeLimit=12:00:00"]
            )
            changed.append(job_id)

        held_after = {job_id: scheduler_record(job_id) for job_id in runs}
        for job_id, run in runs.items():
            require_job(run, held_after[job_id], shortened=True)
            if "Reason=JobHeldUser" not in held_after[job_id]:
                raise RuntimeError(f"E95 Qwen job {job_id} escaped held audit")

        timestamp = dt.datetime.now(dt.timezone.utc).isoformat()
        payload: dict[str, Any] = {
            "schema": "e95_qwen_backfill_walltime_amendment_v1",
            "applied_at": timestamp,
            "released": False,
            "protocol": str(PROTOCOL),
            "protocol_sha256": digest(PROTOCOL),
            "source_amendment": str(SOURCE),
            "source_amendment_sha256": digest(SOURCE),
            "old_time_limit": "1-12:00:00",
            "new_time_limit": "12:00:00",
            "jobs": [
                {
                    "job_id": job_id,
                    "domain": run["domain"],
                    "seed": run["seed"],
                    "run_dir": run["run_dir"],
                    "before": before[job_id],
                    "held_after": held_after[job_id],
                }
                for job_id, run in runs.items()
            ],
        }
        atomic_json(OUT, payload)
        update_ledger(
            {
                "artifact": str(OUT),
                "protocol": str(PROTOCOL),
                "applied_at": timestamp,
                "scientific_change": False,
                "change": "walltime 36h to 12h for A5000 backfill",
            }
        )
        for job_id in runs:
            command(["scontrol", "release", str(job_id)])
        released_after = {job_id: scheduler_record(job_id) for job_id in runs}
        for job_id, record in released_after.items():
            if "Reason=JobHeldUser" in record:
                raise RuntimeError(f"E95 Qwen job {job_id} remained user-held")
        payload["released"] = True
        payload["released_after"] = released_after
        atomic_json(OUT, payload)
    except Exception:
        if not OUT.exists():
            for job_id in changed:
                subprocess.run(
                    [
                        "scontrol",
                        "update",
                        f"JobId={job_id}",
                        "TimeLimit=1-12:00:00",
                    ],
                    check=False,
                )
            for job_id in held:
                subprocess.run(["scontrol", "release", str(job_id)], check=False)
        raise

    print(f"released {len(runs)} E95 Qwen jobs with 12h walltime; artifact={OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
