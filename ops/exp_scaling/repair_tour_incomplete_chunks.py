#!/usr/bin/env python3
"""Submit one audited resume chunk for each incomplete E94/E96 Tour cell."""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import os
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = (
    ROOT
    / "paper/preregistration/tour_incomplete_chunk_repair_20260814.md"
)
REPAIR_LEDGER = ROOT / "var/artifacts/tour_incomplete_chunk_repair_jobs.json"
NODES = "node103,node104,node205,node207,node805"
TARGET_STEPS = 3072
TARGETS = (
    ("e96pt_point_maze_tour_semantic_maxent_jobs.json", "semantic", 43, 2688),
    ("e96pt_point_maze_tour_semantic_maxent_jobs.json", "semantic", 45, 2880),
    ("e94pt_falcon_point_maze_tour_jobs.json", "control", 58, 2880),
    ("e94pt_falcon_point_maze_tour_jobs.json", "control", 59, 2880),
    ("e94pt_falcon_point_maze_tour_jobs.json", "replay", 59, 2880),
)


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


def submit_line(record: str) -> list[str]:
    marker, end_marker = " SubmitLine=", " WorkDir="
    if marker not in record or end_marker not in record:
        raise RuntimeError("stored Tour scheduler record lacks its SubmitLine")
    command = shlex.split(record.split(marker, 1)[1].split(end_marker, 1)[0])
    if not command or command[0] != "sbatch":
        raise RuntimeError("stored Tour SubmitLine is not an sbatch command")
    return command


def resume_command(record: str) -> list[str]:
    source = submit_line(record)
    command: list[str] = []
    saw_name = saw_export = False
    for token in source:
        if token.startswith("--dependency="):
            continue
        if token.startswith("--nodelist="):
            continue
        if token.startswith("--job-name="):
            command.append(token + "-r1")
            saw_name = True
        else:
            command.append(token)
            saw_export = saw_export or token.startswith("--export=")
    if not (saw_name and saw_export and "--parsable" in command):
        raise RuntimeError("stored Tour command lacks name/export/parsable safeguards")
    command.insert(command.index("--parsable") + 1, "--hold")
    command.insert(-1, f"--nodelist={NODES}")
    return command


def scontrol_record(job_id: int) -> str:
    result = subprocess.run(
        ["scontrol", "show", "job", "-dd", "-o", str(job_id)],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or f"cannot inspect job {job_id}")
    return result.stdout.strip()


def submit_held(command: list[str]) -> int:
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "Tour sbatch failed")
    return int(result.stdout.strip().split(";", 1)[0])


def audit_held(job_id: int, *, arm: str, seed: int) -> str:
    record = scontrol_record(job_id)
    required = (
        "JobState=PENDING",
        "Reason=JobHeldUser",
        f"ReqNodeList={NODES}",
        "TimeLimit=01:00:00",
        "gres/gpu:a6000=1",
        f"OAT_ZERO_ARM={arm}",
        f"OAT_ZERO_SEED={seed}",
        "OAT_ZERO_TOUR_PASSES=8",
    )
    missing = [item for item in required if item not in record]
    if missing:
        raise RuntimeError(f"Tour repair job {job_id} lacks {missing}")
    if "Dependency=" in record and "Dependency=(null)" not in record:
        raise RuntimeError(f"Tour repair job {job_id} unexpectedly has a dependency")
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    if not PROTOCOL.is_file():
        raise SystemExit(f"missing repair protocol: {PROTOCOL}")
    if REPAIR_LEDGER.exists():
        raise SystemExit(f"refusing duplicate Tour repair: {REPAIR_LEDGER}")

    ledger_paths = {
        ROOT / "var/artifacts" / filename for filename, _arm, _seed, _done in TARGETS
    }
    ledgers = {
        path: json.loads(path.read_text(encoding="utf-8")) for path in ledger_paths
    }
    originals = copy.deepcopy(ledgers)
    candidates: list[dict[str, Any]] = []
    for filename, arm, seed, expected_update in TARGETS:
        path = ROOT / "var/artifacts" / filename
        run = next(
            row
            for row in ledgers[path]["runs"]
            if row["arm"] == arm and int(row["seed"]) == seed
        )
        checkpoint = Path(run["checkpoint_dir"]) / "COMPLETE.json"
        state = json.loads(checkpoint.read_text(encoding="utf-8"))
        observed_update = int(state["update"])
        if observed_update != expected_update or observed_update >= TARGET_STEPS:
            raise SystemExit(
                f"{filename} {arm} s{seed} checkpoint is {observed_update}, "
                f"expected {expected_update}"
            )
        command = resume_command(str(run["scheduler_record"]))
        candidates.append(
            {
                "ledger_path": path,
                "run": run,
                "arm": arm,
                "seed": seed,
                "old_job_id": int(run["job_id"]),
                "checkpoint_update_before": observed_update,
                "command": command,
            }
        )

    if not args.submit:
        for candidate in candidates:
            print(shlex.join(candidate["command"]))
        return 0

    submitted: list[int] = []
    wrote_primary = False
    try:
        for candidate in candidates:
            job_id = submit_held(candidate["command"])
            submitted.append(job_id)
            candidate["new_job_id"] = job_id
            candidate["held_scheduler_record"] = audit_held(
                job_id, arm=candidate["arm"], seed=candidate["seed"]
            )

        submitted_at = dt.datetime.now(dt.timezone.utc).isoformat()
        repair_records: list[dict[str, Any]] = []
        for candidate in candidates:
            run = candidate["run"]
            repair = {
                "reason": "afterany chain consumed one slot after a failed chunk",
                "protocol": str(PROTOCOL),
                "submitted_at": submitted_at,
                "checkpoint_update_before": candidate["checkpoint_update_before"],
                "target_steps": TARGET_STEPS,
                "old_terminal_job_id": candidate["old_job_id"],
                "new_resume_job_id": candidate["new_job_id"],
                "nodes": NODES,
                "held_scheduler_record": candidate["held_scheduler_record"],
            }
            run.setdefault("repair_history", []).append(repair)
            run["job_id"] = candidate["new_job_id"]
            run["chunk_job_ids"].append(candidate["new_job_id"])
            run["scheduler_record"] = candidate["held_scheduler_record"]
            repair_records.append(
                {
                    key: value
                    for key, value in repair.items()
                    if key != "held_scheduler_record"
                }
                | {
                    "ledger": str(candidate["ledger_path"]),
                    "arm": candidate["arm"],
                    "seed": candidate["seed"],
                    "held_scheduler_record": candidate["held_scheduler_record"],
                }
            )

        payload = {
            "schema": "tour_incomplete_chunk_repair_jobs_v1",
            "released": False,
            "protocol": str(PROTOCOL),
            "nodes": NODES,
            "target_steps": TARGET_STEPS,
            "records": repair_records,
        }
        for path, ledger in ledgers.items():
            atomic_json(path, ledger)
        wrote_primary = True
        atomic_json(REPAIR_LEDGER, payload)
        for job_id in submitted:
            subprocess.run(["scontrol", "release", str(job_id)], check=True)
        payload["released"] = True
        atomic_json(REPAIR_LEDGER, payload)
    except Exception:
        for job_id in submitted:
            subprocess.run(["scancel", str(job_id)], check=False)
        if wrote_primary:
            for path, ledger in originals.items():
                atomic_json(path, ledger)
        raise

    print("released Tour repair jobs " + " ".join(str(value) for value in submitted))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
