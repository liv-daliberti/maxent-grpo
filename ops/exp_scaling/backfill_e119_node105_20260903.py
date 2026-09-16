#!/usr/bin/env python3
"""Move eight high-progress pending E119 cells onto idle node105 capacity.

This is a scheduler-only continuation: scientific cells, run directories,
frozen source, datasets, objectives, and seeds remain unchanged. New jobs are
submitted held, audited, committed to the E119 continuation ledger, and only
then released after the superseded pending allocations are canceled.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import campaign_stats
import launch_e119_level2_qwen05b_factorial as e119
import status_e78 as shared


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "var/artifacts"
LEDGER = ROOT / e119.LEDGER
CONTINUATIONS = ARTIFACTS / "e119_level2_continuation_jobs.json"
AUDIT = ARTIFACTS / "e119_node105_backfill_20260903.json"
PVL = "node[004-008,020-026,101,103-104,403,805-808,901-902,906-909,911-914]"
TARGET_ORIGINALS = (
    31014399,  # Countdown Re:Dr s43, checkpoint 1728
    31014400,  # Countdown MaxRL s43, checkpoint 1536
    31014445,  # MathIR Re:Max s44, checkpoint 1536
    31014404,  # Countdown MaxRL s44, checkpoint 1536
    31014398,  # Countdown Dr.GRPO s43, checkpoint 1344
    31014403,  # Countdown Re:Dr s44, checkpoint 1344
    31014497,  # Python Re:Max s47, checkpoint 1344
    31014401,  # Countdown Re:Max s43, checkpoint 1152
)
EXPECTED = {
    31014399: ("countdown", "replay_drgrpo", 43),
    31014400: ("countdown", "maxrl", 43),
    31014445: ("mathir", "replay_maxrl", 44),
    31014404: ("countdown", "maxrl", 44),
    31014398: ("countdown", "drgrpo", 43),
    31014403: ("countdown", "replay_drgrpo", 44),
    31014497: ("python_factors", "replay_maxrl", 47),
    31014401: ("countdown", "replay_maxrl", 43),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def show(job_id: int) -> str:
    result = subprocess.run(
        ["scontrol", "show", "job", "-dd", "-o", str(job_id)],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def scheduler_state(job_id: int) -> str:
    record = show(job_id)
    for token in record.split():
        if token.startswith("JobState="):
            return token.split("=", 1)[1]
    raise RuntimeError(f"job {job_id} has no JobState")


def node105_command(command: list[str]) -> list[str]:
    replacements = {
        "--mem=64G": "--mem=40G",
    }
    rewritten = [replacements.get(part, part) for part in command]
    return [*rewritten[:-1], f"--exclude={PVL}", rewritten[-1]]


def submit_held(command: list[str], cell: dict[str, Any]) -> tuple[int, str]:
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "sbatch failed")
    job_id = int(result.stdout.strip().split(";", 1)[0])
    record = show(job_id)
    e119.audit(str(job_id), cell)
    required = (
        "JobState=PENDING",
        "Reason=JobHeldUser",
        "Account=mltheory",
        "Partition=mltheory",
        "ReqNodeList=node105",
        "MinMemoryNode=40G",
        "gres/gpu:a5000=1",
        f"ExcNodeList={PVL}",
        "OAT_ZERO_AUTO_RESUME=1",
        f"RUN_STAMP={cell['run_stamp']}",
        f"SAVE_PATH={cell['run_dir']}",
    )
    missing = [token for token in required if token not in record]
    if missing:
        raise RuntimeError(f"held node105 job {job_id} lacks {missing}")
    return job_id, record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()

    if AUDIT.exists():
        raise SystemExit(f"node105 backfill already installed: {AUDIT}")

    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    continuation = json.loads(CONTINUATIONS.read_text(encoding="utf-8"))
    if ledger.get("schema") != "e119_level2_qwen05b_factorial_jobs_v1":
        raise RuntimeError("E119 ledger schema drift")
    if ledger.get("launcher_sha256") != sha256(Path(e119.__file__).resolve()):
        raise RuntimeError("E119 launcher drift")
    mapping = campaign_stats.e119_continuation_jobs(LEDGER)
    if not mapping:
        raise RuntimeError("E119 continuation ledger failed validation")

    runs = {int(run["job_id"]): run for run in ledger["runs"]}
    templates = {
        (str(template["domain"]), int(template["seed"])): template
        for template in e119.templates(ROOT)
    }
    snapshot = Path(ledger["snapshot_root"])
    prior_by_original = {
        int(record["original_job_id"]): record
        for record in continuation["continuations"]
    }
    plans: list[dict[str, Any]] = []

    for original_id in TARGET_ORIGINALS:
        run = runs.get(original_id)
        if run is None:
            raise RuntimeError(f"missing original E119 job {original_id}")
        identity = (str(run["domain"]), str(run["arm"]), int(run["seed"]))
        if identity != EXPECTED[original_id]:
            raise RuntimeError(
                f"identity drift for {original_id}: {identity} != {EXPECTED[original_id]}"
            )
        old_id = int(mapping.get(original_id, original_id))
        old_record = show(old_id)
        if "JobState=PENDING" not in old_record or "Reason=JobHeldUser" in old_record:
            raise RuntimeError(f"effective job {old_id} is not an eligible pending job")

        template = templates[(identity[0], identity[2])]
        env, target = e119.environment(ROOT, template, identity[1], snapshot)
        if str(target) != str(run["run_dir"]):
            raise RuntimeError(f"run-directory drift for {original_id}")
        cell = {
            "original_job_id": original_id,
            "old_job_id": old_id,
            "domain": identity[0],
            "arm": identity[1],
            "seed": identity[2],
            "run_stamp": str(run["run_stamp"]),
            "run_dir": str(run["run_dir"]),
            "step": shared.run_step(Path(run["run_dir"])),
            "checkpoint": shared.checkpoint_step(Path(run["run_dir"])),
        }
        cell["command"] = node105_command(
            e119.command(ROOT, template, identity[1], env)
        )
        plans.append(cell)

    ranked_pending = [
        row for row in campaign_stats.load_smoke_snapshot(LEDGER)["rows"]
        if row["state"] == "PENDING"
    ]
    ranked_pending.sort(key=lambda row: -int(row["step"]))
    expected_top = tuple(int(row["original_job_id"]) for row in ranked_pending[:8])
    if set(expected_top) != set(TARGET_ORIGINALS):
        raise RuntimeError(
            f"highest-progress pending set drifted: {expected_top}"
        )

    print("E119 node105 scheduler-only backfill plan")
    for cell in plans:
        print(
            f"{cell['old_job_id']} -> node105 | {cell['domain']} | "
            f"{cell['arm']} | s{cell['seed']} | "
            f"step={cell['step']} checkpoint={cell['checkpoint']}"
        )
    if not args.submit:
        print("dry run only; pass --submit to install")
        return 0

    submitted: list[int] = []
    records: list[dict[str, Any]] = []
    committed = False
    try:
        for cell in plans:
            new_id, held_record = submit_held(cell["command"], cell)
            submitted.append(new_id)
            cell["new_job_id"] = new_id
            cell["held_scheduler_record"] = held_record

        for cell in plans:
            if scheduler_state(int(cell["old_job_id"])) != "PENDING":
                raise RuntimeError(
                    f"superseded job {cell['old_job_id']} left PENDING during audit"
                )

        for cell in plans:
            original_id = int(cell["original_job_id"])
            prior = prior_by_original.get(original_id)
            old_id = int(cell["old_job_id"])
            previous = [] if prior is None else list(
                prior.get("previous_continuation_job_ids", [])
            )
            if old_id != original_id and old_id not in previous:
                previous.append(old_id)
            prior_by_original[original_id] = {
                "original_job_id": original_id,
                "continuation_job_id": int(cell["new_job_id"]),
                "domain": cell["domain"],
                "arm": cell["arm"],
                "seed": int(cell["seed"]),
                "run_dir": cell["run_dir"],
                "run_stamp": cell["run_stamp"],
                "repair_kind": "node105_mltheory_backfill",
                "previous_continuation_job_ids": previous,
            }
            records.append(
                {
                    key: cell[key]
                    for key in (
                        "original_job_id", "old_job_id", "new_job_id",
                        "domain", "arm", "seed", "run_dir", "run_stamp",
                        "step", "checkpoint",
                    )
                }
            )

        continuation_before_sha = sha256(CONTINUATIONS)
        continuation["continuations"] = list(prior_by_original.values())
        continuation["operational_change"] = (
            str(continuation.get("operational_change", "")).rstrip("; ")
            + "; eight high-progress pending E119 cells backfilled onto "
              "mltheory node105 at 40G"
        ).lstrip("; ")
        continuation["released"] = False
        audit = {
            "schema": "e119-node105-backfill-v1",
            "reason": "use idle entitled mltheory node105 capacity",
            "selection_policy": (
                "eight pending E119 cells with greatest realized optimizer depth"
            ),
            "same_scientific_cells": True,
            "same_run_directories": True,
            "automatic_checkpoint_resume": True,
            "scheduler_only_change": True,
            "outcomes_inspected": False,
            "partition": "mltheory",
            "account": "mltheory",
            "node": "node105",
            "memory_per_job": "40G",
            "pvl_exclusion": PVL,
            "original_ledger": str(LEDGER),
            "original_ledger_sha256": sha256(LEDGER),
            "continuation_ledger_before_sha256": continuation_before_sha,
            "installer": str(Path(__file__).resolve()),
            "installer_sha256": sha256(Path(__file__).resolve()),
            "released": False,
            "replacements": records,
        }
        atomic(CONTINUATIONS, continuation)
        atomic(AUDIT, audit)
        committed = True

        subprocess.run(
            ["scancel", *[str(cell["old_job_id"]) for cell in plans]],
            check=True,
        )
        for _ in range(20):
            states = {
                int(cell["old_job_id"]): shared.scheduler_states(
                    [int(cell["old_job_id"])]
                ).get(int(cell["old_job_id"]), "NOT_IN_QUEUE")
                for cell in plans
            }
            if all(state not in {"PENDING", "RUNNING", "CONFIGURING"}
                   for state in states.values()):
                break
            time.sleep(0.5)
        else:
            raise RuntimeError(f"superseded jobs remain active: {states}")

        for new_id in submitted:
            subprocess.run(["scontrol", "release", str(new_id)], check=True)

        new_states = shared.scheduler_states(submitted)
        bad = {
            job_id: state for job_id, state in new_states.items()
            if state not in {"PENDING", "RUNNING", "CONFIGURING"}
        }
        if bad:
            raise RuntimeError(f"released node105 jobs entered bad states: {bad}")

        continuation["released"] = True
        audit["released"] = True
        audit["post_release_states"] = {
            str(job_id): new_states.get(job_id, "NOT_IN_QUEUE")
            for job_id in submitted
        }
        atomic(CONTINUATIONS, continuation)
        atomic(AUDIT, audit)
    except Exception:
        if submitted and not committed:
            subprocess.run(
                ["scancel", *[str(job_id) for job_id in submitted]],
                check=False,
            )
        raise

    print(
        "released node105 continuations: "
        + ",".join(str(job_id) for job_id in submitted)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
