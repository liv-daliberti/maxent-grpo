#!/usr/bin/env python3
"""Replace timed-out E118 Qwen-3B cells with checkpoint continuations."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import launch_e118q3_qwen3b_maxrl_extension as launch


ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / launch.LEDGER
REPAIR = ROOT / "var/artifacts/e118q3_timeout_continuation_repair_20260902.json"
EXPECTED = {31010882, 31010883}


def state(job_id: int) -> str:
    result = subprocess.run(
        ["sacct", "-n", "-X", "-j", str(job_id), "--format=State", "-P"],
        capture_output=True, text=True, check=True,
    )
    return result.stdout.strip().splitlines()[0].split()[0]


def main() -> int:
    if REPAIR.exists():
        raise SystemExit(f"repair already exists: {REPAIR}")
    payload = json.loads(LEDGER.read_text(encoding="utf-8"))
    selected = [r for r in payload["runs"] if int(r["job_id"]) in EXPECTED]
    if {int(r["job_id"]) for r in selected} != EXPECTED:
        raise SystemExit("timeout cells do not match the frozen repair set")
    templates = {
        (str(r["domain"]), int(r["seed"])): r for r in launch.e80.references(ROOT)
    }
    snapshot = Path(payload["snapshot_root"])
    model = launch.e80.model_root(ROOT)
    submitted: list[str] = []
    replacements = []
    try:
        for old in selected:
            old_id = int(old["job_id"])
            if state(old_id) != "TIMEOUT":
                raise RuntimeError(f"job {old_id} is not TIMEOUT")
            checkpoint = Path(old["run_dir"]) / f"debug_job{old_id}/checkpoints/step_01728"
            if not checkpoint.is_dir():
                raise RuntimeError(f"missing resume checkpoint: {checkpoint}")
            template = templates[(str(old["domain"]), int(old["seed"]))]
            env, target = launch.environment(ROOT, template, str(old["arm"]), snapshot, model)
            if str(target) != str(old["run_dir"]):
                raise RuntimeError("continuation target drifted")
            command = launch.command(ROOT, template, str(old["arm"]), env)
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode:
                raise RuntimeError(result.stderr.strip())
            new_id = result.stdout.strip().split(";", 1)[0]
            if not new_id.isdigit():
                raise RuntimeError(f"invalid replacement id: {result.stdout!r}")
            submitted.append(new_id)
            held = launch.audit(new_id, template, str(old["arm"]), model)
            replacements.append({
                "old_job_id": old_id, "new_job_id": int(new_id),
                "domain": old["domain"], "arm": old["arm"], "seed": old["seed"],
                "resume_checkpoint": str(checkpoint), "held_scheduler_record": held,
            })
        for replacement in replacements:
            for run in payload["runs"]:
                if int(run["job_id"]) == replacement["old_job_id"]:
                    run["previous_job_ids"] = [
                        *run.get("previous_job_ids", []), replacement["old_job_id"]
                    ]
                    run["job_id"] = replacement["new_job_id"]
                    run["held_scheduler_record"] = replacement["held_scheduler_record"]
        repair = {
            "schema": "e118q3_timeout_continuation_repair_v1",
            "reason": "healthy Qwen-3B cells exceeded the fixed 12-hour allocation",
            "scientific_configuration_changed": False,
            "wall_time": "12:00:00",
            "replacements": replacements,
            "released": False,
        }
        launch.e80.atomic_json(LEDGER, payload)
        launch.e80.atomic_json(REPAIR, repair)
        for job_id in submitted:
            result = subprocess.run(["scontrol", "release", job_id], capture_output=True, text=True)
            if result.returncode:
                raise RuntimeError(result.stderr.strip())
        repair["released"] = True
        launch.e80.atomic_json(REPAIR, repair)
    except Exception:
        launch.e80.cancel(submitted)
        raise
    print(f"repaired {len(replacements)} Qwen-3B timeout cells: {submitted}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
