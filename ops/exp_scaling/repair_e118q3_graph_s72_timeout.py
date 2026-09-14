#!/usr/bin/env python3
"""Replace the timed-out E118 Qwen-3B Graph/MaxRL seed-72 cell."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import launch_e118q3_qwen3b_maxrl_extension as launch


ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / launch.LEDGER
REPAIR = ROOT / "var/artifacts/e118q3_graph_s72_timeout_repair_20260902.json"
OLD_JOB_ID = 31010886
CHECKPOINT_STEP = "01920"
PVL_NODES = (
    "node[004-008,020-026,101,103-104,403,805-808,"
    "901-902,906-909,911-914]"
)


def main() -> int:
    if REPAIR.exists():
        raise SystemExit(f"repair already exists: {REPAIR}")
    payload = json.loads(LEDGER.read_text(encoding="utf-8"))
    old = next(
        (run for run in payload["runs"] if int(run["job_id"]) == OLD_JOB_ID),
        None,
    )
    if old is None:
        raise SystemExit("failed E118 cell is absent from the source ledger")
    state = subprocess.run(
        ["sacct", "-n", "-X", "-j", str(OLD_JOB_ID), "--format=State", "-P"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip().splitlines()[0].split()[0]
    if state != "TIMEOUT":
        raise SystemExit(f"expected TIMEOUT, found {state}")

    checkpoint = Path(old["run_dir"]) / (
        f"debug_job{OLD_JOB_ID}/checkpoints/step_{CHECKPOINT_STEP}"
    )
    if not checkpoint.is_dir():
        raise SystemExit(f"missing validated checkpoint: {checkpoint}")

    templates = {
        (str(run["domain"]), int(run["seed"])): run
        for run in launch.e80.references(ROOT)
    }
    template = templates[(str(old["domain"]), int(old["seed"]))]
    snapshot = Path(payload["snapshot_root"])
    model = launch.e80.model_root(ROOT)
    env, target = launch.environment(ROOT, template, str(old["arm"]), snapshot, model)
    if str(target) != str(old["run_dir"]):
        raise SystemExit("continuation target drifted")

    result = subprocess.run(
        launch.command(ROOT, template, str(old["arm"]), env),
        capture_output=True,
        text=True,
        check=True,
    )
    new_job_id = result.stdout.strip().split(";", 1)[0]
    if not new_job_id.isdigit():
        raise RuntimeError(f"invalid replacement job id: {result.stdout!r}")
    try:
        held = launch.audit(new_job_id, template, str(old["arm"]), model)
        subprocess.run(
            ["scontrol", "update", f"JobId={new_job_id}",
             f"ExcNodeList={PVL_NODES}"],
            check=True,
        )
        old["previous_job_ids"] = [*old.get("previous_job_ids", []), OLD_JOB_ID]
        old["job_id"] = int(new_job_id)
        old["held_scheduler_record"] = held
        repair = {
            "schema": "e118q3_graph_s72_timeout_repair_v1",
            "reason": "healthy Qwen-3B cell exceeded the 12-hour allocation",
            "scientific_configuration_changed": False,
            "old_job_id": OLD_JOB_ID,
            "new_job_id": int(new_job_id),
            "domain": old["domain"],
            "arm": old["arm"],
            "seed": old["seed"],
            "resume_checkpoint": str(checkpoint),
            "pvl_nodes_excluded": PVL_NODES,
            "released": False,
        }
        launch.e80.atomic_json(LEDGER, payload)
        launch.e80.atomic_json(REPAIR, repair)
        subprocess.run(["scontrol", "release", new_job_id], check=True)
        repair["released"] = True
        launch.e80.atomic_json(REPAIR, repair)
    except Exception:
        launch.e80.cancel([new_job_id])
        raise

    print(f"repaired E118 job {OLD_JOB_ID} as {new_job_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
