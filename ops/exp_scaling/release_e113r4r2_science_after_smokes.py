#!/usr/bin/env python3
"""Release E113-R4-R2 science only after both official-DAPO smokes pass."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import e113r4_official_dapo_common as common
import launch_e113r4_official_dapo as launch
import recover_e113r4_vllm_scheduler_config as recovery


ROOT = common.ROOT
LEDGER = common.LEDGER
SLURM_SCRIPT = ROOT / "ops/slurm/e113r4_official_dapo.slurm"
SCHEDULER_AMENDMENT = ROOT / (
    "paper/preregistration/"
    "e113r4r2s1_completed_smoke_dependency_expiry_20260824.md"
)


def load_ledger() -> dict[str, Any]:
    payload = json.loads(LEDGER.read_text(encoding="utf-8"))
    if payload.get("schema") != "e113r4_official_verl_dapo_jobs_v1":
        raise RuntimeError("unexpected E113-R4 ledger schema")
    amendment = payload.get("vllm_scheduler_recovery")
    if not isinstance(amendment, dict):
        raise RuntimeError("R4-R2 scheduler recovery is not recorded")
    if amendment.get("schema") != "e113r4_r4r2_vllm_scheduler_recovery_v1":
        raise RuntimeError("unexpected R4-R2 recovery schema")
    if amendment.get("science_replacements_submitted") is True:
        raise RuntimeError("R4-R2 science replacements are already submitted")
    expected_snapshot = str(amendment["recovery_snapshot"])
    if payload["validation"]["runtime_snapshot"] != expected_snapshot:
        raise RuntimeError("authoritative R4-R2 runtime snapshot drifted")
    return payload


def smoke_gate(payload: dict[str, Any]) -> dict[str, Any]:
    amendment = payload["vllm_scheduler_recovery"]
    smokes = payload.get("smokes", {})
    if set(smokes) != set(common.FAMILIES):
        raise RuntimeError("R4-R2 smoke family surface drifted")
    expected_ids = [
        int(value)
        for value in amendment["replacement_smoke_job_ids"]
    ]
    observed_ids = [
        int(smokes[family]["job_id"])
        for family in common.FAMILIES
    ]
    if observed_ids != expected_ids:
        raise RuntimeError(
            f"authoritative R4-R2 smoke IDs drifted: {observed_ids}"
        )

    states = recovery.accounting(observed_ids)
    reports: list[dict[str, Any]] = []
    for family in common.FAMILIES:
        row = smokes[family]
        job_id = int(row["job_id"])
        state = states[job_id]
        run_dir = Path(str(row["run_dir"]))
        receipt_path = run_dir / "TRAINING_COMPLETE.json"
        checkpoint = run_dir / "checkpoints/global_step_1"
        log_path = Path(str(row["log_path"]))
        violations: list[str] = []

        if state["state"] != "COMPLETED":
            violations.append(
                f"Slurm state is {state['state']}; expected COMPLETED"
            )
        if state["exit_code"] != "0:0":
            violations.append(
                f"exit code is {state['exit_code']}; expected 0:0"
            )
        if state["elapsed"] == "00:00:00":
            violations.append("smoke has zero elapsed runtime")

        receipt: dict[str, Any] | None = None
        if not receipt_path.is_file():
            violations.append(f"missing completion receipt: {receipt_path}")
        else:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            expected_receipt = {
                "schema": (
                    "e113r4_official_verl_dapo_training_complete_v1"
                ),
                "family": family,
                "domain": str(row["domain"]),
                "seed": int(row["seed"]),
                "total_training_steps": 1,
                "runtime_snapshot": str(amendment["recovery_snapshot"]),
                "slurm_job_id": str(job_id),
            }
            wrong_receipt = {
                key: (value, receipt.get(key))
                for key, value in expected_receipt.items()
                if receipt.get(key) != value
            }
            if wrong_receipt:
                violations.append(
                    f"completion receipt drifted: {wrong_receipt}"
                )

        if not checkpoint.is_dir():
            violations.append(
                f"missing terminal global_step_1 checkpoint: {checkpoint}"
            )
        elif not (checkpoint / "actor").is_dir():
            violations.append(
                f"terminal checkpoint lacks actor state: {checkpoint / 'actor'}"
            )
        latest = run_dir / "checkpoints/latest_checkpointed_iteration.txt"
        if not latest.is_file() or latest.read_text(encoding="utf-8").strip() != "1":
            violations.append(
                f"latest checkpoint pointer is not global step 1: {latest}"
            )

        log_text = (
            log_path.read_text(encoding="utf-8", errors="replace")
            if log_path.is_file()
            else ""
        )
        for marker in (
            "train/num_gen_batches",
            "Final validation metrics:",
            "local_global_step_folder:",
        ):
            if marker not in log_text:
                violations.append(f"missing trainer evidence in log: {marker}")

        reports.append(
            {
                "family": family,
                "job_id": job_id,
                "state": state,
                "run_dir": str(run_dir),
                "receipt": receipt,
                "checkpoint": str(checkpoint),
                "passed": not violations,
                "violations": violations,
            }
        )
    return {
        "schema": "e113r4_r4r2_smoke_gate_v1",
        "passed": len(reports) == 2
        and all(row["passed"] for row in reports),
        "smokes": reports,
    }


def validate_science_inputs(
    payload: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[int, dict[str, str]]]:
    amendment = payload["vllm_scheduler_recovery"]
    old_runs = deepcopy(
        amendment["superseded_runs_pending_replacement"]
    )
    if len(old_runs) != 50 or len(
        {int(row["job_id"]) for row in old_runs}
    ) != 50:
        raise RuntimeError("frozen R4-R1 science matrix drifted")
    observed_cells = {
        (str(row["family"]), str(row["domain"]), int(row["seed"]))
        for row in old_runs
    }
    expected_cells = {
        (family, domain, seed)
        for family in common.FAMILIES
        for domain in common.DOMAINS
        for seed in common.SEEDS[family]
    }
    if observed_cells != expected_cells:
        raise RuntimeError("frozen R4-R1 science cell identities drifted")

    job_ids = [int(row["job_id"]) for row in old_runs]
    states = recovery.accounting(job_ids)
    wrong = {
        job_id: states[job_id]
        for job_id in job_ids
        if not states[job_id]["state"].startswith("CANCELLED")
        or states[job_id]["elapsed"] != "00:00:00"
        or states[job_id]["start"] not in {"", "None"}
    }
    if wrong:
        raise RuntimeError(
            f"superseded science accounting drifted: {wrong}"
        )
    existing_outputs = [
        str(row["run_dir"])
        for row in old_runs
        if Path(str(row["run_dir"])).exists()
    ]
    if existing_outputs:
        raise RuntimeError(
            f"superseded science output paths now exist: {existing_outputs}"
        )
    return old_runs, states


def science_name(row: dict[str, Any]) -> str:
    family_tag = "q05" if row["family"] == "qwen05b" else "f1"
    domain_tag = common.DOMAIN_TAGS[str(row["domain"])][:5]
    return f"e113r4r2-{family_tag}-{domain_tag}-s{row['seed']}"


def science_env(
    row: dict[str, Any], snapshot: Path
) -> dict[str, str]:
    env = {
        str(key): str(value)
        for key, value in row["environment"].items()
    }
    env.update(
        {
            "E113R4_RUNTIME_SNAPSHOT": str(snapshot),
            "E113R4_VERL_ROOT": str(snapshot / "verl"),
            "E113R4_RUN_SCRIPT": str(
                snapshot / "ops/run_e113r4_official_dapo.sh"
            ),
        }
    )
    return env


def sbatch_command(
    row: dict[str, Any],
    env: dict[str, str],
    dependency_ids: list[int],
) -> tuple[list[str], Path]:
    name = science_name(row)
    log = ROOT / "var/artifacts/logs" / f"{name}-%j.out"
    command = [
        "sbatch",
        "--parsable",
        "--hold",
        "--requeue",
        f"--job-name={name}",
        "--partition=all",
        "--account=allcs",
        "--gres=gpu:a6000:1",
        "--nodes=1",
        "--cpus-per-task=16",
        "--mem=128G",
        "--time=7-00:00:00",
        "--nice=0",
        f"--output={log}",
        f"--error={log.with_suffix('.err')}",
        "--export=ALL,"
        + ",".join(f"{key}={value}" for key, value in env.items()),
        str(SLURM_SCRIPT),
    ]
    return command, log


def submit(command: list[str]) -> int:
    response = recovery.run(command).stdout.strip().split(
        ";", maxsplit=1
    )[0]
    if not response.isdigit():
        raise RuntimeError(f"invalid sbatch response: {response!r}")
    job_id = int(response)
    recovery.run(
        [
            "scontrol",
            "update",
            f"JobId={job_id}",
            "Partition=all",
        ]
    )
    return job_id


def audit_held(
    job_id: int,
    env: dict[str, str],
    dependency_ids: list[int],
) -> str:
    record = recovery.run(
        ["scontrol", "show", "job", "-dd", "-o", str(job_id)]
    ).stdout.strip()
    expected = {
        "JobState": "PENDING",
        "Reason": "JobHeldUser",
        "Requeue": "1",
        "Account": "allcs",
        "Partition": "all",
        "NumCPUs": "16",
        "MinMemoryNode": "128G",
        "TimeLimit": "7-00:00:00",
        "Nice": "0",
        "TresPerNode": "gres/gpu:a6000:1",
    }
    wrong = {
        key: (value, recovery.field(record, key))
        for key, value in expected.items()
        if recovery.field(record, key) != value
    }
    if recovery.field(record, "NumNodes") not in {"1", "1-1"}:
        wrong["NumNodes"] = (
            "one node",
            recovery.field(record, "NumNodes"),
        )
    submit_line = recovery.field(record, "SubmitLine") or ""
    if "--dependency=" in submit_line:
        wrong["SubmitLine dependency"] = (
            "absent after completed-smoke controller expiry",
            submit_line,
        )
    if wrong:
        raise RuntimeError(
            f"held R4-R2 science job {job_id} failed audit: {wrong}"
        )
    missing_env = [
        key
        for key, value in env.items()
        if f"{key}={value}" not in record
    ]
    if missing_env:
        raise RuntimeError(
            f"held R4-R2 science job {job_id} lost environment: "
            f"{missing_env}"
        )
    return record


def make_row(
    old: dict[str, Any],
    job_id: int,
    command: list[str],
    env: dict[str, str],
    log_template: Path,
    held: str,
    dependency_ids: list[int],
) -> dict[str, Any]:
    row = deepcopy(old)
    row.update(
        {
            "job_id": job_id,
            "replaces_job_id": int(old["job_id"]),
            "stage": "full_r2",
            "run_dir": env["E113R4_OUTPUT"],
            "log_path": str(log_template).replace("%j", str(job_id)),
            "command": command,
            "environment": env,
            "held_scheduler_record": held,
            "dependency_smoke_job_ids": dependency_ids,
            "recovery": "vllm_scheduler_cap_v1",
        }
    )
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()

    payload = load_ledger()
    gate = smoke_gate(payload)
    amendment = payload["vllm_scheduler_recovery"]
    old_runs, old_accounting = validate_science_inputs(payload)
    validation = {
        "smoke_gate": gate,
        "science_cells": len(old_runs),
        "recovery_snapshot": amendment["recovery_snapshot"],
        "recovery_snapshot_sha256": (
            amendment["recovery_snapshot_sha256"]
        ),
        "science_parameters_changed": False,
    }
    if not args.launch:
        print(json.dumps(validation, indent=2, sort_keys=True))
        print(
            "dry-run only; --launch is accepted only after "
            "both smoke gates pass"
        )
        return 0 if gate["passed"] else 1
    if not gate["passed"]:
        raise RuntimeError(
            "refusing science submission because R4-R2 smoke gate failed"
        )

    snapshot = Path(str(amendment["recovery_snapshot"]))
    smoke_ids = [
        int(payload["smokes"][family]["job_id"])
        for family in common.FAMILIES
    ]
    new_runs: list[dict[str, Any]] = []
    submitted: list[int] = []
    ledger_written = False
    try:
        for old in old_runs:
            output = Path(str(old["run_dir"]))
            if output.exists():
                raise RuntimeError(
                    f"refusing existing science output: {output}"
                )
            env = science_env(old, snapshot)
            command, log = sbatch_command(
                old, env, smoke_ids
            )
            job_id = submit(command)
            submitted.append(job_id)
            held = audit_held(job_id, env, smoke_ids)
            new_runs.append(
                make_row(
                    old,
                    job_id,
                    command,
                    env,
                    log,
                    held,
                    smoke_ids,
                )
            )

        amendment["smoke_gate_passed"] = True
        amendment["smoke_gate"] = gate
        amendment["smoke_gate_passed_at"] = (
            datetime.now(timezone.utc).isoformat()
        )
        amendment["science_replacements_submitted"] = True
        amendment["science_replacements_submitted_at"] = (
            datetime.now(timezone.utc).isoformat()
        )
        amendment["replacement_science_job_ids"] = [
            int(row["job_id"]) for row in new_runs
        ]
        amendment["superseded_science_accounting_at_release"] = {
            str(job_id): old_accounting[job_id]
            for job_id in sorted(old_accounting)
        }
        amendment[
            "all_science_replacements_held_and_audited_before_release"
        ] = True
        amendment["science_release_launcher"] = str(Path(__file__))
        amendment["science_release_launcher_sha256"] = (
            recovery.digest(Path(__file__))
        )
        amendment["science_scheduler_dependency_amendment"] = {
            "schema": "e113r4_r4r2s1_dependency_expiry_v1",
            "protocol": str(SCHEDULER_AMENDMENT),
            "protocol_sha256": recovery.digest(SCHEDULER_AMENDMENT),
            "smoke_job_ids": smoke_ids,
            "cluster_min_job_age_seconds": 300,
            "science_job_ids_assigned_by_failed_attempt": [],
            "dependency_enforcement": (
                "outcome-blind release auditor passed before submission"
            ),
            "scientific_parameters_changed": False,
        }
        payload["runs"] = new_runs
        payload["released"] = False
        launch.atomic_json(LEDGER, payload)
        ledger_written = True

        recovery.run(
            [
                "scontrol",
                "release",
                *(str(job_id) for job_id in submitted),
            ]
        )
        payload["released"] = True
        payload["released_at"] = datetime.now(timezone.utc).isoformat()
        launch.atomic_json(LEDGER, payload)
    except BaseException:
        if submitted and not ledger_written:
            recovery.run(
                ["scancel", *(str(job_id) for job_id in submitted)],
                check=False,
            )
        raise

    print(
        "released 50 R4-R2 science jobs after both smoke gates: "
        f"{new_runs[0]['job_id']}--{new_runs[-1]['job_id']}"
    )
    print(f"ledger: {LEDGER}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
