#!/usr/bin/env python3
"""Recover the one E113-R4 cell that exhausted DAPO group filtering."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Any

import e113r4_official_dapo_common as common
import launch_e113r4_official_dapo as launch


ROOT = common.ROOT
PROTOCOL = ROOT / (
    "paper/preregistration/" "e113r4r2s5_filter_group_exhaustion_recovery_20260829.md"
)
RUNNER = ROOT / "ops/run_e113r4_official_dapo.sh"
SLURM_SCRIPT = ROOT / "ops/slurm/e113r4_official_dapo.slurm"
RECORD = ROOT / ("var/artifacts/" "e113r4r2s5_filter_group_exhaustion_recovery.json")
FAILED_JOB_ID = 30869122
REJECTED_HELD_JOB_IDS = (30971093, 30971112)
CAP_BEFORE = 10
CAP_AFTER = 20
MAX_EPOCHS_BEFORE = 240
MAX_EPOCHS_AFTER = 480
DISCARDED_PREFIX_STEPS = 2
FAILURE_TEXT = (
    "ValueError: num_gen_batches=10 >= max_num_gen_batches=10. "
    "Generated too many. Please check your data."
)
SHORTFALL_TEXT = "num_prompt_in_batch=122 < prompt_bsz=128"
REPLACEMENT_NAME = "e113r4r2s5-q05-pyth-s44"


def run(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=check,
    )


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def field(record: str, key: str) -> str | None:
    match = re.search(rf"(?:^|\s){re.escape(key)}=(\S*)", record)
    return match.group(1) if match else None


def accounting(job_id: int) -> dict[str, str]:
    result = run(
        [
            "sacct",
            "-X",
            "-j",
            str(job_id),
            "--starttime",
            "2026-08-29",
            "-n",
            "-P",
            "-o",
            (
                "JobIDRaw,State,ExitCode,Elapsed,Start,End,NodeList,"
                "Account,Partition,QOS,Timelimit,Restarts"
            ),
        ]
    )
    for line in result.stdout.splitlines():
        values = line.split("|")
        if len(values) != 12 or values[0] != str(job_id):
            continue
        keys = (
            "job_id",
            "state",
            "exit_code",
            "elapsed",
            "start",
            "end",
            "node_list",
            "account",
            "partition",
            "qos",
            "time_limit",
            "restarts",
        )
        return dict(zip(keys, values, strict=True))
    raise RuntimeError(f"Slurm accounting lacks E113-R4 job {job_id}")


def validate_rejected_held_attempts() -> dict[str, dict[str, str]]:
    expected = {
        "exit_code": "0:0",
        "elapsed": "00:00:00",
        "start": "None",
        "account": "mltheory",
        "partition": "all",
        "qos": "short",
        "time_limit": "12:00:00",
        "restarts": "0",
    }
    observed: dict[str, dict[str, str]] = {}
    for job_id in REJECTED_HELD_JOB_IDS:
        state = accounting(job_id)
        wrong = {
            key: (value, state.get(key))
            for key, value in expected.items()
            if state.get(key) != value
        }
        if not state["state"].startswith("CANCELLED"):
            wrong["state"] = ("CANCELLED", state["state"])
        if wrong:
            raise RuntimeError(
                f"rejected held-attempt {job_id} accounting drifted: {wrong}"
            )
        observed[str(job_id)] = state
    return observed


def validate_runner() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    default_marker = 'max_num_gen_batches="' + "$" + '{E113R4_MAX_NUM_GEN_BATCHES:-10}"'
    required = (
        default_marker,
        ('"algorithm.filter_groups.max_num_gen_batches=' '$max_num_gen_batches"'),
        '"$max_num_gen_batches" =~ ^[1-9][0-9]*$',
    )
    missing = [contract for contract in required if contract not in source]
    if missing:
        raise RuntimeError(f"filter-cap runner contract failed: {missing}")
    run(["bash", "-n", str(RUNNER)])


def validate_ledger() -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    payload = json.loads(common.LEDGER.read_text(encoding="utf-8"))
    if payload.get("schema") != "e113r4_official_verl_dapo_jobs_v1":
        raise RuntimeError("unexpected E113-R4 ledger schema")
    if payload.get("released") is not True:
        raise RuntimeError("E113-R4 ledger is not in its released state")
    if payload.get("filter_group_exhaustion_recovery") is not None:
        raise RuntimeError("filter-group recovery is already recorded")
    if RECORD.exists():
        raise RuntimeError(f"recovery record already exists: {RECORD}")

    runs = payload.get("runs")
    if not isinstance(runs, list) or len(runs) != 50:
        raise RuntimeError("authoritative E113-R4 run count drifted")
    if len({int(row["job_id"]) for row in runs}) != 50:
        raise RuntimeError("authoritative E113-R4 job IDs are not unique")
    matches = [row for row in runs if int(row["job_id"]) == FAILED_JOB_ID]
    if len(matches) != 1:
        raise RuntimeError("failed E113-R4 job is not one exact ledger row")
    old = deepcopy(matches[0])
    expected_cell = {
        "family": "qwen05b",
        "domain": "python_factors",
        "seed": 44,
        "arm": "dapo",
    }
    drift = {
        key: (expected, old.get(key))
        for key, expected in expected_cell.items()
        if old.get(key) != expected
    }
    if drift:
        raise RuntimeError(f"failed scientific cell drifted: {drift}")

    state = accounting(FAILED_JOB_ID)
    expected_accounting = {
        "state": "FAILED",
        "exit_code": "1:0",
        "node_list": "node205",
        "account": "mltheory",
        "partition": "all",
        "qos": "long",
        "time_limit": "12:00:00",
        "restarts": "0",
    }
    wrong = {
        key: (expected, state.get(key))
        for key, expected in expected_accounting.items()
        if state.get(key) != expected
    }
    if wrong:
        raise RuntimeError(f"failed job accounting drifted: {wrong}")
    node = run(["scontrol", "show", "node", state["node_list"], "-o"]).stdout.lower()
    if "gpu:a6000" not in node:
        raise RuntimeError("failed job node is no longer an A6000 node")

    stderr = Path(str(old["log_path"])).with_suffix(".err")
    stdout = Path(str(old["log_path"]))
    if FAILURE_TEXT not in stderr.read_text(encoding="utf-8"):
        raise RuntimeError(f"failure signature drifted: {stderr}")
    output_text = stdout.read_text(encoding="utf-8")
    if "step:2 -" not in output_text or SHORTFALL_TEXT not in output_text:
        raise RuntimeError(
            "failed job did not retain the preregistered two-step/shortfall " "evidence"
        )

    run_dir = Path(str(old["run_dir"]))
    if (run_dir / "TRAINING_COMPLETE.json").exists():
        raise RuntimeError("failed job unexpectedly has a completion receipt")
    if any((run_dir / "checkpoints").glob("global_step_*")):
        raise RuntimeError("failed job unexpectedly has a checkpoint")
    if list(run_dir.glob("TRAINING_COMPLETE*.json")):
        raise RuntimeError("failed job unexpectedly has a training receipt")

    old_env = {str(key): str(value) for key, value in old["environment"].items()}
    if "E113R4_MAX_NUM_GEN_BATCHES" in old_env:
        raise RuntimeError("failed job already overrode the generation cap")
    if old_env.get("E113R4_MAX_EPOCHS") not in {
        None,
        str(MAX_EPOCHS_BEFORE),
    }:
        raise RuntimeError("failed job max-epoch budget drifted")
    base = Path(old_env["E113R4_RUNTIME_SNAPSHOT"])
    old_runner = (base / "ops/run_e113r4_official_dapo.sh").read_text(encoding="utf-8")
    if "algorithm.filter_groups.max_num_gen_batches=10" not in old_runner:
        raise RuntimeError("failed snapshot did not pin the original cap")

    queued = run(["squeue", "-h", "-n", REPLACEMENT_NAME]).stdout.strip()
    if queued:
        raise RuntimeError(f"replacement-name collision before submission: {queued}")
    return payload, old, state


def recovery_snapshot(
    old: dict[str, Any],
) -> tuple[Path, str, str]:
    old_env = {str(key): str(value) for key, value in old["environment"].items()}
    base = Path(old_env["E113R4_RUNTIME_SNAPSHOT"])
    base_identity_path = base / "SNAPSHOT_IDENTITY.json"
    base_metadata = json.loads(base_identity_path.read_text(encoding="utf-8"))
    base_identity = str(base_metadata["sha256"])
    runner_hash = digest(RUNNER)
    protocol_hash = digest(PROTOCOL)
    identity = hashlib.sha256(
        (
            f"{base_identity}\n{runner_hash}\n{protocol_hash}\n"
            "filter-group-exhaustion-v1\n"
        ).encode("utf-8")
    ).hexdigest()
    target = (
        ROOT
        / "var/artifacts/source_snapshots"
        / f"e113r4_official_verl_dapo_filtercap_{identity[:16]}"
    )
    if target.is_dir():
        metadata = json.loads(
            (target / "SNAPSHOT_IDENTITY.json").read_text(encoding="utf-8")
        )
        if metadata.get("sha256") != identity:
            raise RuntimeError(f"recovery snapshot identity drifted: {target}")
        if digest(target / "ops/run_e113r4_official_dapo.sh") != runner_hash:
            raise RuntimeError("recovery snapshot runner drifted")
        return target, identity, base_identity

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        shutil.copytree(base, temporary, dirs_exist_ok=True)
        shutil.copy2(
            RUNNER,
            temporary / "ops/run_e113r4_official_dapo.sh",
        )
        launch.atomic_json(
            temporary / "SNAPSHOT_IDENTITY.json",
            {
                "schema": ("e113r4_official_verl_dapo_runtime_recovery_v3"),
                "sha256": identity,
                "base_snapshot": str(base),
                "base_snapshot_sha256": base_identity,
                "runner_sha256": runner_hash,
                "protocol_sha256": protocol_hash,
                "upstream_commit": common.VERL_COMMIT,
                "only_runtime_change": (
                    "the failed Qwen/Python/seed-44 DAPO cell may collect "
                    "up to 20 generation batches to construct the unchanged "
                    "128-prompt accepted training batch"
                ),
            },
        )
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return target, identity, base_identity


def replacement_environment(old: dict[str, Any], snapshot: Path) -> dict[str, str]:
    env = {str(key): str(value) for key, value in old["environment"].items()}
    env.update(
        {
            "E113R4_RUNTIME_SNAPSHOT": str(snapshot),
            "E113R4_VERL_ROOT": str(snapshot / "verl"),
            "E113R4_RUN_SCRIPT": str(snapshot / "ops/run_e113r4_official_dapo.sh"),
            "E113R4_MAX_NUM_GEN_BATCHES": str(CAP_AFTER),
            "E113R4_MAX_EPOCHS": str(MAX_EPOCHS_AFTER),
        }
    )
    if env.get("E113R4_OUTPUT") != str(old["run_dir"]):
        raise RuntimeError("replacement output path drifted")
    return env


def sbatch_command(
    env: dict[str, str],
) -> tuple[list[str], Path]:
    log = ROOT / "var/artifacts/logs" / f"{REPLACEMENT_NAME}-%j.out"
    command = [
        "sbatch",
        "--parsable",
        "--hold",
        "--requeue",
        f"--job-name={REPLACEMENT_NAME}",
        "--partition=all",
        "--account=allcs",
        "--qos=long",
        "--gres=gpu:a6000:1",
        "--nodes=1",
        "--cpus-per-task=16",
        "--mem=128G",
        "--time=12:00:00",
        "--nice=0",
        f"--output={log}",
        f"--error={log.with_suffix('.err')}",
        "--export=ALL," + ",".join(f"{key}={value}" for key, value in env.items()),
        str(SLURM_SCRIPT),
    ]
    return command, log


def submit(command: list[str]) -> int:
    response = run(command).stdout.strip().split(";", maxsplit=1)[0]
    if not response.isdigit():
        raise RuntimeError(f"invalid sbatch response: {response!r}")
    job_id = int(response)
    run(
        [
            "scontrol",
            "update",
            f"JobId={job_id}",
            "Account=mltheory",
            "Partition=all",
        ]
    )
    run(
        [
            "scontrol",
            "update",
            f"JobId={job_id}",
            "QOS=long",
        ]
    )
    return job_id


def scheduler_record(job_id: int) -> str:
    return run(["scontrol", "show", "job", "-dd", "-o", str(job_id)]).stdout.strip()


def audit_held(job_id: int, env: dict[str, str]) -> str:
    record = scheduler_record(job_id)
    expected = {
        "JobState": "PENDING",
        "Reason": "JobHeldUser",
        "Requeue": "1",
        "Account": "mltheory",
        "Partition": "all",
        "QOS": "long",
        "NumCPUs": "16",
        "MinMemoryNode": "128G",
        "TimeLimit": "12:00:00",
        "Nice": "0",
        "TresPerNode": "gres/gpu:a6000:1",
        "Dependency": "(null)",
    }
    wrong = {
        key: (value, field(record, key))
        for key, value in expected.items()
        if field(record, key) != value
    }
    if field(record, "NumNodes") not in {"1", "1-1"}:
        wrong["NumNodes"] = ("one node", field(record, "NumNodes"))
    if wrong:
        raise RuntimeError(
            f"held filter-cap replacement {job_id} failed audit: {wrong}"
        )
    missing_env = [key for key, value in env.items() if f"{key}={value}" not in record]
    if missing_env:
        raise RuntimeError(f"held replacement {job_id} lost environment: {missing_env}")
    return record


def make_row(
    old: dict[str, Any],
    job_id: int,
    command: list[str],
    env: dict[str, str],
    log_template: Path,
    held: str,
) -> dict[str, Any]:
    row = deepcopy(old)
    row.update(
        {
            "job_id": job_id,
            "prior_job_id": FAILED_JOB_ID,
            "replaces_job_id": FAILED_JOB_ID,
            "stage": "full_r2s5",
            "run_dir": env["E113R4_OUTPUT"],
            "log_path": str(log_template).replace("%j", str(job_id)),
            "command": command,
            "environment": env,
            "held_scheduler_record": held,
            "recovery": "filter_group_exhaustion_cap_v1",
            "max_num_gen_batches": CAP_AFTER,
        }
    )
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()

    if not PROTOCOL.is_file():
        raise SystemExit(f"missing prospective recovery protocol: {PROTOCOL}")
    validate_runner()
    original_ledger_sha256 = digest(common.LEDGER)
    payload, old, failed_accounting = validate_ledger()
    rejected_held_accounting = validate_rejected_held_attempts()
    snapshot, snapshot_identity, base_identity = recovery_snapshot(old)
    env = replacement_environment(old, snapshot)
    command, log = sbatch_command(env)
    validation = {
        "protocol": str(PROTOCOL),
        "protocol_sha256": digest(PROTOCOL),
        "launcher": str(Path(__file__).resolve()),
        "launcher_sha256": digest(Path(__file__)),
        "runner": str(RUNNER),
        "runner_sha256": digest(RUNNER),
        "slurm_script": str(SLURM_SCRIPT),
        "slurm_script_sha256": digest(SLURM_SCRIPT),
        "base_snapshot": old["environment"]["E113R4_RUNTIME_SNAPSHOT"],
        "base_snapshot_sha256": base_identity,
        "recovery_snapshot": str(snapshot),
        "recovery_snapshot_sha256": snapshot_identity,
        "failed_job_id": FAILED_JOB_ID,
        "failed_job_accounting": failed_accounting,
        "rejected_held_job_ids": list(REJECTED_HELD_JOB_IDS),
        "rejected_held_job_accounting": rejected_held_accounting,
        "same_scientific_cell": True,
        "accepted_batch_definition_changed": False,
        "max_num_gen_batches_before": CAP_BEFORE,
        "max_num_gen_batches_after": CAP_AFTER,
        "max_epochs_before": MAX_EPOCHS_BEFORE,
        "max_epochs_after": MAX_EPOCHS_AFTER,
        "restart_from_initial_model": True,
        "checkpoint_reused": False,
        "discarded_uncheckpointed_prefix_steps": DISCARDED_PREFIX_STEPS,
        "efficacy_outcomes_inspected": False,
        "replacement_command": command,
    }
    if not args.launch:
        print(json.dumps(validation, indent=2, sort_keys=True))
        print(
            "dry-run only; pass --launch to submit, audit, record, and "
            "release the one replacement"
        )
        return 0

    submitted: int | None = None
    ledger_written = False
    try:
        submitted = submit(command)
        held = audit_held(submitted, env)
        replacement = make_row(old, submitted, command, env, log, held)
        old_ids = [int(row["job_id"]) for row in payload["runs"]]
        replacement_ids = [
            submitted if job_id == FAILED_JOB_ID else job_id for job_id in old_ids
        ]
        if sum(job_id == submitted for job_id in replacement_ids) != 1:
            raise RuntimeError("replacement did not change exactly one row")
        if [job_id for job_id in replacement_ids if job_id != submitted] != [
            job_id for job_id in old_ids if job_id != FAILED_JOB_ID
        ]:
            raise RuntimeError("unaffected E113-R4 job IDs drifted")

        now = datetime.now(timezone.utc).isoformat()
        recovery = {
            "schema": ("e113r4_r4r2s5_filter_group_exhaustion_recovery_v1"),
            "recorded_at": now,
            "root_cause": FAILURE_TEXT,
            **validation,
            "original_ledger_sha256": original_ledger_sha256,
            "original_job": old,
            "replacement_job_id": submitted,
            "same_model": True,
            "same_domain": True,
            "same_seed": True,
            "same_data": True,
            "same_prompt_and_mini_batch_sizes": True,
            "same_reward_and_loss": True,
            "same_a6000_hardware_class": True,
            "all_replacement_jobs_held_and_audited_before_release": True,
            "held_scheduler_record": held,
            "released": False,
        }
        new_runs = [
            replacement if int(row["job_id"]) == FAILED_JOB_ID else row
            for row in payload["runs"]
        ]
        if sum(int(row["job_id"]) == submitted for row in new_runs) != 1:
            raise RuntimeError("ledger replacement cardinality is not one")
        payload["runs"] = new_runs
        payload["filter_group_exhaustion_recovery"] = recovery
        launch.atomic_json(common.LEDGER, payload)
        launch.atomic_json(RECORD, recovery)
        ledger_written = True

        run(["scontrol", "release", str(submitted)])
        released_record = scheduler_record(submitted)
        if field(released_record, "Reason") == "JobHeldUser":
            raise RuntimeError("replacement remained held after release")
        released_at = datetime.now(timezone.utc).isoformat()
        recovery["released"] = True
        recovery["released_at"] = released_at
        recovery["released_scheduler_record"] = released_record
        replacement["released_at"] = released_at
        replacement["released_scheduler_record"] = released_record
        payload["filter_group_exhaustion_recovery"] = recovery
        payload["runs"] = [
            replacement if int(row["job_id"]) == submitted else row
            for row in payload["runs"]
        ]
        launch.atomic_json(common.LEDGER, payload)
        recovery_record = deepcopy(recovery)
        recovery_record["ledger_sha256_after_release"] = digest(common.LEDGER)
        launch.atomic_json(RECORD, recovery_record)
    except BaseException:
        if submitted is not None and not ledger_written:
            run(["scancel", str(submitted)], check=False)
        raise

    print("released one repaired E113-R4 cell: " f"{FAILED_JOB_ID} -> {submitted}")
    print(f"ledger: {common.LEDGER}")
    print(f"recovery record: {RECORD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
