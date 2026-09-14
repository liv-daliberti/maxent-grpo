#!/usr/bin/env python3
"""Recover all five E113-R4 Qwen/Python DAPO cells under one cap policy."""

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
import tempfile
import time
from typing import Any

import e113r4_official_dapo_common as common
import launch_e113r4_official_dapo as launch
import recover_e113r4_filter_group_exhaustion as prior


ROOT = common.ROOT
PROTOCOL = ROOT / (
    "paper/preregistration/"
    "e113r4r2s6_common_python_filter_exhaustion_recovery_20260829.md"
)
RUNNER = ROOT / "ops/run_e113r4_official_dapo.sh"
SLURM_SCRIPT = ROOT / "ops/slurm/e113r4_official_dapo.slurm"
RECORD = ROOT / (
    "var/artifacts/" "e113r4r2s6_common_python_filter_exhaustion_recovery.json"
)
FAILED_BY_SEED = {
    43: 30869121,
    45: 30869123,
    46: 30869124,
    47: 30869125,
}
PENDING_BY_SEED = {44: 30971120}
OLD_BY_SEED = {**FAILED_BY_SEED, **PENDING_BY_SEED}
TERMINAL_ACCEPTED = {43: 50, 45: 44, 46: 55, 47: 64}
CAP_AFTER = 40
MAX_EPOCHS_AFTER = 960
CHECKPOINT_STEP = 10
DISCARDED_STEP = 11
FAILURE_TEXT = (
    "ValueError: num_gen_batches=10 >= max_num_gen_batches=10. "
    "Generated too many. Please check your data."
)
RECOVERY_KEY = "common_python_filter_group_exhaustion_recovery"


def validate_checkpoint(run_dir: Path) -> dict[str, int]:
    checkpoint = run_dir / "checkpoints" / f"global_step_{CHECKPOINT_STEP}"
    pointer = run_dir / "checkpoints" / "latest_checkpointed_iteration.txt"
    if pointer.read_text(encoding="utf-8").strip() != str(CHECKPOINT_STEP):
        raise RuntimeError(f"latest checkpoint pointer drifted: {pointer}")
    required = (
        "actor/model_world_size_1_rank_0.pt",
        "actor/optim_world_size_1_rank_0.pt",
        "actor/extra_state_world_size_1_rank_0.pt",
        "actor/huggingface/config.json",
        "actor/huggingface/tokenizer.json",
        "data.pt",
    )
    manifest: dict[str, int] = {}
    for relative in required:
        path = checkpoint / relative
        if not path.is_file() or path.stat().st_size <= 0:
            raise RuntimeError(f"incomplete step-10 checkpoint file: {path}")
        manifest[relative] = path.stat().st_size
    return manifest


def validate_failed_cell(
    row: dict[str, Any], seed: int
) -> tuple[dict[str, str], dict[str, int]]:
    job_id = FAILED_BY_SEED[seed]
    state = prior.accounting(job_id)
    expected = {
        "state": "FAILED",
        "exit_code": "1:0",
        "account": "mltheory",
        "partition": "all",
        "qos": "long",
        "time_limit": "12:00:00",
        "restarts": "0",
    }
    wrong = {
        key: (value, state.get(key))
        for key, value in expected.items()
        if state.get(key) != value
    }
    if wrong:
        raise RuntimeError(f"failed job {job_id} accounting drifted: {wrong}")
    node = prior.run(
        ["scontrol", "show", "node", state["node_list"], "-o"]
    ).stdout.lower()
    if "gpu:a6000" not in node:
        raise RuntimeError(f"failed job {job_id} was not on an A6000")

    stderr = Path(str(row["log_path"])).with_suffix(".err")
    stdout = Path(str(row["log_path"]))
    if FAILURE_TEXT not in stderr.read_text(encoding="utf-8"):
        raise RuntimeError(f"failure signature drifted: {stderr}")
    output = stdout.read_text(encoding="utf-8")
    steps = [int(value) for value in re.findall(r"step:(\d+) -", output)]
    if not steps or max(steps) != DISCARDED_STEP:
        raise RuntimeError(f"realized step sequence drifted for job {job_id}")
    shortfall = f"num_prompt_in_batch={TERMINAL_ACCEPTED[seed]} " "< prompt_bsz=128"
    if shortfall not in output:
        raise RuntimeError(f"terminal shortfall drifted for job {job_id}")

    run_dir = Path(str(row["run_dir"]))
    if (run_dir / "TRAINING_COMPLETE.json").exists():
        raise RuntimeError(f"failed job {job_id} has a completion receipt")
    manifest = validate_checkpoint(run_dir)
    env = {str(key): str(value) for key, value in row["environment"].items()}
    if "E113R4_MAX_NUM_GEN_BATCHES" in env:
        raise RuntimeError(f"failed job {job_id} did not use cap 10")
    if env.get("E113R4_MAX_EPOCHS") != "240":
        raise RuntimeError(f"failed job {job_id} epoch horizon drifted")
    return state, manifest


def validate_pending_seed44(row: dict[str, Any]) -> str:
    record = prior.scheduler_record(PENDING_BY_SEED[44])
    expected = {
        "JobState": "PENDING",
        "RunTime": "00:00:00",
        "Account": "mltheory",
        "Partition": "all",
        "QOS": "long",
        "Requeue": "1",
        "Restarts": "0",
    }
    wrong = {
        key: (value, prior.field(record, key))
        for key, value in expected.items()
        if prior.field(record, key) != value
    }
    if wrong:
        raise RuntimeError(f"pending seed-44 job drifted: {wrong}")
    env = {str(key): str(value) for key, value in row["environment"].items()}
    if env.get("E113R4_MAX_NUM_GEN_BATCHES") != "20":
        raise RuntimeError("pending seed-44 job is not the cap-20 replacement")
    if env.get("E113R4_MAX_EPOCHS") != "480":
        raise RuntimeError("pending seed-44 epoch horizon drifted")
    run_dir = Path(str(row["run_dir"]))
    if (run_dir / "TRAINING_COMPLETE.json").exists():
        raise RuntimeError("pending seed-44 job has a completion receipt")
    if any((run_dir / "checkpoints").glob("global_step_*")):
        raise RuntimeError("pending seed-44 job unexpectedly has a checkpoint")
    return record


def validate_ledger() -> (
    tuple[
        dict[str, Any],
        dict[int, dict[str, Any]],
        dict[str, dict[str, str]],
        dict[str, dict[str, int]],
        str,
    ]
):
    payload = json.loads(common.LEDGER.read_text(encoding="utf-8"))
    if payload.get("schema") != "e113r4_official_verl_dapo_jobs_v1":
        raise RuntimeError("unexpected E113-R4 ledger schema")
    if payload.get("released") is not True:
        raise RuntimeError("E113-R4 ledger is not released")
    if payload.get(RECOVERY_KEY) is not None:
        raise RuntimeError("common Python recovery is already recorded")
    if RECORD.exists():
        raise RuntimeError(f"common Python recovery record exists: {RECORD}")
    prior_recovery = payload.get("filter_group_exhaustion_recovery")
    if (
        not isinstance(prior_recovery, dict)
        or int(prior_recovery.get("replacement_job_id", -1)) != PENDING_BY_SEED[44]
        or prior_recovery.get("released") is not True
    ):
        raise RuntimeError("seed-44 cap-20 recovery provenance drifted")

    runs = payload.get("runs")
    if not isinstance(runs, list) or len(runs) != 50:
        raise RuntimeError("authoritative E113-R4 run count drifted")
    rows_by_id = {int(row["job_id"]): deepcopy(row) for row in runs}
    if len(rows_by_id) != 50:
        raise RuntimeError("authoritative E113-R4 job IDs are not unique")
    selected: dict[int, dict[str, Any]] = {}
    failed_accounting: dict[str, dict[str, str]] = {}
    checkpoint_manifests: dict[str, dict[str, int]] = {}
    pending_record = ""
    for seed, job_id in OLD_BY_SEED.items():
        if job_id not in rows_by_id:
            raise RuntimeError(f"missing authoritative job {job_id}")
        row = rows_by_id[job_id]
        expected = {
            "family": "qwen05b",
            "domain": "python_factors",
            "seed": seed,
            "arm": "dapo",
        }
        drift = {
            key: (value, row.get(key))
            for key, value in expected.items()
            if row.get(key) != value
        }
        if drift:
            raise RuntimeError(f"scientific cell {job_id} drifted: {drift}")
        selected[seed] = row
        if seed in FAILED_BY_SEED:
            state, manifest = validate_failed_cell(row, seed)
            failed_accounting[str(job_id)] = state
            checkpoint_manifests[str(job_id)] = manifest
        else:
            pending_record = validate_pending_seed44(row)

    for seed in OLD_BY_SEED:
        name = f"e113r4r2s6-q05-pyth-s{seed}"
        collision = prior.run(["squeue", "-h", "-n", name]).stdout.strip()
        if collision:
            raise RuntimeError(f"replacement-name collision: {collision}")
    return (
        payload,
        selected,
        failed_accounting,
        checkpoint_manifests,
        pending_record,
    )


def recovery_snapshot(rows: dict[int, dict[str, Any]]) -> tuple[Path, str, str]:
    base = Path(str(rows[43]["environment"]["E113R4_RUNTIME_SNAPSHOT"]))
    metadata = json.loads((base / "SNAPSHOT_IDENTITY.json").read_text(encoding="utf-8"))
    base_identity = str(metadata["sha256"])
    runner_hash = prior.digest(RUNNER)
    protocol_hash = prior.digest(PROTOCOL)
    launcher_hash = prior.digest(Path(__file__))
    identity = hashlib.sha256(
        (
            f"{base_identity}\n{runner_hash}\n{protocol_hash}\n"
            f"{launcher_hash}\ncommon-python-filter-cap-40-v1\n"
        ).encode("utf-8")
    ).hexdigest()
    target = (
        ROOT
        / "var/artifacts/source_snapshots"
        / f"e113r4_official_verl_dapo_filtercap40_{identity[:16]}"
    )
    if target.is_dir():
        target_metadata = json.loads(
            (target / "SNAPSHOT_IDENTITY.json").read_text(encoding="utf-8")
        )
        if target_metadata.get("sha256") != identity:
            raise RuntimeError(f"snapshot identity drifted: {target}")
        if prior.digest(target / "ops/run_e113r4_official_dapo.sh") != runner_hash:
            raise RuntimeError("snapshot runner drifted")
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
                "schema": ("e113r4_official_verl_dapo_runtime_recovery_v4"),
                "sha256": identity,
                "base_snapshot": str(base),
                "base_snapshot_sha256": base_identity,
                "runner_sha256": runner_hash,
                "protocol_sha256": protocol_hash,
                "launcher_sha256": launcher_hash,
                "upstream_commit": common.VERL_COMMIT,
                "only_runtime_change": (
                    "Qwen/Python seeds 43-47 may collect up to 40 "
                    "generation batches to construct the unchanged "
                    "128-prompt accepted training batch"
                ),
            },
        )
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return target, identity, base_identity


def replacement_environment(row: dict[str, Any], snapshot: Path) -> dict[str, str]:
    env = {str(key): str(value) for key, value in row["environment"].items()}
    env.update(
        {
            "E113R4_RUNTIME_SNAPSHOT": str(snapshot),
            "E113R4_VERL_ROOT": str(snapshot / "verl"),
            "E113R4_RUN_SCRIPT": str(snapshot / "ops/run_e113r4_official_dapo.sh"),
            "E113R4_MAX_NUM_GEN_BATCHES": str(CAP_AFTER),
            "E113R4_MAX_EPOCHS": str(MAX_EPOCHS_AFTER),
        }
    )
    if env.get("E113R4_OUTPUT") != str(row["run_dir"]):
        raise RuntimeError("replacement output path drifted")
    return env


def sbatch_command(seed: int, env: dict[str, str]) -> tuple[list[str], Path]:
    name = f"e113r4r2s6-q05-pyth-s{seed}"
    log = ROOT / "var/artifacts/logs" / f"{name}-%j.out"
    command = [
        "sbatch",
        "--parsable",
        "--hold",
        "--requeue",
        f"--job-name={name}",
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
    response = prior.run(command).stdout.strip().split(";", maxsplit=1)[0]
    if not response.isdigit():
        raise RuntimeError(f"invalid sbatch response: {response!r}")
    job_id = int(response)
    prior.run(
        [
            "scontrol",
            "update",
            f"JobId={job_id}",
            "Account=mltheory",
            "Partition=all",
        ]
    )
    prior.run(
        [
            "scontrol",
            "update",
            f"JobId={job_id}",
            "QOS=long",
        ]
    )
    return job_id


def hold_seed44(env: dict[str, str]) -> str:
    job_id = PENDING_BY_SEED[44]
    prior.run(["scontrol", "hold", str(job_id)])
    record = prior.scheduler_record(job_id)
    expected = {
        "JobState": "PENDING",
        "Reason": "JobHeldUser",
        "RunTime": "00:00:00",
        "Account": "mltheory",
        "Partition": "all",
        "QOS": "long",
    }
    wrong = {
        key: (value, prior.field(record, key))
        for key, value in expected.items()
        if prior.field(record, key) != value
    }
    missing_env = [key for key, value in env.items() if f"{key}={value}" not in record]
    if wrong or missing_env:
        raise RuntimeError(
            f"seed-44 supersession hold failed: {wrong}, env={missing_env}"
        )
    return record


def make_row(
    old: dict[str, Any],
    seed: int,
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
            "prior_job_id": int(old["job_id"]),
            "replaces_job_id": int(old["job_id"]),
            "stage": "full_r2s6",
            "run_dir": env["E113R4_OUTPUT"],
            "log_path": str(log_template).replace("%j", str(job_id)),
            "command": command,
            "environment": env,
            "held_scheduler_record": held,
            "recovery": "common_python_filter_cap40_v1",
            "max_num_gen_batches": CAP_AFTER,
        }
    )
    if seed in FAILED_BY_SEED:
        row.update(
            {
                "resume_checkpoint_step": CHECKPOINT_STEP,
                "resume_checkpoint": str(
                    Path(str(old["run_dir"]))
                    / "checkpoints"
                    / f"global_step_{CHECKPOINT_STEP}"
                ),
                "discarded_uncheckpointed_step": DISCARDED_STEP,
                "restart_from_initial_model": False,
            }
        )
    else:
        row.update(
            {
                "resume_checkpoint_step": None,
                "resume_checkpoint": None,
                "restart_from_initial_model": True,
            }
        )
    return row


def canceled_accounting(job_id: int) -> dict[str, str]:
    for _ in range(10):
        state = prior.accounting(job_id)
        if state["state"].startswith("CANCELLED"):
            if state["elapsed"] != "00:00:00" or state["start"] != "None":
                raise RuntimeError(
                    f"superseded seed-44 job ran before cancellation: {state}"
                )
            return state
        time.sleep(1)
    raise RuntimeError(f"superseded seed-44 job did not cancel: {state}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()

    if not PROTOCOL.is_file():
        raise SystemExit(f"missing prospective recovery protocol: {PROTOCOL}")
    prior.validate_runner()
    original_ledger_sha256 = prior.digest(common.LEDGER)
    (
        payload,
        rows,
        failed_accounting,
        checkpoint_manifests,
        pending_record,
    ) = validate_ledger()
    snapshot, snapshot_identity, base_identity = recovery_snapshot(rows)
    envs = {seed: replacement_environment(row, snapshot) for seed, row in rows.items()}
    commands = {seed: sbatch_command(seed, env) for seed, env in envs.items()}
    validation = {
        "protocol": str(PROTOCOL),
        "protocol_sha256": prior.digest(PROTOCOL),
        "launcher": str(Path(__file__).resolve()),
        "launcher_sha256": prior.digest(Path(__file__)),
        "runner": str(RUNNER),
        "runner_sha256": prior.digest(RUNNER),
        "slurm_script": str(SLURM_SCRIPT),
        "slurm_script_sha256": prior.digest(SLURM_SCRIPT),
        "base_snapshot": rows[43]["environment"]["E113R4_RUNTIME_SNAPSHOT"],
        "base_snapshot_sha256": base_identity,
        "recovery_snapshot": str(snapshot),
        "recovery_snapshot_sha256": snapshot_identity,
        "failed_job_ids": list(FAILED_BY_SEED.values()),
        "superseded_pending_job_id": PENDING_BY_SEED[44],
        "failed_job_accounting": failed_accounting,
        "checkpoint_manifests": checkpoint_manifests,
        "pending_seed44_scheduler_record": pending_record,
        "terminal_accepted_after_ten_batches": TERMINAL_ACCEPTED,
        "max_num_gen_batches_after": CAP_AFTER,
        "max_epochs_after": MAX_EPOCHS_AFTER,
        "accepted_batch_definition_changed": False,
        "same_scientific_cells": True,
        "common_recovery_source": True,
        "efficacy_outcomes_used_for_repair": False,
        "replacement_commands": {
            str(seed): command for seed, (command, _) in commands.items()
        },
    }
    if not args.launch:
        print(json.dumps(validation, indent=2, sort_keys=True))
        print(
            "dry-run only; pass --launch to hold seed 44, submit and audit "
            "five replacements, record them, cancel the zero-runtime "
            "superseded job, and release the replacements"
        )
        return 0

    submitted: dict[int, int] = {}
    seed44_held = False
    ledger_written = False
    try:
        seed44_hold_record = hold_seed44(
            {str(key): str(value) for key, value in rows[44]["environment"].items()}
        )
        seed44_held = True
        replacement_rows: dict[int, dict[str, Any]] = {}
        held_records: dict[str, str] = {}
        for seed in sorted(rows):
            command, log = commands[seed]
            job_id = submit(command)
            submitted[seed] = job_id
            held = prior.audit_held(job_id, envs[seed])
            held_records[str(job_id)] = held
            replacement_rows[seed] = make_row(
                rows[seed],
                seed,
                job_id,
                command,
                envs[seed],
                log,
                held,
            )

        old_ids = [int(row["job_id"]) for row in payload["runs"]]
        selected_ids = set(OLD_BY_SEED.values())
        unaffected_before = [job_id for job_id in old_ids if job_id not in selected_ids]
        new_runs = [
            replacement_rows[int(row["seed"])]
            if int(row["job_id"]) in selected_ids
            else row
            for row in payload["runs"]
        ]
        unaffected_after = [
            int(row["job_id"])
            for row in new_runs
            if int(row["job_id"]) not in set(submitted.values())
        ]
        if unaffected_after != unaffected_before:
            raise RuntimeError("unaffected E113-R4 job IDs drifted")
        if len(new_runs) != 50 or len({int(row["job_id"]) for row in new_runs}) != 50:
            raise RuntimeError("replacement graph is not 50 unique cells")

        now = datetime.now(timezone.utc).isoformat()
        recovery = {
            "schema": ("e113r4_r4r2s6_common_python_" "filter_exhaustion_recovery_v1"),
            "recorded_at": now,
            "root_cause": FAILURE_TEXT,
            **validation,
            "original_ledger_sha256": original_ledger_sha256,
            "original_jobs_by_seed": {str(seed): row for seed, row in rows.items()},
            "replacement_job_ids_by_seed": {
                str(seed): job_id for seed, job_id in submitted.items()
            },
            "seed44_hold_record": seed44_hold_record,
            "all_replacements_held_and_audited_before_release": True,
            "held_scheduler_records": held_records,
            "resumed_from_step_10_seeds": sorted(FAILED_BY_SEED),
            "discarded_uncheckpointed_step_11": True,
            "initial_model_restart_seeds": [44],
            "same_model": True,
            "same_domain": True,
            "same_seeds": True,
            "same_data": True,
            "same_prompt_and_mini_batch_sizes": True,
            "same_reward_loss_and_optimizer": True,
            "same_a6000_hardware_class": True,
            "released": False,
        }
        payload["runs"] = new_runs
        payload[RECOVERY_KEY] = recovery
        launch.atomic_json(common.LEDGER, payload)
        launch.atomic_json(RECORD, recovery)
        ledger_written = True

        prior.run(["scancel", str(PENDING_BY_SEED[44])])
        superseded_accounting = canceled_accounting(PENDING_BY_SEED[44])
        prior.run(
            [
                "scontrol",
                "release",
                *(str(submitted[seed]) for seed in sorted(submitted)),
            ]
        )
        released_records = {
            str(job_id): prior.scheduler_record(job_id) for job_id in submitted.values()
        }
        still_held = {
            job_id: record
            for job_id, record in released_records.items()
            if prior.field(record, "Reason") == "JobHeldUser"
        }
        if still_held:
            raise RuntimeError(f"replacement jobs remained held: {sorted(still_held)}")

        released_at = datetime.now(timezone.utc).isoformat()
        recovery["released"] = True
        recovery["released_at"] = released_at
        recovery["superseded_seed44_accounting"] = superseded_accounting
        recovery["released_scheduler_records"] = released_records
        for row in payload["runs"]:
            if int(row["job_id"]) in set(submitted.values()):
                row["released_at"] = released_at
                row["released_scheduler_record"] = released_records[str(row["job_id"])]
        payload[RECOVERY_KEY] = recovery
        launch.atomic_json(common.LEDGER, payload)
        recovery_record = deepcopy(recovery)
        recovery_record["ledger_sha256_after_release"] = prior.digest(common.LEDGER)
        launch.atomic_json(RECORD, recovery_record)
    except BaseException:
        if not ledger_written:
            if submitted:
                prior.run(
                    [
                        "scancel",
                        *(str(job_id) for job_id in submitted.values()),
                    ],
                    check=False,
                )
            if seed44_held:
                prior.run(
                    ["scontrol", "release", str(PENDING_BY_SEED[44])],
                    check=False,
                )
        raise

    print(
        "released common cap-40 E113-R4 Qwen/Python replacements: "
        + ", ".join(f"s{seed}={submitted[seed]}" for seed in sorted(submitted))
    )
    print(f"ledger: {common.LEDGER}")
    print(f"recovery record: {RECORD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
