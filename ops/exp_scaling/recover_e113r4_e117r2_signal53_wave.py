#!/usr/bin/env python3
"""Replace the exact E113-R4/E117-R2 cells lost in the signal-53 wave."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any


TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
import apply_e117r2s1_all_one_hour_scheduler_amendment as e117s1  # noqa: E402
import e113r4_official_dapo_common as e113  # noqa: E402
import launch_e113r4_official_dapo as e113_launch  # noqa: E402
import launch_e117r2_same_plumbing_repair as e117  # noqa: E402
import recover_e113r4_filter_group_exhaustion as e113_prior  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / (
    "paper/preregistration/e113r4_e117r2_signal53_wave_recovery_20260830.md"
)
RECORD = ROOT / "var/artifacts/e113r4_e117r2_signal53_wave_recovery.json"
E113_LEDGER = e113.LEDGER
E117_LEDGER = ROOT / e117.LEDGER
E117_AUDIT_JOB_ARTIFACT = ROOT / e117.AUDIT_JOB
E113_SLURM = ROOT / "ops/slurm/e113r4_official_dapo.slurm"
E117_SLURM = ROOT / "ops/slurm/train_node302.slurm"

E113_PROCESS_FAILURE_IDS = (
    30869134,
    30869139,
    30869140,
    30869142,
    30869143,
    30869144,
    30869145,
    30869146,
    30869147,
)
E113_SIGNAL53_IDS = tuple(range(30869148, 30869161)) + tuple(range(30972013, 30972018))
E113_FAILED_IDS = tuple(sorted(E113_PROCESS_FAILURE_IDS + E113_SIGNAL53_IDS))
E113_RESUME_STEPS: dict[int, int | None] = {
    30869134: 20,
    30869139: 20,
    30869140: 20,
    30869142: 15,
    30869143: 15,
    30869144: 15,
    30869145: None,
    30869146: None,
    30869147: None,
    **{job_id: None for job_id in range(30869148, 30869161)},
    30972013: 10,
    30972014: None,
    30972015: 10,
    30972016: 10,
    30972017: 10,
}

E117_COMPLETED_ID = 30970803
E117_PROCESS_FAILURE_IDS = (30970804,)
E117_SIGNAL53_IDS = tuple(range(30970805, 30970815))
E117_FAILED_IDS = E117_PROCESS_FAILURE_IDS + E117_SIGNAL53_IDS
E117_OLD_AUDIT_ID = 30970815

RECOVERY_KEY = "signal53_batch_wave_recovery"
ROOT_CAUSE = "transient_cross_node_batch_launch_interruption"


def run(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if check and result.returncode != 0:
        raise RuntimeError(
            f"command failed {command}: {result.stderr.strip() or result.stdout.strip()}"
        )
    return result


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def text_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def accounting(job_ids: tuple[int, ...]) -> dict[int, dict[str, str]]:
    result = run(
        [
            "sacct",
            "-X",
            "-n",
            "-P",
            "-S",
            "2026-08-20",
            "-j",
            ",".join(str(job_id) for job_id in job_ids),
            "-o",
            (
                "JobIDRaw,JobName,State,ExitCode,Elapsed,Start,End,NodeList,"
                "Reason,Partition,Account,QOS,Timelimit,Restarts"
            ),
        ]
    )
    keys = (
        "job_id",
        "job_name",
        "state",
        "exit_code",
        "elapsed",
        "start",
        "end",
        "node",
        "reason",
        "partition",
        "account",
        "qos",
        "time_limit",
        "restarts",
    )
    rows: dict[int, dict[str, str]] = {}
    wanted = set(job_ids)
    for line in result.stdout.splitlines():
        values = line.split("|")
        if len(values) != len(keys) or not values[0].isdigit():
            continue
        job_id = int(values[0])
        if job_id in wanted:
            row = dict(zip(keys, values, strict=True))
            row["state"] = row["state"].split()[0].split("+", 1)[0]
            rows[job_id] = row
    missing = wanted - set(rows)
    if missing:
        raise RuntimeError(f"Slurm accounting lacks jobs: {sorted(missing)}")
    return rows


def scheduler_record(job_id: int) -> str:
    return run(["scontrol", "show", "job", "-dd", "-o", str(job_id)]).stdout.strip()


def field(record: str, name: str) -> str:
    match = re.search(rf"(?:^|\s){re.escape(name)}=(\S*)", record)
    if match is None:
        raise RuntimeError(f"scheduler record lacks {name}")
    return match.group(1)


def submit(command: list[str]) -> int:
    value = run(command).stdout.strip().split(";", 1)[0]
    if not value.isdigit():
        raise RuntimeError(f"invalid sbatch response: {value!r}")
    return int(value)


def validate_checkpoint(run_dir: Path, step: int | None) -> dict[str, int]:
    checkpoint_root = run_dir / "checkpoints"
    pointer = checkpoint_root / "latest_checkpointed_iteration.txt"
    checkpoint_dirs = sorted(checkpoint_root.glob("global_step_*"))
    if step is None:
        if pointer.exists() or checkpoint_dirs:
            raise RuntimeError(f"unexpected resumable checkpoint in {run_dir}")
        return {}
    if pointer.read_text(encoding="utf-8").strip() != str(step):
        raise RuntimeError(f"checkpoint pointer drifted in {run_dir}")
    checkpoint = checkpoint_root / f"global_step_{step}"
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
            raise RuntimeError(f"incomplete checkpoint member: {path}")
        manifest[relative] = path.stat().st_size
    return manifest


def validate_e113() -> (
    tuple[
        dict[str, Any],
        dict[int, dict[str, Any]],
        dict[int, dict[str, str]],
        dict[str, dict[str, int]],
    ]
):
    payload = json.loads(E113_LEDGER.read_text(encoding="utf-8"))
    if payload.get("schema") != "e113r4_official_verl_dapo_jobs_v1":
        raise RuntimeError("unexpected E113-R4 ledger schema")
    if payload.get("released") is not True or payload.get(RECOVERY_KEY) is not None:
        raise RuntimeError("E113-R4 ledger is not eligible for signal-53 recovery")
    runs = payload.get("runs")
    if not isinstance(runs, list) or len(runs) != 50:
        raise RuntimeError("E113-R4 must contain exactly 50 cells")
    by_id = {int(row["job_id"]): deepcopy(row) for row in runs}
    if len(by_id) != 50 or not set(E113_FAILED_IDS).issubset(by_id):
        raise RuntimeError("E113-R4 authoritative job graph drifted")
    states = accounting(tuple(sorted(by_id)))
    checkpoint_manifests: dict[str, dict[str, int]] = {}
    for job_id, row in by_id.items():
        run_dir = Path(str(row["run_dir"]))
        receipt = run_dir / "TRAINING_COMPLETE.json"
        state = states[job_id]
        if job_id not in E113_FAILED_IDS:
            if state["state"] != "COMPLETED" or state["exit_code"] != "0:0":
                raise RuntimeError(f"unaffected E113 job {job_id} is not complete")
            if not receipt.is_file() or receipt.stat().st_size <= 0:
                raise RuntimeError(f"unaffected E113 receipt is absent: {job_id}")
            continue
        expected_exit = "0:53" if job_id in E113_SIGNAL53_IDS else "1:0"
        if state["state"] != "FAILED" or state["exit_code"] != expected_exit:
            raise RuntimeError(f"E113 failure boundary drifted for {job_id}: {state}")
        if receipt.exists():
            raise RuntimeError(f"failed E113 job {job_id} has a completion receipt")
        env = {str(key): str(value) for key, value in row["environment"].items()}
        if env.get("E113R4_OUTPUT") != str(run_dir):
            raise RuntimeError(f"E113 output identity drifted for {job_id}")
        checkpoint_manifests[str(job_id)] = validate_checkpoint(
            run_dir, E113_RESUME_STEPS[job_id]
        )
    return payload, by_id, states, checkpoint_manifests


def e113_command(row: dict[str, Any]) -> tuple[list[str], Path, dict[str, str]]:
    env = {str(key): str(value) for key, value in row["environment"].items()}
    family = "q05" if row["family"] == "qwen05b" else "f1"
    domain = {
        "graph_coloring": "graph",
        "countdown": "count",
        "python_factors": "pytho",
        "mathir": "mathi",
        "pantry_plan": "pantr",
    }[str(row["domain"])]
    name = f"e113r4r2s7-{family}-{domain}-s{row['seed']}"
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
        str(E113_SLURM),
    ]
    return command, log, env


def normalize_e113(job_id: int) -> None:
    run(
        [
            "scontrol",
            "update",
            f"JobId={job_id}",
            "Account=mltheory",
            "Partition=all",
            "QOS=long",
            "TimeLimit=12:00:00",
        ]
    )


def validate_e113_held(job_id: int, env: dict[str, str]) -> str:
    return e113_prior.audit_held(job_id, env)


def validate_e117() -> (
    tuple[
        dict[str, Any],
        dict[int, dict[str, Any]],
        dict[int, dict[str, str]],
        dict[str, Any],
    ]
):
    payload = json.loads(E117_LEDGER.read_text(encoding="utf-8"))
    if payload.get("schema") != "e117_same_plumbing_component_preflight_jobs_v1":
        raise RuntimeError("unexpected E117-R2 ledger schema")
    if payload.get("released") is not True or payload.get(RECOVERY_KEY) is not None:
        raise RuntimeError("E117-R2 ledger is not eligible for signal-53 recovery")
    runs = payload.get("runs")
    if not isinstance(runs, list) or len(runs) != 12:
        raise RuntimeError("E117-R2 must contain exactly 12 cells")
    by_id = {int(row["job_id"]): deepcopy(row) for row in runs}
    expected_ids = {E117_COMPLETED_ID, *E117_FAILED_IDS}
    if set(by_id) != expected_ids:
        raise RuntimeError("E117-R2 authoritative job graph drifted")
    states = accounting(tuple(sorted(expected_ids | {E117_OLD_AUDIT_ID})))
    completed = states[E117_COMPLETED_ID]
    if (
        completed["state"] != "COMPLETED"
        or completed["exit_code"] != "0:0"
        or completed["elapsed"] != "00:11:07"
    ):
        raise RuntimeError("E117-R2 completed reference drifted")
    for job_id in E117_FAILED_IDS:
        state = states[job_id]
        expected_exit = "0:53" if job_id in E117_SIGNAL53_IDS else "1:0"
        if state["state"] != "FAILED" or state["exit_code"] != expected_exit:
            raise RuntimeError(f"E117-R2 failure boundary drifted for {job_id}")
        row = by_id[job_id]
        expected_environment = e117s1.expected_environment(row)
        if text_digest(expected_environment) != row["scientific_environment_sha256"]:
            raise RuntimeError(f"E117 environment hash drifted for {job_id}")
        run_dir = Path(str(row["run_dir"]))
        material_outputs = (
            [path for path in run_dir.rglob("*") if path.is_file() or path.is_symlink()]
            if run_dir.exists()
            else []
        )
        if material_outputs:
            raise RuntimeError(f"failed E117 cell produced material output: {run_dir}")
    old_audit = states[E117_OLD_AUDIT_ID]
    if old_audit["state"] != "FAILED" or old_audit["exit_code"] != "0:53":
        raise RuntimeError("E117-R2 failed audit boundary drifted")
    old_audit_artifact = json.loads(E117_AUDIT_JOB_ARTIFACT.read_text(encoding="utf-8"))
    if int(old_audit_artifact.get("audit_job_id", -1)) != E117_OLD_AUDIT_ID:
        raise RuntimeError("E117-R2 durable audit identity drifted")
    return payload, by_id, states, old_audit_artifact


def e117_command(row: dict[str, Any]) -> tuple[list[str], str]:
    export = e117s1.expected_environment(row)
    scale = "q05" if row["scale"] == "qwen05b" else "f1"
    domain = {
        "graph_coloring": "graph",
        "countdown": "count",
        "python_factors": "pytho",
        "mathir": "mathi",
    }[str(row["domain"])]
    name = f"e117r2r1-{scale}-{domain}-{row['arm']}"
    log = ROOT / "var/artifacts/logs" / f"{name}-%j.out"
    command = [
        "sbatch",
        "--parsable",
        "--hold",
        "--requeue",
        f"--job-name={name}",
        "--partition=all",
        "--account=allcs",
        f"--nodelist={row['node']}",
        "--gres=gpu:1",
        "--cpus-per-task=8",
        "--mem=64G",
        "--time=01:00:00",
        "--nice=0",
        f"--output={log}",
        f"--error={log.with_suffix('.err')}",
        f"--export={export}",
        str(E117_SLURM),
    ]
    return command, export


def normalize_e117(job_id: int) -> None:
    run(
        [
            "scontrol",
            "update",
            f"JobId={job_id}",
            "Account=mltheory",
            "Partition=all",
            "TimeLimit=01:00:00",
        ]
    )


def validate_e117_held(job_id: int, row: dict[str, Any], expected_export: str) -> str:
    record = scheduler_record(job_id)
    expected = {
        "JobState": "PENDING",
        "Reason": "JobHeldUser",
        "Requeue": "1",
        "Restarts": "0",
        "Account": "mltheory",
        "Partition": "all",
        "QOS": "none",
        "TimeLimit": "01:00:00",
        "NumCPUs": "8",
        "MinMemoryNode": "64G",
        "ReqNodeList": str(row["node"]),
        "TresPerNode": "gres/gpu:1",
        "Dependency": "(null)",
    }
    wrong = {
        key: (value, field(record, key))
        for key, value in expected.items()
        if field(record, key) != value
    }
    if e117s1.environment(record) != expected_export:
        wrong["Environment"] = (
            text_digest(expected_export),
            text_digest(e117s1.environment(record)),
        )
    if wrong:
        raise RuntimeError(f"held E117 replacement {job_id} drifted: {wrong}")
    return record


def released_record(job_id: int) -> str:
    record = scheduler_record(job_id)
    if field(record, "JobState") not in {"PENDING", "RUNNING"}:
        raise RuntimeError(f"released replacement {job_id} is not live")
    if field(record, "Reason") in {"JobHeldUser", "job_requeued_in_held_state"}:
        raise RuntimeError(f"released replacement {job_id} remained held")
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    if not PROTOCOL.is_file():
        raise SystemExit(f"missing prospective protocol: {PROTOCOL}")
    if RECORD.exists():
        raise SystemExit(f"refusing duplicate recovery: {RECORD}")

    original_e113_sha = digest(E113_LEDGER)
    original_e117_sha = digest(E117_LEDGER)
    original_audit_sha = digest(E117_AUDIT_JOB_ARTIFACT)
    e113_payload, e113_rows, e113_states, checkpoint_manifests = validate_e113()
    e117_payload, e117_rows, e117_states, old_audit_artifact = validate_e117()

    validation = {
        "schema": "e113r4_e117r2_signal53_wave_recovery_v1",
        "protocol": str(PROTOCOL),
        "protocol_sha256": digest(PROTOCOL),
        "launcher": str(Path(__file__).resolve()),
        "launcher_sha256": digest(Path(__file__)),
        "root_cause": ROOT_CAUSE,
        "compute_space_root_cause_proven": False,
        "login_tmp_observed_full": True,
        "e113_failed_job_ids": list(E113_FAILED_IDS),
        "e113_signal53_job_ids": list(E113_SIGNAL53_IDS),
        "e113_process_failure_job_ids": list(E113_PROCESS_FAILURE_IDS),
        "e117_completed_job_id": E117_COMPLETED_ID,
        "e117_failed_job_ids": list(E117_FAILED_IDS),
        "e117_signal53_job_ids": list(E117_SIGNAL53_IDS),
        "e117_process_failure_job_ids": list(E117_PROCESS_FAILURE_IDS),
        "e117_failed_audit_job_id": E117_OLD_AUDIT_ID,
        "collateral_job_id_excluded": 30971429,
        "scientific_environment_changed": False,
        "source_snapshot_changed": False,
        "run_directories_changed": False,
        "efficacy_outcomes_inspected": False,
        "e113_checkpoint_manifests": checkpoint_manifests,
    }
    if not args.launch:
        print(json.dumps(validation, indent=2, sort_keys=True))
        print(
            "dry-run only; pass --launch to hold, audit, record, and release "
            "27 E113 plus 11 E117 science replacements"
        )
        return 0

    original_e113_payload = deepcopy(e113_payload)
    original_e117_payload = deepcopy(e117_payload)
    submitted_e113: dict[int, int] = {}
    submitted_e117: dict[int, int] = {}
    audit_job_id = 0
    ledgers_written = False
    release_attempted = False
    try:
        e113_replacements: dict[int, dict[str, Any]] = {}
        e113_held: dict[str, str] = {}
        for old_id in E113_FAILED_IDS:
            old = e113_rows[old_id]
            command, log, env = e113_command(old)
            new_id = submit(command)
            submitted_e113[old_id] = new_id
            normalize_e113(new_id)
            held = validate_e113_held(new_id, env)
            e113_held[str(new_id)] = held
            replacement = deepcopy(old)
            replacement.update(
                {
                    "job_id": new_id,
                    "incident_replaces_job_id": old_id,
                    "incident_recovery": ROOT_CAUSE,
                    "stage": "full_r2s7",
                    "command": command,
                    "log_path": str(log).replace("%j", str(new_id)),
                    "environment": env,
                    "held_scheduler_record": held,
                    "resume_checkpoint_step": E113_RESUME_STEPS[old_id],
                    "restart_from_initial_model": E113_RESUME_STEPS[old_id] is None,
                }
            )
            e113_replacements[old_id] = replacement

        e117_replacements: dict[int, dict[str, Any]] = {}
        e117_held: dict[str, str] = {}
        for old_id in E117_FAILED_IDS:
            old = e117_rows[old_id]
            command, export = e117_command(old)
            new_id = submit(command)
            submitted_e117[old_id] = new_id
            normalize_e117(new_id)
            held = validate_e117_held(new_id, old, export)
            e117_held[str(new_id)] = held
            replacement = deepcopy(old)
            replacement.update(
                {
                    "job_id": new_id,
                    "incident_replaces_job_id": old_id,
                    "incident_recovery": ROOT_CAUSE,
                    "command": command,
                    "held_scheduler_record": held,
                    "restart_from_initial_model": True,
                }
            )
            e117_replacements[old_id] = replacement

        new_e113_runs = [
            e113_replacements.get(int(row["job_id"]), row)
            for row in e113_payload["runs"]
        ]
        new_e117_runs = [
            e117_replacements.get(int(row["job_id"]), row)
            for row in e117_payload["runs"]
        ]
        if len({int(row["job_id"]) for row in new_e113_runs}) != 50:
            raise RuntimeError("E113 replacement graph is not 50 unique cells")
        if len({int(row["job_id"]) for row in new_e117_runs}) != 12:
            raise RuntimeError("E117 replacement graph is not 12 unique cells")

        staged_e117_payload = deepcopy(e117_payload)
        staged_e117_payload["runs"] = new_e117_runs
        staged_e117_payload["released"] = False
        e113_launch.atomic_json(E117_LEDGER, staged_e117_payload)
        audit_job_id, audit_record = e117.schedule_audit(
            root=ROOT,
            snapshot=Path(str(e117_payload["snapshot_root"])),
            ledger=E117_LEDGER,
            job_ids=list(submitted_e117.values()),
        )
        e113_launch.atomic_json(E117_LEDGER, original_e117_payload)

        now = datetime.now(timezone.utc).isoformat()
        e113_recovery = {
            **validation,
            "recorded_at": now,
            "original_ledger_sha256": original_e113_sha,
            "failed_accounting": {
                str(job_id): e113_states[job_id] for job_id in E113_FAILED_IDS
            },
            "replacement_job_ids": list(submitted_e113.values()),
            "mapping": {
                str(old_id): new_id for old_id, new_id in submitted_e113.items()
            },
            "held_scheduler_records": e113_held,
            "all_replacements_held_and_audited_before_release": True,
            "released": False,
        }
        e117_recovery = {
            **validation,
            "recorded_at": now,
            "original_ledger_sha256": original_e117_sha,
            "original_audit_artifact_sha256": original_audit_sha,
            "failed_accounting": {
                str(job_id): e117_states[job_id]
                for job_id in (*E117_FAILED_IDS, E117_OLD_AUDIT_ID)
            },
            "replacement_job_ids": list(submitted_e117.values()),
            "mapping": {
                str(old_id): new_id for old_id, new_id in submitted_e117.items()
            },
            "held_scheduler_records": e117_held,
            "already_completed_job_ids": [E117_COMPLETED_ID],
            "live_dependency_job_ids": list(submitted_e117.values()),
            "replacement_audit_job_id": audit_job_id,
            "replacement_audit_scheduler_record": audit_record,
            "all_replacements_held_and_audited_before_release": True,
            "released": False,
        }
        e113_payload["runs"] = new_e113_runs
        e113_payload[RECOVERY_KEY] = e113_recovery
        e117_payload["runs"] = new_e117_runs
        e117_payload["released"] = False
        e117_payload[RECOVERY_KEY] = e117_recovery
        e117_payload["audit_job_id"] = audit_job_id
        audit_payload = {
            "schema": "e117r2_same_plumbing_component_preflight_audit_job_v1",
            "audit_job_id": audit_job_id,
            "dependency_job_ids": [E117_COMPLETED_ID, *submitted_e117.values()],
            "live_dependency_job_ids": list(submitted_e117.values()),
            "already_completed_job_ids": [E117_COMPLETED_ID],
            "replaces_audit_job_id": E117_OLD_AUDIT_ID,
            "ledger": str(E117_LEDGER),
            "audit_output": str(ROOT / e117.AUDIT),
            "snapshot_root": str(e117_payload["snapshot_root"]),
            "audit_script_sha256": e117_payload["audit_script_sha256"],
            "scheduler_record": audit_record,
            "efficacy_outcomes_used": False,
            "released": False,
        }
        record_payload = {
            **validation,
            "recorded_at": now,
            "e113": e113_recovery,
            "e117": e117_recovery,
            "old_e117_audit_artifact": old_audit_artifact,
            "released": False,
        }
        e113_launch.atomic_json(E113_LEDGER, e113_payload)
        e113_launch.atomic_json(E117_LEDGER, e117_payload)
        e113_launch.atomic_json(E117_AUDIT_JOB_ARTIFACT, audit_payload)
        e113_launch.atomic_json(RECORD, record_payload)
        ledgers_written = True

        release_ids = [*submitted_e113.values(), *submitted_e117.values()]
        release_attempted = True
        run(["scontrol", "release", *(str(job_id) for job_id in release_ids)])
        released_records = {
            str(job_id): released_record(job_id) for job_id in release_ids
        }
        released_at = datetime.now(timezone.utc).isoformat()
        e113_recovery.update(
            {
                "released": True,
                "released_at": released_at,
                "released_scheduler_records": {
                    str(job_id): released_records[str(job_id)]
                    for job_id in submitted_e113.values()
                },
            }
        )
        e117_recovery.update(
            {
                "released": True,
                "released_at": released_at,
                "released_scheduler_records": {
                    str(job_id): released_records[str(job_id)]
                    for job_id in submitted_e117.values()
                },
            }
        )
        e113_payload[RECOVERY_KEY] = e113_recovery
        e117_payload[RECOVERY_KEY] = e117_recovery
        e117_payload["released"] = True
        audit_payload["released"] = True
        record_payload.update(
            {
                "e113": e113_recovery,
                "e117": e117_recovery,
                "released": True,
                "released_at": released_at,
            }
        )
        e113_launch.atomic_json(E113_LEDGER, e113_payload)
        e113_launch.atomic_json(E117_LEDGER, e117_payload)
        e113_launch.atomic_json(E117_AUDIT_JOB_ARTIFACT, audit_payload)
        e113_launch.atomic_json(RECORD, record_payload)
    except BaseException:
        new_ids = [*submitted_e113.values(), *submitted_e117.values()]
        if audit_job_id:
            new_ids.append(audit_job_id)
        if new_ids:
            run(["scancel", *(str(job_id) for job_id in new_ids)], check=False)
        if not release_attempted:
            if ledgers_written or E117_LEDGER.exists():
                e113_launch.atomic_json(E113_LEDGER, original_e113_payload)
                e113_launch.atomic_json(E117_LEDGER, original_e117_payload)
                e113_launch.atomic_json(E117_AUDIT_JOB_ARTIFACT, old_audit_artifact)
                if RECORD.exists():
                    RECORD.unlink()
        raise

    print(
        f"[signal53-recovery] e113_released={len(submitted_e113)} "
        f"e117_released={len(submitted_e117)} e117_audit={audit_job_id}"
    )
    print(f"record: {RECORD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
