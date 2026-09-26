#!/usr/bin/env python3
"""Stage-one recovery for E113-R4's pre-training vLLM scheduler failure."""

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
    "paper/preregistration/e113r4r2_vllm_scheduler_recovery_20260824.md"
)
RUNNER = ROOT / "ops/run_e113r4_official_dapo.sh"
SLURM_SCRIPT = ROOT / "ops/slurm/e113r4_official_dapo.slurm"
A6000_POOL = "node[103-104,205-208]"
FAILED_SMOKES = (30855240, 30855241)
FAILURE_TEXT = (
    "max_num_batched_tokens (448) must be greater than or equal to "
    "max_num_seqs (1024)"
)
ROLLOUT_MAX_NUM_SEQS = 1024
ROLLOUT_MAX_NUM_BATCHED_TOKENS = 1024


def run(
    command: list[str], *, check: bool = True, timeout: int | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=check,
        timeout=timeout,
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


def accounting(job_ids: list[int]) -> dict[int, dict[str, str]]:
    result = run(
        [
            "sacct",
            "-X",
            "-j",
            ",".join(map(str, job_ids)),
            "--starttime",
            "2026-08-23",
            "-n",
            "-P",
            "-o",
            "JobIDRaw,State,ExitCode,Elapsed,Start,End,NodeList,Reason",
        ]
    )
    observed: dict[int, dict[str, str]] = {}
    for line in result.stdout.splitlines():
        values = line.split("|")
        if len(values) != 8 or not values[0].isdigit():
            continue
        observed[int(values[0])] = {
            "state": values[1],
            "exit_code": values[2],
            "elapsed": values[3],
            "start": values[4],
            "end": values[5],
            "node_list": values[6],
            "reason": values[7],
        }
    missing = set(job_ids) - set(observed)
    if missing:
        raise RuntimeError(
            f"Slurm accounting lacks E113-R4-R1 jobs: {sorted(missing)}"
        )
    return observed


def validate_runner() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    required = (
        'rollout_max_num_seqs=1024',
        'rollout_max_num_batched_tokens="$max_token_length"',
        "rollout_max_num_batched_tokens < rollout_max_num_seqs",
        (
            '"actor_rollout_ref.rollout.max_num_batched_tokens='
            '$rollout_max_num_batched_tokens"'
        ),
        (
            '"actor_rollout_ref.rollout.max_num_seqs='
            '$rollout_max_num_seqs"'
        ),
        'export RAY_TMPDIR="$job_tmp"',
        '--bind "$job_tmp:$job_tmp"',
    )
    missing = [contract for contract in required if contract not in source]
    if missing:
        raise RuntimeError(
            f"corrected E113-R4 runner contract failed: {missing}"
        )


def validate_vllm_scheduler() -> str:
    code = (
        "from vllm.config import SchedulerConfig; "
        "SchedulerConfig("
        f"max_num_batched_tokens={ROLLOUT_MAX_NUM_BATCHED_TOKENS},"
        f"max_num_seqs={ROLLOUT_MAX_NUM_SEQS},"
        "max_model_len=448"
        "); print('scheduler-config-valid')"
    )
    result = run(
        [
            "apptainer",
            "exec",
            str(common.IMAGE),
            "python3",
            "-c",
            code,
        ],
        timeout=300,
    )
    if "scheduler-config-valid" not in result.stdout:
        raise RuntimeError(
            "pinned vLLM SchedulerConfig preflight did not pass: "
            f"stdout={result.stdout!r} stderr={result.stderr!r}"
        )
    return result.stdout.strip()


def validate_ledger() -> tuple[dict[str, Any], dict[int, dict[str, str]]]:
    payload = json.loads(common.LEDGER.read_text(encoding="utf-8"))
    if payload.get("schema") != "e113r4_official_verl_dapo_jobs_v1":
        raise RuntimeError("unexpected E113-R4 ledger schema")
    if payload.get("vllm_scheduler_recovery") is not None:
        raise RuntimeError("E113-R4 vLLM scheduler recovery is already recorded")
    prior = payload.get("ray_socket_recovery")
    if not isinstance(prior, dict):
        raise RuntimeError("R4-R1 Ray recovery provenance is missing")

    smokes = list(payload.get("smokes", {}).values())
    if sorted(int(row["job_id"]) for row in smokes) != list(FAILED_SMOKES):
        raise RuntimeError("authoritative R4-R1 smoke IDs drifted")
    science = list(payload.get("runs", []))
    if len(science) != 50 or len(
        {int(row["job_id"]) for row in science}
    ) != 50:
        raise RuntimeError("authoritative R4-R1 science graph drifted")

    job_ids = [
        *FAILED_SMOKES,
        *(int(row["job_id"]) for row in science),
    ]
    states = accounting(job_ids)
    wrong_smokes = {
        job_id: states[job_id]
        for job_id in FAILED_SMOKES
        if states[job_id]["state"] != "FAILED"
        or states[job_id]["exit_code"] != "1:0"
    }
    if wrong_smokes:
        raise RuntimeError(
            f"failed R4-R1 smoke accounting drifted: {wrong_smokes}"
        )
    wrong_science = {
        int(row["job_id"]): states[int(row["job_id"])]
        for row in science
        if not states[int(row["job_id"])]["state"].startswith("CANCELLED")
        or states[int(row["job_id"])]["elapsed"] != "00:00:00"
        or states[int(row["job_id"])]["start"] not in {"", "None"}
    }
    if wrong_science:
        raise RuntimeError(
            "R4-R1 science jobs were not zero-runtime cancellations: "
            f"{wrong_science}"
        )

    for row in smokes:
        stderr = Path(str(row["log_path"])).with_suffix(".err")
        if FAILURE_TEXT not in stderr.read_text(encoding="utf-8"):
            raise RuntimeError(
                f"smoke failure signature drifted: {stderr}"
            )
        run_dir = Path(str(row["run_dir"]))
        if (run_dir / "TRAINING_COMPLETE.json").exists() or any(
            (run_dir / "checkpoints").glob("global_step_*")
        ):
            raise RuntimeError(
                f"failed R4-R1 smoke has scientific output: {run_dir}"
            )
    existing_science = [
        str(row["run_dir"])
        for row in science
        if Path(str(row["run_dir"])).exists()
    ]
    if existing_science:
        raise RuntimeError(
            f"zero-runtime science output paths exist: {existing_science}"
        )

    base = Path(str(payload["validation"]["runtime_snapshot"]))
    identity_path = base / "SNAPSHOT_IDENTITY.json"
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    if (
        identity.get("sha256")
        != payload["validation"]["runtime_snapshot_sha256"]
    ):
        raise RuntimeError("R4-R1 runtime snapshot identity drifted")
    if digest(SLURM_SCRIPT) != payload["validation"]["slurm_script_sha256"]:
        raise RuntimeError("E113-R4 Slurm wrapper drifted")
    return payload, states


def recovery_snapshot(
    payload: dict[str, Any],
) -> tuple[Path, str]:
    base = Path(str(payload["validation"]["runtime_snapshot"]))
    base_identity = str(
        payload["validation"]["runtime_snapshot_sha256"]
    )
    runner_hash = digest(RUNNER)
    protocol_hash = digest(PROTOCOL)
    identity = hashlib.sha256(
        (
            f"{base_identity}\n{runner_hash}\n{protocol_hash}\n"
            "vllm-scheduler-v1\n"
        ).encode("utf-8")
    ).hexdigest()
    target = (
        ROOT
        / "var/artifacts/source_snapshots"
        / f"e113r4_official_verl_dapo_vllmsched_{identity[:16]}"
    )
    if target.is_dir():
        metadata = json.loads(
            (target / "SNAPSHOT_IDENTITY.json").read_text(encoding="utf-8")
        )
        if metadata.get("sha256") != identity:
            raise RuntimeError(
                f"recovery snapshot identity drifted: {target}"
            )
        if digest(
            target / "ops/run_e113r4_official_dapo.sh"
        ) != runner_hash:
            raise RuntimeError("recovery snapshot runner drifted")
        return target, identity

    temporary = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent)
    )
    try:
        shutil.copytree(base, temporary, dirs_exist_ok=True)
        shutil.copy2(
            RUNNER,
            temporary / "ops/run_e113r4_official_dapo.sh",
        )
        launch.atomic_json(
            temporary / "SNAPSHOT_IDENTITY.json",
            {
                "schema": (
                    "e113r4_official_verl_dapo_runtime_recovery_v2"
                ),
                "sha256": identity,
                "base_snapshot": str(base),
                "base_snapshot_sha256": base_identity,
                "runner_sha256": runner_hash,
                "protocol_sha256": protocol_hash,
                "upstream_commit": common.VERL_COMMIT,
                "only_runtime_change": (
                    "vLLM aggregate scheduler cap satisfies "
                    "max_num_batched_tokens >= max_num_seqs"
                ),
            },
        )
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return target, identity


def recovery_name(row: dict[str, Any]) -> str:
    family_tag = "q05" if row["family"] == "qwen05b" else "f1"
    domain_tag = common.DOMAIN_TAGS[str(row["domain"])][:5]
    return f"e113r4r2s0-{family_tag}-{domain_tag}-s{row['seed']}"


def recovery_env(
    row: dict[str, Any], snapshot: Path, smoke_output: Path
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
            "E113R4_OUTPUT": str(smoke_output),
        }
    )
    return env


def sbatch_command(
    row: dict[str, Any], env: dict[str, str]
) -> tuple[list[str], Path]:
    name = recovery_name(row)
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
        "--time=1-00:00:00",
        "--nice=0",
        f"--output={log}",
        f"--error={log.with_suffix('.err')}",
        "--export=ALL,"
        + ",".join(f"{key}={value}" for key, value in env.items()),
        f"--nodelist={A6000_POOL}",
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
            "Partition=all",
        ]
    )
    return job_id


def audit_held(job_id: int, env: dict[str, str]) -> str:
    record = run(
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
        "TimeLimit": "1-00:00:00",
        "Nice": "0",
        "TresPerNode": "gres/gpu:a6000:1",
        "ReqNodeList": A6000_POOL,
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
            f"held R4-R2 smoke {job_id} failed audit: {wrong}"
        )
    missing_env = [
        key
        for key, value in env.items()
        if f"{key}={value}" not in record
    ]
    if missing_env:
        raise RuntimeError(
            f"held R4-R2 smoke {job_id} lost environment: {missing_env}"
        )
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
            "replaces_job_id": int(old["job_id"]),
            "stage": "operational_smoke_r2",
            "run_dir": env["E113R4_OUTPUT"],
            "log_path": str(log_template).replace("%j", str(job_id)),
            "command": command,
            "environment": env,
            "held_scheduler_record": held,
            "recovery": "vllm_scheduler_cap_v1",
        }
    )
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch-smokes", action="store_true")
    args = parser.parse_args()

    if not PROTOCOL.is_file():
        raise SystemExit(
            f"missing prospective recovery protocol: {PROTOCOL}"
        )
    validate_runner()
    payload, old_accounting = validate_ledger()
    scheduler_preflight = validate_vllm_scheduler()
    snapshot, snapshot_identity = recovery_snapshot(payload)
    validation = {
        "protocol": str(PROTOCOL),
        "protocol_sha256": digest(PROTOCOL),
        "base_snapshot": payload["validation"]["runtime_snapshot"],
        "base_snapshot_sha256": (
            payload["validation"]["runtime_snapshot_sha256"]
        ),
        "recovery_snapshot": str(snapshot),
        "recovery_snapshot_sha256": snapshot_identity,
        "runner_sha256": digest(RUNNER),
        "failed_smoke_job_ids": list(FAILED_SMOKES),
        "canceled_science_job_count": 50,
        "scheduler_preflight": scheduler_preflight,
        "rollout_max_num_batched_tokens": (
            ROLLOUT_MAX_NUM_BATCHED_TOKENS
        ),
        "rollout_max_num_seqs": ROLLOUT_MAX_NUM_SEQS,
        "scientific_parameters_changed": False,
        "official_upstream_changed": False,
        "two_stage_release": True,
    }
    if not args.launch_smokes:
        print(json.dumps(validation, indent=2, sort_keys=True))
        print(
            "dry-run only; pass --launch-smokes to submit only "
            "the two repaired operational smokes"
        )
        return 0

    old_smokes = deepcopy(payload["smokes"])
    old_runs = deepcopy(payload["runs"])
    new_smokes: dict[str, dict[str, Any]] = {}
    submitted: list[int] = []
    ledger_written = False
    try:
        for family in common.FAMILIES:
            old = deepcopy(old_smokes[family])
            smoke_output = ROOT / (
                "var/data/"
                f"e113r4r2s0_{family}_"
                f"{common.DOMAIN_TAGS[str(old['domain'])]}_"
                f"official_dapo_s{old['seed']}"
            )
            if smoke_output.exists():
                raise RuntimeError(
                    f"refusing existing R4-R2 smoke output: {smoke_output}"
                )
            env = recovery_env(old, snapshot, smoke_output)
            command, log = sbatch_command(old, env)
            job_id = submit(command)
            submitted.append(job_id)
            held = audit_held(job_id, env)
            new_smokes[family] = make_row(
                old, job_id, command, env, log, held
            )

        smoke_ids = [
            int(new_smokes[family]["job_id"])
            for family in common.FAMILIES
        ]
        now = datetime.now(timezone.utc).isoformat()
        payload["vllm_scheduler_recovery"] = {
            "schema": "e113r4_r4r2_vllm_scheduler_recovery_v1",
            "recorded_at": now,
            "root_cause": FAILURE_TEXT,
            **validation,
            "failed_smoke_accounting": {
                str(job_id): old_accounting[job_id]
                for job_id in FAILED_SMOKES
            },
            "zero_runtime_science_job_ids": [
                int(row["job_id"]) for row in old_runs
            ],
            "zero_runtime_science_accounting": {
                str(row["job_id"]): old_accounting[int(row["job_id"])]
                for row in old_runs
            },
            "replacement_smoke_job_ids": smoke_ids,
            "replacement_science_job_ids": [],
            "science_replacements_submitted": False,
            "smoke_gate_passed": False,
            "superseded_smokes": old_smokes,
            "superseded_runs_pending_replacement": old_runs,
            "all_smokes_held_and_audited_before_release": True,
            "scientific_outcome_observed_before_recovery": False,
        }
        payload["smokes"] = new_smokes
        payload["released"] = False
        payload["validation"]["runtime_snapshot"] = str(snapshot)
        payload["validation"][
            "runtime_snapshot_sha256"
        ] = snapshot_identity
        payload["validation"][
            "vllm_scheduler_recovery_launcher_sha256"
        ] = digest(Path(__file__))
        launch.atomic_json(common.LEDGER, payload)
        ledger_written = True

        run(
            [
                "scontrol",
                "release",
                *(str(job_id) for job_id in submitted),
            ]
        )
        payload["released"] = True
        payload["released_at"] = datetime.now(timezone.utc).isoformat()
        launch.atomic_json(common.LEDGER, payload)
    except BaseException:
        if submitted and not ledger_written:
            run(
                ["scancel", *(str(job_id) for job_id in submitted)],
                check=False,
            )
        raise

    print(
        "released repaired E113-R4-R2 smokes only: "
        f"{[row['job_id'] for row in new_smokes.values()]}"
    )
    print("science replacements submitted: 0 (awaiting both smoke receipts)")
    print(f"ledger: {common.LEDGER}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
