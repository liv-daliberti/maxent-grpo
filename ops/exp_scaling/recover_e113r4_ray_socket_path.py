#!/usr/bin/env python3
"""Recover E113-R4 from the pre-training Ray AF_UNIX path-length failure."""

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
    "paper/preregistration/e113r4r1_ray_socket_recovery_20260823.md"
)
RUNNER = ROOT / "ops/run_e113r4_official_dapo.sh"
SLURM_SCRIPT = ROOT / "ops/slurm/e113r4_official_dapo.slurm"
A6000_POOL = "node[103-104,205-208]"
FAILED_SMOKES = (30800804, 30800805)
HELD_AUDIT_REJECTIONS = (30855187, 30855219)
FAILURE_TEXT = "AF_UNIX path length cannot exceed 107 bytes"


def run(
    command: list[str], *, check: bool = True
) -> subprocess.CompletedProcess[str]:
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


def accounting(job_ids: list[int]) -> dict[int, dict[str, str]]:
    result = run(
        [
            "sacct",
            "-X",
            "-j",
            ",".join(map(str, job_ids)),
            "--starttime",
            "2026-08-20",
            "-n",
            "-P",
            "-o",
            "JobIDRaw,State,ExitCode,Elapsed,NodeList,Reason",
        ]
    )
    observed: dict[int, dict[str, str]] = {}
    for line in result.stdout.splitlines():
        values = line.split("|")
        if len(values) != 6 or not values[0].isdigit():
            continue
        observed[int(values[0])] = {
            "state": values[1],
            "exit_code": values[2],
            "elapsed": values[3],
            "node_list": values[4],
            "reason": values[5],
        }
    missing = set(job_ids) - set(observed)
    if missing:
        raise RuntimeError(
            f"Slurm accounting lacks E113-R4 jobs: {sorted(missing)}"
        )
    return observed


def validate_runner() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    slurm_job_expansion = "$" + "{SLURM_JOB_ID:-manual}"
    required = (
        f'mktemp -d "/tmp/e113r4-{slurm_job_expansion}-XXXXXX"',
        'export TMPDIR="$job_tmp"',
        'export RAY_TMPDIR="$job_tmp"',
        '--bind "$job_tmp:$job_tmp"',
    )
    missing = [contract for contract in required if contract not in source]
    if missing or 'RAY_TMPDIR="$E113R4_OUTPUT/tmp/ray"' in source:
        raise RuntimeError(
            f"short Ray temporary-path contract failed: {missing}"
        )


def validate_ledger() -> tuple[dict[str, Any], dict[int, dict[str, str]]]:
    payload = json.loads(common.LEDGER.read_text(encoding="utf-8"))
    if payload.get("schema") != "e113r4_official_verl_dapo_jobs_v1":
        raise RuntimeError("unexpected E113-R4 ledger schema")
    if payload.get("ray_socket_recovery") is not None:
        raise RuntimeError("E113-R4 Ray socket recovery is already recorded")
    smokes = list(payload.get("smokes", {}).values())
    if sorted(int(row["job_id"]) for row in smokes) != list(FAILED_SMOKES):
        raise RuntimeError("authoritative E113-R4 smoke IDs drifted")
    science = list(payload.get("runs", []))
    if len(science) != 50 or len(
        {int(row["job_id"]) for row in science}
    ) != 50:
        raise RuntimeError("authoritative E113-R4 science graph drifted")

    job_ids = [
        *FAILED_SMOKES,
        *HELD_AUDIT_REJECTIONS,
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
            f"failed smoke accounting drifted: {wrong_smokes}"
        )
    wrong_audit_rejections = {
        job_id: states[job_id]
        for job_id in HELD_AUDIT_REJECTIONS
        if not states[job_id]["state"].startswith("CANCELLED")
        or states[job_id]["elapsed"] != "00:00:00"
    }
    if wrong_audit_rejections:
        raise RuntimeError(
            f"held audit rejection drifted: {wrong_audit_rejections}"
        )
    wrong_science = {
        int(row["job_id"]): states[int(row["job_id"])]
        for row in science
        if states[int(row["job_id"])]["state"] != "CANCELLED"
        or states[int(row["job_id"])]["elapsed"] != "00:00:00"
    }
    if wrong_science:
        raise RuntimeError(
            "dependent science jobs were not zero-runtime canceled: "
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
                f"failed smoke has scientific output: {run_dir}"
            )
    existing_science = [
        str(row["run_dir"])
        for row in science
        if Path(str(row["run_dir"])).exists()
    ]
    if existing_science:
        raise RuntimeError(
            f"canceled science output paths exist: {existing_science}"
        )

    base = Path(str(payload["validation"]["runtime_snapshot"]))
    identity_path = base / "SNAPSHOT_IDENTITY.json"
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    if (
        identity.get("sha256")
        != payload["validation"]["runtime_snapshot_sha256"]
    ):
        raise RuntimeError("base runtime snapshot identity drifted")
    if (
        digest(SLURM_SCRIPT)
        != payload["validation"]["slurm_script_sha256"]
    ):
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
    identity = hashlib.sha256(
        f"{base_identity}\n{runner_hash}\nray-tmp-v1\n".encode("utf-8")
    ).hexdigest()
    target = (
        ROOT
        / "var/artifacts/source_snapshots"
        / f"e113r4_official_verl_dapo_raytmp_{identity[:16]}"
    )
    if target.is_dir():
        metadata = json.loads(
            (target / "SNAPSHOT_IDENTITY.json").read_text()
        )
        if metadata.get("sha256") != identity:
            raise RuntimeError(
                f"recovery snapshot identity drifted: {target}"
            )
        return target, identity

    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{target.name}.",
            dir=target.parent,
        )
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
                    "e113r4_official_verl_dapo_runtime_recovery_v1"
                ),
                "sha256": identity,
                "base_snapshot": str(base),
                "base_snapshot_sha256": base_identity,
                "runner_sha256": runner_hash,
                "upstream_commit": common.VERL_COMMIT,
                "only_runtime_change": (
                    "short node-local TMPDIR/RAY_TMPDIR"
                ),
            },
        )
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return target, identity


def recovery_name(
    row: dict[str, Any], *, smoke: bool
) -> str:
    family_tag = "q05" if row["family"] == "qwen05b" else "f1"
    prefix = "e113r4r1s0" if smoke else "e113r4r1"
    domain_tag = common.DOMAIN_TAGS[str(row["domain"])][:5]
    return f"{prefix}-{family_tag}-{domain_tag}-s{row['seed']}"


def recovery_env(
    row: dict[str, Any],
    snapshot: Path,
    *,
    smoke_output: Path | None = None,
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
    if smoke_output is not None:
        env["E113R4_OUTPUT"] = str(smoke_output)
    return env


def sbatch_command(
    row: dict[str, Any],
    env: dict[str, str],
    *,
    smoke: bool,
    dependency: str | None,
) -> tuple[list[str], Path]:
    name = recovery_name(row, smoke=smoke)
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
        f"--time={'1-00:00:00' if smoke else '7-00:00:00'}",
        "--nice=0",
        f"--output={log}",
        f"--error={log.with_suffix('.err')}",
        "--export=ALL,"
        + ",".join(
            f"{key}={value}" for key, value in env.items()
        ),
    ]
    if smoke:
        command.append(f"--nodelist={A6000_POOL}")
    if dependency:
        command.extend(
            [
                f"--dependency=afterok:{dependency}",
                "--kill-on-invalid-dep=yes",
            ]
        )
    command.append(str(SLURM_SCRIPT))
    return command, log


def submit(command: list[str]) -> int:
    response = (
        run(command).stdout.strip().split(";", maxsplit=1)[0]
    )
    if not response.isdigit():
        raise RuntimeError(
            f"invalid sbatch response: {response!r}"
        )
    job_id = int(response)
    # The site submission hook may normalize allcs requests to cs.
    # R4-R1 retains the approved broad, non-preempting all placement.
    run(
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
    *,
    smoke: bool,
    dependency_ids: list[int] | None,
) -> str:
    record = run(
        [
            "scontrol",
            "show",
            "job",
            "-dd",
            "-o",
            str(job_id),
        ]
    ).stdout.strip()
    expected = {
        "JobState": "PENDING",
        "Reason": "JobHeldUser",
        "Requeue": "1",
        "Account": "allcs",
        "Partition": "all",
        "NumCPUs": "16",
        "MinMemoryNode": "128G",
        "TimeLimit": (
            "1-00:00:00" if smoke else "7-00:00:00"
        ),
        "Nice": "0",
        "TresPerNode": "gres/gpu:a6000:1",
    }
    wrong = {
        key: (value, field(record, key))
        for key, value in expected.items()
        if field(record, key) != value
    }
    if field(record, "NumNodes") not in {"1", "1-1"}:
        wrong["NumNodes"] = (
            "one node",
            field(record, "NumNodes"),
        )
    if (
        smoke
        and field(record, "ReqNodeList") != A6000_POOL
    ):
        wrong["ReqNodeList"] = (
            A6000_POOL,
            field(record, "ReqNodeList"),
        )
    if dependency_ids:
        dependency = field(record, "Dependency") or ""
        missing = [
            job_id
            for job_id in dependency_ids
            if f"afterok:{job_id}" not in dependency
        ]
        if missing:
            wrong["Dependency"] = (
                dependency_ids,
                dependency,
            )
    elif field(record, "Dependency") != "(null)":
        wrong["Dependency"] = (
            "(null)",
            field(record, "Dependency"),
        )
    if wrong:
        raise RuntimeError(
            f"held recovery job {job_id} failed audit: {wrong}"
        )
    missing_env = [
        key
        for key, value in env.items()
        if f"{key}={value}" not in record
    ]
    if missing_env:
        raise RuntimeError(
            f"held recovery job {job_id} lost environment: "
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
    *,
    smoke: bool,
    dependency_ids: list[int] | None,
) -> dict[str, Any]:
    row = deepcopy(old)
    row.update(
        {
            "job_id": job_id,
            "replaces_job_id": int(old["job_id"]),
            "stage": (
                "operational_smoke_r1" if smoke else "full_r1"
            ),
            "run_dir": env["E113R4_OUTPUT"],
            "log_path": str(log_template).replace(
                "%j", str(job_id)
            ),
            "command": command,
            "environment": env,
            "held_scheduler_record": held,
            "recovery": "short_node_local_ray_tmp_v1",
        }
    )
    if not smoke:
        row["dependency_smoke_job_ids"] = list(
            dependency_ids or []
        )
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()

    if not PROTOCOL.is_file():
        raise SystemExit(
            f"missing prospective recovery protocol: {PROTOCOL}"
        )
    validate_runner()
    payload, old_accounting = validate_ledger()
    snapshot, snapshot_identity = recovery_snapshot(payload)
    validation = {
        "protocol": str(PROTOCOL),
        "protocol_sha256": digest(PROTOCOL),
        "base_snapshot": (
            payload["validation"]["runtime_snapshot"]
        ),
        "base_snapshot_sha256": (
            payload["validation"]["runtime_snapshot_sha256"]
        ),
        "recovery_snapshot": str(snapshot),
        "recovery_snapshot_sha256": snapshot_identity,
        "runner_sha256": digest(RUNNER),
        "failed_smoke_job_ids": list(FAILED_SMOKES),
        "held_audit_rejection_job_ids": list(HELD_AUDIT_REJECTIONS),
        "canceled_science_job_count": 50,
        "scientific_parameters_changed": False,
        "official_upstream_changed": False,
    }
    if not args.launch:
        print(json.dumps(validation, indent=2, sort_keys=True))
        print(
            "dry-run only; pass --launch to submit "
            "2 repaired smokes + 50 gated cells"
        )
        return 0

    old_smokes = deepcopy(payload["smokes"])
    old_runs = deepcopy(payload["runs"])
    new_smokes: dict[str, dict[str, Any]] = {}
    new_runs: list[dict[str, Any]] = []
    submitted: list[int] = []
    ledger_written = False
    try:
        for family in common.FAMILIES:
            old = deepcopy(old_smokes[family])
            smoke_output = ROOT / (
                "var/data/"
                f"e113r4r1s0_{family}_"
                f"{common.DOMAIN_TAGS[str(old['domain'])]}_"
                f"official_dapo_s{old['seed']}"
            )
            if smoke_output.exists():
                raise RuntimeError(
                    "refusing existing recovery smoke output: "
                    f"{smoke_output}"
                )
            env = recovery_env(
                old,
                snapshot,
                smoke_output=smoke_output,
            )
            command, log = sbatch_command(
                old,
                env,
                smoke=True,
                dependency=None,
            )
            job_id = submit(command)
            submitted.append(job_id)
            held = audit_held(
                job_id,
                env,
                smoke=True,
                dependency_ids=None,
            )
            new_smokes[family] = make_row(
                old,
                job_id,
                command,
                env,
                log,
                held,
                smoke=True,
                dependency_ids=None,
            )

        smoke_ids = [
            int(new_smokes[family]["job_id"])
            for family in common.FAMILIES
        ]
        dependency = ":".join(map(str, smoke_ids))
        for old in old_runs:
            output = Path(str(old["run_dir"]))
            if output.exists():
                raise RuntimeError(
                    f"refusing existing scientific output: {output}"
                )
            env = recovery_env(old, snapshot)
            command, log = sbatch_command(
                old,
                env,
                smoke=False,
                dependency=dependency,
            )
            job_id = submit(command)
            submitted.append(job_id)
            held = audit_held(
                job_id,
                env,
                smoke=False,
                dependency_ids=smoke_ids,
            )
            new_runs.append(
                make_row(
                    old,
                    job_id,
                    command,
                    env,
                    log,
                    held,
                    smoke=False,
                    dependency_ids=smoke_ids,
                )
            )

        now = datetime.now(timezone.utc).isoformat()
        payload["ray_socket_recovery"] = {
            "schema": (
                "e113r4_r4r1_ray_socket_recovery_v1"
            ),
            "recorded_at": now,
            "root_cause": FAILURE_TEXT,
            **validation,
            "failed_smoke_accounting": {
                str(job_id): old_accounting[job_id]
                for job_id in FAILED_SMOKES
            },
            "held_audit_rejection_accounting": {
                str(job_id): old_accounting[job_id]
                for job_id in HELD_AUDIT_REJECTIONS
            },
            "canceled_science_job_ids": [
                int(row["job_id"]) for row in old_runs
            ],
            "canceled_science_accounting": {
                str(row["job_id"]): old_accounting[
                    int(row["job_id"])
                ]
                for row in old_runs
            },
            "replacement_smoke_job_ids": smoke_ids,
            "replacement_science_job_ids": [
                int(row["job_id"]) for row in new_runs
            ],
            "superseded_smokes": old_smokes,
            "superseded_runs": old_runs,
            "all_replacements_held_and_audited_before_release": True,
            "scientific_outcome_observed_before_recovery": False,
        }
        payload["smokes"] = new_smokes
        payload["runs"] = new_runs
        payload["released"] = False
        payload["validation"]["runtime_snapshot"] = str(
            snapshot
        )
        payload["validation"][
            "runtime_snapshot_sha256"
        ] = snapshot_identity
        payload["validation"][
            "recovery_launcher_sha256"
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
        payload["released_at"] = (
            datetime.now(timezone.utc).isoformat()
        )
        launch.atomic_json(common.LEDGER, payload)
    except BaseException:
        if submitted and not ledger_written:
            run(
                [
                    "scancel",
                    *(str(job_id) for job_id in submitted),
                ],
                check=False,
            )
        raise

    print(
        "released repaired E113-R4 smokes: "
        f"{[row['job_id'] for row in new_smokes.values()]}"
    )
    print(
        "released 50 dependency-gated E113-R4 science jobs: "
        f"{new_runs[0]['job_id']}--{new_runs[-1]['job_id']}"
    )
    print(f"ledger: {common.LEDGER}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
