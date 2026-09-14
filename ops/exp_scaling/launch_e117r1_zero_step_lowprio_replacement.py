#!/usr/bin/env python3
"""Replace E117's zero-step unschedulable jobs on explicit lowprio."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import apply_e117s1_lowprio_partition_repair as e117s1  # noqa: E402
import launch_e117_same_plumbing_component_preflight as e117  # noqa: E402


ORIGINAL_AUDIT_JOB_ID = 30873569
PROTOCOL = (
    "paper/preregistration/e117r1_zero_step_lowprio_replacement_20260824.md"
)
LEDGER = "var/artifacts/e117r1_same_plumbing_component_preflight_jobs.json"
RETIREMENT = "var/artifacts/e117r1_zero_step_lowprio_replacement.json"
AUDIT = "var/artifacts/e117r1_same_plumbing_component_preflight_audit.json"
AUDIT_JOB = "var/artifacts/e117r1_same_plumbing_component_preflight_audit_job.json"


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"command failed {command}: {result.stderr.strip()}")
    return result


def accounting(job_ids: list[int]) -> dict[int, dict[str, Any]]:
    result = run(
        [
            "sacct",
            "-X",
            "-n",
            "-P",
            "-j",
            ",".join(str(job_id) for job_id in job_ids),
            "--format=JobIDRaw,State,Elapsed,ExitCode,Restarts",
        ]
    )
    rows: dict[int, dict[str, Any]] = {}
    for line in result.stdout.splitlines():
        parts = line.split("|")
        if len(parts) < 5 or not parts[0].isdigit():
            continue
        job_id = int(parts[0])
        if job_id in job_ids:
            rows[job_id] = {
                "state": parts[1].split()[0].split("+", 1)[0],
                "elapsed": parts[2],
                "exit_code": parts[3],
                "restarts": int(parts[4] or 0),
            }
    return rows


def replacement_command(command: list[str]) -> list[str]:
    replaced = [
        "--partition=lowprio" if token.startswith("--partition=") else token
        for token in command
    ]
    if not any(token == "--partition=lowprio" for token in replaced):
        raise RuntimeError("replacement command lacks lowprio")
    return replaced


def schedule_audit(
    *, root: Path, snapshot: Path, ledger: Path, job_ids: list[int]
) -> tuple[int, str]:
    dependency = "afterany:" + ":".join(str(job_id) for job_id in job_ids)
    audit_script = (
        snapshot
        / "ops/exp_scaling/audit_e117_same_plumbing_component_preflight.py"
    )
    output = root / AUDIT
    command = [
        "sbatch",
        "--parsable",
        f"--dependency={dependency}",
        "--job-name=e117r1-audit",
        "--partition=all",
        "--account=allcs",
        "--cpus-per-task=2",
        "--mem=4G",
        "--time=01:00:00",
        "--nice=0",
        "--no-requeue",
        f"--chdir={root}",
        f"--output={root / 'var/artifacts/logs/e117r1-audit-%j.out'}",
        f"--error={root / 'var/artifacts/logs/e117r1-audit-%j.err'}",
        "--wrap="
        + f"python {audit_script} --ledger {ledger} --output {output}",
    ]
    result = run(command)
    raw = result.stdout.strip().split(";", 1)[0]
    if not raw.isdigit():
        raise RuntimeError(f"invalid E117-R1 audit job id: {result.stdout!r}")
    job_id = int(raw)
    record = e117s1.show(job_id)
    required = (
        "JobState=PENDING",
        "Reason=Dependency",
        "Requeue=0",
        "Account=allcs",
        "Partition=all",
        f"Dependency={dependency.replace(':', '(unfulfilled),afterany:', 1)}",
    )
    # Dependency formatting is site-specific; exact job IDs and script path are
    # checked directly below instead of depending on the synthetic string.
    del required
    if (
        e117s1.field(record, "JobState") != "PENDING"
        or e117s1.field(record, "Reason") != "Dependency"
        or e117s1.field(record, "Requeue") != "0"
        or e117s1.field(record, "Account") != "allcs"
        or e117s1.field(record, "Partition") != "all"
        or str(audit_script) not in record
        or any(str(value) not in e117s1.field(record, "Dependency") for value in job_ids)
    ):
        raise RuntimeError("E117-R1 audit dependency failed held inspection")
    return job_id, record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    if not args.submit:
        raise SystemExit("pass --submit to install E117-R1")
    root = e117.repo_root()
    original_ledger_path = root / e117.LEDGER
    ledger_path = root / LEDGER
    retirement_path = root / RETIREMENT
    audit_job_path = root / AUDIT_JOB
    protocol_path = root / PROTOCOL
    for path in (ledger_path, retirement_path, audit_job_path):
        if path.exists():
            raise SystemExit(f"refusing duplicate E117-R1 installation: {path}")
    original = json.loads(original_ledger_path.read_text(encoding="utf-8"))
    original_runs = list(original.get("runs", []))
    if original.get("released") is not True or len(original_runs) != 12:
        raise SystemExit("original E117 ledger is incomplete")
    snapshot = Path(str(original["snapshot_root"]))
    planned = e117.planned_cells(root, snapshot)
    by_identity = {
        (str(row["scale"]), str(row["domain"]), str(row["arm"])): row
        for row in original_runs
    }
    original_ids = sorted(int(row["job_id"]) for row in original_runs)
    states = accounting(original_ids + [ORIGINAL_AUDIT_JOB_ID])
    originals_already_retired = all(
        states.get(job_id, {}).get("state") == "CANCELLED"
        and states.get(job_id, {}).get("elapsed") == "00:00:00"
        and states.get(job_id, {}).get("restarts") == 0
        for job_id in original_ids
    )
    if originals_already_retired:
        if states.get(ORIGINAL_AUDIT_JOB_ID, {}).get("state") != "CANCELLED":
            raise SystemExit("original E117 audit was not retired with its jobs")
        if any(Path(str(row["run_dir"])).exists() for row in original_runs):
            raise SystemExit("retired original E117 unexpectedly has a run directory")
    else:
        for row in original_runs:
            identity = (str(row["scale"]), str(row["domain"]), str(row["arm"]))
            job_id = int(row["job_id"])
            record = e117s1.show(job_id)
            expected_environment = e117s1.environment(
                str(row["held_scheduler_record"])
            )
            e117s1.assert_zero_runtime(
                record,
                node=str(row["node"]),
                expected_environment=expected_environment,
            )
            if (
                e117s1.field(record, "Partition") != "mltheory"
                or e117s1.field(record, "Reason") != "JobHeldAdmin"
                or states.get(job_id, {}).get("elapsed") != "00:00:00"
                or states.get(job_id, {}).get("restarts") != 0
                or Path(str(row["run_dir"])).exists()
                or identity not in by_identity
            ):
                raise SystemExit(
                    f"original E117 zero-step boundary drifted: {job_id}"
                )
        audit_record = e117s1.show(ORIGINAL_AUDIT_JOB_ID)
        if (
            e117s1.field(audit_record, "JobState") != "PENDING"
            or e117s1.field(audit_record, "Reason") != "Dependency"
        ):
            raise SystemExit("original E117 audit job is no longer stale/pending")

    submitted: list[int] = []
    replacements: list[dict[str, Any]] = []
    originals_retired = originals_already_retired
    replacements_released = False
    try:
        for cell in planned:
            identity = (
                str(cell["scale"]),
                str(cell["domain"]),
                str(cell["arm"]),
            )
            original_row = by_identity[identity]
            command = replacement_command(list(cell["command"]))
            result = run(command)
            raw = result.stdout.strip().split(";", 1)[0]
            if not raw.isdigit():
                raise RuntimeError(f"invalid replacement job id: {result.stdout!r}")
            job_id = int(raw)
            submitted.append(job_id)
            held = e117.held_job_audit(
                str(job_id),
                scale=identity[0],
                domain=identity[1],
                arm=identity[2],
                snapshot=snapshot,
            )
            if e117s1.field(held, "Partition") != "lowprio":
                raise RuntimeError(f"replacement {job_id} is not lowprio")
            original_environment = e117s1.environment(
                str(original_row["held_scheduler_record"])
            )
            if e117s1.environment(held) != original_environment:
                raise RuntimeError(f"replacement {job_id} environment drifted")
            replacements.append(
                {
                    key: cell[key]
                    for key in (
                        "scale",
                        "model_tag",
                        "domain",
                        "seed",
                        "arm",
                        "arm_label",
                        "node",
                        "run_stamp",
                        "run_dir",
                    )
                }
                | {
                    "job_id": job_id,
                    "original_job_id": int(original_row["job_id"]),
                    "held_scheduler_record": held,
                    "scientific_environment_sha256": e117s1.sha256_text(
                        original_environment
                    ),
                }
            )
        if len(replacements) != 12:
            raise RuntimeError("E117-R1 did not hold exactly 12 replacements")

        replacement_payload = dict(original)
        replacement_payload.update(
            {
                "protocol": str(protocol_path),
                "protocol_sha256": e117.digest(protocol_path),
                "launcher": str(Path(__file__).resolve()),
                "launcher_sha256": e117.digest(Path(__file__)),
                "original_ledger": str(original_ledger_path),
                "original_ledger_sha256": e117.digest(original_ledger_path),
                "original_job_ids": original_ids,
                "replacement_partition": "lowprio",
                "scheduler_only_replacement": True,
                "runs": replacements,
                "released": False,
            }
        )
        retirement_payload = {
            "schema": "e117r1_zero_step_lowprio_replacement_v1",
            "protocol": str(protocol_path),
            "protocol_sha256": e117.digest(protocol_path),
            "original_ledger": str(original_ledger_path),
            "original_ledger_sha256": e117.digest(original_ledger_path),
            "original_audit_job_id": ORIGINAL_AUDIT_JOB_ID,
            "exact_original_job_ids": original_ids,
            "exact_replacement_job_ids": submitted,
            "mapping": [
                {
                    "original_job_id": row["original_job_id"],
                    "replacement_job_id": row["job_id"],
                    "scale": row["scale"],
                    "domain": row["domain"],
                    "arm": row["arm"],
                    "scientific_environment_sha256": row[
                        "scientific_environment_sha256"
                    ],
                }
                for row in replacements
            ],
            "originals_zero_runtime": True,
            "scientific_environment_changed": False,
            "partition_changed_only": True,
            "outcomes_inspected": False,
            "pointmaze": "excluded",
            "installed": False,
        }
        e117.e111.e81.atomic_json(ledger_path, replacement_payload)
        e117.e111.e81.atomic_json(retirement_path, retirement_payload)
        if not originals_retired:
            run(["scancel", str(ORIGINAL_AUDIT_JOB_ID)])
            run(["scancel", *[str(job_id) for job_id in original_ids]])
            retired_states = accounting(original_ids + [ORIGINAL_AUDIT_JOB_ID])
            if any(
                retired_states.get(job_id, {}).get("state") != "CANCELLED"
                for job_id in original_ids + [ORIGINAL_AUDIT_JOB_ID]
            ):
                raise RuntimeError(
                    "E117 original retirement did not become terminal"
                )
            originals_retired = True

        for job_id in submitted:
            run(["scontrol", "release", str(job_id)])
        replacements_released = True
        replacement_payload["released"] = True
        retirement_payload["installed"] = True
        e117.e111.e81.atomic_json(ledger_path, replacement_payload)
        e117.e111.e81.atomic_json(retirement_path, retirement_payload)

        audit_job_id, new_audit_record = schedule_audit(
            root=root,
            snapshot=snapshot,
            ledger=ledger_path,
            job_ids=submitted,
        )
        replacement_payload["audit_job_id"] = audit_job_id
        retirement_payload["replacement_audit_job_id"] = audit_job_id
        audit_payload = {
            "schema": "e117r1_same_plumbing_component_preflight_audit_job_v1",
            "audit_job_id": audit_job_id,
            "dependency_job_ids": submitted,
            "ledger": str(ledger_path),
            "audit_output": str(root / AUDIT),
            "scheduler_record": new_audit_record,
            "scientific_configuration_changed": False,
        }
        e117.e111.e81.atomic_json(ledger_path, replacement_payload)
        e117.e111.e81.atomic_json(retirement_path, retirement_payload)
        e117.e111.e81.atomic_json(audit_job_path, audit_payload)
    except Exception:
        if not replacements_released:
            e117.e111.e81.cancel([str(job_id) for job_id in submitted])
            for path in (ledger_path, retirement_path, audit_job_path):
                if path.exists():
                    path.unlink()
        raise

    print(
        f"[e117r1] replacements=12 released=12 "
        f"jobs={','.join(str(job_id) for job_id in submitted)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
