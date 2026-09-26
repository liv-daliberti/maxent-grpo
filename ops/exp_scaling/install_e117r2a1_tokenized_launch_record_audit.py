#!/usr/bin/env python3
"""Freeze and install E117-R2-A1's tokenized launch-record re-audit."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Iterable


TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
import apply_e117s1_lowprio_partition_repair as scheduler  # noqa: E402
import launch_e117_same_plumbing_component_preflight as e117  # noqa: E402
import launch_e117r2_same_plumbing_repair as e117r2  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / (
    "paper/preregistration/"
    "e117r2a1_tokenized_launch_record_audit_20260830.md"
)
AUDITOR = ROOT / "ops/exp_scaling/audit_e117_same_plumbing_component_preflight.py"
TEST = ROOT / "tests/test_e117_mechanism_audit.py"
LEDGER = ROOT / e117r2.LEDGER
AUDIT_JOB = ROOT / e117r2.AUDIT_JOB
AUDIT_OUTPUT = ROOT / e117r2.AUDIT
HISTORY = ROOT / "var/artifacts/e117r2a1_audit_job_replacement.json"
SUPERSEDED_AUDIT = (
    ROOT / "var/artifacts/e117r2a1_superseded_audit_30977278.json"
)
OLD_AUDIT_JOB_ID = 30977278
DERIVATION_TAG = "e117r2a1-tokenized-launch-record-audit-v1"
AUDITOR_RELATIVE = Path(
    "ops/exp_scaling/audit_e117_same_plumbing_component_preflight.py"
)
IDENTITY_RELATIVE = Path("SNAPSHOT_IDENTITY.json")


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"command failed {command}: {detail}")
    return result


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized_state(value: str) -> str:
    return value.split()[0].split("+", 1)[0]


def accounting(job_ids: Iterable[int]) -> dict[int, dict[str, str]]:
    ordered = list(dict.fromkeys(int(value) for value in job_ids))
    result = run(
        [
            "sacct",
            "-X",
            "-n",
            "-P",
            "-S",
            "2026-08-20",
            "-j",
            ",".join(str(value) for value in ordered),
            "--format=JobIDRaw,State,ExitCode,Elapsed,Restarts,NodeList",
        ]
    )
    rows: dict[int, dict[str, str]] = {}
    for line in result.stdout.splitlines():
        values = line.split("|")
        if len(values) != 6 or not values[0].isdigit():
            continue
        job_id = int(values[0])
        if job_id not in ordered:
            continue
        rows[job_id] = {
            "state": normalized_state(values[1]),
            "exit_code": values[2],
            "elapsed": values[3],
            "restarts": values[4],
            "node_list": values[5],
        }
    missing = set(ordered) - set(rows)
    if missing:
        raise RuntimeError(f"Slurm accounting lacks jobs: {sorted(missing)}")
    return rows


def tree_hash(root: Path, excluded: set[Path]) -> str:
    result = hashlib.sha256()
    for path in sorted(value for value in root.rglob("*") if value.is_file()):
        relative = path.relative_to(root)
        if relative in excluded or "__pycache__" in relative.parts:
            continue
        if path.suffix == ".pyc":
            continue
        result.update(str(relative).encode("utf-8"))
        result.update(b"\0")
        result.update(path.read_bytes())
        result.update(b"\0")
    return result.hexdigest()


def audit_snapshot(base: Path) -> tuple[Path, dict[str, Any]]:
    identity_path = base / IDENTITY_RELATIVE
    if not identity_path.is_file():
        raise RuntimeError(f"E117-R2 base snapshot identity is absent: {identity_path}")
    base_metadata = json.loads(identity_path.read_text(encoding="utf-8"))
    base_identity = str(base_metadata.get("sha256", ""))
    if not base_identity:
        raise RuntimeError("E117-R2 base snapshot identity is invalid")

    audit_sha256 = digest(AUDITOR)
    protocol_sha256 = digest(PROTOCOL)
    derived_identity = hashlib.sha256(
        (
            base_identity
            + "\n"
            + digest(identity_path)
            + "\n"
            + audit_sha256
            + "\n"
            + protocol_sha256
            + "\n"
            + DERIVATION_TAG
            + "\n"
        ).encode("utf-8")
    ).hexdigest()
    target = (
        ROOT
        / "var/artifacts/source_snapshots"
        / f"e117r2a1_audit_{derived_identity[:16]}"
    )
    excluded = {IDENTITY_RELATIVE, AUDITOR_RELATIVE}
    base_unchanged_tree_sha256 = tree_hash(base, excluded)
    metadata = {
        "schema": "e117r2_audit_amendment_snapshot_v1",
        "sha256": derived_identity,
        "base_snapshot": str(base),
        "base_snapshot_identity": base_identity,
        "base_snapshot_identity_sha256": digest(identity_path),
        "base_unchanged_tree_sha256": base_unchanged_tree_sha256,
        "audit_script": str(AUDITOR_RELATIVE),
        "audit_script_sha256": audit_sha256,
        "protocol": str(PROTOCOL),
        "protocol_sha256": protocol_sha256,
        "derivation_tag": DERIVATION_TAG,
        "training_source_changed": False,
        "only_runtime_change": "tokenized E117 launch-record export parsing",
    }

    if target.is_dir():
        actual = json.loads(
            (target / IDENTITY_RELATIVE).read_text(encoding="utf-8")
        )
        if actual != metadata:
            raise RuntimeError(f"E117-R2-A1 audit snapshot identity drifted: {target}")
        if digest(target / AUDITOR_RELATIVE) != audit_sha256:
            raise RuntimeError("E117-R2-A1 frozen auditor drifted")
        if tree_hash(target, excluded) != base_unchanged_tree_sha256:
            raise RuntimeError("E117-R2-A1 non-audit snapshot bytes drifted")
        return target, metadata

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent)
    )
    try:
        shutil.copytree(base, temporary, dirs_exist_ok=True)
        shutil.copy2(AUDITOR, temporary / AUDITOR_RELATIVE)
        e117.e111.e81.atomic_json(temporary / IDENTITY_RELATIVE, metadata)
        if digest(temporary / AUDITOR_RELATIVE) != audit_sha256:
            raise RuntimeError("staged E117-R2-A1 auditor digest drifted")
        if tree_hash(temporary, excluded) != base_unchanged_tree_sha256:
            raise RuntimeError("staged E117-R2-A1 changed non-audit runtime bytes")
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return target, metadata


def schedule_audit(
    snapshot: Path, job_ids: list[int]
) -> tuple[int, str, list[str]]:
    dependency = "afterany:" + ":".join(str(value) for value in job_ids)
    audit_script = snapshot / AUDITOR_RELATIVE
    command = [
        "sbatch",
        "--parsable",
        "--hold",
        f"--dependency={dependency}",
        "--job-name=e117r2a1-audit",
        "--partition=all",
        "--account=allcs",
        "--cpus-per-task=2",
        "--mem=4G",
        "--time=01:00:00",
        "--nice=0",
        "--no-requeue",
        f"--chdir={ROOT}",
        "--output="
        + str(ROOT / "var/artifacts/logs/e117r2a1-audit-%j.out"),
        "--error="
        + str(ROOT / "var/artifacts/logs/e117r2a1-audit-%j.err"),
        "--wrap="
        + f"python {audit_script} --ledger {LEDGER} --output {AUDIT_OUTPUT}",
    ]
    result = run(command)
    raw = result.stdout.strip().split(";", 1)[0]
    if not raw.isdigit():
        raise RuntimeError(f"invalid E117-R2-A1 audit job id: {result.stdout!r}")
    job_id = int(raw)
    try:
        record = scheduler.show(job_id)
        expected = {
            "JobState": "PENDING",
            "Reason": "JobHeldUser",
            "RunTime": "00:00:00",
            "Restarts": "0",
            "Requeue": "0",
            "Partition": "all",
            "Account": "allcs",
            "JobName": "e117r2a1-audit",
        }
        drifted = {
            key: (scheduler.field(record, key), value)
            for key, value in expected.items()
            if scheduler.field(record, key) != value
        }
        required_text = (
            f"--dependency={dependency}",
            str(audit_script),
            str(LEDGER),
            str(AUDIT_OUTPUT),
        )
        absent = [value for value in required_text if value not in record]
        if drifted or absent:
            raise RuntimeError(
                f"E117-R2-A1 held audit drifted: fields={drifted} absent={absent}"
            )
    except Exception:
        subprocess.run(["scancel", str(job_id)], cwd=ROOT, check=False)
        raise
    return job_id, record, command


def validate_trigger(
    ledger: dict[str, Any],
    audit_job: dict[str, Any],
    failed_audit: dict[str, Any],
) -> tuple[list[int], dict[int, dict[str, str]]]:
    runs = list(ledger.get("runs", []))
    job_ids = [int(row["job_id"]) for row in runs]
    if (
        ledger.get("released") is not True
        or len(job_ids) != 12
        or len(set(job_ids)) != 12
        or int(ledger.get("audit_job_id", 0)) != OLD_AUDIT_JOB_ID
        or int(audit_job.get("audit_job_id", 0)) != OLD_AUDIT_JOB_ID
        or [int(value) for value in audit_job.get("dependency_job_ids", [])]
        != job_ids
    ):
        raise RuntimeError("E117-R2 ledger or audit-job identity drifted")
    failures = list(failed_audit.get("failures", []))
    scheduler_failures = [
        value
        for value in failures
        if value.endswith("scheduler record lacks a bounded export block")
    ]
    other_failures = [
        value for value in failures if value not in scheduler_failures
    ]
    if (
        failed_audit.get("schema")
        != "e117_same_plumbing_component_preflight_audit_v3"
        or failed_audit.get("terminal") is not True
        or failed_audit.get("passed") is not False
        or failed_audit.get("efficacy_outcomes_used") is not False
        or failed_audit.get("endpoint_contrasts_computed") is not False
        or len(scheduler_failures) != 11
        or other_failures
        != ["qwen05b/countdown: launch block lacks C/P/F"]
    ):
        raise RuntimeError("E117-R2-A1 audit-only trigger evidence drifted")

    states = accounting(job_ids + [OLD_AUDIT_JOB_ID])
    bad_training = {
        value: states[value]
        for value in job_ids
        if states[value]["state"] != "COMPLETED"
        or states[value]["exit_code"] != "0:0"
    }
    if bad_training:
        raise RuntimeError(f"E117-R2 training completion drifted: {bad_training}")
    old_state = states[OLD_AUDIT_JOB_ID]
    if old_state["state"] != "FAILED" or old_state["exit_code"] != "1:0":
        raise RuntimeError(f"E117-R2 failed audit accounting drifted: {old_state}")
    return job_ids, states


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install", action="store_true")
    args = parser.parse_args()
    if not args.install:
        raise SystemExit("pass --install to freeze and submit E117-R2-A1")
    for required in (PROTOCOL, AUDITOR, TEST, LEDGER, AUDIT_JOB, AUDIT_OUTPUT):
        if not required.is_file():
            raise SystemExit(f"required E117-R2-A1 input is absent: {required}")
    existing = [value for value in (HISTORY, SUPERSEDED_AUDIT) if value.exists()]
    if existing:
        raise SystemExit(f"refusing duplicate E117-R2-A1 installation: {existing}")

    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    audit_job = json.loads(AUDIT_JOB.read_text(encoding="utf-8"))
    failed_audit = json.loads(AUDIT_OUTPUT.read_text(encoding="utf-8"))
    job_ids, states = validate_trigger(ledger, audit_job, failed_audit)
    base_snapshot = Path(str(ledger.get("snapshot_root", "")))
    if (
        not base_snapshot.is_dir()
        or digest(base_snapshot / IDENTITY_RELATIVE)
        != ledger.get("snapshot_identity_sha256")
    ):
        raise SystemExit("E117-R2 immutable training snapshot identity drifted")

    test_command = [sys.executable, "-m", "pytest", "-q", str(TEST)]
    test_result = run(test_command)
    audit_snapshot_root, snapshot_metadata = audit_snapshot(base_snapshot)
    new_job_id = 0
    released = False
    ledger_before = deepcopy(ledger)
    audit_job_before = deepcopy(audit_job)
    try:
        new_job_id, new_record, submit_command = schedule_audit(
            audit_snapshot_root, job_ids
        )
        shutil.copy2(AUDIT_OUTPUT, SUPERSEDED_AUDIT)
        if digest(SUPERSEDED_AUDIT) != digest(AUDIT_OUTPUT):
            raise RuntimeError("superseded E117-R2 audit archive drifted")

        superseded = list(audit_job.get("superseded_audit_job_ids", []))
        superseded.append(OLD_AUDIT_JOB_ID)
        superseded = list(dict.fromkeys(int(value) for value in superseded))
        audit_script = audit_snapshot_root / AUDITOR_RELATIVE
        history = {
            "schema": "e117r2a1_audit_job_replacement_v1",
            "protocol": str(PROTOCOL),
            "protocol_sha256": digest(PROTOCOL),
            "previous_audit_job_id": OLD_AUDIT_JOB_ID,
            "previous_audit_artifact": str(SUPERSEDED_AUDIT),
            "previous_audit_artifact_sha256": digest(SUPERSEDED_AUDIT),
            "previous_audit_state": states[OLD_AUDIT_JOB_ID],
            "replacement_audit_job_id": new_job_id,
            "replacement_scheduler_record": new_record,
            "replacement_submit_command": submit_command,
            "dependency_job_ids": job_ids,
            "training_job_states": {
                str(value): states[value] for value in job_ids
            },
            "training_snapshot_root": str(base_snapshot),
            "training_snapshot_identity_sha256": digest(
                base_snapshot / IDENTITY_RELATIVE
            ),
            "audit_snapshot_root": str(audit_snapshot_root),
            "audit_snapshot_identity": snapshot_metadata,
            "audit_snapshot_identity_sha256": digest(
                audit_snapshot_root / IDENTITY_RELATIVE
            ),
            "audit_script": str(audit_script),
            "audit_script_sha256": digest(audit_script),
            "regression_test_command": test_command,
            "regression_test_stdout": test_result.stdout.strip(),
            "regression_test_stderr": test_result.stderr.strip(),
            "training_jobs_changed": False,
            "training_configuration_changed": False,
            "efficacy_outcomes_used": False,
            "endpoint_contrasts_computed": False,
            "released": False,
        }
        amendment_history = list(ledger.get("audit_amendment_history", []))
        prior_amendment = ledger.get("audit_amendment")
        if prior_amendment and prior_amendment not in amendment_history:
            amendment_history.append(prior_amendment)
        amendment_history.append(str(PROTOCOL))
        ledger.update(
            {
                "audit_job_id": new_job_id,
                "audit_script": str(audit_script),
                "audit_script_sha256": digest(audit_script),
                "audit_snapshot_root": str(audit_snapshot_root),
                "audit_snapshot_identity_sha256": digest(
                    audit_snapshot_root / IDENTITY_RELATIVE
                ),
                "audit_amendment": str(PROTOCOL),
                "audit_amendment_sha256": digest(PROTOCOL),
                "audit_amendment_history": amendment_history,
                "superseded_audit_job_ids": superseded,
                "superseded_audit_artifact": str(SUPERSEDED_AUDIT),
                "superseded_audit_artifact_sha256": digest(SUPERSEDED_AUDIT),
            }
        )
        audit_job.update(
            {
                "schema": "e117r2_same_plumbing_component_preflight_audit_job_v2",
                "audit_job_id": new_job_id,
                "audit_script": str(audit_script),
                "audit_script_sha256": digest(audit_script),
                "audit_snapshot_root": str(audit_snapshot_root),
                "audit_snapshot_identity_sha256": digest(
                    audit_snapshot_root / IDENTITY_RELATIVE
                ),
                "scheduler_record": new_record,
                "supersedes_audit_job_id": OLD_AUDIT_JOB_ID,
                "superseded_audit_job_ids": superseded,
                "audit_protocol": str(PROTOCOL),
                "audit_protocol_sha256": digest(PROTOCOL),
                "released": False,
            }
        )
        e117.e111.e81.atomic_json(HISTORY, history)
        e117.e111.e81.atomic_json(LEDGER, ledger)
        e117.e111.e81.atomic_json(AUDIT_JOB, audit_job)

        run(["scontrol", "release", str(new_job_id)])
        released = True
        history["released"] = True
        ledger["audit_released"] = True
        audit_job["released"] = True
        e117.e111.e81.atomic_json(HISTORY, history)
        e117.e111.e81.atomic_json(LEDGER, ledger)
        e117.e111.e81.atomic_json(AUDIT_JOB, audit_job)
    except Exception:
        if new_job_id and not released:
            subprocess.run(["scancel", str(new_job_id)], cwd=ROOT, check=False)
            e117.e111.e81.atomic_json(LEDGER, ledger_before)
            e117.e111.e81.atomic_json(AUDIT_JOB, audit_job_before)
            for value in (HISTORY, SUPERSEDED_AUDIT):
                if value.exists():
                    value.unlink()
        raise

    print(
        f"[e117r2a1] previous={OLD_AUDIT_JOB_ID} replacement={new_job_id} "
        f"training_jobs=12 snapshot={audit_snapshot_root}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
