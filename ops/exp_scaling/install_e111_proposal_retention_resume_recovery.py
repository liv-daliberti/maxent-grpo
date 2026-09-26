#!/usr/bin/env python3
"""Install and record the outcome-blind E111 proposal-retention resume fix."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "var/artifacts/e111_verified_support_discovery_mechanism_gate_jobs.json"
PROTOCOL = ROOT / "paper/preregistration/e111_proposal_retention_resume_recovery_amendment_20260818.md"
RECORD = ROOT / "var/artifacts/e111_proposal_retention_resume_recovery.json"
RUNTIME = ROOT / "var/artifacts/source_snapshots/e76_tuned_scale_2febfdc12e36650d"
PRODUCTION_PYTHON = ROOT / "var/seed_paper_eval/paper310/bin/python"
PRODUCTION_LIB = ROOT / "var/seed_paper_eval/paper310/lib"
FILES = (
    "src/oat_drgrpo/admission_retention.py",
    "src/oat_drgrpo/online_canonical_bank.py",
)
BEFORE_SHA256 = {
    "src/oat_drgrpo/admission_retention.py": "c555846c9c0cb1f6ea1591cb29414c3e06afa367c94f8a11ea2761173aeb91e1",
    "src/oat_drgrpo/online_canonical_bank.py": "86316337cd7497030ba8813a220adf8e9591038cd7d750f735b3617f1f70edee",
}
AFTER_SHA256 = {
    "src/oat_drgrpo/admission_retention.py": "beeb6b758cea2fb0afe7d5bf37d0f815197ad8d00f2ec1e5143aa238405aea4c",
    "src/oat_drgrpo/online_canonical_bank.py": "e066538d32bdaedf7f7cce6956766e409bf6844c567090a5b79b7fa3f05c5e4b",
}
FAILURE_MARKER = "proposal retention state refers to a non-proposal exemplar"
EXPECTED_FAILURE_JOBS = {30674729, 30674733, 30674754}
EXPECTED_TEST_SUMMARY = "24 passed"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as sink:
            json.dump(payload, sink, indent=2, sort_keys=True)
            sink.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_install(source: Path, target: Path) -> None:
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as sink:
            sink.write(source.read_bytes())
            sink.flush()
            os.fsync(sink.fileno())
        os.chmod(temporary, source.stat().st_mode & 0o777)
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> int:
    if RECORD.exists():
        raise SystemExit(f"refusing duplicate E111 recovery installation: {RECORD}")
    if not LEDGER.is_file() or not PROTOCOL.is_file():
        raise SystemExit("E111 recovery prerequisites are absent")
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    if ledger.get("released") is not True or ledger.get("pointmaze") != "excluded":
        raise SystemExit("E111 ledger identity drifted")
    runs = list(ledger.get("runs", []))
    if len(runs) != 15:
        raise SystemExit("E111 recovery expected exactly 15 released cells")

    failures: list[dict[str, object]] = []
    for run in runs:
        stdout = Path(str(run.get("stdout", "")))
        value = stdout.read_text(encoding="utf-8", errors="replace") if stdout.is_file() else ""
        occurrences = value.count(FAILURE_MARKER)
        if occurrences:
            failures.append(
                {
                    "job_id": int(run["job_id"]),
                    "scale": str(run["scale"]),
                    "domain": str(run["domain"]),
                    "occurrences": occurrences,
                }
            )
    observed_ids = {int(row["job_id"]) for row in failures}
    if not EXPECTED_FAILURE_JOBS.issubset(observed_ids):
        raise SystemExit("E111 recovery trigger evidence is incomplete")

    before = {relative: digest(RUNTIME / relative) for relative in FILES}
    root_after = {relative: digest(ROOT / relative) for relative in FILES}
    if before != BEFORE_SHA256:
        raise SystemExit(f"E111 runtime source before-digest drifted: {before}")
    if root_after != AFTER_SHA256:
        raise SystemExit(f"tested root recovery source digest drifted: {root_after}")

    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "src")
    existing = environment.get("LD_LIBRARY_PATH", "")
    environment["LD_LIBRARY_PATH"] = str(PRODUCTION_LIB) + (
        f":{existing}" if existing else ""
    )
    command = [
        str(PRODUCTION_PYTHON),
        "-m",
        "pytest",
        "-q",
        "tests/test_online_canonical_bank.py",
    ]
    result = subprocess.run(
        command, cwd=ROOT, env=environment, capture_output=True, text=True, check=False
    )
    if result.returncode != 0 or EXPECTED_TEST_SUMMARY not in result.stdout:
        raise SystemExit("E111 recovery regression suite did not pass")

    # Install the tracker accessor first. Old bank code is compatible with it;
    # the reverse transient order could expose new bank code to an old tracker.
    for relative in FILES:
        atomic_install(ROOT / relative, RUNTIME / relative)
    installed = {relative: digest(RUNTIME / relative) for relative in FILES}
    if installed != AFTER_SHA256:
        raise SystemExit("E111 recovery runtime installation digest mismatch")

    payload: dict[str, object] = {
        "schema": "e111_proposal_retention_resume_recovery_v1",
        "ledger": str(LEDGER),
        "ledger_sha256": digest(LEDGER),
        "protocol": str(PROTOCOL),
        "protocol_sha256": digest(PROTOCOL),
        "runtime_snapshot": str(RUNTIME),
        "runtime_snapshot_identity_sha256": digest(RUNTIME / "SNAPSHOT_IDENTITY.json"),
        "files": list(FILES),
        "before_sha256": before,
        "after_sha256": installed,
        "root_after_sha256": root_after,
        "test_file": str(ROOT / "tests/test_online_canonical_bank.py"),
        "test_file_sha256": digest(ROOT / "tests/test_online_canonical_bank.py"),
        "test_command": command,
        "test_returncode": result.returncode,
        "test_stdout": result.stdout,
        "test_stderr": result.stderr,
        "failure_marker": FAILURE_MARKER,
        "trigger_failures": failures,
        "all_e111_job_ids": [int(run["job_id"]) for run in runs],
        "runtime_source_changed": True,
        "checkpoint_deserialization_only": True,
        "checkpoint_schema_changed": False,
        "optimizer_update_changed": False,
        "treatment_changed": False,
        "environment_changed": False,
        "jobs_signaled": False,
        "jobs_reset": False,
        "outcomes_inspected": False,
        "pointmaze": "excluded",
        "installed": True,
    }
    atomic_json(RECORD, payload)
    print(f"[e111-resume-recovery] installed=True failures={len(failures)} record={RECORD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
