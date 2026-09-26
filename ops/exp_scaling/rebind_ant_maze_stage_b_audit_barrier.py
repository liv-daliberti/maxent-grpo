#!/usr/bin/env python3
"""Rebind the live AntMaze audit to an explicit ten-job AND barrier."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[2]
IDENTITY = ROOT / "var/artifacts/ant_maze_stage_b_05b_12pass_identity.json"
SUBMISSION = ROOT / "var/artifacts/ant_maze_stage_b_05b_12pass_submission.json"
RUNNER = ROOT / "var/artifacts/ant_maze_stage_b_05b_12pass_audit_runner_identity.json"
MANIFEST = ROOT / "var/artifacts/ant_maze_stage_b_05b_12pass_jobs.tsv"
OUTPUT = ROOT / "var/artifacts/ant_maze_stage_b_05b_12pass_audit.json"
PROTOCOL = ROOT / "paper/preregistration/ant_maze_stage_b_audit_runtime_repair_r2_20260730.md"
OLD_AUDIT_JOB = 30203042
ARMS = ("grpo", "verified_first_global_replay_canonical")
SEEDS = (43, 44, 45, 46, 47)


def run(command: Sequence[str], *, check: bool = True) -> str:
    result = subprocess.run(
        list(command), cwd=ROOT, check=check, capture_output=True, text=True
    )
    return result.stdout.strip()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic(path: Path, payload: Any) -> None:
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, allow_nan=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def validate() -> tuple[dict[str, Any], dict[str, Any]]:
    for path in (IDENTITY, SUBMISSION, RUNNER, MANIFEST, PROTOCOL):
        if not path.is_file():
            raise FileNotFoundError(path)
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    identity = json.loads(IDENTITY.read_text())
    submission = json.loads(SUBMISSION.read_text())
    jobs = identity.get("jobs", {})
    expected = {f"{arm}/s{seed}" for seed in SEEDS for arm in ARMS}
    if (
        identity.get("schema") != "ant-maze-stage-b-05b-12pass-identity-v1"
        or set(jobs) != expected
        or not all(isinstance(value, int) for value in jobs.values())
        or identity.get("audit_job_id") != OLD_AUDIT_JOB
        or submission.get("jobs") != jobs
        or submission.get("identity_sha256") != sha(IDENTITY)
    ):
        raise RuntimeError("AntMaze identity is not the frozen live cohort")
    accounting = run([
        "sacct", "-X", "-j", str(OLD_AUDIT_JOB),
        "--format=State,ExitCode", "-n", "-P",
    ])
    errors = ROOT / f"var/artifacts/logs/audit-ant-stage-b-{OLD_AUDIT_JOB}.err"
    if (
        "FAILED|1:0" not in accounting
        or not errors.is_file()
        or "ant_waypoint_v11.evaluation.json" not in errors.read_text(errors="replace")
    ):
        raise RuntimeError("original AntMaze audit lacks the frozen runtime failure")
    return identity, submission


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("config", "run"))
    args = parser.parse_args()
    identity, submission = validate()
    if args.phase == "config":
        print("[ant-stage-b-audit-barrier-r1] configuration passed; no job submitted")
        return
    jobs = identity["jobs"]
    dependency = ",".join(
        f"afterany:{jobs[f'{arm}/s{seed}']}" for seed in SEEDS for arm in ARMS
    )
    execution_root = Path(identity["execution_root"])
    source_root = Path(identity["source_root"])
    batch = execution_root / "audit_ant_maze_stage_b_05b_12pass.slurm"
    if not batch.is_file():
        raise FileNotFoundError(batch)
    output = run([
        "sbatch", "--parsable", "--hold", f"--dependency={dependency}",
        "--export=ALL,"
        f"ROOT_DIR={ROOT},OAT_ZERO_REPO_ROOT={ROOT},OAT_ZERO_SOURCE_ROOT={source_root},OAT_ZERO_EXECUTION_ROOT={execution_root}",
        str(batch),
    ])
    job_id = int(output.split(";", 1)[0])
    try:
        record = run(["scontrol", "show", "job", "-o", str(job_id)])
        if "JobState=PENDING" not in record or "Reason=JobHeldUser" not in record:
            raise RuntimeError("replacement AntMaze audit is not held")
        archives = {
            IDENTITY: ROOT / "var/artifacts/ant_maze_stage_b_05b_12pass_pre_audit_runtime_repair_identity.json",
            SUBMISSION: ROOT / "var/artifacts/ant_maze_stage_b_05b_12pass_pre_audit_runtime_repair_submission.json",
            RUNNER: ROOT / "var/artifacts/ant_maze_stage_b_05b_12pass_pre_audit_runtime_repair_audit_runner_identity.json",
        }
        for source, target in archives.items():
            if target.exists():
                raise FileExistsError(target)
            shutil.copy2(source, target)
        identity = dict(identity)
        identity.update({
            "audit_job_id": job_id,
            "audit_dependency": dependency,
            "audit_barrier_repair": {
                "schema": "ant-maze-stage-b-audit-runtime-repair-r2",
                "old_audit_job_id": OLD_AUDIT_JOB,
                "new_audit_job_id": job_id,
                "protocol_sha256": sha(PROTOCOL),
                "launcher_sha256": sha(Path(__file__).resolve()),
                "training_jobs_changed": False,
                "training_jobs_all_completed_0_0": True,
                "old_audit_failure": "missing OAT_ZERO_REPO_ROOT",
            },
        })
        atomic(IDENTITY, identity)
        submission = dict(submission)
        submission["identity_sha256"] = sha(IDENTITY)
        submission["audit_barrier_repair"] = "r2"
        atomic(SUBMISSION, submission)
        atomic(RUNNER, {
            "schema": "ant-maze-stage-b-05b-12pass-audit-runner-v1",
            "audit_job_id": job_id,
            "dependency": dependency,
            "identity_sha256": sha(IDENTITY),
            "execution_hash": identity["execution_hash"],
            "audit_barrier_repair": "r2",
        })
        run(["scontrol", "release", str(job_id)])
    except BaseException:
        run(["scancel", str(job_id)], check=False)
        raise
    print(f"[ant-stage-b-audit-barrier-r1] submitted audit_job={job_id}")


if __name__ == "__main__":
    main()
