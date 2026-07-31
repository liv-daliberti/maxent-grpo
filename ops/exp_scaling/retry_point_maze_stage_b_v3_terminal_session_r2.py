#!/usr/bin/env python3
"""Restart all ten PointMaze cells after the terminal-session leak repair."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[2]
PYTHON = ROOT / "var/seed_paper_eval/paper310/bin/python"
IDENTITY = ROOT / "var/artifacts/point_maze_stage_b_05b_12pass_identity.json"
SUBMISSION = ROOT / "var/artifacts/point_maze_stage_b_05b_12pass_submission.json"
MANIFEST = ROOT / "var/artifacts/point_maze_stage_b_05b_12pass_jobs.tsv"
RUNNER = ROOT / "var/artifacts/point_maze_stage_b_05b_12pass_audit_runner_identity.json"
OUTPUT = ROOT / "var/artifacts/point_maze_stage_b_05b_12pass_audit.json"
INITIAL_IDENTITY = ROOT / "var/artifacts/point_maze_stage_b_05b_12pass_initial_attempt_identity.json"
PROTOCOL = ROOT / "paper/preregistration/point_maze_stage_b_v3_terminal_session_repair_r2_20260730.md"
CONTROL = "grpo"
TREATMENT = "verified_first_global_replay_canonical"
ARMS = (CONTROL, TREATMENT)
SEEDS = (43, 44, 45, 46, 47)
INITIAL_JOBS = {
    f"{CONTROL}/s43": 30203021,
    f"{TREATMENT}/s43": 30203022,
    f"{CONTROL}/s44": 30203023,
    f"{TREATMENT}/s44": 30203024,
    f"{CONTROL}/s45": 30203025,
    f"{TREATMENT}/s45": 30203026,
    f"{CONTROL}/s46": 30203027,
    f"{TREATMENT}/s46": 30203028,
    f"{CONTROL}/s47": 30203029,
    f"{TREATMENT}/s47": 30203030,
}
PARTIAL_UPDATES = {
    30203021: 62, 30203022: 62,
    30203023: 64, 30203024: 64,
    30203025: 64, 30203026: 64,
    30203027: 60, 30203028: 60,
    30203029: 60, 30203030: 60,
}
PLACEMENT_JOBS = {
    f"{CONTROL}/s46": 30203550,
    f"{TREATMENT}/s46": 30203551,
    f"{CONTROL}/s47": 30203552,
    f"{TREATMENT}/s47": 30203553,
}
EXECUTION_HASH = "a54ebba3a4d8e905d2d109ea68072b31c100414521ca5183d5d188bae26bf0d0"
TIMEOUT_TEXT = "interactive PointMaze worker request timed out"


def run(
    command: Sequence[str], *, check: bool = True, env: dict[str, str] | None = None
) -> str:
    result = subprocess.run(
        list(command), cwd=ROOT, check=check, capture_output=True, text=True, env=env
    )
    return result.stdout.strip()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def atomic(path: Path, payload: Any) -> None:
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, allow_nan=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def cell_paths(label: str) -> dict[str, Path]:
    arm, seed_label = label.split("/s", 1)
    stem = ROOT / f"var/artifacts/point_maze_stage_b_05b_12pass_{arm}_s{seed_label}"
    return {
        "receipt": Path(str(stem) + ".json"),
        "metrics": Path(str(stem) + ".metrics.jsonl"),
        "state_replay": Path(str(stem) + ".state_replay.jsonl"),
    }


def archived(path: Path, job_id: int) -> Path:
    suffixes = "".join(path.suffixes)
    base = path.name[: -len(suffixes)] if suffixes else path.name
    return path.with_name(f"{base}.failed_job{job_id}{suffixes}")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def initial_partial(label: str, job_id: int) -> Path:
    path = cell_paths(label)["metrics"]
    return path if path.exists() else archived(path, job_id)


def initial_replay(label: str, job_id: int) -> Path:
    path = cell_paths(label)["state_replay"]
    return path if path.exists() else archived(path, job_id)


def snapshot_source() -> tuple[Path, str]:
    source = ROOT / "src"
    digest = tree_hash(source)
    parent = ROOT / f"var/artifacts/source_snapshots/point_stage_b_05b_12pass_source_{digest}"
    target = parent / "src"
    if not target.is_dir():
        parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".snapshot.", dir=parent))
        shutil.copytree(source, staging / "src")
        os.replace(staging / "src", target)
        staging.rmdir()
    if tree_hash(target) != digest:
        raise RuntimeError("PointMaze r2 source snapshot mismatch")
    return target, digest


def validate() -> tuple[dict[str, Any], dict[str, Any]]:
    for required in (IDENTITY, SUBMISSION, MANIFEST, RUNNER, INITIAL_IDENTITY, PROTOCOL):
        if not required.is_file():
            raise FileNotFoundError(required)
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    identity = json.loads(IDENTITY.read_text())
    initial = json.loads(INITIAL_IDENTITY.read_text())
    if (
        identity.get("schema") != "point-maze-stage-b-05b-12pass-identity-v1"
        or identity.get("execution_hash") != EXECUTION_HASH
        or initial.get("jobs") != INITIAL_JOBS
        or identity.get("infrastructure_repair", {}).get("schema")
        != "point-maze-stage-b-node203-timeout-repair-r1"
    ):
        raise RuntimeError("PointMaze state is not the preserved r1/initial cohort")
    evidence: dict[str, Any] = {}
    for label, job_id in INITIAL_JOBS.items():
        accounting = run([
            "sacct", "-X", "-j", str(job_id), "--format=State,ExitCode", "-n", "-P"
        ])
        error_files = list((ROOT / "var/artifacts/logs").glob(f"*-{job_id}.err"))
        if (
            "FAILED|1:0" not in accounting
            or len(error_files) != 1
            or TIMEOUT_TEXT not in error_files[0].read_text(errors="replace")
        ):
            raise RuntimeError(f"{label} lacks the common initial timeout")
        metrics_path = initial_partial(label, job_id)
        replay_path = initial_replay(label, job_id)
        if not metrics_path.is_file() or not replay_path.is_file():
            raise FileNotFoundError(f"{label} initial partial evidence")
        metrics = read_jsonl(metrics_path)
        training = [row for row in metrics if row.get("schema") == "point-maze-stage-b-training-metric-v1"]
        evaluations = [row for row in metrics if row.get("schema") == "point-maze-stage-b-evaluation-v1"]
        replay = read_jsonl(replay_path)
        updates = PARTIAL_UPDATES[job_id]
        if [row.get("learning_round") for row in training] != list(range(1, updates + 1)):
            raise RuntimeError(f"{label} initial training boundary drift")
        if [row.get("learning_round") for row in evaluations] != list(range(0, updates, 2)):
            raise RuntimeError(f"{label} initial evaluation boundary drift")
        if [row.get("update") for row in replay] != list(range(1, updates + 1)):
            raise RuntimeError(f"{label} initial replay boundary drift")
        if cell_paths(label)["receipt"].exists():
            raise RuntimeError(f"{label} unexpectedly has an initial receipt")
        evidence[label] = {
            "job_id": job_id,
            "updates": updates,
            "evaluation_coordinates": updates // 2,
            "state_replay_rows": updates,
            "metrics_sha256": sha(metrics_path),
            "state_replay_sha256": sha(replay_path),
            "terminal_receipt": False,
            "failure_signature": TIMEOUT_TEXT,
        }
    for label, job_id in PLACEMENT_JOBS.items():
        accounting = run([
            "sacct", "-X", "-j", str(job_id), "--format=State,Elapsed", "-n", "-P"
        ])
        if "CANCELLED" not in accounting:
            raise RuntimeError(f"{label} placement attempt is not canceled")
        paths = cell_paths(label)
        if any(path.exists() for path in paths.values()):
            raise RuntimeError(f"{label} placement attempt emitted an artifact")
    return identity, evidence


def write_manifest(jobs: dict[str, int]) -> None:
    with MANIFEST.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("arm", "seed", "job_id", "receipt", "metrics", "state_replay"),
            delimiter="\t", lineterminator="\n",
        )
        writer.writeheader()
        for seed in SEEDS:
            for arm in ARMS:
                label = f"{arm}/s{seed}"
                paths = cell_paths(label)
                writer.writerow({
                    "arm": arm, "seed": seed, "job_id": jobs[label],
                    **{name: path.relative_to(ROOT).as_posix() for name, path in paths.items()},
                })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("config", "run"))
    args = parser.parse_args()
    identity, failure_evidence = validate()
    environment = dict(os.environ)
    environment["PYTHONPATH"] = f"{ROOT / 'ops'}:{ROOT / 'src'}"
    if args.phase == "config":
        run([str(PYTHON), "-m", "pytest", "-q",
             str(ROOT / "tests/test_point_maze_interactive_worker.py"),
             str(ROOT / "tests/test_point_maze_stage_b_05b_12pass.py")], env=environment)
        print("[point-stage-b-terminal-session-r2] configuration passed; no job submitted")
        return

    source_root, source_hash = snapshot_source()
    execution_root = Path(identity["execution_root"])
    train_batch = execution_root / "train_point_maze_stage_b_05b_12pass.slurm"
    audit_batch = execution_root / "audit_point_maze_stage_b_05b_12pass.slurm"
    new_jobs: dict[str, int] = {}
    audit_job: int | None = None
    try:
        for seed in SEEDS:
            for arm in ARMS:
                label = f"{arm}/s{seed}"
                short = "grpo" if arm == CONTROL else "maxent"
                output = run([
                    "sbatch", "--parsable", "--hold",
                    f"--job-name=point-stage-b-r2-{short}-s{seed}",
                    "--partition=all", "--account=allcs",
                    "--export=ALL,"
                    f"ROOT_DIR={ROOT},OAT_ZERO_SOURCE_ROOT={source_root},"
                    f"OAT_ZERO_EXECUTION_ROOT={execution_root},OAT_ZERO_SOURCE_HASH={source_hash},"
                    f"OAT_ZERO_EXECUTION_HASH={EXECUTION_HASH},OAT_ZERO_ARM={arm},OAT_ZERO_SEED={seed}",
                    str(train_batch),
                ])
                job_id = int(output.split(";", 1)[0])
                new_jobs[label] = job_id
                run(["scontrol", "update", f"JobId={job_id}", "Requeue=0"])
                record = run(["scontrol", "show", "job", "-o", str(job_id)])
                for required in (
                    "JobState=PENDING", "Reason=JobHeldUser", "gres/gpu:a5000:1",
                    "NumCPUs=8", "MinMemoryNode=64G", "TimeLimit=3-00:00:00", "Requeue=0",
                ):
                    if required not in record:
                        raise RuntimeError(f"held r2 {label} lacks {required}")
        dependency = ",".join(
            f"afterany:{new_jobs[f'{arm}/s{seed}']}" for seed in SEEDS for arm in ARMS
        )
        output = run([
            "sbatch", "--parsable", "--hold", f"--dependency={dependency}",
            "--export=ALL,"
            f"ROOT_DIR={ROOT},OAT_ZERO_REPO_ROOT={ROOT},OAT_ZERO_SOURCE_ROOT={source_root},OAT_ZERO_EXECUTION_ROOT={execution_root}",
            str(audit_batch),
        ])
        audit_job = int(output.split(";", 1)[0])
        audit_record = run(["scontrol", "show", "job", "-o", str(audit_job)])
        if "JobState=PENDING" not in audit_record or "Reason=JobHeldUser" not in audit_record:
            raise RuntimeError("PointMaze r2 audit is not held")

        archives = {
            IDENTITY: ROOT / "var/artifacts/point_maze_stage_b_05b_12pass_placement_r1_identity.json",
            SUBMISSION: ROOT / "var/artifacts/point_maze_stage_b_05b_12pass_placement_r1_submission.json",
            MANIFEST: ROOT / "var/artifacts/point_maze_stage_b_05b_12pass_placement_r1_jobs.tsv",
            RUNNER: ROOT / "var/artifacts/point_maze_stage_b_05b_12pass_placement_r1_audit_runner_identity.json",
        }
        for source, target in archives.items():
            if target.exists():
                raise FileExistsError(target)
            shutil.copy2(source, target)
        for label, initial_job in INITIAL_JOBS.items():
            paths = cell_paths(label)
            for name in ("metrics", "state_replay"):
                path = paths[name]
                if path.exists():
                    destination = archived(path, initial_job)
                    if destination.exists():
                        raise FileExistsError(destination)
                    os.replace(path, destination)

        write_manifest(new_jobs)
        identity = dict(identity)
        identity.update({
            "source_root": str(source_root),
            "source_hash": source_hash,
            "jobs": new_jobs,
            "manifest_sha256": sha(MANIFEST),
            "audit_job_id": audit_job,
            "audit_dependency": dependency,
            "resume": False,
            "infrastructure_repair": {
                "schema": "point-maze-stage-b-terminal-session-repair-r2",
                "repair_protocol_sha256": sha(PROTOCOL),
                "retry_launcher_sha256": sha(Path(__file__).resolve()),
                "initial_attempt_identity_sha256": sha(INITIAL_IDENTITY),
                "placement_r1_identity_sha256": sha(archives[IDENTITY]),
                "initial_failures": failure_evidence,
                "placement_attempt_jobs": PLACEMENT_JOBS,
                "placement_attempt_model_samples": 0,
                "replacement_jobs": new_jobs,
                "from_scratch": True,
                "partial_artifacts_eligible": False,
                "terminal_session_reclamation": True,
                "policy_or_environment_semantics_changed": False,
            },
        })
        atomic(IDENTITY, identity)
        atomic(SUBMISSION, {
            "schema": "point-maze-stage-b-05b-12pass-submission-v1",
            "identity_sha256": sha(IDENTITY),
            "manifest_sha256": sha(MANIFEST),
            "jobs": new_jobs,
            "held_job_audit": "pass",
            "released": True,
            "infrastructure_repair": "terminal-session-r2",
        })
        atomic(RUNNER, {
            "schema": "point-maze-stage-b-05b-12pass-audit-runner-v1",
            "audit_job_id": audit_job,
            "dependency": dependency,
            "identity_sha256": sha(IDENTITY),
            "execution_hash": EXECUTION_HASH,
            "source_hash": source_hash,
            "infrastructure_repair": "terminal-session-r2",
        })
        for job_id in new_jobs.values():
            run(["scontrol", "release", str(job_id)])
        run(["scontrol", "release", str(audit_job)])
    except BaseException:
        for job_id in [*new_jobs.values(), *([] if audit_job is None else [audit_job])]:
            run(["scancel", str(job_id)], check=False)
        raise
    print(
        f"[point-stage-b-terminal-session-r2] released jobs={new_jobs} audit_job={audit_job} source_hash={source_hash}"
    )


if __name__ == "__main__":
    main()
