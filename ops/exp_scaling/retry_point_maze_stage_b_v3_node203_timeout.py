#!/usr/bin/env python3
"""Fail-closed, from-scratch retry for four PointMaze node203 timeouts."""

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
AUDIT_RUNNER = ROOT / "var/artifacts/point_maze_stage_b_05b_12pass_audit_runner_identity.json"
AUDIT_OUTPUT = ROOT / "var/artifacts/point_maze_stage_b_05b_12pass_audit.json"
REPAIR_PROTOCOL = ROOT / "paper/preregistration/point_maze_stage_b_v3_node203_timeout_repair_r1_20260730.md"
CONTROL = "grpo"
TREATMENT = "verified_first_global_replay_canonical"
ARMS = (CONTROL, TREATMENT)
SEEDS = (43, 44, 45, 46, 47)
FAILED = {
    f"{CONTROL}/s46": 30203027,
    f"{TREATMENT}/s46": 30203028,
    f"{CONTROL}/s47": 30203029,
    f"{TREATMENT}/s47": 30203030,
}
EXPECTED_SOURCE_HASH = "ef84cca31972dd556d79ef2127496328e5b170fde4e0c1f58138c5b8092c11e6"
EXPECTED_EXECUTION_HASH = "a54ebba3a4d8e905d2d109ea68072b31c100414521ca5183d5d188bae26bf0d0"
TIMEOUT_TEXT = "interactive PointMaze worker request timed out"


def run(command: Sequence[str], *, check: bool = True) -> str:
    result = subprocess.run(
        list(command), cwd=ROOT, check=check, capture_output=True, text=True
    )
    return result.stdout.strip()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def archived(path: Path, job_id: int) -> Path:
    suffixes = "".join(path.suffixes)
    base = path.name[: -len(suffixes)] if suffixes else path.name
    return path.with_name(f"{base}.failed_job{job_id}{suffixes}")


def validate() -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    for required in (IDENTITY, SUBMISSION, MANIFEST, AUDIT_RUNNER, REPAIR_PROTOCOL):
        if not required.is_file():
            raise FileNotFoundError(required)
    if AUDIT_OUTPUT.exists():
        raise FileExistsError("PointMaze terminal audit already exists")
    identity = json.loads(IDENTITY.read_text())
    if (
        identity.get("schema") != "point-maze-stage-b-05b-12pass-identity-v1"
        or identity.get("source_hash") != EXPECTED_SOURCE_HASH
        or identity.get("execution_hash") != EXPECTED_EXECUTION_HASH
        or identity.get("resume") is not False
        or identity.get("jobs", {}).get(f"{CONTROL}/s46") != FAILED[f"{CONTROL}/s46"]
        or identity.get("jobs", {}).get(f"{TREATMENT}/s46") != FAILED[f"{TREATMENT}/s46"]
        or identity.get("jobs", {}).get(f"{CONTROL}/s47") != FAILED[f"{CONTROL}/s47"]
        or identity.get("jobs", {}).get(f"{TREATMENT}/s47") != FAILED[f"{TREATMENT}/s47"]
    ):
        raise RuntimeError("initial PointMaze identity is not the frozen failed cohort")
    evidence: dict[str, dict[str, Any]] = {}
    for label, job_id in FAILED.items():
        accounting = run([
            "sacct", "-X", "-j", str(job_id), "--format=State,ExitCode", "-n", "-P"
        ])
        if "FAILED|1:0" not in accounting:
            raise RuntimeError(f"{label} is not an exact failed scheduler attempt")
        errors = list((ROOT / "var/artifacts/logs").glob(f"*-{job_id}.err"))
        if len(errors) != 1 or TIMEOUT_TEXT not in errors[0].read_text(errors="replace"):
            raise RuntimeError(f"{label} lacks the frozen worker-timeout signature")
        paths = cell_paths(label)
        if paths["receipt"].exists():
            raise RuntimeError(f"{label} unexpectedly has a terminal receipt")
        if not paths["metrics"].is_file() or not paths["state_replay"].is_file():
            raise RuntimeError(f"{label} lacks exact partial artifacts")
        metrics = read_jsonl(paths["metrics"])
        training = [row for row in metrics if row.get("schema") == "point-maze-stage-b-training-metric-v1"]
        evaluations = [row for row in metrics if row.get("schema") == "point-maze-stage-b-evaluation-v1"]
        replay = read_jsonl(paths["state_replay"])
        if [row.get("learning_round") for row in training] != list(range(1, 61)):
            raise RuntimeError(f"{label} partial training boundary is not 60 updates")
        if [row.get("learning_round") for row in evaluations] != list(range(0, 60, 2)):
            raise RuntimeError(f"{label} partial evaluation boundary is not rounds 0..58")
        if [row.get("update") for row in replay] != list(range(1, 61)):
            raise RuntimeError(f"{label} partial replay boundary is not 60 updates")
        for path in (paths["metrics"], paths["state_replay"]):
            if archived(path, job_id).exists():
                raise FileExistsError(archived(path, job_id))
        evidence[label] = {
            "failed_job_id": job_id,
            "scheduler_state": "FAILED",
            "exit_code": "1:0",
            "node": "node203",
            "training_rows": 60,
            "evaluation_rows": 30,
            "state_replay_rows": 60,
            "terminal_receipt": False,
            "metrics_sha256": sha(paths["metrics"]),
            "state_replay_sha256": sha(paths["state_replay"]),
            "failure_signature": TIMEOUT_TEXT,
        }
    return identity, evidence


def write_manifest(jobs: dict[str, int]) -> None:
    with MANIFEST.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("arm", "seed", "job_id", "receipt", "metrics", "state_replay"),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for seed in SEEDS:
            for arm in ARMS:
                label = f"{arm}/s{seed}"
                paths = cell_paths(label)
                writer.writerow({
                    "arm": arm,
                    "seed": seed,
                    "job_id": jobs[label],
                    **{name: path.relative_to(ROOT).as_posix() for name, path in paths.items()},
                })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("config", "run"))
    args = parser.parse_args()
    identity, failure_evidence = validate()
    if args.phase == "config":
        print("[point-stage-b-timeout-r1] configuration passed; no job submitted")
        return

    source_root = Path(identity["source_root"])
    execution_root = Path(identity["execution_root"])
    train_batch = execution_root / "train_point_maze_stage_b_05b_12pass.slurm"
    audit_batch = execution_root / "audit_point_maze_stage_b_05b_12pass.slurm"
    if not train_batch.is_file() or not audit_batch.is_file():
        raise FileNotFoundError("immutable PointMaze execution snapshot is incomplete")

    new_jobs: dict[str, int] = {}
    audit_job: int | None = None
    try:
        for label in FAILED:
            arm, seed_label = label.split("/s", 1)
            short = "grpo" if arm == CONTROL else "maxent"
            output = run([
                "sbatch", "--parsable", "--hold", "--nodelist=node202",
                f"--job-name=point-stage-b-r1-{short}-s{seed_label}",
                "--partition=all", "--account=allcs",
                "--export=ALL,"
                f"ROOT_DIR={ROOT},OAT_ZERO_SOURCE_ROOT={source_root},"
                f"OAT_ZERO_EXECUTION_ROOT={execution_root},OAT_ZERO_SOURCE_HASH={identity['source_hash']},"
                f"OAT_ZERO_EXECUTION_HASH={identity['execution_hash']},OAT_ZERO_ARM={arm},OAT_ZERO_SEED={seed_label}",
                str(train_batch),
            ])
            job_id = int(output.split(";", 1)[0])
            new_jobs[label] = job_id
            run(["scontrol", "update", f"JobId={job_id}", "Requeue=0"])
            record = run(["scontrol", "show", "job", "-o", str(job_id)])
            for required in (
                "JobState=PENDING", "Reason=JobHeldUser", "ReqNodeList=node202",
                "gres/gpu:a5000:1", "NumCPUs=8", "MinMemoryNode=64G",
                "TimeLimit=3-00:00:00", "Requeue=0",
            ):
                if required not in record:
                    raise RuntimeError(f"held retry {label} lacks {required}")

        jobs = {str(key): int(value) for key, value in identity["jobs"].items()}
        jobs.update(new_jobs)
        dependency = ",".join(
            f"afterany:{jobs[f'{arm}/s{seed}']}" for seed in SEEDS for arm in ARMS
        )
        audit_output = run([
            "sbatch", "--parsable", f"--dependency={dependency}",
            "--export=ALL,"
            f"ROOT_DIR={ROOT},OAT_ZERO_SOURCE_ROOT={source_root},OAT_ZERO_EXECUTION_ROOT={execution_root}",
            str(audit_batch),
        ])
        audit_job = int(audit_output.split(";", 1)[0])

        originals = {
            IDENTITY: ROOT / "var/artifacts/point_maze_stage_b_05b_12pass_initial_attempt_identity.json",
            SUBMISSION: ROOT / "var/artifacts/point_maze_stage_b_05b_12pass_initial_attempt_submission.json",
            MANIFEST: ROOT / "var/artifacts/point_maze_stage_b_05b_12pass_initial_attempt_jobs.tsv",
            AUDIT_RUNNER: ROOT / "var/artifacts/point_maze_stage_b_05b_12pass_initial_attempt_audit_runner_identity.json",
        }
        for source, target in originals.items():
            if target.exists():
                raise FileExistsError(target)
            shutil.copy2(source, target)
        run(["scancel", str(identity["audit_job_id"])], check=False)
        for label, failed_job in FAILED.items():
            paths = cell_paths(label)
            os.replace(paths["metrics"], archived(paths["metrics"], failed_job))
            os.replace(paths["state_replay"], archived(paths["state_replay"], failed_job))

        write_manifest(jobs)
        identity = dict(identity)
        identity.update({
            "jobs": jobs,
            "manifest_sha256": sha(MANIFEST),
            "audit_job_id": audit_job,
            "audit_dependency": dependency,
            "resume": False,
            "infrastructure_repair": {
                "schema": "point-maze-stage-b-node203-timeout-repair-r1",
                "repair_protocol_sha256": sha(REPAIR_PROTOCOL),
                "retry_launcher_sha256": sha(Path(__file__).resolve()),
                "initial_identity_sha256": sha(originals[IDENTITY]),
                "initial_submission_sha256": sha(originals[SUBMISSION]),
                "initial_manifest_sha256": sha(originals[MANIFEST]),
                "failed_attempts": failure_evidence,
                "replacement_jobs": new_jobs,
                "placement": "node202",
                "from_scratch": True,
                "partial_artifacts_eligible": False,
                "model_outcomes_used_for_repair": False,
            },
        })
        atomic_json(IDENTITY, identity)
        atomic_json(SUBMISSION, {
            "schema": "point-maze-stage-b-05b-12pass-submission-v1",
            "identity_sha256": sha(IDENTITY),
            "manifest_sha256": sha(MANIFEST),
            "jobs": jobs,
            "held_job_audit": "pass",
            "released": True,
            "infrastructure_repair": "node203-timeout-r1",
        })
        atomic_json(AUDIT_RUNNER, {
            "schema": "point-maze-stage-b-05b-12pass-audit-runner-v1",
            "audit_job_id": audit_job,
            "dependency": dependency,
            "identity_sha256": sha(IDENTITY),
            "execution_hash": identity["execution_hash"],
            "infrastructure_repair": "node203-timeout-r1",
        })
        for job_id in new_jobs.values():
            run(["scontrol", "release", str(job_id)])
    except BaseException:
        for job_id in [*new_jobs.values(), *([] if audit_job is None else [audit_job])]:
            run(["scancel", str(job_id)], check=False)
        raise
    print(
        f"[point-stage-b-timeout-r1] released replacements={new_jobs} audit_job={audit_job}"
    )


if __name__ == "__main__":
    main()
