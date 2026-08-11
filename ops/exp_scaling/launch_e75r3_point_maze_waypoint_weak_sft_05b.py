#!/usr/bin/env python3
"""Validate and submit E75R3, the calibrated-warm-start PointMaze successor."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
PYTHON = ROOT / "var/seed_paper_eval/paper310/bin/python"
WORKER = ROOT / "var/maze_runtime/venv/bin/python"
MODEL = (
    ROOT / "var/cache/huggingface/transformers/"
    "models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/"
    "7ae557604adf67be50417f59c2c2f167def9a775"
)
PROTOCOL = ROOT / "paper/preregistration/e75r3_point_maze_waypoint_weak_sft_05b_20260804.md"
E75_DATA_IDENTITY = (
    ROOT / "var/data/point_maze_waypoint_pilot_v1/identity.json"
)
E75R1_DATA_IDENTITY = (
    ROOT / "var/data/point_maze_waypoint_pilot_e75r1/identity.json"
)
E75R2_DATA_IDENTITY = (
    ROOT / "var/data/point_maze_waypoint_pilot_e75r2/identity.json"
)
DATA = ROOT / "var/data/point_maze_waypoint_pilot_e75r3"
SFT_DATA = ROOT / "var/data/point_maze_waypoint_warmstart_e75r3"
E75_DEV_QUALIFICATION = (
    ROOT / "var/artifacts/e75_point_maze_waypoint_dev_qualification.json"
)
E75R1_DEV_QUALIFICATION = (
    ROOT / "var/artifacts/e75r1_point_maze_waypoint_dev_qualification.json"
)
E75R2_DEV_QUALIFICATION = (
    ROOT / "var/artifacts/e75r2_point_maze_waypoint_dev_qualification.json"
)
SFT_MODEL = ROOT / "var/models/point_maze_waypoint_warmstart_e75r3"
SFT_RECEIPT = ROOT / "var/artifacts/e75r3_point_maze_waypoint_warmstart.json"
DEV_STEM = ROOT / "var/artifacts/e75r3_point_maze_waypoint_dev"
DEV_QUALIFICATION = (
    ROOT / "var/artifacts/e75r3_point_maze_waypoint_dev_qualification.json"
)
IDENTITY = ROOT / "var/artifacts/e75r3_point_maze_waypoint_05b_identity.json"
MANIFEST = ROOT / "var/artifacts/e75r3_point_maze_waypoint_05b_jobs.tsv"
SUBMISSION = ROOT / "var/artifacts/e75r3_point_maze_waypoint_05b_submission.json"
SOURCE_SNAPSHOT_PARENT = ROOT / "var/artifacts/source_snapshots"

PREPARE = ROOT / "ops/slurm/e75r3_point_maze_waypoint_prepare.slurm"
SFT = ROOT / "ops/slurm/e75r3_point_maze_waypoint_sft.slurm"
DEV = ROOT / "ops/slurm/e75r3_point_maze_waypoint_dev.slurm"
TRAIN = ROOT / "ops/slurm/e75r3_point_maze_waypoint_train.slurm"
MAKE_DATA = ROOT / "ops/make_point_maze_waypoint_pilot_v1_data.py"
MAKE_SFT_DATA = ROOT / "ops/materialize_point_maze_waypoint_warmstart_v1.py"
TRAIN_SFT = ROOT / "ops/train_point_maze_waypoint_warmstart_v1.py"
TRAIN_PILOT = ROOT / "ops/train_point_maze_waypoint_pilot_v1.py"
SHARED_TRAINER = ROOT / "ops/train_point_maze_interactive_paired_smoke_v1.py"
QUALIFIER = ROOT / "ops/qualify_point_maze_waypoint_dev_v1.py"

SEED = 88504
GPU_NODE = "node208"
ARMS = (
    "grpo",
    "verified_first_global_replay_canonical",
    "verified_first_delayed_singleton_replay_canonical",
)
ARM_SHORT = {
    "grpo": "grpo",
    "verified_first_global_replay_canonical": "current",
    "verified_first_delayed_singleton_replay_canonical": "delayed",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, allow_nan=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def run(command: Sequence[str], *, env: Mapping[str, str] | None = None) -> str:
    completed = subprocess.run(
        list(command),
        cwd=ROOT,
        env=None if env is None else dict(env),
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def arm_paths(arm: str) -> dict[str, Path]:
    stem = ROOT / f"var/artifacts/e75r3_point_maze_waypoint_{arm}_s{SEED}"
    return {
        "model": ROOT / f"var/models/e75r3_point_maze_waypoint_{arm}_s{SEED}",
        "receipt": Path(str(stem) + ".json"),
        "metrics": Path(str(stem) + ".metrics.jsonl"),
        "replay": Path(str(stem) + ".replay.jsonl"),
    }


def prerequisites() -> None:
    required = (
        PYTHON,
        WORKER,
        MODEL / "config.json",
        PROTOCOL,
        E75_DATA_IDENTITY,
        E75R1_DATA_IDENTITY,
        E75R2_DATA_IDENTITY,
        E75_DEV_QUALIFICATION,
        E75R1_DEV_QUALIFICATION,
        E75R2_DEV_QUALIFICATION,
        PREPARE,
        SFT,
        DEV,
        TRAIN,
        MAKE_DATA,
        MAKE_SFT_DATA,
        TRAIN_SFT,
        TRAIN_PILOT,
        SHARED_TRAINER,
        QUALIFIER,
        ROOT / "ops/repo_env.sh",
    )
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)


def validate() -> None:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = f"{ROOT / 'ops'}:{ROOT / 'src'}"
    library = str(ROOT / "var/seed_paper_eval/paper310/lib")
    environment["LD_LIBRARY_PATH"] = library + (
        ":" + environment["LD_LIBRARY_PATH"]
        if environment.get("LD_LIBRARY_PATH")
        else ""
    )
    run(
        [
            str(PYTHON),
            "-m",
            "py_compile",
            str(MAKE_DATA),
            str(MAKE_SFT_DATA),
            str(TRAIN_SFT),
            str(TRAIN_PILOT),
            str(QUALIFIER),
            str(Path(__file__).resolve()),
        ],
        env=environment,
    )
    for script in (PREPARE, SFT, DEV, TRAIN):
        run(["bash", "-n", str(script)])
    run(
        [
            str(PYTHON),
            "-m",
            "pytest",
            "-q",
            str(ROOT / "tests/test_point_maze_waypoint.py"),
            str(ROOT / "tests/test_point_maze_paired_smoke.py"),
            str(ROOT / "tests/test_point_maze_interactive_policy.py"),
            str(ROOT / "tests/test_e75r3_point_maze_waypoint_launch.py"),
        ],
        env=environment,
    )


def snapshot_source() -> tuple[Path, str]:
    source = ROOT / "src"
    digest = tree_sha256(source)
    parent = SOURCE_SNAPSHOT_PARENT / f"e75r3_point_maze_waypoint_src_{digest}"
    target = parent / "src"
    if not target.is_dir():
        parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".source.", dir=parent))
        shutil.copytree(source, staging / "src")
        os.replace(staging / "src", target)
        staging.rmdir()
    if tree_sha256(target) != digest:
        raise RuntimeError("E75R3 source snapshot hash mismatch")
    return target, digest


def snapshot_execution() -> tuple[Path, str]:
    inputs = (
        PREPARE,
        SFT,
        DEV,
        TRAIN,
        MAKE_DATA,
        MAKE_SFT_DATA,
        TRAIN_SFT,
        TRAIN_PILOT,
        SHARED_TRAINER,
        QUALIFIER,
        PROTOCOL,
    )
    SOURCE_SNAPSHOT_PARENT.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=".e75r3-point-waypoint.", dir=SOURCE_SNAPSHOT_PARENT)
    )
    for source in inputs:
        shutil.copy2(source, temporary / source.name)
    digest = tree_sha256(temporary)
    target = SOURCE_SNAPSHOT_PARENT / f"e75r3_point_maze_waypoint_ops_{digest}"
    if not target.is_dir():
        os.replace(temporary, target)
    else:
        shutil.rmtree(temporary)
    if tree_sha256(target) != digest:
        raise RuntimeError("E75R3 execution snapshot hash mismatch")
    return target, digest


def fresh_outputs() -> list[Path]:
    paths = [
        DATA,
        SFT_DATA,
        SFT_MODEL,
        SFT_RECEIPT,
        Path(str(DEV_STEM) + ".json"),
        Path(str(DEV_STEM) + ".metrics.jsonl"),
        Path(str(DEV_STEM) + ".replay.jsonl"),
        DEV_QUALIFICATION,
        IDENTITY,
        MANIFEST,
        SUBMISSION,
    ]
    for arm in ARMS:
        paths.extend(arm_paths(arm).values())
    return paths


def export_spec(
    *,
    source_root: Path,
    source_hash: str,
    execution_root: Path,
    execution_hash: str,
    arm: str | None = None,
) -> str:
    values = {
        "ROOT_DIR": str(ROOT),
        "OAT_ZERO_SOURCE_ROOT": str(source_root),
        "OAT_ZERO_EXECUTION_ROOT": str(execution_root),
        "OAT_ZERO_SOURCE_HASH": source_hash,
        "OAT_ZERO_EXECUTION_HASH": execution_hash,
        "OAT_ZERO_BASE_MODEL": str(MODEL),
    }
    if arm is not None:
        values["OAT_ZERO_ARM"] = arm
    return "ALL," + ",".join(f"{key}={value}" for key, value in values.items())


def submit_held(
    *,
    name: str,
    script: Path,
    source_root: Path,
    source_hash: str,
    execution_root: Path,
    execution_hash: str,
    dependency: int | None = None,
    arm: str | None = None,
    gpu: bool,
    time_limit: str,
) -> int:
    command = [
        "sbatch",
        "--parsable",
        "--hold",
        f"--job-name={name}",
        "--partition=all",
        "--account=allcs",
        "--cpus-per-task=8",
        "--mem=64G",
        f"--time={time_limit}",
        "--export="
        + export_spec(
            source_root=source_root,
            source_hash=source_hash,
            execution_root=execution_root,
            execution_hash=execution_hash,
            arm=arm,
        ),
    ]
    if dependency is not None:
        command.append(f"--dependency=afterok:{dependency}")
    if gpu:
        command.extend([f"--nodelist={GPU_NODE}", "--gres=gpu:a6000:1"])
    command.append(str(execution_root / script.name))
    output = run(command)
    job_id = int(output.split(";", 1)[0])
    # The site's submit router can canonicalize ``all`` to ``cs``. Force the
    # intended partition after submission so a node208 pin remains satisfiable.
    run(
        ["scontrol", "update", f"JobId={job_id}", "Partition=all", "Requeue=0"]
    )
    return job_id


def validate_job(
    *,
    job_id: int,
    name: str,
    dependency: int | None,
    gpu: bool,
    source_hash: str,
    execution_hash: str,
) -> str:
    record = run(["scontrol", "show", "job", "-o", str(job_id)])
    required = [
        f"JobName={name}",
        "JobState=PENDING",
        "Reason=JobHeldUser",
        "Account=allcs",
        "Partition=all",
        "NumCPUs=8",
        "MinMemoryNode=64G",
        "Requeue=0",
        f"OAT_ZERO_SOURCE_HASH={source_hash}",
        f"OAT_ZERO_EXECUTION_HASH={execution_hash}",
    ]
    if dependency is not None:
        required.append(f"Dependency=afterok:{dependency}")
    if gpu:
        required.extend(["gres/gpu:a6000:1", GPU_NODE])
    missing = [value for value in required if value not in record]
    if missing:
        raise RuntimeError(f"E75R3 held job {job_id} is missing {missing}")
    return record


def write_manifest(jobs: Mapping[str, int]) -> None:
    rows = [
        {
            "stage": "prepare",
            "arm": "",
            "job_id": jobs["prepare"],
            "dependency": "",
            "primary_output": str(DATA / "identity.json"),
        },
        {
            "stage": "sft",
            "arm": "",
            "job_id": jobs["sft"],
            "dependency": jobs["prepare"],
            "primary_output": str(SFT_RECEIPT),
        },
        {
            "stage": "dev",
            "arm": "",
            "job_id": jobs["dev"],
            "dependency": jobs["sft"],
            "primary_output": str(DEV_QUALIFICATION),
        },
    ]
    for arm in ARMS:
        rows.append(
            {
                "stage": "online",
                "arm": arm,
                "job_id": jobs[arm],
                "dependency": jobs["dev"],
                "primary_output": str(arm_paths(arm)["receipt"]),
            }
        )
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("stage", "arm", "job_id", "dependency", "primary_output"),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("config", "run"))
    args = parser.parse_args()
    prerequisites()
    validate()
    if args.phase == "config":
        print("[e75r3-point-waypoint] configuration passed; no jobs submitted")
        return 0

    for path in fresh_outputs():
        if path.exists():
            raise FileExistsError(f"fresh E75R3 output required: {path}")

    source_root, source_hash = snapshot_source()
    execution_root, execution_hash = snapshot_execution()
    jobs: dict[str, int] = {}
    submitted: list[int] = []
    records: dict[str, str] = {}
    try:
        jobs["prepare"] = submit_held(
            name="e75r3pmw-prep",
            script=PREPARE,
            source_root=source_root,
            source_hash=source_hash,
            execution_root=execution_root,
            execution_hash=execution_hash,
            gpu=False,
            time_limit="04:00:00",
        )
        submitted.append(jobs["prepare"])
        jobs["sft"] = submit_held(
            name="e75r3pmw-sft",
            script=SFT,
            source_root=source_root,
            source_hash=source_hash,
            execution_root=execution_root,
            execution_hash=execution_hash,
            dependency=jobs["prepare"],
            gpu=True,
            time_limit="12:00:00",
        )
        submitted.append(jobs["sft"])
        jobs["dev"] = submit_held(
            name="e75r3pmw-dev",
            script=DEV,
            source_root=source_root,
            source_hash=source_hash,
            execution_root=execution_root,
            execution_hash=execution_hash,
            dependency=jobs["sft"],
            gpu=True,
            time_limit="08:00:00",
        )
        submitted.append(jobs["dev"])
        for arm in ARMS:
            jobs[arm] = submit_held(
                name=f"e75r3pmw-{ARM_SHORT[arm]}",
                script=TRAIN,
                source_root=source_root,
                source_hash=source_hash,
                execution_root=execution_root,
                execution_hash=execution_hash,
                dependency=jobs["dev"],
                arm=arm,
                gpu=True,
                time_limit="1-00:00:00",
            )
            submitted.append(jobs[arm])

        records["prepare"] = validate_job(
            job_id=jobs["prepare"],
            name="e75r3pmw-prep",
            dependency=None,
            gpu=False,
            source_hash=source_hash,
            execution_hash=execution_hash,
        )
        records["sft"] = validate_job(
            job_id=jobs["sft"],
            name="e75r3pmw-sft",
            dependency=jobs["prepare"],
            gpu=True,
            source_hash=source_hash,
            execution_hash=execution_hash,
        )
        records["dev"] = validate_job(
            job_id=jobs["dev"],
            name="e75r3pmw-dev",
            dependency=jobs["sft"],
            gpu=True,
            source_hash=source_hash,
            execution_hash=execution_hash,
        )
        for arm in ARMS:
            records[arm] = validate_job(
                job_id=jobs[arm],
                name=f"e75r3pmw-{ARM_SHORT[arm]}",
                dependency=jobs["dev"],
                gpu=True,
                source_hash=source_hash,
                execution_hash=execution_hash,
            )

        write_manifest(jobs)
        identity = {
            "schema": "e75r3-point-maze-waypoint-05b-identity-v1",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "status": "launched",
            "experiment": "E75R3",
            "model": "Qwen2.5-0.5B-Instruct",
            "model_path": str(MODEL),
            "model_tree_sha256": tree_sha256(MODEL),
            "protocol_sha256": sha256_file(PROTOCOL),
            "launcher_sha256": sha256_file(Path(__file__).resolve()),
            "source_root": str(source_root),
            "source_sha256": source_hash,
            "execution_root": str(execution_root),
            "execution_sha256": execution_hash,
            "manifest_sha256": sha256_file(MANIFEST),
            "seed": SEED,
            "sft_seed": 88404,
            "data_seed": 88103,
            "sft_optimizer_updates": 72,
            "predecessors": [
                {
                    "experiment": "E75",
                    "data_identity_sha256": sha256_file(E75_DATA_IDENTITY),
                    "dev_qualification_sha256": sha256_file(E75_DEV_QUALIFICATION),
                },
                {
                    "experiment": "E75R1",
                    "data_identity_sha256": sha256_file(E75R1_DATA_IDENTITY),
                    "dev_qualification_sha256": sha256_file(E75R1_DEV_QUALIFICATION),
                },
                {
                    "experiment": "E75R2",
                    "data_identity_sha256": sha256_file(E75R2_DATA_IDENTITY),
                    "dev_qualification_sha256": sha256_file(E75R2_DEV_QUALIFICATION),
                },
            ],
            "arms": list(ARMS),
            "jobs": jobs,
            "dependency_chain": {
                "prepare_to_sft": "afterok",
                "sft_to_dev": "afterok",
                "dev_qualification_to_online": "afterok",
            },
            "gpu_node": GPU_NODE,
            "gpu_type": "a6000",
            "train_maps": 64,
            "dev_maps": 32,
            "eval_maps_held_out": 64,
            "online_updates_per_arm": 64,
            "evaluation_job_submitted": False,
            "development_only": True,
            "held_job_audit": "pass",
        }
        atomic_json(IDENTITY, identity)
        for job_id in submitted:
            run(["scontrol", "release", str(job_id)])
        atomic_json(
            SUBMISSION,
            {
                "schema": "e75r3-point-maze-waypoint-05b-submission-v1",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "identity_sha256": sha256_file(IDENTITY),
                "manifest_sha256": sha256_file(MANIFEST),
                "jobs": jobs,
                "released": True,
                "evaluation_job_submitted": False,
            },
        )
    except BaseException:
        if submitted:
            subprocess.run(["scancel", *[str(job) for job in submitted]], check=False)
        raise

    print(
        json.dumps(
            {"experiment": "E75R3", "jobs": jobs, "released": True}, sort_keys=True
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
