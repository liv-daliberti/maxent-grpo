#!/usr/bin/env python3
"""Validate and submit the E75F1 paper-scale PointMaze cohort."""

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
PROTOCOL = (
    ROOT / "paper/preregistration/e75f1_point_maze_waypoint_05b_12pass_20260804.md"
)
AMENDMENT = ROOT / "paper/preregistration/e75f1_submission_dependency_validation_amendment_20260804.md"

E75_DATA_IDENTITY = ROOT / "var/data/point_maze_waypoint_pilot_v1/identity.json"
E75R1_DATA_IDENTITY = ROOT / "var/data/point_maze_waypoint_pilot_e75r1/identity.json"
E75R2_DATA_IDENTITY = ROOT / "var/data/point_maze_waypoint_pilot_e75r2/identity.json"
E75R3_DATA_IDENTITY = ROOT / "var/data/point_maze_waypoint_pilot_e75r3/identity.json"
DATA = ROOT / "var/data/point_maze_waypoint_e75f1"
E75R3_DEV_QUALIFICATION = ROOT / "var/artifacts/e75r3_point_maze_waypoint_dev_qualification.json"
E75R3_FINAL_ANALYSIS = ROOT / "var/artifacts/e75r3_point_maze_waypoint_final_analysis.json"
SFT_MODEL = ROOT / "var/models/point_maze_waypoint_warmstart_e75r3"
SFT_RECEIPT = ROOT / "var/artifacts/e75r3_point_maze_waypoint_warmstart.json"
DEV_STEM = ROOT / "var/artifacts/e75f1_point_maze_waypoint_dev"
DEV_QUALIFICATION = ROOT / "var/artifacts/e75f1_point_maze_waypoint_dev_qualification.json"
IDENTITY = ROOT / "var/artifacts/e75f1_point_maze_waypoint_05b_identity.json"
MANIFEST = ROOT / "var/artifacts/e75f1_point_maze_waypoint_05b_jobs.tsv"
SUBMISSION = ROOT / "var/artifacts/e75f1_point_maze_waypoint_05b_submission.json"
AUDIT_OUTPUT = ROOT / "var/artifacts/e75f1_point_maze_waypoint_05b_12pass_audit.json"
SOURCE_SNAPSHOT_PARENT = ROOT / "var/artifacts/source_snapshots"

PREPARE = ROOT / "ops/slurm/e75f1_point_maze_waypoint_prepare.slurm"
DEV = ROOT / "ops/slurm/e75f1_point_maze_waypoint_dev.slurm"
TRAIN = ROOT / "ops/slurm/e75f1_point_maze_waypoint_train.slurm"
AUDIT_SLURM = ROOT / "ops/slurm/e75f1_point_maze_waypoint_audit.slurm"
MAKE_DATA = ROOT / "ops/make_point_maze_waypoint_pilot_v1_data.py"
TRAIN_PILOT = ROOT / "ops/train_point_maze_waypoint_pilot_v1.py"
SHARED_TRAINER = ROOT / "ops/train_point_maze_interactive_paired_smoke_v1.py"
QUALIFIER = ROOT / "ops/qualify_e75f1_point_maze_waypoint_dev.py"
AUDITOR = ROOT / "ops/audit_e75f1_point_maze_waypoint_05b_12pass.py"

SEEDS = (43, 44, 45, 46, 47)
GPU_NODE = "node208"
ARMS = ("grpo", "verified_first_global_replay_canonical")
ARM_SHORT = {
    "grpo": "g",
    "verified_first_global_replay_canonical": "x",
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


def arm_paths(arm: str, seed: int) -> dict[str, Path]:
    stem = ROOT / f"var/artifacts/e75f1_point_maze_waypoint_{arm}_s{seed}"
    return {
        "model": ROOT / f"var/models/e75f1_point_maze_waypoint_{arm}_s{seed}",
        "receipt": Path(str(stem) + ".json"),
        "metrics": Path(str(stem) + ".metrics.jsonl"),
        "replay": Path(str(stem) + ".replay.jsonl"),
    }


def prerequisites() -> None:
    required = (
        PYTHON, WORKER, MODEL / "config.json", PROTOCOL, AMENDMENT,
        E75_DATA_IDENTITY, E75R1_DATA_IDENTITY, E75R2_DATA_IDENTITY,
        E75R3_DATA_IDENTITY, E75R3_DEV_QUALIFICATION, E75R3_FINAL_ANALYSIS,
        SFT_MODEL / "config.json", SFT_RECEIPT, PREPARE, DEV, TRAIN,
        AUDIT_SLURM, MAKE_DATA, TRAIN_PILOT, SHARED_TRAINER, QUALIFIER,
        AUDITOR, ROOT / "ops/repo_env.sh",
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
            str(TRAIN_PILOT),
            str(QUALIFIER),
            str(AUDITOR),
            str(Path(__file__).resolve()),
        ],
        env=environment,
    )
    for script in (PREPARE, DEV, TRAIN, AUDIT_SLURM):
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
            str(ROOT / "tests/test_e75f1_point_maze_waypoint_launch.py"),
        ],
        env=environment,
    )


def snapshot_source() -> tuple[Path, str]:
    source = ROOT / "src"
    digest = tree_sha256(source)
    parent = SOURCE_SNAPSHOT_PARENT / f"e75f1_point_maze_waypoint_src_{digest}"
    target = parent / "src"
    if not target.is_dir():
        parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".source.", dir=parent))
        shutil.copytree(source, staging / "src")
        os.replace(staging / "src", target)
        staging.rmdir()
    if tree_sha256(target) != digest:
        raise RuntimeError("E75F1 source snapshot hash mismatch")
    return target, digest


def snapshot_execution() -> tuple[Path, str]:
    inputs = (
        PREPARE,
        DEV,
        TRAIN,
        AUDIT_SLURM,
        MAKE_DATA,
        TRAIN_PILOT,
        SHARED_TRAINER,
        QUALIFIER,
        PROTOCOL,
        AMENDMENT,
        AUDITOR,
    )
    SOURCE_SNAPSHOT_PARENT.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=".e75f1-point-waypoint.", dir=SOURCE_SNAPSHOT_PARENT)
    )
    for source in inputs:
        shutil.copy2(source, temporary / source.name)
    digest = tree_sha256(temporary)
    target = SOURCE_SNAPSHOT_PARENT / f"e75f1_point_maze_waypoint_ops_{digest}"
    if not target.is_dir():
        os.replace(temporary, target)
    else:
        shutil.rmtree(temporary)
    if tree_sha256(target) != digest:
        raise RuntimeError("E75F1 execution snapshot hash mismatch")
    return target, digest


def fresh_outputs() -> list[Path]:
    paths = [
        DATA, Path(str(DEV_STEM) + ".json"),
        Path(str(DEV_STEM) + ".metrics.jsonl"),
        Path(str(DEV_STEM) + ".replay.jsonl"), DEV_QUALIFICATION,
        IDENTITY, MANIFEST, SUBMISSION, AUDIT_OUTPUT,
    ]
    for arm in ARMS:
        for seed in SEEDS:
            paths.extend(arm_paths(arm, seed).values())
    return paths


def export_spec(
    *, source_root: Path, source_hash: str, execution_root: Path,
    execution_hash: str, arm: str | None = None, seed: int | None = None,
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
    if seed is not None:
        values["OAT_ZERO_SEED"] = str(seed)
    return "ALL," + ",".join(f"{key}={value}" for key, value in values.items())


def submit_held(
    *, name: str, script: Path, source_root: Path, source_hash: str,
    execution_root: Path, execution_hash: str, dependencies: Sequence[int] = (),
    arm: str | None = None, seed: int | None = None, gpu: bool, time_limit: str,
) -> int:
    command = [
        "sbatch", "--parsable", "--hold", f"--job-name={name}",
        "--partition=all", "--account=allcs", "--cpus-per-task=8",
        "--mem=64G", f"--time={time_limit}",
        "--export=" + export_spec(
            source_root=source_root, source_hash=source_hash,
            execution_root=execution_root, execution_hash=execution_hash,
            arm=arm, seed=seed,
        ),
    ]
    if dependencies:
        command.append("--dependency=afterok:" + ":".join(str(value) for value in dependencies))
    if gpu:
        command.extend([f"--nodelist={GPU_NODE}", "--gres=gpu:a6000:1"])
    command.append(str(execution_root / script.name))
    job_id = int(run(command).split(";", 1)[0])
    run(["scontrol", "update", f"JobId={job_id}", "Partition=all", "Requeue=0"])
    return job_id


def validate_job(
    *, job_id: int, name: str, dependencies: Sequence[int], gpu: bool,
    source_hash: str, execution_hash: str,
) -> str:
    record = run(["scontrol", "show", "job", "-o", str(job_id)])
    required = [
        f"JobName={name}", "JobState=PENDING", "Reason=JobHeldUser",
        "Account=allcs", "Partition=all", "NumCPUs=8", "MinMemoryNode=64G",
        "Requeue=0", f"OAT_ZERO_SOURCE_HASH={source_hash}",
        f"OAT_ZERO_EXECUTION_HASH={execution_hash}",
    ]
    if dependencies:
        if len(dependencies) == 1:
            required.append(f"Dependency=afterok:{dependencies[0]}")
        else:
            required.extend(
                f"afterok:{value}" for value in dependencies
            )
    if gpu:
        required.extend(["gres/gpu:a6000:1", GPU_NODE])
    missing = [value for value in required if value not in record]
    if missing:
        raise RuntimeError(f"E75F1 held job {job_id} is missing {missing}")
    return record


def write_manifest(jobs: Mapping[str, Any]) -> None:
    rows = [
        {
            "stage": "prepare", "arm": "", "seed": "",
            "job_id": jobs["prepare"], "dependency": "",
            "primary_output": str(DATA / "identity.json"),
        },
        {
            "stage": "dev", "arm": "", "seed": "",
            "job_id": jobs["dev"], "dependency": jobs["prepare"],
            "primary_output": str(DEV_QUALIFICATION),
        },
    ]
    for arm in ARMS:
        for seed in SEEDS:
            key = f"{arm}:s{seed}"
            rows.append({
                "stage": "online", "arm": arm, "seed": seed,
                "job_id": jobs["online"][key], "dependency": jobs["dev"],
                "primary_output": str(arm_paths(arm, seed)["receipt"]),
            })
    rows.append({
        "stage": "audit", "arm": "", "seed": "", "job_id": jobs["audit"],
        "dependency": ":".join(str(value) for value in jobs["online"].values()),
        "primary_output": str(AUDIT_OUTPUT),
    })
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("stage", "arm", "seed", "job_id", "dependency", "primary_output"),
            delimiter="\t", lineterminator="\n",
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
        print("[e75f1-point-waypoint] configuration passed; no jobs submitted")
        return 0

    for path in fresh_outputs():
        if path.exists():
            raise FileExistsError(f"fresh E75F1 output required: {path}")

    source_root, source_hash = snapshot_source()
    execution_root, execution_hash = snapshot_execution()
    jobs: dict[str, Any] = {"online": {}}
    submitted: list[int] = []
    try:
        jobs["prepare"] = submit_held(
            name="e75f1pmw-prep", script=PREPARE, source_root=source_root,
            source_hash=source_hash, execution_root=execution_root,
            execution_hash=execution_hash, gpu=False, time_limit="12:00:00",
        )
        submitted.append(jobs["prepare"])
        jobs["dev"] = submit_held(
            name="e75f1pmw-dev", script=DEV, source_root=source_root,
            source_hash=source_hash, execution_root=execution_root,
            execution_hash=execution_hash, dependencies=(jobs["prepare"],),
            gpu=True, time_limit="08:00:00",
        )
        submitted.append(jobs["dev"])
        for arm in ARMS:
            for seed in SEEDS:
                key = f"{arm}:s{seed}"
                name = f"e75f1pmw-{ARM_SHORT[arm]}-s{seed}"
                jobs["online"][key] = submit_held(
                    name=name, script=TRAIN, source_root=source_root,
                    source_hash=source_hash, execution_root=execution_root,
                    execution_hash=execution_hash, dependencies=(jobs["dev"],),
                    arm=arm, seed=seed, gpu=True, time_limit="3-00:00:00",
                )
                submitted.append(jobs["online"][key])
        online_ids = tuple(jobs["online"].values())
        jobs["audit"] = submit_held(
            name="e75f1pmw-audit", script=AUDIT_SLURM, source_root=source_root,
            source_hash=source_hash, execution_root=execution_root,
            execution_hash=execution_hash, dependencies=online_ids,
            gpu=False, time_limit="04:00:00",
        )
        submitted.append(jobs["audit"])

        validate_job(
            job_id=jobs["prepare"], name="e75f1pmw-prep", dependencies=(),
            gpu=False, source_hash=source_hash, execution_hash=execution_hash,
        )
        validate_job(
            job_id=jobs["dev"], name="e75f1pmw-dev",
            dependencies=(jobs["prepare"],), gpu=True,
            source_hash=source_hash, execution_hash=execution_hash,
        )
        for arm in ARMS:
            for seed in SEEDS:
                key = f"{arm}:s{seed}"
                validate_job(
                    job_id=jobs["online"][key],
                    name=f"e75f1pmw-{ARM_SHORT[arm]}-s{seed}",
                    dependencies=(jobs["dev"],), gpu=True,
                    source_hash=source_hash, execution_hash=execution_hash,
                )
        validate_job(
            job_id=jobs["audit"], name="e75f1pmw-audit",
            dependencies=online_ids, gpu=False, source_hash=source_hash,
            execution_hash=execution_hash,
        )

        write_manifest(jobs)
        identity = {
            "schema": "e75f1-point-maze-waypoint-05b-identity-v1",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "status": "launched", "experiment": "E75F1",
            "base_model": "Qwen2.5-0.5B-Instruct",
            "base_model_path": str(MODEL),
            "base_model_tree_sha256": tree_sha256(MODEL),
            "warmstart_model_path": str(SFT_MODEL),
            "warmstart_model_tree_sha256": tree_sha256(SFT_MODEL),
            "warmstart_receipt_sha256": sha256_file(SFT_RECEIPT),
            "protocol_sha256": sha256_file(PROTOCOL),
            "operations_amendment_sha256": sha256_file(AMENDMENT),
            "launcher_sha256": sha256_file(Path(__file__).resolve()),
            "source_root": str(source_root), "source_sha256": source_hash,
            "execution_root": str(execution_root),
            "execution_sha256": execution_hash,
            "manifest_sha256": sha256_file(MANIFEST),
            "data_seed": 88104, "expected_split_counts": {
                "train": 384, "dev": 64, "eval": 128,
            },
            "predecessors": [
                {"experiment": "E75", "data_identity_sha256": sha256_file(E75_DATA_IDENTITY)},
                {"experiment": "E75R1", "data_identity_sha256": sha256_file(E75R1_DATA_IDENTITY)},
                {"experiment": "E75R2", "data_identity_sha256": sha256_file(E75R2_DATA_IDENTITY)},
                {"experiment": "E75R3", "data_identity_sha256": sha256_file(E75R3_DATA_IDENTITY)},
            ],
            "e75r3_dev_qualification_sha256": sha256_file(E75R3_DEV_QUALIFICATION),
            "e75r3_final_analysis_sha256": sha256_file(E75R3_FINAL_ANALYSIS),
            "arms": list(ARMS), "seeds": list(SEEDS), "jobs": jobs,
            "optimizer_updates_per_cell": 4608, "passes": 12,
            "evaluation_interval_updates": 96, "evaluation_coordinates": 49,
            "dependency_chain": {
                "prepare_to_dev": "afterok",
                "dev_qualification_to_online": "afterok",
                "all_online_to_audit": "afterok",
            },
            "gpu_node": GPU_NODE, "gpu_type": "a6000",
            "held_job_audit": "pass", "efficacy_gate": False,
        }
        atomic_json(IDENTITY, identity)
        for job_id in submitted:
            run(["scontrol", "release", str(job_id)])
        atomic_json(SUBMISSION, {
            "schema": "e75f1-point-maze-waypoint-05b-submission-v1",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "identity_sha256": sha256_file(IDENTITY),
            "manifest_sha256": sha256_file(MANIFEST), "jobs": jobs,
            "released": True, "conditional_on_dev_gate": True,
        })
    except BaseException:
        if submitted:
            subprocess.run(["scancel", *[str(job) for job in submitted]], check=False)
        raise

    print(json.dumps({"experiment": "E75F1", "jobs": jobs, "released": True}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

