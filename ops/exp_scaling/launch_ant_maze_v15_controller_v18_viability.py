#!/usr/bin/env python3
"""Configure or launch the frozen Ant v15/controller-v18 model gate."""

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
PYTHON = ROOT / "var/seed_paper_eval/paper310/bin/python"
WORKER = ROOT / "var/maze_runtime/venv/bin/python"
MODEL = ROOT / "var/models/ant_maze_interactive_warmstart_v13"
DATA = ROOT / "var/data/ant_maze_modebench_v15_controller_v18"
ADMISSION = (
    ROOT
    / "var/artifacts/"
    "ant_maze_modebench_v15_controller_v18_admission_audit.json"
)
ADMISSION_IDENTITY = (
    ROOT
    / "var/artifacts/"
    "ant_maze_v15_controller_v18_admission_identity.json"
)
PROTOCOL = (
    ROOT
    / "paper/preregistration/"
    "ant_maze_v15_controller_v18_05b_viability_20260730.md"
)
EVALUATOR = ROOT / "ops/evaluate_ant_maze_interactive_viability_v18.py"
BASE_EVALUATOR = ROOT / "ops/evaluate_point_maze_interactive_viability.py"
QUALIFIER = (
    ROOT / "ops/qualify_ant_maze_v15_controller_v18_viability.py"
)
BATCH = (
    ROOT
    / "ops/slurm/evaluate_ant_maze_v15_controller_v18_viability.slurm"
)
OUTPUT = (
    ROOT / "var/artifacts/ant_maze_v15_controller_v18_05b_viability.json"
)
QUALIFICATION = (
    ROOT / "var/artifacts/ant_maze_v15_controller_v18_qualification.json"
)
IDENTITY = (
    ROOT
    / "var/artifacts/"
    "ant_maze_v15_controller_v18_viability_identity.json"
)
SUBMISSION = (
    ROOT
    / "var/artifacts/"
    "ant_maze_v15_controller_v18_viability_submission.json"
)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, allow_nan=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def run(command: Sequence[str], *, env: dict[str, str] | None = None) -> str:
    completed = subprocess.run(
        list(command),
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def environment() -> dict[str, str]:
    value = dict(os.environ)
    value["PYTHONPATH"] = f"{ROOT / 'ops'}:{ROOT / 'src'}"
    library = str(ROOT / "var/seed_paper_eval/paper310/lib")
    value["LD_LIBRARY_PATH"] = library + (
        ":" + value["LD_LIBRARY_PATH"] if value.get("LD_LIBRARY_PATH") else ""
    )
    return value


def validate_static() -> None:
    for path in (
        PYTHON,
        WORKER,
        PROTOCOL,
        EVALUATOR,
        BASE_EVALUATOR,
        QUALIFIER,
        BATCH,
        MODEL / "config.json",
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    env = environment()
    run(
        [
            str(PYTHON),
            "-m",
            "py_compile",
            str(EVALUATOR),
            str(BASE_EVALUATOR),
            str(QUALIFIER),
            str(Path(__file__).resolve()),
            str(ROOT / "src/oat_drgrpo/ant_maze_interactive_worker_v18.py"),
            str(ROOT / "src/oat_drgrpo/ant_maze_interactive_process_v18.py"),
        ],
        env=env,
    )
    run(["bash", "-n", str(BATCH)])
    run(
        [
            str(PYTHON),
            "-m",
            "pytest",
            "-q",
            str(ROOT / "tests/test_ant_maze_interactive_v18.py"),
            str(ROOT / "tests/test_ant_maze_v15_v18_viability.py"),
            str(ROOT / "tests/test_ant_maze_worker_v18.py"),
            str(ROOT / "tests/test_ant_maze_interactive_v13.py"),
        ],
        env=env,
    )


def validate_runtime() -> None:
    for path in (
        DATA / "identity.json",
        DATA / "dev/dataset_dict.json",
        ADMISSION,
        ADMISSION_IDENTITY,
        ROOT / "var/maze_runtime/controllers/ant_stable_handoff_v18.zip",
        ROOT
        / "var/maze_runtime/controllers/"
        "ant_stable_handoff_v18.evaluation.json",
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    admission = json.loads(ADMISSION.read_text(encoding="utf-8"))
    if (
        admission.get("status") != "pass"
        or admission.get("decision")
        != "admitted_to_ant_v15_v18_frozen_model_viability_gate"
    ):
        raise RuntimeError("Ant v15/v18 admission did not authorize viability")


def snapshot_source() -> tuple[Path, str]:
    digest = tree_hash(ROOT / "src")
    parent = (
        ROOT
        / "var/artifacts/source_snapshots/"
        f"ant_v15_v18_viability_source_{digest}"
    )
    target = parent / "src"
    if not target.exists():
        parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".snapshot.", dir=parent))
        shutil.copytree(ROOT / "src", staging / "src")
        os.replace(staging / "src", target)
        staging.rmdir()
    if tree_hash(target) != digest:
        raise RuntimeError("Ant v15/v18 source snapshot mismatch")
    return target, digest


def snapshot_execution() -> tuple[Path, str]:
    staging = Path(
        tempfile.mkdtemp(
            prefix=".ant-v15-v18-viability.",
            dir=ROOT / "var/artifacts/source_snapshots",
        )
    )
    for source in (PROTOCOL, EVALUATOR, BASE_EVALUATOR, QUALIFIER, BATCH):
        shutil.copy2(source, staging / source.name)
    digest = tree_hash(staging)
    target = (
        ROOT
        / "var/artifacts/source_snapshots/"
        f"ant_v15_v18_viability_ops_{digest}"
    )
    if target.exists():
        shutil.rmtree(staging)
    else:
        os.replace(staging, target)
    if tree_hash(target) != digest:
        raise RuntimeError("Ant v15/v18 execution snapshot mismatch")
    return target, digest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("config", "run"))
    args = parser.parse_args()
    validate_static()
    if args.phase == "config":
        print("[ant-v15-v18-viability] static configuration passed")
        return
    validate_runtime()
    for path in (OUTPUT, QUALIFICATION, IDENTITY, SUBMISSION):
        if path.exists():
            raise FileExistsError(f"fresh Ant v18 viability artifact required: {path}")
    source_root, source_hash = snapshot_source()
    execution_root, execution_hash = snapshot_execution()
    job_text = run(
        [
            "sbatch",
            "--parsable",
            "--hold",
            "--partition=all",
            "--account=allcs",
            "--export=ALL,"
            f"ROOT_DIR={ROOT},OAT_ZERO_SOURCE_ROOT={source_root},"
            f"OAT_ZERO_EXECUTION_ROOT={execution_root},"
            f"OAT_ZERO_SOURCE_HASH={source_hash},"
            f"OAT_ZERO_EXECUTION_HASH={execution_hash},"
            f"OAT_ZERO_VIABILITY_IDENTITY={IDENTITY}",
            str(execution_root / BATCH.name),
        ]
    )
    job_id = int(job_text.split(";", 1)[0])
    try:
        atomic(
            IDENTITY,
            {
                "schema_version": (
                    "ant-maze-v15-controller-v18-viability-identity-v1"
                ),
                "job_id": job_id,
                "source_root": str(source_root),
                "source_hash": source_hash,
                "execution_root": str(execution_root),
                "execution_hash": execution_hash,
                "protocol_sha256": sha(PROTOCOL),
                "evaluator_sha256": sha(EVALUATOR),
                "base_evaluator_sha256": sha(BASE_EVALUATOR),
                "qualifier_sha256": sha(QUALIFIER),
                "batch_sha256": sha(BATCH),
                "data_tree_sha256": tree_hash(DATA),
                "admission_sha256": sha(ADMISSION),
                "admission_identity_sha256": sha(ADMISSION_IDENTITY),
                "model_tree_sha256": tree_hash(MODEL),
                "seed": 76701,
                "sample_count_per_prompt": 64,
                "prefix_count": 16,
                "stable_planar_speed": 1.0,
                "development_only": True,
                "evaluation_rows_loaded": False,
                "final_seed": False,
            },
        )
        record = run(["scontrol", "show", "job", str(job_id), "-o"])
        for required in (
            "JobState=PENDING",
            "Reason=JobHeldUser",
            "gres/gpu:a5000:1",
            "MinMemoryNode=64G",
            "TimeLimit=12:00:00",
        ):
            if required not in record:
                raise RuntimeError(f"held Ant v18 viability job lacks {required}")
        atomic(
            SUBMISSION,
            {
                "schema_version": (
                    "ant-maze-v15-controller-v18-viability-submission-v1"
                ),
                "job_id": job_id,
                "identity_sha256": sha(IDENTITY),
                "held_job_record": record,
                "released": True,
            },
        )
        run(["scontrol", "update", f"JobId={job_id}", "Requeue=0"])
        run(["scontrol", "release", str(job_id)])
    except BaseException:
        subprocess.run(["scancel", str(job_id)], cwd=ROOT, check=False)
        raise
    print(f"[ant-v15-v18-viability] released job {job_id}")


if __name__ == "__main__":
    main()
