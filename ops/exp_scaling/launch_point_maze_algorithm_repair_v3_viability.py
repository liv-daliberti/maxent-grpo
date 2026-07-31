#!/usr/bin/env python3
"""Configure or launch the frozen PointMaze v3 K=16 viability gate."""

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
PROTOCOL = (
    ROOT
    / "paper/preregistration/"
    "point_maze_algorithm_repair_v3_viability_20260730.md"
)
EVALUATOR = ROOT / "ops/evaluate_point_maze_interactive_viability.py"
QUALIFIER = ROOT / "ops/qualify_point_maze_algorithm_repair_v3.py"
BATCH = ROOT / "ops/slurm/evaluate_point_maze_algorithm_repair_v3.slurm"
DATA = ROOT / "var/data/point_maze_algorithm_repair_v3"
ADMISSION = (
    ROOT
    / "var/artifacts/"
    "point_maze_algorithm_repair_v3_admission_audit.json"
)
MODEL = ROOT / "var/models/point_maze_interactive_warmstart_v3"
OUTPUT = (
    ROOT / "var/artifacts/point_maze_algorithm_repair_v3_viability.json"
)
QUALIFICATION = (
    ROOT / "var/artifacts/point_maze_algorithm_repair_v3_qualification.json"
)
IDENTITY = (
    ROOT
    / "var/artifacts/"
    "point_maze_algorithm_repair_v3_viability_identity.json"
)
SUBMISSION = (
    ROOT
    / "var/artifacts/"
    "point_maze_algorithm_repair_v3_viability_submission.json"
)


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
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
    )
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, allow_nan=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def run(command: Sequence[str], *, env: dict[str, str] | None = None) -> str:
    result = subprocess.run(
        list(command),
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def snapshot_tree(source: Path) -> tuple[Path, str]:
    digest = tree_hash(source)
    parent = (
        ROOT
        / "var/artifacts/source_snapshots/"
        f"point_repair_v3_viability_source_{digest}"
    )
    target = parent / "src"
    if not target.exists():
        parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".snapshot.", dir=parent))
        shutil.copytree(source, staging / "src")
        os.replace(staging / "src", target)
        staging.rmdir()
    if tree_hash(target) != digest:
        raise RuntimeError("PointMaze v3 viability source snapshot mismatch")
    return target, digest


def snapshot_execution() -> tuple[Path, str]:
    staging = Path(
        tempfile.mkdtemp(
            prefix=".point-repair-v3-viability.",
            dir=ROOT / "var/artifacts/source_snapshots",
        )
    )
    for source in (PROTOCOL, EVALUATOR, QUALIFIER, BATCH):
        shutil.copy2(source, staging / source.name)
    digest = tree_hash(staging)
    target = (
        ROOT
        / "var/artifacts/source_snapshots/"
        f"point_repair_v3_viability_ops_{digest}"
    )
    if target.exists():
        shutil.rmtree(staging)
    else:
        os.replace(staging, target)
    if tree_hash(target) != digest:
        raise RuntimeError("PointMaze v3 viability execution snapshot mismatch")
    return target, digest


def validate() -> None:
    for path in (
        PYTHON,
        PROTOCOL,
        EVALUATOR,
        QUALIFIER,
        BATCH,
        DATA / "identity.json",
        ADMISSION,
        MODEL / "config.json",
        ROOT / "var/maze_runtime/venv/bin/python",
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    admission = json.loads(ADMISSION.read_text(encoding="utf-8"))
    if (
        admission.get("status") != "pass"
        or admission.get("decision")
        != "admitted_to_point_maze_v3_balanced_viability_gate"
    ):
        raise RuntimeError("PointMaze v3 executable admission did not pass")
    data = json.loads(
        (DATA / "identity.json").read_text(encoding="utf-8")
    )
    if (
        data.get("schema_version")
        != "point-maze-algorithm-repair-data-v3"
        or data.get("orientation_balanced_within_each_split") is not True
        or data.get("executable_task_overlap_count") != 0
    ):
        raise RuntimeError("PointMaze v3 data identity drift")
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
            str(EVALUATOR),
            str(QUALIFIER),
            str(Path(__file__).resolve()),
        ],
        env=environment,
    )
    run(["bash", "-n", str(BATCH)])
    run(
        [
            str(PYTHON),
            "-m",
            "pytest",
            "-q",
            str(ROOT / "tests/test_point_maze_modebench.py"),
            str(ROOT / "tests/test_point_maze_interactive_policy.py"),
        ],
        env=environment,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("config", "run"))
    args = parser.parse_args()
    validate()
    if args.phase == "config":
        print("[point-repair-v3] viability configuration passed")
        return
    for path in (OUTPUT, QUALIFICATION, IDENTITY, SUBMISSION):
        if path.exists():
            raise FileExistsError(
                f"fresh PointMaze v3 viability artifact required: {path}"
            )
    source_root, source_hash = snapshot_tree(ROOT / "src")
    execution_root, execution_hash = snapshot_execution()
    output = run(
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
    job_id = int(output.split(";", 1)[0])
    try:
        run(["scontrol", "update", f"JobId={job_id}", "Requeue=0"])
        record = run(["scontrol", "show", "job", "-o", str(job_id)])
        for required in (
            "JobState=PENDING",
            "Reason=JobHeldUser",
            "Account=allcs",
            "gres/gpu:a5000:1",
            "NumCPUs=8",
            "MinMemoryNode=64G",
            "TimeLimit=12:00:00",
            "Requeue=0",
            f"OAT_ZERO_SOURCE_HASH={source_hash}",
            f"OAT_ZERO_EXECUTION_HASH={execution_hash}",
            f"OAT_ZERO_VIABILITY_IDENTITY={IDENTITY}",
        ):
            if required not in record:
                raise RuntimeError(
                    f"held PointMaze v3 viability lacks {required}"
                )
        atomic(
            IDENTITY,
            {
                "schema_version": (
                    "point-maze-algorithm-repair-v3-viability-identity-v1"
                ),
                "job_id": job_id,
                "source_root": str(source_root),
                "source_hash": source_hash,
                "execution_root": str(execution_root),
                "execution_hash": execution_hash,
                "protocol_sha256": sha(PROTOCOL),
                "evaluator_sha256": sha(EVALUATOR),
                "qualifier_sha256": sha(QUALIFIER),
                "batch_sha256": sha(BATCH),
                "data_tree_sha256": tree_hash(DATA),
                "data_identity_sha256": sha(DATA / "identity.json"),
                "admission_audit_sha256": sha(ADMISSION),
                "model_tree_sha256": tree_hash(MODEL),
                "seed": 76530,
                "sample_count_per_prompt": 64,
                "prefix_count": 16,
                "split": "development_only",
                "orientation_counts": {
                    "0": 1,
                    "1": 1,
                    "2": 1,
                    "3": 1,
                },
                "held_scheduler_record": record,
                "final_seed": False,
                "evaluation_rows_loaded": False,
                "v2_terminal_pair_outcome_used": False,
            },
        )
        atomic(
            SUBMISSION,
            {
                "schema_version": (
                    "point-maze-algorithm-repair-v3-viability-submission-v1"
                ),
                "job_id": job_id,
                "identity_sha256": sha(IDENTITY),
                "released": True,
            },
        )
        run(["scontrol", "release", str(job_id)])
    except BaseException:
        subprocess.run(["scancel", str(job_id)], cwd=ROOT, check=False)
        raise
    print(f"[point-repair-v3] released viability job {job_id}")


if __name__ == "__main__":
    main()
