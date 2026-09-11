#!/usr/bin/env python3
"""Configure or launch PointMaze's unseen-geometry repair viability gate."""

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
PROTOCOL = ROOT / "paper/preregistration/point_maze_algorithm_repair_v1_20260730.md"
EVALUATOR = ROOT / "ops/evaluate_point_maze_interactive_viability.py"
BATCH = ROOT / "ops/slurm/evaluate_point_maze_algorithm_repair_v1.slurm"
DATA = ROOT / "var/data/point_maze_algorithm_repair_v1"
ADMISSION = ROOT / "var/artifacts/point_maze_algorithm_repair_v1_admission_audit.json"
MODEL = ROOT / "var/models/point_maze_interactive_warmstart_v3"
OUTPUT = ROOT / "var/artifacts/point_maze_algorithm_repair_v1_viability.json"
IDENTITY = ROOT / "var/artifacts/point_maze_algorithm_repair_v1_viability_identity.json"
SUBMISSION = (
    ROOT / "var/artifacts/point_maze_algorithm_repair_v1_viability_submission.json"
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
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, allow_nan=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def run(command: Sequence[str], *, env: dict[str, str] | None = None) -> str:
    result = subprocess.run(
        list(command), cwd=ROOT, env=env, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def snapshot_tree(source: Path, prefix: str) -> tuple[Path, str]:
    digest = tree_hash(source)
    parent = ROOT / f"var/artifacts/source_snapshots/{prefix}_{digest}"
    target = parent / source.name
    if not target.exists():
        parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".snapshot.", dir=parent))
        shutil.copytree(source, staging / source.name)
        os.replace(staging / source.name, target)
        staging.rmdir()
    if tree_hash(target) != digest:
        raise RuntimeError(f"{prefix} source snapshot mismatch")
    return target, digest


def snapshot_execution() -> tuple[Path, str]:
    staging = Path(
        tempfile.mkdtemp(
            prefix=".point-repair-v1.", dir=ROOT / "var/artifacts/source_snapshots"
        )
    )
    for source in (PROTOCOL, EVALUATOR, BATCH):
        shutil.copy2(source, staging / source.name)
    digest = tree_hash(staging)
    target = ROOT / f"var/artifacts/source_snapshots/point_repair_v1_ops_{digest}"
    if target.exists():
        shutil.rmtree(staging)
    else:
        os.replace(staging, target)
    if tree_hash(target) != digest:
        raise RuntimeError("PointMaze repair execution snapshot mismatch")
    return target, digest


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
            str(EVALUATOR),
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
    parsed = parser.parse_args()
    for path in (
        PYTHON,
        PROTOCOL,
        EVALUATOR,
        BATCH,
        DATA / "identity.json",
        ADMISSION,
        MODEL / "config.json",
        ROOT / "var/maze_runtime/venv/bin/python",
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    admission = json.loads(ADMISSION.read_text())
    if (
        admission.get("status") != "pass"
        or admission.get("decision") != "admitted_for_0.5b_viability_sampling"
    ):
        raise RuntimeError("PointMaze repair route admission did not pass")
    validate()
    if parsed.phase == "config":
        print("[point-repair-v1] viability configuration passed; no model sampled")
        return
    for path in (OUTPUT, IDENTITY, SUBMISSION):
        if path.exists():
            raise FileExistsError(f"fresh PointMaze repair artifact required: {path}")
    source_root, source_hash = snapshot_tree(ROOT / "src", "point_repair_v1_source")
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
            f"OAT_ZERO_EXECUTION_HASH={execution_hash}",
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
        ):
            if required not in record:
                raise RuntimeError(f"held PointMaze repair viability lacks {required}")
        atomic(
            IDENTITY,
            {
                "schema": "point-maze-algorithm-repair-v1-viability-identity",
                "job_id": job_id,
                "source_root": str(source_root),
                "source_hash": source_hash,
                "execution_root": str(execution_root),
                "execution_hash": execution_hash,
                "protocol_sha256": sha(PROTOCOL),
                "data_tree_sha256": tree_hash(DATA),
                "admission_audit_sha256": sha(ADMISSION),
                "model_tree_sha256": tree_hash(MODEL),
                "seed": 76500,
                "sample_count_per_prompt": 64,
                "split": "development_only",
                "held_scheduler_record": record,
                "final_seed": False,
            },
        )
        atomic(
            SUBMISSION,
            {
                "schema": "point-maze-algorithm-repair-v1-viability-submission",
                "job_id": job_id,
                "identity_sha256": sha(IDENTITY),
                "released": True,
            },
        )
        run(["scontrol", "release", str(job_id)])
    except BaseException:
        subprocess.run(["scancel", str(job_id)], cwd=ROOT, check=False)
        raise
    print(f"[point-repair-v1] released viability job {job_id}")


if __name__ == "__main__":
    main()

