#!/usr/bin/env python3
"""Configure or launch the frozen Ant stable-handoff v18 controller."""

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
PYTHON = ROOT / "var/maze_runtime/venv/bin/python"
TRAINER = ROOT / "ops/train_ant_stable_handoff_controller_v18.py"
V17_TRAINER = ROOT / "ops/train_ant_sequential_waypoint_controller_v17.py"
V16_TRAINER = ROOT / "ops/train_ant_sequential_waypoint_controller_v16.py"
CONTROLLER_BASE = ROOT / "ops/train_ant_waypoint_controller_v7.py"
BATCH = ROOT / "ops/slurm/train_ant_stable_handoff_controller_v18.slurm"
PROTOCOL = (
    ROOT
    / "paper/preregistration/"
    "ant_stable_handoff_controller_v18_20260730.md"
)
INITIAL_MODEL = (
    ROOT / "var/maze_runtime/controllers/ant_sequential_waypoint_v17.zip"
)
V17_RECEIPT = (
    ROOT
    / "var/maze_runtime/controllers/"
    "ant_sequential_waypoint_v17.evaluation.json"
)
IDENTITY = (
    ROOT / "var/artifacts/ant_stable_handoff_controller_v18_identity.json"
)
SUBMISSION = (
    ROOT / "var/artifacts/ant_stable_handoff_controller_v18_submission.json"
)
MODEL = ROOT / "var/maze_runtime/controllers/ant_stable_handoff_v18.zip"
RECEIPT = (
    ROOT / "var/maze_runtime/controllers/ant_stable_handoff_v18.evaluation.json"
)
EXPECTED_INITIAL = (
    "7a964daa7ebc02d52e4717e62ec7d02eed70d5b3155d4910728e24a5e10bbf0d"
)
EXPECTED_V17_RECEIPT = (
    "9b1e04b85acad936b673976d4bc2af11324d15ad139ea70e019aaee1df95f148"
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


def validate() -> None:
    for path in (
        PYTHON,
        TRAINER,
        V17_TRAINER,
        V16_TRAINER,
        CONTROLLER_BASE,
        BATCH,
        PROTOCOL,
        INITIAL_MODEL,
        V17_RECEIPT,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    if sha(INITIAL_MODEL) != EXPECTED_INITIAL:
        raise RuntimeError("Ant v18 initialization hash mismatch")
    if sha(V17_RECEIPT) != EXPECTED_V17_RECEIPT:
        raise RuntimeError("Ant v18 sealed v17 receipt hash mismatch")
    v17 = json.loads(V17_RECEIPT.read_text(encoding="utf-8"))
    summary = v17.get("evaluation", {}).get("summary", {})
    if (
        v17.get("status") != "fail"
        or v17.get("decision") != "ant_waypoint_v17_ineligible"
        or summary.get("episodes") != 96
        or summary.get("success_rate") != 0.21875
        or summary.get("unhealthy_termination_rate") != 0.28125
    ):
        raise RuntimeError("v18 requires the exact immutable v17 antecedent")
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "ops")
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    run(
        [
            str(PYTHON),
            "-m",
            "py_compile",
            str(TRAINER),
            str(V17_TRAINER),
            str(V16_TRAINER),
            str(CONTROLLER_BASE),
            str(Path(__file__).resolve()),
        ],
        env=environment,
    )
    run(["bash", "-n", str(BATCH)])
    run(
        [
            str(PYTHON),
            "-c",
            (
                "import train_ant_stable_handoff_controller_v18 as v;"
                "assert len(v.TRAINING_PATTERNS)==448;"
                "assert len(v.EVALUATION_PATTERNS)==24;"
                "assert not (set(v.TRAINING_PATTERNS)&set(v.EVALUATION_PATTERNS));"
                "assert not (set(v.v17.EVALUATION_PATTERNS)&set(v.EVALUATION_PATTERNS));"
                "assert len(v.TRAIN_MAPS)==4 and len(v.DEVELOPMENT_MAPS)==4;"
                "assert all(0<c<14 and 0<r<14 for p in v.TRAINING_PATTERNS "
                "for r,c in [v._goal_cell((7,7),p)]);"
                "assert all(0<c<18 and 0<r<18 for p in v.EVALUATION_PATTERNS "
                "for r,c in [v._goal_cell((9,9),p)]);"
                "e=v.AntStableHandoffEnv(rank=0,episode_steps=1200);"
                "o,_=e.reset(seed=73018);assert o.shape[-1]>2;e.close()"
            ),
        ],
        env=environment,
    )


def snapshot_execution() -> tuple[Path, str]:
    staging = Path(
        tempfile.mkdtemp(
            prefix=".ant-stable-v18.",
            dir=ROOT / "var/artifacts/source_snapshots",
        )
    )
    for source in (
        TRAINER,
        V17_TRAINER,
        V16_TRAINER,
        CONTROLLER_BASE,
        BATCH,
    ):
        shutil.copy2(source, staging / source.name)
    digest = tree_hash(staging)
    target = ROOT / f"var/artifacts/source_snapshots/ant_stable_v18_ops_{digest}"
    if target.exists():
        shutil.rmtree(staging)
    else:
        os.replace(staging, target)
    if tree_hash(target) != digest:
        raise RuntimeError("Ant stable v18 execution snapshot mismatch")
    return target, digest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("config", "run"))
    args = parser.parse_args()
    validate()
    if args.phase == "config":
        print("[ant-stable-v18] configuration passed; no job launched")
        return
    for path in (IDENTITY, SUBMISSION, MODEL, RECEIPT):
        if path.exists():
            raise FileExistsError(
                f"fresh Ant stable v18 artifact required: {path}"
            )
    execution_root, execution_hash = snapshot_execution()
    output = run(
        [
            "sbatch",
            "--parsable",
            "--hold",
            "--partition=all",
            "--account=allcs",
            "--export=ALL,"
            f"ROOT_DIR={ROOT},OAT_ZERO_EXECUTION_ROOT={execution_root},"
            f"OAT_ZERO_PROTOCOL_IDENTITY={IDENTITY}",
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
            "NumCPUs=12",
            "MinMemoryNode=32G",
            "TimeLimit=04:00:00",
            "Requeue=0",
        ):
            if required not in record:
                raise RuntimeError(f"held Ant stable v18 job lacks {required}")
        atomic(
            IDENTITY,
            {
                "schema_version": (
                    "ant-stable-handoff-controller-v18-identity-v1"
                ),
                "job_id": job_id,
                "execution_root": str(execution_root),
                "execution_hash": execution_hash,
                "protocol_sha256": sha(PROTOCOL),
                "trainer_sha256": sha(TRAINER),
                "trainer_base_sha256": sha(V17_TRAINER),
                "controller_base_sha256": sha(CONTROLLER_BASE),
                "batch_sha256": sha(BATCH),
                "initial_model_sha256": EXPECTED_INITIAL,
                "v17_failure_receipt_sha256": EXPECTED_V17_RECEIPT,
                "v17_failure_success_rate": 0.21875,
                "v17_failure_unhealthy_rate": 0.28125,
                "seed": 73018,
                "timesteps": 6_000_000,
                "workers": 8,
                "learning_rate": 2e-7,
                "training_map_count": 4,
                "training_map_size": 15,
                "training_pattern_count": 448,
                "stable_planar_speed": 1.0,
                "development_map_count": 4,
                "development_map_size": 19,
                "development_pattern_count": 24,
                "development_pattern_length": 8,
                "development_episode_count": 96,
                "v15_map_loaded_for_training": False,
                "v15_trajectory_loaded_for_training": False,
                "language_model_sampled": False,
                "secondary_post_outcome_repair": True,
                "held_scheduler_record": record,
            },
        )
        atomic(
            SUBMISSION,
            {
                "schema_version": (
                    "ant-stable-handoff-controller-v18-submission-v1"
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
    print(f"[ant-stable-v18] released controller job {job_id}")


if __name__ == "__main__":
    main()
