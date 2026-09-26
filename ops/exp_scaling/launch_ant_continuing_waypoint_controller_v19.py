#!/usr/bin/env python3
"""Configure or launch the frozen Ant continuing-task v19 controller."""

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
TRAINER = ROOT / "ops/train_ant_continuing_waypoint_controller_v19.py"
V18_TRAINER = ROOT / "ops/train_ant_stable_handoff_controller_v18.py"
V17_TRAINER = ROOT / "ops/train_ant_sequential_waypoint_controller_v17.py"
V16_TRAINER = ROOT / "ops/train_ant_sequential_waypoint_controller_v16.py"
CONTROLLER_BASE = ROOT / "ops/train_ant_waypoint_controller_v7.py"
BATCH = (
    ROOT / "ops/slurm/train_ant_continuing_waypoint_controller_v19.slurm"
)
PROTOCOL = (
    ROOT
    / "paper/preregistration/"
    "ant_continuing_waypoint_controller_v19_20260804.md"
)
INITIAL_MODEL = ROOT / "var/maze_runtime/controllers/ant_waypoint_v11.zip"
V11_RECEIPT = (
    ROOT / "var/maze_runtime/controllers/ant_waypoint_v11.evaluation.json"
)
V18_RECEIPT = (
    ROOT
    / "var/maze_runtime/controllers/"
    "ant_stable_handoff_v18.evaluation.json"
)
SIMULATOR_SOURCE = (
    ROOT
    / "var/maze_runtime/venv/lib/python3.11/site-packages/"
    "gymnasium_robotics/envs/maze/ant_maze_v5.py"
)
TERMINATION_SOURCE = (
    ROOT
    / "var/maze_runtime/venv/lib/python3.11/site-packages/"
    "gymnasium_robotics/envs/maze/maze_v4.py"
)
IDENTITY = (
    ROOT
    / "var/artifacts/"
    "ant_continuing_waypoint_controller_v19_identity.json"
)
SUBMISSION = (
    ROOT
    / "var/artifacts/"
    "ant_continuing_waypoint_controller_v19_submission.json"
)
MODEL = (
    ROOT / "var/maze_runtime/controllers/ant_continuing_waypoint_v19.zip"
)
RECEIPT = (
    ROOT
    / "var/maze_runtime/controllers/"
    "ant_continuing_waypoint_v19.evaluation.json"
)
EXPECTED_INITIAL = (
    "e6d202bd525be5469135b35b63b2bf0884459cc630b71e76f8c49e86dcb8f913"
)
EXPECTED_V11_RECEIPT = (
    "a2e5fb9b1ec757675eeb9c5f998181adbf959f4e6a5d0eeb516baed6f7b3ed93"
)
EXPECTED_V18_RECEIPT = (
    "55398e12736325c8e02791659a439be9a36354112893cd26ba315ab27a6c8229"
)
EXPECTED_SIMULATOR_SOURCE = (
    "203df28af76033022fc2f266723269a8cbc10f7bb4d7b22473adaab4a6094938"
)
EXPECTED_TERMINATION_SOURCE = (
    "1912148fc047355e4e33a3f165064e87ed9bc7a98f7d40e09fcf0eaa45089d56"
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
        V18_TRAINER,
        V17_TRAINER,
        V16_TRAINER,
        CONTROLLER_BASE,
        BATCH,
        PROTOCOL,
        INITIAL_MODEL,
        V11_RECEIPT,
        V18_RECEIPT,
        SIMULATOR_SOURCE,
        TERMINATION_SOURCE,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    if sha(INITIAL_MODEL) != EXPECTED_INITIAL:
        raise RuntimeError("Ant v19 initialization hash mismatch")
    if sha(V11_RECEIPT) != EXPECTED_V11_RECEIPT:
        raise RuntimeError("Ant v19 admitted-v11 receipt hash mismatch")
    if sha(V18_RECEIPT) != EXPECTED_V18_RECEIPT:
        raise RuntimeError("Ant v19 sealed-v18 receipt hash mismatch")
    if sha(SIMULATOR_SOURCE) != EXPECTED_SIMULATOR_SOURCE:
        raise RuntimeError("Ant v19 simulator source hash mismatch")
    if sha(TERMINATION_SOURCE) != EXPECTED_TERMINATION_SOURCE:
        raise RuntimeError("Ant v19 termination source hash mismatch")
    v11 = json.loads(V11_RECEIPT.read_text(encoding="utf-8"))
    v18 = json.loads(V18_RECEIPT.read_text(encoding="utf-8"))
    v11_summary = v11.get("evaluation", {}).get("summary", {})
    v18_summary = v18.get("evaluation", {}).get("summary", {})
    if (
        v11.get("status") != "pass"
        or v11.get("decision") != "admitted_to_fresh_maze_route_gate_v11"
        or v11_summary.get("episodes") != 96
        or v11_summary.get("success_rate") != 0.9270833333333334
    ):
        raise RuntimeError("v19 requires the exact admitted v11 antecedent")
    if (
        v18.get("status") != "fail"
        or v18.get("decision") != "ant_waypoint_v18_ineligible"
        or v18_summary.get("episodes") != 96
        or v18_summary.get("success_rate") != 0.2708333333333333
        or v18_summary.get("unhealthy_termination_rate")
        != 0.5208333333333334
    ):
        raise RuntimeError("v19 requires the exact immutable v18 antecedent")
    simulator_source = SIMULATOR_SOURCE.read_text(encoding="utf-8")
    for required in (
        "ant_obs, _, _, _, info = self.ant_env.step(action)",
        "terminated = self.compute_terminated",
    ):
        if required not in simulator_source:
            raise RuntimeError("Ant v19 termination diagnosis no longer binds")
    termination_source = TERMINATION_SOURCE.read_text(encoding="utf-8")
    if "if not self.continuing_task:" not in termination_source:
        raise RuntimeError("Ant v19 continuing-task diagnosis no longer binds")
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "ops")
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    run(["bash", "-n", str(BATCH)])
    run(
        [
            str(PYTHON),
            "-c",
            (
                "import train_ant_continuing_waypoint_controller_v19 as v;"
                "assert len(v.TRAINING_PATTERNS)==512;"
                "assert len(v.EVALUATION_PATTERNS)==24;"
                "assert len(set(v.EVALUATION_PATTERNS))==24;"
                "assert not (set(v.TRAINING_PATTERNS)&set(v.EVALUATION_PATTERNS));"
                "assert not (set(v.v18.v17.EVALUATION_PATTERNS)&set(v.EVALUATION_PATTERNS));"
                "assert not (set(v.v18.EVALUATION_PATTERNS)&set(v.EVALUATION_PATTERNS));"
                "assert len(v.TRAIN_MAPS)==4 and len(v.DEVELOPMENT_MAPS)==4;"
                "assert all(0<c<16 and 0<r<16 for p in v.TRAINING_PATTERNS "
                "for r,c in [v._goal_cell((8,8),p)]);"
                "assert all(0<c<20 and 0<r<20 for p in v.EVALUATION_PATTERNS "
                "for r,c in [v._goal_cell((10,10),p)]);"
                "e=v.AntContinuingStableHandoffEnv(rank=0,episode_steps=1600);"
                "o,_=e.reset(seed=73019);"
                "assert o.shape[-1]>2;"
                "assert e.env.unwrapped.continuing_task is True;"
                "assert e.env.unwrapped.reset_target is False;"
                "assert isinstance(v._ant_is_healthy(e),bool);"
                "e.close()"
            ),
        ],
        env=environment,
    )


def snapshot_execution() -> tuple[Path, str]:
    staging = Path(
        tempfile.mkdtemp(
            prefix=".ant-cont-v19.",
            dir=ROOT / "var/artifacts/source_snapshots",
        )
    )
    for source in (
        TRAINER,
        V18_TRAINER,
        V17_TRAINER,
        V16_TRAINER,
        CONTROLLER_BASE,
        BATCH,
    ):
        shutil.copy2(source, staging / source.name)
    digest = tree_hash(staging)
    target = ROOT / f"var/artifacts/source_snapshots/ant_cont_v19_ops_{digest}"
    if target.exists():
        shutil.rmtree(staging)
    else:
        os.replace(staging, target)
    if tree_hash(target) != digest:
        raise RuntimeError("Ant continuing v19 execution snapshot mismatch")
    return target, digest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("config", "run"))
    args = parser.parse_args()
    validate()
    if args.phase == "config":
        print("[ant-cont-v19] configuration passed; no job launched")
        return
    for path in (IDENTITY, SUBMISSION, MODEL, RECEIPT):
        if path.exists():
            raise FileExistsError(
                f"fresh Ant continuing v19 artifact required: {path}"
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
                raise RuntimeError(f"held Ant continuing v19 job lacks {required}")
        atomic(
            IDENTITY,
            {
                "schema_version": (
                    "ant-continuing-waypoint-controller-v19-identity-v1"
                ),
                "job_id": job_id,
                "execution_root": str(execution_root),
                "execution_hash": execution_hash,
                "protocol_sha256": sha(PROTOCOL),
                "trainer_sha256": sha(TRAINER),
                "trainer_base_sha256": sha(V18_TRAINER),
                "controller_base_sha256": sha(CONTROLLER_BASE),
                "batch_sha256": sha(BATCH),
                "simulator_source_sha256": EXPECTED_SIMULATOR_SOURCE,
                "termination_source_sha256": EXPECTED_TERMINATION_SOURCE,
                "initial_model_sha256": EXPECTED_INITIAL,
                "v11_admission_receipt_sha256": EXPECTED_V11_RECEIPT,
                "v18_failure_receipt_sha256": EXPECTED_V18_RECEIPT,
                "v18_failure_success_rate": 0.2708333333333333,
                "v18_confounded_termination_rate": 0.5208333333333334,
                "seed": 73019,
                "timesteps": 6_000_000,
                "workers": 8,
                "learning_rate": 5e-7,
                "episode_steps": 1600,
                "training_map_count": 4,
                "training_map_size": 17,
                "training_pattern_count": 512,
                "continuing_task": True,
                "reset_target": False,
                "explicit_ant_health": True,
                "stable_planar_speed": 1.0,
                "development_map_count": 4,
                "development_map_size": 21,
                "development_pattern_count": 24,
                "development_pattern_length": 8,
                "development_episode_count": 96,
                "evaluation_seed_offset": 13_000_000,
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
                    "ant-continuing-waypoint-controller-v19-submission-v1"
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
    print(f"[ant-cont-v19] released controller job {job_id}")


if __name__ == "__main__":
    main()
