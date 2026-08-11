#!/usr/bin/env python3
"""Submit the prospective PointMaze extension to E78."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import launch_e76_tuned_scale as snapshot_util  # noqa: E402


PROTOCOL = ROOT / "paper/preregistration/e78pm_point_maze_verified_replay_only_05b_20260804.md"
PREPARE = ROOT / "ops/slurm/e78pm_point_maze_prepare.slurm"
TRAIN = ROOT / "ops/slurm/e78pm_point_maze_train.slurm"
RUNNER = ROOT / "ops/train_point_maze_verified_replay_only.py"
GENERATOR = ROOT / "ops/make_point_maze_waypoint_pilot_v1_data.py"
DATA = ROOT / "var/data/point_maze_waypoint_e78pm"
MODEL = ROOT / "var/models/point_maze_waypoint_warmstart_e75r3"
WORKER = ROOT / "var/maze_runtime/venv/bin/python"
PYTHON = ROOT / "var/seed_paper_eval/paper310/bin/python"
LEDGER = ROOT / "var/artifacts/e78pm_point_maze_verified_replay_only_05b_jobs.json"
NODE = "node208"
ARMS = ("control", "replay")
SEEDS = (43, 44, 45, 46, 47)
PASSES = 8
TRAIN_ROWS = 384
DEV_ROWS = 64
EVAL_ROWS = 128
CHECKPOINT_INTERVAL = TRAIN_ROWS // 2
TARGET_STEPS = PASSES * TRAIN_ROWS
REPLAY_WEIGHT = 0.10
DATA_SEED = 88104
EXCLUDED_IDENTITIES = (
    ROOT / "var/data/point_maze_waypoint_pilot_v1/identity.json",
    ROOT / "var/data/point_maze_waypoint_pilot_e75r1/identity.json",
    ROOT / "var/data/point_maze_waypoint_pilot_e75r2/identity.json",
    ROOT / "var/data/point_maze_waypoint_pilot_e75r3/identity.json",
)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    if path.is_dir():
        for child in sorted(item for item in path.rglob("*") if item.is_file()):
            relative = child.relative_to(path).as_posix().encode("utf-8")
            value.update(len(relative).to_bytes(8, "big"))
            value.update(relative)
            value.update(child.read_bytes())
    else:
        value.update(path.read_bytes())
    return value.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, allow_nan=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def output_paths(arm: str, seed: int) -> dict[str, Path]:
    stem = ROOT / f"var/artifacts/e78pm_point_maze_{arm}_s{seed}"
    return {
        "receipt": Path(str(stem) + ".json"),
        "metrics": Path(str(stem) + ".metrics.jsonl"),
        "replay": Path(str(stem) + ".replay.jsonl"),
        "model": ROOT / f"var/models/e78pm_point_maze_{arm}_s{seed}",
        "checkpoint": ROOT / f"var/checkpoints/e78pm_point_maze_{arm}_s{seed}",
    }


def prerequisites() -> None:
    for path in (
        PROTOCOL,
        PREPARE,
        TRAIN,
        RUNNER,
        GENERATOR,
        MODEL / "config.json",
        WORKER,
        PYTHON,
        *EXCLUDED_IDENTITIES,
    ):
        if not path.exists():
            raise FileNotFoundError(path)


def validate() -> None:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = f"{ROOT / 'ops'}:{ROOT / 'src'}"
    subprocess.run(
        [
            str(PYTHON),
            "-m",
            "py_compile",
            str(RUNNER),
            str(Path(__file__).resolve()),
        ],
        cwd=ROOT,
        env=environment,
        check=True,
    )
    for script in (PREPARE, TRAIN):
        subprocess.run(["bash", "-n", str(script)], cwd=ROOT, check=True)
    subprocess.run(
        [
            str(PYTHON),
            "-m",
            "pytest",
            "-q",
            str(ROOT / "tests/test_point_maze_waypoint.py"),
            str(ROOT / "tests/test_point_maze_interactive_policy.py"),
            str(ROOT / "tests/test_e78pm_point_maze_verified_replay_only.py"),
        ],
        cwd=ROOT,
        env=environment,
        check=True,
    )


def prepare_command(snapshot: Path) -> list[str]:
    exports = ",".join(
        (
            f"ROOT_DIR={ROOT}",
            f"OAT_ZERO_SOURCE_ROOT={snapshot / 'src'}",
            f"OAT_ZERO_EXECUTION_ROOT={snapshot / 'ops'}",
        )
    )
    return [
        "sbatch",
        "--parsable",
        "--hold",
        "--job-name=e78pm-prepare",
        f"--export=ALL,{exports}",
        "--partition=all",
        "--account=allcs",
        f"--nodelist={NODE}",
        "--cpus-per-task=4",
        "--mem=24G",
        "--time=02:00:00",
        "--nice=100",
        str(snapshot / "ops/slurm/e78pm_point_maze_prepare.slurm"),
    ]


def train_command(snapshot: Path, arm: str, seed: int, prepare_job: str) -> list[str]:
    exports = ",".join(
        (
            f"ROOT_DIR={ROOT}",
            f"OAT_ZERO_SOURCE_ROOT={snapshot / 'src'}",
            f"OAT_ZERO_EXECUTION_ROOT={snapshot / 'ops'}",
            f"OAT_ZERO_ARM={arm}",
            f"OAT_ZERO_SEED={seed}",
        )
    )
    return [
        "sbatch",
        "--parsable",
        "--hold",
        f"--dependency=afterok:{prepare_job}",
        f"--job-name=e78pm-pm-{arm[:3]}-s{seed}",
        f"--export=ALL,{exports}",
        "--partition=all",
        "--account=allcs",
        f"--nodelist={NODE}",
        "--gres=gpu:a6000:1",
        "--cpus-per-task=8",
        "--mem=64G",
        "--time=7-00:00:00",
        "--nice=100",
        "--requeue",
        str(snapshot / "ops/slurm/e78pm_point_maze_train.slurm"),
    ]


def submit(command: list[str]) -> str:
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "sbatch failed")
    job_id = result.stdout.strip().split(";", 1)[0]
    if not job_id.isdigit():
        raise RuntimeError(f"invalid Slurm job id: {result.stdout!r}")
    return job_id


def scheduler_record(job_id: str, required: Iterable[str]) -> str:
    result = subprocess.run(
        ["scontrol", "show", "job", "-dd", "-o", job_id],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise RuntimeError(f"cannot inspect held job {job_id}")
    missing = [literal for literal in required if literal not in result.stdout]
    if missing:
        raise RuntimeError(f"held job {job_id} lacks {missing}")
    return result.stdout


def cancel(job_ids: Iterable[str]) -> None:
    values = [value for value in job_ids if value.isdigit()]
    if values:
        subprocess.run(["scancel", *values], cwd=ROOT, check=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--snapshot-root", type=Path)
    args = parser.parse_args()
    if args.submit and args.dry_run:
        raise SystemExit("choose --submit or --dry-run")
    prerequisites()
    validate()
    if LEDGER.exists():
        raise SystemExit(f"refusing duplicate E78-PM ledger: {LEDGER}")
    if DATA.exists():
        raise SystemExit(f"refusing pre-existing E78-PM data: {DATA}")
    for arm in ARMS:
        for seed in SEEDS:
            for path in output_paths(arm, seed).values():
                if path.exists():
                    raise SystemExit(f"refusing pre-existing E78-PM output: {path}")

    snapshot = snapshot_util.ensure_snapshot(ROOT, args.snapshot_root)
    if args.dry_run or not args.submit:
        print(" ".join(shlex.quote(part) for part in prepare_command(snapshot)))
        for arm in ARMS:
            for seed in SEEDS:
                print(
                    " ".join(
                        shlex.quote(part)
                        for part in train_command(snapshot, arm, seed, "PREPARE_JOB")
                    )
                )
        print(f"[e78pm] dry_run=True cells=10 snapshot={snapshot}")
        return 0

    jobs: list[str] = []
    try:
        prepare_job = submit(prepare_command(snapshot))
        jobs.append(prepare_job)
        prepare_record = scheduler_record(
            prepare_job,
            (
                "JobState=PENDING",
                "Reason=JobHeldUser",
                "JobName=e78pm-prepare",
                f"ReqNodeList={NODE}",
                "OAT_ZERO_EXECUTION_ROOT=",
            ),
        )
        runs = []
        for arm in ARMS:
            for seed in SEEDS:
                job_id = submit(train_command(snapshot, arm, seed, prepare_job))
                jobs.append(job_id)
                record = scheduler_record(
                    job_id,
                    (
                        "JobState=PENDING",
                        "Reason=JobHeldUser",
                        f"JobName=e78pm-pm-{arm[:3]}-s{seed}",
                        f"ReqNodeList={NODE}",
                        f"Dependency=afterok:{prepare_job}",
                        f"OAT_ZERO_ARM={arm}",
                        f"OAT_ZERO_SEED={seed}",
                    ),
                )
                paths = output_paths(arm, seed)
                runs.append(
                    {
                        "domain": "point_maze",
                        "arm": arm,
                        "seed": seed,
                        "job_id": int(job_id),
                        "metrics_path": str(paths["metrics"]),
                        "receipt_path": str(paths["receipt"]),
                        "checkpoint_dir": str(paths["checkpoint"]),
                        "held_scheduler_record": record,
                    }
                )
        payload = {
            "schema": "e78pm_point_maze_verified_replay_only_jobs_v1",
            "experiment": "E78-PM",
            "relationship_to_e78": "prospective_separately_identified_sixth_domain_extension",
            "protocol": str(PROTOCOL),
            "protocol_sha256": digest(PROTOCOL),
            "launcher_sha256": digest(Path(__file__)),
            "snapshot_root": str(snapshot),
            "snapshot_identity_sha256": digest(snapshot / "SNAPSHOT_IDENTITY.json"),
            "initial_model": str(MODEL),
            "initial_model_tree_sha256": digest(MODEL),
            "data_root": str(DATA),
            "data_seed": DATA_SEED,
            "excluded_data_identity_sha256": [digest(path) for path in EXCLUDED_IDENTITIES],
            "domains": ["point_maze"],
            "arms": list(ARMS),
            "seeds": list(SEEDS),
            "train_rows": TRAIN_ROWS,
            "dev_rows": DEV_ROWS,
            "eval_rows": EVAL_ROWS,
            "passes": PASSES,
            "target_steps": TARGET_STEPS,
            "checkpoint_interval_steps": CHECKPOINT_INTERVAL,
            "registered_passes": [index / 2 for index in range(17)],
            "replay_weight": REPLAY_WEIGHT,
            "objective": "uniform_verified_likelihood_only",
            "prepare_job_id": int(prepare_job),
            "prepare_held_scheduler_record": prepare_record,
            "runs": runs,
        }
        atomic_json(LEDGER, payload)
        subprocess.run(["scontrol", "release", *jobs[1:]], cwd=ROOT, check=True)
        subprocess.run(["scontrol", "release", prepare_job], cwd=ROOT, check=True)
        print(
            f"[e78pm] submitted prepare={prepare_job} cells=10 ledger={LEDGER}",
            flush=True,
        )
        return 0
    except BaseException:
        cancel(jobs)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
