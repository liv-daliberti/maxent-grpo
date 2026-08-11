#!/usr/bin/env python3
"""Submit Falcon3-1B warm start and paired PointMaze replay experiment."""

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


PROTOCOL = (
    ROOT
    / "paper/preregistration/e79pm_falcon_point_maze_verified_replay_only_20260805.md"
)
SFT_SLURM = ROOT / "ops/slurm/e79pm_falcon_point_maze_sft.slurm"
TRAIN_SLURM = ROOT / "ops/slurm/e79pm_falcon_point_maze_train.slurm"
SFT_RUNNER = ROOT / "ops/train_point_maze_waypoint_warmstart_v1.py"
TRAIN_RUNNER = ROOT / "ops/train_point_maze_verified_replay_only.py"
BASE_MODEL = (
    ROOT
    / "var/cache/huggingface/transformers/models--tiiuae--Falcon3-1B-Instruct"
    / "snapshots/28ba2251970a01dd1edc7ba7dad2eb71216ccfdf"
)
MODEL_REVISION = "28ba2251970a01dd1edc7ba7dad2eb71216ccfdf"
SFT_DATA = ROOT / "var/data/point_maze_waypoint_warmstart_e75r3"
SFT_MODEL = ROOT / "var/models/e79pm_falcon_point_maze_warmstart"
SFT_RECEIPT = ROOT / "var/artifacts/e79pm_falcon_point_maze_warmstart.json"
ONLINE_DATA = ROOT / "var/data/point_maze_waypoint_e78pm"
E78PM_LEDGER = (
    ROOT / "var/artifacts/e78pm_point_maze_verified_replay_only_05b_jobs.json"
)
WORKER = ROOT / "var/maze_runtime/venv/bin/python"
PYTHON = ROOT / "var/seed_paper_eval/paper310/bin/python"
LEDGER = ROOT / "var/artifacts/e79pm_falcon_point_maze_verified_replay_jobs.json"
NODE = "node208"
ARMS = ("control", "replay")
SEEDS = (55, 56, 57, 58, 59)
PASSES = 8
TRAIN_ROWS = 384
DEV_ROWS = 64
EVAL_ROWS = 128
CHECKPOINT_INTERVAL = TRAIN_ROWS // 2
TARGET_STEPS = PASSES * TRAIN_ROWS
REPLAY_WEIGHT = 0.10


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            value.update(chunk)
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
    stem = ROOT / f"var/artifacts/e79pm_falcon_point_maze_{arm}_s{seed}"
    return {
        "receipt": Path(str(stem) + ".json"),
        "metrics": Path(str(stem) + ".metrics.jsonl"),
        "replay": Path(str(stem) + ".replay.jsonl"),
        "model": ROOT / f"var/models/e79pm_falcon_point_maze_{arm}_s{seed}",
        "checkpoint": ROOT / f"var/checkpoints/e79pm_falcon_point_maze_{arm}_s{seed}",
    }


def shared_prepare_job() -> str:
    payload = json.loads(E78PM_LEDGER.read_text(encoding="utf-8"))
    if (
        payload.get("schema")
        != "e78pm_point_maze_verified_replay_only_jobs_v1"
        or int(payload.get("train_rows", -1)) != TRAIN_ROWS
        or int(payload.get("dev_rows", -1)) != DEV_ROWS
        or int(payload.get("eval_rows", -1)) != EVAL_ROWS
        or Path(str(payload.get("data_root", ""))).resolve() != ONLINE_DATA.resolve()
    ):
        raise SystemExit("shared E78-PM data contract changed")
    job_id = str(payload.get("prepare_job_id", ""))
    if not job_id.isdigit():
        raise SystemExit("shared E78-PM preparation job is missing")
    return job_id


def prerequisites() -> None:
    for path in (
        PROTOCOL,
        SFT_SLURM,
        TRAIN_SLURM,
        SFT_RUNNER,
        TRAIN_RUNNER,
        BASE_MODEL / "config.json",
        BASE_MODEL / "tokenizer_config.json",
        SFT_DATA / "identity.json",
        SFT_DATA / "examples.jsonl",
        E78PM_LEDGER,
        WORKER,
        PYTHON,
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
            str(Path(__file__).resolve()),
            str(SFT_RUNNER),
            str(TRAIN_RUNNER),
        ],
        cwd=ROOT,
        env=environment,
        check=True,
    )
    for script in (SFT_SLURM, TRAIN_SLURM):
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
            str(ROOT / "tests/test_e79pm_falcon_point_maze_verified_replay.py"),
        ],
        cwd=ROOT,
        env=environment,
        check=True,
    )
    script = (
        "from pathlib import Path; "
        "from transformers import AutoTokenizer; "
        "from oat_drgrpo.point_maze_waypoint_policy import "
        "convert_point_waypoint_prompt; "
        f"p=Path({str(BASE_MODEL)!r}); "
        "t=AutoTokenizer.from_pretrained(p,local_files_only=True," 
        "trust_remote_code=False); "
        "ids=[t.encode(x,add_special_tokens=False) for x in 'ABCD']; "
        "assert all(len(x)==1 for x in ids) and len({x[0] for x in ids})==4; "
        f"e=Path({str(SFT_DATA / 'examples.jsonl')!r}).open().readline(); "
        "import json; q=json.loads(e)['prompt']; "
        "f=convert_point_waypoint_prompt(q,prompt_format='falcon3'); "
        "assert f.startswith('<|system|>\\n') and f.endswith('<|assistant|>\\n')"
    )
    subprocess.run([str(PYTHON), "-c", script], cwd=ROOT, env=environment, check=True)


def sft_command(snapshot: Path) -> list[str]:
    exports = ",".join(
        (
            f"ROOT_DIR={ROOT}",
            f"OAT_ZERO_SOURCE_ROOT={snapshot / 'src'}",
            f"OAT_ZERO_EXECUTION_ROOT={snapshot / 'ops'}",
            f"OAT_ZERO_BASE_MODEL={BASE_MODEL}",
        )
    )
    return [
        "sbatch",
        "--parsable",
        "--hold",
        "--job-name=e79pm-falcon-sft",
        f"--export=ALL,{exports}",
        "--partition=all",
        "--account=allcs",
        f"--nodelist={NODE}",
        "--gres=gpu:a6000:1",
        "--cpus-per-task=8",
        "--mem=64G",
        "--time=12:00:00",
        "--nice=100",
        str(snapshot / "ops/slurm/e79pm_falcon_point_maze_sft.slurm"),
    ]


def train_command(
    snapshot: Path,
    arm: str,
    seed: int,
    *,
    sft_job: str,
    prepare_job: str,
) -> list[str]:
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
        f"--dependency=afterok:{sft_job}:{prepare_job}",
        f"--job-name=e79pm-pm-{arm[:3]}-s{seed}",
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
        str(snapshot / "ops/slurm/e79pm_falcon_point_maze_train.slurm"),
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
    prepare_job = shared_prepare_job()
    if LEDGER.exists():
        raise SystemExit(f"refusing duplicate E79-PM ledger: {LEDGER}")
    for path in (SFT_MODEL, SFT_RECEIPT):
        if path.exists():
            raise SystemExit(f"refusing pre-existing Falcon warm start: {path}")
    for arm in ARMS:
        for seed in SEEDS:
            for path in output_paths(arm, seed).values():
                if path.exists():
                    raise SystemExit(f"refusing pre-existing E79-PM output: {path}")

    snapshot = snapshot_util.ensure_snapshot(ROOT, args.snapshot_root)
    if args.dry_run or not args.submit:
        print(" ".join(shlex.quote(part) for part in sft_command(snapshot)))
        for arm in ARMS:
            for seed in SEEDS:
                command = train_command(
                    snapshot,
                    arm,
                    seed,
                    sft_job="SFT_JOB",
                    prepare_job=prepare_job,
                )
                print(" ".join(shlex.quote(part) for part in command))
        print(
            f"[e79pm] dry_run=True cells=10 snapshot={snapshot} "
            f"prepare_job={prepare_job}"
        )
        return 0

    jobs: list[str] = []
    try:
        sft_job = submit(sft_command(snapshot))
        jobs.append(sft_job)
        sft_record = scheduler_record(
            sft_job,
            (
                "JobState=PENDING",
                "Reason=JobHeldUser",
                "JobName=e79pm-falcon-sft",
                f"ReqNodeList={NODE}",
                f"OAT_ZERO_BASE_MODEL={BASE_MODEL}",
                f"OAT_ZERO_EXECUTION_ROOT={snapshot / 'ops'}",
            ),
        )
        runs = []
        for arm in ARMS:
            for seed in SEEDS:
                job_id = submit(
                    train_command(
                        snapshot,
                        arm,
                        seed,
                        sft_job=sft_job,
                        prepare_job=prepare_job,
                    )
                )
                jobs.append(job_id)
                record = scheduler_record(
                    job_id,
                    (
                        "JobState=PENDING",
                        "Reason=JobHeldUser",
                        f"JobName=e79pm-pm-{arm[:3]}-s{seed}",
                        f"ReqNodeList={NODE}",
                        f"afterok:{sft_job}",
                        prepare_job,
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
            "schema": "e79pm_falcon_point_maze_verified_replay_jobs_v1",
            "experiment": "E79-PM",
            "relationship_to_e79": (
                "prospective_separately_identified_sixth_domain_extension"
            ),
            "protocol": str(PROTOCOL),
            "protocol_sha256": digest(PROTOCOL),
            "launcher_sha256": digest(Path(__file__)),
            "snapshot_root": str(snapshot),
            "snapshot_identity_sha256": digest(snapshot / "SNAPSHOT_IDENTITY.json"),
            "base_model": str(BASE_MODEL),
            "base_model_revision": MODEL_REVISION,
            "base_model_config_sha256": digest(BASE_MODEL / "config.json"),
            "prompt_format": "falcon3",
            "warmstart": {
                "job_id": int(sft_job),
                "model": str(SFT_MODEL),
                "receipt": str(SFT_RECEIPT),
                "data_identity_sha256": digest(SFT_DATA / "identity.json"),
                "examples_sha256": digest(SFT_DATA / "examples.jsonl"),
                "seed": 88404,
                "optimizer_steps": 72,
                "held_scheduler_record": sft_record,
            },
            "data_root": str(ONLINE_DATA),
            "prepare_job_id": int(prepare_job),
            "shared_data_prepare_job_id": int(prepare_job),
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
            "optimizer": {
                "name": "AdamW",
                "learning_rate": 2e-7,
                "scheduler": "constant",
                "adam_betas": [0.9, 0.999],
                "epsilon": 1e-8,
                "weight_decay": 0.0,
                "max_grad_norm": 1.0,
            },
            "runs": runs,
            "released": False,
        }
        atomic_json(LEDGER, payload)
        subprocess.run(["scontrol", "release", *jobs[1:]], cwd=ROOT, check=True)
        subprocess.run(["scontrol", "release", sft_job], cwd=ROOT, check=True)
        payload["released"] = True
        atomic_json(LEDGER, payload)
        print(
            f"[e79pm] warmstart={sft_job} cells=10 released=11 ledger={LEDGER}",
            flush=True,
        )
        return 0
    except BaseException:
        cancel(jobs)
        if LEDGER.exists():
            LEDGER.unlink()
        raise


if __name__ == "__main__":
    raise SystemExit(main())
