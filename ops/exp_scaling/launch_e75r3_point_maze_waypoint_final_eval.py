#!/usr/bin/env python3
"""Freeze and submit the one-shot E75R3 untouched final evaluation."""

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
DATA_IDENTITY = ROOT / "var/data/point_maze_waypoint_pilot_e75r3/identity.json"
E75R3_IDENTITY = ROOT / "var/artifacts/e75r3_point_maze_waypoint_05b_identity.json"
DEV_QUALIFICATION = ROOT / "var/artifacts/e75r3_point_maze_waypoint_dev_qualification.json"
PROTOCOL = ROOT / "paper/preregistration/e75r3_point_maze_waypoint_final_eval_20260804.md"
PLAN = ROOT / "var/artifacts/e75r3_point_maze_waypoint_final_eval_plan.json"
MANIFEST = ROOT / "var/artifacts/e75r3_point_maze_waypoint_final_eval_jobs.tsv"
SUBMISSION = ROOT / "var/artifacts/e75r3_point_maze_waypoint_final_eval_submission.json"
ANALYSIS_JSON = ROOT / "var/artifacts/e75r3_point_maze_waypoint_final_analysis.json"
ANALYSIS_MARKDOWN = ROOT / "var/artifacts/e75r3_point_maze_waypoint_final_analysis.md"
SNAPSHOT_PARENT = ROOT / "var/artifacts/source_snapshots"
EVAL_SLURM = ROOT / "ops/slurm/e75r3_point_maze_waypoint_final_eval.slurm"
ANALYSIS_SLURM = ROOT / "ops/slurm/e75r3_point_maze_waypoint_final_analysis.slurm"
ANALYZER = ROOT / "ops/exp_scaling/analyze_e75r3_point_maze_waypoint_final_eval.py"
TEST = ROOT / "tests/test_e75r3_point_maze_waypoint_final_eval.py"
SEED = 88504
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
        list(command), cwd=ROOT, env=None if env is None else dict(env),
        check=True, capture_output=True, text=True,
    )
    return completed.stdout.strip()


def training_paths(arm: str) -> dict[str, Path]:
    stem = ROOT / f"var/artifacts/e75r3_point_maze_waypoint_{arm}_s{SEED}"
    return {
        "model": ROOT / f"var/models/e75r3_point_maze_waypoint_{arm}_s{SEED}",
        "receipt": Path(str(stem) + ".json"),
        "metrics": Path(str(stem) + ".metrics.jsonl"),
        "replay": Path(str(stem) + ".replay.jsonl"),
    }


def final_paths(arm: str) -> dict[str, Path]:
    stem = ROOT / f"var/artifacts/e75r3_point_maze_waypoint_final_eval_{ARM_SHORT[arm]}"
    return {
        "receipt": Path(str(stem) + ".json"),
        "metrics": Path(str(stem) + ".metrics.jsonl"),
        "replay": Path(str(stem) + ".replay.jsonl"),
    }


def prerequisites() -> tuple[dict[str, Any], dict[str, Any]]:
    required = (
        PYTHON, WORKER, DATA_IDENTITY, E75R3_IDENTITY, DEV_QUALIFICATION,
        PROTOCOL, EVAL_SLURM, ANALYSIS_SLURM, ANALYZER, TEST,
        ROOT / "ops/repo_env.sh",
    )
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    identity = json.loads(E75R3_IDENTITY.read_text())
    qualification = json.loads(DEV_QUALIFICATION.read_text())
    if identity.get("schema") != "e75r3-point-maze-waypoint-05b-identity-v1":
        raise RuntimeError("unexpected E75R3 identity")
    if identity.get("evaluation_job_submitted") is not False or identity.get("eval_maps_held_out") != 64:
        raise RuntimeError("E75R3 identity does not preserve the evaluation firewall")
    if qualification.get("status") != "pass" or qualification.get("eligible_for_online_pilot") is not True:
        raise RuntimeError("E75R3 development qualification is not a pass")
    data_sha256 = sha256_file(DATA_IDENTITY)
    checkpoints: dict[str, Any] = {}
    for arm in ARMS:
        paths = training_paths(arm)
        for path in paths.values():
            if not path.exists():
                raise FileNotFoundError(path)
        receipt = json.loads(paths["receipt"].read_text())
        expected = {
            "status": "complete", "arm": arm, "seed": SEED,
            "evaluation_only": False, "evaluation_split": "dev",
            "evaluation_prompt_count": 32, "optimizer_updates": 64,
        }
        for key, value in expected.items():
            if receipt.get(key) != value:
                raise RuntimeError(f"{arm}: training receipt {key} is not frozen terminal state")
        if receipt.get("data_identity_sha256") != data_sha256:
            raise RuntimeError(f"{arm}: training receipt data identity mismatch")
        model_hash = tree_sha256(paths["model"])
        if model_hash != receipt.get("output_model_tree_sha256"):
            raise RuntimeError(f"{arm}: terminal checkpoint hash mismatch")
        if sha256_file(paths["metrics"]) != receipt.get("metrics_sha256"):
            raise RuntimeError(f"{arm}: terminal metrics hash mismatch")
        if sha256_file(paths["replay"]) != receipt.get("state_replay_sha256"):
            raise RuntimeError(f"{arm}: terminal replay hash mismatch")
        checkpoints[arm] = {
            "online_update": 64,
            "path": str(paths["model"].resolve()),
            "tree_sha256": model_hash,
            "training_receipt": str(paths["receipt"].resolve()),
            "training_receipt_sha256": sha256_file(paths["receipt"]),
            "training_metrics": str(paths["metrics"].resolve()),
            "training_metrics_sha256": sha256_file(paths["metrics"]),
            "training_replay": str(paths["replay"].resolve()),
            "training_replay_sha256": sha256_file(paths["replay"]),
        }
    return identity, checkpoints


def validate() -> None:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = f"{ROOT / 'ops'}:{ROOT / 'src'}"
    library = str(ROOT / "var/seed_paper_eval/paper310/lib")
    environment["LD_LIBRARY_PATH"] = library + (":" + environment["LD_LIBRARY_PATH"] if environment.get("LD_LIBRARY_PATH") else "")
    run([str(PYTHON), "-m", "py_compile", str(Path(__file__).resolve()), str(ANALYZER)], env=environment)
    run(["bash", "-n", str(EVAL_SLURM)])
    run(["bash", "-n", str(ANALYSIS_SLURM)])
    run([str(PYTHON), "-m", "pytest", "-q", str(TEST)], env=environment)


def snapshot_execution(identity: Mapping[str, Any]) -> tuple[Path, str]:
    prior = Path(identity["execution_root"])
    inputs = (
        prior / "train_point_maze_waypoint_pilot_v1.py",
        prior / "train_point_maze_interactive_paired_smoke_v1.py",
        EVAL_SLURM, ANALYSIS_SLURM, ANALYZER, PROTOCOL,
    )
    SNAPSHOT_PARENT.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".e75r3-final-eval.", dir=SNAPSHOT_PARENT))
    for source in inputs:
        shutil.copy2(source, temporary / source.name)
    digest = tree_sha256(temporary)
    target = SNAPSHOT_PARENT / f"e75r3_point_maze_final_eval_ops_{digest}"
    if not target.is_dir():
        os.replace(temporary, target)
    else:
        shutil.rmtree(temporary)
    if tree_sha256(target) != digest:
        raise RuntimeError("E75R3 final evaluator snapshot hash mismatch")
    return target, digest


def fresh_outputs() -> list[Path]:
    paths = [PLAN, MANIFEST, SUBMISSION, ANALYSIS_JSON, ANALYSIS_MARKDOWN]
    for arm in ARMS:
        paths.extend(final_paths(arm).values())
    return paths


def export_spec(*, identity: Mapping[str, Any], execution_root: Path, execution_hash: str, plan_hash: str, arm: str | None = None) -> str:
    values = {
        "ROOT_DIR": str(ROOT),
        "OAT_ZERO_SOURCE_ROOT": str(identity["source_root"]),
        "OAT_ZERO_SOURCE_HASH": str(identity["source_sha256"]),
        "OAT_ZERO_EXECUTION_ROOT": str(execution_root),
        "OAT_ZERO_EXECUTION_HASH": execution_hash,
        "OAT_ZERO_E75R3_FINAL_PLAN": str(PLAN),
        "OAT_ZERO_E75R3_FINAL_PLAN_SHA256": plan_hash,
    }
    if arm is not None:
        values["OAT_ZERO_ARM"] = arm
    return "ALL," + ",".join(f"{key}={value}" for key, value in values.items())


def submit_held(*, name: str, script: Path, identity: Mapping[str, Any], execution_root: Path, execution_hash: str, plan_hash: str, arm: str | None = None, dependencies: Sequence[int] = (), gpu: bool) -> int:
    command = [
        "sbatch", "--parsable", "--hold", f"--job-name={name}",
        "--partition=all", "--account=allcs",
        f"--cpus-per-task={8 if gpu else 2}", f"--mem={'64G' if gpu else '8G'}",
        f"--time={'02:00:00' if gpu else '00:20:00'}",
        "--export=" + export_spec(identity=identity, execution_root=execution_root, execution_hash=execution_hash, plan_hash=plan_hash, arm=arm),
    ]
    if dependencies:
        command.append("--dependency=afterok:" + ":".join(str(job) for job in dependencies))
    if gpu:
        command.append("--gres=gpu:a6000:1")
    command.append(str(execution_root / script.name))
    job_id = int(run(command).split(";", 1)[0])
    run(["scontrol", "update", f"JobId={job_id}", "Partition=all", "Requeue=0"])
    return job_id


def validate_job(*, job_id: int, name: str, plan_hash: str, execution_hash: str, dependencies: Sequence[int] = (), gpu: bool) -> str:
    record = run(["scontrol", "show", "job", "-o", str(job_id)])
    required = [
        f"JobName={name}", "JobState=PENDING", "Reason=JobHeldUser",
        "Account=allcs", "Partition=all", "Requeue=0",
        f"OAT_ZERO_EXECUTION_HASH={execution_hash}",
        f"OAT_ZERO_E75R3_FINAL_PLAN_SHA256={plan_hash}",
    ]
    if dependencies:
        required.extend(f"afterok:{job}" for job in dependencies)
    if gpu:
        required.extend(["NumCPUs=8", "MinMemoryNode=64G", "gres/gpu:a6000:1"])
    else:
        required.extend(["NumCPUs=2", "MinMemoryNode=8G"])
    missing = [value for value in required if value not in record]
    if missing:
        raise RuntimeError(f"E75R3 final held job {job_id} is missing {missing}")
    return record


def write_manifest(jobs: Mapping[str, int]) -> None:
    rows = []
    for arm in ARMS:
        rows.append({"stage": "evaluation", "arm": arm, "job_id": jobs[arm], "dependency": "", "primary_output": str(final_paths(arm)["receipt"])})
    rows.append({"stage": "analysis", "arm": "", "job_id": jobs["analysis"], "dependency": ":".join(str(jobs[arm]) for arm in ARMS), "primary_output": str(ANALYSIS_JSON)})
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("stage", "arm", "job_id", "dependency", "primary_output"), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("config", "run", "resume"))
    args = parser.parse_args()
    identity, checkpoints = prerequisites()
    validate()
    if args.phase == "config":
        print("[e75r3-final] configuration and frozen inputs passed; no jobs submitted")
        return 0
    required_fresh = fresh_outputs() if args.phase == "run" else [
        MANIFEST,
        SUBMISSION,
        ANALYSIS_JSON,
        ANALYSIS_MARKDOWN,
        *(path for arm in ARMS for path in final_paths(arm).values()),
    ]
    for path in required_fresh:
        if path.exists():
            raise FileExistsError(f"fresh one-shot E75R3 final output required: {path}")
    if tree_sha256(Path(identity["source_root"])) != identity["source_sha256"]:
        raise RuntimeError("E75R3 frozen source tree hash mismatch")
    if args.phase == "resume":
        if not PLAN.is_file():
            raise FileNotFoundError("resume requires the already-frozen E75R3 plan")
        plan = json.loads(PLAN.read_text())
        if plan.get("schema") != "e75r3-point-maze-waypoint-final-eval-plan-v1":
            raise RuntimeError("cannot resume an unexpected E75R3 plan")
        if plan.get("prior_eval_access") is not False or plan.get("exclusions") != []:
            raise RuntimeError("cannot resume a changed E75R3 plan")
        if plan.get("checkpoints") != checkpoints:
            raise RuntimeError("frozen E75R3 checkpoint identities changed")
        execution_root = Path(plan["execution_snapshot"]["path"])
        execution_hash = str(plan["execution_snapshot"]["tree_sha256"])
        if tree_sha256(execution_root) != execution_hash:
            raise RuntimeError("frozen E75R3 evaluator snapshot changed")
        plan_hash = sha256_file(PLAN)
    else:
        execution_root, execution_hash = snapshot_execution(identity)
        plan = {
        "schema": "e75r3-point-maze-waypoint-final-eval-plan-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment": "E75R3",
        "status": "frozen_before_eval_submission",
        "prior_eval_access": False,
        "e75r3_identity": {"path": str(E75R3_IDENTITY.resolve()), "sha256": sha256_file(E75R3_IDENTITY)},
        "dev_qualification": {"path": str(DEV_QUALIFICATION.resolve()), "sha256": sha256_file(DEV_QUALIFICATION)},
        "data_identity": {"path": str(DATA_IDENTITY.resolve()), "sha256": sha256_file(DATA_IDENTITY), "eval_maps": 64},
        "source_snapshot": {"path": str(identity["source_root"]), "tree_sha256": identity["source_sha256"]},
        "execution_snapshot": {"path": str(execution_root), "tree_sha256": execution_hash},
        "protocol": {"path": str(PROTOCOL.resolve()), "sha256": sha256_file(PROTOCOL)},
        "launcher_sha256": sha256_file(Path(__file__).resolve()),
        "arms": list(ARMS),
        "arm_short": ARM_SHORT,
        "checkpoints": checkpoints,
        "exclusions": [],
        "evaluation": {"split": "eval", "maps": 64, "trajectories_per_map": 8, "fixed_horizon": 64, "seed": SEED, "common_random_numbers_across_arms": True, "one_evaluation_per_arm": True},
        "analysis": {"paired_unit": "map_id", "metrics": ["mean8", "pass8", "distinct8", "modes_per_success"], "comparisons": ["current_minus_grpo", "delayed_minus_grpo", "delayed_minus_current"], "bootstrap_replicates": 10000, "bootstrap_base_seed": 756400, "interval": [0.025, 0.975], "confirmatory_tests": False},
        }
        # This write is the scientific freeze point and precedes every sbatch call.
        atomic_json(PLAN, plan)
        plan_hash = sha256_file(PLAN)
    jobs: dict[str, int] = {}
    submitted: list[int] = []
    released = False
    resume_audit = ({"operational_resume_after_pre_release_cancellation": [30257704, 30257705, 30257706, 30257707], "resume_launcher_sha256": sha256_file(Path(__file__).resolve())} if args.phase == "resume" else {})
    try:
        for arm in ARMS:
            jobs[arm] = submit_held(name=f"e75r3-eval-{ARM_SHORT[arm]}", script=EVAL_SLURM, identity=identity, execution_root=execution_root, execution_hash=execution_hash, plan_hash=plan_hash, arm=arm, gpu=True)
            submitted.append(jobs[arm])
        dependencies = [jobs[arm] for arm in ARMS]
        jobs["analysis"] = submit_held(name="e75r3-eval-analysis", script=ANALYSIS_SLURM, identity=identity, execution_root=execution_root, execution_hash=execution_hash, plan_hash=plan_hash, dependencies=dependencies, gpu=False)
        submitted.append(jobs["analysis"])
        records = {}
        for arm in ARMS:
            records[arm] = validate_job(job_id=jobs[arm], name=f"e75r3-eval-{ARM_SHORT[arm]}", plan_hash=plan_hash, execution_hash=execution_hash, gpu=True)
        records["analysis"] = validate_job(job_id=jobs["analysis"], name="e75r3-eval-analysis", plan_hash=plan_hash, execution_hash=execution_hash, dependencies=dependencies, gpu=False)
        write_manifest(jobs)
        atomic_json(SUBMISSION, {"schema": "e75r3-point-maze-waypoint-final-eval-submission-v1", "generated_at": datetime.now(timezone.utc).isoformat(), "plan_sha256": plan_hash, "manifest_sha256": sha256_file(MANIFEST), "jobs": jobs, "held_job_audit": "pass", "released": False, **resume_audit})
        run(["scontrol", "release", *[str(job) for job in submitted]])
        released = True
        atomic_json(SUBMISSION, {"schema": "e75r3-point-maze-waypoint-final-eval-submission-v1", "generated_at": datetime.now(timezone.utc).isoformat(), "plan_sha256": plan_hash, "manifest_sha256": sha256_file(MANIFEST), "jobs": jobs, "held_job_audit": "pass", "released": True, **resume_audit})
    except BaseException:
        if submitted and not released:
            subprocess.run(["scancel", *[str(job) for job in submitted]], check=False)
        raise
    print(json.dumps({"experiment": "E75R3-final", "jobs": jobs, "plan_sha256": plan_hash, "released": True}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
