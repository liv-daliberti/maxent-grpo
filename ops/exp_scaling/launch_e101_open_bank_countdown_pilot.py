#!/usr/bin/env python3
"""Submit E101: a three-arm, sub-hour open-bank Countdown mechanism pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import launch_e72_b3a_replay_ablation as base  # noqa: E402
import launch_e76_tuned_scale as snapshot_util  # noqa: E402

ARMS = ("mass", "balance", "open")
VARIANTS = {
    "mass": "verified_first_replay_rehearsal_only",
    "balance": "verified_first_replay_only_ablation",
    "open": "open_bank_maxent_replay",
}
SEED = 101
TRAIN_ROWS = 32
EVAL_ROWS = 32
PASSES = 4
TARGET_STEPS = TRAIN_ROWS * PASSES
EVAL_INTERVAL = 64
REPLAY_WEIGHT = 0.10
DATA_ROOT = "var/data/e101_open_bank_countdown_tiny"
SOURCE_MANIFEST = "var/artifacts/e72_frontier_source_runs.json"
PROTOCOL = "paper/preregistration/e101_open_bank_countdown_pilot_20260814.md"
LEDGER = "var/artifacts/e101_open_bank_countdown_pilot_jobs.json"
MODEL_TAG = "qwen25_0p5b_instruct"
NODES = "node103,node104,node205,node206"
PARTITION = "all"
ACCOUNT = "allcs"
GRES = "gpu:a6000:1"
MEMORY = "48G"
TIME_LIMIT = "00:55:00"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_digest(path: Path) -> str:
    value = hashlib.sha256()
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        value.update(str(item.relative_to(path)).encode())
        value.update(b"\0")
        value.update(item.read_bytes())
        value.update(b"\0")
    return value.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as sink:
            json.dump(payload, sink, indent=2, sort_keys=True)
            sink.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def reference(root: Path) -> dict[str, Any]:
    payload = json.loads((root / SOURCE_MANIFEST).read_text(encoding="utf-8"))
    matches = [
        run
        for run in payload["runs"]
        if run["arm"] == "xgrpo"
        and run["domain"] == "countdown"
        and int(run["seed"]) == 43
    ]
    if len(matches) != 1:
        raise SystemExit(f"expected one Countdown/s43 source template, found {len(matches)}")
    run = matches[0]
    evaluation = run["inherited_eval_config"]
    training = run["inherited_train_config"]
    if int(evaluation["num_samples"]) != 16:
        raise SystemExit("source template group size drifted from 16")
    if float(training["learning_rate"]) != 2e-7 or float(training["beta"]) != 0.0:
        raise SystemExit("source optimizer drifted from lr=2e-7, beta=0")
    return base.reseed(run, SEED)


def run_stamp(arm: str) -> str:
    return f"e101r1_open_bank_countdown_{arm}_s{SEED}"


def save_path(root: Path, arm: str) -> Path:
    return root / "var/data" / f"xdr_{MODEL_TAG}_{VARIANTS[arm]}_{run_stamp(arm)}"


def fixed_objective(arm: str) -> dict[str, str]:
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    split = arm != "mass"
    proposal = arm == "open"
    return {
        "OAT_ZERO_VARIANT": VARIANTS[arm],
        "OAT_ZERO_POLICY_ENTROPY_COEF": "0.0",
        "OAT_ZERO_XDR_TAU": "inf",
        "OAT_ZERO_SEED_ENTROPY_ALPHA": "0.0",
        "OAT_ZERO_MAXENT_ALPHA": "0.0",
        "OAT_ZERO_MAXENT_INVERSE_ADAPTATION": "0",
        "OAT_ZERO_SEMANTIC_SHANNON_COEF": "0.0",
        "OAT_ZERO_SEMANTIC_SHANNON_SEPARATE_ADVANTAGE": "0",
        "OAT_ZERO_SEMANTIC_SHANNON_QUALITY_GATED_ADVANTAGE": "0",
        "OAT_ZERO_SEMANTIC_SHANNON_SUCCESS_CONDITIONED_SIGNED_ADVANTAGE": "0",
        "OAT_ZERO_ONLINE_CANONICAL_BANK_ALPHA": "0.0",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY": "1",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_OBJECTIVE": (
            "split_mass_balance_per_rollout"
            if split
            else "verified_likelihood_per_rollout"
        ),
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_ALPHA": repr(REPLAY_WEIGHT),
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_MASS_ALPHA": repr(REPLAY_WEIGHT),
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_CAPACITY": "16",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_GLOBAL_GROUPS_PER_STEP": "1",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_GLOBAL_BOOTSTRAP_STEPS": "0",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_COMPUTE_ONLY": "0",
        "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_PROPOSALS": "1" if proposal else "0",
        "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_SEPARATE_OBJECTIVE_SUPPORT": (
            "1" if proposal else "0"
        ),
        "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_SINGLETON_ONLY": (
            "1" if proposal else "0"
        ),
        "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_TRANSFORM_PROPOSALS": (
            "0" if proposal else "1"
        ),
        "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_ANCHOR_MAX_TOKENS": "256",
        "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_MAX_ATTEMPTS": "1",
        "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_SAMPLING_TEMPERATURE": "1.0",
        "OAT_ZERO_BETA": "0.0",
    }


def build_env(
    root: Path,
    source: dict[str, Any],
    arm: str,
    snapshot_root: Path,
) -> tuple[dict[str, str], Path]:
    target = save_path(root, arm)
    env = base.build_export_vars(root, source, target, "b1a")
    data = root / DATA_ROOT
    env.update(
        {
            "SAVE_PATH": str(target),
            "RUN_STAMP": run_stamp(arm),
            "OAT_ZERO_SOURCE_ROOT": str(snapshot_root / "src"),
            "OAT_ZERO_OPS_SNAPSHOT_ROOT": str(snapshot_root / "ops"),
            "OAT_ZERO_PROMPT_DATA": str(data / "train"),
            "OAT_ZERO_EVAL_DATA": str(data / "eval"),
            "OAT_ZERO_TEST_SPLIT": "multi_answer",
            "OAT_ZERO_MAX_TRAIN": str(TRAIN_ROWS),
            "OAT_ZERO_SEED": str(SEED),
            "OAT_ZERO_NUM_PROMPT_EPOCH": str(PASSES),
            "OAT_ZERO_MAX_PROMPT_EPOCHS": str(PASSES),
            "OAT_ZERO_EVAL_PROMPT_INTERVAL": str(EVAL_INTERVAL),
            "OAT_ZERO_ALLOW_SPARSE_EVAL": "1",
            "OAT_ZERO_EVAL_BATCH_SIZE": str(EVAL_ROWS),
            "OAT_ZERO_EVAL_MODE_COVERAGE_K": "8",
            "OAT_ZERO_EVAL_MODE_COVERAGE_DRAWS": "2",
            "OAT_ZERO_EVAL_MODE_COVERAGE_SEED": "810101",
            "OAT_ZERO_SAVE_CKPT": "0",
            "OAT_ZERO_EXPORT_STEPS": "0",
            "OAT_ZERO_AUTO_RESUME": "0",
            "OAT_ZERO_WATCHDOG_REQUEUE": "0",
            "OAT_ZERO_USE_WB": "0",
            "OAT_ZERO_NUM_GPUS_PER_ACTOR": "1",
            "OAT_ZERO_REPLICATED_FREEFORM_SAMPLING": "1",
            "OAT_ZERO_LOCAL_ACTOR_WEIGHT_SYNC": "1",
            "OAT_ZERO_VLLM_SLEEP_LEVEL": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    env.update(fixed_objective(arm))
    return env, target


def sbatch_command(root: Path, arm: str, env: dict[str, str]) -> list[str]:
    export_pairs = ",".join(f"{key}={value}" for key, value in env.items())
    return [
        "sbatch",
        "--parsable",
        "--hold",
        f"--job-name=e101-{arm}-s{SEED}",
        f"--export=ALL,{export_pairs}",
        f"--partition={PARTITION}",
        f"--account={ACCOUNT}",
        f"--nodelist={NODES}",
        f"--gres={GRES}",
        "--cpus-per-task=8",
        f"--mem={MEMORY}",
        f"--time={TIME_LIMIT}",
        "--nice=100",
        str(root / "ops/slurm/train_node302.slurm"),
    ]


def held_job_audit(job_id: str, arm: str) -> str:
    result = subprocess.run(
        ["scontrol", "show", "job", "-dd", "-o", job_id],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"cannot inspect held job {job_id}: {result.stderr.strip()}")
    record = result.stdout
    required = (
        "JobState=PENDING",
        "Reason=JobHeldUser",
        f"JobName=e101-{arm}-s{SEED}",
        f"Account={ACCOUNT}",
        f"Partition={PARTITION}",
        "gres/gpu:a6000:1",
        f"OAT_ZERO_VARIANT={VARIANTS[arm]}",
        f"OAT_ZERO_SEED={SEED}",
        f"OAT_ZERO_MAX_TRAIN={TRAIN_ROWS}",
        f"OAT_ZERO_NUM_PROMPT_EPOCH={PASSES}",
        "OAT_ZERO_SEMANTIC_SHANNON_COEF=0.0",
        "OAT_ZERO_ONLINE_CANONICAL_BANK_ALPHA=0.0",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY=1",
        "OAT_ZERO_NUM_GPUS_PER_ACTOR=1",
        "OAT_ZERO_REPLICATED_FREEFORM_SAMPLING=1",
        "OAT_ZERO_LOCAL_ACTOR_WEIGHT_SYNC=1",
        "OAT_ZERO_VLLM_SLEEP_LEVEL=1",
    )
    missing = [item for item in required if item not in record]
    if missing:
        raise RuntimeError(f"held job {job_id} lacks {missing}")
    objective = fixed_objective(arm)
    for key in (
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_OBJECTIVE",
        "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_PROPOSALS",
        "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_SEPARATE_OBJECTIVE_SUPPORT",
        "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_SINGLETON_ONLY",
        "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_TRANSFORM_PROPOSALS",
    ):
        expected = f"{key}={objective[key]}"
        if expected not in record:
            raise RuntimeError(f"held job {job_id} lacks {expected}")
    return record


def cancel(job_ids: list[str]) -> None:
    if job_ids:
        subprocess.run(["scancel", *job_ids], check=False)


def main() -> int:
    root = repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--snapshot-root", type=Path)
    args = parser.parse_args()
    if args.submit and args.dry_run:
        raise SystemExit("choose --submit or --dry-run, not both")

    required = (
        root / PROTOCOL,
        root / SOURCE_MANIFEST,
        root / DATA_ROOT / "train",
        root / DATA_ROOT / "eval",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit(f"required frozen inputs are absent: {missing}")
    ledger = root / LEDGER
    if args.submit and ledger.exists():
        raise SystemExit(f"refusing duplicate E101 submission: {ledger}")

    source = reference(root)
    snapshot_root = snapshot_util.ensure_snapshot(root, args.snapshot_root)
    planned: list[dict[str, Any]] = []
    for arm in ARMS:
        env, target = build_env(root, source, arm, snapshot_root)
        if target.exists():
            raise SystemExit(f"refusing to overwrite existing E101 run: {target}")
        planned.append(
            {
                "arm": arm,
                "run_stamp": run_stamp(arm),
                "run_dir": str(target),
                "env": env,
                "command": sbatch_command(root, arm, env),
            }
        )

    if args.dry_run or not args.submit:
        for cell in planned:
            print(" ".join(shlex.quote(part) for part in cell["command"]))
        print(f"[e101] dry_run=True cells={len(planned)} snapshot={snapshot_root}")
        return 0

    submitted: list[str] = []
    records: list[dict[str, Any]] = []
    try:
        for cell in planned:
            result = subprocess.run(
                cell["command"], capture_output=True, text=True, check=False
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"submission failed for {cell['run_stamp']}: {result.stderr.strip()}"
                )
            job_id = result.stdout.strip().split(";", 1)[0]
            if not job_id.isdigit():
                raise RuntimeError(f"invalid job id: {result.stdout!r}")
            submitted.append(job_id)
            held_record = held_job_audit(job_id, str(cell["arm"]))
            records.append(
                {
                    "arm": cell["arm"],
                    "variant": VARIANTS[str(cell["arm"])],
                    "seed": SEED,
                    "run_stamp": cell["run_stamp"],
                    "run_dir": cell["run_dir"],
                    "job_id": int(job_id),
                    "stdout": str(root / f"var/artifacts/logs/e101-{cell['arm']}-s{SEED}-{job_id}.out"),
                    "stderr": str(root / f"var/artifacts/logs/e101-{cell['arm']}-s{SEED}-{job_id}.err"),
                    "objective": fixed_objective(str(cell["arm"])),
                    "held_scheduler_record": held_record,
                }
            )

        protocol = root / PROTOCOL
        manifest = root / SOURCE_MANIFEST
        data = root / DATA_ROOT
        payload = {
            "schema": "e101_open_bank_countdown_pilot_jobs_v1",
            "protocol": str(protocol),
            "protocol_sha256": digest(protocol),
            "launcher_sha256": digest(Path(__file__)),
            "source_manifest": str(manifest),
            "source_manifest_sha256": digest(manifest),
            "snapshot_root": str(snapshot_root),
            "data_root": str(data),
            "data_tree_sha256": tree_digest(data),
            "model": "Qwen2.5-0.5B-Instruct",
            "arms": list(ARMS),
            "seed": SEED,
            "train_rows": TRAIN_ROWS,
            "eval_rows": EVAL_ROWS,
            "passes": PASSES,
            "target_steps": TARGET_STEPS,
            "registered_evaluation_steps": [0, EVAL_INTERVAL, TARGET_STEPS],
            "group_size": 16,
            "coverage_k": 8,
            "coverage_draws": 2,
            "learning_rate": 2e-7,
            "replay_weight": REPLAY_WEIGHT,
            "time_limit": TIME_LIMIT,
            "placement": {
                "partition": PARTITION,
                "account": ACCOUNT,
                "nodes": NODES.split(","),
                "gres": GRES,
                "memory": MEMORY,
            },
            "runs": records,
            "released": False,
        }
        atomic_json(ledger, payload)
        for job_id in submitted:
            release = subprocess.run(
                ["scontrol", "release", job_id],
                capture_output=True,
                text=True,
                check=False,
            )
            if release.returncode != 0:
                raise RuntimeError(f"release failed for {job_id}: {release.stderr.strip()}")
        payload["released"] = True
        atomic_json(ledger, payload)
    except Exception:
        cancel(submitted)
        if ledger.exists():
            ledger.unlink()
        raise

    print(
        f"[e101] cells={len(records)} released={len(submitted)} "
        f"snapshot={snapshot_root} ledger={ledger}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
