#!/usr/bin/env python3
"""Submit the 192-step Falcon Python admission-horizon replacement gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any


sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_e104_group_centered_semantic_repair as mechanism  # noqa: E402
import launch_e79_falcon1b_aligned_verified_replay as e79  # noqa: E402
import launch_e81_semantic_maxent_verified_replay_05b as e81  # noqa: E402
import launch_e106_python_lambda_normalization_three_scale as e106  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "paper/preregistration/e110_falcon_python_admission_horizon_20260818.md"
LEDGER = "var/artifacts/e110_falcon_python_admission_horizon_jobs.json"
E79_LEDGER = "var/artifacts/e79_falcon1b_aligned_verified_replay_jobs.json"
E106_LEDGER = e106.LEDGER
JOB_ID_SUPERSEDED = 30640330
HISTORICAL_JOB_ID = 30269053
SCALE = "falcon1b"
DOMAIN = "python_factors"
SEED = 55
TRAIN_ROWS = 192
PASSES = 1
TARGET_STEPS = TRAIN_ROWS * PASSES
CHECKPOINT_INTERVAL = 64
SUBMIT_PARTITION = "cs"
PARTITION = "all"
ACCOUNT = "allcs"
NODE_LIST = "node[103-104,205-208,805]"
TIME_LIMIT = "03:00:00"
RUN_STAMP = "e110_falcon1b_python_admission_horizon_s55"
CANCELLED_ZERO_RUNTIME_ATTEMPT = 30647351


def load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"required artifact is absent: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return e81.digest(path)


def run_dir(root: Path) -> Path:
    return root / "var/data" / (
        "xdr_falcon3_1b_instruct_"
        "verified_replay_semantic_maxent_group_centered_"
        f"{RUN_STAMP}"
    )


def historical_admission(root: Path) -> dict[str, Any]:
    ledger_path = root / E79_LEDGER
    ledger = load(ledger_path)
    matches = [
        item
        for item in ledger.get("runs", [])
        if int(item.get("job_id", -1)) == HISTORICAL_JOB_ID
        and item.get("domain") == DOMAIN
        and int(item.get("seed", -1)) == SEED
    ]
    if len(matches) != 1:
        raise RuntimeError("historical Falcon Python seed-55 run drifted")
    run = matches[0]
    record = str(run.get("held_scheduler_record", ""))
    required = (
        "OAT_ZERO_VARIANT=verified_first_replay_rehearsal_only",
        "OAT_ZERO_SEED=55",
        "OAT_ZERO_PROMPT_TEMPLATE=falcon_boxed",
        "OAT_ZERO_GENERATE_MAX_LENGTH=512",
        "OAT_ZERO_EVAL_GENERATE_MAX_LENGTH=512",
        "OAT_ZERO_NUM_SAMPLES=16",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY=1",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_COMPUTE_ONLY=0",
    )
    missing = [needle for needle in required if needle not in record]
    if missing:
        raise RuntimeError(f"historical Falcon Python surface lacks {missing}")
    metrics = Path(str(run["run_dir"])) / (
        f"debug_job{HISTORICAL_JOB_ID}/train_metrics.jsonl"
    )
    if not metrics.is_file():
        raise RuntimeError("historical Falcon Python metrics are absent")
    admitting: list[dict[str, float]] = []
    for line in metrics.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        bank = float(row.get("train/online_canonical_bank_size_after_mean", 0.0))
        if bank > 0.0:
            admitting.append(
                {
                    "step": float(row["misc/global_step"]),
                    "bank_size": bank,
                }
            )
    if not admitting or int(admitting[0]["step"]) != 179:
        raise RuntimeError("historical first Falcon Python admission drifted")
    return {
        "job_id": HISTORICAL_JOB_ID,
        "run_dir": str(run["run_dir"]),
        "metrics": str(metrics),
        "metrics_sha256": digest(metrics),
        "first_admission_step": 179,
        "max_bank_size": max(item["bank_size"] for item in admitting),
        "surface": {
            "model": "Falcon3-1B-Instruct",
            "domain": DOMAIN,
            "seed": SEED,
            "prompt_template": "falcon_boxed",
            "generate_max_length": 512,
            "num_samples": 16,
        },
        "mechanism_only": True,
        "evaluation_outcomes_inspected": False,
    }


def failed_e106_evidence(root: Path) -> dict[str, Any]:
    ledger_path = root / E106_LEDGER
    ledger = load(ledger_path)
    matches = [
        item
        for item in ledger.get("runs", [])
        if int(item.get("job_id", -1)) == JOB_ID_SUPERSEDED
    ]
    if len(matches) != 1:
        raise RuntimeError("failed E106 Falcon Python cell drifted")
    run = matches[0]
    report, violations = mechanism.parse_run(Path(str(run["run_dir"])))
    expected_violation = "verified replay never produced an applied update"
    unexpected = [item for item in violations if item != expected_violation]
    if unexpected or violations.count(expected_violation) != 1:
        raise RuntimeError(f"failed E106 trace is malformed: {violations}")
    if (
        int(report["last_step"]) != 64
        or float(report["bank_size_after_max"]) != 0.0
        or float(report["history_rows_added_max"]) != 0.0
        or float(report["replay_gradient_l2_max"]) != 0.0
        or float(report["group_centered_active_min"]) != 1.0
        or float(report["legacy_active_max"]) != 0.0
        or float(report["controller_active_max"]) != 0.0
    ):
        raise RuntimeError("failed E106 mechanism diagnosis drifted")
    marker = Path(str(run["run_dir"])) / "TRAINING_COMPLETE.json"
    metrics = Path(report["metric_paths"][0])
    if not marker.is_file() or not metrics.is_file():
        raise RuntimeError("failed E106 completion evidence is absent")
    record = str(run.get("held_scheduler_record", ""))
    for needle in (
        "OAT_ZERO_PROMPT_TEMPLATE=falcon_boxed",
        "OAT_ZERO_GENERATE_MAX_LENGTH=512",
        "OAT_ZERO_NUM_SAMPLES=16",
        "OAT_ZERO_SEED=55",
    ):
        if needle not in record:
            raise RuntimeError(f"failed E106 surface lacks {needle}")
    return {
        "job_id": JOB_ID_SUPERSEDED,
        "run_dir": str(run["run_dir"]),
        "last_step": 64,
        "bank_size_after_max": 0.0,
        "history_rows_added_max": 0.0,
        "replay_gradient_l2_max": 0.0,
        "metrics": str(metrics),
        "metrics_sha256": digest(metrics),
        "completion_marker": str(marker),
        "completion_marker_sha256": digest(marker),
        "mechanism_only": True,
        "evaluation_outcomes_inspected": False,
    }


def build_env(root: Path, snapshot: Path) -> tuple[dict[str, str], dict[str, Any]]:
    template = e106.python_template(root, SCALE)
    env, _target = e106.build_env(root, SCALE, template, snapshot)
    env.update(
        {
            "SAVE_PATH": str(run_dir(root)),
            "RUN_STAMP": RUN_STAMP,
            "OAT_ZERO_MAX_TRAIN": str(TRAIN_ROWS),
            "OAT_ZERO_NUM_PROMPT_EPOCH": str(PASSES),
            "OAT_ZERO_MAX_PROMPT_EPOCHS": str(PASSES),
            "OAT_ZERO_EVAL_PROMPT_INTERVAL": str(CHECKPOINT_INTERVAL),
            "OAT_ZERO_SAVE_STEPS": str(CHECKPOINT_INTERVAL),
            "OAT_ZERO_SAVE_FROM": str(CHECKPOINT_INTERVAL),
            "OAT_ZERO_RESUME_STEPS": str(CHECKPOINT_INTERVAL),
            "OAT_ZERO_SOURCE_ROOT": str(snapshot / "src"),
            "OAT_ZERO_OPS_SNAPSHOT_ROOT": str(snapshot / "ops"),
        }
    )
    return env, template


def sbatch_command(
    root: Path, template: dict[str, Any], env: dict[str, str]
) -> list[str]:
    original = e106.sbatch_command(root, SCALE, template, env)
    output: list[str] = []
    for token in original:
        if token.startswith("--job-name="):
            output.append("--job-name=e110-f1-python")
        elif token.startswith("--partition="):
            output.append(f"--partition={SUBMIT_PARTITION}")
        elif token.startswith("--account="):
            output.append(f"--account={ACCOUNT}")
        elif token.startswith("--nodelist="):
            output.append(f"--nodelist={NODE_LIST}")
        elif token.startswith("--time="):
            output.append(f"--time={TIME_LIMIT}")
        elif token.startswith("--nice="):
            output.append("--nice=0")
        else:
            output.append(token)
    return output


def scheduler_record(job_id: str | int) -> str:
    result = subprocess.run(
        ["scontrol", "show", "job", "-dd", "-o", str(job_id)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"cannot inspect E110 job {job_id}")
    return result.stdout


def cancelled_attempt_evidence() -> str:
    record = scheduler_record(CANCELLED_ZERO_RUNTIME_ATTEMPT)
    required = (
        "JobState=CANCELLED",
        "RunTime=00:00:00",
        "Partition=cs",
        "Reason=JobHeldUser",
        "--partition=all",
        "JobName=e110-f1-python",
    )
    missing = [needle for needle in required if needle not in record]
    if missing:
        raise RuntimeError(f"cancelled E110 attempt lacks {missing}")
    return record


def update_held_partition(job_id: str) -> None:
    result = subprocess.run(
        [
            "scontrol",
            "update",
            f"JobId={job_id}",
            f"Partition={PARTITION}",
            f"Account={ACCOUNT}",
            f"NodeList={NODE_LIST}",
            "Gres=gpu:a6000:1",
            f"TimeLimit={TIME_LIMIT}",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"E110 held placement update failed: {detail}")


def held_audit(job_id: str, snapshot: Path) -> str:
    record = scheduler_record(job_id)
    required = (
        "JobState=PENDING",
        "Reason=JobHeldUser",
        "JobName=e110-f1-python",
        f"Partition={PARTITION}",
        f"Account={ACCOUNT}",
        f"ReqNodeList={NODE_LIST}",
        "TresPerNode=gres/gpu:a6000:1",
        "NumCPUs=8",
        "MinMemoryNode=64G",
        f"TimeLimit={TIME_LIMIT}",
        f"SAVE_PATH={run_dir(ROOT)}",
        f"OAT_ZERO_SOURCE_ROOT={snapshot / 'src'}",
        f"OAT_ZERO_OPS_SNAPSHOT_ROOT={snapshot / 'ops'}",
        "OAT_ZERO_SEED=55",
        "OAT_ZERO_MAX_TRAIN=192",
        "OAT_ZERO_NUM_PROMPT_EPOCH=1",
        "OAT_ZERO_EVAL_PROMPT_INTERVAL=64",
        "OAT_ZERO_GENERATE_MAX_LENGTH=512",
        "OAT_ZERO_EVAL_GENERATE_MAX_LENGTH=512",
        "OAT_ZERO_NUM_SAMPLES=16",
        "OAT_ZERO_SEMANTIC_SHANNON_COEF=0.1",
        "OAT_ZERO_SEMANTIC_SHANNON_SUCCESS_CONDITIONED_GROUP_CENTERED_ADVANTAGE=1",
        "OAT_ZERO_SEMANTIC_RMS_CONTROL=0",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY=1",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_ALPHA=0.1",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_COMPUTE_ONLY=0",
    )
    missing = [needle for needle in required if needle not in record]
    if missing:
        raise RuntimeError(f"held E110 job {job_id} lacks {missing}")
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    if args.dry_run == args.submit:
        raise SystemExit("choose exactly one of --dry-run or --submit")
    root = ROOT
    protocol = root / PROTOCOL
    ledger_path = root / LEDGER
    snapshot = (root / e106.SNAPSHOT).resolve()
    if not protocol.is_file():
        raise SystemExit(f"E110 protocol is absent: {protocol}")
    e106.verify_snapshot(root, snapshot)
    if ledger_path.exists():
        raise SystemExit(f"refusing duplicate E110 ledger: {ledger_path}")
    target = run_dir(root)
    if target.exists():
        raise SystemExit(f"refusing existing E110 run directory: {target}")
    history = historical_admission(root)
    failed = failed_e106_evidence(root)
    cancelled_attempt = cancelled_attempt_evidence()
    env, template = build_env(root, snapshot)
    command = sbatch_command(root, template, env)
    if args.dry_run:
        print(" ".join(shlex.quote(token) for token in command))
        print(
            f"scontrol update JobId=<held_job_id> Partition={PARTITION} "
            f"Account={ACCOUNT} NodeList={NODE_LIST} Gres=gpu:a6000:1 "
            f"TimeLimit={TIME_LIMIT}"
        )
        print(
            f"[e110-dry-run] historical_first_admission="
            f"{history['first_admission_step']} failed_gate_step={failed['last_step']}"
        )
        return 0

    submitted: list[str] = []
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip()
            raise RuntimeError(f"E110 submission failed: {detail}")
        job_id = result.stdout.strip().split(";", 1)[0]
        if not job_id.isdigit():
            raise RuntimeError(f"invalid E110 job id: {result.stdout!r}")
        submitted.append(job_id)
        submitted_record = scheduler_record(job_id)
        if (
            "JobState=PENDING" not in submitted_record
            or "Reason=JobHeldUser" not in submitted_record
            or "RunTime=00:00:00" not in submitted_record
            or f"Partition={SUBMIT_PARTITION}" not in submitted_record
        ):
            raise RuntimeError("E110 submit-time held placement drifted")
        update_held_partition(job_id)
        held = held_audit(job_id, snapshot)
        payload = {
            "schema": "e110_falcon_python_admission_horizon_jobs_v1",
            "protocol": PROTOCOL,
            "protocol_sha256": digest(protocol),
            "launcher": str(Path(__file__).relative_to(root)),
            "launcher_sha256": digest(Path(__file__)),
            "snapshot_root": str(snapshot),
            "snapshot_sha256": e106.SNAPSHOT_SHA256,
            "base_e79_ledger": E79_LEDGER,
            "base_e79_ledger_sha256": digest(root / E79_LEDGER),
            "base_e106_ledger": E106_LEDGER,
            "base_e106_ledger_sha256": digest(root / E106_LEDGER),
            "historical_admission_evidence": history,
            "failed_e106_mechanism_evidence": failed,
            "cancelled_zero_runtime_attempt": {
                "job_id": CANCELLED_ZERO_RUNTIME_ATTEMPT,
                "scheduler_record": cancelled_attempt,
            },
            "submitted_held_scheduler_record": submitted_record,
            "submitted_partition": SUBMIT_PARTITION,
            "effective_partition": PARTITION,
            "supersedes_job_id": JOB_ID_SUPERSEDED,
            "replacement_scope": [SCALE, DOMAIN, SEED],
            "models": [SCALE],
            "domains": [DOMAIN],
            "seeds": [SEED],
            "train_rows": TRAIN_ROWS,
            "passes": PASSES,
            "target_steps": TARGET_STEPS,
            "checkpoint_interval_steps": CHECKPOINT_INTERVAL,
            "mechanism_gate_used_outcome_metrics": False,
            "post_update_outcome_metrics_inspected": False,
            "pointmaze": "excluded",
            "runs": [
                {
                    "scale": SCALE,
                    "model_tag": "falcon3_1b_instruct",
                    "domain": DOMAIN,
                    "seed": SEED,
                    "run_stamp": RUN_STAMP,
                    "run_dir": str(target),
                    "job_id": int(job_id),
                    "stdout": str(
                        root / "var/artifacts/logs" / f"e110-f1-python-{job_id}.out"
                    ),
                    "stderr": str(
                        root / "var/artifacts/logs" / f"e110-f1-python-{job_id}.err"
                    ),
                    "held_scheduler_record": held,
                }
            ],
            "released": False,
        }
        e81.atomic_json(ledger_path, payload)
        release = subprocess.run(
            ["scontrol", "release", job_id],
            capture_output=True,
            text=True,
            check=False,
        )
        if release.returncode != 0:
            raise RuntimeError(
                "E110 release failed: "
                + (release.stderr.strip() or release.stdout.strip())
            )
        payload["released"] = True
        e81.atomic_json(ledger_path, payload)
    except Exception:
        if submitted:
            subprocess.run(["scancel", *submitted], check=False)
        if ledger_path.exists():
            ledger_path.unlink()
        raise
    print(f"[e110] cells=1 released=1 ledger={ledger_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
