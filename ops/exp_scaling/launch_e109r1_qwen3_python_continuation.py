#!/usr/bin/env python3
"""Continue the two preempted E109 Qwen-3B Python comparator cells."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "ops"))
import apply_e117s1_lowprio_partition_repair as scheduler  # noqa: E402
import launch_e117_same_plumbing_component_preflight as e117  # noqa: E402
import status_e78 as status  # noqa: E402
import validate_deepspeed_checkpoint as checkpoint  # noqa: E402


PROTOCOL = ROOT / (
    "paper/preregistration/"
    "e109r1_qwen3_python_preemption_continuation_20260824.md"
)
ROUTING_AMENDMENT = ROOT / (
    "paper/preregistration/e109r1s1_submit_route_repair_20260825.md"
)
ORIGINAL_LEDGER = ROOT / (
    "var/artifacts/e109_repaired_python_replay_comparators_jobs.json"
)
ARTIFACT = ROOT / (
    "var/artifacts/e109r1_qwen3_python_continuation_jobs.json"
)
TARGETS = {
    73: {"original_job_id": 30659554, "progress": 2391, "checkpoint": 2304},
    74: {"original_job_id": 30659555, "progress": 2197, "checkpoint": 2112},
}
NODE_LIST = "node[104,205-207,805]"


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"command failed {command}: {result.stderr.strip()}")
    return result


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def select_targets(ledger: dict[str, Any]) -> list[dict[str, Any]]:
    if (
        ledger.get("schema")
        != "e109_repaired_python_replay_comparators_jobs_v1"
        or ledger.get("released") is not True
        or ledger.get("pointmaze") != "excluded"
    ):
        raise RuntimeError("E109 ledger identity drifted")
    selected = [
        row
        for row in ledger.get("runs", [])
        if row.get("scale") == "qwen3b" and int(row.get("seed", -1)) in TARGETS
    ]
    if len(selected) != 2:
        raise RuntimeError("E109 continuation requires exactly two Qwen-3B cells")
    for row in selected:
        seed = int(row["seed"])
        if int(row["job_id"]) != TARGETS[seed]["original_job_id"]:
            raise RuntimeError(f"E109 seed {seed} original job drifted")
        if row.get("domain") != "python_factors" or row.get("arm") != "replay":
            raise RuntimeError(f"E109 seed {seed} scientific cell drifted")
    return sorted(selected, key=lambda row: int(row["seed"]))


def original_accounting(job_id: int) -> str:
    result = run(
        [
            "sacct",
            "-n",
            "-X",
            "-j",
            str(job_id),
            "--format=JobIDRaw,State,ExitCode,Elapsed,Restarts,NodeList,Partition",
            "--parsable2",
        ]
    )
    rows = [line for line in result.stdout.splitlines() if line.startswith(f"{job_id}|")]
    if len(rows) != 1 or rows[0].split("|", 2)[1].split()[0] != "PREEMPTED":
        raise RuntimeError(f"E109 original job {job_id} is not uniquely PREEMPTED")
    return rows[0]


def build_command(row: dict[str, Any], environment: str) -> list[str]:
    seed = int(row["seed"])
    return [
        "sbatch",
        "--parsable",
        "--hold",
        f"--job-name=e109r1-q3-python-s{seed}",
        f"--export={environment}",
        "--partition=all",
        "--account=allcs",
        f"--nodelist={NODE_LIST}",
        "--gres=gpu:a6000:1",
        "--cpus-per-task=16",
        "--mem=128G",
        "--time=3-00:00:00",
        "--nice=100",
        f"--chdir={ROOT}",
        f"--output={ROOT}/var/artifacts/logs/e109r1-q3-python-s{seed}-%j.out",
        f"--error={ROOT}/var/artifacts/logs/e109r1-q3-python-s{seed}-%j.err",
        str(ROOT / "ops/slurm/train_node302.slurm"),
    ]


def audit_held(
    record: str, *, seed: int, environment: str, partition: str
) -> None:
    required = {
        "JobName": f"e109r1-q3-python-s{seed}",
        "JobState": "PENDING",
        "Reason": "JobHeldUser",
        "RunTime": "00:00:00",
        "Restarts": "0",
        "Partition": partition,
        "Account": "allcs",
        "ReqNodeList": NODE_LIST,
        "NumCPUs": "16",
        "MinMemoryNode": "128G",
        "TimeLimit": "3-00:00:00",
        "Nice": "100",
        "TresPerNode": "gres/gpu:a6000:1",
    }
    failures = {
        field: (scheduler.field(record, field), expected)
        for field, expected in required.items()
        if scheduler.field(record, field) != expected
    }
    if scheduler.environment(record) != environment:
        failures["environment_sha256"] = (
            scheduler.sha256_text(scheduler.environment(record)),
            scheduler.sha256_text(environment),
        )
    if failures:
        raise RuntimeError(f"held E109-R1 identity drifted: {failures}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    if not args.submit:
        raise SystemExit("pass --submit to launch E109-R1")
    if ARTIFACT.exists():
        raise SystemExit(f"refusing duplicate E109-R1 launch: {ARTIFACT}")
    ledger = json.loads(ORIGINAL_LEDGER.read_text(encoding="utf-8"))
    rows = select_targets(ledger)
    original_ids = [int(row["job_id"]) for row in rows]
    live = run(["squeue", "-h", "-j", ",".join(map(str, original_ids)), "-o", "%i"])
    if live.stdout.strip():
        raise SystemExit(f"E109 originals are still active: {live.stdout.strip()}")

    prepared = []
    for row in rows:
        seed = int(row["seed"])
        run_dir = Path(str(row["run_dir"]))
        if not run_dir.is_dir() or status.receipt_step(run_dir) != 0:
            raise SystemExit(f"E109 seed {seed} is absent or already terminal")
        progress = status.run_step(run_dir)
        if progress != TARGETS[seed]["progress"]:
            raise SystemExit(f"E109 seed {seed} progress drifted: {progress}")
        selected, rejected = checkpoint.select_latest_checkpoint(run_dir)
        expected_checkpoint = TARGETS[seed]["checkpoint"]
        if (
            selected is None
            or int(selected.name.removeprefix("step_")) != expected_checkpoint
            or checkpoint.validate_checkpoint(selected)
        ):
            raise SystemExit(f"E109 seed {seed} lacks checkpoint {expected_checkpoint}")
        environment = scheduler.environment(str(row["held_scheduler_record"]))
        required_exports = (
            f"OAT_ZERO_SEED={seed}",
            f"SAVE_PATH={run_dir}",
            "OAT_ZERO_AUTO_RESUME=1",
            "OAT_ZERO_MAX_TRAIN=384",
            "OAT_ZERO_NUM_PROMPT_EPOCH=8",
            "OAT_ZERO_SEMANTIC_SHANNON_COEF=0.0",
            "OAT_ZERO_ONLINE_CANONICAL_REPLAY=1",
        )
        missing = [value for value in required_exports if value not in environment]
        if missing:
            raise SystemExit(f"E109 seed {seed} export drifted: {missing}")
        prepared.append(
            {
                "row": row,
                "seed": seed,
                "run_dir": run_dir,
                "progress": progress,
                "checkpoint": selected,
                "checkpoint_rejections": rejected,
                "environment": environment,
                "original_accounting": original_accounting(int(row["job_id"])),
            }
        )

    submitted: list[int] = []
    records = []
    try:
        for item in prepared:
            result = run(build_command(item["row"], item["environment"]))
            job_text = result.stdout.strip().split(";", 1)[0]
            if not job_text.isdigit():
                raise RuntimeError(f"invalid E109-R1 job id: {result.stdout!r}")
            job_id = int(job_text)
            submitted.append(job_id)
            routed_held = scheduler.show(job_id)
            audit_held(
                routed_held,
                seed=item["seed"],
                environment=item["environment"],
                partition="cs",
            )
            run(
                [
                    "scontrol",
                    "update",
                    f"JobId={job_id}",
                    "Partition=all",
                    "Account=allcs",
                ]
            )
            held = scheduler.show(job_id)
            audit_held(
                held,
                seed=item["seed"],
                environment=item["environment"],
                partition="all",
            )
            records.append(
                {
                    "seed": item["seed"],
                    "original_job_id": int(item["row"]["job_id"]),
                    "continuation_job_id": job_id,
                    "run_dir": str(item["run_dir"]),
                    "run_stamp": str(item["row"]["run_stamp"]),
                    "training_progress_before": item["progress"],
                    "resume_checkpoint": str(item["checkpoint"]),
                    "resume_checkpoint_step": TARGETS[item["seed"]]["checkpoint"],
                    "checkpoint_rejections": item["checkpoint_rejections"],
                    "environment_sha256": scheduler.sha256_text(item["environment"]),
                    "original_accounting": item["original_accounting"],
                    "routed_held_scheduler_record": routed_held,
                    "held_scheduler_record": held,
                }
            )
        payload = {
            "schema": "e109r1_qwen3_python_continuation_jobs_v1",
            "protocol": str(PROTOCOL),
            "protocol_sha256": e117.digest(PROTOCOL),
            "routing_amendment": str(ROUTING_AMENDMENT),
            "routing_amendment_sha256": e117.digest(ROUTING_AMENDMENT),
            "zero_runtime_canceled_attempt_job_id": 30873997,
            "launcher": str(Path(__file__).resolve()),
            "launcher_sha256": e117.digest(Path(__file__)),
            "original_ledger": str(ORIGINAL_LEDGER),
            "original_ledger_sha256": e117.digest(ORIGINAL_LEDGER),
            "exact_original_job_ids": original_ids,
            "exact_seeds": [73, 74],
            "same_scientific_cells": True,
            "same_run_directories": True,
            "same_a6000_hardware_class": True,
            "scientific_environment_changed": False,
            "outcomes_inspected": False,
            "pointmaze": "excluded",
            "partition": "all",
            "account": "allcs",
            "node_list": NODE_LIST,
            "records": records,
            "released": False,
        }
        atomic_json(ARTIFACT, payload)
        run(["scontrol", "release", *map(str, submitted)])
        for record in records:
            current = scheduler.show(int(record["continuation_job_id"]))
            if scheduler.environment(current) != prepared[
                record["seed"] - 73
            ]["environment"]:
                raise RuntimeError("E109-R1 environment changed after release")
            if scheduler.field(current, "Partition") != "all":
                raise RuntimeError("E109-R1 left partition all after release")
            record["released_scheduler_record"] = current
        payload["released"] = True
        atomic_json(ARTIFACT, payload)
    except Exception:
        if submitted:
            subprocess.run(
                ["scancel", *map(str, submitted)],
                capture_output=True,
                text=True,
                check=False,
            )
        if ARTIFACT.exists():
            ARTIFACT.unlink()
        raise
    print(f"[e109r1] continuations={submitted} released=2")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
