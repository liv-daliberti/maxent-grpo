#!/usr/bin/env python3
"""Retry E98-R1 Pantry behind a resident-vLLM A100 smoke and audit."""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PRIMARY_LEDGER = ROOT / "var/artifacts/e98r1_sparse_rlep_dr_05b_jobs.json"
PRIOR_REPAIR = ROOT / "var/artifacts/e98r1_pantry_action_surface_repair_jobs.json"
OUT = ROOT / "var/artifacts/e98r1_pantry_memory_recovery_jobs.json"
PROTOCOL = ROOT / "paper/preregistration/e98r1_pantry_memory_recovery_20260818.md"
SMOKE_ROOT = ROOT / "var/data/e98r1_pantry_memory_safe_smoke_s44"
SMOKE_AUDIT = "ops/exp_scaling/audit_e98r1_sparse_rlep_smoke.py"
SEEDS = (43, 44, 45, 46, 47)
PRIOR_JOBS = {
    43: 30579531,
    44: 30579532,
    45: 30579533,
    46: 30579534,
    47: 30579535,
}
SNAPSHOT_NAME = "e98r1_pantry_action_surface_a15c103a95a0bfee"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def run(argv: list[str]) -> str:
    result = subprocess.run(argv, capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return result.stdout.strip()


def submit_held(argv: list[str]) -> int:
    return int(run(argv).split(";", 1)[0])


def scheduler_record(job_id: int) -> str:
    return run(["scontrol", "show", "job", "-dd", "-o", str(job_id)])


def scheduler_state(job_id: int) -> str:
    output = run(
        [
            "sacct",
            "-n",
            "-X",
            "-j",
            str(job_id),
            "--format=JobIDRaw,State",
            "--parsable2",
        ]
    )
    rows = [line.split("|") for line in output.splitlines() if line.strip()]
    row = next((fields for fields in rows if fields[0] == str(job_id)), None)
    if row is None:
        raise RuntimeError(f"cannot resolve E98-R1 job {job_id}")
    return row[1].split()[0].split("+", 1)[0]


def load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"missing E98-R1 artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def submit_line(record: str) -> list[str]:
    if " SubmitLine=" not in record or " WorkDir=" not in record:
        raise RuntimeError("E98-R1 scheduler record lacks SubmitLine")
    argv = shlex.split(record.split(" SubmitLine=", 1)[1].split(" WorkDir=", 1)[0])
    if not argv or argv[0] != "sbatch":
        raise RuntimeError("E98-R1 SubmitLine is not sbatch")
    return argv


def patch_export(token: str, updates: dict[str, str], additions: dict[str, str]) -> str:
    if not token.startswith("--export="):
        raise RuntimeError("not an export token")
    fields = token.split("=", 1)[1].split(",")
    output: list[str] = []
    remaining = dict(updates)
    seen: set[str] = set()
    for field in fields:
        name = field.split("=", 1)[0]
        if name in remaining:
            output.append(f"{name}={remaining.pop(name)}")
        else:
            output.append(field)
        seen.add(name)
    if remaining:
        raise RuntimeError(f"E98-R1 export lacks {sorted(remaining)}")
    for name, value in additions.items():
        if name in seen:
            raise RuntimeError(f"E98-R1 export already defines {name}")
        output.append(f"{name}={value}")
    return "--export=" + ",".join(output)


def common_memory_additions() -> dict[str, str]:
    return {
        "OAT_ZERO_VLLM_SLEEP": "0",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }


def smoke_command(run_record: dict[str, Any], snapshot: Path) -> list[str]:
    updates = {
        "OAT_ZERO_SOURCE_ROOT": str(snapshot / "src"),
        "OAT_ZERO_OPS_SNAPSHOT_ROOT": str(snapshot / "ops"),
        "SAVE_PATH": str(SMOKE_ROOT),
        "RUN_STAMP": "e98r1_pantry_memory_safe_smoke_s44",
        "OAT_ZERO_MAX_TRAIN": "32",
        "OAT_ZERO_NUM_PROMPT_EPOCH": "1",
        "OAT_ZERO_MAX_PROMPT_EPOCHS": "1",
        "OAT_ZERO_EVAL_PROMPT_INTERVAL": "32",
        "OAT_ZERO_SAVE_STEPS": "32",
        "OAT_ZERO_SAVE_FROM": "32",
        "OAT_ZERO_AUTO_RESUME": "0",
        "OAT_ZERO_WATCHDOG_REQUEUE": "0",
        "OAT_ZERO_VLLM_GPU_RATIO": "0.10",
    }
    result: list[str] = []
    for token in submit_line(str(run_record["held_scheduler_record"])):
        if token == "--hold" or token.startswith(("--dependency=", "--nodelist=")):
            continue
        if token.startswith("--job-name="):
            result.append("--job-name=e98r1-pantry-memory-smoke")
        elif token.startswith("--export="):
            result.append(patch_export(token, updates, common_memory_additions()))
        elif token.startswith("--partition="):
            result.append("--partition=mltheory")
        elif token.startswith("--account="):
            result.append("--account=mltheory")
        elif token.startswith("--gres="):
            result.append("--gres=gpu:a100:1")
        elif token.startswith("--mem="):
            result.append("--mem=32G")
        elif token.startswith("--time="):
            result.append("--time=02:00:00")
        elif token.startswith("--nice="):
            result.append("--nice=0")
        else:
            result.append(token)
    result.insert(result.index("--parsable") + 1, "--hold")
    result.insert(-1, "--nodelist=node302")
    return result


def audit_command(snapshot: Path, smoke_id: int) -> list[str]:
    python = ROOT / "var/seed_paper_eval/paper310/bin/python"
    wrapped = shlex.join(
        [
            str(python),
            str(snapshot / SMOKE_AUDIT),
            "--run-root",
            str(SMOKE_ROOT),
            "--expected-terminal-step",
            "32",
        ]
    )
    return [
        "sbatch",
        "--parsable",
        "--hold",
        "--job-name=e98r1-pantry-memory-audit",
        f"--dependency=afterok:{smoke_id}",
        f"--export=ALL,PYTHONPATH={snapshot / 'src'}",
        "--partition=all",
        "--account=allcs",
        "--cpus-per-task=2",
        "--mem=8G",
        "--time=00:15:00",
        "--nice=0",
        f"--output={ROOT / 'var/artifacts/logs'}/%x-%j.out",
        f"--error={ROOT / 'var/artifacts/logs'}/%x-%j.err",
        "--wrap",
        wrapped,
    ]


def placement(seed: int) -> tuple[str, str, str, str]:
    if seed in (43, 44):
        return "cs", "allcs", "gpu:a5000:1", "node202,node203,node204"
    return "mltheory", "mltheory", "gpu:a100:1", "node302"


def science_command(
    run_record: dict[str, Any], *, snapshot: Path, audit_id: int
) -> list[str]:
    seed = int(run_record["seed"])
    partition, account, gres, nodes = placement(seed)
    updates = {
        "OAT_ZERO_SOURCE_ROOT": str(snapshot / "src"),
        "OAT_ZERO_OPS_SNAPSHOT_ROOT": str(snapshot / "ops"),
        "OAT_ZERO_VLLM_GPU_RATIO": "0.10",
    }
    result: list[str] = []
    for token in submit_line(str(run_record["held_scheduler_record"])):
        if token == "--hold" or token.startswith(("--dependency=", "--nodelist=")):
            continue
        if token.startswith("--job-name="):
            result.append(f"--job-name=e98r1-pantry-s{seed}-rr2")
        elif token.startswith("--export="):
            result.append(patch_export(token, updates, common_memory_additions()))
        elif token.startswith("--partition="):
            result.append(f"--partition={partition}")
        elif token.startswith("--account="):
            result.append(f"--account={account}")
        elif token.startswith("--gres="):
            result.append(f"--gres={gres}")
        elif token.startswith("--nice="):
            result.append("--nice=0")
        else:
            result.append(token)
    result.insert(result.index("--parsable") + 1, "--hold")
    result.insert(-1, f"--dependency=afterok:{audit_id}")
    result.insert(-1, f"--nodelist={nodes}")
    return result


def audit_held(job_id: int, needles: tuple[str, ...]) -> str:
    record = scheduler_record(job_id)
    required = ("JobState=PENDING", "Reason=JobHeldUser", *needles)
    missing = [needle for needle in required if needle not in record]
    if missing:
        raise RuntimeError(f"E98-R1 recovery job {job_id} lacks {missing}")
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    if not PROTOCOL.is_file():
        raise SystemExit(f"missing E98-R1 recovery protocol: {PROTOCOL}")
    if args.submit and OUT.exists():
        raise SystemExit(f"refusing duplicate E98-R1 recovery: {OUT}")
    if args.submit and SMOKE_ROOT.exists():
        raise SystemExit(f"refusing existing E98-R1 smoke root: {SMOKE_ROOT}")

    primary = load(PRIMARY_LEDGER)
    original_primary = copy.deepcopy(primary)
    prior = load(PRIOR_REPAIR)
    if prior.get("released") is not True:
        raise RuntimeError("prior E98-R1 Pantry repair was not released")
    snapshot = Path(str(prior["repaired_snapshot"])).resolve()
    if snapshot.name != SNAPSHOT_NAME or not snapshot.is_dir():
        raise RuntimeError("E98-R1 repaired snapshot identity drifted")
    runs = {
        int(row["seed"]): row
        for row in primary.get("runs", [])
        if str(row.get("domain")) == "pantry_plan"
    }
    if set(runs) != set(SEEDS):
        raise RuntimeError("E98-R1 primary ledger lacks Pantry seeds")
    for seed, job_id in PRIOR_JOBS.items():
        if int(runs[seed]["job_id"]) != job_id:
            raise RuntimeError(f"E98-R1 Pantry s{seed} identity drifted")
        if scheduler_state(job_id) != "CANCELLED":
            raise RuntimeError(f"E98-R1 Pantry s{seed} is not cancelled")
        if list(Path(str(runs[seed]["run_dir"])).glob("**/checkpoints/step_*")):
            raise RuntimeError(f"E98-R1 Pantry s{seed} has a checkpoint")

    smoke_argv = smoke_command(runs[44], snapshot)
    if not args.submit:
        print(shlex.join(smoke_argv))
        print("<audit afterok smoke>")
        for seed in SEEDS:
            print(f"<Pantry s{seed} afterok audit>")
        return 0

    submitted: list[int] = []
    committed = False
    try:
        smoke_id = submit_held(smoke_argv)
        submitted.append(smoke_id)
        smoke_record = audit_held(
            smoke_id,
            (
                "JobName=e98r1-pantry-memory-smoke",
                "Partition=mltheory",
                "ReqNodeList=node302",
                "TresPerNode=gres/gpu:a100:1",
                "MinMemoryNode=32G",
                "OAT_ZERO_MAX_TRAIN=32",
                "OAT_ZERO_VLLM_GPU_RATIO=0.10",
                "OAT_ZERO_VLLM_SLEEP=0",
                "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
                f"OAT_ZERO_SOURCE_ROOT={snapshot / 'src'}",
            ),
        )
        audit_id = submit_held(audit_command(snapshot, smoke_id))
        submitted.append(audit_id)
        audit_record = audit_held(
            audit_id,
            (
                "JobName=e98r1-pantry-memory-audit",
                f"Dependency=afterok:{smoke_id}",
                str(SMOKE_ROOT),
                "--expected-terminal-step",
            ),
        )

        replacements: list[dict[str, Any]] = []
        for seed in SEEDS:
            job_id = submit_held(
                science_command(runs[seed], snapshot=snapshot, audit_id=audit_id)
            )
            submitted.append(job_id)
            partition, account, gres, nodes = placement(seed)
            normalized_nodes = "node[202-204]" if seed in (43, 44) else "node302"
            normalized_gres = (
                "gres/gpu:a5000=1" if seed in (43, 44) else "gres/gpu=1"
            )
            record = audit_held(
                job_id,
                (
                    f"JobName=e98r1-pantry-s{seed}-rr2",
                    f"Dependency=afterok:{audit_id}",
                    f"Partition={partition}",
                    f"Account={account}",
                    f"ReqNodeList={normalized_nodes}",
                    normalized_gres,
                    f"OAT_ZERO_SEED={seed}",
                    "OAT_ZERO_RLEP_REPLAY_COUNT=2",
                    "OAT_ZERO_RLEP_SPARSE_FALLBACK=1",
                    "OAT_ZERO_VLLM_GPU_RATIO=0.10",
                    "OAT_ZERO_VLLM_SLEEP=0",
                    f"SAVE_PATH={runs[seed]['run_dir']}",
                    f"OAT_ZERO_SOURCE_ROOT={snapshot / 'src'}",
                ),
            )
            replacements.append(
                {
                    "seed": seed,
                    "old_job_id": PRIOR_JOBS[seed],
                    "new_job_id": job_id,
                    "placement": {
                        "partition": partition,
                        "account": account,
                        "gres": gres,
                        "nodes": nodes,
                    },
                    "held_scheduler_record": record,
                }
            )

        timestamp = dt.datetime.now(dt.timezone.utc).isoformat()
        for replacement in replacements:
            seed = int(replacement["seed"])
            row = runs[seed]
            row.setdefault("replaced_job_ids", []).append(PRIOR_JOBS[seed])
            row.setdefault("repair_history", []).append(
                {
                    "reason": "prior action-surface smoke hit contaminated-GPU CuMem wake OOM",
                    "protocol": str(PROTOCOL),
                    "submitted_at": timestamp,
                    "old_job_id": PRIOR_JOBS[seed],
                    "old_state": "CANCELLED",
                    "new_job_id": replacement["new_job_id"],
                    "snapshot_root": str(snapshot),
                    "smoke_job_id": smoke_id,
                    "smoke_audit_job_id": audit_id,
                    "placement": replacement["placement"],
                    "vllm_sleep": False,
                    "vllm_gpu_ratio": 0.10,
                    "held_scheduler_record": replacement["held_scheduler_record"],
                }
            )
            row["job_id"] = replacement["new_job_id"]
            row["held_scheduler_record"] = replacement["held_scheduler_record"]
            row["snapshot_root"] = str(snapshot)
            row["smoke_audit_dependency_job_id"] = audit_id

        payload: dict[str, Any] = {
            "schema": "e98r1_pantry_memory_recovery_jobs_v1",
            "released": False,
            "protocol": str(PROTOCOL),
            "protocol_sha256": digest(PROTOCOL),
            "primary_ledger": str(PRIMARY_LEDGER),
            "prior_repair": str(PRIOR_REPAIR),
            "snapshot_root": str(snapshot),
            "memory_recovery": {
                "vllm_sleep": False,
                "vllm_gpu_ratio": 0.10,
                "pytorch_cuda_alloc_conf": "expandable_segments:True",
            },
            "smoke": {
                "job_id": smoke_id,
                "audit_job_id": audit_id,
                "run_root": str(SMOKE_ROOT),
                "target_steps": 32,
                "held_scheduler_record": smoke_record,
                "held_audit_scheduler_record": audit_record,
            },
            "replacements": replacements,
        }
        primary.setdefault("repair_amendments", []).append(
            {
                "protocol": str(PROTOCOL),
                "repair_ledger": str(OUT),
                "snapshot_root": str(snapshot),
                "domains": ["pantry_plan"],
                "seeds": list(SEEDS),
                "scientific_change": False,
            }
        )
        atomic_json(PRIMARY_LEDGER, primary)
        atomic_json(OUT, payload)
        committed = True
        for job_id in submitted:
            run(["scontrol", "release", str(job_id)])
        payload["released"] = True
        payload["released_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        payload["released_scheduler_records"] = {
            str(job_id): scheduler_record(job_id) for job_id in submitted
        }
        atomic_json(OUT, payload)
    except Exception:
        if not committed:
            for job_id in submitted:
                subprocess.run(["scancel", str(job_id)], check=False)
            atomic_json(PRIMARY_LEDGER, original_primary)
        raise

    print(
        f"released E98-R1 smoke {smoke_id}, audit {audit_id}, replacements "
        + " ".join(str(item["new_job_id"]) for item in replacements)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
