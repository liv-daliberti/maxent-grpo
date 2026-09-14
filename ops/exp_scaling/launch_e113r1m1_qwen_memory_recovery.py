#!/usr/bin/env python3
"""Submit the one-job Qwen capacity repair for the DAPO recovery gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any


sys.path.insert(0, str(Path(__file__).resolve().parent))
import launch_e113_dapo_direct_baseline as e113  # noqa: E402
import launch_e113r1_dapo_recovery_smokes as r1  # noqa: E402
import launch_e79_falcon1b_aligned_verified_replay as e79  # noqa: E402


COHORT = "e113r1m1"
FAMILY = "qwen05b"
DOMAIN = "graph_coloring"
LEDGER = "var/artifacts/e113r1m1_qwen_memory_recovery_jobs.json"
PROTOCOL = "paper/preregistration/e113r1m1_qwen_memory_recovery_20260819.md"
R1_LEDGER = r1.LEDGER
TARGET = "var/data/e113r1m1_qwen05b_graph_dapo_smoke"
JOB_NAME = "e113r1m1-q05-graph-dapo-smoke"
NODELIST = "node[103-104,205-208,805]"
PARTITION = "lowprio"
ACCOUNT = "mltheory"
GRES = "gpu:a6000:1"
ORIGINAL_JOB_ID = 30790111
ORIGINAL_LOG = "var/artifacts/logs/e113r1-q05-graph-dapo-smoke-30790111.out"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def source_run(root: Path) -> dict[str, Any]:
    return next(
        run
        for run in e113.references(root, FAMILY)
        if str(run["domain"]) == DOMAIN
        and int(run["seed"]) == int(e113.FAMILY_SEEDS[FAMILY][0])
    )


def failure_evidence(root: Path) -> dict[str, Any]:
    r1_ledger = load_json(root / R1_LEDGER)
    smoke = r1_ledger["smokes"][FAMILY]
    if int(smoke["job_id"]) != ORIGINAL_JOB_ID:
        raise SystemExit("E113-R1 Qwen job identity drifted")
    run_dir = Path(str(smoke["run_dir"]))
    if (run_dir / "TRAINING_COMPLETE.json").exists():
        raise SystemExit("refusing M1 because the original Qwen smoke completed")
    rows: list[dict[str, Any]] = []
    for path in run_dir.glob("debug_job*/train_metrics.jsonl"):
        rows.extend(
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    accepted = [
        row
        for row in rows
        if float(row.get("actor/dapo_accepted_groups", 0.0)) == 1.0
    ]
    log_path = root / ORIGINAL_LOG
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    required = (
        "torch.OutOfMemoryError: CUDA out of memory",
        "Tried to allocate 428.00 MiB",
        "torch.Size([16, 367])",
    )
    if len(accepted) != 6 or any(value not in log_text for value in required):
        raise SystemExit("original Qwen OOM evidence is incomplete or drifted")
    return {
        "job_id": ORIGINAL_JOB_ID,
        "run_dir": str(run_dir),
        "accepted_updates": len(accepted),
        "last_query_step": max(
            float(row.get("misc/query_step", 0.0)) for row in rows
        ),
        "failure": "cuda_oom_backward",
        "failed_sequence_shape": [16, 367],
        "requested_allocation_mib": 428,
        "log": str(log_path),
        "log_sha256": e113.e78.digest(log_path),
    }


def build_env(
    root: Path,
    run: dict[str, Any],
    snapshot: Path,
) -> dict[str, str]:
    env = r1.recovery_env(root, FAMILY, run, snapshot, e79.model_root(root))
    env.update(
        {
            "SAVE_PATH": str(root / TARGET),
            "RUN_STAMP": "e113r1m1_qwen05b_graph_dapo_smoke",
            "OAT_ZERO_ADAM_OFFLOAD": "0",
            "OAT_ZERO_ACTIVATION_OFFLOADING": "0",
        }
    )
    return env


def command(root: Path, run: dict[str, Any], env: dict[str, str]) -> list[str]:
    base = r1.recovery_command(root, FAMILY, run, env)
    replacements = {
        "--job-name=": f"--job-name={JOB_NAME}",
        "--partition=": f"--partition={PARTITION}",
        "--account=": f"--account={ACCOUNT}",
        "--nodelist=": f"--nodelist={NODELIST}",
        "--gres=": f"--gres={GRES}",
    }
    output: list[str] = []
    for token in base:
        replacement = next(
            (value for prefix, value in replacements.items() if token.startswith(prefix)),
            None,
        )
        output.append(replacement if replacement is not None else token)
    return output


def audit_held(job_id: str, root: Path, snapshot: Path) -> str:
    result = subprocess.run(
        ["scontrol", "show", "job", "-dd", "-o", job_id],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(f"cannot inspect held E113-R1-M1 job {job_id}")
    required = (
        "JobState=PENDING",
        "Reason=JobHeldUser",
        "Dependency=(null)",
        f"JobName={JOB_NAME}",
        f"Partition={PARTITION}",
        f"Account={ACCOUNT}",
        f"ReqNodeList={NODELIST}",
        "TresPerNode=gres/gpu:a6000:1",
        f"SAVE_PATH={root / TARGET}",
        "RUN_STAMP=e113r1m1_qwen05b_graph_dapo_smoke",
        f"OAT_ZERO_SOURCE_ROOT={snapshot / 'src'}",
        "OAT_ZERO_VARIANT=dapo",
        "OAT_ZERO_DAPO_ENABLED=1",
        "OAT_ZERO_DAPO_MAX_NUM_GEN_BATCHES=10",
        "OAT_ZERO_MAX_QUERIES=5120",
        "OAT_ZERO_TRAIN_BATCH_SIZE=16",
        "OAT_ZERO_TRAIN_BATCH_SIZE_PER_DEVICE=16",
        "OAT_ZERO_ADAM_OFFLOAD=0",
        "OAT_ZERO_ACTIVATION_OFFLOADING=0",
    )
    missing = [value for value in required if value not in result.stdout]
    if missing:
        raise RuntimeError(f"held E113-R1-M1 job {job_id} lacks {missing}")
    return result.stdout


def main() -> int:
    root = e113.repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.submit and args.dry_run:
        raise SystemExit("choose --submit or --dry-run, not both")

    protocol_path = root / PROTOCOL
    ledger_path = root / LEDGER
    r1_path = root / R1_LEDGER
    for required in (protocol_path, r1_path, root / ORIGINAL_LOG):
        if not required.is_file():
            raise SystemExit(f"required E113-R1-M1 input is absent: {required}")
    if ledger_path.exists() and args.submit:
        raise SystemExit(f"refusing duplicate E113-R1-M1 submission: {ledger_path}")
    if (root / TARGET).exists():
        raise SystemExit(f"refusing to overwrite E113-R1-M1 output: {root / TARGET}")

    original = load_json(r1_path)
    snapshot = Path(str(original["snapshot_root"])).resolve()
    e113.verify_snapshot(snapshot)
    evidence = failure_evidence(root)
    run = source_run(root)
    env = build_env(root, run, snapshot)
    submit_command = command(root, run, env)

    if args.dry_run or not args.submit:
        print(shlex.join(submit_command))
        print(
            f"[e113r1m1] scientific=0 replacement=1 device=a6000 "
            f"original_accepted={evidence['accepted_updates']}"
        )
        return 0

    job_id = ""
    try:
        job_id = e113.submit_held(submit_command)
        held = audit_held(job_id, root, snapshot)
        smoke = {
            "scientific": False,
            "family": FAMILY,
            "domain": DOMAIN,
            "seed": int(run["seed"]),
            "job_id": int(job_id),
            "run_stamp": env["RUN_STAMP"],
            "run_dir": env["SAVE_PATH"],
            "max_train": r1.SMOKE_MAX_TRAIN,
            "max_queries": r1.SMOKE_MAX_QUERIES,
            "held_scheduler_record": held,
        }
        payload = {
            "schema": "e113r1m1_qwen_memory_recovery_jobs_v1",
            "cohort": COHORT,
            "released": False,
            "scientific_cells": 0,
            "target_steps": r1.SMOKE_MAX_TRAIN,
            "passes": 1,
            "smoke_domain": DOMAIN,
            "capacity_repair": {
                "partition": PARTITION,
                "account": ACCOUNT,
                "nodelist": NODELIST,
                "gres": GRES,
                "optimizer_offload": False,
                "activation_offload": False,
            },
            "protocol": str(protocol_path),
            "protocol_sha256": e113.e78.digest(protocol_path),
            "launcher_sha256": e113.e78.digest(Path(__file__)),
            "r1_ledger": str(r1_path),
            "r1_ledger_sha256": e113.e78.digest(r1_path),
            "failed_qwen_smoke": evidence,
            "snapshot_root": str(snapshot),
            "snapshot_identity": original["snapshot_identity"],
            "objective": e113.objective(),
            "smokes": {FAMILY: smoke},
            "runs": [],
        }
        e113.e78.atomic_json(ledger_path, payload)
        subprocess.run(["scontrol", "release", job_id], check=True)
        payload["released"] = True
        e113.e78.atomic_json(ledger_path, payload)
    except Exception:
        e113.cancel([job_id] if job_id else [])
        raise

    print(f"[e113r1m1] released Qwen memory-capacity recovery job {job_id}")
    print(f"[e113r1m1] ledger {ledger_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
