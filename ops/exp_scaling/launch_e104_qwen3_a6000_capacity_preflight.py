#!/usr/bin/env python3
"""Submit E104's separate one-update Qwen-3B A6000 capacity preflight."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import launch_e104_group_centered_semantic_repair_three_scale as e104  # noqa: E402


DOMAIN = "graph_coloring"
SCALE = "qwen3b"
SEED = 70
TRAIN_ROWS = 1
CHECKPOINT_INTERVAL = 1
NODE_LIST = "node[103-104,205-208,805]"
PARTITION = "lowprio"
LEDGER = "var/artifacts/e104_qwen3_a6000_capacity_preflight_jobs.json"
AUDIT = "var/artifacts/e104_qwen3_a6000_capacity_preflight_gate.json"
PROTOCOL = (
    "paper/preregistration/"
    "e104_qwen3_a6000_capacity_preflight_20260817.md"
)
JOB_NAME = "e104-q3-a6-pre"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def run_stamp() -> str:
    return "e104_qwen3b_a6000_capacity_preflight_s70"


def save_path(root: Path) -> Path:
    return root / "var/data" / (
        "xdr_qwen25_3b_instruct_"
        f"{e104.VARIANT}_{run_stamp()}"
    )


def build_plan(root: Path) -> dict[str, Any]:
    e104_ledger_path = root / e104.LEDGER
    if not e104_ledger_path.is_file():
        raise SystemExit("Qwen-3B A6000 preflight requires the E104 ledger")
    e104_ledger = json.loads(e104_ledger_path.read_text(encoding="utf-8"))
    snapshot = Path(str(e104_ledger["snapshot_root"]))
    e104.verify_snapshot(snapshot)
    run = next(
        item
        for item in e104.references(root, SCALE)
        if str(item["domain"]) == DOMAIN
    )
    env, _ = e104.build_env(root, SCALE, run, snapshot)
    target = save_path(root)
    env.update(
        {
            "SAVE_PATH": str(target),
            "RUN_STAMP": run_stamp(),
            "OAT_ZERO_MAX_TRAIN": str(TRAIN_ROWS),
            "OAT_ZERO_NUM_PROMPT_EPOCH": "1",
            "OAT_ZERO_MAX_PROMPT_EPOCHS": "1",
            "OAT_ZERO_EVAL_PROMPT_INTERVAL": str(CHECKPOINT_INTERVAL),
            "OAT_ZERO_EVAL_MODE_COVERAGE_K": "0",
            "OAT_ZERO_SAVE_STEPS": str(CHECKPOINT_INTERVAL),
            "OAT_ZERO_SAVE_FROM": str(CHECKPOINT_INTERVAL),
            "OAT_ZERO_RESUME_STEPS": str(CHECKPOINT_INTERVAL),
            "OAT_ZERO_AUTO_RESUME": "0",
            "OAT_ZERO_WATCHDOG_REQUEUE": "0",
        }
    )
    command = e104.base_command(root, SCALE, run, env)
    rewritten: list[str] = []
    replacements = {
        "--job-name=": f"--job-name={JOB_NAME}",
        "--partition=": f"--partition={PARTITION}",
        "--account=": "--account=mltheory",
        "--nodelist=": f"--nodelist={NODE_LIST}",
        "--gres=": "--gres=gpu:a6000:1",
        "--time=": "--time=02:00:00",
        "--nice=": "--nice=0",
    }
    for token in command:
        replacement = next(
            (
                value
                for prefix, value in replacements.items()
                if token.startswith(prefix)
            ),
            None,
        )
        rewritten.append(replacement if replacement is not None else token)
    return {
        "snapshot": snapshot,
        "run": run,
        "env": env,
        "target": target,
        "command": rewritten,
    }


def held_audit(job_id: str) -> str:
    result = subprocess.run(
        ["scontrol", "show", "job", "-dd", "-o", job_id],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"cannot inspect held preflight job {job_id}")
    record = result.stdout
    required = (
        "JobState=PENDING",
        "Reason=JobHeldUser",
        f"Partition={PARTITION}",
        "Account=mltheory",
        f"ReqNodeList={NODE_LIST}",
        "TresPerNode=gres/gpu:a6000:1",
        f"OAT_ZERO_MAX_TRAIN={TRAIN_ROWS}",
        "OAT_ZERO_EVAL_MODE_COVERAGE_K=0",
        "OAT_ZERO_SEMANTIC_SHANNON_SUCCESS_CONDITIONED_GROUP_CENTERED_ADVANTAGE=1",
        "OAT_ZERO_SEMANTIC_SHANNON_SUCCESS_CONDITIONED_SIGNED_ADVANTAGE=0",
        "OAT_ZERO_SEMANTIC_RMS_CONTROL=0",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY=1",
    )
    missing = [needle for needle in required if needle not in record]
    if missing:
        raise RuntimeError(f"held preflight job {job_id} lacks {missing}")
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.submit and args.dry_run:
        raise SystemExit("choose --submit or --dry-run, not both")
    root = repo_root()
    protocol = root / PROTOCOL
    ledger_path = root / LEDGER
    if not protocol.is_file():
        raise SystemExit(f"preflight protocol is absent: {protocol}")
    if args.submit and ledger_path.exists():
        raise SystemExit(f"refusing duplicate preflight: {ledger_path}")
    plan = build_plan(root)
    if args.submit and plan["target"].exists():
        raise SystemExit(f"refusing to overwrite preflight: {plan['target']}")
    if args.dry_run or not args.submit:
        print(" ".join(shlex.quote(token) for token in plan["command"]))
        return 0
    result = subprocess.run(
        plan["command"], capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "preflight submission failed")
    job_id = result.stdout.strip().split(";", 1)[0]
    if not job_id.isdigit():
        raise RuntimeError(f"invalid preflight job id: {result.stdout!r}")
    try:
        held = held_audit(job_id)
        payload = {
            "schema": "e104_qwen3_a6000_capacity_preflight_jobs_v1",
            "protocol": str(protocol),
            "protocol_sha256": e104.digest(protocol),
            "launcher": str(Path(__file__)),
            "launcher_sha256": e104.digest(Path(__file__)),
            "e104_ledger": str(root / e104.LEDGER),
            "e104_ledger_sha256": e104.digest(root / e104.LEDGER),
            "snapshot_root": str(plan["snapshot"]),
            "scale": SCALE,
            "domain": DOMAIN,
            "seed": SEED,
            "target_steps": TRAIN_ROWS,
            "run_stamp": run_stamp(),
            "run_dir": str(plan["target"]),
            "job_id": int(job_id),
            "stdout": str(root / f"var/artifacts/logs/{JOB_NAME}-{job_id}.out"),
            "stderr": str(root / f"var/artifacts/logs/{JOB_NAME}-{job_id}.err"),
            "held_scheduler_record": held,
            "released": False,
        }
        e104.e81.atomic_json(ledger_path, payload)
        release = subprocess.run(
            ["scontrol", "release", job_id],
            capture_output=True,
            text=True,
            check=False,
        )
        if release.returncode != 0:
            raise RuntimeError(f"cannot release preflight job {job_id}")
        payload["released"] = True
        e104.e81.atomic_json(ledger_path, payload)
    except Exception:
        e104.e81.cancel([job_id])
        if ledger_path.exists():
            ledger_path.unlink()
        raise
    print(
        f"[e104-q3-a6000] job={job_id} released=True "
        f"snapshot={plan['snapshot']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
