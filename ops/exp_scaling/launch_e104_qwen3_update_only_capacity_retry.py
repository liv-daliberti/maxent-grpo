#!/usr/bin/env python3
"""Submit the corrected update-only Qwen-3B A6000 capacity retry."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import launch_e104_group_centered_semantic_repair_three_scale as e104  # noqa: E402
import launch_e104_qwen3_update_only_capacity_preflight as failed  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
DOMAIN = failed.DOMAIN
SCALE = failed.SCALE
SEED = failed.SEED
TRAIN_ROWS = 1
NODE_LIST = failed.NODE_LIST
PARTITION = failed.PARTITION
JOB_NAME = "e104-q3-up-r1"
LEDGER = "var/artifacts/e104_qwen3_update_only_capacity_retry_jobs.json"
AUDIT = "var/artifacts/e104_qwen3_update_only_capacity_retry_gate.json"
PROTOCOL = (
    "paper/preregistration/"
    "e104_qwen3_update_only_capacity_retry_20260817.md"
)
OPS_OVERLAY = ROOT / (
    "var/artifacts/source_snapshots/"
    "e104_qwen3_capacity_no_eval_ops_v2/ops"
)
OVERLAY_PATCH_SHA256 = {
    "run_experiment.sh": "be0c43ba139561e1be9238cad6f5a73fc8a7e6a49e52a4263791982a8e7a09f7",
    "train.sh": "d6f14306725bcdd6b0e012819987227a8c1a895c5a2bcffa8b88a5a5e13c8761",
}


def run_stamp() -> str:
    return "e104_qwen3b_update_only_capacity_retry_s70"


def save_path() -> Path:
    return ROOT / "var/data" / (
        "xdr_qwen25_3b_instruct_"
        f"{e104.VARIANT}_{run_stamp()}"
    )


def tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def verify_overlay() -> None:
    snapshot = Path(
        json.loads((ROOT / e104.LEDGER).read_text(encoding="utf-8"))[
            "snapshot_root"
        ]
    )
    base = snapshot / "ops"
    base_files = {
        path.relative_to(base) for path in base.rglob("*") if path.is_file()
    }
    overlay_files = {
        path.relative_to(OPS_OVERLAY)
        for path in OPS_OVERLAY.rglob("*")
        if path.is_file()
    }
    if overlay_files != base_files:
        raise SystemExit("retry ops overlay file set differs from E104")
    changed = {
        str(relative)
        for relative in base_files
        if e104.digest(base / relative) != e104.digest(OPS_OVERLAY / relative)
    }
    if changed != set(OVERLAY_PATCH_SHA256):
        raise SystemExit(f"unexpected retry ops changes: {sorted(changed)}")
    for relative, expected in OVERLAY_PATCH_SHA256.items():
        if e104.digest(OPS_OVERLAY / relative) != expected:
            raise SystemExit(f"retry ops digest mismatch: {relative}")
    run_text = (OPS_OVERLAY / "run_experiment.sh").read_text(encoding="utf-8")
    train_text = (OPS_OVERLAY / "train.sh").read_text(encoding="utf-8")
    required = (
        "EVAL_STEPS=0",
        "export OAT_ZERO_RESUME_STEPS=-1",
        'EVAL_CADENCE_POLICY="capacity_preflight_none"',
    )
    if any(needle not in run_text for needle in required):
        raise SystemExit("retry ops overlay lacks bounded runtime hooks")
    if "cmd+=(--debug)" not in train_text:
        raise SystemExit("retry ops overlay lacks initial-eval skip")


def build_plan() -> dict[str, Any]:
    ledger = json.loads((ROOT / e104.LEDGER).read_text(encoding="utf-8"))
    snapshot = Path(str(ledger["snapshot_root"]))
    e104.verify_snapshot(snapshot)
    verify_overlay()
    run = next(
        item
        for item in e104.references(ROOT, SCALE)
        if str(item["domain"]) == DOMAIN
    )
    env, _ = e104.build_env(ROOT, SCALE, run, snapshot)
    target = save_path()
    env.update(
        {
            "SAVE_PATH": str(target),
            "RUN_STAMP": run_stamp(),
            "OAT_ZERO_MAX_TRAIN": "1",
            "OAT_ZERO_NUM_PROMPT_EPOCH": "1",
            "OAT_ZERO_MAX_PROMPT_EPOCHS": "1",
            "OAT_ZERO_OPS_SNAPSHOT_ROOT": str(OPS_OVERLAY),
            "OAT_ZERO_CAPACITY_PREFLIGHT_SKIP_EVAL": "1",
            "OAT_ZERO_EVAL_MODE_COVERAGE_K": "0",
            "OAT_ZERO_EXPORT_STEPS": "-1",
            "OAT_ZERO_SAVE_STEPS": "0",
            "OAT_ZERO_RESUME_STEPS": "-1",
            "OAT_ZERO_PRUNE_RESUME_ON_SUCCESS": "0",
            "OAT_ZERO_AUTO_RESUME": "0",
            "OAT_ZERO_WATCHDOG_REQUEUE": "0",
        }
    )
    command = e104.base_command(ROOT, SCALE, run, env)
    replacements = {
        "--job-name=": f"--job-name={JOB_NAME}",
        "--partition=": f"--partition={PARTITION}",
        "--account=": "--account=mltheory",
        "--nodelist=": f"--nodelist={NODE_LIST}",
        "--gres=": "--gres=gpu:a6000:1",
        "--time=": "--time=02:00:00",
        "--nice=": "--nice=0",
    }
    rewritten: list[str] = []
    for token in command:
        replacement = next(
            (value for prefix, value in replacements.items() if token.startswith(prefix)),
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
        raise RuntimeError(f"cannot inspect held retry job {job_id}")
    record = result.stdout
    required = (
        "JobState=PENDING",
        "Reason=JobHeldUser",
        f"Partition={PARTITION}",
        "Account=mltheory",
        f"ReqNodeList={NODE_LIST}",
        "TresPerNode=gres/gpu:a6000:1",
        f"OAT_ZERO_OPS_SNAPSHOT_ROOT={OPS_OVERLAY}",
        "OAT_ZERO_CAPACITY_PREFLIGHT_SKIP_EVAL=1",
        "OAT_ZERO_MAX_TRAIN=1",
        "OAT_ZERO_EVAL_MODE_COVERAGE_K=0",
        "OAT_ZERO_EXPORT_STEPS=-1",
        "OAT_ZERO_SAVE_STEPS=0",
        "OAT_ZERO_RESUME_STEPS=-1",
        "OAT_ZERO_SEMANTIC_SHANNON_SUCCESS_CONDITIONED_GROUP_CENTERED_ADVANTAGE=1",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY=1",
    )
    missing = [needle for needle in required if needle not in record]
    if missing:
        raise RuntimeError(f"held retry job {job_id} lacks {missing}")
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.submit and args.dry_run:
        raise SystemExit("choose --submit or --dry-run, not both")
    protocol = ROOT / PROTOCOL
    ledger_path = ROOT / LEDGER
    if not protocol.is_file():
        raise SystemExit(f"retry protocol is absent: {protocol}")
    if args.submit and ledger_path.exists():
        raise SystemExit(f"refusing duplicate retry: {ledger_path}")
    plan = build_plan()
    if args.submit and plan["target"].exists():
        raise SystemExit(f"refusing to overwrite retry: {plan['target']}")
    if args.dry_run or not args.submit:
        print(" ".join(shlex.quote(token) for token in plan["command"]))
        return 0
    result = subprocess.run(
        plan["command"], capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "retry submission failed")
    job_id = result.stdout.strip().split(";", 1)[0]
    if not job_id.isdigit():
        raise RuntimeError(f"invalid retry job id: {result.stdout!r}")
    try:
        held = held_audit(job_id)
        payload = {
            "schema": "e104_qwen3_update_only_capacity_retry_jobs_v1",
            "protocol": str(protocol),
            "protocol_sha256": e104.digest(protocol),
            "launcher": str(Path(__file__)),
            "launcher_sha256": e104.digest(Path(__file__)),
            "failed_preflight_ledger": str(ROOT / failed.LEDGER),
            "failed_preflight_ledger_sha256": e104.digest(ROOT / failed.LEDGER),
            "failed_preflight_audit": str(ROOT / failed.AUDIT),
            "failed_preflight_audit_sha256": e104.digest(ROOT / failed.AUDIT),
            "e104_ledger": str(ROOT / e104.LEDGER),
            "e104_ledger_sha256": e104.digest(ROOT / e104.LEDGER),
            "snapshot_root": str(plan["snapshot"]),
            "ops_overlay": str(OPS_OVERLAY),
            "ops_overlay_tree_sha256": tree_digest(OPS_OVERLAY),
            "scale": SCALE,
            "domain": DOMAIN,
            "seed": SEED,
            "target_steps": TRAIN_ROWS,
            "run_stamp": run_stamp(),
            "run_dir": str(plan["target"]),
            "job_id": int(job_id),
            "stdout": str(ROOT / f"var/artifacts/logs/{JOB_NAME}-{job_id}.out"),
            "stderr": str(ROOT / f"var/artifacts/logs/{JOB_NAME}-{job_id}.err"),
            "held_scheduler_record": held,
            "released": False,
        }
        e104.e81.atomic_json(ledger_path, payload)
        release = subprocess.run(
            ["scontrol", "release", job_id], capture_output=True, text=True, check=False
        )
        if release.returncode != 0:
            raise RuntimeError(f"cannot release retry job {job_id}")
        payload["released"] = True
        e104.e81.atomic_json(ledger_path, payload)
    except Exception:
        e104.e81.cancel([job_id])
        if ledger_path.exists():
            ledger_path.unlink()
        raise
    print(f"[e104-q3-update-only-retry] job={job_id} released=True")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
